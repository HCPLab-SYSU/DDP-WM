"""DDP-WM world model: predicts the next frame through a small set of foreground tokens.

See ``models/ddp_predictor.py`` for the three training stages. The per-environment settings
(use_history, mask_dilation, num_motion_tokens) come from the ``motion`` config group.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision import transforms

from .ddp_predictor import CLS_POS_WEIGHT, PATCH_GRID, FrozenLocalizer, LabelGenerator

# Number of frames the predictor attends to when use_history is on (current frame + 2 past frames).
NUM_HISTORY_FRAMES = 3
# Predicted patch features with an L2 norm at or below this value keep the previous frame's feature.
FEATURE_NORM_THRESHOLD = 10.0


class DDPVWorldModel(nn.Module):
    def __init__(
        self,
        image_size,
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        training_stage,
        motion,
        ckpt_path=None,
        proprio_dim=0,
        action_dim=0,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
        **unused,
    ):
        """``decoder`` is accepted for config compatibility but not used (the model never decodes images).

        ``ckpt_path`` (stages 2 and 3): checkpoint the frozen history fusion, localizer and proprio/action
        encoders are taken from; any checkpoint from stage 1 on carries them.
        """
        super().__init__()
        assert concat_dim == 1, "Only concat_dim=1 (feature concatenation) is supported"
        self.training_stage = training_stage
        self.use_history = motion["use_history"]
        self.num_hist = 3 if training_stage == "localization" else 5
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = None
        self.predictor = predictor
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.concat_dim = concat_dim
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat
        self.emb_dim = self.encoder.emb_dim + self.action_dim + self.proprio_dim
        self.d_visual = self.emb_dim - self.action_dim - self.proprio_dim
        self.d_state = self.d_visual + self.proprio_dim

        if "dino" in self.encoder.name:
            self.encoder_transform = transforms.Compose([transforms.Resize(PATCH_GRID * encoder.patch_size)])
        else:
            self.encoder_transform = lambda x: x
        self.emb_criterion = nn.MSELoss()

        self.label_generator = LabelGenerator(
            training_stage=training_stage,
            mask_dilation=motion["mask_dilation"],
            num_motion_tokens=motion["num_motion_tokens"],
            d_state=self.d_state,
        )

        if training_stage != "localization":
            # Frozen stage-1 modules; the proprio/action encoders come from the same checkpoint.
            self.frozen_localizer = FrozenLocalizer(
                ckpt_path, use_history=self.use_history, num_queries=motion["num_motion_tokens"]
            )
            checkpoint = self.frozen_localizer.checkpoint
            for name in ("proprio_encoder", "action_encoder"):
                module = getattr(self, name)
                module = getattr(module, "module", module)  # unwrap DDP
                module.load_state_dict(checkpoint[name])
                for param in module.parameters():
                    param.requires_grad_(False)
            # The predictor shares them, so that its checkpoints carry the whole model.
            target = getattr(self.predictor, "module", self.predictor)
            if self.use_history:
                target.history_fusion = self.frozen_localizer.history_fusion
            target.localizer = self.frozen_localizer.localizer

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        return self

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        return self

    # ------------------------------------------------------------------ encoding

    def encode(self, obs, act):
        """obs: dict of (b, t, ...) tensors, act: (b, t, action_dim) -> z: (b, t, num_patches, emb_dim)."""
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        num_patches = z_dct["visual"].shape[2]
        proprio = repeat(z_dct["proprio"].unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
        proprio = proprio.repeat(1, 1, 1, self.num_proprio_repeat)
        act_emb = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
        act_emb = act_emb.repeat(1, 1, 1, self.num_action_repeat)
        return torch.cat([z_dct["visual"], proprio, act_emb], dim=3)

    def encode_act(self, act):
        return self.action_encoder(act)

    def encode_proprio(self, proprio):
        return self.proprio_encoder(proprio)

    def encode_obs(self, obs):
        """obs: "visual" (b, t, 3, h, w), "proprio" (b, t, d) -> "visual" (b, t, num_patches, d_visual), "proprio" (b, t, d_proprio)."""
        visual = obs["visual"]
        b = visual.shape[0]
        visual = self._resize_frames(visual)
        visual_embs = self.encoder.forward(rearrange(visual, "b t ... -> (b t) ..."))
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)
        return {"visual": visual_embs, "proprio": self.encode_proprio(obs["proprio"])}

    def separate_emb(self, z):
        """z: (b, t, num_patches, emb_dim) -> z_obs dict, z_act (b, t, action_emb_dim)."""
        z_visual = z[..., : -(self.proprio_dim + self.action_dim)]
        z_proprio = z[..., -(self.proprio_dim + self.action_dim) : -self.action_dim]
        z_act = z[..., -self.action_dim :]
        # remove tiled dimensions
        z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
        z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        return {"visual": z_visual, "proprio": z_proprio}, z_act

    def replace_actions_from_z(self, z, act):
        act_emb = repeat(self.encode_act(act).unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
        z[..., -self.action_dim :] = act_emb.repeat(1, 1, 1, self.num_action_repeat)
        return z

    def _resize_frames(self, visual):
        b = visual.shape[0]
        visual = self.encoder_transform(rearrange(visual, "b t ... -> (b t) ..."))
        return rearrange(visual, "(b t) ... -> b t ...", b=b)

    # ------------------------------------------------------------------ prediction

    def _merge_prediction(self, z_t, pred):
        """Next-frame state: predicted features where their norm exceeds the threshold, z_t elsewhere."""
        keep = torch.norm(pred, p=2, dim=-1, keepdim=True) > FEATURE_NORM_THRESHOLD
        return torch.where(keep, pred, z_t[..., : pred.size(-1)])

    def _history_window(self, z):
        return z[:, -NUM_HISTORY_FRAMES:] if self.use_history else z[:, -1:]

    def _rollout_history(self, z, visual, num_frames):
        """Teacher-forced rollout used during training (stages 2 and 3).

        Rebuilds frames 1.. of z one step at a time, applying the ground-truth next-frame targets
        (restricted to the selected foreground tokens in stage 2) to the previously rebuilt frame.
        Returns the last num_frames frames.
        """
        z = z.clone()
        for i in range(1, z.size(1)):
            frames = self._resize_frames(visual[:, i - 1 : i + 1])
            targets, _ = self.label_generator(z[:, i - 1 : i + 1], frames)
            z[:, i, :, : self.d_state] = self._merge_prediction(z[:, i - 1], targets.squeeze(1))
        return z[:, -num_frames:]

    def forward(self, obs, act):
        """Training step: predict the last frame of the window from the frames before it."""
        z = self.encode(obs, act)
        if self.training_stage != "localization":
            num_frames = self.num_hist if self.use_history else 1
            z = torch.cat([self._rollout_history(z[:, :-1], obs["visual"], num_frames), z[:, -1:]], dim=1)
        else:
            z = z[:, -(self.num_hist + 1) :]
        if self.use_history:
            z = z[:, -(NUM_HISTORY_FRAMES + 1) :]
        z_history = z[:, :-1]
        frames = self._resize_frames(obs["visual"])[:, -2:]

        pred_mask = self.frozen_localizer(z_history).unsqueeze(1) if self.training_stage != "localization" else None
        targets, mask = self.label_generator(z[:, -2:], frames, pred_mask)
        mask = mask.squeeze(1).float()

        if self.training_stage == "localization":
            logits = self.predictor(z_history)
            pos_weight = torch.tensor(CLS_POS_WEIGHT, device=logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
        else:
            pred = self.predictor(z_history, mask)
            loss = F.mse_loss(pred[..., : self.d_state], targets.squeeze(1)[..., : self.d_state])

        loss_components = {"loss_total": loss, "loss": loss}
        return None, None, None, loss, loss_components

    def _predict_next(self, z_hist, z_pair, frame_pair):
        """z_hist: (b, t, num_patches, emb_dim) -> next frame (b, 1, num_patches, emb_dim) with zero actions.

        The token selection goes through the same label generator as in training. Only the localizer's
        change scores are used at planning time, so z_pair / frame_pair are placeholders for the unknown
        next frame.
        """
        pred_mask = self.frozen_localizer(z_hist).unsqueeze(1)
        _, mask = self.label_generator(z_pair, frame_pair, pred_mask)
        mask = mask.squeeze(1).float()
        pred = self.predictor(z_hist, mask)
        z_next = self._merge_prediction(z_hist[:, -1], pred)
        return torch.cat([z_next, torch.zeros_like(z_next[..., : self.action_dim])], dim=-1).unsqueeze(1)

    def rollout(self, obs_0, act):
        """Open-loop rollout (stages 2 and 3).

        obs_0: "visual" (b, n, 3, h, w), "proprio" (b, n, d); act: (b, n + T, action_dim)
        returns z_obses (dict) and z: (b, n + T + 1, num_patches, emb_dim)
        """
        assert self.training_stage != "localization", "Rollout needs a stage-2 or stage-3 model"
        self.eval()
        num_obs_init = obs_0["visual"].shape[1]
        z = self.encode(obs_0, act[:, :num_obs_init])
        frames = self._resize_frames(obs_0["visual"])
        future_actions = act[:, num_obs_init:]
        for t in range(future_actions.shape[1]):
            z_hist = self._history_window(z)
            # random placeholders for the (unknown) next frame pair passed to the label generator
            frame_pair = torch.rand_like(frames[:, -1:].repeat(1, 2, 1, 1, 1))
            z_pair = torch.rand_like(z_hist[:, -1:].repeat(1, 2, 1, 1))
            z_new = self._predict_next(z_hist, z_pair, frame_pair)
            z_new = self.replace_actions_from_z(z_new, future_actions[:, t : t + 1])
            z = torch.cat([z, z_new], dim=1)
        # one more frame beyond the action sequence (its action embedding stays zero)
        z = torch.cat([z, self._predict_next(self._history_window(z), z_pair, frame_pair)], dim=1)
        z_obses, _ = self.separate_emb(z)
        return z_obses, z
