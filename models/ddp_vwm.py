import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from einops import rearrange, repeat
import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Dict, List, Optional, Tuple
from .utils import *


class DDPVWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor, 
        label_generator,
        cls_pos_weight=10,
        predictor_criterion=None, 
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        training_stage=''
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.label_generator = label_generator
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.concat_dim = concat_dim
        self.cls_pos_weight = cls_pos_weight
        self.training_stage = training_stage

        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) 
        print("Model emb_dim: ", self.emb_dim)
        self.d_visual = self.emb_dim - self.action_dim - self.proprio_dim
        self.d_state = self.d_visual + self.proprio_dim

        # --- loss function ---
        # According to different training stages，Define required loss functions
        self.localization_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.cls_pos_weight))
        self.dynamics_loss = nn.MSELoss()
        self.lrm_loss = nn.MSELoss()
        
        if "dino" in self.encoder.name:
            decoder_scale = 16
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose([transforms.Resize(self.encoder_image_size)])
        else:
            self.encoder_transform = lambda x: x

    def __getattr__(self, name):
        """
        A convenient attribute getter method.if DDPVWorldModel does not have this attribute itself，
        it will try to get from self.predictor (DDP_Predictor) find in.
        This makes accessing attributes of sub-modules externally simple (e.g., model.training_stage).
        """
        try:
            # Prioritize returning its own attributes
            return super().__getattr__(name)
        except AttributeError:
            # If it doesn't have it itself，then try to get from self.predictor get from
            if hasattr(self, 'predictor') and self.predictor is not None and hasattr(self.predictor, name):
                return getattr(self.predictor, name)
            else:
                # if predictor also doesn't have，then throw the original AttributeError
                raise

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        Execute single-step inference prediction.
        This method encapsulates DDP-WM at 'inference' complete forward propagation in mode.

        Args:
            z_history (torch.Tensor): the input history feature sequence，shape [B, T_hist, N, D_model].

        Returns:
            torch.Tensor: the predicted complete feature map of the next frame z_{t+1}, shape [B, 1, N, D_model].
        """
        # 1. Confirm that the predictor is in inference mode
        if self.predictor.training_stage != 'inference':
            raise RuntimeError(f"call predict() when, DDP_Predictor 's training_stage must be 'inference', but currently is '{self.predictor.training_stage}'")
        
        # 2. Call the predictor directly
        # at 'inference' in mode，It will return a dictionary containing the final prediction
        predictor_outputs = self.predictor(z_history)
        
        # 3. Extract and return the final complete feature map prediction
        z_t_plus_1 = predictor_outputs['final_prediction'] # Shape: [B, 1, N, D_model]
        
        return z_t_plus_1

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act


    def forward(self, obs, act):
        # -------------------------------------------------------------
        # step 1: Initial encoding (general)
        # -------------------------------------------------------------
        z = self.encode(obs, act) # (B, T, N, D)
        B, T_full, N, D = z.shape

        # -------------------------------------------------------------
        # step 2: process `trainrollout` logic
        # -------------------------------------------------------------
        if self.training_stage not in ['localization', 'lrm']:
            # This step will userollout's results to replacez's history part
            z_roll = z[:, :-1]
            z_roll = get_rollout_z(z_roll, self, obs['visual'], self.num_hist)
            z = torch.cat([z_roll, z[:, -1:]], dim=1)

        # -------------------------------------------------------------
        # step 3: Prepare predictor input and ground truth
        # -------------------------------------------------------------
        # Regardless of the stage，The predictor requires history input z_history
        z_history = z[:, -self.num_hist-1:-1, :, :] # (B, T-1, N, D)
        
        # -------------------------------------------------------------
        # step 4: Calculate loss based on the training stage
        # -------------------------------------------------------------
        loss = 0
        loss_components = {}
        
        predictor_outputs = self.predictor(z_history)

        if self.predictor.training_stage == 'localization':
            mask_logits = predictor_outputs['mask_logits']
            
            # Apply transform when generating labels
            images_dict = get_ref(obs['visual'],  self)
            # z_dict temporarily pass inNone，becauselabel_generatorispixelmode
            gt_mask = self.label_generator(images_dict=images_dict, z_dict=None) 
            
            loss = self.localization_loss(mask_logits, gt_mask.float())
            loss_components['loss_localization'] = loss

        else: # 'primary_predictor' or 'lrm' stage
            # key_point:generated by the frozen localizermaskto split the ground truth
            predicted_mask = predictor_outputs['mask']
            
            # Get the ground truth features of the next frame
            z_next_gt_full = z[:, -1, :, :] # (B, N, D)
            
            if self.predictor.training_stage == 'primary_predictor':
                pred_fg = predictor_outputs['pred_fg'] # (B, K, D)

                # use predicted_mask from z_next_gt_full extract the ground truth foreground from
                K = pred_fg.shape[1]
                gt_fg_next = z_next_gt_full[predicted_mask].view(B, K, D)
                
                loss = self.dynamics_loss(pred_fg, gt_fg_next.detach())
                loss_components['loss_primary'] = loss

            elif self.predictor.training_stage == 'lrm':
                pred_bg = predictor_outputs['pred_bg'] # (B, N-K, D)
                
                # use predicted_mask from z_next_gt_full extract the ground truth background from
                num_bg = pred_bg.shape[1]
                gt_bg_next = z_next_gt_full[~predicted_mask].view(B, num_bg, D)
                
                loss = self.lrm_loss(pred_bg, gt_bg_next.detach())
                loss_components['loss_lrm'] = loss

        # return，Let the external accelerator.backward(loss) process
        return None, None, None, loss, loss_components


    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        use DDP-WM predictor performs multi-step execution in an open loop rollout.

        Args:
            obs_0 (dict): Initial observation, "visual" shape [B, num_obs_init, C, H, W].
            act (torch.Tensor): The complete action sequence, shape [B, T_total, action_dim].
                                T_total should be equal to num_obs_init + T_future.
        
        Returns:
            Tuple[Dict, torch.Tensor]: the separated observation embeddings and the completezsequence.
        """
        self.eval() # Ensure the entire model is in evaluation mode

        # 1. Encode the initial observation sequence
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        z = self.encode(obs_0, act_0)  # Shape: [B, num_obs_init, N, D_model]
        
        # 2. Prepare the future action sequence
        future_actions = act[:, num_obs_init:] # Shape: [B, T_future, action_dim]
        T_future = future_actions.shape[1]
        
        t = 0
        inc = 1 # predict one frame at a time

        # 3. Loop for multi-step prediction
        while t < T_future:
            # a. Prepare the history window for prediction
            # here self.num_hist is the history length in the model configuration
            z_history_window = z[:, -self.num_hist:] # Shape: [B, num_hist, N, D_model]
            
            # b. Call the single-step prediction method
            z_pred_next = self.predict(z_history_window) # Shape: [B, 1, N, D_model]
            
            # c. Inject the next future action
            next_action = future_actions[:, t : t + inc] # Shape: [B, 1, action_dim]
            z_pred_with_action = self.replace_actions_from_z(z_pred_next, next_action)
            
            # d. Concatenate the newly predicted frame to the sequence
            z = torch.cat([z, z_pred_with_action], dim=1)
            t += inc

        # 4. (optional)Predict one extra step(beyond the last step of the action sequence)
        z_history_window = z[:, -self.num_hist:]
        z_pred_final = self.predict(z_history_window)
        # There is no new action in this step，so it will reuse the last action
        z = torch.cat([z, z_pred_final], dim=1)
        
        # 5. Separate and return the final result
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z