"""DDP-WM predictor.

Training runs in three stages (selected by ``training_stage``):

1. ``localization``: Historical Information Fusion + Dynamic Localization Network. For every patch of
   the current frame, predict which of its 2x2 sub-patches will change in the next frame (labels come
   from pixel differences).
2. ``primary_predictor``: with the stage-1 modules frozen, the localizer picks ``num_motion_tokens``
   patches and the Sparse Primary Dynamics Predictor predicts their next-frame features.
3. ``lrm``: starting from a stage-2 model with everything else frozen, the Low-Rank Correction Module
   updates the remaining background patches from the predicted foreground.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .vit import ViTPredictor

STAGES = ("localization", "primary_predictor", "lrm")

PATCH_GRID = 14  # DINOv2 patch grid is PATCH_GRID x PATCH_GRID
NUM_PATCHES = PATCH_GRID * PATCH_GRID
MOTION_THRESHOLD = 0.1  # pixel RMS difference above which a (sub-)patch counts as changing
CLS_POS_WEIGHT = 10.0  # positive-class weight of the localization BCE loss


class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU activations."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            layers.append(nn.Linear(n, k))
            layers.append(nn.ReLU(inplace=True))
        layers.pop()  # no activation after the last layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CrossAttentionDecoderLayer(nn.Module):
    """Post-norm cross-attention block: tgt attends to memory. Inputs are (L, B, D)."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)


class TransformerDecoderLayer(nn.Module):
    """Post-norm DETR decoder layer: self-attention over the queries, then cross-attention to memory."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(q, k, value=tgt)[0]))
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


# ====================================================================
#  Historical Information Fusion Module
# ====================================================================
class HistoricalInformationFusion(nn.Module):
    """Cross-attention from the current frame to the previous num_frames - 1 frames."""

    def __init__(self, d_model, num_frames, n_heads=4, dropout=0.1):
        super().__init__()
        self.history_encoder = CrossAttentionDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.hist_query_pos = nn.Parameter(torch.randn(NUM_PATCHES, d_model))
        self.hist_mem_pos = nn.Parameter(torch.randn(NUM_PATCHES, d_model))
        self.hist_time_embeds = nn.Parameter(torch.zeros(num_frames - 1, d_model))

    def forward(self, z):
        """(B, T, N, D) -> (B, 1, N, D). With a single frame the input is returned unchanged
        (plus zero-valued terms so that every parameter stays in the autograd graph)."""
        B, t_len = z.shape[:2]
        if t_len > 1:
            current_frame = z[:, -1].permute(1, 0, 2)  # (N, B, D)
            history_frames = rearrange(z[:, :-1], "b t n d -> (t n) b d")
            query_pos = self.hist_query_pos.unsqueeze(1).repeat(1, B, 1)
            mem_pos = self.hist_mem_pos.unsqueeze(0).repeat(t_len - 1, 1, 1)
            mem_pos = mem_pos + self.hist_time_embeds[: t_len - 1].unsqueeze(1)
            mem_pos = rearrange(mem_pos, "t n d -> (t n) d").unsqueeze(1).repeat(1, B, 1)
            encoded = self.history_encoder(tgt=current_frame, memory=history_frames, query_pos=query_pos, pos=mem_pos)
            return encoded.permute(1, 0, 2).unsqueeze(1)
        dummy_out = self.history_encoder(z[:1, 0, :1], z[:1, 0, :1])
        return (
            z
            + self.hist_time_embeds.sum() * 0
            + self.hist_query_pos.sum() * 0
            + self.hist_mem_pos.sum() * 0
            + dummy_out.sum() * 0
        )


# ====================================================================
#  Dynamic Localization Network
# ====================================================================
class DynamicLocalizationNetwork(nn.Module):
    """Small ViT over reduced visual features; outputs 4 logits (2x2 sub-patches) per patch.

    Once frozen (stages 2 and 3) it stays in eval mode.
    """

    def __init__(self, d_visual, d_other, reduced_dim=192, num_layers=6, n_heads=3, mlp_dim=768, dropout=0.1):
        super().__init__()
        self.d_visual = d_visual
        dim = reduced_dim + d_other
        self.dino_wm_ti = ViTPredictor(
            dim=dim, depth=num_layers, heads=n_heads, mlp_dim=mlp_dim,
            num_frames=1, num_patches=NUM_PATCHES, dropout=dropout, emb_dropout=0, pool="mean",
        )
        self.dimensionality_reduction_layer = nn.Linear(d_visual, reduced_dim)
        self.cls_head = MLP(dim, dim, 4, num_layers=3)
        self.frozen = False

    def train(self, mode=True):
        return super().train(mode and not self.frozen)

    def forward(self, z):
        """(B, 1, N, D) -> (B, N, 4) logits."""
        z_reduced = torch.cat([self.dimensionality_reduction_layer(z[..., : self.d_visual]), z[..., self.d_visual :]], dim=-1)
        z_reduced = rearrange(z_reduced, "b t p d -> b (t p) d")
        return self.cls_head(self.dino_wm_ti(z_reduced))


# ====================================================================
#  Sparse Primary Dynamics Predictor
# ====================================================================
class SparsePrimaryDynamicsPredictor(nn.Module):
    """ViT over the selected foreground tokens; predicts their next-frame features."""

    def __init__(self, d_model, num_layers=6, n_heads=16, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.dino_wm = ViTPredictor(
            dim=d_model, depth=num_layers, heads=n_heads, mlp_dim=mlp_dim, num_frames=1,
            num_patches=NUM_PATCHES, dropout=dropout, emb_dropout=0, pool="mean",
        )

    def forward(self, x, mask):
        """x: (B, N, D), mask: (B, N) bool with k Trues per row -> (B, k, D)."""
        return self.dino_wm(x, mask)


# ====================================================================
#  Low-Rank Correction Module (LRM)
# ====================================================================
class LowRankCorrectionModule(nn.Module):
    """Cross-attention from the background patches to the predicted foreground patches.

    Holds a learnable absolute position embedding for all patches; ``mem_mask`` marks the
    foreground patches (exactly k per row) and decides which embeddings go to memory and target.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.0, n_patches=NUM_PATCHES):
        super().__init__()
        self.n_patches = n_patches
        self.d_model = d_model
        self.ape = nn.Parameter(torch.randn(n_patches, d_model))
        self.cross_attn = CrossAttentionDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)

    @staticmethod
    def mask_to_indices(mem_mask):
        """(B, N) mask with the same number of Trues per row -> sorted (B, k) and (B, N-k) indices."""
        if mem_mask.dtype != torch.bool:
            mem_mask = mem_mask > 0.5
        B, N = mem_mask.shape
        pos = torch.arange(N, device=mem_mask.device).unsqueeze(0).expand(B, -1)
        return pos[mem_mask].view(B, -1), pos[~mem_mask].view(B, -1)

    def forward(self, tgt_tokens, mem_tokens, mem_mask):
        """tgt_tokens: (B, N-k, D), mem_tokens: (B, k, D), mem_mask: (B, N) -> (B, N-k, D)."""
        B, n_tgt, D = tgt_tokens.shape
        k = mem_tokens.shape[1]
        idx_mem, idx_tgt = self.mask_to_indices(mem_mask)
        if idx_mem.shape[1] != k or idx_tgt.shape[1] != n_tgt:
            raise ValueError(f"mask selects {idx_mem.shape[1]} tokens, expected {k}")
        ape_mem = self.ape.index_select(0, idx_mem.reshape(-1)).view(B, k, D)
        ape_tgt = self.ape.index_select(0, idx_tgt.reshape(-1)).view(B, n_tgt, D)
        out = self.cross_attn(
            tgt=tgt_tokens.transpose(0, 1),
            memory=mem_tokens.transpose(0, 1),
            pos=ape_mem.transpose(0, 1),
            query_pos=ape_tgt.transpose(0, 1),
        )
        return out.transpose(0, 1)


def dilate_mask(mask, connectivity):
    """Dilate a (B, 196) patch mask on the 14x14 grid; connectivity 0 returns it unchanged."""
    if connectivity not in (4, 8):
        return mask
    b, n = mask.shape
    assert n == NUM_PATCHES, f"Expected {NUM_PATCHES} patches, got {n}"
    if connectivity == 4:
        kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    else:
        kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    kernel = torch.tensor(kernel, dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    dilated = F.conv2d(mask.view(b, 1, PATCH_GRID, PATCH_GRID), kernel, padding=1) > 0
    return dilated.float().view(b, n)


def force_fixed_k_mask(mask, k):
    """Return a (B, N) bool mask with exactly k Trues per row.

    Higher mask values are kept first; ties are broken randomly, and if fewer than k entries
    are positive the rest is filled with random patches.
    """
    B, N = mask.shape
    if N < k:
        raise ValueError(f"Number of patches N={N} must be >= k={k}")
    scores = mask.float() * 2.0 + torch.rand(B, N, device=mask.device)
    _, indices = torch.topk(scores, k, dim=1)
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask.scatter_(1, indices, True)
    return new_mask


class LabelGenerator(nn.Module):
    """Builds change masks and training targets for a pair of frames (no parameters)."""

    def __init__(self, training_stage, mask_dilation, num_motion_tokens, d_state, threshold=MOTION_THRESHOLD):
        super().__init__()
        self.training_stage = training_stage
        self.mask_dilation = mask_dilation
        self.num_motion_tokens = num_motion_tokens
        self.d_state = d_state
        self.threshold = threshold

    def pixel_motion_mask(self, frames):
        """frames: (B, 2, C, H, W) -> (B, 1, G*G) float mask of patches whose RMS pixel change exceeds the threshold.

        G is the patch grid (14), or twice that in stage 1 where the localizer predicts 2x2 sub-patches.
        """
        B = frames.shape[0]
        pixel_diff = (frames[:, 1:] - frames[:, :-1]).flatten(0, 1)  # (B, C, H, W)
        H, W = pixel_diff.shape[-2:]
        patch_h, patch_w = H // PATCH_GRID, W // PATCH_GRID
        if self.training_stage == "localization":
            patch_h, patch_w = patch_h // 2, patch_w // 2
        pixel_norms_sq = torch.sum(pixel_diff * pixel_diff, dim=1, keepdim=True)
        mean_sq_norms = F.avg_pool2d(pixel_norms_sq, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        norms = torch.sqrt(mean_sq_norms.view(B, 1, -1))
        return (norms > self.threshold).float()

    def select_tokens(self, mask):
        """(B, 1, N) change scores -> (B, 1, N) bool mask with exactly num_motion_tokens Trues."""
        mask = dilate_mask(mask.squeeze(1), connectivity=self.mask_dilation)
        return force_fixed_k_mask(mask, self.num_motion_tokens).unsqueeze(1)

    def forward(self, z, frames, pred_mask=None):
        """
        z: (B, 2, N, E) features of frames t and t+1; frames: (B, 2, C, H, W).
        pred_mask: optional (B, 1, N) change scores from the frozen localizer (stages 2 and 3).
        Returns (targets, mask):
          stage 1:    targets is None, mask is the (B, 1, 4N) sub-patch label;
          stage 2/3:  mask is the (B, 1, N) bool token selection, targets are the (B, 1, N, d_state)
                      next-frame features (zeroed outside the selection in stage 2).
        """
        mask = self.pixel_motion_mask(frames) if pred_mask is None else pred_mask
        if self.training_stage == "localization":
            return None, mask
        mask = self.select_tokens(mask)
        targets = z[:, 1:, :, : self.d_state]
        if self.training_stage == "primary_predictor":
            targets = targets * mask.unsqueeze(-1)
        return targets, mask


class _QueryDecoderMixin:
    """DETR-style query decoder the DDP-WM predictors are derived from: learnable queries attend to the patch
    features and a cross-attention layer decodes them back onto the patch grid.
    """

    def _build_query_decoder(self, d_model, d_visual, d_out, num_queries, coord_and_delta_heads, sub_patch_head):
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, nhead=4, dim_feedforward=2048, dropout=0.1) for _ in range(6)]
        )
        self.cls_head = MLP(d_model, d_model, 1, num_layers=3)
        if coord_and_delta_heads:
            self.coord_head = MLP(d_model, d_model, 2, num_layers=3)
            self.delta_head = nn.Linear(d_model, d_out)
        if sub_patch_head:
            self.cls_head = MLP(d_model, d_model, 4, num_layers=3)
        dense_pos = torch.cat([torch.randn(NUM_PATCHES, d_visual), torch.zeros(NUM_PATCHES, d_model - d_visual)], dim=-1)
        self.register_buffer("dense_query_pos", dense_pos)
        self.dense_decoder = CrossAttentionDecoderLayer(d_model=d_model, nhead=4, dropout=0.1)

    def _drop_query_decoder(self):
        for name in ("query_embed", "decoder_layers", "cls_head", "coord_head", "delta_head", "dense_query_pos", "dense_decoder"):
            if hasattr(self, name):
                delattr(self, name)


class FrozenLocalizer(_QueryDecoderMixin, nn.Module):
    """Frozen stage-1 modules (history fusion + localizer) used in stages 2 and 3 to choose the foreground tokens.

    Loads them from the ``history_fusion.*`` / ``localizer.*`` predictor keys of ``ckpt_path`` (a stage-1
    checkpoint, or any later checkpoint, which carries them too) and keeps the full checkpoint in
    ``checkpoint`` so that the world model can also take the proprio/action encoders from it.
    """

    def __init__(
        self, ckpt_path, use_history, num_queries=32, d_visual=384, d_action_embed=10, d_proprio_embed=10, num_frames=3
    ):
        super().__init__()
        self.use_history = use_history
        d_model = d_visual + d_action_embed + d_proprio_embed
        self._build_query_decoder(
            d_model, d_visual, d_visual + d_proprio_embed, num_queries, coord_and_delta_heads=False, sub_patch_head=True
        )
        if use_history and num_frames > 1:
            self.history_fusion = HistoricalInformationFusion(d_model, num_frames)
        self.localizer = DynamicLocalizationNetwork(d_visual, d_action_embed + d_proprio_embed)
        self._drop_query_decoder()

        self.checkpoint = torch.load(ckpt_path, map_location="cpu")
        missing, _ = self.load_state_dict(self.checkpoint["predictor"], strict=False)
        if missing:
            raise ValueError(f"{ckpt_path} has no localizer weights: missing {missing}")
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.localizer.frozen = True

    def train(self, mode=True):
        """Always stays in eval mode."""
        return super().train(False)

    def forward(self, z):
        """z: (B, T, N, D) history -> (B, N) number of changing sub-patches (0..4) per patch."""
        if self.use_history:
            z = self.history_fusion(z)
        logits = self.localizer(z)
        return (logits.sigmoid() > 0.5).sum(-1).float()


class DDP_Predictor(_QueryDecoderMixin, nn.Module):
    """DDP-WM predictor for all three training stages.

    localization:       history fusion + localizer, returns (B, 4N) sub-patch logits.
    primary_predictor:  ViT over the selected foreground tokens, returns (B, N, d_visual + d_proprio)
                        next-frame features (zero outside the selection).
    lrm:                primary_predictor plus the LRM for the background patches; ``init_ckpt`` (a stage-2
                        checkpoint) is loaded and everything except ``lrm`` is frozen.
    In stages 2/3 the frozen history fusion and localizer are attached by the world model.
    Extra keyword arguments from the hydra config are accepted and ignored.
    """

    def __init__(
        self,
        d_visual=384,
        d_action_embed=10,
        d_proprio_embed=10,
        num_frames=3,
        training_stage="lrm",
        use_history=True,
        init_ckpt=None,
        num_queries=32,
        **unused,
    ):
        super().__init__()
        assert training_stage in STAGES, f"training_stage must be one of {STAGES}"
        self.training_stage = training_stage
        self.use_history = use_history
        self.d_visual = d_visual
        self.d_proprio_embed = d_proprio_embed
        d_model = d_visual + d_action_embed + d_proprio_embed
        self._build_query_decoder(
            d_model, d_visual, d_visual + d_proprio_embed, num_queries,
            coord_and_delta_heads=True, sub_patch_head=(training_stage == "localization"),
        )

        if training_stage == "localization":
            if use_history and num_frames > 1:
                self.history_fusion = HistoricalInformationFusion(d_model, num_frames)
            self.localizer = DynamicLocalizationNetwork(d_visual, d_action_embed + d_proprio_embed)
            self._drop_query_decoder()
            return

        self.primary_predictor = SparsePrimaryDynamicsPredictor(d_model)
        self._drop_query_decoder()
        if training_stage == "lrm":
            if init_ckpt is not None:
                self.load_state_dict(torch.load(init_ckpt, map_location="cpu")["predictor"], strict=False)
            self.lrm = LowRankCorrectionModule(d_model=d_model, n_patches=NUM_PATCHES)
            for p in self.parameters():
                p.requires_grad = False
            for p in self.lrm.parameters():
                p.requires_grad = True
            self.primary_predictor.eval()

    def forward(self, z_history, motion_mask=None):
        """
        z_history: (B, T, N, D) concatenated visual / proprio / action embeddings.
        motion_mask: (B, N) token selection (stages 2 and 3).
        """
        if self.use_history:
            z_history = self.history_fusion(z_history)

        if self.training_stage == "localization":
            logits = self.localizer(z_history)  # (B, N, 4)
            return rearrange(logits, "b (h w) (p1 p2) -> b (h p1 w p2)", h=PATCH_GRID, w=PATCH_GRID, p1=2, p2=2)

        mask = motion_mask > 0.01
        x = z_history[:, -1]  # (B, N, D)
        B, N, D = x.shape
        mem_tokens = self.primary_predictor(x, mask)  # (B, k, D)
        pred = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        pred[mask] = mem_tokens.view(-1, D)
        if self.training_stage == "lrm":
            _, idx_tgt = self.lrm.mask_to_indices(mask)
            idx_tgt = idx_tgt.unsqueeze(-1).expand(-1, -1, D)
            updated_tgt = self.lrm(tgt_tokens=x.gather(1, idx_tgt), mem_tokens=mem_tokens, mem_mask=mask)
            pred.scatter_(1, idx_tgt, updated_tgt)
        return pred[..., : self.d_visual + self.d_proprio_embed]
