# file: models/ddp_predictor.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, Literal, Optional, Tuple

from .utils import *
from .vit import ViTPredictor


# ====================================================================
#  Stage 1: HistoricalInformationFusionModule (Historical Information Fusion)
# ====================================================================
class HistoricalInformationFusion(nn.Module):
    """
    paper Stage 1: Historical Information Fusion Module.
    Use a single layer of Cross-Attention the history frame information (Z_hist) fuse into the current frame (z_t) in.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, num_frames=3, num_patches=196):
        super().__init__()
        # Instantiate the cross-attention layer
        self.history_encoder = CrossAttentionDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=False,
            normalize_before=False
        )
        
        # Instantiate position encoding parameters (These were originally in Predictor 's __init__ in)
        self.hist_query_pos = nn.Parameter(torch.randn(num_patches, d_model))
        self.hist_mem_pos = nn.Parameter(torch.randn(num_patches, d_model))
        if num_frames > 1:
            self.hist_time_embeds = nn.Parameter(torch.zeros(num_frames - 1, d_model))
        else:
            self.hist_time_embeds = None

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_history (torch.Tensor): the input history sequence, shape [B, T, N, D].
                                      the last frame is the current frame t, preceded by history frames.

        Returns:
            torch.Tensor: current frame features after fusing history information z_t', shape [B, 1, N, D].
        """
        B, t_len, N, D = z_history.shape

        if t_len <= 1:
            # If there is no history information, return the current frame directly, but ensure the computation graph can pass(for gradient checking)
            if self.hist_time_embeds is not None:
                # This is a trick, Ensure that even without history frames, these parameters are also included in the computation graph
                # Their gradients will be0, but it won't cause an error for being unused
                dummy_sum = self.history_encoder.parameters().__next__().sum() * 0 + \
                            self.hist_query_pos.sum() * 0 + \
                            self.hist_mem_pos.sum() * 0 + \
                            self.hist_time_embeds.sum() * 0
                return z_history + dummy_sum
            return z_history

        # prepare Query: current_frame
        current_frame = z_history[:, -1, :, :].permute(1, 0, 2)  # (N, B, D)
        query_pos = self.hist_query_pos.unsqueeze(1).repeat(1, B, 1) # (N, B, D)
        
        # prepare Key/Value: history_frame
        history_frames = rearrange(z_history[:, :-1, :, :], 'b t n d -> (t n) b d') # ((t-1)*N, B, D)
        
        # prepare Key/Value 's position encoding
        hist_len = t_len - 1
        mem_pos_spatial = self.hist_mem_pos.unsqueeze(0).repeat(hist_len, 1, 1) # (t-1, N, D)
        
        if self.hist_time_embeds is not None:
            time_pe = self.hist_time_embeds[:hist_len].unsqueeze(1)
            mem_pos_full = mem_pos_spatial + time_pe
        else:
            mem_pos_full = mem_pos_spatial
        
        mem_pos = rearrange(mem_pos_full, 't n d -> (t n) d').unsqueeze(1).repeat(1, B, 1) # ((t-1)*N, B, D)

        # Execute cross-attention, with an internal residual connection
        encoded_frame = self.history_encoder(
            tgt=current_frame,
            memory=history_frames,
            query_pos=query_pos,
            pos=mem_pos
        )  # (N, B, D)
        
        # Restore dimensions and return
        z_t_prime = encoded_frame.permute(1, 0, 2).unsqueeze(1) # [B, 1, N, D]
        
        return z_t_prime


# ====================================================================
#  Stage 2: DynamicLocalizationNetwork (Dynamic Localization Network)
# ====================================================================
class DynamicLocalizationNetwork(nn.Module):
    """
    paper Stage 2: Dynamic Localization Network.
    A lightweight ViT, receives the fused z_t' and action, predict the changed region mask M.
    """
    def __init__(self, d_visual, d_action_embed, d_proprio_embed, 
                 reduced_dim, num_layers, n_heads, mlp_dim, n_patches_hw, partition_precision, **kwargs):
        super().__init__()
        self.n_patches_hw = n_patches_hw
        
        self.dimensionality_reduction_layer = nn.Linear(d_visual, reduced_dim)
        
        dino_wm_ti_dim = reduced_dim + d_action_embed + d_proprio_embed
        self.dino_wm_ti = ViTPredictor(
            dim=dino_wm_ti_dim, 
            depth=num_layers, 
            heads=n_heads, 
            mlp_dim=mlp_dim,
            num_frames=1, 
            num_patches=n_patches_hw[0] * n_patches_hw[1]
        )
        
        # Determine output dimension based on configuration
        self.cls_head = MLP(dino_wm_ti_dim, dino_wm_ti_dim, partition_precision, num_layers=3)

    def forward(self, z_t_prime_vis: torch.Tensor, z_t_prime_prio: torch.Tensor, z_t_prime_act: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t_prime_vis (torch.Tensor): visual part, shape [B, 1, N, D_vis]
            z_t_prime_prio (torch.Tensor): proprioceptive part, shape [B, 1, N, D_prio]
            z_t_prime_act (torch.Tensor): action part, shape [B, 1, N, D_act]
        
        Returns:
            torch.Tensor: mask's logits, shape [B, N*4] or [B, N]
        """
        # as prio and act expand patch dimension
        B, _, N, _ = z_t_prime_vis.shape

        # Dimensionality reduction and concatenation
        z_vis_reduction = self.dimensionality_reduction_layer(z_t_prime_vis)
        z_reduction = torch.cat([z_vis_reduction, z_t_prime_prio, z_t_prime_act], dim=-1)
        
        # ViT Adjust input format
        z_reduction = rearrange(z_reduction, "b t p d -> b (t p) d")
        
        # through ViT and classification head
        z_reduction = self.dino_wm_ti(z_reduction)
        logits = self.cls_head(z_reduction)
        
        return logits



# ====================================================================
#  Stage 3: SparsePrimaryDynamicsPredictor (Sparse Primary Dynamics Predictor)
# ====================================================================
class SparsePrimaryDynamicsPredictor(nn.Module):
    """
    paper Stage 3: Sparse Primary Dynamics Predictor.
    A powerful Transformer, on the localized sparse foreground token perform self-attention on.
    """
    def __init__(self, d_model, num_layers, n_heads, mlp_dim, **kwargs):
        super().__init__()
        # Used here ViTPredictor, because it is essentially a token operating on the sequence Transformer
        self.dino_wm = ViTPredictor(
            dim=d_model, 
            depth=num_layers, 
            heads=n_heads, 
            mlp_dim=mlp_dim,
            num_frames=1, # Only process the foreground of a single frame
            num_patches=196 
        )

    def forward(self, z_t_prime: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t_prime (torch.Tensor): fused current frame, shape [B, 1, N, D_model].
            mask (torch.Tensor): boolean mask, shape [B, N], True represents foreground.

        Returns:
            torch.Tensor: the predicted foreground features of the next frame, shape [B, K, D_model].
        """
        B, _, N, D = z_t_prime.shape
        z_t_prime = z_t_prime.squeeze(1)

        # 1. Extract foreground using the mask token
        # z_t_prime[mask] will return a (B*K, D) 's flattened tensor
        # We need reshape back (B, K, D)
        K = mask.sum(dim=1)[0].item() # Assuming that for each sample, theKsame
        fg_tokens = z_t_prime[mask].view(B, K, D)

        # 2. Perform self-attention through the primary predictor
        # dino_wm input [B, K, D], output [B, K, D]
        next_fg_tokens = self.dino_wm(fg_tokens)

        return next_fg_tokens

        
# ====================================================================
#  Stage 4: LowRankCorrectionModule (LRM / Low-Rank Correction Module)
# ====================================================================
class LowRankCorrectionModule(nn.Module):
    """
    paper Stage 4: Low-Rank Correction Module (LRM).

    This module is responsible for efficiently updating background features.Its core mechanism is:
    1. background token as Query.
    2. the updated foreground token as Key and Value.
    3. Through a single cross-attention layer, Let each background token "query" changes in the foreground, and adjust itself accordingly.
    4. Use Absolute Position Encoding (APE) as Query and Key/Value provide spatial information.
    """
    def __init__(self, d_model: int, n_heads: int, n_patches: int, dropout: float = 0.0):
        """
        initialize LRM module.

        Args:
            d_model (int): The feature dimension of the model.
            n_heads (int): The number of heads for multi-head attention.
            n_patches (int): patch 's total count (for example 14*14=196).
            dropout (float): Dropout 's probability.
        """
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches

        # 1. Directly instantiate the cross-attention layer
        self.cross_attn = CrossAttentionDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=False,  # PyTorch Transformerlayer expects by default (L, B, D)
            normalize_before=False
        )

        # 2. Create learnable Absolute Position Encoding (APE)
        #    thisAPEwill be used for all 196  patch provide position information
        self.ape = nn.Parameter(torch.randn(n_patches, d_model))

    @staticmethod
    def _mask_to_indices(mem_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A static helper function, convert a boolean mask into foreground and background indices.

        Args:
            mem_mask (torch.Tensor): foreground mask, shape [B, N], Truerepresents foreground.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (foreground index, background index).
        """
        B, N = mem_mask.shape
        device = mem_mask.device
        
        # Create a grid containing all position indices
        pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)

        # Separate indices based on the mask
        idx_mem = pos[mem_mask].view(B, -1)     # (B, K)
        idx_tgt = pos[~mem_mask].view(B, -1)   # (B, N-K)
        
        return idx_mem, idx_tgt

    def forward(self, z_t_prime: torch.Tensor, mask: torch.Tensor, pred_fg_tokens: torch.Tensor) -> torch.Tensor:
        """
        execute LRM 's forward propagation.

        Args:
            z_t_prime (torch.Tensor): fused current frame features, shape [B, N, D].
            mask (torch.Tensor): boolean mask, shape [B, N], True represents foreground.
            pred_fg_tokens (torch.Tensor): the predicted foreground features of the next frame, shape [B, K, D].

        Returns:
            torch.Tensor: updated next-frame background features, shape [B, N-K, D].
        """
        B, N, D = z_t_prime.shape
        K = pred_fg_tokens.shape[1]
        num_bg = N - K
        
        # 1. Extract original background token (as Query)
        bg_tokens = z_t_prime[~mask].view(B, num_bg, D)
        
        # 2. Prepare inputs for cross-attention (tgt, memory, pos, query_pos)
        
        # 2a. Get foreground and background indices
        idx_mem, idx_tgt = self._mask_to_indices(mask)
        
        # 2b. Assign position encodings
        #    - `pos` is memory (foreground) 's position encoding
        #    - `query_pos` is tgt (background) 's position encoding
        ape_mem = self.ape.index_select(0, idx_mem.reshape(-1)).view(B, K, D)
        ape_bg = self.ape.index_select(0, idx_tgt.reshape(-1)).view(B, num_bg, D)
        
        # 2c. Adjust dimensions to match PyTorch Transformer API (L, B, D)
        # Query (background)
        tgt = bg_tokens.transpose(0, 1)          # (N-K, B, D)
        query_pos = ape_bg.transpose(0, 1)       # (N-K, B, D)
        
        # Key/Value (foreground)
        memory = pred_fg_tokens.transpose(0, 1)  # (K, B, D)
        pos = ape_mem.transpose(0, 1)            # (K, B, D)
        
        # 3. Execute cross-attention
        #    background(tgt)query foreground(memory)
        updated_bg_tokens_transposed = self.cross_attn(
            tgt=tgt,
            memory=memory,
            pos=pos,
            query_pos=query_pos
        )
        
        # 4. Restore dimensions to (B, N-K, D) and return
        updated_bg_tokens = updated_bg_tokens_transposed.transpose(0, 1)
        
        return updated_bg_tokens


# ====================================================================
#  Final Assembly: DDP-WM predictor
# ====================================================================
class DDP_Predictor(nn.Module):
    """
    complete DDP-WM predictor, Supports staged initialization and training.
    through `training_stage` parameter controls the instantiation and forward propagation logic of the module.
    """
    def __init__(self, d_visual, d_action_embed, d_proprio_embed, num_frames, num_patches, 
                 training_stage: str = 'inference', **kwargs):
        """
        Args:
            training_stage (str): training stage, optional_values:
                - 'localization': Train only history fusion and localization network.
                - 'primary_predictor': Train the Primary Dynamics Predictor (Freeze preceding modules).
                - 'lrm': Train the Low-Rank Correction Module (Freeze preceding modules).
                - 'inference': inference mode, All modules are loaded and frozen.
        """
        super().__init__()
        
        # --- 1. Save core configuration ---
        self.d_visual = d_visual
        self.d_proprio = d_proprio_embed
        self.d_action = d_action_embed
        self.training_stage = training_stage
        
        d_model = d_visual + d_action_embed + d_proprio_embed
        n_patches_hw = (int(num_patches**0.5), int(num_patches**0.5))

        # --- 2. According to the training stage, Conditionally initialize modules as needed ---
        
        # All modules are by default None
        self.history_fusion: Optional[HistoricalInformationFusion] = None
        self.localizer: Optional[DynamicLocalizationNetwork] = None
        self.primary_predictor: Optional[SparsePrimaryDynamicsPredictor] = None
        self.lrm: Optional[LowRankCorrectionModule] = None
        self.delta_head: Optional[MLP] = None
        
        print(f"Initializing DDP_Predictor in '{self.training_stage}' stage.")

        # Stage 1 & 2: Always required or as a dependency
        if self.training_stage in ['localization', 'primary_predictor', 'lrm', 'inference']:
            self.history_fusion = HistoricalInformationFusion(
                d_model=d_model, num_frames=num_frames, num_patches=num_patches, **kwargs['history_fusion']
            )
            self.localizer = DynamicLocalizationNetwork(
                d_visual=d_visual, d_action_embed=d_action_embed, d_proprio_embed=d_proprio_embed,
                n_patches_hw=n_patches_hw, **kwargs['localizer']
            )

        # Stage 3: at 'primary_predictor' and subsequent stages require
        if self.training_stage in ['primary_predictor', 'lrm', 'inference']:
            self.primary_predictor = SparsePrimaryDynamicsPredictor(
                d_model=d_model, **kwargs['primary_predictor']
            )
        
        # Stage 4: at 'lrm' and subsequent stages require
        if self.training_stage in ['lrm', 'inference']:
            self.lrm = LowRankCorrectionModule(
                d_model=d_model, n_patches=num_patches, **kwargs['lrm']
            )
        
        # --- 3. Freeze pre-trained modules according to the stage ---
        self._freeze_stages()

    def _freeze_stages(self):
        """According to the current training stage, Freeze modules that do not need to be trained."""
        if self.training_stage == 'primary_predictor':
            print("Freezing modules for 'primary_predictor' stage...")
            freeze_module(self.history_fusion)
            freeze_module(self.localizer)
        
        elif self.training_stage == 'lrm':
            print("Freezing modules for 'lrm' stage...")
            freeze_module(self.history_fusion)
            freeze_module(self.localizer)
            freeze_module(self.primary_predictor)

        elif self.training_stage == 'inference':
            print("Freezing all modules for 'inference' stage.")
            freeze_module(self) # Freeze all its own parameters

    def _process_mask(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """Internal helper function, used tologitsconvert to the final binarized mask."""
        mask = (mask_logits.sigmoid() > 0.5)
        if len(mask.shape) == 3: # Handle multi-class output cases
            mask = mask.sum(-1) > 0
        # mask = dilate_mask(mask, connectivity=connectivity)
        mask = force_fixed_k_mask(mask)
        return mask

    def forward(self, z_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        executed in stagesDDP-WMforward propagation.
        according to `self.training_stage` return different intermediate results.
        """
        # --- Stage 1: History Fusion ---
        # This step is always needed, even if it is primary_predictor stage, also needs it as no_grad 's input
        with torch.no_grad() if self.training_stage not in ['localization'] else torch.enable_grad():
            z_t_prime_full = self.history_fusion(z_history) # [B, 1, N, D]
        
        # --- Stage 2: Dynamic Localization ---
        with torch.no_grad() if self.training_stage not in ['localization'] else torch.enable_grad():
            z_vis = z_t_prime_full[..., :self.d_visual]
            z_prio = z_t_prime_full[..., self.d_visual : self.d_visual + self.d_proprio]
            z_act = z_t_prime_full[..., -self.d_action:]
            mask_logits = self.localizer(z_vis, z_prio, z_act)
        
        if self.training_stage == 'localization':
            return {'mask_logits': mask_logits}

        # --- Stage 3: Sparse Primary Dynamics Prediction ---
        with torch.no_grad() if self.training_stage not in ['primary_predictor'] else torch.enable_grad():
            mask = self._process_mask(mask_logits)
            pred_fg_tokens = self.primary_predictor(z_t_prime_full, mask)
        
        if self.training_stage == 'primary_predictor':
            return {'pred_fg': pred_fg_tokens, 'mask': mask}

        # --- Stage 4: Low-Rank Correction ---
        with torch.no_grad() if self.training_stage not in ['lrm'] else torch.enable_grad():
            updated_bg_tokens = self.lrm(z_t_prime_full.squeeze(1), mask, pred_fg_tokens)

        if self.training_stage == 'lrm':
            return {'pred_bg': updated_bg_tokens, 'mask': mask}
        
        # --- Combine foreground and background, form the next frame ---
        z_t_plus_1 = torch.zeros_like(z_t_prime_full)
        z_t_plus_1[mask] = pred_fg_tokens.view(-1, z_t_plus_1.size(-1))
        z_t_plus_1[~mask] = updated_bg_tokens.view(-1, z_t_plus_1.size(-1))

        # --- Inference stage ---
        return {'final_prediction': z_t_plus_1.unsqueeze(1)}



class LabelGenerator(nn.Module):
    """
    A module specifically for generating binarized foreground masks from raw data during training.

    This module, based on the specified mode ('pixel' or 'feature') calculate the change between consecutive frames, 
    Apply threshold, finally generate a
    clean、for supervising downstream tasks(like localization、segmentation)'s boolean mask.

    It does not participate in the model's inference process.

    Args:
        mode (str): 'pixel' or 'feature'.Determines whether to generate the mask based on pixel difference or feature difference.
        pixel_threshold (float): at 'pixel' in mode, The pixel difference norm threshold used to determine significant changes.
        feature_threshold (float): at 'feature' in mode, The feature difference norm threshold used to determine significant changes.
        d_feature (int): at 'feature' in mode, The feature dimension used to calculate the norm.
        grid_h (int): feature_map/of the mask in the height direction patch count.
        grid_w (int): feature_map/of the mask in the width direction patch count.
    """
    def __init__(self,
                 mode: Literal['pixel', 'feature'] = 'pixel',
                 pixel_threshold: float = 0.1,
                 feature_threshold: float = 45.0,
                 d_feature: int = 384,
                 grid_h: int = 14,
                 grid_w: int = 14,
                 partition_precision: int = 4,
                 **kwargs): # Absorb extra configuration parameters
        super().__init__()

        assert mode in ['pixel', 'feature'], "mode must be 'pixel' or 'feature'"
        
        self.mode = mode
        self.pixel_threshold = pixel_threshold
        self.feature_threshold = feature_threshold
        self.d_feature = d_feature
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_patches = grid_h * grid_w
        self.partition_precision = partition_precision

        # set the module to evaluation mode and freeze it, because it does not contain trainable parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, 
                images_dict: Dict[str, torch.Tensor],
                z_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate a foreground mask based on the input data.

        Args:
            images_dict (Dict): a dictionary, should contain 'current' and 'next' key, 
                               corresponds to t and t+1 's image tensor at time.
                               Shape: [B, C, H, W].
            z_dict (Dict): a dictionary, should contain 'current' and 'next' key, 
                           corresponds to t and t+1 's feature tensor at time.
                           Shape: [B, N, D].

        Returns:
            torch.Tensor: the final foreground mask M_final, shape [B, N], dtype=torch.bool.
        """
        B = images_dict['current'].shape[0]
        device = images_dict['current'].device

        # --- step A: Calculate the original change norm map (Norm Map) ---
        if self.mode == 'pixel':
            images_current = images_dict['current']
            images_next = images_dict['next']
            
            # Calculate pixel difference
            pixel_diff = images_next - images_current  # shape: [B, C, H, W]

            # Calculate for each pixel L2 the square of the norm (sum over the channel dimension)
            pixel_norms_sq = torch.sum(pixel_diff.pow(2), dim=1, keepdim=True) # shape: [B, 1, H, W]

            # aggregate pixel-level differences to the patch-level using average pooling patch level
            patch_size_h = images_current.shape[2] // self.grid_h // int(math.sqrt(self.partition_precision))
            patch_size_w = images_current.shape[3] // self.grid_w // int(math.sqrt(self.partition_precision))
            pool_kernel = (patch_size_h, patch_size_w)

            patch_norms_sq = F.avg_pool2d(
                pixel_norms_sq, 
                kernel_size=pool_kernel, 
                stride=pool_kernel
            ) # shape: [B, 1, grid_h, grid_w]

            # take the square root and flatten, get for each patch 's RMS norm
            norms = torch.sqrt(patch_norms_sq).view(B, self.n_patches, -1) # shape: [B, N, _]
            threshold = self.pixel_threshold

        elif self.mode == 'feature':
            z_current = z_dict['current']
            z_next = z_dict['next']
            
            # Calculate feature difference
            delta_z = z_next - z_current # shape: [B, N, D]
            
            # Calculate the feature difference's L2 norm
            norms = torch.norm(delta_z[..., :self.d_feature], p=2, dim=-1).view(B, self.n_patches, -1) # shape: [B, N, _]
            threshold = self.feature_threshold
        
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        # --- step B: Apply threshold ---
        
        # 1. Apply threshold, obtain the original binary mask
        M_final = (norms > threshold) # shape: [B, N], dtype=torch.bool

        return M_final
