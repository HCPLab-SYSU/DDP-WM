import os
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Callable, Dict, Tuple
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PositionEmbeddingSine2D(nn.Module):
    """
    a2DfixedSinusoidalposition embedding module.
    it pre-computes the embeddings at initialization，and atforwardreturn when.
    """
    def __init__(self, d_model_visual, H=14, W=14, temperature=10000):
        super().__init__()
        
        if d_model_visual % 2 != 0:
            raise ValueError(f"d_model_visual (got {d_model_visual}) must be an even number，to be assigned toxandy")
        
        dim_t = d_model_visual // 2
        
        y_embed, x_embed = torch.meshgrid(
            torch.arange(H, dtype=torch.float32), 
            torch.arange(W, dtype=torch.float32), 
            indexing="ij"
        )
        
        dim_t_tensor = torch.arange(dim_t, dtype=torch.float32)
        dim_t_tensor = temperature ** (2 * (dim_t_tensor // 2) / dim_t)
        
        pos_x = x_embed[:, :, None] / dim_t_tensor
        pos_y = y_embed[:, :, None] / dim_t_tensor
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # concatenatexandy's embedding
        pos_embed_2d = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1) # (H*W, d_model_visual)
        
        # register as buffer，making it part of the module's state，but not a trainable parameter
        self.register_buffer('pos_embed', pos_embed_2d)

    def forward(self):
        # directly return the pre-computed embeddings
        # shape: (N, d_visual) i.e. (196, 384)
        return self.pos_embed


class MLP(nn.Module):
    """ A simple Multi-Layer Perceptron (MLP) (FFN) """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for n, k in zip([input_dim] + h, h + [output_dim]):
            layers.append(nn.Linear(n, k))
            layers.append(nn.ReLU(inplace=True))
        layers.pop() # Remove the last layer'sReLU
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    
class TransformerSelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) 

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # Self Attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Feed Forward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, **kwargs):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)

class CrossAttentionDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1,
                 batch_first=False, normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


def dilate_mask(mask, connectivity=4):
    """
    pair mask perform dilation operation，support 4 connected or 8 connectivity.
    
    parameter:
    mask: with shape (b, 196) 's float tensor，with value 0 or 1
    connectivity: integer，4 or 8.
                  4 represents cross-shaped dilation (up, down, left, right)
                  8 represents square-shaped dilation (all around)
    
    return:
    dilated_mask: with shape (b, 196) 's after dilation mask
    """
    if connectivity not in [4, 8]:
        return mask
    b, n = mask.shape
    assert n == 196, f"Expected n=196, got {n}"
    
    # 1. reshape to 14x14 's 2D map (b, 1, 14, 14)
    mask_2d = mask.view(b, 1, 14, 14)
    
    # 2. Define the convolution kernel based on parameters
    if connectivity == 4:
        # cross-shaped convolution kernel
        kernel_data = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
    elif connectivity == 8:
        # nine-patch grid (all1) kernel
        kernel_data = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    # create kernel，ensure device and dtype is consistent with the input
    kernel = torch.tensor(kernel_data, dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)

    # 3. Apply convolution (padding=1 keep the size unchanged)
    # As long as there is one point within the range that is 1，the convolution result will be greater than 0
    conv_result = F.conv2d(mask_2d, kernel, padding=1)
    
    # 4. Binarization result (greater than 0 i.e., set to 1)
    dilated_2d = (conv_result > 0).float()
    
    # 5. Restore original shape (b, 196)
    dilated_mask = dilated_2d.view(b, 196)
    
    return dilated_mask

def force_fixed_k_mask(mask, k_min=32):
    """
    adjust mask，such that each row has at least k  True.
    - If the original True < k: randomly select False complete.
    
    Args:
        mask (torch.Tensor): with shape (B, N) 's bool or 0/1 tensor
        k (int): target True the number of (default 32)
        
    Returns:
        torch.Tensor: with shape (B, N) 's bool tensor，each_row sum are all k
    """
    B, N = mask.shape
    if N < k_min:
        raise ValueError(f"sequence length N={N} must be greater than or equal to the threshold k={k_min}")
    k = max(k_min, mask.bool().int().sum(1).max())

    # 1. Generate random noise of the same shape (0, 1)
    noise = torch.rand(B, N, device=mask.device)
    
    # 2. Calculate score
    # mask as True 's base score is 2.0，after adding noise is (2.0, 3.0)
    # mask as False 's base score is 0.0，after adding noise is (0.0, 1.0)
    # This ensures that the original ones are prioritized True，if True if too many, then True randomly select internally，
    # if True if not enough, select after finishing True then at False randomly select internally.
    scores = mask.float() * 2.0 + noise
    
    # 3. Select the one with the highest score k indices
    # values unimportant，We only need indices
    _, indices = torch.topk(scores, k, dim=1)
    
    # 4. generate a new full False mask，and set the selected indices to True
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    # scatter_(dim, index, src) -> will True fill in indices the specified position
    new_mask.scatter_(1, indices, True)
    
    return new_mask

def freeze_module(module: Optional[nn.Module]):
    """
    A general helper function，used to freeze aPyTorchall parameters of the module and set it to evaluation mode.

    Args:
        module (Optional[nn.Module]): The module to freeze.If it isNone，then no action is performed.
    """
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def get_ref(visual, model):
    B=visual.size(0)
    visual = rearrange(visual, "b t ... -> (b t) ...")
    visual = model.encoder_transform(visual)
    visual = rearrange(visual, "(b t) ... -> b t ...", b=B)
    ref = {'current': visual[:,-2], 'next': visual[:,-1]}
    return ref

def get_rollout_z(z, model, visual, T):
    for i in range(1, z.size(1)):
        images_dict = get_ref(visual[:,i-1:i+1], model)
        mask = model.label_generator(images_dict, None)
        mask = model.predictor._process_mask(mask)
        z_t = z[:,i-1]
        z_t_plus_1 = z_t
        mask_expanded = mask.unsqueeze(-1) > 0
        z_t_plus_1 = torch.where(mask_expanded, z_t_plus_1, z_t)
        z[:,i][...,:model.d_state] = z_t_plus_1[...,:model.d_state]
    z_out = z[:,-T:]
    return z_out
