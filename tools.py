"""Utilities for inspecting predictions with the original DINO-WM checkpoint.

The DINO-WM PushT checkpoint contains a VQ-VAE decoder that maps DINOv2 patch features back to images.
``load_feature_decoder`` loads it (plan.py does this before building the world model when
``feature_decoder_model`` is set) and ``decode_features`` renders predicted features with it.
"""
from pathlib import Path

import hydra
import torch
from einops import rearrange
from omegaconf import OmegaConf

MODEL_KEYS = ("encoder", "predictor", "decoder", "proprio_encoder", "action_encoder")

_feature_decoder = None


def load_dinowm_model(model_dir, device):
    """Load an original DINO-WM checkpoint, whose modules are stored as pickled objects."""
    model_dir = Path(model_dir)
    cfg = OmegaConf.load(model_dir / "hydra.yaml")
    ckpt = torch.load(model_dir / "checkpoints" / "model_latest.pth", map_location=device, weights_only=False)
    modules = {k: v.to(device) for k, v in ckpt.items() if k in MODEL_KEYS}
    if "encoder" not in modules:  # DINO-WM does not store the frozen DINOv2 encoder
        modules["encoder"] = hydra.utils.instantiate(cfg.encoder).to(device)
    model = hydra.utils.instantiate(
        cfg.model,
        **modules,
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.get("num_action_repeat", 1),
        num_proprio_repeat=cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model


def load_feature_decoder(model_dir, device):
    """Load the image decoder of a DINO-WM checkpoint for decode_features."""
    global _feature_decoder
    _feature_decoder = load_dinowm_model(model_dir, device).decoder
    return _feature_decoder


@torch.no_grad()
def decode_features(z_visual):
    """z_visual: (b, t, num_patches, 384) DINOv2 patch features -> images (b, t, 3, 224, 224) in [-1, 1]."""
    if _feature_decoder is None:
        raise RuntimeError("Call load_feature_decoder() first.")
    visual, _ = _feature_decoder(z_visual)
    return rearrange(visual, "(b t) c h w -> b t c h w", t=z_visual.shape[1])
