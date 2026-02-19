# merge_decoder.py
import torch
from pathlib import Path
import logging
import argparse
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def find_config_file(model_dir: Path) -> Path:
    """
    Looking for main configuration file in model directory.
    Compatible with multiple possible paths.
    """
    candidates = [
        model_dir / "hydra.yaml",
        model_dir / ".hydra" / "config.yaml",
        model_dir / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Configuration file not found in directory {model_dir}.")


def merge_and_prepare_for_inference(model_dir: Path, decoder_ckpt_path: Path, output_dir: Path):
    """
    Merge DINO-WM decoder into the specified model checkpoint and update config to prepare for inference.

    Args:
        model_dir (Path): Output directory of your own trained model.
        decoder_ckpt_path (Path): Path to DINO-WM checkpoint containing pre-trained decoder.
        output_dir (Path): New directory path to save merged checkpoint and new config.
    """
    # --- 0. Path and Validity Checks ---
    if not model_dir.is_dir():
        log.error(f"Error: Input model path is not a directory: {model_dir}")
        return

    if not decoder_ckpt_path.exists():
        log.error(f"Error: DINO-WM checkpoint not found: {decoder_ckpt_path}")
        return

    ckpt_path = model_dir / "checkpoints" / "model_latest.pth"
    if not ckpt_path.exists():
        log.error(f"Error: Cannot find in {model_dir} 'checkpoints/model_latest.pth'。")
        return
        
    config_path = find_config_file(model_dir)
    log.info(f"Found config file: {config_path}")

    device = 'cpu'  # Operate on CPU to avoid unnecessary GPU memory usage
    log.info(f"Will load and process checkpoint on '{device}' device.")

    # --- 1. Load your own checkpoint ---
    log.info(f"Loading your model weights from: {ckpt_path}")
    my_checkpoint = torch.load(ckpt_path, map_location=device)
    
    if 'decoder' in my_checkpoint:
        log.warning(f"Warning: Your checkpoint '{ckpt_path.name}' already contains 'decoder' key. It will be overwritten by DINO-WM decoder.")

    # --- 2. Load and Extract Decoder ---
    log.info(f"Loading DINO-WM decoder weights from: {decoder_ckpt_path}")
    dinowm_checkpoint = torch.load(decoder_ckpt_path, map_location=device)

    decoder_payload = dinowm_checkpoint.get('decoder')
    if decoder_payload is None:
        log.error(f"Error: DINO-WM checkpoint '{decoder_ckpt_path.name}' does not contain 'decoder' key.")
        return

    # Compatible with two possible formats: directly a state_dict or a model object
    if isinstance(decoder_payload, dict):
        decoder_state_dict = decoder_payload
        log.info("Directly extracted from DINO-WM checkpoint:  'decoder' state_dict。")
    else:
        decoder_state_dict = decoder_payload.state_dict()
        log.info("Extracted from DINO-WM checkpoint:  'decoder' model and retrieved its state_dict.")
    
    # --- 3. Merge Checkpoints ---
    merged_checkpoint = my_checkpoint.copy()
    merged_checkpoint['decoder'] = decoder_state_dict
    
    # --- 4. Modify Configuration File ---
    log.info("Modifying configuration file for inference...")
    cfg = OmegaConf.load(config_path)
    try:
        OmegaConf.update(cfg, "model.training_stage", "inference", force_add=True)
        OmegaConf.update(cfg, "predictor.training_stage", "inference", force_add=True)
        OmegaConf.update(cfg, "has_decoder", "true", force_add=True)
        log.info("Changed 'model.training_stage' to 'inference'")
    except Exception as e:
        log.error(f"Failed to modify config file: {e}. Please check config file structure.")
        return

    # --- 5. Save new checkpoint and config to new directory ---
    try:
        # Creating output directory structure
        output_checkpoints_dir = output_dir / "checkpoints"
        output_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Saving merged checkpoint
        output_ckpt_path = output_checkpoints_dir / "model_latest.pth"
        torch.save(merged_checkpoint, output_ckpt_path)
        log.info(f"Merged new checkpoint saved to: {output_ckpt_path}")
        
        # Saving modified configuration file
        output_config_path = output_dir / "hydra.yaml"
        OmegaConf.save(config=cfg, f=output_config_path)
        log.info(f"New config file for inference saved to: {output_config_path}")

        log.info(f"✅ Preparation complete! Resources required for inference generated in directory: {output_dir}")

    except Exception as e:
        log.error(f"❌ Error: Failed to save output files: {e}")

def main():
    """Main function, processing command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge DINO-WM decoder into the model and generate a new directory for inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Output directory of your own trained model (e.g.: 'outputs/YYYY-MM-DD/HH-MM-SS')。"
    )
    
    parser.add_argument(
        "--decoder_ckpt",
        type=str,
        required=True,
        help="Path to original DINO-WM checkpoint file containing pre-trained decoder."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to new directory for storing merged model and configuration."
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    decoder_ckpt_path = Path(args.decoder_ckpt)
    output_dir = Path(args.output_dir)
    
    merge_and_prepare_for_inference(model_dir, decoder_ckpt_path, output_dir)


if __name__ == '__main__':
    main()
