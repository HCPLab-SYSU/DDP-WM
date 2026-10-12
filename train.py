import os
import hydra
import torch
import wandb
import logging
import warnings
import itertools
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from accelerate import Accelerator
from pathlib import Path
from collections import OrderedDict
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
# W&B logs are kept locally unless WANDB_MODE is set (e.g. WANDB_MODE=online)
os.environ.setdefault("WANDB_MODE", "offline")


class Trainer:
    def __init__(self, cfg):
        if cfg.model.training_stage != "localization":
            # stages 2 and 3 train on teacher-forced rollouts over 5 history frames
            cfg.num_hist = 5
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        self.accelerator = Accelerator(log_with="wandb")
        self.device = self.accelerator.device
        log.info(f"rank: {self.accelerator.local_process_index}  device: {self.device}  model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process and not self.cfg.eval_only:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg.get("wandb_run_id", None)
                if wandb_run_id:
                    log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
            self.wandb_run = wandb.init(
                project="dino_wm_debug" if self.cfg.debug else "dino_wm",
                config=wandb_dict,
                id=wandb_run_id,
                resume="allow",
            )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = model_name
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )
        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False,  # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }
        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")
        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        log.info(f"Train encoder, predictor: {self.train_encoder} {self.train_predictor}")

        self._keys_to_save = ["epoch"]
        if self.train_encoder:
            self._keys_to_save += ["encoder", "encoder_optimizer"]
        if self.train_predictor and self.cfg.has_predictor:
            self._keys_to_save += ["predictor", "predictor_optimizer"]
        # action_encoder_optimizer also holds the proprio encoder parameters
        self._keys_to_save += ["action_encoder", "proprio_encoder", "action_encoder_optimizer"]

        self.init_models()
        self.init_optimizers()

        # Decide which checkpoint to load, if any
        model_ckpt_path = None
        if self.cfg.eval_only:
            if not self.cfg.eval_ckpt_path:
                raise ValueError("eval_only=True, but eval_ckpt_path is not specified.")
            model_ckpt_path = Path(self.cfg.eval_ckpt_path)
            if not model_ckpt_path.exists():
                raise FileNotFoundError(f"Evaluation checkpoint not found at: {self.cfg.eval_ckpt_path}")
            log.info(f"Running in EVALUATION-ONLY mode. Loading checkpoint from: {model_ckpt_path}")
        else:
            resume_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
            if resume_ckpt.exists():
                model_ckpt_path = resume_ckpt
                log.info(f"Found checkpoint. Resuming training from: {model_ckpt_path}")

        if model_ckpt_path:
            self.load_ckpt(model_ckpt_path)
        else:
            log.info("No checkpoint found. Starting training from scratch at epoch 0.")

        self.epoch_log = OrderedDict()
        # Optional batch limits for quick tests: +smoke.max_train_batches=N +smoke.max_val_batches=M
        try:
            self.smoke_max_train_batches = int(self.cfg.smoke.max_train_batches)
        except Exception:
            self.smoke_max_train_batches = None
        try:
            self.smoke_max_val_batches = int(self.cfg.smoke.max_val_batches)
        except Exception:
            self.smoke_max_val_batches = None

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {}
        # minimal dataset information needed to rebuild the model for planning
        try:
            ckpt["dataset_info"] = {
                "proprio_dim": self.datasets["train"].proprio_dim,
                "action_dim": self.datasets["train"].action_dim,
            }
        except Exception as e:
            log.error(f"Failed to get dataset dims, saving checkpoint without dataset_info: {e}")

        for k in self._keys_to_save:
            if k == "epoch":
                ckpt[k] = self.epoch
                continue
            obj = self.__dict__[k]
            if hasattr(obj, "module"):
                obj = self.accelerator.unwrap_model(obj)
            ckpt[k] = obj.state_dict()

        torch.save(ckpt, "checkpoints/model_latest.pth")
        torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
        log.info(f"Saved model state_dict to {os.getcwd()}")

    def load_ckpt(self, filename):
        log.info(f"Loading checkpoint state_dict from {filename}...")
        ckpt = torch.load(filename, map_location=self.device)
        not_in_ckpt = []
        for k in self._keys_to_save:
            if k not in ckpt:
                not_in_ckpt.append(k)
                continue
            if k == "epoch":
                self.epoch = ckpt[k]
                continue
            obj = self.__dict__[k]
            try:
                if k.endswith("optimizer"):
                    obj.load_state_dict(ckpt[k])
                else:
                    # models were wrapped by accelerator.prepare; checkpoints hold unwrapped state_dicts
                    self.accelerator.unwrap_model(obj).load_state_dict(ckpt[k])
            except Exception as e:
                log.error(f"Failed to load state_dict for key '{k}', skipping it. Error: {e}")
        if not_in_ckpt:
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)
        log.info(f"Checkpoint loaded. Resuming from epoch {self.epoch}.")

    def init_models(self):
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(self.cfg.encoder)
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.proprio_encoder is None:
            self.proprio_encoder = hydra.utils.instantiate(
                self.cfg.proprio_encoder,
                in_chans=self.datasets["train"].proprio_dim,
                emb_dim=self.cfg.proprio_emb_dim,
            )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        if self.action_encoder is None:
            self.action_encoder = hydra.utils.instantiate(
                self.cfg.action_encoder,
                in_chans=self.datasets["train"].action_dim,
                emb_dim=self.cfg.action_emb_dim,
            )
        action_emb_dim = self.action_encoder.emb_dim
        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process and not self.cfg.eval_only:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        motion = OmegaConf.to_container(self.cfg.motion, resolve=True)
        training_stage = self.cfg.model.training_stage
        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_frames=self.cfg.num_hist,
                    d_visual=self.encoder.emb_dim,
                    d_action_embed=proprio_emb_dim * self.cfg.num_proprio_repeat,
                    d_proprio_embed=action_emb_dim * self.cfg.num_action_repeat,
                    training_stage=training_stage,
                    use_history=motion["use_history"],
                    init_ckpt=self.cfg.ckpt_path if training_stage == "lrm" else None,
                    num_queries=motion["num_motion_tokens"],
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        decoder = None
        if self.cfg.get("has_decoder", False):
            decoder = hydra.utils.instantiate(self.cfg.decoder, emb_dim=self.encoder.emb_dim)

        self.encoder, self.predictor = self.accelerator.prepare(self.encoder, self.predictor)
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=decoder,
            motion=motion,
            ckpt_path=self.cfg.ckpt_path,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )
        self.model.to(self.accelerator.device)

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.training.encoder_lr)
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=self.cfg.training.predictor_lr)
            self.predictor_optimizer = self.accelerator.prepare(self.predictor_optimizer)
            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(self.action_encoder.parameters(), self.proprio_encoder.parameters()),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(self.action_encoder_optimizer)

    def run(self):
        if self.cfg.eval_only:
            log.info(f"--- Evaluating checkpoint from epoch {self.epoch} ---")
            self.accelerator.wait_for_everyone()
            self.val()
            self.accelerator.wait_for_everyone()
            self.logs_flash(epoch=self.epoch)
            return

        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(epoch=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                self.save_ckpt()

    def _history_slices(self, full_obs, full_act):
        """Yield (hist_len, obs, act) windows of 1..num_hist history frames plus the target frame.

        Without history, stage 1 only sees the last history frame and the target.
        """
        T_full = full_obs["visual"].shape[1]
        for hist_len in range(1, self.model.num_hist + 1):
            T_slice = hist_len + 1
            if T_full < T_slice:
                continue
            T_start = T_slice - 2 if not self.model.use_history and self.model.training_stage == "localization" else 0
            obs_slice = {k: v[:, T_start:T_slice, ...] for k, v in full_obs.items()}
            yield hist_len, obs_slice, full_act[:, T_start:T_slice, ...]

    def _log_losses(self, prefix, loss_components):
        loss_components = self.accelerator.gather_for_metrics(loss_components)
        self.logs_update({f"{prefix}{k}": [v.mean().item()] for k, v in loss_components.items()})

    def train(self):
        log_freq = len(self.dataloaders["train"]) // 10
        for i, data in enumerate(tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")):
            if self.smoke_max_train_batches is not None and i > self.smoke_max_train_batches:
                break
            if log_freq > 0 and i % log_freq == log_freq - 1:
                self.logs_flash(self.epoch, step=i, flash=False)

            full_obs, full_act, _ = data
            self.encoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            # Gradients from every history length are accumulated before a single optimizer step.
            for hist_len, obs_slice, act_slice in self._history_slices(full_obs, full_act):
                self.model.train()
                _, _, _, loss, loss_components = self.model(obs_slice, act_slice)
                self.accelerator.backward(loss)
                self._log_losses(f"train_h{hist_len}_", loss_components)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

    def val(self):
        self.model.eval()
        self.accelerator.wait_for_everyone()
        for i, data in enumerate(tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")):
            if self.smoke_max_val_batches is not None and i >= self.smoke_max_val_batches:
                break
            full_obs, full_act, _ = data
            for hist_len, obs_slice, act_slice in self._history_slices(full_obs, full_act):
                self.model.eval()
                with torch.no_grad():
                    _, _, _, loss, loss_components = self.model(obs_slice, act_slice)
                self._log_losses(f"val_h{hist_len}_", loss_components)

    def logs_update(self, logs):
        for key, value in logs.items():
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (count + len(value), total + sum(value))

    def logs_flash(self, epoch, step=-1, flash=True):
        epoch_log = OrderedDict((key, total / count) for key, (count, total) in self.epoch_log.items())
        epoch_log["epoch"] = epoch

        if self.cfg.eval_only:
            summary = f"Validation results for epoch {self.epoch} | Loss: {epoch_log.get('loss_total', 'N/A')}"
        else:
            # report the longest-but-one history length
            max_hist = self.model.num_hist - 1
            train_val = epoch_log.get(f"train_h{max_hist}_loss_total", "N/A")
            val_val = epoch_log.get(f"val_h{max_hist}_loss_total", "N/A")
            summary = f"Epoch {self.epoch} step {step} (h={max_hist}) | Loss: Train={train_val} Val={val_val}"
            if self.accelerator.is_main_process:
                self.wandb_run.log(epoch_log)
        log.info(summary)

        if self.accelerator.is_main_process:
            lines = [summary] + [f"  {key}: {val:.4f}" for key, val in epoch_log.items() if key != "epoch"]
            for line in lines[1:]:
                log.info(line)
            try:
                with open(os.path.join(os.getcwd(), "train.log"), "a") as f:
                    f.write("\n".join(lines) + "\n")
                    f.write(f"--- End of Epoch {self.epoch} Eval ---\n")
            except Exception as e:
                log.warning(f"Failed to write to train.log: {e}")

        if flash:
            self.epoch_log = OrderedDict()


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    if cfg.model.training_stage != "localization" and not cfg.ckpt_path:
        raise ValueError("Stages 2 and 3 need ckpt_path: the stage-1 (or stage-2 for lrm) checkpoint")
    # hydra changes the working directory to the run directory; make the checkpoint path absolute
    if cfg.ckpt_path:
        cfg.ckpt_path = hydra.utils.to_absolute_path(cfg.ckpt_path)
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
