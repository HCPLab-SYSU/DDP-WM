import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from custom_resolvers import replace_slash  # noqa: F401  (registers the resolver used in plan_*.yaml)
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
# W&B logs are kept locally unless WANDB_MODE is set (e.g. WANDB_MODE=online)
os.environ.setdefault("WANDB_MODE", "offline")


def _resolve_train_cfg_path(model_dir: Path) -> Path:
    """Find the training config saved next to a checkpoint."""
    candidates = [
        model_dir / "hydra.yaml",
        model_dir / ".hydra" / "config.yaml",
        model_dir / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find training config in: {model_dir}. Tried: " + ", ".join(str(x) for x in candidates)
    )


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(cfg_dict["objective"])

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(self.wandb_run, wandb.sdk.lib.disabled.RunDisabled):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # the planning horizon equals the goal horizon
        from planning.mpc import MPCPlanner

        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(traj_len=2)
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(self.eval_seed)
            if self.env_name == "deformable_env":  # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(
                traj_len=self.frameskip * self.goal_H + 1
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = np.array([x[0] for x in states])
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(self.eval_seed, init_state, exec_actions.numpy())
            self.obs_0 = {key: np.expand_dims(arr[:, 0], axis=1) for key, arr in rollout_obses.items()}
            self.obs_g = {key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()}
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        print(f"Dumped plan targets to {os.path.abspath('plan_targets.pkl')}")

    def perform_planning(self):
        actions_init = self.gt_actions if self.debug_dset_init else None
        actions, action_len = self.planner.plan(obs_0=self.obs_0, obs_g=self.obs_g, actions=actions_init)
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (value.item() if isinstance(value, (np.float32, np.int32, np.int64)) else value)
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_model(model_ckpt, train_cfg, num_action_repeat, device, motion, feature_decoder_dir=None):
    """Instantiate the world model from its training config and load a state_dict checkpoint.

    ``motion`` holds the per-environment DDP-WM settings (see conf/motion). The frozen history fusion
    and localizer that the checkpoint carries are loaded by the world model.
    """
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {model_ckpt}")
    payload = torch.load(model_ckpt, map_location=device)
    if "dataset_info" not in payload:
        raise ValueError(f"Checkpoint {model_ckpt} has no 'dataset_info'; it was not saved by train.py.")
    if not train_cfg.has_predictor:
        raise ValueError("Predictor is required for planning.")
    log.info(f"Loading state_dict from epoch {payload.get('epoch', 'N/A')}: {model_ckpt}")

    proprio_emb_dim = train_cfg.proprio_emb_dim
    action_emb_dim = train_cfg.action_emb_dim
    encoder = hydra.utils.instantiate(train_cfg.encoder)
    proprio_encoder = hydra.utils.instantiate(
        train_cfg.proprio_encoder, in_chans=payload["dataset_info"]["proprio_dim"], emb_dim=proprio_emb_dim
    )
    action_encoder = hydra.utils.instantiate(
        train_cfg.action_encoder, in_chans=payload["dataset_info"]["action_dim"], emb_dim=action_emb_dim
    )
    predictor = hydra.utils.instantiate(
        train_cfg.predictor,
        num_frames=train_cfg.num_hist,
        d_visual=encoder.emb_dim,
        d_action_embed=proprio_emb_dim * train_cfg.num_proprio_repeat,
        d_proprio_embed=action_emb_dim * train_cfg.num_action_repeat,
        training_stage=train_cfg.model.training_stage,
        use_history=motion["use_history"],
        init_ckpt=None,  # the weights come from model_ckpt
        num_queries=motion["num_motion_tokens"],
    )
    decoder = None
    if train_cfg.get("has_decoder", False):
        decoder = hydra.utils.instantiate(train_cfg.decoder, emb_dim=encoder.emb_dim)

    for name, module in [
        ("encoder", encoder), ("proprio_encoder", proprio_encoder), ("action_encoder", action_encoder), ("decoder", decoder)
    ]:
        if module is not None and name in payload:
            module.load_state_dict(payload[name])
    # the history fusion and localizer keys are loaded by the world model (FrozenLocalizer)
    predictor.load_state_dict(payload["predictor"], strict=False)

    if feature_decoder_dir:
        # image decoder of the original DINO-WM checkpoint, for rendering predicted features (tools.decode_features)
        import tools

        tools.load_feature_decoder(feature_decoder_dir, device)

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
        motion=motion,
        ckpt_path=model_ckpt,
        proprio_dim=proprio_emb_dim,
        action_dim=action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    print(cfg_dict)
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict)
        wandb.run.name = output_dir.split("plan_outputs/")[-1]
    else:
        wandb_run = None

    # model_path points straight at a run directory; otherwise use ckpt_base_path/outputs/model_name
    model_path = cfg_dict.get("model_path", None)
    if model_path is None or str(model_path).strip() == "":
        model_path = f"{cfg_dict['ckpt_base_path']}/outputs/{cfg_dict['model_name']}/"
    model_dir = Path(str(model_path)).expanduser()
    if not model_dir.is_absolute():
        # hydra changes the working directory, so resolve relative paths against the original one
        model_dir = Path(hydra.utils.get_original_cwd()) / model_dir
    model_dir = model_dir.resolve()

    model_cfg = OmegaConf.load(_resolve_train_cfg_path(model_dir))
    # DDP-WM settings: those saved with the model, optionally overridden by a `motion` block in the planning config
    motion = OmegaConf.to_container(
        OmegaConf.merge(model_cfg.get("motion", {}), cfg_dict.get("motion") or {}), resolve=True
    )

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.get("num_action_repeat", 1)

    # checkpoint: checkpoint_name if given, else model_{model_epoch}.pth (latest -> model_latest.pth)
    checkpoint_name = cfg_dict.get("checkpoint_name", None)
    if checkpoint_name is not None and str(checkpoint_name).strip() != "":
        model_ckpt = model_dir / "checkpoints" / str(checkpoint_name)
    else:
        model_ckpt = model_dir / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"

    model = load_model(
        model_ckpt, model_cfg, num_action_repeat, device=device, motion=motion,
        feature_decoder_dir=cfg_dict.get("feature_decoder_model"),
    )

    # use dummy vector envs for wall and the deformable envs
    if model_cfg.env.name == "deformable_env":
        # PyFleX is a process-wide singleton and crashes when several scenes are created in one process,
        # so a single simulator serves all evaluation instances
        from env.serial_vector_env import LazySerialVectorEnv

        env = LazySerialVectorEnv(
            gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs), cfg_dict["n_evals"]
        )
    elif model_cfg.env.name == "wall":
        from env.serial_vector_env import SerialVectorEnv

        env = SerialVectorEnv(
            [
                gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        from env.venv import SubprocVectorEnv

        env = SubprocVectorEnv(
            [
                lambda: gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )
    return plan_workspace.perform_planning()


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    OmegaConf.resolve(cfg)
    cfg_dict = cfg_to_dict(cfg)
    # hydra changes the working directory; make checkpoint paths absolute
    cfg_dict["ckpt_base_path"] = hydra.utils.to_absolute_path(cfg_dict["ckpt_base_path"])
    if cfg_dict.get("feature_decoder_model"):
        cfg_dict["feature_decoder_model"] = hydra.utils.to_absolute_path(cfg_dict["feature_decoder_model"])
    # results are written to logs.json; W&B logging is optional (wandb_logging=true)
    cfg_dict["wandb_logging"] = bool(cfg.get("wandb_logging", False))
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
