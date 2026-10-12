import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from models.ddp_predictor import dilate_mask
from utils import move_to_device

PIXEL_DIFF_THRESH = 0.1
USE_MASKED_HEATMAP = 1  # mask the heatmaps in the evaluator's plots (the planning cost uses use_cost_mask)

metric = nn.MSELoss(reduction="none")


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        use_cost_mask=True,
        mask_dilation=8,
        **kwargs,
    ):
        """use_cost_mask: restrict the visual cost to the patches where start and goal differ (Sparse MPC Cost Mask);
        otherwise the plain objective_fn is used.
        mask_dilation: dilation of that mask on the patch grid, 0 (off), 4 or 8 connectivity."""
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.use_cost_mask = use_cost_mask
        self.mask_dilation = mask_dilation

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def _compute_visual_patch_mask_last(self, z_obs_pred, z_obs_tgt, start_visual, goal_visual):
        """
        Args:
            z_obs_pred: dict with 'visual': (B, T, P, D), used for getting P
            z_obs_tgt:  unused (kept for signature consistency / future extension)
            start_visual: (B0, T0, C, H, W) or (B0, C, H, W)
            goal_visual:  (B0, T0, C, H, W) or (B0, C, H, W)

        Returns:
            patch_mask: (B0, P) float tensor in {0,1}, where 1 indicates this patch has enough pixel diffs.
        """
        # take first frame from start and goal
        if start_visual.dim() == 5:
            start_img = start_visual[:, 0]  # (B0,C,H,W)
        elif start_visual.dim() == 4:
            start_img = start_visual
        else:
            raise ValueError(f"Unexpected start_visual shape: {tuple(start_visual.shape)}")

        if goal_visual.dim() == 5:
            goal_img = goal_visual[:, 0]  # (B0,C,H,W)
        elif goal_visual.dim() == 4:
            goal_img = goal_visual
        else:
            raise ValueError(f"Unexpected goal_visual shape: {tuple(goal_visual.shape)}")

        # pixel diff mask
        diff = (start_img - goal_img).abs().mean(dim=1)  # (B0,H,W)
        bsz, H, W = diff.shape

        # patch grid from latent token count
        P = int(z_obs_pred["visual"].shape[2])
        h_p = int(math.sqrt(P))
        while h_p > 1 and (P % h_p != 0):
            h_p -= 1
        w_p = P // h_p

        pooled = F.adaptive_avg_pool2d(
            diff.unsqueeze(1), output_size=(h_p, w_p)
        )  # (B0,1,hp,wp)

        patch_active = (pooled >= PIXEL_DIFF_THRESH).float()  # (B0,1,hp,wp)
        patch_active = patch_active.view(bsz, -1)  # (B0,P) if grid matches perfectly
        patch_active = dilate_mask(patch_active, connectivity=self.mask_dilation)

        # safety: ensure length matches P
        if patch_active.shape[1] != P:
            if patch_active.shape[1] > P:
                patch_active = patch_active[:, :P]
            else:
                patch_active = F.pad(patch_active, (0, P - patch_active.shape[1]), value=0.0)

        return patch_active

    def _masked_visual_loss_last(self, z_obs_pred, z_obs_tgt, patch_mask=None):
        """
        Args:
            z_obs_pred: dict with 'visual': (B, T, P, D), 'proprio': (B, T, ...)
            z_obs_tgt:  dict with 'visual': (B, 1, P, D), 'proprio': (B, 1, ...)
            patch_mask: (B, P) float in {0,1}; if None -> fallback to unmasked visual loss

        Returns:
            loss_visual_masked: (B,)
        """
        pred_last = z_obs_pred["visual"][:, -1:]  # (B,1,P,D)
        tgt = z_obs_tgt["visual"]  # (B,1,P,D)

        per_elem = metric(pred_last, tgt)  # (B,1,P,D)

        if patch_mask is None:
            # fallback to original behavior: mean over (T,P,D) with T=1 here
            return per_elem.mean(dim=tuple(range(1, per_elem.ndim)))

        per_patch = per_elem.mean(dim=-1)  # (B,1,P)
        patch_mask = patch_mask.to(device=per_patch.device, dtype=per_patch.dtype).unsqueeze(1)  # (B,1,P)

        masked_sum = (per_patch * patch_mask).sum(dim=-1)  # (B,1)
        denom = patch_mask.sum(dim=-1).clamp(min=1.0)  # (B,1)
        loss = (masked_sum / denom).mean(dim=1)  # (B,)

        return loss

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        # Simulator-backed world models (if any) are synced to the initial states.
        state0_batch = None
        if self.evaluator is not None and hasattr(self.evaluator, "get_init_cond"):
            _, state0_batch = self.evaluator.get_init_cond()
        if state0_batch is not None and hasattr(self.wm, "assign_init_state_batch"):
            self.wm.assign_init_state_batch(state0_batch)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself

                if state0_batch is not None and hasattr(self.wm, "set_current_state"):
                    self.wm.set_current_state(state0_batch[traj])
                    self.wm.set_current_state_index(traj)

                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                if self.use_cost_mask:
                    # compute traj-level patch mask from (start frame, goal frame), then broadcast to num_samples
                    start_visual_1 = trans_obs_0["visual"][traj : traj + 1]  # (1, t0, C, H, W) or (1,C,H,W)
                    goal_visual_1 = trans_obs_g["visual"][traj : traj + 1]   # (1, t0, C, H, W) or (1,C,H,W)
                    patch_mask_1 = self._compute_visual_patch_mask_last(
                        i_z_obses, cur_z_obs_g, start_visual=start_visual_1, goal_visual=goal_visual_1
                    )  # (1,P)
                    patch_mask = patch_mask_1.repeat(self.num_samples, 1)  # (num_samples,P)

                    # 1) masked visual loss (only where start-goal pixel differs)
                    loss_visual_masked = self._masked_visual_loss_last(i_z_obses, cur_z_obs_g, patch_mask=patch_mask)

                    # 2) keep original proprio weighting via objective_fn (force visual loss = 0)
                    if self.objective_fn is None:
                        loss = loss_visual_masked
                    else:
                        goal_visual_full = cur_z_obs_g["visual"].expand(
                            -1, i_z_obses["visual"].shape[1], -1, -1
                        )
                        i_z_obses_for_proprio = dict(i_z_obses)
                        i_z_obses_for_proprio["visual"] = goal_visual_full
                        loss_proprio_weighted = self.objective_fn(i_z_obses_for_proprio, cur_z_obs_g)

                        loss = loss_visual_masked + loss_proprio_weighted
                else:
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        return mu, np.full(n_evals, np.inf)  # all actions are valid
