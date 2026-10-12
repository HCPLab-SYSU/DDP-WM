import os
import math
import torch
import torch.nn.functional as F
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils
from .cem import PIXEL_DIFF_THRESH, USE_MASKED_HEATMAP, dilate_mask

import json


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types.
    See: https://stackoverflow.com/a/47626762
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip
        self.use_masked_heatmap = USE_MASKED_HEATMAP

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def _latent_diff_to_heatmap(self, i_z_visual, goal_z_visual, out_hw, patch_mask=None, eps=1e-8):
        """
        Compute per-patch L2 diff between predicted latent visual tokens and goal latent visual tokens,
        then upsample to image-space heatmap.

        Args:
            i_z_visual: (b, t, p, d)
            goal_z_visual: (b, 1, p, d) or (b, p, d)
            out_hw: (H, W)
            patch_mask: optional (b, t, p) or (b, 1, p), float in {0,1};
                        will be applied BEFORE normalization by setting masked diffs to 0.

        Returns:
            heat_rgb: (b, t, 3, H, W) in [0, 1]
            heat_scalar: (b, t, 1, H, W) in [0, 1]
        """
        if goal_z_visual.dim() == 3:
            goal_z_visual = goal_z_visual.unsqueeze(1)

        # (b, t, p, d)
        diff = i_z_visual - goal_z_visual
        diff_mag = torch.norm(diff.float(), dim=-1)  # (b, t, p)

        # apply mask FIRST: masked parts are set to 0 (not "no heatmap")
        if patch_mask is not None:
            if patch_mask.dim() == 3:
                pm = patch_mask
            elif patch_mask.dim() == 3:  # kept for clarity
                pm = patch_mask
            elif patch_mask.dim() == 2:
                pm = patch_mask.unsqueeze(1)
            else:
                pm = patch_mask
            pm = pm.to(device=diff_mag.device, dtype=diff_mag.dtype)
            diff_mag = diff_mag * pm

        # normalize per-row (per b) across all timesteps and patches -> [0,1]
        minv = diff_mag.amin(dim=(1, 2), keepdim=True)  # (b,1,1)
        maxv = diff_mag.amax(dim=(1, 2), keepdim=True)  # (b,1,1)
        diff_norm = (diff_mag - minv) / (maxv - minv + eps)  # (b, t, p)

        bsz, tsteps, p = diff_norm.shape

        # infer patch grid (h_p, w_p)
        h_p = int(math.sqrt(p))
        w_p = h_p
        if h_p * w_p != p:
            h_p = int(math.sqrt(p))
            while h_p > 1 and (p % h_p != 0):
                h_p -= 1
            w_p = p // h_p

        diff_grid = diff_norm.reshape(bsz * tsteps, 1, h_p, w_p)
        heat_scalar = F.interpolate(
            diff_grid, size=out_hw, mode="nearest"
        ).reshape(bsz, tsteps, 1, out_hw[0], out_hw[1])

        # jet-like colormap approximation
        x = heat_scalar.clamp(0, 1)
        r = (1.5 - torch.abs(4 * x - 3)).clamp(0, 1)
        g = (1.5 - torch.abs(4 * x - 2)).clamp(0, 1)
        b = (1.5 - torch.abs(4 * x - 1)).clamp(0, 1)
        heat_rgb = torch.cat([r, g, b], dim=2)

        return heat_rgb, heat_scalar




    def _overlay_heatmap(self, imgs, heat_rgb, alpha=0.3):
        """
        Args:
            imgs: (b, t, 3, H, W) in [0,1] or [-1,1]
            heat_rgb: (b, t, 3, H, W) in [0,1]
        Returns:
            blended: same range as imgs
        """
        if heat_rgb is None:
            return imgs

        heat_rgb = heat_rgb.to(device=imgs.device, dtype=imgs.dtype)

        in_01 = imgs.min().item() >= 0.0
        if in_01:
            base = imgs.clamp(0, 1)
            blended = (1 - alpha) * base + alpha * heat_rgb
            return blended.clamp(0, 1)

        base = ((imgs.clamp(-1, 1) + 1) / 2)
        blended = (1 - alpha) * base + alpha * heat_rgb
        blended = blended.clamp(0, 1)
        return blended * 2 - 1

    def _compute_heatmap_patch_mask_img(self, start_visual, goal_visual, num_patches, tsteps=1, return_patch_mask=False):
        """
        Args:
            start_visual: (B, 1, C, H, W) or (B, C, H, W)  initial frame visual (same pixel domain as goal_visual)
            goal_visual:  (B, 1, C, H, W) or (B, C, H, W)  goal frame visual (same pixel domain as start_visual)
            num_patches:  int (P)
            tsteps:       int, rollout length in latent time steps (used to broadcast the fixed mask across time)
            return_patch_mask: if True, also return patch_mask (B,T,P)

        Returns:
            mask_img: (B, T, 1, H, W) float in {0,1}
            patch_mask (optional): (B, T, P) float in {0,1}
        """
        if not torch.is_tensor(start_visual):
            start_visual = torch.as_tensor(start_visual)
        if not torch.is_tensor(goal_visual):
            goal_visual = torch.as_tensor(goal_visual)

        # take first frame if time dimension exists
        if start_visual.dim() == 5:
            start_img = start_visual[:, 0]  # (B,C,H,W)
        elif start_visual.dim() == 4:
            start_img = start_visual
        else:
            raise ValueError(f"Unexpected start_visual shape: {tuple(start_visual.shape)}")

        if goal_visual.dim() == 5:
            goal_img = goal_visual[:, 0]  # (B,C,H,W)
        elif goal_visual.dim() == 4:
            goal_img = goal_visual
        else:
            raise ValueError(f"Unexpected goal_visual shape: {tuple(goal_visual.shape)}")

        if start_img.shape != goal_img.shape:
            raise ValueError(
                f"start_img and goal_img shapes must match, got {tuple(start_img.shape)} vs {tuple(goal_img.shape)}"
            )

        B, C, H, W = start_img.shape

        # per-pixel mean abs diff
        diff = (start_img - goal_img).abs().mean(dim=1)  # (B,H,W)

        P = int(num_patches)
        h_p = int(math.sqrt(P))
        while h_p > 1 and (P % h_p != 0):
            h_p -= 1
        w_p = P // h_p

        pooled = F.adaptive_avg_pool2d(diff.unsqueeze(1), output_size=(h_p, w_p))  # (B,1,hp,wp)

        # threshold in patch space (avg abs diff per patch)
        patch_active = (pooled >= PIXEL_DIFF_THRESH).float()  # (B,1,hp,wp)

        # dilate in patch grid (spatial)
        patch_active_flat = patch_active.reshape(B, -1)  # (B,P')
        patch_active_flat = dilate_mask(patch_active_flat, connectivity=8)  # (B,P')
        patch_active = patch_active_flat.reshape(B, 1, h_p, w_p)

        # image-space mask (fixed over time, then broadcast)
        mask_img_1 = F.interpolate(patch_active, size=(H, W), mode="nearest")  # (B,1,H,W)
        mask_img = mask_img_1.unsqueeze(1).expand(B, int(tsteps), 1, H, W)  # (B,T,1,H,W)

        if not return_patch_mask:
            return mask_img

        patch_mask_1 = patch_active.reshape(B, 1, h_p, w_p).reshape(B, -1)  # (B,P')
        if patch_mask_1.shape[-1] != P:
            if patch_mask_1.shape[-1] > P:
                patch_mask_1 = patch_mask_1[..., :P]
            else:
                patch_mask_1 = F.pad(patch_mask_1, (0, P - patch_mask_1.shape[-1]), value=0.0)

        patch_mask = patch_mask_1.unsqueeze(1).expand(B, int(tsteps), P)  # (B,T,P)

        return mask_img, patch_mask


    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        
        # simulator-backed world models (if any) start from the batch of initial states (B, ...)
        if hasattr(self.wm, "assign_init_state_batch"):
            self.wm.assign_init_state_batch(self.state_0)

        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
            goal_z_obs = self.wm.encode_obs(trans_obs_g)

        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        path_prefix = os.path.dirname(filename)
        name = os.path.basename(filename)
        state_dir = os.path.join(path_prefix, "state")
        os.makedirs(state_dir, exist_ok=True)
        filename = os.path.join(state_dir, name)

        np.save(f"{filename}_actions.npy", actions.cpu().numpy())
        with open(f"{filename}_logs.json", "w") as f:
            json.dump(logs, f, cls=NumpyEncoder, indent=4)
        with open(f"{filename}_successes.json", "w") as f:
            json.dump(successes, f, cls=NumpyEncoder, indent=4)

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals_list = []
            for i in range(i_z_obses["visual"].size(1)):
                i_visual = self.wm.decode_obs(
                    {"visual": i_z_obses["visual"][:, i : i + 1], "proprio": i_z_obses["proprio"][:, i : i + 1]}
                )[0]["visual"].cpu()
                i_visuals_list.append(i_visual)
            i_visuals = torch.cat(i_visuals_list, dim=1)
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                i_z_visual=i_z_obses["visual"],
                goal_z_visual=goal_z_obs["visual"],
                action_len=action_len,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states


    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        return logs, successes

    def _plot_rollout_compare(
        self,
        e_visuals,
        i_visuals,
        successes,
        i_z_visual=None,
        goal_z_visual=None,
        action_len=None,
        save_video=False,
        filename="",
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, c, h, w)
        i_visuals: (b, t, c, h, w)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        if action_len is not None:
            action_len = action_len[: self.n_plot_samples]

        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)
        if goal_visual.dim() == 4:
            goal_visual = goal_visual.unsqueeze(1)

        heat_rgb = None
        if (i_z_visual is not None) and (goal_z_visual is not None):
            i_z_visual = i_z_visual[: self.n_plot_samples]
            goal_z_visual = goal_z_visual[: self.n_plot_samples]

            out_hw = (i_visuals.shape[-2], i_visuals.shape[-1])

            patch_mask = None
            if self.use_masked_heatmap:
                num_patches = int(i_z_visual.shape[2])
                tsteps = int(i_z_visual.shape[1])

                start_visual = e_visuals[:, :1]
                if start_visual.dim() == 4:
                    start_visual = start_visual.unsqueeze(1)

                _, patch_mask = self._compute_heatmap_patch_mask_img(
                    start_visual=start_visual,
                    goal_visual=goal_visual,
                    num_patches=num_patches,
                    tsteps=tsteps,
                    return_patch_mask=True,
                )
                if action_len is not None:
                    patch_mask = self._mask_traj(patch_mask, action_len + 1)

            heat_rgb, _ = self._latent_diff_to_heatmap(
                i_z_visual=i_z_visual,
                goal_z_visual=goal_z_visual,
                out_hw=out_hw,
                patch_mask=patch_mask,
            )

            # mask heatmap by action_len (latent has action_len+1 states)
            if action_len is not None:
                heat_rgb = self._mask_traj(heat_rgb, action_len + 1)

            # move to same device as decoded visuals (often cpu)
            heat_rgb = heat_rgb.to(i_visuals.device)

        # pad i_visuals (due to frameskip)
        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        # pad heatmap in the same way, then overlay onto predicted visuals
        if heat_rgb is not None:
            heat_rgb = heat_rgb.unsqueeze(2)
            heat_rgb = torch.cat(
                [heat_rgb] + [heat_rgb] * (self.frameskip - 1),
                dim=2,
            )
            heat_rgb = rearrange(heat_rgb, "b t n c h w -> b (t n) c h w")
            heat_rgb = heat_rgb[:, : heat_rgb.shape[1] - (self.frameskip - 1)]
            i_visuals = self._overlay_heatmap(i_visuals, heat_rgb, alpha=0.45)

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
