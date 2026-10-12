import numpy as np
from utils import aggregate_dct

# for debugging environments, and envs that are not compatible with SubprocVectorEnv
class SerialVectorEnv:
    """
    obs, reward, done, info
    obs: dict, each key has shape (num_env, ...)
    reward: (num_env, )
    done: (num_env, )
    info: tuple of length num_env, each element is a dict
    """

    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)

    def sample_random_init_goal_states(self, seed):
        init_state, goal_state = zip(*(self.envs[i].sample_random_init_goal_states(seed[i]) for i in range(self.num_envs)))
        return np.stack(init_state), np.stack(goal_state)
    
    def update_env(self, env_info):
        [self.envs[i].update_env(env_info[i]) for i in range(self.num_envs)]
    
    def eval_state(self, goal_state, cur_state): 
        eval_result = []
        for i in range(self.num_envs):
            env = self.envs[i]
            eval_result.append(env.eval_state(goal_state[i], cur_state[i]))
        eval_result = aggregate_dct(eval_result)
        return eval_result

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (num_envs, H, W, C)
        state: tuple of num_envs dicts
        """
        obs = []
        state = []
        for i in range(self.num_envs):
            env = self.envs[i]
            cur_seed = seed[i]
            cur_init_state = init_state[i]
            o, s = env.prepare(cur_seed, cur_init_state)
            obs.append(o)
            state.append(s)
        obs = aggregate_dct(obs)
        state = np.stack(state)
        return obs, state

    def step_multiple(self, actions):
        """
        actions: (num_envs, T, action_dim)
        obses: (num_envs, T, H, W, C)
        infos: tuple of length num_envs, each element is a dict
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            env = self.envs[i]
            cur_actions = actions[i]
            obs, reward, done, info = env.step_multiple(cur_actions)
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        obses = np.stack(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = tuple(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        obses: (num_envs, T, H, W, C)
        states: (num_envs, T, D)
        proprios: (num_envs, T, D_p)
        """
        obses = []
        states = []
        for i in range(self.num_envs):
            env = self.envs[i]
            cur_seed = seed[i]
            cur_init_state = init_state[i]
            cur_actions = actions[i]
            obs, state = env.rollout(cur_seed, cur_init_state, cur_actions)
            obses.append(obs)
            states.append(state)
        obses = aggregate_dct(obses)
        states = np.stack(states)
        return obses, states


class LazySerialVectorEnv:
    """
    A "virtual vector env" that only holds ONE simulator env instance,
    and multiplexes num_envs logical environments by saving/restoring states.

    Required underlying env API:
      - prepare(seed, init_state) -> (obs, state)
      - rollout(seed, init_state, actions) -> (obs_traj, state_traj)
      - eval_state(goal_state, cur_state) -> dict
    Optional (for faster switching / correct state update after step_multiple):
      - set_state(state)
      - get_state()
      - update_env(env_info)
      - step_multiple(actions) -> (obs, reward, done, info)
    """

    def __init__(self, env, num_envs: int, strict: bool = True):
        self.env = env
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")

        self.strict = bool(strict)

        # per-logical-env cache
        self._seeds = [None] * self.num_envs
        self._states = [None] * self.num_envs
        self._env_infos = [None] * self.num_envs  # optional per-env config

        # fast path (if env supports)
        self._set_state = getattr(env, "set_state", None)
        self._get_state = getattr(env, "get_state", None)
        self._update_env = getattr(env, "update_env", None)

    # -------------------------
    # utilities
    # -------------------------
    def _require_cached(self, i: int):
        if self._seeds[i] is None or self._states[i] is None:
            raise RuntimeError(
                f"Logical env[{i}] has no cached (seed, state). "
                "Call prepare(seed, init_state) first, or provide init_state explicitly."
            )

    def _apply_env_info(self, i: int):
        if self._update_env is None:
            return
        info = self._env_infos[i]
        if info is not None:
            self._update_env(info)

    def _activate(self, i: int, seed=None, state=None):
        """
        Switch the single simulator to logical env i.
        Preference:
          1) env.set_state(state) if available
          2) env.prepare(seed, state) otherwise
        """
        if seed is None or state is None:
            self._require_cached(i)
            seed = self._seeds[i]
            state = self._states[i]

        self._apply_env_info(i)

        if callable(self._set_state):
            # fast state restore
            self._set_state(state)
        else:
            # generic restore via prepare()
            # we intentionally ignore returned obs here
            _, _ = self.env.prepare(seed, state)

    def _update_cache_after_step(self, i: int, info=None):
        """
        Try to update cached state after step_multiple().
        """
        new_state = None

        if callable(self._get_state):
            new_state = self._get_state()
        elif isinstance(info, dict) and ("state" in info):
            new_state = info["state"]
        elif hasattr(self.env, "state"):
            new_state = getattr(self.env, "state")

        if new_state is None:
            if self.strict:
                raise RuntimeError(
                    "Underlying env.step_multiple() did not provide a way to obtain next state. "
                    "Please implement env.get_state()/env.set_state(), or include state in info['state'], "
                    "or set strict=False to skip state updates (not recommended)."
                )
            return

        self._states[i] = new_state

    # -------------------------
    # public API (vectorized)
    # -------------------------
    def update_env(self, env_info):
        """
        Store per-logical-env env_info, applied lazily before running that logical env.
        env_info: list/tuple of length num_envs
        """
        if env_info is None:
            self._env_infos = [None] * self.num_envs
            return
        if len(env_info) != self.num_envs:
            raise ValueError(f"env_info length {len(env_info)} != num_envs {self.num_envs}")
        self._env_infos = list(env_info)

    def sample_random_init_goal_states(self, seed):
        """
        seed: list/array of length num_envs
        """
        if len(seed) != self.num_envs:
            raise ValueError(f"seed length {len(seed)} != num_envs {self.num_envs}")

        init_states = []
        goal_states = []
        for i in range(self.num_envs):
            self._apply_env_info(i)
            s0, sg = self.env.sample_random_init_goal_states(seed[i])
            init_states.append(s0)
            goal_states.append(sg)
        return np.stack(init_states), np.stack(goal_states)

    def eval_state(self, goal_state, cur_state):
        """
        goal_state: (num_envs, ...)
        cur_state:  (num_envs, ...)
        """
        eval_result = []
        for i in range(self.num_envs):
            self._apply_env_info(i)
            eval_result.append(self.env.eval_state(goal_state[i], cur_state[i]))
        return aggregate_dct(eval_result)

    def prepare(self, seed, init_state):
        """
        Reset each logical env with controlled init_state, but using ONLY ONE simulator instance.
        seed:      (num_envs,)
        init_state:(num_envs, ...)
        Returns:
          obs: dict with shape (num_envs, ...)
          state: np.stack(list_of_states) (often dtype=object if state is dict-like)
        """
        if len(seed) != self.num_envs:
            raise ValueError(f"seed length {len(seed)} != num_envs {self.num_envs}")
        if len(init_state) != self.num_envs:
            raise ValueError(f"init_state length {len(init_state)} != num_envs {self.num_envs}")

        obs_list = []
        state_list = []
        for i in range(self.num_envs):
            self._apply_env_info(i)
            o, s = self.env.prepare(seed[i], init_state[i])
            self._seeds[i] = seed[i]
            self._states[i] = s
            obs_list.append(o)
            state_list.append(s)

        obs = aggregate_dct(obs_list)
        state = np.stack(state_list)
        return obs, state

    def rollout(self, seed, init_state, actions):
        """
        seed:      (num_envs,)
        init_state:(num_envs, ...)  (if you pass None, will use cached states)
        actions:   (num_envs, T, action_dim)
        Returns:
          obses:  dict, each key shape (num_envs, T+1, ...)
          states: np.stack of state trajectories, shape (num_envs, T+1, ...) or object
        """
        if len(seed) != self.num_envs:
            raise ValueError(f"seed length {len(seed)} != num_envs {self.num_envs}")
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions batch {actions.shape[0]} != num_envs {self.num_envs}")

        obses_list = []
        states_list = []
        for i in range(self.num_envs):
            self._apply_env_info(i)

            s0 = init_state[i] if init_state is not None else None
            if s0 is None:
                self._require_cached(i)
                s0 = self._states[i]

            obs_traj, state_traj = self.env.rollout(seed[i], s0, actions[i])
            obses_list.append(obs_traj)
            states_list.append(state_traj)

            # cache last state for future switching
            try:
                self._states[i] = state_traj[-1]
            except Exception:
                # if state_traj isn't indexable, keep old cache unless strict
                if self.strict:
                    raise

            self._seeds[i] = seed[i]

        obses = aggregate_dct(obses_list)
        states = np.stack(states_list)
        return obses, states

    def step_multiple(self, actions):
        """
        actions: (num_envs, T, action_dim)
        Returns (same as your old API):
          obses:  (num_envs, T, ...) or dict aggregated (depending on env output)
          rewards:(num_envs, )
          dones:  (num_envs, )
          infos:  tuple(len=num_envs)
        """
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions batch {actions.shape[0]} != num_envs {self.num_envs}")

        obses = []
        rewards = []
        dones = []
        infos = []

        step_fn = getattr(self.env, "step_multiple", None)
        if not callable(step_fn):
            raise AttributeError("Underlying env has no step_multiple(actions). Use rollout() instead.")

        for i in range(self.num_envs):
            self._activate(i)  # switch to cached state
            obs, reward, done, info = step_fn(actions[i])
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            # update cached state if possible
            self._update_cache_after_step(i, info=info)

        # stack/aggregate outputs
        if len(obses) > 0 and isinstance(obses[0], dict):
            obses_out = aggregate_dct(obses)
        else:
            obses_out = np.stack(obses)

        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = tuple(infos)
        return obses_out, rewards, dones, infos
