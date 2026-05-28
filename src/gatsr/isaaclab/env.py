"""Isaac Lab G1 environment wrapper.

Wraps an Isaac Lab ManagerBasedRLEnv (Unitree G1 velocity-tracking task by
default) so the GATS-R agent sees the same simple API as the CPU toy env:

    env.reset(seed=...)              -> obs (num_envs, obs_dim)
    env.step(actions)                 -> (obs, reward, term, trunc, info)
    env.recover_step(actions)         -> same, but flagged as a recovery edge
    env.physical_state                -> (num_envs, P) proprio tensor
    env.is_fallen()                   -> (num_envs,) bool
    env.is_crashed()                  -> (num_envs,) bool
    env.action_dim, env.obs_dim, env.physical_dim, env.num_envs

The physical_state is the concatenation of base linear velocity, base angular
velocity, base orientation (gravity-projected), and joint positions/velocities
- the proprioceptive features that the CBF and recovery modules actually use.
This is *not* the policy observation; it's a deliberately smaller, structured
view that the layered world model can reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class IsaacLabG1Config:
    task: str = "Isaac-Velocity-Flat-G1-v0"
    num_envs: int = 16
    device: str = "cuda:0"
    use_fabric: bool = True
    fall_tilt: float = 0.6  # rad — base tilt above which we call it "fallen"
    crash_tilt: float = 1.0  # past this the env will terminate anyway
    seed: Optional[int] = None


class IsaacLabG1Env:
    """Thin facade over Isaac Lab's gym env. Must be created *after*
    ``AppLauncher`` has booted Isaac Sim."""

    def __init__(self, cfg: IsaacLabG1Config | None = None):
        self.cfg = cfg if cfg is not None else IsaacLabG1Config()

        # imports deferred so this file can be imported when Isaac Sim is not running
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  (registers gym IDs)
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(
            self.cfg.task,
            device=self.cfg.device,
            num_envs=self.cfg.num_envs,
            use_fabric=self.cfg.use_fabric,
        )
        if self.cfg.seed is not None:
            env_cfg.seed = self.cfg.seed
        self._env = gym.make(self.cfg.task, cfg=env_cfg)
        # First reset so unwrapped attributes are populated.
        obs, _ = self._env.reset()
        self._obs = obs
        self._recovery_mask = torch.zeros(
            self.cfg.num_envs, dtype=torch.bool, device=self.cfg.device
        )

    # ----- API mirror of BalanceBotEnv -------------------------------------

    @property
    def num_envs(self) -> int:
        return self._env.unwrapped.num_envs

    @property
    def device(self) -> torch.device:
        return self._env.unwrapped.device

    @property
    def action_dim(self) -> int:
        shape = self._env.action_space.shape
        # gym vectorized envs report (num_envs, A); take A
        return int(shape[-1])

    @property
    def obs_dim(self) -> int:
        space = self._env.observation_space
        if isinstance(space, dict) or hasattr(space, "spaces"):
            return int(space["policy"].shape[-1])
        return int(space.shape[-1])

    @property
    def physical_dim(self) -> int:
        return self.physical_state.shape[-1]

    # ----- core ops -------------------------------------------------------

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        obs, _ = self._env.reset()
        self._obs = obs
        self._recovery_mask.zero_()
        return self._extract_policy(obs)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(self.device, dtype=torch.float32)
        obs, reward, term, trunc, info = self._env.step(actions)
        self._obs = obs
        self._recovery_mask.zero_()
        return self._extract_policy(obs), reward, term, trunc, info

    def recover_step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Same dynamics as ``step``, but we mark these timesteps as recovery
        in the per-env mask so logging/metrics can attribute them correctly."""
        result = self.step(actions)
        self._recovery_mask.fill_(True)
        return result

    def close(self) -> None:
        self._env.close()

    # ----- proprio / safety queries ---------------------------------------

    @property
    def physical_state(self) -> torch.Tensor:
        """Returns (num_envs, P) tensor: [base_lin_vel(3), base_ang_vel(3),
        projected_gravity(3), joint_pos(N), joint_vel(N)]. Falls back to the
        policy observation tail when the scene articulation isn't yet
        accessible (e.g., immediately after construction)."""
        env = self._env.unwrapped
        robot = self._get_robot(env)
        if robot is None:
            # fallback: take a fixed prefix of the policy obs
            return self._extract_policy(self._obs).clone()
        try:
            lin_vel = robot.data.root_lin_vel_b  # body frame
            ang_vel = robot.data.root_ang_vel_b
            grav = robot.data.projected_gravity_b
            jp = robot.data.joint_pos
            jv = robot.data.joint_vel
            return torch.cat([lin_vel, ang_vel, grav, jp, jv], dim=-1)
        except AttributeError:
            return self._extract_policy(self._obs).clone()

    def is_fallen(self) -> torch.Tensor:
        """A G1 with the projected-gravity z component above this threshold
        (i.e., gravity pointing more sideways than down in body frame) is
        considered fallen. Returns (num_envs,) bool."""
        env = self._env.unwrapped
        robot = self._get_robot(env)
        if robot is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        grav = robot.data.projected_gravity_b
        # in body frame, fully upright has grav ~= (0, 0, -1). When tilted the
        # horizontal component grows. Use horizontal magnitude as the tilt.
        tilt = torch.linalg.norm(grav[..., :2], dim=-1)
        return tilt > torch.sin(torch.as_tensor(self.cfg.fall_tilt, device=self.device))

    def is_crashed(self) -> torch.Tensor:
        env = self._env.unwrapped
        robot = self._get_robot(env)
        if robot is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        grav = robot.data.projected_gravity_b
        tilt = torch.linalg.norm(grav[..., :2], dim=-1)
        return tilt > torch.sin(torch.as_tensor(self.cfg.crash_tilt, device=self.device))

    @property
    def recovery_mask(self) -> torch.Tensor:
        return self._recovery_mask.clone()

    # ----- internals ------------------------------------------------------

    @staticmethod
    def _extract_policy(obs) -> torch.Tensor:
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    @staticmethod
    def _get_robot(env):
        # Isaac Lab manager-based envs expose .scene["robot"] when present
        try:
            return env.scene["robot"]
        except Exception:
            return None
