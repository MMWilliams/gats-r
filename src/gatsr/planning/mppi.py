"""MPPI baseline: model-predictive path-integral control.

This is the standard TD-MPC2 inner-loop optimizer (Hansen et al., 2024). We
keep the implementation framework-agnostic over the *world model* by accepting
any callable `rollout(state, action_seq) -> (traj, eps)` interface, so the
same planner can drive the analytic L1 model, the L2 ensemble, the layered
orchestrator, or a TD-MPC2-lite baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class MPPIConfig:
    horizon: int = 12
    n_samples: int = 64
    n_iter: int = 2
    n_elite: int = 12
    init_std: float = 0.7
    min_std: float = 0.1
    temperature: float = 1.0
    action_dim: int = 1
    seed: int = 0


class MPPIPlanner:
    """Continuous MPC using path-integral sampling with elite refinement."""

    def __init__(
        self,
        cfg: MPPIConfig,
        rollout_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        cost_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ):
        """
        rollout_fn(state: (S,), actions: (B, H, A)) -> (traj: (B, H, S), eps: (B, H))
        cost_fn(traj: (B, H, S), actions: (B, H, A), eps: (B, H)) -> (B,)
        """
        self.cfg = cfg
        self.rollout = rollout_fn
        self.cost = cost_fn
        self.rng = np.random.default_rng(cfg.seed)
        self.prev_mean = np.zeros((cfg.horizon, cfg.action_dim))

    def reset(self) -> None:
        self.prev_mean = np.zeros((self.cfg.horizon, self.cfg.action_dim))

    def plan(self, state: np.ndarray) -> np.ndarray:
        mean = self.prev_mean.copy()
        std = np.full_like(mean, self.cfg.init_std)
        last_costs = None
        for _ in range(self.cfg.n_iter):
            noise = self.rng.standard_normal(
                (self.cfg.n_samples, self.cfg.horizon, self.cfg.action_dim)
            )
            actions = np.clip(mean + noise * std, -1.0, 1.0)
            traj, eps = self.rollout(state, actions)
            costs = self.cost(traj, actions, eps)
            last_costs = costs
            # path-integral weighting
            min_c = np.min(costs)
            w = np.exp(-(costs - min_c) / max(self.cfg.temperature, 1e-6))
            w = w / (w.sum() + 1e-9)
            new_mean = (w[:, None, None] * actions).sum(axis=0)
            new_std = np.sqrt(
                (w[:, None, None] * (actions - new_mean[None]) ** 2).sum(axis=0) + 1e-6
            )
            mean = new_mean
            std = np.maximum(new_std, self.cfg.min_std)
        # shift mean for warm start
        self.prev_mean = np.concatenate([mean[1:], np.zeros((1, self.cfg.action_dim))], axis=0)
        return mean  # full action sequence; caller usually takes mean[0]

    def plan_action(self, state: np.ndarray) -> np.ndarray:
        return self.plan(state)[0]
