"""MPPI baseline (no skill graph, no monitor, no recovery, no CBF).

Uses the same L2 latent model as GATS-R for fairness; the only thing turned
off is the higher-level structure. This isolates the value-add of the skill
graph + monitor + recovery in the ablation."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from ..envs.balance_env import BalanceBotEnv
from ..planning.mppi import MPPIPlanner, MPPIConfig
from ..world_models.latent import EnsembleLatentModel


class MPPIAgent:
    name = "mppi"

    def __init__(self, env: BalanceBotEnv, latent_model: EnsembleLatentModel, seed: int = 0):
        self.env = env
        self.latent = latent_model
        self.seed = seed

        def rollout_fn(state: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return self.latent.rollout_np(state, actions)

        self._rollout_fn = rollout_fn
        self.planner = MPPIPlanner(
            MPPIConfig(horizon=12, n_samples=96, seed=seed),
            rollout_fn=rollout_fn,
            cost_fn=self._cost_fn,
        )

    def _cost_fn(self, traj: np.ndarray, actions: np.ndarray, eps: np.ndarray) -> np.ndarray:
        # traj: (B, H, S=4)
        goal_x = self.env.current_goal()
        end_dist = np.abs(traj[:, -1, 0] - goal_x)
        upright = -np.mean(np.cos(traj[:, :, 2]), axis=-1)
        action_cost = 0.01 * np.mean(actions ** 2, axis=(-1, -2))
        return end_dist + 0.5 * upright + action_cost

    def evaluate(self, episodes: int = 5, seed_offset: int = 0):
        stats_list = []
        for ep in range(episodes):
            self.env.reset(seed=self.seed + seed_offset + ep)
            self.planner.reset()
            done = False
            ep_return = 0.0
            steps = 0
            plan_ms = 0.0
            while not done:
                ps = self.env.physical_state
                t0 = time.perf_counter()
                a = self.planner.plan_action(ps)
                plan_ms += (time.perf_counter() - t0) * 1000.0
                _obs, r, done, info = self.env.step(a)
                ep_return += r
                steps += 1
            success = int(info.get("terminated", "") == "success")
            stats_list.append(
                dict(
                    ep_return=ep_return,
                    success=success,
                    steps=steps,
                    failures_detected=0,
                    recoveries_attempted=0,
                    recoveries_succeeded=0,
                    safety_violations=0,
                    time_to_recover=-1.0,
                    planning_ms=plan_ms / max(1, steps),
                )
            )
        return stats_list
