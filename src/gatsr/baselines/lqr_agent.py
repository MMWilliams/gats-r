"""LQR-only baseline: pure analytic stabilizer with a P-controller toward the
current goal_x. Cheap, fast, locally optimal — a strong baseline for short-
horizon balance, weak for long-horizon multi-goal sequences with disturbances.
"""

from __future__ import annotations

import numpy as np

from ..envs.balance_env import BalanceBotEnv
from ..world_models.analytic import AnalyticModel


class LQRAgent:
    name = "lqr"

    def __init__(self, env: BalanceBotEnv, seed: int = 0):
        self.env = env
        # The LQR baseline is exactly the layered model's L1 controller: a
        # goal-tracking LQR on the linearized cart-pole. This makes "LQR" and
        # "GATS-R's L1 layer" the same code, so the ablation isolates what the
        # graph/monitor/recovery/L2 layers add on top of L1.
        self.policy = AnalyticModel(
            mass_cart=env.cfg.mass_cart,
            mass_pole=env.cfg.mass_pole,
            pole_length=env.cfg.pole_length,
        )
        self.seed = seed

    def evaluate(self, episodes: int = 5, seed_offset: int = 0):
        stats_list = []
        for ep in range(episodes):
            self.env.reset(seed=self.seed + seed_offset + ep)
            done = False
            ep_return = 0.0
            steps = 0
            while not done:
                ps = self.env.physical_state
                goal_x = self.env.current_goal()
                a = self.policy.control(ps, goal_x=goal_x)
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
                    planning_ms=0.0,
                )
            )
        return stats_list
