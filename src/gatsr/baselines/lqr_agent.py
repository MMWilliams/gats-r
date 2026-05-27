"""LQR-only baseline: pure analytic stabilizer with a P-controller toward the
current goal_x. Cheap, fast, locally optimal — a strong baseline for short-
horizon balance, weak for long-horizon multi-goal sequences with disturbances.
"""

from __future__ import annotations

import numpy as np

from ..envs.balance_env import BalanceBotEnv
from ..recovery.recovery_policy import LQRRecoveryPolicy, RecoveryConfig


class LQRAgent:
    name = "lqr"

    def __init__(self, env: BalanceBotEnv, seed: int = 0):
        self.env = env
        self.policy = LQRRecoveryPolicy(
            RecoveryConfig(
                pole_length=env.cfg.pole_length,
                mass_pole=env.cfg.mass_pole,
                mass_cart=env.cfg.mass_cart,
            )
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
                # bias toward current goal by shifting reference x
                goal_x = self.env.current_goal()
                ps_shifted = ps.copy()
                ps_shifted[0] = ps[0] - goal_x
                a = self.policy(ps_shifted)
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
