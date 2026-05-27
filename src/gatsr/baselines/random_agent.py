from __future__ import annotations

import numpy as np

from ..envs.balance_env import BalanceBotEnv


class RandomAgent:
    name = "random"

    def __init__(self, env: BalanceBotEnv, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def evaluate(self, episodes: int = 5, seed_offset: int = 0):
        stats_list = []
        for ep in range(episodes):
            self.env.reset(seed=seed_offset + ep)
            done = False
            ep_return = 0.0
            steps = 0
            while not done:
                a = self.rng.uniform(-1.0, 1.0, size=(1,))
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
