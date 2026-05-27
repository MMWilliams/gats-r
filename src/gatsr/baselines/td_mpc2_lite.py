"""TD-MPC2-lite: the same ensemble latent model + MPPI, but with a learned
value head bootstrapping the planning horizon (rather than just the path-
integral cost). This is the closest spiritual analog to TD-MPC2 (Hansen et
al., 2024) we can fit on CPU with no GPU training.

Differences vs. the GATS-R agent:
    * no skill graph; no high-level planning
    * no Sentinel-style monitor
    * no graph-indexed recovery
    * no CBF filter

This isolates the value of the *graph + monitor + recovery* layers.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
from torch import nn

from ..envs.balance_env import BalanceBotEnv
from ..planning.mppi import MPPIPlanner, MPPIConfig
from ..world_models.latent import EnsembleLatentModel


class _ValueHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class TDMPC2LiteAgent:
    name = "td_mpc2_lite"

    def __init__(self, env: BalanceBotEnv, latent_model: EnsembleLatentModel, seed: int = 0):
        self.env = env
        self.latent = latent_model
        self.value_head = _ValueHead(latent_model.cfg.latent_dim).to(latent_model.cfg.device)
        self.seed = seed
        self.planner = MPPIPlanner(
            MPPIConfig(horizon=12, n_samples=96, seed=seed),
            rollout_fn=lambda s, a: self.latent.rollout_np(s, a),
            cost_fn=self._cost_fn,
        )

    def fit_value(self, states: np.ndarray, returns: np.ndarray, epochs: int = 8) -> dict:
        """Quick TD-style value regression on Monte-Carlo returns."""
        device = self.latent.cfg.device
        s_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(self.value_head.parameters(), lr=3e-3, weight_decay=1e-5)
        losses = []
        for _ in range(epochs):
            idx = torch.randperm(s_t.shape[0], device=device)
            for i in range(0, s_t.shape[0], 256):
                b = idx[i : i + 256]
                with torch.no_grad():
                    z = self.latent.encode(s_t[b])
                pred = self.value_head(z)
                loss = nn.functional.mse_loss(pred, ret[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
        return {"final_loss": losses[-1] if losses else float("nan")}

    def _cost_fn(self, traj: np.ndarray, actions: np.ndarray, eps: np.ndarray) -> np.ndarray:
        goal_x = self.env.current_goal()
        end_dist = np.abs(traj[:, -1, 0] - goal_x)
        upright = -np.mean(np.cos(traj[:, :, 2]), axis=-1)
        action_cost = 0.01 * np.mean(actions ** 2, axis=(-1, -2))
        # value bootstrap at horizon end
        with torch.no_grad():
            s_end = torch.as_tensor(traj[:, -1], dtype=torch.float32, device=self.latent.cfg.device)
            z_end = self.latent.encode(s_end)
            v_end = self.value_head(z_end).cpu().numpy()
        # we minimise cost, so subtract value bootstrap
        return end_dist + 0.5 * upright + action_cost - 0.05 * v_end

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
