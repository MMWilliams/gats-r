"""Dreamer-lite: GRU-based recurrent state-space model with a small actor head
and reconstruction loss. Stripped-down compared to DreamerV3 — no symlog,
twohot, or KL-balanced posterior — but the architecture maps directly onto
the RSSM diagram and serves as a representative reconstruction-based baseline
distinct from TD-MPC2's value-equivalent latent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..envs.balance_env import BalanceBotEnv


@dataclass
class DreamerLiteConfig:
    state_dim: int = 4
    action_dim: int = 1
    deter_dim: int = 32
    stoch_dim: int = 16
    hidden: int = 64
    lr: float = 3e-3
    epochs: int = 8
    batch_size: int = 64
    seq_len: int = 16
    seed: int = 0


class DreamerLiteAgent(nn.Module):
    name = "dreamer_lite"

    def __init__(self, env: BalanceBotEnv, cfg: DreamerLiteConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else DreamerLiteConfig()
        self.env = env
        d, s, h, a = (
            self.cfg.deter_dim,
            self.cfg.stoch_dim,
            self.cfg.hidden,
            self.cfg.action_dim,
        )
        self.gru = nn.GRUCell(s + a, d)
        self.posterior = nn.Sequential(
            nn.Linear(d + self.cfg.state_dim, h), nn.SiLU(), nn.Linear(h, s)
        )
        self.prior = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, s))
        self.decoder = nn.Sequential(
            nn.Linear(d + s, h), nn.SiLU(), nn.Linear(h, self.cfg.state_dim)
        )
        self.actor = nn.Sequential(
            nn.Linear(d + s, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, a), nn.Tanh()
        )
        self.value = nn.Sequential(
            nn.Linear(d + s, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1)
        )

    # --- training -------------------------------------------------------

    def fit(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> dict:
        """Simple one-step prediction + reconstruction; no imagination training
        (which would require many more iterations than is fair to spend on a
        baseline in a CPU benchmark)."""
        device = "cpu"
        N = states.shape[0]
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        s = torch.as_tensor(states, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32)
        sp = torch.as_tensor(next_states, dtype=torch.float32)
        losses = []
        for ep in range(self.cfg.epochs):
            idx = torch.randperm(N)
            ep_losses = []
            for i in range(0, N, self.cfg.batch_size):
                b = idx[i : i + self.cfg.batch_size]
                deter = torch.zeros(b.numel(), self.cfg.deter_dim)
                stoch = torch.zeros(b.numel(), self.cfg.stoch_dim)
                inp = torch.cat([stoch, a[b]], dim=-1)
                deter = self.gru(inp, deter)
                post = self.posterior(torch.cat([deter, sp[b]], dim=-1))
                pri = self.prior(deter)
                recon = self.decoder(torch.cat([deter, post], dim=-1))
                loss = (
                    nn.functional.mse_loss(recon, sp[b])
                    + 0.1 * nn.functional.mse_loss(pri, post.detach())
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_losses.append(float(loss.detach()))
            losses.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
        return {"final_loss": losses[-1] if losses else float("nan"), "loss_curve": losses}

    # --- inference ------------------------------------------------------

    def _hidden_state(self, s_t: torch.Tensor) -> torch.Tensor:
        deter = torch.zeros(s_t.shape[0], self.cfg.deter_dim)
        stoch = self.posterior(torch.cat([deter, s_t], dim=-1))
        return torch.cat([deter, stoch], dim=-1)

    @torch.no_grad()
    def select_action(self, physical_state: np.ndarray) -> np.ndarray:
        # bias toward goal by feeding offset state to the actor
        goal_x = self.env.current_goal()
        ps = physical_state.copy()
        ps[0] = ps[0] - goal_x
        s_t = torch.as_tensor(ps[:4], dtype=torch.float32).unsqueeze(0)
        h = self._hidden_state(s_t)
        a = self.actor(h).detach().cpu().numpy().flatten()
        return np.clip(a, -1.0, 1.0)

    def evaluate(self, episodes: int = 5, seed_offset: int = 0):
        stats_list = []
        for ep in range(episodes):
            self.env.reset(seed=self.cfg.seed + seed_offset + ep)
            done = False
            ep_return = 0.0
            steps = 0
            plan_ms = 0.0
            while not done:
                ps = self.env.physical_state
                t0 = time.perf_counter()
                a = self.select_action(ps)
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
