"""Batched MPPI on top of the G1 latent world model.

Each of N parallel envs has its own MPPI optimizer that shares the same latent
dynamics model. Within an env, K sample trajectories are rolled out in the
latent model, scored by a user-supplied cost function on the *predicted*
trajectory, and the elite-weighted mean is taken as the action sequence.

Compared to the toy MPPI (numpy, single env), this version:
    * stays entirely on the configured CUDA device,
    * batches the K samples across all N envs into a single (N*K, H, A) tensor,
    * dispatches the rollout via the latent model's multi-GPU split so both
      RTX 5090s contribute when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .latent import G1EnsembleLatentModel


@dataclass
class G1MPPIConfig:
    horizon: int = 16
    n_samples: int = 96
    n_iter: int = 2
    init_std: float = 0.4
    min_std: float = 0.05
    temperature: float = 1.0
    device: str = "cuda:0"


class G1BatchedMPPI:
    def __init__(
        self,
        cfg: G1MPPIConfig,
        model: G1EnsembleLatentModel,
        num_envs: int,
        action_dim: int,
    ):
        self.cfg = cfg
        self.model = model
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(cfg.device)
        self.prev_mean = torch.zeros(num_envs, cfg.horizon, action_dim, device=self.device)

    def reset(self) -> None:
        self.prev_mean.zero_()

    @torch.inference_mode()
    def plan(
        self,
        physical_state: torch.Tensor,
        cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """physical_state: (N, S). cost_fn(traj: (N, K, H, S), actions: (N, K, H, A),
        eps: (N, K, H)) -> (N, K) costs."""
        N, K, H, A = self.num_envs, self.cfg.n_samples, self.cfg.horizon, self.action_dim
        mean = self.prev_mean.clone()
        std = torch.full_like(mean, self.cfg.init_std)
        physical_state = physical_state.to(self.device)
        # broadcast initial state across K samples
        s_init = physical_state.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)
        for _ in range(self.cfg.n_iter):
            noise = torch.randn(N, K, H, A, device=self.device)
            actions = (mean.unsqueeze(1) + noise * std.unsqueeze(1)).clamp(-1.0, 1.0)
            actions_flat = actions.reshape(N * K, H, A)
            traj_flat, eps_flat = self.model.rollout(s_init, actions_flat)
            traj = traj_flat.reshape(N, K, H, -1)
            eps = eps_flat.reshape(N, K, H)
            costs = cost_fn(traj, actions, eps)  # (N, K)
            min_c = costs.min(dim=-1, keepdim=True).values
            w = torch.exp(-(costs - min_c) / max(self.cfg.temperature, 1e-6))
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
            new_mean = (w.unsqueeze(-1).unsqueeze(-1) * actions).sum(dim=1)
            diff = actions - new_mean.unsqueeze(1)
            new_std = ((w.unsqueeze(-1).unsqueeze(-1) * (diff ** 2)).sum(dim=1) + 1e-6).sqrt()
            mean = new_mean
            std = new_std.clamp_min(self.cfg.min_std)
        # warm-start
        self.prev_mean = torch.cat([mean[:, 1:], torch.zeros_like(mean[:, :1])], dim=1)
        return mean  # (N, H, A); caller takes mean[:, 0]
