"""L2: learned latent dynamics with epistemic uncertainty via ensemble
disagreement.

Faithful but minimal implementation of the TD-MPC2 idea adapted to the toy
domain: an encoder maps the physical state into a small latent, an MLP
predicts the residual delta-latent given (z, a), and a decoder maps z back to
the next physical state. An *ensemble* of dynamics heads provides the
epistemic-uncertainty signal that the runtime monitor consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


def _mlp(sizes: Iterable[int], act: type = nn.SiLU, last_act: bool = False) -> nn.Sequential:
    sizes = list(sizes)
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


@dataclass
class LatentModelConfig:
    state_dim: int = 4
    action_dim: int = 1
    latent_dim: int = 16
    hidden: int = 64
    n_ensemble: int = 4
    lr: float = 3e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 20
    device: str = "cpu"


class _DynamicsHead(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp([latent_dim + action_dim, hidden, hidden, latent_dim])

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))


class EnsembleLatentModel(nn.Module):
    """Encoder + ensemble dynamics + decoder. Returns mean rollouts and per-step
    epistemic-uncertainty (ensemble std)."""

    def __init__(self, cfg: LatentModelConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else LatentModelConfig()
        self.encoder = _mlp([self.cfg.state_dim, self.cfg.hidden, self.cfg.latent_dim])
        self.decoder = _mlp([self.cfg.latent_dim, self.cfg.hidden, self.cfg.state_dim])
        self.heads = nn.ModuleList(
            [
                _DynamicsHead(self.cfg.latent_dim, self.cfg.action_dim, self.cfg.hidden)
                for _ in range(self.cfg.n_ensemble)
            ]
        )
        self.to(self.cfg.device)

    # ----- core ops --------------------------------------------------------

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) over the ensemble for the next latent."""
        outs = torch.stack([h(z, a) for h in self.heads], dim=0)
        return outs.mean(0), outs.std(0)

    # ----- numpy convenience for planners ----------------------------------

    @torch.no_grad()
    def predict_np(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float]:
        """One-step prediction returning (next_state, epistemic_uncertainty)."""
        device = self.cfg.device
        st = torch.as_tensor(s, dtype=torch.float32, device=device).view(-1, self.cfg.state_dim)
        at = torch.as_tensor(a, dtype=torch.float32, device=device).view(-1, self.cfg.action_dim)
        z = self.encode(st)
        z_next_mean, z_next_std = self.step_latent(z, at)
        s_next = self.decode(z_next_mean).cpu().numpy().reshape(s.shape)
        eps = float(z_next_std.mean().cpu().item())
        return s_next, eps

    @torch.no_grad()
    def rollout_np(
        self, s: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Roll out H steps. actions shape (B, H, A) or (H, A).

        Returns (predicted_trajectory, per_step_epistemic_uncertainty)."""
        device = self.cfg.device
        if actions.ndim == 2:
            actions = actions[None]  # (1, H, A)
            squeeze = True
        else:
            squeeze = False
        B, H, _ = actions.shape
        s_t = torch.as_tensor(s, dtype=torch.float32, device=device).view(-1, self.cfg.state_dim)
        if s_t.shape[0] == 1 and B > 1:
            s_t = s_t.expand(B, -1).contiguous()
        z = self.encode(s_t)
        a_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
        traj_states = np.zeros((B, H, self.cfg.state_dim), dtype=np.float64)
        eps_per_step = np.zeros((B, H), dtype=np.float64)
        for h in range(H):
            z, z_std = self.step_latent(z, a_all[:, h])
            s_next = self.decode(z).cpu().numpy()
            traj_states[:, h] = s_next
            eps_per_step[:, h] = z_std.mean(dim=-1).cpu().numpy()
        if squeeze:
            return traj_states[0], eps_per_step[0]
        return traj_states, eps_per_step

    # ----- training -------------------------------------------------------

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        verbose: bool = False,
    ) -> dict:
        """Fit on a flat buffer of (s, a, s')."""
        device = self.cfg.device
        s = torch.as_tensor(states, dtype=torch.float32, device=device)
        a = torch.as_tensor(actions, dtype=torch.float32, device=device)
        sp = torch.as_tensor(next_states, dtype=torch.float32, device=device)

        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        N = s.shape[0]
        losses = []
        for ep in range(self.cfg.epochs):
            idx = torch.randperm(N, device=device)
            ep_losses = []
            for i in range(0, N, self.cfg.batch_size):
                b = idx[i : i + self.cfg.batch_size]
                if b.numel() < 8:
                    continue
                s_b, a_b, sp_b = s[b], a[b], sp[b]
                z = self.encode(s_b)
                z_target = self.encode(sp_b).detach()
                # ensemble bootstrap: each head sees a random 80% subset
                loss = 0.0
                for h_idx, head in enumerate(self.heads):
                    mask = torch.rand(b.numel(), device=device) < 0.8
                    if mask.sum() < 2:
                        continue
                    z_pred = head(z[mask], a_b[mask])
                    loss_dyn = nn.functional.mse_loss(z_pred, z_target[mask])
                    loss = loss + loss_dyn
                # add reconstruction so the encoder/decoder are aligned
                s_rec = self.decode(z)
                loss = loss + 0.5 * nn.functional.mse_loss(s_rec, s_b)
                # one-step decoder consistency on predicted next state
                z_pred_avg, _ = self.step_latent(z.detach(), a_b)
                s_next_rec = self.decode(z_pred_avg)
                loss = loss + 0.5 * nn.functional.mse_loss(s_next_rec, sp_b)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                opt.step()
                ep_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
            if verbose:
                print(f"[L2] epoch {ep + 1}/{self.cfg.epochs} loss={losses[-1]:.4f}")
        return {"final_loss": losses[-1] if losses else float("nan"), "loss_curve": losses}
