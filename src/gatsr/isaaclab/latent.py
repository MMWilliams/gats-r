"""GPU-batched ensemble latent dynamics model for the G1.

Functionally identical to ``gatsr.world_models.latent.EnsembleLatentModel`` but
keeps everything as torch tensors on the configured CUDA device, supports
distributed-data-parallel-style multi-GPU rollouts (one head per GPU), and
operates on the G1's structured proprioceptive state vector (~92-D) rather
than the toy 4-D cart-pole state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

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
class G1LatentConfig:
    state_dim: int = 92
    action_dim: int = 37
    latent_dim: int = 64
    hidden: int = 256
    n_ensemble: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 5
    device: str = "cuda:0"
    multi_gpu_rollouts: bool = False  # disabled by default; gpu:1 may be on a slow PCIe lane


class _DynamicsHead(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp([latent_dim + action_dim, hidden, hidden, latent_dim])

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))


class G1EnsembleLatentModel(nn.Module):
    def __init__(self, cfg: G1LatentConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else G1LatentConfig()
        c = self.cfg
        self.encoder = _mlp([c.state_dim, c.hidden, c.latent_dim])
        self.decoder = _mlp([c.latent_dim, c.hidden, c.state_dim])
        self.heads = nn.ModuleList(
            [_DynamicsHead(c.latent_dim, c.action_dim, c.hidden) for _ in range(c.n_ensemble)]
        )
        self.to(c.device)

        # optional multi-GPU rollouts: replicate the ensemble heads on the
        # second visible CUDA device for true parallelism.
        self._mirror = None
        if c.multi_gpu_rollouts and torch.cuda.device_count() > 1:
            self._mirror_device = torch.device("cuda:1")
            # we copy the whole module (cheap; ~MB of params) to the second
            # device; the encode/step/decode pass batches large enough to
            # justify the cross-device sync
            try:
                self._mirror = self._replicate_to(self._mirror_device)
            except Exception:
                self._mirror = None

    def _replicate_to(self, device: torch.device) -> "G1EnsembleLatentModel":
        c = self.cfg
        clone = G1EnsembleLatentModel(
            G1LatentConfig(**{**c.__dict__, "device": str(device), "multi_gpu_rollouts": False})
        )
        clone.load_state_dict(self.state_dict())
        return clone

    # ----- core ops -------------------------------------------------------

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = torch.stack([h(z, a) for h in self.heads], dim=0)
        return outs.mean(0), outs.std(0)

    @torch.inference_mode()
    def rollout(
        self, s: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rollout H steps with batched actions.

        actions: (B, H, A) or (H, A).
        returns (traj: (B, H, S), eps: (B, H)).

        If a mirror exists on a second GPU, the batch is split across both
        devices for a near-2x throughput on rollout-heavy planners (MPPI)."""
        squeeze = False
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
            squeeze = True
        B, H, _ = actions.shape
        device = next(self.parameters()).device
        s = s.to(device, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0).expand(B, -1).contiguous()
        elif s.shape[0] != B:
            s = s.expand(B, -1).contiguous()
        actions = actions.to(device, dtype=torch.float32)

        if self._mirror is None or B < 64:
            traj, eps = self._rollout_single_device(s, actions)
        else:
            half = B // 2
            t1, e1 = self._rollout_single_device(s[:half], actions[:half])
            with torch.cuda.device(self._mirror_device):
                s2 = s[half:].to(self._mirror_device, non_blocking=True)
                a2 = actions[half:].to(self._mirror_device, non_blocking=True)
                t2, e2 = self._mirror._rollout_single_device(s2, a2)
                t2 = t2.to(device, non_blocking=True)
                e2 = e2.to(device, non_blocking=True)
            traj = torch.cat([t1, t2], dim=0)
            eps = torch.cat([e1, e2], dim=0)
        if squeeze:
            return traj.squeeze(0), eps.squeeze(0)
        return traj, eps

    def _rollout_single_device(
        self, s: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, _ = actions.shape
        z = self.encode(s)
        traj = torch.zeros((B, H, self.cfg.state_dim), device=s.device)
        eps = torch.zeros((B, H), device=s.device)
        for h in range(H):
            z, z_std = self.step_latent(z, actions[:, h])
            traj[:, h] = self.decode(z)
            eps[:, h] = z_std.mean(dim=-1)
        return traj, eps

    @torch.inference_mode()
    def predict(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step prediction. s: (B, S), a: (B, A). returns (s', epi: (B,))."""
        device = next(self.parameters()).device
        s = s.to(device, dtype=torch.float32)
        a = a.to(device, dtype=torch.float32)
        z = self.encode(s)
        z_next, z_std = self.step_latent(z, a)
        return self.decode(z_next), z_std.mean(dim=-1)

    # ----- training ------------------------------------------------------

    def fit(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        verbose: bool = True,
    ) -> dict:
        device = next(self.parameters()).device
        s = states.to(device, dtype=torch.float32)
        a = actions.to(device, dtype=torch.float32)
        sp = next_states.to(device, dtype=torch.float32)
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
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
                with torch.no_grad():
                    z_tgt = self.encode(sp_b)
                loss = torch.zeros((), device=device)
                for head in self.heads:
                    mask = torch.rand(b.numel(), device=device) < 0.8
                    if mask.sum() < 2:
                        continue
                    z_pred = head(z[mask], a_b[mask])
                    loss = loss + nn.functional.mse_loss(z_pred, z_tgt[mask])
                s_rec = self.decode(z)
                loss = loss + 0.5 * nn.functional.mse_loss(s_rec, s_b)
                z_pred_avg, _ = self.step_latent(z.detach(), a_b)
                s_next_rec = self.decode(z_pred_avg)
                loss = loss + 0.5 * nn.functional.mse_loss(s_next_rec, sp_b)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                opt.step()
                ep_losses.append(float(loss.detach().item()))
            losses.append(float(sum(ep_losses) / max(1, len(ep_losses))))
            if verbose:
                # protect against BrokenPipeError on Windows when stdout is redirected to nul
                try:
                    print(f"[L2] epoch {ep + 1}/{self.cfg.epochs} loss={losses[-1]:.4f}", flush=True)
                except Exception:
                    pass
        # refresh mirror after training
        if self._mirror is not None:
            self._mirror.load_state_dict(self.state_dict())
        return {"final_loss": losses[-1] if losses else float("nan"), "loss_curve": losses}
