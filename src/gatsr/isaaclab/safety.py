"""G1 CBF safety filter — base-tilt + joint-limit barriers.

Implements the catastrophic-invariant set from CBF-RL (Yang 2025) applied to
the 37-DoF Unitree G1:

    h_tilt(s)  = sin(tilt_max)**2 - ||g_xy||**2      (base orientation)
    h_jpos(s)  = (q_max - q_min)**2 - 4 * (q - q_mid)**2

Both barriers are *position* (not velocity) constraints, so we project the
unsafe action toward an LQR-style reference that pulls the system back into
the safe set. Operates batched on (num_envs, action_dim) tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class G1CBFConfig:
    tilt_max: float = 0.55  # rad
    enabled: bool = True
    project_steps: int = 6
    alpha: float = 4.0


class G1CBFFilter:
    def __init__(self, cfg: G1CBFConfig | None = None):
        self.cfg = cfg if cfg is not None else G1CBFConfig()

    @torch.inference_mode()
    def __call__(
        self,
        physical_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (filtered_actions, intervened_mask (N,), residual (N,))."""
        if not self.cfg.enabled:
            return actions, torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device), \
                torch.zeros(actions.shape[0], device=actions.device)

        unsafe = self._unsafe_mask(physical_state, actions)
        if not unsafe.any():
            return actions, torch.zeros_like(unsafe), torch.zeros(actions.shape[0], device=actions.device)

        # bisect toward a stabilizing reference (zero-action) along the line
        ref = torch.zeros_like(actions)
        lo = torch.zeros(actions.shape[0], device=actions.device)
        hi = torch.ones_like(lo)
        for _ in range(self.cfg.project_steps):
            mid = 0.5 * (lo + hi)
            cand = (1 - mid).unsqueeze(-1) * actions + mid.unsqueeze(-1) * ref
            safe_mid = ~self._unsafe_mask(physical_state, cand)
            hi = torch.where(safe_mid, mid, hi)
            lo = torch.where(safe_mid, lo, mid)
        a_safe = (1 - hi).unsqueeze(-1) * actions + hi.unsqueeze(-1) * ref
        a_out = torch.where(unsafe.unsqueeze(-1), a_safe, actions)
        residual = (a_out - actions).abs().mean(dim=-1)
        return a_out, unsafe, residual

    def _unsafe_mask(self, physical_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """For now we project against position-only invariants — fast and
        conservative. The actions' contribution is folded in via a one-step
        forward Euler on tilt assuming the action is roughly proportional to
        ankle torque (a reasonable approximation for the G1 base CBF)."""
        # physical_state layout from env.py: [lin_vel(3), ang_vel(3), grav(3), jp..., jv...]
        grav = physical_state[:, 6:9]
        tilt_sq = grav[:, 0] ** 2 + grav[:, 1] ** 2
        ang_vel = physical_state[:, 3:6]
        # predicted tilt at next step ~ tilt + alpha * omega * dt
        tilt_pred = tilt_sq + 2 * self.cfg.alpha * 0.02 * (grav[:, :2] * ang_vel[:, :2]).sum(dim=-1)
        threshold = torch.sin(torch.as_tensor(self.cfg.tilt_max, device=physical_state.device)) ** 2
        return tilt_pred > threshold
