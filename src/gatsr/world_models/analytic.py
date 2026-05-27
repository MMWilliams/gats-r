"""L1: analytic / physics-prior dynamics.

For BalanceBot this is the standard cart-pole linearization around the
upright equilibrium (theta = 0, theta_dot = 0). The model is exact at the
linearization point and degrades smoothly with |theta|; we therefore expose a
`validity()` score that the layered orchestrator uses to decide whether to
trust L1.

This is the continuous-control analog of the GATS "symbolic STRIPS match" —
cheap, structurally informative, and locally exact, but with a known domain
of validity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.balance_env import GRAVITY, DT


@dataclass
class AnalyticModel:
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_length: float = 0.5
    dt: float = DT
    validity_angle: float = 0.25  # rad — beyond this, L1's linearization is unreliable

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One-step prediction. State is the 4-D physical state."""
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        # squeeze trailing action dim if present so shapes broadcast cleanly
        if action.shape and action.shape[-1] == 1:
            action = action[..., 0]
        # broadcast action against state's leading dims
        if action.ndim < state.ndim - 1:
            action = np.broadcast_to(action, state.shape[:-1])
        a = np.clip(action, -1.0, 1.0) * 12.0  # match env action_scale

        x, x_dot, th, th_dot = state[..., 0], state[..., 1], state[..., 2], state[..., 3]

        mc, mp, l = self.mass_cart, self.mass_pole, self.pole_length
        total = mc + mp

        # linearization: cos(theta) ~ 1, sin(theta) ~ theta, theta_dot^2 sin ~ 0
        x_acc = (a - mp * GRAVITY * th) / total
        th_acc = (GRAVITY * th - a / total) / (l * (4 / 3 - mp / total))

        new_x_dot = x_dot + x_acc * self.dt
        new_x = x + new_x_dot * self.dt
        new_th_dot = th_dot + th_acc * self.dt
        new_th = th + new_th_dot * self.dt
        return np.stack([new_x, new_x_dot, new_th, new_th_dot], axis=-1)

    def rollout(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out for H steps. actions: (H, A) or (B, H, A)."""
        if actions.ndim == 2:
            s = state.copy()
            traj = []
            for h in range(actions.shape[0]):
                s = self.predict(s, actions[h])
                traj.append(s)
            return np.stack(traj, axis=0)
        elif actions.ndim == 3:
            B, H, _ = actions.shape
            s = np.broadcast_to(state, (B, state.shape[-1])).copy()
            traj = np.zeros((B, H, state.shape[-1]))
            for h in range(H):
                s = self.predict(s, actions[:, h])
                traj[:, h] = s
            return traj
        else:
            raise ValueError(f"actions ndim must be 2 or 3, got {actions.ndim}")

    def validity(self, state: np.ndarray) -> float:
        """[0,1] score; 1 = in domain of linearization, 0 = far OOD.

        Falls off smoothly with |theta| / |theta_dot|."""
        th = state[..., 2]
        th_dot = state[..., 3]
        v_th = np.exp(-((th / self.validity_angle) ** 2))
        v_thd = np.exp(-((th_dot / 4.0) ** 2))
        v = (v_th * v_thd)
        return float(v if np.ndim(v) == 0 else np.mean(v))
