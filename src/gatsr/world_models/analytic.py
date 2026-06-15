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

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are

from ..envs.balance_env import GRAVITY, DT


@dataclass
class AnalyticModel:
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_length: float = 0.5
    dt: float = DT
    validity_angle: float = 0.5  # rad — beyond ~0.5 rad the cart-pole linearization degrades
    # Goal-tracking LQR weights. Cart-position weight is deliberately higher
    # than the recovery stabilizer's so the controller actually drives the cart
    # to each goal setpoint (not just balance at the origin).
    q_diag: tuple[float, ...] = (20.0, 1.0, 30.0, 5.0)
    r_weight: float = 0.1
    action_scale: float = 12.0
    _K: np.ndarray | None = field(default=None, repr=False, compare=False)

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

    # ----- analytic control (L1 goal-tracking LQR) -----------------------

    def _gain(self) -> np.ndarray:
        """Time-invariant LQR gain on the linearized cart-pole."""
        if self._K is not None:
            return self._K
        l = self.pole_length
        mp, mc = self.mass_pole, self.mass_cart
        total = mc + mp
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -mp * GRAVITY / total, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, total * GRAVITY / (l * mc), 0.0],
            ]
        )
        B = np.array([[0.0], [1.0 / total], [0.0], [-1.0 / (l * mc)]])
        Q = np.diag(self.q_diag)
        R = np.array([[self.r_weight]])
        P = solve_continuous_are(A, B, Q, R)
        self._K = np.linalg.inv(R) @ B.T @ P  # (1, 4)
        return self._K

    def control(self, state: np.ndarray, goal_x: float = 0.0) -> np.ndarray:
        """Goal-tracking LQR action in [-1, 1].

        Regulates the state to the setpoint [goal_x, 0, 0, 0], i.e. drive the
        cart to the goal while keeping the pole upright. This is L1's *control*
        counterpart to its `predict`/`validity` interface, used by the agent
        whenever L1 is in its domain of validity."""
        K = self._gain()
        s = np.asarray(state[:4], dtype=np.float64).reshape(4, 1)
        s_ref = np.array([[float(goal_x)], [0.0], [0.0], [0.0]])
        u = float(-(K @ (s - s_ref)).reshape(-1)[0])
        a = float(np.clip(u / self.action_scale, -1.0, 1.0))
        return np.array([a])

    def validity(self, state: np.ndarray) -> float:
        """[0,1] score; 1 = in domain of linearization, 0 = far OOD.

        Falls off smoothly with |theta| / |theta_dot|."""
        th = state[..., 2]
        th_dot = state[..., 3]
        v_th = np.exp(-((th / self.validity_angle) ** 2))
        v_thd = np.exp(-((th_dot / 4.0) ** 2))
        v = (v_th * v_thd)
        return float(v if np.ndim(v) == 0 else np.mean(v))
