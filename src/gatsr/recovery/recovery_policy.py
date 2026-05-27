"""Recovery policy: an LQR stabilizer around the upright equilibrium.

In the full thesis this slot is occupied by FRASA / FIRM / "Learning to Get
Up Across Morphologies". For BalanceBot the natural high-authority controller
is the LQR gain on the linearized model — analytic, fast, and provably
locally stabilizing.

The point is *not* the specific algorithm but the interface: a black-box
"recover from this state" controller that the graph-indexed dispatcher can
swap out for FRASA/FIRM-style policies in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from ..envs.balance_env import GRAVITY


@dataclass
class RecoveryConfig:
    pole_length: float = 0.5
    mass_pole: float = 0.1
    mass_cart: float = 1.0
    q_diag: tuple[float, ...] = (1.0, 1.0, 30.0, 5.0)
    r: float = 0.1
    action_scale: float = 12.0
    max_steps: int = 60


class LQRRecoveryPolicy:
    """Time-invariant LQR gain on the linearized 4-D state."""

    def __init__(self, cfg: RecoveryConfig | None = None):
        self.cfg = cfg if cfg is not None else RecoveryConfig()
        self.K = self._compute_gain()
        self.steps_in_recovery = 0

    def _compute_gain(self) -> np.ndarray:
        L = self.cfg.pole_length
        mp, mc = self.cfg.mass_pole, self.cfg.mass_cart
        total = mc + mp
        # linearized cart-pole around theta = 0
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -mp * GRAVITY / total, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, total * GRAVITY / (L * mc), 0.0],
            ]
        )
        B = np.array(
            [
                [0.0],
                [1.0 / total],
                [0.0],
                [-1.0 / (L * mc)],
            ]
        )
        Q = np.diag(self.cfg.q_diag)
        R = np.array([[self.cfg.r]])
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P  # (1, 4)
        return K

    def reset(self) -> None:
        self.steps_in_recovery = 0

    def __call__(self, physical_state: np.ndarray) -> np.ndarray:
        s = np.asarray(physical_state[:4], dtype=np.float64).reshape(4, 1)
        u = -(self.K @ s).flatten()
        a = float(np.clip(u / self.cfg.action_scale, -1.0, 1.0))
        self.steps_in_recovery += 1
        return np.array([a])

    def recovered(self, physical_state: np.ndarray) -> bool:
        x, x_dot, th, th_dot = physical_state[:4]
        return bool(abs(th) < 0.1 and abs(th_dot) < 0.3 and abs(x_dot) < 0.5)
