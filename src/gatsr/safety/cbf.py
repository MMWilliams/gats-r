"""Control Barrier Function (CBF) safety filter — CBF-RL pattern.

We use the CBF-RL recipe of Yang et al. (arXiv 2510.14959, Oct 2025): apply
the filter *during training* so the policy internalizes the safety constraint,
then turn the filter off (or keep it on as a thin runtime check) at deployment.

For BalanceBot the catastrophic invariants are:
    1. |theta| < theta_max  (don't crash the pole — the get-up controller
       cannot recover beyond `crash_angle`)
    2. |x| < cart_limit     (don't fall off the rail)

The barrier function is
    h_theta(s) = (theta_max ** 2) - theta ** 2
    h_cart(s)  = (cart_limit ** 2) - x ** 2

and we enforce  dh/dt + alpha * h >= 0  (exponential CBF; Ames et al.) on the
linearized dynamics. The filter projects an unsafe action toward the nearest
feasible one using a 1-D scalar search (sufficient for a 1-D action space).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.balance_env import GRAVITY


@dataclass
class CBFConfig:
    theta_max: float = 0.6
    cart_max: float = 2.2
    alpha_theta: float = 6.0
    alpha_cart: float = 3.0
    enabled: bool = True
    project_steps: int = 8


class CBFSafetyFilter:
    """Project an unsafe action toward the safe set. Returns (a_safe,
    intervened: bool, residual_norm: float)."""

    def __init__(self, cfg: CBFConfig | None = None):
        self.cfg = cfg if cfg is not None else CBFConfig()

    def __call__(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, bool, float]:
        if not self.cfg.enabled:
            return action, False, 0.0
        a = float(np.clip(action, -1.0, 1.0))
        # Pre-evaluate the constraints at the current action
        if self._is_safe(state, np.array([a])):
            return np.array([a]), False, 0.0
        # binary search along the line between unsafe a and a "stabilizing" reference
        ref = self._stabilizing_reference(state)
        lo, hi = 0.0, 1.0
        for _ in range(self.cfg.project_steps):
            mid = 0.5 * (lo + hi)
            cand = (1 - mid) * a + mid * ref
            if self._is_safe(state, np.array([cand])):
                hi = mid
            else:
                lo = mid
        a_safe = (1 - hi) * a + hi * ref
        return np.array([a_safe]), True, float(abs(a_safe - a))

    def _stabilizing_reference(self, state: np.ndarray) -> float:
        """Simple LQR-like reference: push opposite to the tilt and velocity."""
        x, x_dot, th, th_dot = state[:4]
        ref = -2.5 * th - 0.5 * th_dot - 0.4 * x - 0.4 * x_dot
        return float(np.clip(ref / 12.0, -1.0, 1.0))

    def _is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        x, x_dot, th, th_dot = state[:4]
        a = float(action[0]) * 12.0  # match action_scale

        # Linearized accelerations
        th_acc = (GRAVITY * th - a * 1.0) / 0.5
        x_acc = a / 1.1

        # h_theta = theta_max^2 - theta^2  -> dh/dt = -2 * theta * theta_dot
        # second time derivative used implicitly via th_acc
        h_theta = self.cfg.theta_max ** 2 - th ** 2
        # exponential CBF discrete check
        dh_theta = -2.0 * th * th_dot
        ddh_theta = -2.0 * (th_dot ** 2 + th * th_acc)
        cbf_theta = ddh_theta + self.cfg.alpha_theta * dh_theta + self.cfg.alpha_theta * h_theta

        h_cart = self.cfg.cart_max ** 2 - x ** 2
        dh_cart = -2.0 * x * x_dot
        ddh_cart = -2.0 * (x_dot ** 2 + x * x_acc)
        cbf_cart = ddh_cart + self.cfg.alpha_cart * dh_cart + self.cfg.alpha_cart * h_cart

        return (cbf_theta >= -1e-3) and (cbf_cart >= -1e-3) and (h_theta > 0) and (h_cart > 0)
