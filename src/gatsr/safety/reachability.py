"""Reduced-order-model (ROM) reachability check.

The full thesis recommends *post-hoc reachability verification on a reduced
model* — Hamilton-Jacobi on the full G1 is infeasible, but it is feasible on
a low-dimensional reduced model (e.g., 3D-LIPM or capture point).

For BalanceBot the reduced model is a 2-D linear-inverted-pendulum
(`theta`, `theta_dot`). We compute the forward-reachable set under bounded
control and check that it stays within the safe envelope. This module
exposes `is_reachable_safe(state, horizon)` that other components can call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..envs.balance_env import GRAVITY, DT


@dataclass
class ROMReachabilityChecker:
    pole_length: float = 0.5
    dt: float = DT
    horizon: int = 8
    theta_max: float = 0.7
    u_max: float = 1.0
    n_samples: int = 32

    def reachable_set(self, theta0: float, theta_dot0: float) -> np.ndarray:
        """Sample-based forward reachable set under bounded control.

        Linearized: theta'' = (g/l) * theta - (1/l) * u
        u in [-u_max, +u_max]. Returns array of (theta_h, theta_dot_h) states."""
        L = self.pole_length
        rng = np.random.default_rng(0)
        # sample N control histories, simulate H steps
        u_hist = rng.uniform(-self.u_max, self.u_max, size=(self.n_samples, self.horizon))
        thetas = np.full(self.n_samples, theta0)
        theta_dots = np.full(self.n_samples, theta_dot0)
        out = np.zeros((self.n_samples, 2))
        for h in range(self.horizon):
            theta_ddot = (GRAVITY / L) * thetas - (1.0 / L) * u_hist[:, h]
            theta_dots = theta_dots + theta_ddot * self.dt
            thetas = thetas + theta_dots * self.dt
        out[:, 0] = thetas
        out[:, 1] = theta_dots
        return out

    def is_reachable_safe(self, theta0: float, theta_dot0: float) -> tuple[bool, float]:
        """True iff *at least one* control history stays within |theta| < theta_max
        for the whole horizon, AND the final |theta| is below the limit. We
        also return a margin: max over controls of (theta_max - max |theta|)."""
        L = self.pole_length
        rng = np.random.default_rng(0)
        u_hist = rng.uniform(-self.u_max, self.u_max, size=(self.n_samples, self.horizon))
        thetas = np.full(self.n_samples, theta0)
        theta_dots = np.full(self.n_samples, theta_dot0)
        max_abs_theta = np.zeros(self.n_samples)
        for h in range(self.horizon):
            theta_ddot = (GRAVITY / L) * thetas - (1.0 / L) * u_hist[:, h]
            theta_dots = theta_dots + theta_ddot * self.dt
            thetas = thetas + theta_dots * self.dt
            max_abs_theta = np.maximum(max_abs_theta, np.abs(thetas))
        safe_per_traj = max_abs_theta < self.theta_max
        is_safe = bool(safe_per_traj.any())
        margin = float(self.theta_max - np.min(max_abs_theta))
        return is_safe, margin
