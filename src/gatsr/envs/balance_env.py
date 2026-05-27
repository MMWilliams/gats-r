"""BalanceBot environment — a planar inverted-pendulum-on-a-cart with multiple
goals, stochastic disturbances, and an explicit recovery channel.

The environment is a controlled stand-in for the Isaac-Lab Unitree-G1 setup in
the parent thesis: every architectural mechanism (falling, recovery, OOD
generalization, long-horizon multi-goal planning) is preserved while running on
CPU in a few milliseconds per episode.

State (R^7):
    [x, x_dot, theta, theta_dot, goal_x, goal_idx_norm, fallen]

    - x, x_dot:        cart position / velocity
    - theta, theta_dot pole angle (rad, 0 = upright) / angular velocity
    - goal_x:          x-coordinate of the current goal
    - goal_idx_norm:   index of current goal / total goals (progress signal)
    - fallen:          binary flag — pole tilted beyond recoverable angle

Action (R^1):
    Continuous force on the cart in [-action_scale, +action_scale].

A separate `recover_step` channel applies a higher-authority controller for
get-up; the agent invokes it only when its runtime monitor decides the
nominal stack is OOD. This mirrors the FRASA / FIRM "recovery edge" idea.

Dynamics are intentionally simple but not trivial: the cart-pole is unstable
at theta = 0, the goal sequence requires planning over a horizon longer than
MPC can solve in one shot, and disturbances are calibrated so a non-robust
policy will eventually fall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# --- physical constants -----------------------------------------------------

GRAVITY = 9.81
DT = 0.02
MASS_CART_NOMINAL = 1.0
MASS_POLE_NOMINAL = 0.1
LENGTH_NOMINAL = 0.5
FRICTION_NOMINAL = 0.01


@dataclass
class BalanceBotConfig:
    """Configuration for BalanceBotEnv. Defaults match the in-distribution
    training regime; OOD evaluation scales `ood_level` to perturb dynamics,
    push impulses, payloads and friction simultaneously."""

    # episode structure
    max_steps: int = 500
    n_goals: int = 4
    goal_tolerance: float = 0.15
    track_half_width: float = 2.4

    # control
    action_scale: float = 12.0
    control_dt: float = DT

    # safety / failure
    fall_angle: float = 0.7  # ~40 deg — pole tipped, recovery required
    crash_angle: float = 1.2  # ~70 deg — past point of no return (episode ends)
    cart_limit: float = 2.4  # cart leaving the rail ends episode
    safety_margin_angle: float = 0.5  # ~28 deg — CBF trigger zone

    # disturbances and OOD
    base_force_noise: float = 0.05  # always-on actuator noise
    push_prob: float = 0.01  # per-step prob of an impulsive push
    push_strength: float = 4.0  # base push magnitude
    ood_level: float = 0.0  # in [0, 1] — scales perturbations during eval

    # reward shaping
    upright_bonus: float = 1.0
    goal_reward: float = 50.0
    fall_penalty: float = -20.0
    crash_penalty: float = -100.0
    step_penalty: float = 0.0

    # physical (set at reset; depends on ood_level)
    mass_cart: float = MASS_CART_NOMINAL
    mass_pole: float = MASS_POLE_NOMINAL
    pole_length: float = LENGTH_NOMINAL
    friction: float = FRICTION_NOMINAL

    seed: int = 0


class BalanceBotEnv:
    """Self-contained gym-like environment. No gymnasium dependency."""

    state_dim: int = 7
    action_dim: int = 1

    def __init__(self, config: BalanceBotConfig | None = None):
        self.cfg = config if config is not None else BalanceBotConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._state = np.zeros(4, dtype=np.float64)
        self.goals: np.ndarray = np.zeros(self.cfg.n_goals)
        self.goal_idx = 0
        self.t = 0
        self._dyn_cart_mass = self.cfg.mass_cart
        self._dyn_pole_mass = self.cfg.mass_pole
        self._dyn_pole_len = self.cfg.pole_length
        self._dyn_friction = self.cfg.friction

    # ----- gym API ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # sample physical parameters; OOD level widens the band
        s = 1.0 + 0.3 * self.cfg.ood_level
        self._dyn_cart_mass = self.cfg.mass_cart * self.rng.uniform(1 / s, s)
        self._dyn_pole_mass = self.cfg.mass_pole * self.rng.uniform(1 / s, s)
        self._dyn_pole_len = self.cfg.pole_length * self.rng.uniform(1 / s, s)
        self._dyn_friction = max(
            0.0,
            self.cfg.friction + 0.05 * self.cfg.ood_level * self.rng.standard_normal(),
        )

        # initial state: near upright, small velocity
        self._state = np.array(
            [
                self.rng.uniform(-0.05, 0.05),
                0.0,
                self.rng.uniform(-0.02, 0.02),
                0.0,
            ],
            dtype=np.float64,
        )
        # goals span the rail; force the agent to revisit positions in sequence
        self.goals = self.rng.uniform(
            -0.7 * self.cfg.track_half_width,
            0.7 * self.cfg.track_half_width,
            size=self.cfg.n_goals,
        )
        self.goal_idx = 0
        self.t = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        x, x_dot, th, th_dot = self._state
        goal_x = self.goals[self.goal_idx] if self.goal_idx < len(self.goals) else 0.0
        goal_idx_norm = self.goal_idx / max(1, len(self.goals))
        fallen = float(abs(th) > self.cfg.fall_angle)
        return np.array([x, x_dot, th, th_dot, goal_x, goal_idx_norm, fallen], dtype=np.float64)

    @property
    def physical_state(self) -> np.ndarray:
        """The 4-D physical state (x, x_dot, theta, theta_dot)."""
        return self._state.copy()

    def is_fallen(self) -> bool:
        return abs(self._state[2]) > self.cfg.fall_angle

    def is_crashed(self) -> bool:
        return (
            abs(self._state[2]) > self.cfg.crash_angle
            or abs(self._state[0]) > self.cfg.cart_limit
        )

    def current_goal(self) -> float:
        if self.goal_idx >= len(self.goals):
            return 0.0
        return float(self.goals[self.goal_idx])

    def goals_remaining(self) -> int:
        return max(0, len(self.goals) - self.goal_idx)

    # ----- step variants --------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Standard environment step (nominal control)."""
        return self._integrate(action, recovery=False)

    def recover_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Recovery channel: same dynamics, but the agent declares it is in a
        recovery mode (for logging) and disturbances are gated down to mimic
        the "stoppability + safe-set" assumption of FRASA/FIRM."""
        return self._integrate(action, recovery=True)

    # ----- dynamics -------------------------------------------------------

    def _integrate(
        self, action: np.ndarray, recovery: bool
    ) -> Tuple[np.ndarray, float, bool, dict]:
        a = float(np.clip(action, -1.0, 1.0)) * self.cfg.action_scale

        # actuator noise + impulsive disturbance
        noise = self.cfg.base_force_noise * (1.0 + 2.0 * self.cfg.ood_level)
        a += self.rng.standard_normal() * noise

        push = 0.0
        push_prob = self.cfg.push_prob * (1.0 + 3.0 * self.cfg.ood_level)
        if (not recovery) and self.rng.random() < push_prob:
            push = (
                self.cfg.push_strength
                * (1.0 + 2.0 * self.cfg.ood_level)
                * np.sign(self.rng.standard_normal())
            )

        x, x_dot, th, th_dot = self._state
        mc, mp, l, mu = (
            self._dyn_cart_mass,
            self._dyn_pole_mass,
            self._dyn_pole_len,
            self._dyn_friction,
        )

        f = a + push
        total_mass = mc + mp
        sin_t, cos_t = np.sin(th), np.cos(th)

        # classic cart-pole equations of motion with friction term
        denom = total_mass - mp * cos_t**2
        x_acc = (
            f
            + mp * l * (th_dot**2 * sin_t - GRAVITY * sin_t * cos_t / 1.0)
            - mu * x_dot
        ) / max(denom, 1e-6)
        th_acc = (GRAVITY * sin_t - cos_t * (f / total_mass)) / max(
            l * (4 / 3 - mp * cos_t**2 / total_mass / 1.0), 1e-6
        )

        x_dot += x_acc * self.cfg.control_dt
        x += x_dot * self.cfg.control_dt
        th_dot += th_acc * self.cfg.control_dt
        th += th_dot * self.cfg.control_dt

        self._state = np.array([x, x_dot, th, th_dot], dtype=np.float64)
        self.t += 1

        # reward
        reward = self.cfg.upright_bonus * np.cos(th) - self.cfg.step_penalty
        # goal progress
        if self.goal_idx < len(self.goals):
            goal_x = self.goals[self.goal_idx]
            dist = abs(x - goal_x)
            reward += -0.3 * dist
            if dist < self.cfg.goal_tolerance and abs(x_dot) < 1.0 and abs(th) < 0.2:
                reward += self.cfg.goal_reward
                self.goal_idx += 1

        info = {
            "fallen": self.is_fallen(),
            "crashed": self.is_crashed(),
            "recovery_active": recovery,
            "push": push,
            "goal_idx": self.goal_idx,
            "n_goals": len(self.goals),
            "physical_state": self._state.copy(),
        }
        # terminal conditions
        done = False
        if self.is_crashed():
            reward += self.cfg.crash_penalty
            done = True
            info["terminated"] = "crash"
        elif self.goal_idx >= len(self.goals):
            done = True
            info["terminated"] = "success"
        elif self.t >= self.cfg.max_steps:
            done = True
            info["terminated"] = "timeout"

        if self.is_fallen() and not done:
            reward += self.cfg.fall_penalty / max(1, self.cfg.max_steps)

        return self.observe(), float(reward), done, info

    # ----- helpers --------------------------------------------------------

    def clone(self) -> "BalanceBotEnv":
        """Cheap state clone for tree search / MCTS rollouts."""
        env = BalanceBotEnv(self.cfg)
        env._state = self._state.copy()
        env.goals = self.goals.copy()
        env.goal_idx = self.goal_idx
        env.t = self.t
        env._dyn_cart_mass = self._dyn_cart_mass
        env._dyn_pole_mass = self._dyn_pole_mass
        env._dyn_pole_len = self._dyn_pole_len
        env._dyn_friction = self._dyn_friction
        # use a derived rng so rollouts do not consume the real env's noise stream
        env.rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        return env

    def deterministic_clone(self) -> "BalanceBotEnv":
        """Clone with disturbances disabled — used by MPC/MCTS to score plans
        without stochastic distraction; the policy still faces noise at exec."""
        cfg = BalanceBotConfig(**{**self.cfg.__dict__})
        cfg.base_force_noise = 0.0
        cfg.push_prob = 0.0
        env = BalanceBotEnv(cfg)
        env._state = self._state.copy()
        env.goals = self.goals.copy()
        env.goal_idx = self.goal_idx
        env.t = self.t
        env._dyn_cart_mass = self._dyn_cart_mass
        env._dyn_pole_mass = self._dyn_pole_mass
        env._dyn_pole_len = self._dyn_pole_len
        env._dyn_friction = self._dyn_friction
        env.rng = np.random.default_rng(0)
        return env
