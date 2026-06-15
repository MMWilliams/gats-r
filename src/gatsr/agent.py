"""GATS-R agent — graph-augmented layered-world-model RL with graph-indexed recovery.

This module wires every component together in a clean closed loop:

    obs -> physical_state
        -> layered world model (L1/L2/L3) reports validity + epistemic
        -> hybrid planner (skill-graph A* + MCTS+VPW or MPPI) proposes action
        -> CBF filter projects action onto safe set (optional)
        -> monitor (Sentinel-style) decides OOD?
              YES -> graph-indexed recovery takes over until stable
              NO  -> nominal action goes to environment
        -> env.step / env.recover_step

The same class is reused by ablations (disable mcts, disable monitor, disable
CBF, disable recovery, ...) and by training (collect rollouts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch

from .envs.balance_env import BalanceBotEnv
from .world_models.analytic import AnalyticModel
from .world_models.latent import EnsembleLatentModel, LatentModelConfig
from .world_models.fallback import FallbackProposer
from .world_models.layered import LayeredWorldModel, Layer
from .planning.skill_graph import SkillGraph
from .planning.planner import HybridPlanner, HybridPlannerConfig
from .safety.cbf import CBFSafetyFilter, CBFConfig
from .monitoring.monitor import RuntimeMonitor, MonitorConfig, MonitorDecision
from .recovery.recovery_policy import LQRRecoveryPolicy, RecoveryConfig
from .recovery.recovery_graph import GraphIndexedRecovery


@dataclass
class AgentConfig:
    use_layered: bool = True
    use_skill_graph: bool = True
    use_mcts: bool = True  # otherwise MPPI inner loop
    use_cbf: bool = True
    use_monitor: bool = True
    use_recovery: bool = True
    planning_horizon: int = 12
    n_mppi_samples: int = 64
    n_mcts_simulations: int = 48
    seed: int = 0


@dataclass
class EpisodeStats:
    steps: int = 0
    ep_return: float = 0.0
    success: int = 0
    failures_detected: int = 0
    recoveries_attempted: int = 0
    recoveries_succeeded: int = 0
    safety_violations: int = 0
    time_to_recover_accum: float = 0.0
    time_to_recover_count: int = 0
    planning_ms_sum: float = 0.0


class GATSRAgent:
    """The full integrator."""

    def __init__(
        self,
        cfg: AgentConfig,
        env: BalanceBotEnv,
        latent_model: EnsembleLatentModel,
        skill_graph: SkillGraph,
    ):
        self.cfg = cfg
        self.env = env

        # world model layers
        self.analytic = AnalyticModel(
            mass_cart=env.cfg.mass_cart,
            mass_pole=env.cfg.mass_pole,
            pole_length=env.cfg.pole_length,
        )
        self.latent = latent_model
        self.fallback = FallbackProposer(rng=np.random.default_rng(cfg.seed))
        self.layered = LayeredWorldModel(
            analytic=self.analytic, latent=self.latent, fallback=self.fallback
        )

        # planner
        def rollout_fn(state: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # use L2 ensemble for rollouts (the planner's workhorse model)
            return self.latent.rollout_np(state, actions)

        def step_fn(state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
            return self.latent.predict_np(state, action)

        self.skill_graph = skill_graph
        self.planner = HybridPlanner(
            HybridPlannerConfig(
                use_mcts=cfg.use_mcts,
                horizon=cfg.planning_horizon,
                n_samples=cfg.n_mppi_samples,
                n_mcts_sims=cfg.n_mcts_simulations,
                seed=cfg.seed,
            ),
            graph=self.skill_graph,
            rollout_fn=rollout_fn,
            step_fn=step_fn,
        )

        # safety, monitor, recovery
        self.cbf = CBFSafetyFilter(CBFConfig(enabled=cfg.use_cbf))
        self.monitor = RuntimeMonitor(MonitorConfig(enabled=cfg.use_monitor))
        self.recovery = GraphIndexedRecovery(
            graph=self.skill_graph,
            default_policy=LQRRecoveryPolicy(
                RecoveryConfig(
                    pole_length=env.cfg.pole_length,
                    mass_pole=env.cfg.mass_pole,
                    mass_cart=env.cfg.mass_cart,
                )
            ),
        )

    # ----- actions ------------------------------------------------------

    def act(
        self, physical_state: np.ndarray, current_goal: float, step: int
    ) -> tuple[np.ndarray, MonitorDecision, dict]:
        info: dict = {}

        # 1. Set / refresh planner subgoal
        target = np.array([current_goal, 0.0, 0.0, 0.0])
        if not self.cfg.use_skill_graph:
            # Ablation: bypass the landmark graph entirely, drive straight to goal.
            self.planner.set_goal_direct(target)
        elif not self.planner._current_path:
            self.planner.set_goal_physical(physical_state, target)
        # advance through landmarks if reached
        self.planner.advance_if_reached(physical_state)

        # 2. Inner-loop plan
        t0 = time.perf_counter()
        nominal_action, plan_info = self.planner.plan_action(physical_state)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        info["plan_ms"] = plan_ms
        info["plan_info"] = plan_info

        # 3. Layered world model selects the controller.
        #    L1 (analytic) is used for *control* when in its domain of validity:
        #    the goal-tracking LQR is exact near upright and far cheaper/safer
        #    than planning through the learned L2 model. Outside L1 validity we
        #    fall back to the L2 planner's action (computed above). This makes
        #    the L1/L2/L3 selection drive control, not just report uncertainty.
        if self.cfg.use_layered:
            choice = self.layered.predict(
                physical_state, nominal_action, goal_x=current_goal
            )
            epistemic = choice.epistemic_uncertainty
            info["layer"] = choice.layer.value
            info["validity"] = choice.analytic_validity
            info["epistemic"] = epistemic
            if choice.layer == Layer.L1:
                nominal_action = self.analytic.control(physical_state, goal_x=current_goal)
                info["control_source"] = "L1"
            else:
                info["control_source"] = "L2"
        else:
            epistemic = 0.0
            info["layer"] = "n/a"
            info["control_source"] = "L2"

        # 4. CBF filter
        action_safe, intervened, residual = self.cbf(physical_state, nominal_action)
        info["cbf_intervened"] = intervened
        info["cbf_residual"] = residual

        # 5. Monitor
        if self.cfg.use_monitor:
            decision = self.monitor.update(
                action_safe, epistemic_uncertainty=epistemic, physical_state=physical_state
            )
        else:
            decision = MonitorDecision(
                ood=False, epistemic=epistemic, temporal_variance=0.0, triggered_by="disabled"
            )
        info["monitor"] = decision

        # 6. Decide nominal vs recovery
        if self.cfg.use_recovery and (
            decision.ood or self.env.is_fallen() or self.recovery.active
        ):
            if not self.recovery.active:
                self.recovery.begin(physical_state, step)
            a_rec, recovered = self.recovery.step(physical_state, step)
            info["recovery_active"] = True
            info["recovered_now"] = recovered
            if recovered:
                # reset the high-level path so we re-plan from the recovered state
                self.planner._current_path = []
                self.planner._current_subgoal_idx = None
            return a_rec, decision, info
        info["recovery_active"] = False
        return action_safe, decision, info

    # ----- training data collection ---------------------------------------

    @torch.no_grad()
    def random_collect(
        self, n_steps: int = 4000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Roll random actions to seed the latent model + skill graph."""
        states = []
        actions = []
        next_states = []
        physicals = []
        obs = self.env.reset(seed=self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)
        for _ in range(n_steps):
            ps = self.env.physical_state
            a = rng.uniform(-1.0, 1.0, size=(1,))
            obs, _r, done, _info = self.env.step(a)
            states.append(ps)
            actions.append(a)
            next_states.append(self.env.physical_state)
            physicals.append(self.env.physical_state)
            if done:
                obs = self.env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        return (
            np.array(states, dtype=np.float64),
            np.array(actions, dtype=np.float64),
            np.array(next_states, dtype=np.float64),
            np.array(physicals, dtype=np.float64),
        )

    # ----- evaluation ---------------------------------------------------

    def evaluate(
        self,
        episodes: int = 5,
        seed_offset: int = 0,
    ) -> list[EpisodeStats]:
        out: list[EpisodeStats] = []
        for ep in range(episodes):
            self.env.reset(seed=self.cfg.seed + seed_offset + ep)
            self.monitor.reset()
            self.recovery.reset_stats()
            stats = EpisodeStats()
            done = False
            while not done:
                ps = self.env.physical_state
                a, decision, info = self.act(ps, self.env.current_goal(), stats.steps)
                if info.get("cbf_intervened", False):
                    stats.safety_violations += 1
                if decision.ood:
                    stats.failures_detected += 1
                in_recov = info.get("recovery_active", False)
                if in_recov:
                    obs, r, done, einfo = self.env.recover_step(a)
                else:
                    obs, r, done, einfo = self.env.step(a)
                stats.ep_return += r
                stats.steps += 1
                stats.planning_ms_sum += info.get("plan_ms", 0.0)
                if info.get("recovered_now", False):
                    stats.recoveries_succeeded += 1
                if self.env.is_crashed():
                    break
            stats.recoveries_attempted = self.recovery.attempts
            stats.recoveries_succeeded = self.recovery.successes
            stats.time_to_recover_accum = self.recovery._time_to_recover_accum
            stats.time_to_recover_count = self.recovery.successes
            term = einfo.get("terminated", "")
            stats.success = int(term == "success")
            out.append(stats)
        return out
