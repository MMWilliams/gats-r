"""Hybrid two-level planner.

High level: A* / Dijkstra on a learned skill graph to pick the next latent
sub-goal landmark.
Low level: continuous MCTS-with-VPW (or MPPI as a baseline switch) to drive
the system toward that sub-goal in the learned latent world model.

This is the direct continuous-control analog of the GATS "graph + search"
structure: the graph cuts the long-horizon problem into short edges, and the
continuous MCTS solves each edge under a layered model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .mppi import MPPIPlanner, MPPIConfig
from .mcts import ContinuousMCTS, MCTSConfig
from .skill_graph import SkillGraph, SkillNode


@dataclass
class HybridPlannerConfig:
    use_mcts: bool = True  # switch False for MPPI-only at the inner loop
    horizon: int = 12
    n_samples: int = 64
    n_mcts_sims: int = 48
    discount: float = 0.97
    subgoal_tolerance: float = 0.5
    seed: int = 0


class HybridPlanner:
    def __init__(
        self,
        cfg: HybridPlannerConfig,
        graph: SkillGraph,
        rollout_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        step_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        cost_for_subgoal: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        self.cfg = cfg
        self.graph = graph
        self.rollout = rollout_fn
        self.step = step_fn
        self.rng = np.random.default_rng(cfg.seed)
        self._current_subgoal_idx: Optional[int] = None
        self._current_path: List[int] = []
        # When the skill graph is ablated, the planner drives straight to the
        # raw goal instead of routing through landmark nodes.
        self._direct_subgoal_phys: Optional[np.ndarray] = None
        self._cost_subgoal = (
            cost_for_subgoal
            if cost_for_subgoal is not None
            else self._default_subgoal_cost
        )
        # inner planners (lazy)
        self._mppi: Optional[MPPIPlanner] = None
        self._mcts: Optional[ContinuousMCTS] = None

    # --- subgoal selection ----------------------------------------------

    def set_goal_physical(self, physical_state: np.ndarray, target_physical: np.ndarray) -> List[int]:
        self._direct_subgoal_phys = None
        start = self.graph.nearest_physical(physical_state)
        goal = self.graph.nearest_physical(target_physical)
        self._current_path = self.graph.shortest_path(start, goal)
        self._current_subgoal_idx = 1 if len(self._current_path) > 1 else 0
        return self._current_path

    def set_goal_direct(self, target_physical: np.ndarray) -> None:
        """Skill-graph-free goal setting: aim the inner loop straight at the raw
        target, bypassing landmark routing entirely (used by the no-graph
        ablation)."""
        self._direct_subgoal_phys = np.asarray(target_physical, dtype=np.float64)
        self._current_path = []
        self._current_subgoal_idx = None

    def route_to_recovery(self, physical_state: np.ndarray) -> List[int]:
        start = self.graph.nearest_physical(physical_state)
        rec = self.graph.recovery_node().idx
        self._current_path = self.graph.shortest_path(start, rec)
        self._current_subgoal_idx = 1 if len(self._current_path) > 1 else 0
        return self._current_path

    def current_subgoal(self) -> Optional[SkillNode]:
        if self._direct_subgoal_phys is not None:
            return SkillNode(
                idx=-1,
                latent=np.zeros(1),
                physical=self._direct_subgoal_phys,
                description="direct_goal",
            )
        if not self._current_path:
            return None
        i = min(self._current_subgoal_idx or 0, len(self._current_path) - 1)
        return self.graph.nodes[self._current_path[i]]

    def advance_if_reached(self, physical_state: np.ndarray) -> bool:
        sg = self.current_subgoal()
        if sg is None or self._current_subgoal_idx is None:
            return False
        d = float(np.linalg.norm(physical_state[:4] - sg.physical[:4]))
        if d < self.cfg.subgoal_tolerance and self._current_subgoal_idx + 1 < len(self._current_path):
            self._current_subgoal_idx += 1
            return True
        return False

    # --- inner planning --------------------------------------------------

    def _default_subgoal_cost(self, traj: np.ndarray, subgoal_phys: np.ndarray) -> float:
        # distance to subgoal at end of horizon + cumulative upright bonus
        end_dist = np.linalg.norm(traj[..., -1, :4] - subgoal_phys[None, :4], axis=-1)
        upright = -np.cos(traj[..., 2]).mean(axis=-1)
        return end_dist + 0.5 * upright

    def plan_action(self, physical_state: np.ndarray) -> Tuple[np.ndarray, dict]:
        sg = self.current_subgoal()
        if sg is None:
            return np.zeros(1), {"reason": "no_subgoal"}
        subgoal_phys = sg.physical

        # cost function for MPPI / MCTS reward
        def cost_fn(traj_b: np.ndarray, actions_b: np.ndarray, eps_b: np.ndarray) -> np.ndarray:
            # traj_b: (B, H, S)
            end_dist = np.linalg.norm(traj_b[:, -1, :4] - subgoal_phys[None, :4], axis=-1)
            upright = -np.mean(np.cos(traj_b[:, :, 2]), axis=-1)
            action_cost = 0.01 * np.mean(actions_b ** 2, axis=(-1, -2))
            unc_cost = 0.1 * np.mean(eps_b, axis=-1)
            return end_dist + 0.5 * upright + action_cost + unc_cost

        if self.cfg.use_mcts:
            if self._mcts is None:
                mc_cfg = MCTSConfig(
                    horizon=min(6, self.cfg.horizon),
                    n_simulations=self.cfg.n_mcts_sims,
                    action_dim=1,
                    discount=self.cfg.discount,
                    seed=self.cfg.seed,
                )

                def step_fn(s, a):
                    return self.step(s, a)

                def reward_fn(s, a, sp):
                    dist = float(np.linalg.norm(sp[:4] - subgoal_phys[:4]))
                    upright = float(np.cos(sp[2]))
                    return upright - 0.4 * dist - 0.01 * float(np.sum(a ** 2))

                self._mcts = ContinuousMCTS(mc_cfg, step_fn=step_fn, reward_fn=reward_fn)
            # MCTS does not allow per-call subgoal injection cleanly, so rebuild
            # the reward closure each call by mutating in place
            self._mcts.reward_fn = lambda s, a, sp: (
                float(np.cos(sp[2]))
                - 0.4 * float(np.linalg.norm(sp[:4] - subgoal_phys[:4]))
                - 0.01 * float(np.sum(a ** 2))
            )
            a, info = self._mcts.plan(physical_state[:4])
            info["mode"] = "mcts"
            return np.atleast_1d(a), info
        else:
            if self._mppi is None:
                mp_cfg = MPPIConfig(
                    horizon=self.cfg.horizon, n_samples=self.cfg.n_samples, seed=self.cfg.seed
                )
                self._mppi = MPPIPlanner(mp_cfg, rollout_fn=self.rollout, cost_fn=cost_fn)
            else:
                self._mppi.cost = cost_fn
            seq = self._mppi.plan(physical_state[:4])
            return seq[0], {"mode": "mppi"}
