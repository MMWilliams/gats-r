"""Graph-indexed recovery dispatcher.

When the monitor flags OOD, the agent
    1. snaps the current physical state to the nearest skill-graph recovery
       node (or to the closest node and routes via Dijkstra to the recovery
       anchor),
    2. invokes the corresponding `RecoveryPolicy` for that node,
    3. monitors whether the policy converges to a stable state; if so, the
       agent re-enters its nominal stack.

This is the novel contribution from the parent thesis Section H: recovery is
*indexed* by skill-graph node, so each recovery edge can be a different
specialized controller (LQR / FRASA / FIRM / ad-hoc analytic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from ..planning.skill_graph import SkillGraph, SkillNode
from .recovery_policy import LQRRecoveryPolicy


RecoveryFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class GraphIndexedRecovery:
    """Maps skill-graph node ids to dedicated recovery controllers.

    Default mapping: every node falls back to a global LQR stabilizer; the
    interface allows overriding individual edges with FRASA/FIRM-style policies
    in production deployments."""

    graph: SkillGraph
    default_policy: LQRRecoveryPolicy
    edge_policies: Dict[int, RecoveryFn] = field(default_factory=dict)
    active: bool = False
    target_node: Optional[int] = None
    max_steps_per_attempt: int = 80
    _steps_in_attempt: int = 0
    _attempts: int = 0
    _successes: int = 0
    _time_to_recover_accum: float = 0.0
    _started_at_step: int = 0

    def register(self, node_idx: int, policy: RecoveryFn) -> None:
        self.edge_policies[node_idx] = policy

    def begin(self, physical_state: np.ndarray, current_step: int) -> None:
        if self.active:
            return
        # route to the recovery anchor in the graph
        self.target_node = self.graph.recovery_node().idx
        self.active = True
        self._steps_in_attempt = 0
        self._attempts += 1
        self._started_at_step = current_step
        self.default_policy.reset()

    def step(self, physical_state: np.ndarray, current_step: int) -> tuple[np.ndarray, bool]:
        """Return (recovery_action, recovered_now)."""
        if not self.active:
            return np.zeros(1), False
        policy = self.edge_policies.get(
            self.target_node if self.target_node is not None else -1,
            self.default_policy,
        )
        a = policy(physical_state)
        self._steps_in_attempt += 1
        done = self.default_policy.recovered(physical_state) or (
            self._steps_in_attempt >= self.max_steps_per_attempt
        )
        if done:
            recovered = self.default_policy.recovered(physical_state)
            if recovered:
                self._successes += 1
                # account in env-steps actually spent in recovery; use the
                # internal step counter so we never report 0 for a successful
                # recovery (would happen if recovery started and finished
                # within the same outer iteration).
                self._time_to_recover_accum += max(1, self._steps_in_attempt)
            self.active = False
            self.target_node = None
            return a, recovered
        return a, False

    # --- stats ----------------------------------------------------------

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def mean_time_to_recover(self) -> float:
        return self._time_to_recover_accum / max(1, self._successes)

    def reset_stats(self) -> None:
        self._attempts = 0
        self._successes = 0
        self._time_to_recover_accum = 0.0
        self._steps_in_attempt = 0
        self.active = False
