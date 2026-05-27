"""L3: generative / fallback sub-goal proposer.

In the full thesis this is a VLM that proposes sub-goals when L1 is out of
validity and L2 is uncertain. Here we instantiate it as a small heuristic +
random proposer over goal positions and "stand-still recovery" macro-actions.
The interface is intentionally identical to what a VLM-backed proposer would
return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SubGoalProposal:
    """A single proposal returned by the L3 fallback. The agent treats this
    as a candidate node to insert into the skill graph and plan towards."""

    target_state: np.ndarray  # desired physical state to reach
    description: str
    confidence: float  # rough self-confidence used for ranking


class FallbackProposer:
    def __init__(self, n_proposals: int = 4, rng: np.random.Generator | None = None):
        self.n_proposals = n_proposals
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def propose(self, current_state: np.ndarray, goal_x: float) -> List[SubGoalProposal]:
        """Always returns at least one safe, conservative proposal (upright at 0)."""
        proposals: List[SubGoalProposal] = []
        # 1) "regain upright at current x"  -- recovery anchor
        proposals.append(
            SubGoalProposal(
                target_state=np.array([current_state[0], 0.0, 0.0, 0.0]),
                description="regain_upright_in_place",
                confidence=0.9,
            )
        )
        # 2) "move halfway to goal, upright"
        mid_x = 0.5 * (current_state[0] + goal_x)
        proposals.append(
            SubGoalProposal(
                target_state=np.array([mid_x, 0.0, 0.0, 0.0]),
                description="halfway_to_goal_upright",
                confidence=0.7,
            )
        )
        # 3) "go directly to the goal"
        proposals.append(
            SubGoalProposal(
                target_state=np.array([goal_x, 0.0, 0.0, 0.0]),
                description="direct_to_goal",
                confidence=0.6,
            )
        )
        # 4..N) random nearby anchors (acts like VLM diversity)
        for _ in range(max(0, self.n_proposals - 3)):
            jitter = self.rng.uniform(-0.4, 0.4)
            proposals.append(
                SubGoalProposal(
                    target_state=np.array(
                        [current_state[0] + jitter, 0.0, 0.0, 0.0]
                    ),
                    description=f"jittered_anchor_{jitter:+.2f}",
                    confidence=0.4,
                )
            )
        return proposals[: self.n_proposals]
