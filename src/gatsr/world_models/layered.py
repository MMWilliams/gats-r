"""Layered orchestrator: chooses among L1, L2, and (when both fail) L3.

Pattern translated from the GATS paper:
    1. If L1 (analytic) is in its domain of validity → use L1's prediction.
    2. Else, query L2 (ensemble). If epistemic uncertainty is low → use L2.
    3. Else, hand off to L3 to *propose a new sub-goal* (the agent then
       replans toward a recoverable anchor).

The orchestrator returns a `ModelChoice` recording which layer fired and the
associated uncertainty/validity — these are surfaced to the runtime monitor
and used in the OOD-classification logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .analytic import AnalyticModel
from .latent import EnsembleLatentModel
from .fallback import FallbackProposer


class Layer(Enum):
    L1 = "analytic"
    L2 = "latent"
    L3 = "fallback"


@dataclass
class ModelChoice:
    layer: Layer
    predicted_next_state: np.ndarray | None
    epistemic_uncertainty: float
    analytic_validity: float
    fallback_proposals: list | None = None


class LayeredWorldModel:
    def __init__(
        self,
        analytic: AnalyticModel,
        latent: EnsembleLatentModel,
        fallback: FallbackProposer,
        l1_validity_threshold: float = 0.7,
        l2_epistemic_threshold: float = 0.5,
    ):
        self.l1 = analytic
        self.l2 = latent
        self.l3 = fallback
        self.l1_validity_threshold = l1_validity_threshold
        self.l2_epistemic_threshold = l2_epistemic_threshold

    def predict(
        self, physical_state: np.ndarray, action: np.ndarray, goal_x: float = 0.0
    ) -> ModelChoice:
        v = self.l1.validity(physical_state)
        if v >= self.l1_validity_threshold:
            s_next = self.l1.predict(physical_state, action)
            return ModelChoice(
                layer=Layer.L1,
                predicted_next_state=s_next,
                epistemic_uncertainty=0.0,
                analytic_validity=v,
            )
        s_next, eps = self.l2.predict_np(physical_state, action)
        if eps <= self.l2_epistemic_threshold:
            return ModelChoice(
                layer=Layer.L2,
                predicted_next_state=s_next,
                epistemic_uncertainty=eps,
                analytic_validity=v,
            )
        proposals = self.l3.propose(physical_state, goal_x)
        return ModelChoice(
            layer=Layer.L3,
            predicted_next_state=s_next,  # still return L2's guess for continuity
            epistemic_uncertainty=eps,
            analytic_validity=v,
            fallback_proposals=proposals,
        )

    def predict_traj(
        self, physical_state: np.ndarray, action_seq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Joint rollout using the layered policy. Returns
        (trajectory, per_step_epistemic, per_step_layer_index).
        For batched action_seq, the per-step layer choice is taken from the
        validity of the *predicted* state — implementing dynamic fallback over
        the rollout horizon.

        action_seq shape: (H, A) — single rollout."""
        H = action_seq.shape[0]
        traj = np.zeros((H, 4))
        eps = np.zeros(H)
        layer_idx = np.zeros(H, dtype=np.int32)
        s = physical_state.copy()
        for h in range(H):
            mc = self.predict(s, action_seq[h])
            traj[h] = mc.predicted_next_state
            eps[h] = mc.epistemic_uncertainty
            layer_idx[h] = list(Layer).index(mc.layer)
            s = mc.predicted_next_state
        return traj, eps, layer_idx
