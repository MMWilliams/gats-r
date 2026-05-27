"""Runtime monitor — Sentinel-style two-signal OOD detector.

Sentinel (Agia et al., CoRL 2024) found that *combining* ensemble disagreement
with temporal consistency catches ~18% more failures than either alone. We
replicate the structure here:

    1. Ensemble disagreement: epistemic uncertainty from the L2 ensemble.
    2. Temporal consistency: rolling variance of the most recent action chunk
       (a proxy for "model is generating jittery actions because it is lost").

Either signal exceeding its calibrated threshold triggers an OOD decision.

Thresholds are calibrated on in-distribution data via a percentile rule:
take the 95th percentile of each signal during nominal operation and use
that as the trigger. Calibrate once at training time, freeze for evaluation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


@dataclass
class MonitorConfig:
    epistemic_threshold: float = 0.5
    temporal_threshold: float = 0.5
    window: int = 8
    use_safe_stoppability: bool = True  # 2026 preprint: learned safe-stop monitor
    enabled: bool = True


@dataclass
class MonitorDecision:
    ood: bool
    epistemic: float
    temporal_variance: float
    triggered_by: str  # 'none' | 'epistemic' | 'temporal' | 'safe_stop' | 'multiple'
    safe_stoppable: bool = True


@dataclass
class RuntimeMonitor:
    cfg: MonitorConfig = field(default_factory=MonitorConfig)
    _action_window: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    _epistemic_window: Deque[float] = field(default_factory=lambda: deque(maxlen=16))

    def __post_init__(self) -> None:
        self._action_window = deque(maxlen=self.cfg.window)
        self._epistemic_window = deque(maxlen=max(16, self.cfg.window * 2))

    def reset(self) -> None:
        self._action_window.clear()
        self._epistemic_window.clear()

    def update(
        self,
        action: np.ndarray,
        epistemic_uncertainty: float,
        physical_state: Optional[np.ndarray] = None,
    ) -> MonitorDecision:
        a_scalar = float(np.asarray(action).flatten()[0])
        self._action_window.append(a_scalar)
        self._epistemic_window.append(epistemic_uncertainty)

        eps = float(np.mean(self._epistemic_window)) if self._epistemic_window else 0.0
        if len(self._action_window) >= 2:
            t_var = float(np.var(self._action_window))
        else:
            t_var = 0.0
        safe_stoppable = True
        if self.cfg.use_safe_stoppability and physical_state is not None:
            # 2026 safe-stoppability proxy: pole well within recoverable wedge
            #   and cart inside the track. Returns False if we predict a fall
            #   that the recovery policy cannot undo.
            theta = physical_state[2]
            x = physical_state[0]
            safe_stoppable = bool(abs(theta) < 0.9 and abs(x) < 2.2)

        if not self.cfg.enabled:
            return MonitorDecision(
                ood=False,
                epistemic=eps,
                temporal_variance=t_var,
                triggered_by="disabled",
                safe_stoppable=safe_stoppable,
            )

        flags = []
        if eps > self.cfg.epistemic_threshold:
            flags.append("epistemic")
        if t_var > self.cfg.temporal_threshold:
            flags.append("temporal")
        if not safe_stoppable:
            flags.append("safe_stop")

        if not flags:
            return MonitorDecision(
                ood=False,
                epistemic=eps,
                temporal_variance=t_var,
                triggered_by="none",
                safe_stoppable=safe_stoppable,
            )
        triggered = "multiple" if len(flags) > 1 else flags[0]
        return MonitorDecision(
            ood=True,
            epistemic=eps,
            temporal_variance=t_var,
            triggered_by=triggered,
            safe_stoppable=safe_stoppable,
        )

    def calibrate(
        self,
        epistemic_samples: np.ndarray,
        temporal_samples: np.ndarray,
        quantile: float = 0.95,
    ) -> None:
        """Set thresholds to the requested quantile of nominal-data samples."""
        if epistemic_samples.size:
            self.cfg.epistemic_threshold = float(np.quantile(epistemic_samples, quantile))
        if temporal_samples.size:
            self.cfg.temporal_threshold = float(np.quantile(temporal_samples, quantile))
