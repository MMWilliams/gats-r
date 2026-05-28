"""Batched G1 runtime monitor — Sentinel-style ensemble + temporal-consistency
+ safe-stoppability proxy, vectorised across (num_envs,)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch


@dataclass
class G1MonitorConfig:
    epistemic_threshold: float = 1.0
    temporal_threshold: float = 0.3
    window: int = 8
    use_safe_stoppability: bool = True
    enabled: bool = True
    tilt_safe_stop: float = 0.9  # rad


class G1RuntimeMonitor:
    def __init__(self, cfg: G1MonitorConfig | None = None, num_envs: int = 1, device: str = "cuda:0"):
        self.cfg = cfg if cfg is not None else G1MonitorConfig()
        self.num_envs = num_envs
        self.device = torch.device(device)
        # circular buffer of past actions
        self._actions_buf = torch.zeros(self.cfg.window, num_envs, 1, device=self.device)
        self._fill = 0
        self._head = 0

    def reset(self) -> None:
        self._actions_buf.zero_()
        self._fill = 0
        self._head = 0

    @torch.inference_mode()
    def update(
        self,
        actions: torch.Tensor,           # (N, A)
        epistemic: torch.Tensor,         # (N,)
        physical_state: torch.Tensor,    # (N, S)
    ) -> dict:
        # scalarize action via L2 norm so the buffer stays small
        a_norm = actions.norm(dim=-1, keepdim=True)
        self._actions_buf[self._head] = a_norm
        self._head = (self._head + 1) % self.cfg.window
        self._fill = min(self._fill + 1, self.cfg.window)

        # temporal variance
        if self._fill >= 2:
            t_var = self._actions_buf[: self._fill].var(dim=0).squeeze(-1)
        else:
            t_var = torch.zeros(self.num_envs, device=self.device)

        if not self.cfg.enabled:
            return {
                "ood": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "epistemic": epistemic,
                "temporal_variance": t_var,
                "triggered_by": "disabled",
            }

        flag_eps = epistemic > self.cfg.epistemic_threshold
        flag_tvar = t_var > self.cfg.temporal_threshold
        if self.cfg.use_safe_stoppability:
            grav = physical_state[:, 6:9]
            tilt = torch.linalg.norm(grav[..., :2], dim=-1)
            stop_threshold = torch.sin(torch.as_tensor(self.cfg.tilt_safe_stop, device=self.device))
            flag_stop = tilt > stop_threshold
        else:
            flag_stop = torch.zeros_like(flag_eps)

        ood = flag_eps | flag_tvar | flag_stop
        # for diagnostics produce a string label per env (only meaningful when N small)
        triggered = []
        if self.num_envs <= 16:
            for i in range(self.num_envs):
                flags = []
                if flag_eps[i]: flags.append("epistemic")
                if flag_tvar[i]: flags.append("temporal")
                if flag_stop[i]: flags.append("safe_stop")
                triggered.append("|".join(flags) if flags else "none")
        else:
            triggered = "batched"
        return {
            "ood": ood,
            "epistemic": epistemic,
            "temporal_variance": t_var,
            "triggered_by": triggered,
            "safe_stoppable": ~flag_stop,
        }

    def calibrate(
        self,
        epistemic_samples: torch.Tensor,
        temporal_samples: torch.Tensor,
        quantile: float = 0.95,
    ) -> None:
        if epistemic_samples.numel():
            self.cfg.epistemic_threshold = float(epistemic_samples.flatten().quantile(quantile))
        if temporal_samples.numel():
            self.cfg.temporal_threshold = float(temporal_samples.flatten().quantile(quantile))
