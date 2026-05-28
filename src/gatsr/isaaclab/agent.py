"""G1 GATS-R agent — closed-loop integrator.

Single class that:
    1. Asks the latent world model for an epistemic-uncertainty signal.
    2. Runs batched MPPI on the latent model to propose a nominal action.
    3. Projects via the CBF filter.
    4. Updates the Sentinel-style monitor.
    5. Routes per-env to the recovery controller when monitor.ood OR is_fallen.
    6. Returns the final per-env actions for the env.step / env.recover_step.

Returns a populated stats dict so the benchmark can aggregate metrics matching
the CPU port: success_rate, recovery_attempts, recovery_successes, time-to-
recover, safety_violations, planning_ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .env import IsaacLabG1Env
from .latent import G1EnsembleLatentModel
from .planner import G1BatchedMPPI, G1MPPIConfig
from .safety import G1CBFFilter, G1CBFConfig
from .monitor import G1RuntimeMonitor, G1MonitorConfig
from .recovery import G1RecoveryController, G1RecoveryConfig


@dataclass
class G1AgentConfig:
    use_mppi: bool = True
    use_cbf: bool = True
    use_monitor: bool = True
    use_recovery: bool = True
    horizon: int = 16
    n_samples: int = 96
    n_iter: int = 2
    seed: int = 0


class G1Agent:
    def __init__(
        self,
        env: IsaacLabG1Env,
        latent_model: G1EnsembleLatentModel,
        cfg: G1AgentConfig | None = None,
    ):
        self.env = env
        self.model = latent_model
        self.cfg = cfg if cfg is not None else G1AgentConfig()
        N = env.num_envs
        A = env.action_dim
        device = str(env.device)

        self.planner = G1BatchedMPPI(
            G1MPPIConfig(
                horizon=self.cfg.horizon,
                n_samples=self.cfg.n_samples,
                n_iter=self.cfg.n_iter,
                device=device,
            ),
            model=latent_model,
            num_envs=N,
            action_dim=A,
        )
        self.cbf = G1CBFFilter(G1CBFConfig(enabled=self.cfg.use_cbf))
        self.monitor = G1RuntimeMonitor(
            G1MonitorConfig(enabled=self.cfg.use_monitor),
            num_envs=N,
            device=device,
        )
        self.recovery = G1RecoveryController(
            G1RecoveryConfig(),
            num_envs=N,
            action_dim=A,
            device=device,
        )

        # running counters (per env, cumulative within an evaluation)
        self.safety_violations = torch.zeros(N, dtype=torch.long, device=env.device)
        self.failures_detected = torch.zeros(N, dtype=torch.long, device=env.device)
        self.planning_ms_sum = 0.0
        self.planning_steps = 0

    # ----- action selection -----------------------------------------------

    @torch.inference_mode()
    def act(self, physical_state: torch.Tensor, cost_fn) -> tuple[torch.Tensor, dict]:
        info: dict = {}
        t0 = time.perf_counter()
        if self.cfg.use_mppi:
            seq = self.planner.plan(physical_state, cost_fn=cost_fn)
            nominal = seq[:, 0]  # (N, A)
        else:
            nominal = torch.zeros(self.env.num_envs, self.env.action_dim, device=self.env.device)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        info["plan_ms"] = plan_ms
        self.planning_ms_sum += plan_ms
        self.planning_steps += 1

        # epistemic via a one-step prediction sanity check
        _, eps = self.model.predict(physical_state, nominal)
        info["epistemic"] = eps

        # CBF
        nominal, cbf_intervened, cbf_residual = self.cbf(physical_state, nominal)
        self.safety_violations = self.safety_violations + cbf_intervened.long()
        info["cbf_intervened"] = cbf_intervened
        info["cbf_residual"] = cbf_residual

        # monitor
        monitor_out = self.monitor.update(nominal, eps, physical_state)
        ood = monitor_out["ood"]
        self.failures_detected = self.failures_detected + ood.long()
        info["monitor"] = monitor_out

        # recovery
        if self.cfg.use_recovery:
            fallen = self.env.is_fallen()
            rec_action, rec_mask = self.recovery.step(physical_state, ood, fallen)
            actions = torch.where(rec_mask.unsqueeze(-1), rec_action, nominal)
            info["recovery_mask"] = rec_mask
        else:
            actions = nominal
            info["recovery_mask"] = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

        return actions, info

    def reset_stats(self) -> None:
        self.safety_violations.zero_()
        self.failures_detected.zero_()
        self.planning_ms_sum = 0.0
        self.planning_steps = 0
        self.recovery.reset()
        self.monitor.reset()
        self.planner.reset()
