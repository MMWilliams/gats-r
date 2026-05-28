"""G1 recovery controller — a *placeholder* PD stand-up policy.

Production replacements: FRASA / HiFAR / FIRM / "Learning to Get Up Across
Morphologies". For the smoke benchmark we use a simple proportional joint
controller toward the G1's default upright pose, which is a reasonable proxy:
it converges from small disturbances and fails on hard falls (so the metric
"recovery success rate" still has signal).

Interface mirrors ``GraphIndexedRecovery`` from the CPU port.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class G1RecoveryConfig:
    kp: float = 6.0
    kd: float = 0.2
    max_steps_per_attempt: int = 60
    recovered_tilt: float = 0.2


class G1RecoveryController:
    def __init__(
        self,
        cfg: G1RecoveryConfig | None = None,
        num_envs: int = 1,
        action_dim: int = 37,
        device: str = "cuda:0",
        default_joint_pos: torch.Tensor | None = None,
    ):
        self.cfg = cfg if cfg is not None else G1RecoveryConfig()
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        if default_joint_pos is None:
            default_joint_pos = torch.zeros(action_dim, device=self.device)
        self.default_joint_pos = default_joint_pos.to(self.device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.attempts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.successes = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.time_to_recover_accum = torch.zeros(num_envs, dtype=torch.float, device=self.device)

    def reset(self) -> None:
        self.active.zero_()
        self.steps.zero_()
        self.attempts.zero_()
        self.successes.zero_()
        self.time_to_recover_accum.zero_()

    @torch.inference_mode()
    def step(
        self,
        physical_state: torch.Tensor,
        ood: torch.Tensor,
        fallen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (recovery_action: (N, A), recovery_mask: (N,))."""
        N, A = self.num_envs, self.action_dim
        trigger = ood | fallen
        # start new recovery attempts where (trigger AND not active)
        new = trigger & ~self.active
        self.active = self.active | new
        self.attempts = self.attempts + new.long()
        self.steps = torch.where(new, torch.zeros_like(self.steps), self.steps)

        # PD action toward default joint pose
        # state layout: [lin_vel(3), ang_vel(3), grav(3), jp(N), jv(N)]
        jp = physical_state[:, 9 : 9 + A]
        jv = physical_state[:, 9 + A : 9 + 2 * A]
        # if joint counts don't match (e.g., fewer DoF in observed proprio),
        # fall back to a zero action so we don't crash
        if jp.shape[1] != A:
            action = torch.zeros(N, A, device=self.device)
        else:
            action = -self.cfg.kp * (jp - self.default_joint_pos) - self.cfg.kd * jv
            action = action.clamp(-1.0, 1.0)

        # check if recovered
        grav = physical_state[:, 6:9]
        tilt = torch.linalg.norm(grav[..., :2], dim=-1)
        threshold = torch.sin(torch.as_tensor(self.cfg.recovered_tilt, device=self.device))
        recovered = self.active & (tilt < threshold)

        # timeout
        self.steps = torch.where(self.active, self.steps + 1, self.steps)
        timed_out = self.active & (self.steps >= self.cfg.max_steps_per_attempt)

        # accounting
        self.successes = self.successes + recovered.long()
        self.time_to_recover_accum = self.time_to_recover_accum + torch.where(
            recovered, self.steps.float(), torch.zeros_like(self.time_to_recover_accum)
        )

        # deactivate where finished
        finished = recovered | timed_out
        self.active = self.active & ~finished

        # mask: which envs return a recovery action this step
        recovery_mask = self.active | new
        out_action = torch.where(recovery_mask.unsqueeze(-1), action, torch.zeros_like(action))
        return out_action, recovery_mask

    @property
    def mean_time_to_recover(self) -> float:
        s = self.successes.sum().item()
        if s == 0:
            return -1.0
        return float(self.time_to_recover_accum.sum().item() / s)
