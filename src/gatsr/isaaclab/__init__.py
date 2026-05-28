"""Isaac Lab port of GATS-R.

This subpackage adapts every architectural idea from the CPU toy implementation
(BalanceBot) to NVIDIA Isaac Lab on the 37-DoF Unitree G1. Everything in here
*requires* the Isaac Sim runtime; importing the module without the simulator
running will raise. Use ``scripts/run_isaaclab.ps1`` (or the equivalent env-var
setup) to launch.

Components:
    * ``IsaacLabG1Env`` -- wraps an Isaac Lab ManagerBasedRLEnv so the GATS-R
      agent sees the same physical_state / observe / step interface as
      ``BalanceBotEnv``, with vectorised (num_envs,) batch semantics.
    * ``G1EnsembleLatentModel`` -- GPU-resident batched ensemble dynamics
      model trained on tuples collected from the live sim.
    * ``G1HybridPlanner`` -- batched MPPI in the latent model, optionally
      switched to short MCTS rollouts when the per-env budget allows.
    * ``G1CBFFilter`` -- base-tilt / joint-limit barrier on the G1's
      proprioceptive observation.
    * ``G1RuntimeMonitor`` -- ensemble-disagreement + temporal-consistency
      monitor adapted to (num_envs,) batched tensors.
    * ``G1RecoveryController`` -- placeholder PD stand-up controller invoked
      via the graph-indexed dispatcher.
    * ``G1Agent`` -- closed-loop integrator analogous to ``GATSRAgent``.
"""

from .env import IsaacLabG1Env, IsaacLabG1Config  # noqa: F401
from .latent import G1EnsembleLatentModel, G1LatentConfig  # noqa: F401
from .planner import G1BatchedMPPI, G1MPPIConfig  # noqa: F401
from .safety import G1CBFFilter, G1CBFConfig  # noqa: F401
from .monitor import G1RuntimeMonitor, G1MonitorConfig  # noqa: F401
from .recovery import G1RecoveryController, G1RecoveryConfig  # noqa: F401
from .agent import G1Agent, G1AgentConfig  # noqa: F401

__all__ = [
    "IsaacLabG1Env", "IsaacLabG1Config",
    "G1EnsembleLatentModel", "G1LatentConfig",
    "G1BatchedMPPI", "G1MPPIConfig",
    "G1CBFFilter", "G1CBFConfig",
    "G1RuntimeMonitor", "G1MonitorConfig",
    "G1RecoveryController", "G1RecoveryConfig",
    "G1Agent", "G1AgentConfig",
]
