"""Unit tests for the Isaac-Lab port that DO NOT require the simulator.

The wrapper subpackage is split so that pure-torch components (the latent
model, MPPI math, CBF/monitor/recovery numerics) are importable and testable
without booting Isaac Sim. The env wrapper imports gymnasium + isaaclab lazily
inside __init__ — we don't touch it here.

Run as part of the normal CPU test suite:
    pytest -q
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _has_working_cuda() -> bool:
    """Just `torch.cuda.is_available()` isn't enough on systems where the
    installed torch wheel doesn't have kernels for the host GPU (very common
    when the system Python has a different torch build than the Isaac Lab
    conda env). Run a tiny op to verify the kernel actually launches."""
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        _ = (x + 1).item()
        return True
    except Exception:
        return False


DEVICE = "cuda:0" if _has_working_cuda() else "cpu"


def _import_isaaclab_subpkg(name: str):
    """Imports gatsr.isaaclab.<name>; skips if Isaac Lab side imports fail."""
    try:
        return importlib.import_module(f"gatsr.isaaclab.{name}")
    except ModuleNotFoundError as e:
        pytest.skip(f"isaaclab port dep missing: {e}")


def test_latent_model_constructs_and_predicts():
    mod = _import_isaaclab_subpkg("latent")
    model = mod.G1EnsembleLatentModel(
        mod.G1LatentConfig(
            state_dim=16,
            action_dim=4,
            latent_dim=8,
            hidden=16,
            n_ensemble=2,
            device=DEVICE,
            multi_gpu_rollouts=False,
        )
    )
    s = torch.randn(3, 16, device=DEVICE)
    a = torch.randn(3, 4, device=DEVICE)
    s_next, eps = model.predict(s, a)
    assert s_next.shape == (3, 16)
    assert eps.shape == (3,)
    assert torch.all(eps >= 0)


def test_latent_model_rollout_batches():
    mod = _import_isaaclab_subpkg("latent")
    model = mod.G1EnsembleLatentModel(
        mod.G1LatentConfig(
            state_dim=8, action_dim=2, latent_dim=4, hidden=8, n_ensemble=2,
            device=DEVICE, multi_gpu_rollouts=False,
        )
    )
    s = torch.randn(4, 8, device=DEVICE)
    a = torch.randn(4, 6, 2, device=DEVICE)
    traj, eps = model.rollout(s, a)
    assert traj.shape == (4, 6, 8)
    assert eps.shape == (4, 6)


def test_latent_model_fits_loss_finite():
    mod = _import_isaaclab_subpkg("latent")
    model = mod.G1EnsembleLatentModel(
        mod.G1LatentConfig(
            state_dim=8, action_dim=2, latent_dim=4, hidden=8, n_ensemble=2,
            epochs=2, batch_size=64, device=DEVICE, multi_gpu_rollouts=False,
        )
    )
    N = 200
    s = torch.randn(N, 8, device=DEVICE)
    a = torch.rand(N, 2, device=DEVICE) * 2 - 1
    sp = s + 0.05 * torch.randn_like(s)
    info = model.fit(s, a, sp, verbose=False)
    assert info["final_loss"] == info["final_loss"]  # not NaN


def test_mppi_returns_correct_shape():
    latent_mod = _import_isaaclab_subpkg("latent")
    planner_mod = _import_isaaclab_subpkg("planner")
    N, A = 4, 2
    model = latent_mod.G1EnsembleLatentModel(latent_mod.G1LatentConfig(
        state_dim=8, action_dim=A, latent_dim=4, hidden=8, n_ensemble=2,
        device=DEVICE, multi_gpu_rollouts=False,
    ))
    planner = planner_mod.G1BatchedMPPI(
        planner_mod.G1MPPIConfig(horizon=4, n_samples=16, n_iter=1, device=DEVICE),
        model=model, num_envs=N, action_dim=A,
    )
    ps = torch.randn(N, 8, device=DEVICE)

    def cost_fn(traj, actions, eps):
        return traj.norm(dim=-1).sum(dim=-1)  # (N, K)

    seq = planner.plan(ps, cost_fn=cost_fn)
    assert seq.shape == (N, 4, A)
    assert torch.isfinite(seq).all()


def test_cbf_passes_safe_state():
    mod = _import_isaaclab_subpkg("safety")
    cbf = mod.G1CBFFilter(mod.G1CBFConfig())
    # upright G1: gravity in body frame is (0, 0, -1); horizontal ~ 0
    N, A = 4, 6
    ps = torch.zeros(N, 9 + 2 * A, device=DEVICE)
    ps[:, 8] = -1.0  # grav_z
    a = torch.zeros(N, A, device=DEVICE)
    a_out, intervened, residual = cbf(ps, a)
    assert (~intervened).all()
    assert torch.allclose(a_out, a)


def test_cbf_intervenes_when_tilted():
    mod = _import_isaaclab_subpkg("safety")
    cbf = mod.G1CBFFilter(mod.G1CBFConfig(tilt_max=0.1))
    N, A = 4, 6
    ps = torch.zeros(N, 9 + 2 * A, device=DEVICE)
    # heavy lateral grav component + adverse ang vel
    ps[:, 6] = 0.6  # horizontal grav x
    ps[:, 8] = -0.8
    ps[:, 3] = 2.0  # ang vel x = roll
    a = torch.ones(N, A, device=DEVICE)
    a_out, intervened, residual = cbf(ps, a)
    assert intervened.any()


def test_monitor_triggers_on_high_epistemic():
    mod = _import_isaaclab_subpkg("monitor")
    m = mod.G1RuntimeMonitor(mod.G1MonitorConfig(epistemic_threshold=0.1), num_envs=2, device=DEVICE)
    actions = torch.zeros(2, 4, device=DEVICE)
    ps = torch.zeros(2, 9, device=DEVICE)
    ps[:, 8] = -1.0
    eps = torch.tensor([0.5, 0.5], device=DEVICE)
    out = m.update(actions, eps, ps)
    assert out["ood"].all()


def test_monitor_disabled_never_flags():
    mod = _import_isaaclab_subpkg("monitor")
    m = mod.G1RuntimeMonitor(mod.G1MonitorConfig(enabled=False), num_envs=1, device=DEVICE)
    out = m.update(torch.zeros(1, 4, device=DEVICE), torch.tensor([10.0], device=DEVICE), torch.zeros(1, 9, device=DEVICE))
    assert not out["ood"].any()


def test_recovery_counts_attempts_and_successes():
    mod = _import_isaaclab_subpkg("recovery")
    N, A = 2, 4
    rec = mod.G1RecoveryController(num_envs=N, action_dim=A, device=DEVICE)
    # state: tilted (will trigger), then upright (will count success)
    ps = torch.zeros(N, 9 + 2 * A, device=DEVICE)
    ps[:, 6] = 0.5  # tilted
    ps[:, 8] = -0.85
    ood = torch.ones(N, dtype=torch.bool, device=DEVICE)
    fallen = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    a, mask = rec.step(ps, ood, fallen)
    assert mask.all()
    assert int(rec.attempts.sum().item()) == N
    # next step with upright state
    ps2 = ps.clone()
    ps2[:, 6] = 0.0
    ps2[:, 8] = -1.0
    a, mask = rec.step(ps2, torch.zeros_like(ood), fallen)
    assert int(rec.successes.sum().item()) >= 1
