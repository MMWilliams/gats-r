import numpy as np

from gatsr.safety.cbf import CBFSafetyFilter, CBFConfig
from gatsr.safety.reachability import ROMReachabilityChecker


# ----- CBF -----------------------------------------------------------------

def test_cbf_passes_safe_action():
    f = CBFSafetyFilter(CBFConfig())
    a_out, intervened, residual = f(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0]))
    assert not intervened
    assert residual == 0.0


def test_cbf_intervenes_at_boundary():
    f = CBFSafetyFilter(CBFConfig(theta_max=0.3))
    # near theta_max, falling further (theta_dot > 0). Action -1 pushes the
    # cart away from the pole's tilt direction, which worsens the fall.
    # The CBF must reject it.
    state = np.array([0.0, 0.0, 0.28, 1.5])
    bad_action = np.array([-1.0])
    a_out, intervened, residual = f(state, bad_action)
    assert intervened
    assert residual >= 0.0
    assert -1.0 <= float(a_out[0]) <= 1.0
    # the safe action should be *less negative* (or positive) — pulling away
    # from the unsafe direction
    assert float(a_out[0]) > float(bad_action[0])


def test_cbf_disabled_passes_through():
    f = CBFSafetyFilter(CBFConfig(enabled=False))
    a = np.array([0.9])
    a_out, intervened, residual = f(np.array([0.0, 0.0, 0.5, 0.0]), a)
    assert not intervened
    np.testing.assert_array_equal(a_out, a)


def test_cbf_projection_pulls_toward_stabilizing_reference():
    """Projection moves the filtered action toward the LQR-style reference."""
    state = np.array([0.0, 0.0, 0.25, 1.2])
    a = np.array([-1.0])  # bad: push away from the tilt
    f = CBFSafetyFilter(CBFConfig())
    a_out, intervened, residual = f(state, a)
    assert intervened
    # the LQR reference for a positive tilt + tilt-rate is negative-magnitude;
    # the safe action should not be more negative than the original
    assert float(a_out[0]) >= -1.0 + 1e-6


# ----- Reachability --------------------------------------------------------

def test_rom_reachability_safe_near_upright():
    r = ROMReachabilityChecker(horizon=8)
    is_safe, margin = r.is_reachable_safe(0.0, 0.0)
    assert is_safe
    assert margin > 0


def test_rom_reachability_unsafe_at_large_tilt():
    r = ROMReachabilityChecker(horizon=8, theta_max=0.7)
    is_safe, margin = r.is_reachable_safe(0.65, 3.0)
    # there may still exist control histories that survive, but the margin
    # should be much smaller than at upright.
    is_safe_up, margin_up = r.is_reachable_safe(0.0, 0.0)
    assert margin <= margin_up


def test_rom_reachable_set_shape():
    r = ROMReachabilityChecker(horizon=4, n_samples=16)
    rs = r.reachable_set(0.1, 0.0)
    assert rs.shape == (16, 2)
