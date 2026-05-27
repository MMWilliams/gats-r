import numpy as np

from gatsr.monitoring.monitor import RuntimeMonitor, MonitorConfig


def test_monitor_clean_signal_does_not_trigger():
    m = RuntimeMonitor(MonitorConfig(window=8, epistemic_threshold=10.0, temporal_threshold=10.0))
    for _ in range(20):
        d = m.update(np.array([0.0]), epistemic_uncertainty=0.0)
        assert not d.ood


def test_monitor_high_epistemic_triggers():
    m = RuntimeMonitor(MonitorConfig(epistemic_threshold=0.1, temporal_threshold=10.0))
    triggered = False
    for _ in range(20):
        d = m.update(np.array([0.0]), epistemic_uncertainty=0.5)
        triggered = triggered or d.ood
    assert triggered


def test_monitor_high_temporal_variance_triggers():
    m = RuntimeMonitor(MonitorConfig(epistemic_threshold=10.0, temporal_threshold=0.05))
    rng = np.random.default_rng(0)
    triggered = False
    for _ in range(20):
        a = np.array([rng.uniform(-1, 1)])
        d = m.update(a, epistemic_uncertainty=0.0)
        triggered = triggered or d.ood
    assert triggered


def test_monitor_safe_stoppability_proxy():
    m = RuntimeMonitor(MonitorConfig(epistemic_threshold=10.0, temporal_threshold=10.0, use_safe_stoppability=True))
    safe_state = np.array([0.0, 0.0, 0.0, 0.0])
    fall_state = np.array([0.0, 0.0, 1.5, 0.0])
    d_safe = m.update(np.array([0.0]), 0.0, physical_state=safe_state)
    d_fall = m.update(np.array([0.0]), 0.0, physical_state=fall_state)
    assert not d_safe.ood
    assert d_fall.ood


def test_monitor_calibration_sets_threshold():
    m = RuntimeMonitor(MonitorConfig())
    eps = np.linspace(0, 1, 100)
    tvar = np.linspace(0, 0.5, 100)
    m.calibrate(eps, tvar, quantile=0.9)
    assert abs(m.cfg.epistemic_threshold - 0.9) < 0.05
    assert abs(m.cfg.temporal_threshold - 0.45) < 0.05


def test_monitor_disabled_never_flags():
    m = RuntimeMonitor(MonitorConfig(enabled=False, epistemic_threshold=0.0))
    d = m.update(np.array([1.0]), epistemic_uncertainty=10.0)
    assert not d.ood
