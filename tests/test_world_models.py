import numpy as np
import pytest

from gatsr.world_models.analytic import AnalyticModel
from gatsr.world_models.fallback import FallbackProposer
from gatsr.world_models.layered import LayeredWorldModel, Layer
from gatsr.world_models.latent import EnsembleLatentModel, LatentModelConfig


# ----- L1 ------------------------------------------------------------------

def test_analytic_predict_shape():
    m = AnalyticModel()
    s = np.zeros(4)
    sp = m.predict(s, np.array([0.5]))
    assert sp.shape == (4,)


def test_analytic_validity_decreases_with_tilt():
    m = AnalyticModel()
    v_upright = m.validity(np.array([0.0, 0.0, 0.0, 0.0]))
    v_tilted = m.validity(np.array([0.0, 0.0, 0.5, 0.0]))
    assert v_upright > v_tilted
    assert 0.0 <= v_tilted <= v_upright <= 1.0


def test_analytic_rollout_batch():
    m = AnalyticModel()
    s = np.zeros(4)
    actions = np.zeros((3, 5, 1))
    traj = m.rollout(s, actions)
    assert traj.shape == (3, 5, 4)


# ----- L2 ------------------------------------------------------------------

def test_latent_model_fits_and_uncertainty_is_nonneg():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((400, 4))
    a = rng.uniform(-1, 1, size=(400, 1))
    sp = s + 0.1 * np.concatenate([a, a, a, a], axis=-1)
    model = EnsembleLatentModel(
        LatentModelConfig(epochs=3, n_ensemble=3, hidden=32, latent_dim=8)
    )
    info = model.fit(s, a, sp)
    assert np.isfinite(info["final_loss"])
    sp_pred, eps = model.predict_np(s[:1], a[:1])
    assert sp_pred.shape == (1, 4)
    assert eps >= 0.0


def test_latent_rollout_shape():
    model = EnsembleLatentModel(
        LatentModelConfig(epochs=1, n_ensemble=2, hidden=16, latent_dim=4)
    )
    s = np.zeros(4)
    actions = np.zeros((4, 5, 1))
    traj, eps = model.rollout_np(s, actions)
    assert traj.shape == (4, 5, 4)
    assert eps.shape == (4, 5)


# ----- L3 ------------------------------------------------------------------

def test_fallback_returns_n_proposals():
    f = FallbackProposer(n_proposals=4, rng=np.random.default_rng(0))
    props = f.propose(np.array([0.1, 0.0, 0.0, 0.0]), goal_x=1.5)
    assert len(props) == 4
    assert all(p.target_state.shape == (4,) for p in props)
    # at least one is the safe "regain upright" anchor
    assert any("upright" in p.description for p in props)


# ----- Layered orchestrator ------------------------------------------------

def test_layered_uses_l1_when_in_validity(trained_latent):
    layered = LayeredWorldModel(
        analytic=AnalyticModel(),
        latent=trained_latent,
        fallback=FallbackProposer(),
        l1_validity_threshold=0.5,
    )
    mc = layered.predict(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0]))
    assert mc.layer == Layer.L1


def test_layered_falls_back_when_out_of_validity(trained_latent):
    layered = LayeredWorldModel(
        analytic=AnalyticModel(),
        latent=trained_latent,
        fallback=FallbackProposer(),
        l1_validity_threshold=0.95,  # very strict, forces L2/L3
        l2_epistemic_threshold=-1.0,  # impossible threshold, forces L3
    )
    mc = layered.predict(np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0]))
    assert mc.layer == Layer.L3
    assert mc.fallback_proposals is not None and len(mc.fallback_proposals) > 0


def test_layered_traj_returns_three_arrays(trained_latent):
    layered = LayeredWorldModel(
        analytic=AnalyticModel(), latent=trained_latent, fallback=FallbackProposer()
    )
    traj, eps, layers = layered.predict_traj(
        np.array([0.0, 0.0, 0.05, 0.0]), np.zeros((8, 1))
    )
    assert traj.shape == (8, 4)
    assert eps.shape == (8,)
    assert layers.shape == (8,)
    assert traj.dtype == np.float64
