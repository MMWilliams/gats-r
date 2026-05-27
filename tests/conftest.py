"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# allow `import gatsr...` without installing
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gatsr.envs.balance_env import BalanceBotConfig, BalanceBotEnv
from gatsr.utils.seed import set_seed
from gatsr.world_models.latent import EnsembleLatentModel, LatentModelConfig


@pytest.fixture(autouse=True)
def _seed_everything():
    set_seed(0)
    yield


@pytest.fixture
def env() -> BalanceBotEnv:
    cfg = BalanceBotConfig(max_steps=120, n_goals=2, seed=0)
    e = BalanceBotEnv(cfg)
    e.reset(seed=0)
    return e


@pytest.fixture
def env_long() -> BalanceBotEnv:
    cfg = BalanceBotConfig(max_steps=300, n_goals=3, seed=0)
    e = BalanceBotEnv(cfg)
    e.reset(seed=0)
    return e


@pytest.fixture
def trained_latent(env_long) -> EnsembleLatentModel:
    """Quick-train an L2 ensemble on a few hundred random transitions."""
    rng = np.random.default_rng(0)
    s, a, sp = [], [], []
    obs = env_long.reset(seed=0)
    for _ in range(600):
        ps = env_long.physical_state
        act = rng.uniform(-1.0, 1.0, size=(1,))
        env_long.step(act)
        s.append(ps)
        a.append(act)
        sp.append(env_long.physical_state)
        if env_long.is_crashed() or env_long.t >= env_long.cfg.max_steps - 1:
            env_long.reset(seed=int(rng.integers(0, 1e9)))
    states = np.array(s)
    acts = np.array(a)
    nexts = np.array(sp)
    model = EnsembleLatentModel(
        LatentModelConfig(epochs=4, n_ensemble=3, hidden=32, latent_dim=8)
    )
    model.fit(states, acts, nexts)
    env_long.reset(seed=0)
    return model
