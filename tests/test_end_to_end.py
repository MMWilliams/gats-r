"""End-to-end smoke tests for the full GATS-R agent + baselines."""

import numpy as np
import pytest

from gatsr.envs.balance_env import BalanceBotConfig, BalanceBotEnv
from gatsr.agent import GATSRAgent, AgentConfig
from gatsr.baselines.random_agent import RandomAgent
from gatsr.baselines.lqr_agent import LQRAgent
from gatsr.baselines.mppi_agent import MPPIAgent
from gatsr.baselines.dreamer_lite import DreamerLiteAgent
from gatsr.planning.skill_graph import SkillGraph


def _build_agent(env, latent):
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((40, latent.cfg.latent_dim))
    physicals = rng.standard_normal((40, 4)) * 0.1
    g = SkillGraph.from_trajectories(latents, physicals, n_clusters=4, edge_radius=2.0)
    cfg = AgentConfig(
        seed=0,
        planning_horizon=6,
        n_mppi_samples=24,
        n_mcts_simulations=12,
        use_mcts=True,
    )
    return GATSRAgent(cfg, env, latent_model=latent, skill_graph=g)


def test_gatsr_agent_runs_one_episode(trained_latent):
    env = BalanceBotEnv(BalanceBotConfig(max_steps=80, n_goals=2, push_prob=0.0))
    env.reset(seed=0)
    agent = _build_agent(env, trained_latent)
    stats = agent.evaluate(episodes=1)
    assert len(stats) == 1
    assert stats[0].steps > 0


def test_gatsr_ablation_no_recovery_still_runs(trained_latent):
    env = BalanceBotEnv(BalanceBotConfig(max_steps=80, n_goals=2, push_prob=0.0))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    g = SkillGraph.from_trajectories(
        rng.standard_normal((40, trained_latent.cfg.latent_dim)),
        rng.standard_normal((40, 4)) * 0.1,
        n_clusters=4,
        edge_radius=2.0,
    )
    cfg = AgentConfig(
        seed=0,
        planning_horizon=6,
        n_mppi_samples=16,
        n_mcts_simulations=8,
        use_recovery=False,
        use_monitor=False,
        use_cbf=False,
    )
    agent = GATSRAgent(cfg, env, latent_model=trained_latent, skill_graph=g)
    stats = agent.evaluate(episodes=1)
    assert len(stats) == 1


def test_baselines_run(trained_latent):
    cfg = BalanceBotConfig(max_steps=80, n_goals=2, push_prob=0.0)
    env = BalanceBotEnv(cfg)
    env.reset(seed=0)
    for AgentCls in [RandomAgent, LQRAgent]:
        agent = AgentCls(env=env, seed=0)
        stats = agent.evaluate(episodes=1)
        assert len(stats) == 1
        assert stats[0]["steps"] > 0
    # MPPI requires a shared latent
    env2 = BalanceBotEnv(cfg)
    env2.reset(seed=0)
    agent = MPPIAgent(env=env2, latent_model=trained_latent, seed=0)
    stats = agent.evaluate(episodes=1)
    assert stats[0]["steps"] > 0


def test_dreamer_lite_fit_and_eval(env):
    rng = np.random.default_rng(0)
    s = rng.standard_normal((200, 4))
    a = rng.uniform(-1, 1, size=(200, 1))
    sp = s + 0.05 * np.concatenate([a, a, a, a], axis=-1)
    dl = DreamerLiteAgent(env=env)
    info = dl.fit(s, a, sp)
    assert np.isfinite(info["final_loss"])
    stats = dl.evaluate(episodes=1)
    assert stats[0]["steps"] > 0


def test_gatsr_recovery_triggers_on_forced_fall(trained_latent):
    """Force the env into a 'fallen' state and confirm recovery activates."""
    cfg = BalanceBotConfig(max_steps=120, n_goals=2, push_prob=0.0, base_force_noise=0.0)
    env = BalanceBotEnv(cfg)
    env.reset(seed=0)
    agent = _build_agent(env, trained_latent)
    # forcibly tilt past fall_angle so the env reports `is_fallen()`
    env._state = np.array([0.0, 0.0, 0.8, 0.0])
    assert env.is_fallen()
    # one step — the agent must activate recovery on a fallen state
    a, decision, info = agent.act(env.physical_state, env.current_goal(), step=0)
    assert info["recovery_active"]
