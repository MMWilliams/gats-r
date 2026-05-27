import numpy as np
import pytest

from gatsr.envs.balance_env import BalanceBotConfig, BalanceBotEnv


def test_reset_returns_observation_of_correct_shape(env):
    obs = env.reset(seed=0)
    assert obs.shape == (BalanceBotEnv.state_dim,)
    assert env.physical_state.shape == (4,)


def test_step_returns_finite_values(env):
    for _ in range(20):
        obs, r, done, info = env.step(np.array([0.0]))
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
        if done:
            break


def test_action_clipped_inside_step(env):
    obs1, *_ = env.step(np.array([100.0]))
    env.reset(seed=0)
    obs2, *_ = env.step(np.array([1.0]))
    np.testing.assert_allclose(obs1[:4], obs2[:4], atol=1e-6)


def test_deterministic_under_same_seed():
    cfg = BalanceBotConfig(max_steps=50, n_goals=2, seed=42)
    e1 = BalanceBotEnv(cfg)
    e1.reset(seed=42)
    e2 = BalanceBotEnv(cfg)
    e2.reset(seed=42)
    traj1, traj2 = [], []
    for _ in range(40):
        a = np.array([0.1])
        traj1.append(e1.step(a)[0])
        traj2.append(e2.step(a)[0])
    np.testing.assert_array_equal(np.array(traj1), np.array(traj2))


def test_crash_terminates_episode(env):
    # force the pole way past crash_angle
    env._state = np.array([0.0, 0.0, 2.0, 0.0])
    obs, r, done, info = env.step(np.array([0.0]))
    assert done
    assert info["terminated"] == "crash"
    assert r <= 0  # crash penalty


def test_goal_reached_increments(env):
    env._state = np.array([env.current_goal(), 0.0, 0.0, 0.0])
    g0 = env.goal_idx
    obs, r, done, info = env.step(np.array([0.0]))
    assert env.goal_idx == g0 + 1
    assert r > 10  # got the goal reward


def test_ood_level_increases_disturbances():
    cfg_low = BalanceBotConfig(max_steps=200, push_prob=0.5, ood_level=0.0, seed=1)
    cfg_high = BalanceBotConfig(max_steps=200, push_prob=0.5, ood_level=1.0, seed=1)
    e_low = BalanceBotEnv(cfg_low)
    e_low.reset(seed=1)
    e_high = BalanceBotEnv(cfg_high)
    e_high.reset(seed=1)
    pushes_low, pushes_high = 0.0, 0.0
    for _ in range(150):
        _, _, d1, info_l = e_low.step(np.array([0.0]))
        _, _, d2, info_h = e_high.step(np.array([0.0]))
        pushes_low += abs(info_l["push"])
        pushes_high += abs(info_h["push"])
        if d1 and d2:
            break
    assert pushes_high >= pushes_low


def test_clone_is_independent(env):
    for _ in range(5):
        env.step(np.array([0.2]))
    snapshot = env.physical_state.copy()
    clone = env.clone()
    for _ in range(5):
        clone.step(np.array([-0.5]))
    np.testing.assert_array_equal(env.physical_state, snapshot)


def test_recover_step_disables_pushes(env):
    env.cfg = BalanceBotConfig(push_prob=1.0, push_strength=10.0, seed=0)
    env.reset(seed=0)
    # In normal step a guaranteed push fires; in recover_step it must not.
    pushes_recover = 0.0
    for _ in range(30):
        _, _, d, info = env.recover_step(np.array([0.0]))
        pushes_recover += abs(info["push"])
        if d:
            break
    assert pushes_recover == 0.0
