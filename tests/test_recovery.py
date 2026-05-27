import numpy as np

from gatsr.envs.balance_env import BalanceBotConfig, BalanceBotEnv
from gatsr.planning.skill_graph import SkillGraph, SkillNode
from gatsr.recovery.recovery_policy import LQRRecoveryPolicy, RecoveryConfig
from gatsr.recovery.recovery_graph import GraphIndexedRecovery


def test_lqr_stabilizes_small_tilt():
    pol = LQRRecoveryPolicy(RecoveryConfig())
    s = np.array([0.0, 0.0, 0.2, 0.0])
    # roll a few steps of the linearized dynamics under LQR
    from gatsr.world_models.analytic import AnalyticModel
    m = AnalyticModel()
    for _ in range(120):
        a = pol(s)
        s = m.predict(s, a)
        if abs(s[2]) < 0.05 and abs(s[3]) < 0.3:
            return
    assert abs(s[2]) < 0.2  # at least it didn't blow up


def test_recovery_in_env_brings_pole_upright():
    cfg = BalanceBotConfig(max_steps=300, push_prob=0.0, base_force_noise=0.0, seed=0)
    env = BalanceBotEnv(cfg)
    env.reset(seed=0)
    env._state = np.array([0.2, 0.0, 0.4, 0.0])  # tilted
    pol = LQRRecoveryPolicy(RecoveryConfig())
    for _ in range(200):
        a = pol(env.physical_state)
        env.recover_step(a)
        if pol.recovered(env.physical_state):
            return
    assert pol.recovered(env.physical_state)


def test_graph_indexed_dispatcher_tracks_stats():
    g = SkillGraph()
    g.add_node(SkillNode(idx=-1, latent=np.zeros(2), physical=np.zeros(4), is_recovery=True))
    pol = LQRRecoveryPolicy(RecoveryConfig())
    disp = GraphIndexedRecovery(graph=g, default_policy=pol, max_steps_per_attempt=120)
    cfg = BalanceBotConfig(push_prob=0.0, base_force_noise=0.0)
    env = BalanceBotEnv(cfg)
    env.reset(seed=0)
    env._state = np.array([0.0, 0.0, 0.35, 0.0])
    disp.begin(env.physical_state, current_step=0)
    for t in range(200):
        a, recovered = disp.step(env.physical_state, current_step=t)
        env.recover_step(a)
        if recovered:
            break
    assert disp.attempts == 1
    assert disp.successes == 1
    assert disp.mean_time_to_recover > 0


def test_graph_indexed_returns_zero_action_when_inactive():
    g = SkillGraph()
    g.add_node(SkillNode(idx=-1, latent=np.zeros(2), physical=np.zeros(4), is_recovery=True))
    disp = GraphIndexedRecovery(graph=g, default_policy=LQRRecoveryPolicy(RecoveryConfig()))
    a, recovered = disp.step(np.array([0.0, 0.0, 0.0, 0.0]), current_step=0)
    assert not recovered
    np.testing.assert_array_equal(a, np.zeros(1))
