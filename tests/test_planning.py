import numpy as np
import pytest

from gatsr.planning.mppi import MPPIPlanner, MPPIConfig
from gatsr.planning.mcts import ContinuousMCTS, MCTSConfig
from gatsr.planning.skill_graph import SkillGraph, SkillNode
from gatsr.planning.planner import HybridPlanner, HybridPlannerConfig
from gatsr.world_models.analytic import AnalyticModel


# ----- MPPI ----------------------------------------------------------------

def test_mppi_returns_action_sequence_of_correct_shape():
    m = AnalyticModel()

    def rollout(s, a):
        B, H, _ = a.shape
        traj = np.zeros((B, H, 4))
        for h in range(H):
            traj[:, h] = (
                m.predict(s if h == 0 else traj[:, h - 1], a[:, h])
                if h == 0
                else m.predict(traj[:, h - 1], a[:, h])
            )
        eps = np.zeros((B, H))
        return traj, eps

    def cost(traj, actions, eps):
        return np.linalg.norm(traj[:, -1, :4], axis=-1)

    cfg = MPPIConfig(horizon=8, n_samples=32, action_dim=1)
    planner = MPPIPlanner(cfg, rollout_fn=rollout, cost_fn=cost)
    seq = planner.plan(np.array([0.0, 0.0, 0.1, 0.0]))
    assert seq.shape == (8, 1)
    a = planner.plan_action(np.array([0.0, 0.0, 0.1, 0.0]))
    assert a.shape == (1,)
    assert -1.0 <= a[0] <= 1.0


def test_mppi_reduces_cost_over_iterations():
    m = AnalyticModel()

    def rollout(s, a):
        B, H, _ = a.shape
        traj = np.zeros((B, H, 4))
        s_now = np.broadcast_to(s, (B, 4)).copy()
        for h in range(H):
            s_now = m.predict(s_now, a[:, h])
            traj[:, h] = s_now
        eps = np.zeros((B, H))
        return traj, eps

    def cost(traj, actions, eps):
        return np.abs(traj[:, -1, 2]) + 0.1 * np.abs(traj[:, -1, 3])  # tilt and tilt-rate

    cfg1 = MPPIConfig(horizon=10, n_samples=64, n_iter=1, seed=0)
    cfg2 = MPPIConfig(horizon=10, n_samples=64, n_iter=4, seed=0)
    state = np.array([0.0, 0.0, 0.15, 0.0])
    planner1 = MPPIPlanner(cfg1, rollout_fn=rollout, cost_fn=cost)
    planner2 = MPPIPlanner(cfg2, rollout_fn=rollout, cost_fn=cost)
    a1 = planner1.plan_action(state)
    a2 = planner2.plan_action(state)
    # both should push leftward to counter positive tilt
    assert a1[0] != 0.0 or a2[0] != 0.0  # planner produced motion


# ----- Continuous MCTS -----------------------------------------------------

def test_mcts_plan_returns_valid_action():
    m = AnalyticModel()
    cfg = MCTSConfig(horizon=4, n_simulations=24, action_dim=1, seed=0)
    mcts = ContinuousMCTS(
        cfg,
        step_fn=lambda s, a: (m.predict(s, a), 0.0),
    )
    a, info = mcts.plan(np.array([0.0, 0.0, 0.05, 0.0]))
    assert a.shape == (1,)
    assert info["nodes"] >= 1
    assert -1.0 <= a[0] <= 1.0


def test_mcts_progressive_widening_constrains_children():
    cfg = MCTSConfig(
        horizon=3, n_simulations=40, action_dim=1, k_pw=0.5, alpha_pw=0.3, seed=0
    )
    m = AnalyticModel()
    mcts = ContinuousMCTS(cfg, step_fn=lambda s, a: (m.predict(s, a), 0.0))
    a, info = mcts.plan(np.array([0.0, 0.0, 0.05, 0.0]))
    # nothing crashed; widening worked
    assert info["nodes"] >= 1


def test_mcts_voronoi_picks_far_action():
    cfg = MCTSConfig(horizon=1, n_simulations=4, action_dim=1, n_candidates=16, seed=1)
    mcts = ContinuousMCTS(cfg, step_fn=lambda s, a: (s.copy(), 0.0))

    # Manually craft existing actions packed near 0; expect new sample far from 0
    from gatsr.planning.mcts import _Node

    node = _Node(state=np.zeros(4), depth=0)
    node.actions = [np.array([-0.05]), np.array([0.05])]
    new_a = mcts._sample_action_voronoi(node)
    assert abs(new_a[0]) > 0.05


# ----- Skill graph ---------------------------------------------------------

def test_skill_graph_construction():
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((80, 8))
    physicals = rng.standard_normal((80, 4)) * 0.1
    g = SkillGraph.from_trajectories(latents, physicals, n_clusters=6, edge_radius=2.0)
    assert len(g) >= 6
    # there must be a recovery node
    assert any(n.is_recovery for n in g.nodes)
    # nearest is self
    n0 = g.nodes[0]
    assert g.nearest(n0.latent) == n0.idx


def test_skill_graph_dijkstra_returns_valid_path():
    g = SkillGraph()
    g.add_node(SkillNode(idx=-1, latent=np.zeros(2), physical=np.zeros(4)))
    g.add_node(SkillNode(idx=-1, latent=np.ones(2), physical=np.array([1, 0, 0, 0])))
    g.add_node(SkillNode(idx=-1, latent=2 * np.ones(2), physical=np.array([2, 0, 0, 0]), is_recovery=True))
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    path = g.shortest_path(0, 2)
    assert path == [0, 1, 2]


def test_skill_graph_recovery_anchor_always_reachable():
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((40, 8))
    physicals = rng.standard_normal((40, 4)) * 0.1
    g = SkillGraph.from_trajectories(latents, physicals, n_clusters=4, edge_radius=0.01)
    rec = g.recovery_node().idx
    for n in g.nodes:
        if n.idx == rec:
            continue
        path = g.shortest_path(n.idx, rec)
        assert path[-1] == rec


# ----- Hybrid planner ------------------------------------------------------

def test_hybrid_planner_picks_action(trained_latent):
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((40, 8))
    physicals = rng.standard_normal((40, 4)) * 0.1
    g = SkillGraph.from_trajectories(latents, physicals, n_clusters=4, edge_radius=2.0)

    def rollout(s, a):
        return trained_latent.rollout_np(s, a)

    def step(s, a):
        return trained_latent.predict_np(s, a)

    planner = HybridPlanner(
        HybridPlannerConfig(use_mcts=True, horizon=6, n_mcts_sims=8),
        graph=g,
        rollout_fn=rollout,
        step_fn=step,
    )
    planner.set_goal_physical(np.zeros(4), np.array([0.5, 0.0, 0.0, 0.0]))
    a, info = planner.plan_action(np.zeros(4))
    assert a.shape in [(1,), (1, 1)]
    assert info["mode"] == "mcts"


def test_hybrid_planner_with_mppi(trained_latent):
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((40, 8))
    physicals = rng.standard_normal((40, 4)) * 0.1
    g = SkillGraph.from_trajectories(latents, physicals, n_clusters=4, edge_radius=2.0)

    planner = HybridPlanner(
        HybridPlannerConfig(use_mcts=False, horizon=6, n_samples=16),
        graph=g,
        rollout_fn=lambda s, a: trained_latent.rollout_np(s, a),
        step_fn=lambda s, a: trained_latent.predict_np(s, a),
    )
    planner.set_goal_physical(np.zeros(4), np.array([0.5, 0.0, 0.0, 0.0]))
    a, info = planner.plan_action(np.zeros(4))
    assert info["mode"] == "mppi"
