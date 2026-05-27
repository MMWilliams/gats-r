"""Quick end-to-end demo: train an L2 ensemble, build the skill graph, run a
single GATS-R episode, and save a rollout plot.

Run with:
    python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gatsr.agent import AgentConfig, GATSRAgent
from gatsr.envs.balance_env import BalanceBotConfig, BalanceBotEnv
from gatsr.planning.skill_graph import SkillGraph
from gatsr.utils.seed import set_seed
from gatsr.world_models.latent import EnsembleLatentModel, LatentModelConfig


def main():
    set_seed(0)
    print("[demo] collecting random data for L2 training ...")
    env_collect = BalanceBotEnv(BalanceBotConfig(max_steps=200, n_goals=3, seed=0))
    env_collect.reset(seed=0)
    rng = np.random.default_rng(0)
    s, a, sp = [], [], []
    for _ in range(2000):
        ps = env_collect.physical_state
        act = rng.uniform(-1, 1, size=(1,))
        env_collect.step(act)
        s.append(ps)
        a.append(act)
        sp.append(env_collect.physical_state)
        if env_collect.is_crashed() or env_collect.t >= env_collect.cfg.max_steps - 1:
            env_collect.reset(seed=int(rng.integers(0, 1e9)))
    states, actions, next_states = np.array(s), np.array(a), np.array(sp)

    print("[demo] training L2 ensemble ...")
    model = EnsembleLatentModel(LatentModelConfig(epochs=8, n_ensemble=4, latent_dim=16, hidden=64))
    model.fit(states, actions, next_states)

    print("[demo] building skill graph ...")
    with torch.no_grad():
        z = model.encode(torch.as_tensor(states, dtype=torch.float32)).numpy()
    graph = SkillGraph.from_trajectories(z, states, n_clusters=12, edge_radius=2.5)
    print(f"        graph has {len(graph)} nodes")

    print("[demo] running one GATS-R episode (OOD = 0.5) ...")
    env = BalanceBotEnv(BalanceBotConfig(max_steps=300, n_goals=3, ood_level=0.5, seed=0))
    env.reset(seed=0)
    agent = GATSRAgent(
        AgentConfig(planning_horizon=10, n_mcts_simulations=24, n_mppi_samples=32),
        env=env,
        latent_model=model,
        skill_graph=graph,
    )
    traj = []
    is_recovery = []
    ood_flags = []
    while True:
        ps = env.physical_state
        traj.append(ps.copy())
        a, decision, info = agent.act(ps, env.current_goal(), step=len(traj))
        is_recovery.append(info["recovery_active"])
        ood_flags.append(decision.ood)
        if info["recovery_active"]:
            _, _, done, eninfo = env.recover_step(a)
        else:
            _, _, done, eninfo = env.step(a)
        if done:
            break

    traj = np.array(traj)
    is_recovery = np.array(is_recovery, dtype=bool)
    ood_flags = np.array(ood_flags, dtype=bool)

    figdir = ROOT / "results" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    t = np.arange(len(traj))
    axes[0].plot(t, traj[:, 0], color="tab:blue", label="cart x")
    for g in env.goals:
        axes[0].axhline(g, linestyle="--", alpha=0.3, color="gray")
    axes[0].fill_between(t, -2.4, 2.4, where=is_recovery, color="orange", alpha=0.15, label="recovery active")
    axes[0].fill_between(t, -2.4, 2.4, where=(~is_recovery) & ood_flags, color="red", alpha=0.1, label="OOD flag")
    axes[0].set_ylabel("cart x")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, traj[:, 2], color="tab:red")
    axes[1].axhline(0.7, color="black", linestyle=":", alpha=0.5, label="fall_angle")
    axes[1].axhline(-0.7, color="black", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("pole θ (rad)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(t, traj[:, 3], color="tab:green")
    axes[2].set_ylabel("pole θ̇ (rad/s)")
    axes[2].set_xlabel("step")
    axes[2].grid(alpha=0.3)

    fig.suptitle("GATS-R demo episode (OOD=0.5)")
    fig.tight_layout()
    out = figdir / "fig00_demo_episode.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
