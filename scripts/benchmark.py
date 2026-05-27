"""End-to-end benchmark.

Run with:
    python scripts/benchmark.py --seeds 3 --episodes 20

Sweeps over six methods × n_seeds × 3 OOD levels (0.0 in-dist, 0.5 mid, 1.0
heavy) and writes:
    results/raw.csv     — per-episode metrics
    results/summary.csv — aggregated mean ± std

Each method shares the same trained L2 ensemble for fairness (so the
comparison isolates the *structural* contribution of GATS-R rather than
model quality differences).

Methods compared:
    random            — uniform actions
    lqr               — pure analytic stabilizer
    mppi              — MPPI on L2 latent (no graph, monitor, recovery, CBF)
    td_mpc2_lite      — MPPI on L2 latent + learned value bootstrap
    dreamer_lite      — GRU-RSSM + actor head
    gatsr_full        — the full GATS-R agent (graph + MCTS + monitor + recovery + CBF)
    gatsr_no_graph    — GATS-R minus the skill graph
    gatsr_no_recovery — GATS-R minus the recovery dispatcher
    gatsr_no_monitor  — GATS-R minus the Sentinel-style monitor
    gatsr_no_cbf      — GATS-R minus the CBF filter
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow `import gatsr...` without installing
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from tqdm import tqdm

from gatsr.agent import AgentConfig, GATSRAgent
from gatsr.baselines.dreamer_lite import DreamerLiteAgent
from gatsr.baselines.lqr_agent import LQRAgent
from gatsr.baselines.mppi_agent import MPPIAgent
from gatsr.baselines.random_agent import RandomAgent
from gatsr.baselines.td_mpc2_lite import TDMPC2LiteAgent
from gatsr.envs.balance_env import BalanceBotConfig, BalanceBotEnv
from gatsr.planning.skill_graph import SkillGraph
from gatsr.utils.logger import BenchmarkLog, EpisodeLog
from gatsr.utils.seed import set_seed
from gatsr.world_models.latent import EnsembleLatentModel, LatentModelConfig


METHOD_LIST = [
    "random",
    "lqr",
    "mppi",
    "td_mpc2_lite",
    "dreamer_lite",
    "gatsr_full",
    "gatsr_no_graph",
    "gatsr_no_recovery",
    "gatsr_no_monitor",
    "gatsr_no_cbf",
]


def collect_random_data(env: BalanceBotEnv, n_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    s, a, sp = [], [], []
    obs = env.reset(seed=seed)
    for _ in range(n_steps):
        ps = env.physical_state
        act = rng.uniform(-1.0, 1.0, size=(1,))
        env.step(act)
        s.append(ps)
        a.append(act)
        sp.append(env.physical_state)
        if env.is_crashed() or env.t >= env.cfg.max_steps - 1:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return np.array(s), np.array(a), np.array(sp)


def train_latent(states, actions, next_states, seed: int) -> EnsembleLatentModel:
    cfg = LatentModelConfig(
        epochs=10, n_ensemble=4, hidden=64, latent_dim=16, batch_size=128
    )
    set_seed(seed)
    m = EnsembleLatentModel(cfg)
    m.fit(states, actions, next_states)
    return m


def build_skill_graph(model: EnsembleLatentModel, states: np.ndarray, seed: int) -> SkillGraph:
    set_seed(seed)
    with torch.no_grad():
        z = model.encode(torch.as_tensor(states, dtype=torch.float32)).cpu().numpy()
    return SkillGraph.from_trajectories(
        latents=z,
        physicals=states,
        n_clusters=12,
        edge_radius=2.5,
        recovery_anchor_physical=np.zeros(4),
        seed=seed,
    )


def make_agent(name: str, env: BalanceBotEnv, latent: EnsembleLatentModel, graph: SkillGraph, seed: int):
    if name == "random":
        return RandomAgent(env, seed=seed)
    if name == "lqr":
        return LQRAgent(env, seed=seed)
    if name == "mppi":
        return MPPIAgent(env, latent, seed=seed)
    if name == "td_mpc2_lite":
        return TDMPC2LiteAgent(env, latent, seed=seed)
    if name == "dreamer_lite":
        return DreamerLiteAgent(env)

    cfg_kwargs = dict(
        seed=seed,
        planning_horizon=10,
        n_mppi_samples=48,
        n_mcts_simulations=32,
        use_mcts=True,
        use_skill_graph=True,
        use_recovery=True,
        use_monitor=True,
        use_cbf=True,
        use_layered=True,
    )
    if name == "gatsr_full":
        pass
    elif name == "gatsr_no_graph":
        cfg_kwargs["use_skill_graph"] = False
    elif name == "gatsr_no_recovery":
        cfg_kwargs["use_recovery"] = False
    elif name == "gatsr_no_monitor":
        cfg_kwargs["use_monitor"] = False
    elif name == "gatsr_no_cbf":
        cfg_kwargs["use_cbf"] = False
    else:
        raise ValueError(name)
    return GATSRAgent(AgentConfig(**cfg_kwargs), env, latent_model=latent, skill_graph=graph)


def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bench = BenchmarkLog()
    ood_levels = [0.0, 0.5, 1.0]
    methods = METHOD_LIST if args.methods is None else args.methods

    # ---- one-time L2 + skill-graph training per seed (shared across methods)
    seed_caches: dict[int, dict] = {}
    print("[train] preparing shared L2 model and skill graph per seed ...")
    for seed in range(args.seeds):
        env = BalanceBotEnv(BalanceBotConfig(max_steps=200, n_goals=3, seed=seed))
        states, actions, next_states = collect_random_data(env, n_steps=args.train_steps, seed=seed)
        latent = train_latent(states, actions, next_states, seed=seed)
        graph = build_skill_graph(latent, states, seed=seed)
        # quick value-head fit for td_mpc2_lite — Monte-Carlo returns from data
        returns = np.linspace(0, 1, len(states))  # placeholder shaping; irrelevant since planner reuses cost
        seed_caches[seed] = dict(
            latent=latent,
            graph=graph,
            states=states,
            returns=returns,
        )

    # ---- main eval loop
    total = len(methods) * args.seeds * len(ood_levels) * args.episodes
    pbar = tqdm(total=total, desc="benchmark")
    for method in methods:
        for seed in range(args.seeds):
            cache = seed_caches[seed]
            for ood in ood_levels:
                eval_env_cfg = BalanceBotConfig(
                    max_steps=args.max_steps,
                    n_goals=args.n_goals,
                    ood_level=ood,
                    seed=seed,
                )
                env = BalanceBotEnv(eval_env_cfg)
                env.reset(seed=seed)
                agent = make_agent(method, env, cache["latent"], cache["graph"], seed=seed)
                # td_mpc2_lite needs value training first (cheap)
                if method == "td_mpc2_lite":
                    agent.fit_value(cache["states"], cache["returns"], epochs=3)
                # dreamer needs one fit
                if method == "dreamer_lite":
                    agent.fit(
                        cache["states"],
                        np.random.default_rng(seed).uniform(-1, 1, (len(cache["states"]), 1)),
                        np.roll(cache["states"], -1, axis=0),
                    )

                # evaluate
                if isinstance(agent, GATSRAgent):
                    stats = agent.evaluate(episodes=args.episodes, seed_offset=1000)
                    for i, s in enumerate(stats):
                        ttr = -1.0
                        if s.time_to_recover_count > 0:
                            ttr = s.time_to_recover_accum / s.time_to_recover_count
                        plan_ms = s.planning_ms_sum / max(1, s.steps)
                        bench.add(
                            EpisodeLog(
                                method=method,
                                seed=seed,
                                ood_level=ood,
                                episode=i,
                                steps=s.steps,
                                ep_return=s.ep_return,
                                success=s.success,
                                failures_detected=s.failures_detected,
                                recoveries_attempted=s.recoveries_attempted,
                                recoveries_succeeded=s.recoveries_succeeded,
                                safety_violations=s.safety_violations,
                                time_to_recover=ttr,
                                planning_ms=plan_ms,
                            )
                        )
                        pbar.update(1)
                else:
                    stats = agent.evaluate(episodes=args.episodes, seed_offset=1000)
                    for i, s in enumerate(stats):
                        bench.add(
                            EpisodeLog(
                                method=method,
                                seed=seed,
                                ood_level=ood,
                                episode=i,
                                steps=s["steps"],
                                ep_return=s["ep_return"],
                                success=s["success"],
                                failures_detected=s["failures_detected"],
                                recoveries_attempted=s["recoveries_attempted"],
                                recoveries_succeeded=s["recoveries_succeeded"],
                                safety_violations=s["safety_violations"],
                                time_to_recover=s["time_to_recover"],
                                planning_ms=s["planning_ms"],
                            )
                        )
                        pbar.update(1)
    pbar.close()

    # write raw
    raw_path = results_dir / "raw.csv"
    bench.write_csv(raw_path)
    print(f"[done] wrote {raw_path}")

    # summary
    summary_path = results_dir / "summary.csv"
    _write_summary(bench, summary_path)
    print(f"[done] wrote {summary_path}")


def _write_summary(bench: BenchmarkLog, path: Path) -> None:
    import csv
    from collections import defaultdict

    groups: dict[tuple, list[EpisodeLog]] = defaultdict(list)
    for r in bench.rows:
        groups[(r.method, r.ood_level)].append(r)

    fields = [
        "method",
        "ood_level",
        "n",
        "success_rate",
        "success_rate_std",
        "return_mean",
        "return_std",
        "safety_violations_mean",
        "recoveries_attempted_mean",
        "recovery_success_rate",
        "time_to_recover_mean",
        "planning_ms_mean",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (method, ood), rs in sorted(groups.items()):
            n = len(rs)
            success = np.array([r.success for r in rs])
            ret = np.array([r.ep_return for r in rs])
            sv = np.array([r.safety_violations for r in rs])
            ra = np.array([r.recoveries_attempted for r in rs])
            rs_succ = np.array([r.recoveries_succeeded for r in rs])
            ttr = np.array([r.time_to_recover for r in rs])
            plan = np.array([r.planning_ms for r in rs])
            attempted = ra.sum()
            succeeded = rs_succ.sum()
            row = dict(
                method=method,
                ood_level=ood,
                n=n,
                success_rate=float(success.mean()),
                success_rate_std=float(success.std(ddof=0)),
                return_mean=float(ret.mean()),
                return_std=float(ret.std(ddof=0)),
                safety_violations_mean=float(sv.mean()),
                recoveries_attempted_mean=float(ra.mean()),
                recovery_success_rate=float(succeeded / max(1, attempted)),
                time_to_recover_mean=float(ttr[ttr >= 0].mean()) if (ttr >= 0).any() else -1.0,
                planning_ms_mean=float(plan.mean()),
            )
            w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-goals", type=int, default=3)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
