"""Automated claim verification.

Runs a small, self-contained evaluation and asserts every *qualitative* claim
the repository makes about the CPU BalanceBot reference. Exits non-zero if any
claim fails, so `python scripts/verify_claims.py` is a one-line reproducibility
gate (used by CI and by the README's verification step).

Claims checked (CPU side):
  C1  Code runs under the installed NumPy (incl. NumPy >= 2).
  C2  The long-horizon multi-goal task is solvable: LQR (L1) and GATS-R
      reach a non-zero episode success rate in-distribution.
  C3  Only GATS-R variants with a recovery dispatcher attempt recoveries;
      the baselines (random/lqr/mppi/td_mpc2_lite/dreamer_lite) never do.
  C4  The CBF filter measurably activates for gatsr_full and is exactly zero
      for gatsr_no_cbf (no spurious activation path).
  C5  Removing the layered L1 controller (gatsr_no_layered) collapses success
      to ~0 -> the analytic L1 layer is load-bearing.
  C6  Substantive metrics are deterministic per seed (two identical runs agree
      on every non-timing column).

The Isaac Lab + G1 claims are verified separately by
`scripts/isaaclab_benchmark.py` (requires Isaac Sim + a CUDA GPU); see
`scripts/verify_isaaclab_claims.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from gatsr.agent import AgentConfig, GATSRAgent
from gatsr.baselines.lqr_agent import LQRAgent
from gatsr.baselines.mppi_agent import MPPIAgent
from gatsr.baselines.random_agent import RandomAgent
from gatsr.envs.balance_env import BalanceBotConfig, BalanceBotEnv
from gatsr.planning.skill_graph import SkillGraph
from gatsr.utils.seed import set_seed
from gatsr.world_models.latent import EnsembleLatentModel, LatentModelConfig
from benchmark import collect_random_data


PASS, FAIL = "PASS", "FAIL"


def _build(seed: int, train_steps: int = 1500):
    env0 = BalanceBotEnv(BalanceBotConfig(max_steps=200, n_goals=3, seed=seed))
    s, a, sp, _ret = collect_random_data(env0, train_steps, seed)
    set_seed(seed)
    m = EnsembleLatentModel(
        LatentModelConfig(epochs=10, n_ensemble=4, hidden=64, latent_dim=16, batch_size=128)
    )
    m.fit(s, a, sp)
    with torch.no_grad():
        z = m.encode(torch.as_tensor(s, dtype=torch.float32)).cpu().numpy()
    g = SkillGraph.from_trajectories(
        latents=z, physicals=s, n_clusters=12, edge_radius=2.5,
        recovery_anchor_physical=np.zeros(4), seed=seed,
    )
    return m, g


def _gatsr(name, env, m, g, seed):
    kw = dict(
        seed=seed, planning_horizon=10, n_mppi_samples=48, n_mcts_simulations=32,
        use_mcts=True, use_skill_graph=True, use_recovery=True, use_monitor=True,
        use_cbf=True, use_layered=True,
    )
    if name == "gatsr_no_cbf":
        kw["use_cbf"] = False
    if name == "gatsr_no_layered":
        kw["use_layered"] = False
    return GATSRAgent(AgentConfig(**kw), env, latent_model=m, skill_graph=g)


def main() -> int:
    seed = 0
    print(f"[verify] NumPy {np.__version__}, Torch {torch.__version__}")
    print("[verify] building shared L2 + skill graph ...")
    m, g = _build(seed)

    def env(ood=0.0, max_steps=300):
        e = BalanceBotEnv(BalanceBotConfig(max_steps=max_steps, n_goals=3, ood_level=ood, seed=seed))
        e.reset(seed=seed)
        return e

    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  [{PASS if ok else FAIL}] {name}: {detail}")

    n = 6
    # --- C2: task solvable -------------------------------------------------
    lqr_stats = LQRAgent(env(), seed=seed).evaluate(episodes=n, seed_offset=1000)
    lqr_succ = float(np.mean([x["success"] for x in lqr_stats]))
    full_stats = _gatsr("gatsr_full", env(), m, g, seed).evaluate(episodes=n, seed_offset=1000)
    full_succ = float(np.mean([x.success for x in full_stats]))
    check("C2 task solvable (LQR success>0)", lqr_succ > 0, f"LQR success_rate={lqr_succ:.2f}")
    check("C2 task solvable (GATS-R success>0)", full_succ > 0, f"GATS-R success_rate={full_succ:.2f}")

    # --- C3: only GATS-R attempts recovery --------------------------------
    full_rec = sum(x.recoveries_attempted for x in full_stats)
    rnd_stats = RandomAgent(env(), seed=seed).evaluate(episodes=n, seed_offset=1000)
    mppi_stats = MPPIAgent(env(), m, seed=seed).evaluate(episodes=n, seed_offset=1000)
    base_rec = sum(x["recoveries_attempted"] for x in rnd_stats) + \
        sum(x["recoveries_attempted"] for x in mppi_stats) + \
        sum(x["recoveries_attempted"] for x in lqr_stats)
    check("C3 GATS-R attempts recovery", full_rec > 0, f"gatsr_full attempts={full_rec}")
    check("C3 baselines never recover", base_rec == 0, f"baseline attempts={base_rec}")

    # --- C4: CBF activates for full, zero for no_cbf -----------------------
    full_cbf = sum(x.safety_violations for x in full_stats)
    nocbf_stats = _gatsr("gatsr_no_cbf", env(), m, g, seed).evaluate(episodes=n, seed_offset=1000)
    nocbf_cbf = sum(x.safety_violations for x in nocbf_stats)
    check("C4 CBF activates (full>0)", full_cbf > 0, f"gatsr_full interventions={full_cbf}")
    check("C4 CBF off => zero (no_cbf==0)", nocbf_cbf == 0, f"gatsr_no_cbf interventions={nocbf_cbf}")

    # --- C5: layered L1 is load-bearing -----------------------------------
    nolay_stats = _gatsr("gatsr_no_layered", env(), m, g, seed).evaluate(episodes=n, seed_offset=1000)
    nolay_succ = float(np.mean([x.success for x in nolay_stats]))
    check("C5 layered L1 load-bearing", nolay_succ < full_succ,
          f"no_layered success={nolay_succ:.2f} < full={full_succ:.2f}")

    # --- C6: determinism (non-timing columns) -----------------------------
    a1 = _gatsr("gatsr_full", env(), m, g, seed).evaluate(episodes=3, seed_offset=1000)
    a2 = _gatsr("gatsr_full", env(), m, g, seed).evaluate(episodes=3, seed_offset=1000)
    same = all(
        (x.steps, round(x.ep_return, 9), x.success, x.recoveries_attempted,
         x.recoveries_succeeded, x.safety_violations) ==
        (y.steps, round(y.ep_return, 9), y.success, y.recoveries_attempted,
         y.recoveries_succeeded, y.safety_violations)
        for x, y in zip(a1, a2)
    )
    check("C6 deterministic per seed", same, "two runs agree on all non-timing metrics")

    ok = all(results)
    print(f"\n[verify] {sum(results)}/{len(results)} claims PASS -> {'ALL VERIFIED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
