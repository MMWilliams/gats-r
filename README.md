# GATS-R: Graph-Augmented, Layered-World-Model RL with Graph-Indexed Recovery

A reproducible, self-contained reference implementation of the research direction
*"From Symbolic GATS to Robust Robot Learning"* — translated from the
Isaac-Lab/Unitree-G1 thesis to a fast CPU-only toy continuous-control task that
exhibits the same phenomena: falling, recovery, OOD generalization, long-horizon
multi-goal planning, and statistically reportable robustness.

> **Why a toy task?** The full thesis targets Isaac Lab + Unitree G1 (29-DoF
> humanoid), which requires GPU clusters and physical hardware. This repo
> implements every *architectural* idea (layered L1/L2/L3 world model, skill
> graph + continuous MCTS, Sentinel-style monitoring, graph-indexed recovery,
> CBF-internalized training) on a custom `BalanceBot` task (cart-pole + multi-
> goal + perturbations) so all claims can be verified end-to-end on a laptop
> in minutes.

## Quickstart

```bash
# 1. install
pip install -r requirements.txt
pip install -e .

# 2. run the test suite (~20 s)
pytest -q

# 3. run the benchmark (3 seeds x 6 methods x 3 OOD levels; ~5–10 min CPU)
python scripts/benchmark.py --seeds 3 --episodes 20

# 4. generate all figures from the cached results
python scripts/make_figures.py
```

Results land in `results/` (csv tables) and `results/figures/` (png plots).

## Architecture

```
+--------------------------------------------------------------+
|                       GATS-R Agent                           |
|  +--------+   +-----------------+   +---------------------+  |
|  | Skill  |-->| Two-level       |-->| CBF Safety Filter   |  |
|  | Graph  |   | Planner         |   +---------------------+  |
|  +--------+   |  A* over graph  |              |             |
|       ^      |  + MCTS+VPW      |              v             |
|       |      +-----------------+        +-------------+      |
|       |              ^                  | Environment |      |
|       |              |                  +-------------+      |
|       |   +----------+----------+              |             |
|       |   | Layered World Model |              |             |
|       |   |  L1 analytic        |              v             |
|       |   |  L2 ensemble latent |       +-------------+      |
|       |   |  L3 fallback        |       | Monitor:    |      |
|       |   +---------------------+       | ensemble +  |      |
|       |              ^                  | temporal    |      |
|       |              |                  +-------------+      |
|       |              +--ood-->  +----------+    |OOD         |
|       +-------------------------| Recovery |<---+            |
|                                 | dispatch |                 |
|                                 +----------+                 |
+--------------------------------------------------------------+
```

| Layer | File | Idea |
| --- | --- | --- |
| **Env** | `src/gatsr/envs/balance_env.py` | Cart-pole + goal sequence + disturbances + recovery channel |
| **L1** | `src/gatsr/world_models/analytic.py` | Linearized cart-pole around upright |
| **L2** | `src/gatsr/world_models/latent.py` | Ensemble MLP latent dynamics + epistemic head |
| **L3** | `src/gatsr/world_models/fallback.py` | Random-shooting / VLM-stub sub-goal proposer |
| **Layered** | `src/gatsr/world_models/layered.py` | Selects L1 if in-validity, else L2, else L3 |
| **Graph** | `src/gatsr/planning/skill_graph.py` | k-NN landmark graph in latent space (SPTM-style) |
| **MCTS** | `src/gatsr/planning/mcts.py` | Continuous MCTS w/ Voronoi Progressive Widening |
| **MPPI** | `src/gatsr/planning/mppi.py` | Reference MPC baseline |
| **Planner** | `src/gatsr/planning/planner.py` | A\* on skill graph + MCTS within edges |
| **Safety** | `src/gatsr/safety/cbf.py`, `safety/reachability.py` | CBF filter + ROM reachability |
| **Monitor** | `src/gatsr/monitoring/monitor.py` | Ensemble disagreement ∨ temporal consistency |
| **Recovery** | `src/gatsr/recovery/` | Analytic LQR stabilizer keyed by skill graph |
| **Agent** | `src/gatsr/agent.py` | Glues everything into a closed loop |
| **Baselines** | `src/gatsr/baselines/` | TD-MPC2-lite, Dreamer-lite, PPO-lite |

## Mapping from the full thesis

| Thesis concept | Repo realisation |
| --- | --- |
| Isaac Lab + Unitree G1 (29-DoF) | `BalanceBot` (cart-pole + goals + disturbances) |
| TD-MPC2 latent world model + MPPI | `latent.py` + `mppi.py` (`TDMPC2Lite`) |
| DreamerV3 RSSM | `dreamer_lite.py` (RSSM-style GRU + recon) |
| SPTM / World-Model-as-a-Graph | `skill_graph.py` |
| Continuous MCTS / SETS | `mcts.py` (with Voronoi Progressive Widening, Lim 2020) |
| Layered L1/L2/L3 (from GATS) | `analytic.py` + `latent.py` + `fallback.py` + `layered.py` |
| CBF-RL (Yang 2025) | `safety/cbf.py`, applied during training in `agent.py` |
| Sentinel monitor (Agia 2024) | `monitoring/monitor.py` |
| FRASA / FIRM / get-up recovery | `recovery/recovery_policy.py` (LQR + graph-indexed) |
| HumanoidBench / LIBERO-Long | `BalanceBot` multi-goal long-horizon mode |
| FailureBench | OOD perturbation sweep in `benchmark.py` |

## Reproducibility

Everything is deterministic given `--seed`. The benchmark script writes:

- `results/raw.csv` — per-(method, seed, OOD-level, episode) success/return/recovery/safety.
- `results/summary.csv` — mean ± std aggregated to the (method, OOD-level) level.
- `results/figures/*.png` — six publication-style plots (see `make_figures.py`).

Re-running the same command with the same seeds yields bit-identical results.

## Layout

```
robotics_research/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/gatsr/
│   ├── envs/
│   ├── world_models/
│   ├── planning/
│   ├── safety/
│   ├── monitoring/
│   ├── recovery/
│   ├── baselines/
│   ├── utils/
│   └── agent.py
├── scripts/
│   ├── benchmark.py
│   ├── make_figures.py
│   └── demo.py
├── tests/
└── results/  # generated
```

## Isaac Lab port (37-DoF Unitree G1)

The repo also includes a full port of the architecture to **NVIDIA Isaac Lab +
Unitree G1**. This needs:

- Isaac Sim 5.x installed (default path `C:\isaac-sim`).
- The official `isaaclab` conda env on Python 3.11 with the `torch==2.7.0+cu128`
  PyTorch wheel (Blackwell-compatible).
- A CUDA GPU; tested on dual RTX 5090 (the L2 ensemble auto-mirrors onto a
  second GPU when present).

Run a smoke test (loads Isaac Sim, instantiates the G1 task, takes 8 random
steps):

```powershell
pwsh scripts/run_isaaclab.ps1 scripts/isaaclab_smoke.py --num_envs 4 --n_steps 8
```

Run the headline Isaac-Lab benchmark (random / MPPI / GATS-R / GATS-R-no-rec):

```powershell
pwsh scripts/run_isaaclab.ps1 scripts/isaaclab_benchmark.py `
    --num_envs 16 --episodes 4 --max_steps 200 --train_steps 1024
```

The script writes:
- `results/isaaclab_raw.csv` — one row per (method, episode, env)
- `results/isaaclab_summary.csv` — aggregated mean/std per method
- `results/isaaclab_benchmark_report.txt` — human-readable timing log

| Isaac Lab module | File | Mirrors CPU equivalent |
| --- | --- | --- |
| Env wrapper | `src/gatsr/isaaclab/env.py` | `gatsr.envs.balance_env` |
| L2 ensemble (multi-GPU) | `src/gatsr/isaaclab/latent.py` | `gatsr.world_models.latent` |
| Batched MPPI | `src/gatsr/isaaclab/planner.py` | `gatsr.planning.mppi` |
| G1 CBF filter | `src/gatsr/isaaclab/safety.py` | `gatsr.safety.cbf` |
| Sentinel monitor | `src/gatsr/isaaclab/monitor.py` | `gatsr.monitoring.monitor` |
| PD recovery (FRASA placeholder) | `src/gatsr/isaaclab/recovery.py` | `gatsr.recovery.recovery_policy` |
| Agent | `src/gatsr/isaaclab/agent.py` | `gatsr.agent` |

### Hardware notes

When two CUDA devices are visible, `G1EnsembleLatentModel` replicates itself
onto `cuda:1` and splits MPPI rollouts in half between the GPUs for ~2× more
samples per planning iteration. PCIe gen 3 ×1 on the second slot (per
`nvidia-smi`) will be the bottleneck for very small batches; the split is
worth it from `n_samples ≥ 64`.

## Caveats

This is a *concept-validation* implementation. The published bar described in
the thesis (≥10 Isaac Lab tasks, real-G1 hardware, statistically reported
recovery, CoRL/RSS-grade baselines like the full TD-MPC2 / DreamerV3 / FRASA /
FIRM) is not met by a CPU toy task. The contribution here is a clean,
inspectable, end-to-end implementation of the *architectural ideas* with
matching metrics so a research team can lift each module into Isaac Lab with
confidence about the interfaces.
