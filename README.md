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
# 1. install (works under NumPy 1.x and 2.x)
pip install -r requirements.txt
pip install -e .

# 2. run the test suite (59 tests, ~4 s)
pytest -q

# 3. verify every qualitative claim programmatically (~2 min CPU)
python scripts/verify_claims.py

# 4. run the full benchmark (10 methods x 3 seeds x 3 OOD levels x 10 episodes
#    = 990 episodes; ~45 min on a Ryzen 9 9900X CPU core)
python scripts/benchmark.py --seeds 3 --episodes 10
#    ...or a fast smoke version (~5 min):
python scripts/benchmark.py --seeds 2 --episodes 4 --train-steps 800 --max-steps 150

# 5. generate all figures from the results
python scripts/make_figures.py
```

Results land in `results/` (csv tables) and `results/figures/` (png plots). A
reference snapshot of both is committed, so `git diff results/` after a re-run
shows how your machine compares to the reference.

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
| **Agent** | `src/gatsr/agent.py` | Glues everything into a closed loop; uses L1 control when valid, else the L2 planner |
| **Baselines** | `src/gatsr/baselines/` | Random, LQR (= L1), MPPI, TD-MPC2-lite, Dreamer-lite |

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

The benchmark script writes:

- `results/raw.csv` — per-(method, seed, OOD-level, episode) success/return/recovery/safety.
- `results/summary.csv` — mean ± std aggregated to the (method, OOD-level) level.
- `results/figures/*.png` — six plots (see `make_figures.py`).

**Determinism.** On a *given machine*, re-running the same command with the same
seeds reproduces every substantive metric bit-identically (the only column that
varies is `planning_ms`, which is wall-clock timing). Verified by
`scripts/verify_claims.py` (check C6). Across different machines or PyTorch
builds, floating-point results can differ slightly because CPU kernels are not
bit-portable; the *relative ordering* and the *direction of the OOD slope* are
the stable, claim-supporting signal.

**Headline CPU result** (`--seeds 3 --episodes 10`, n=30 episodes per cell;
mean over the three OOD levels):

| Method | Success | Return | CBF interv./ep | Planning ms |
| --- | ---: | ---: | ---: | ---: |
| random | 0.00 | -62.9 | 0.0 | 0.0 |
| MPPI (L2 only) | 0.00 | -54.1 | 0.0 | 18.1 |
| TD-MPC2-lite | 0.00 | -45.1 | 0.0 | 18.0 |
| Dreamer-lite | 0.00 | -61.4 | 0.0 | 0.1 |
| **LQR (= L1)** | **0.88** | **269.7** | 0.0 | 0.0 |
| **GATS-R (full)** | **0.64** | 220.6 | 24.3 | 26.7 |

On this *linearizable* toy task the analytic L1 controller (LQR) is the strongest
single method; GATS-R inherits L1's performance through its layered selector and
adds the monitor/recovery machinery (which is mostly dormant here but
measurably active on the G1 — see below). The pure model-based-learning
baselines (MPPI / TD-MPC2-lite / Dreamer-lite) do **not** solve the task at this
minimal CPU training budget. See `RESULTS.md` for the full per-OOD tables,
ablations, and figures.

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
