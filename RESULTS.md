# Results & metrics

This file documents *which* metrics the benchmark produces and *how to read them*.

## Files

| File | Produced by | Contents |
| --- | --- | --- |
| `results/raw.csv` | `scripts/benchmark.py` | one row per (method, seed, OOD-level, episode) |
| `results/summary.csv` | `scripts/benchmark.py` | mean ± std aggregated to (method, OOD-level) |
| `results/figures/fig01_success_vs_ood.png` | `scripts/make_figures.py` | Robustness curve |
| `results/figures/fig02_return_vs_ood.png` | `scripts/make_figures.py` | Return vs OOD |
| `results/figures/fig03_safety_violations.png` | `scripts/make_figures.py` | CBF activations by method × OOD |
| `results/figures/fig04_recovery.png` | `scripts/make_figures.py` | Recovery success rate + time to recover |
| `results/figures/fig05_planning_latency.png` | `scripts/make_figures.py` | Per-decision compute (with 20 ms G1 control-loop reference line) |
| `results/figures/fig06_ablation.png` | `scripts/make_figures.py` | GATS-R minus each component |
| `results/figures/fig00_demo_episode.png` | `scripts/demo.py` | Single rollout, OOD = 0.5 |

## Metrics

| Column | Meaning | Notes |
| --- | --- | --- |
| `method` | one of `random`, `lqr`, `mppi`, `td_mpc2_lite`, `dreamer_lite`, `gatsr_full`, `gatsr_no_graph`, `gatsr_no_recovery`, `gatsr_no_monitor`, `gatsr_no_cbf` | |
| `seed` | random seed for env + model + planner | |
| `ood_level` | `0.0` (in-dist) / `0.5` (mid) / `1.0` (heavy) | scales push prob, push strength, dynamics jitter, friction noise |
| `episode` | episode index within (method, seed, ood_level) | |
| `steps` | env steps before termination | |
| `ep_return` | total reward in the episode | |
| `success` | 1 iff env terminated with `terminated="success"` (all goals reached) | |
| `failures_detected` | monitor's # OOD flags | only > 0 for GATS-R variants with monitor enabled |
| `recoveries_attempted` | # times the dispatcher entered recovery mode | |
| `recoveries_succeeded` | # times recovery ended in a "recovered" state | recovery success rate = succeeded / attempted |
| `safety_violations` | # times the CBF intervened on a proposed action | proxy for "policy wanted to do something unsafe" |
| `time_to_recover` | mean env-steps between recovery start and end (per episode) | -1 if no recovery occurred |
| `planning_ms` | mean wall-clock ms per decision | for comparison vs. 20 ms G1 control-loop reference |

## Reproducing the published-style summary

```bash
# the default seeds × episodes recommended for a CoRL-style summary table
python scripts/benchmark.py --seeds 5 --episodes 20
python scripts/make_figures.py
```

This takes ~10–20 min on a modern laptop CPU. To trim further:

```bash
python scripts/benchmark.py --seeds 2 --episodes 4 --train-steps 800 --max-steps 150
```

## Expected qualitative findings

1. **Robustness curve** (`fig01`): GATS-R degrades the most gracefully as OOD level rises; baselines like Random/MPPI drop sharply.
2. **Return** (`fig02`): GATS-R ≥ TD-MPC2-lite ≥ MPPI ≥ LQR ≥ Dreamer-lite ≥ Random on this toy task.
3. **Safety** (`fig03`): CBF activations are highest under OOD (the policy *wants* to do unsafe things more often); `gatsr_no_cbf` records zero violations because the filter is off — but it *crashes* more (visible in success / return drop).
4. **Recovery** (`fig04`): GATS-R variants attempt recoveries and succeed at a high rate; baselines without a recovery layer never attempt recoveries (`-1` reported for time-to-recover).
5. **Planning latency** (`fig05`): GATS-R (with MCTS) is the slowest at ~20 ms; MPPI ~10 ms; LQR ~0 ms.
6. **Ablation** (`fig06`): removing the *recovery* dispatcher hurts the most under OOD; removing the *skill graph* hurts long-horizon success; removing the *monitor* allows safety violations to stack up; removing *CBF* shifts costs from "CBF intervention" to "crashes / lost reward."

Numbers will vary across machines and seeds; the *relative ordering* and the *direction of the OOD-curve slope* are the stable, claim-supporting signal — exactly what the thesis Section H predicts.

## Isaac Lab + Unitree G1 results

Produced by `scripts/isaaclab_benchmark.py` on a dual-RTX-5090 host, Isaac Sim
5.1, `Isaac-Velocity-Rough-G1-v0`, 16 envs × 3 episodes × 150 steps, L2 trained
on 512 random transitions (deliberately under-trained for a fast smoke run).

| Method | Return | CBF interventions/ep | Recovery attempts/ep | Recovery success | Time-to-recover (steps) | Planning ms |
| --- | --- | --- | --- | --- | --- | --- |
| random | -4.82 | 0 | 0 | — | — | 0.0 |
| mppi | -3.49 | 0 | 0 | — | — | 4.9 |
| gatsr_no_rec | -3.49 | 0 | 0 | — | — | 4.9 |
| **gatsr_full** | -3.99 | **16.2** | **1.42** | **91.2%** | **~14** | **5.0** |

Reading the table:

1. **The safety/recovery machinery measurably activates on real G1 physics.**
   Only `gatsr_full` records CBF interventions (~16/episode) and recovery
   attempts (1.42/episode at **91% success**). `mppi` and `gatsr_no_rec` are
   bit-identical because with those components off the agent reduces to plain
   MPPI on the same latent model.
2. **Planning fits the control budget.** MPPI-in-latent costs ~5 ms/decision —
   well under the 20 ms G1 control-loop period the thesis flags.
3. **Returns are close and `success_rate` is 0 for all** because the L2 world
   model is trained on only 512 random transitions; nobody survives 150 steps
   of rough terrain with a controller that weak. This is the honest expected
   outcome of a *smoke* run — real training is GPU-hours, not seconds. The
   point this benchmark proves is that the **full closed loop runs on Isaac
   Lab + G1 and the components do what they claim**, not that an under-trained
   controller walks rough terrain.

On flat terrain (`Isaac-Velocity-Flat-G1-v0`) in short episodes nothing falls,
so CBF/recovery never fire and the three MPPI-based methods are identical —
which is why the differentiating benchmark uses rough terrain.

Reproduce:

```powershell
pwsh scripts/run_isaaclab.ps1 scripts/isaaclab_benchmark.py `
    --task Isaac-Velocity-Rough-G1-v0 --num_envs 16 --episodes 3 `
    --max_steps 150 --train_steps 512 `
    --methods random mppi gatsr_no_rec gatsr_full
```
