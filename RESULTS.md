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
| `method` | one of `random`, `lqr`, `mppi`, `td_mpc2_lite`, `dreamer_lite`, `gatsr_full`, `gatsr_no_layered`, `gatsr_no_graph`, `gatsr_no_recovery`, `gatsr_no_monitor`, `gatsr_no_cbf` | |
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

## Reproducing the summary

```bash
# the documented configuration for the tables/figures below
python scripts/benchmark.py --seeds 3 --episodes 10   # ~45 min, 990 episodes
python scripts/make_figures.py
python scripts/verify_claims.py                        # asserts the findings
```

`--seeds 3 --episodes 10` takes ~45 min on one Ryzen-9-9900X core (CPU torch).
To trim to a ~5 min smoke run:

```bash
python scripts/benchmark.py --seeds 2 --episodes 4 --train-steps 800 --max-steps 150
```

## Headline numbers (`--seeds 3 --episodes 10`, n=30 per cell)

Mean over the three OOD levels. Full per-OOD breakdown is in `results/summary.csv`.

| Method | Success | Return | CBF interv./ep | Recovery att./ep | Recovery success | Planning ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0.00 | -62.9 | 0.0 | 0.00 | — | 0.0 |
| mppi (L2 only) | 0.00 | -54.1 | 0.0 | 0.00 | — | 18.1 |
| td_mpc2_lite | 0.00 | -45.1 | 0.0 | 0.00 | — | 18.0 |
| dreamer_lite | 0.00 | -61.4 | 0.0 | 0.00 | — | 0.1 |
| **lqr (= L1)** | **0.88** | **269.7** | 0.0 | 0.00 | — | 0.0 |
| **gatsr_full** | **0.64** | 220.6 | 24.3 | 0.36 | 0.00 | 26.7 |
| gatsr_no_layered | 0.00 | -0.5 | 40.1 | 2.54 | 0.60 | 24.3 |
| gatsr_no_graph | 0.64 | 221.1 | 24.5 | 0.36 | 0.00 | 29.4 |
| gatsr_no_recovery | 0.64 | 220.6 | 23.3 | 0.00 | — | 27.5 |
| gatsr_no_monitor | 0.64 | 220.6 | 24.3 | 0.36 | 0.00 | 27.5 |
| gatsr_no_cbf | 0.79 | 253.0 | 0.0 | 0.21 | 0.00 | 29.9 |

## Findings (what the data actually shows)

1. **The task is solved by the analytic methods, not the learned ones**
   (`fig01`/`fig02`). LQR (the L1 layer) reaches 0.88 success; GATS-R reaches
   0.64; the pure model-based-learning baselines (MPPI, TD-MPC2-lite,
   Dreamer-lite) sit at **0.00** — with only 2000 random transitions of CPU
   training they cannot drive the cart to goals while balancing. This is the
   honest, expected outcome and is exactly why the architecture keeps an
   analytic L1 prior.
2. **Graceful degradation** (`fig01`): GATS-R declines smoothly with OOD
   (0.70 → 0.63 → 0.60); LQR is near-flat (0.90 → 0.90 → 0.83). The learned
   baselines are flat at zero (nothing to degrade).
3. **The layered L1 selector is load-bearing** (`fig06`, `gatsr_no_layered`):
   turning L1 off — so the agent always plans through the under-trained L2 —
   collapses success to **0.00** and return to ~0. This is the single largest
   ablation effect and validates the central architectural choice.
4. **CBF is a safety-vs-performance trade-off, not a free win** (`fig03`,
   `gatsr_no_cbf`): with the filter off, success/return *increase*
   (0.79 / +253 vs 0.64 / +220) and CBF interventions drop to 0 — the policy is
   freer. The filter is conservative; its value is the ~24 interventions/episode
   it makes on the unsafe actions the planner proposes, which matters on the
   G1 (below), not on this benign cart-pole.
5. **Graph / monitor / recovery are largely dormant on this benign task.**
   `no_graph`, `no_monitor`, and `no_recovery` are within noise of full
   (0.64 success, ~220 return); `no_monitor` is bit-identical to full. Once L1
   keeps the pole upright, the monitor rarely flags OOD and recovery rarely
   fires (0.36 attempts/ep, and those that fire do not complete within the
   episode → 0.00 recovery success). The recovery dispatcher's effectiveness is
   demonstrated where it is actually exercised: the Isaac-Lab G1 (91% below) and
   the `no_layered` ablation (0.60 recovery success, where the agent tilts
   constantly and recovery fires ~2.5×/episode).
6. **Planning latency** (`fig05`): GATS-R's MCTS inner loop is the slowest at
   ~27 ms/decision (above the 20 ms G1 control budget — flagged as follow-up);
   MPPI/TD-MPC2 ~18 ms; LQR/random ~0 ms.

`scripts/verify_claims.py` turns findings 1, 3, 4, 5 (and determinism) into
assertions and exits non-zero if any regresses. Numbers vary across machines and
seeds; the *relative ordering* and the *direction of the OOD slope* are the
stable signal.

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
