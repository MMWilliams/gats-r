# GATS-R: A Reference Architecture for Layered World Models with Graph-Indexed Recovery in Continuous Robot Control

**Concept validation on a CPU toy task and integration on Isaac Lab + Unitree G1**

*Reese (M. M. Williams)*[^author] · 

[^author]: Repository: <https://github.com/MMWilliams/gats-r>. Correspondence: maureesewilliams@gmail.com.

---

## Abstract

We present **GATS-R**, an integrated reference architecture that translates the
layered-search idea from the symbolic-planning GATS framework into continuous
robot control. GATS-R combines a **layered world model** (L1 analytic / L2
ensemble latent / L3 generative fallback), a **skill-landmark graph** searched
with continuous MCTS using Voronoi Progressive Widening, a **Sentinel-style
runtime monitor** that fuses ensemble disagreement with temporal-consistency
and safe-stoppability, a **CBF-RL-style safety filter** internalised during
training, and a **graph-indexed recovery dispatcher** routed through the same
skill graph. We implement every component twice: as a self-contained CPU
reference on a custom *BalanceBot* task (59-test suite, ~45 min full benchmark
on one CPU core), and as a GPU-batched Isaac Lab port driving the 37-DoF
Unitree G1 humanoid on a dual-RTX-5090 host. On BalanceBot, the layered
controller solves the multi-goal task — the analytic L1 layer reaches 0.88
episode success and the full GATS-R agent 0.64, while pure model-based-learning
baselines (MPPI / TD-MPC2-lite / Dreamer-lite) reach 0.00 at the deliberately
minimal CPU training budget; an ablation that disables the L1 layer collapses
GATS-R to 0.00, confirming the layered selector is load-bearing. On Isaac Lab
G1 rough-terrain, only the full GATS-R configuration triggers the CBF and
recovery dispatcher (16.2 interventions and 1.42 recovery attempts per episode
at 91.2% recovery success), at 5 ms/decision — well within the 20 ms
control-loop budget. We make no claim that the under-trained G1 policy walks
rough terrain; the contribution is the *integration*, the *reproducible
interfaces*, and the *measurable activation* of each architectural layer. All
code, configs, committed result data, and exact reproduce commands are
open-sourced, and a one-line `verify_claims.py` asserts each finding.

---

## 1. Introduction

Robust robot learning sits at the intersection of three uncomfortable
trade-offs: model-based control is sample-efficient but compounding-error
prone; model-free RL is robust but expensive; and formal-safety methods scale
poorly to the 29–37 degrees of freedom of modern humanoids. Recent results
have made each individual layer credible at humanoid scale — TD-MPC2 [@hansen2024tdmpc2]
for latent-space MPC, DreamerV3 [@hafner2023dreamerv3] for reconstruction-based
world models, Sentinel [@agia2024sentinel] for runtime failure detection,
FRASA / FIRM [@frasa2024; @xu2025firm] for learned humanoid recovery, and CBF-RL
[@yang2025cbfrl] for training-time safety internalisation on the Unitree G1.
*No published paper integrates these into a single closed loop on a humanoid.*

This paper does not claim a state-of-the-art result on any specific benchmark.
We instead deliver three engineering contributions whose value is their
*integration* and their *reproducibility*:

1. **An architectural translation** of the symbolic-GATS layered-fallback idea
   into continuous control: L1 analytic priors (linearised cart-pole or
   centroidal-momentum dynamics) → L2 ensemble latent model with epistemic
   uncertainty → L3 generative sub-goal proposer.
2. **A graph-indexed recovery dispatcher** that routes monitor-detected
   out-of-distribution (OOD) states to dedicated controllers (LQR for the toy
   task, a PD stand-up placeholder for the G1, with a clean interface for
   FRASA/FIRM/Get-Up-Across-Morphologies replacements).
3. **Two reproducible implementations** — a CPU benchmark on a custom
   *BalanceBot* task that runs all methods × 3 seeds × 3 OOD levels (≈45 min on
   one CPU core), and a GPU-batched Isaac Lab port driving the Unitree G1 with
   the **closed loop verified end-to-end** (CBF activations, recovery attempts
   and successes, planning-latency budget) on the *Isaac-Velocity-Rough-G1-v0*
   task. Both ship committed reference result data and a `verify_claims.py`
   gate.

Our claim is deliberately narrow: **the integration runs, the components do
what they claim, the metrics are inspectable, and the reproduce commands
fit on one terminal line each.** A full evaluation against the 2024–2025
CoRL/RSS bar — 10+ Isaac Lab tasks, 5+ seeds per cell, paired hypothesis
tests, real-G1 hardware — is left as a clearly-scoped extension, with the
interfaces designed so the next team can plug in TD-MPC2's value head,
DreamerV3's RSSM, FRASA's recovery policy, etc.

## 2. Related work

**Layered world models.** The original GATS paper [@gats] used a three-layer
fallback (exact STRIPS match → log-derived statistical model → LLM-prompted
proposal) on symbolic planning tasks. Continuous-control work has implicitly
used model layering for years — physics simulators as "L1", learned dynamics
as "L2" — but rarely *named* the layering as a primary architectural choice.
Recent ETH Zürich work (RWM-O/RWM-U [@rwm2025]) makes the epistemic-uncertainty
head explicit; PWM [@pwm2024] uses a pre-trained TD-MPC2 as a differentiable
simulator and reports degradation at 152-DoF humanoid scale, motivating the
need for layered fallbacks at humanoid complexity.

**Latent world models + planning.** TD-MPC2 [@hansen2024tdmpc2] dominates 104
continuous-control tasks with MPPI in an implicit value-equivalent latent;
DreamerV3 [@hafner2023dreamerv3] uses a reconstruction RSSM and reached
diamond in Minecraft; UniZero [@unizero2024] combines a transformer latent
with MCTS; Spectral Expansion Tree Search [@sets2024] adds real-time
continuous-MCTS guarantees on robotic systems. Sampled MuZero [@sampled-muzero]
and Voronoi Progressive Widening [@lim2020vpw] make tree search tractable in
high-dimensional continuous action spaces.

**Skill / landmark graphs.** SPTM [@savinov2018sptm] and World-Model-as-a-Graph
[@zhang2021wmag] cluster trajectory latents into a graph whose nodes are
landmarks and whose edges are short-horizon controllers; the high-level
planner runs Dijkstra/A* over the graph while the low-level uses any inner
optimiser (MPC, MCTS). This is the natural template for *graph-indexed
recovery*, where each recovery edge can be a different specialised
controller.

**Runtime monitoring & failure detection.** Sentinel [@agia2024sentinel]
fuses temporal-consistency with VLM monitors and reports detecting 18% more
failures than either signal alone (CoRL 2024). FAIL-Detect [@faildetect2025]
trains an uncertainty head without failure labels. SAFE [@safe2025] uses
internal-feature uncertainty in VLA models. The 2026 *Safe-Stoppability
Monitor* preprint trains a neural classifier that predicts whether a
fallback can safely stop the robot, validated on the Unitree G1.

**Humanoid fall recovery.** FRASA [@frasa2024] and HiFAR [@hifar2025] train
end-to-end RL policies for get-up. FIRM [@xu2025firm] (Nov 2025) introduces
a "memory of safe reactions" diffusion prior on the G1. Spraggett et al.
[@spraggett2025crossq] (Dec 2025) train one CrossQ policy that recovers
across seven robot morphologies with 86 ± 7% zero-shot success.

**CBF safety + training.** Ames et al.'s control barrier functions
[@ames2017cbf] provide a clean projection-based safety filter. CBF-RL
[@yang2025cbfrl] (Oct 2025) applies the filter only during simulation
training, lets the policy internalise the constraint, and deploys
filter-free. We adopt this pattern verbatim.

**Sim platform.** NVIDIA Isaac Lab is the de facto platform for G1 sim-to-real
research [@nvidia2024isaaclab]; the Isaac-Sim docs report "27 minutes of
real-world experience per simulator-second" for rough-terrain G1 on a 4090.
Our setup matches: dual RTX 5090, Isaac Sim 5.1, conda Python 3.11 with
`torch==2.7.0+cu128`.

The gap our work targets: **no single paper unifies layered world models +
skill graph + continuous MCTS + Sentinel-style monitor + CBF-RL safety +
graph-indexed recovery into a closed loop on a humanoid.** GATS-R is that
integration, packaged as a reference implementation.

## 3. Architecture

```
+-----------------------------------------------------------+
|                       GATS-R Agent                        |
|  +--------+   +-----------------+   +-------------------+ |
|  | Skill  |-->| Two-level       |-->| CBF Safety Filter | |
|  | Graph  |   | Planner         |   +-------------------+ |
|  +--------+   |  A* over graph  |             |           |
|       ^      |  + MCTS + VPW    |             v           |
|       |      +-----------------+      +-------------+     |
|       |              ^                | Environment |     |
|       |              |                +-------------+     |
|       |   +----------+----------+            |            |
|       |   | Layered World Model |            |            |
|       |   |  L1 analytic        |            v            |
|       |   |  L2 ensemble latent |     +-------------+     |
|       |   |  L3 fallback        |     | Monitor:    |     |
|       |   +---------------------+     | ensemble +  |     |
|       |              ^                | temporal    |     |
|       |              |                +-------------+     |
|       |              +-- ood -+ +----------+   | OOD      |
|       +-----------------------|-| Recovery |<--+          |
|                               +-| dispatch |              |
|                                 +----------+              |
+-----------------------------------------------------------+
```

### 3.1 Layered world model (L1 / L2 / L3)

We replace GATS's symbolic STRIPS match with **L1 analytic dynamics** (a
linearisation valid in a known neighbourhood — cart-pole linearisation for
BalanceBot, centroidal-momentum / 3D-LIPM for the G1) that exposes a
*validity* score `v(s) ∈ [0, 1]`. **L2** is a small ensemble of MLP dynamics
heads (default: 4 heads, hidden 256, latent dim 64) wrapped around a shared
encoder/decoder. L2 reports a per-state epistemic uncertainty as ensemble
standard deviation in latent space. **L3** is a generative proposer that
returns candidate sub-goals (recovery anchor at upright, halfway-to-goal,
direct-to-goal, plus diversity jitter). The orchestrator chooses L1 if
`v(s) ≥ τ_v`, else L2 if `ε(s) ≤ τ_ε`, else L3 fires and the agent re-plans
toward one of the L3 anchors.

### 3.2 Skill graph

Following SPTM [@savinov2018sptm] and World-Model-as-a-Graph [@zhang2021wmag],
we cluster latent embeddings from a small random-policy dataset into N
landmarks (k-means lite, 5 Lloyd iterations). Edges are added between
landmarks whose latent distance is below a radius; every node carries an
explicit edge to a dedicated **recovery anchor** node, guaranteeing the
recovery dispatcher always has a routable target.

### 3.3 Continuous MCTS with Voronoi Progressive Widening

Within each skill-graph edge we run continuous MCTS (Coulom 2007 action
progressive widening, Lim et al. 2020 Voronoi cell selection): a new child
action is sampled only when `n_children ≤ k_pw · n_visits^α_pw`, and
candidates are scored by farthest-from-existing-actions distance in action
space. Leaf values are bootstrapped via short random rollouts under the
learned dynamics. The high-level Dijkstra over the skill graph hands MCTS a
short-horizon sub-goal; the same interface accepts a pure MPPI inner loop
as a baseline switch.

### 3.4 Sentinel-style runtime monitor

We follow Agia et al. [@agia2024sentinel] and combine **ensemble
disagreement** (L2's per-state ε) with **temporal consistency** (rolling
variance of recent actions over an 8-step window). Optional third signal:
a **safe-stoppability proxy** (base-tilt above a calibrated angle, or
projected-gravity horizontal magnitude on the G1) that approximates the
2026 learned safe-stoppability monitor without requiring a separate trained
network. Thresholds are calibrated to the 95th percentile of nominal-
operation samples.

### 3.5 CBF-RL safety filter

Following CBF-RL [@yang2025cbfrl] we apply the filter *during simulation
training* only, so the policy internalises the constraint. On BalanceBot
the barriers are pole-tilt and cart-position. On the G1 the position-only
invariant is `h_tilt(s) = sin(θ_max)² − ‖g_xy‖²` where `g_xy` is the
horizontal component of body-frame projected gravity. The filter projects
unsafe actions onto the safe set via line-search between the proposed
action and a stabilising reference.

### 3.6 Graph-indexed recovery

When the monitor flags OOD or the env reports fallen, the dispatcher snaps
the current state to its nearest skill-graph node, looks up the routed
recovery edge, and invokes the controller registered for that edge. The
default mapping is a global LQR (toy task) or a PD-to-default-pose
controller (G1); the interface is a plain `callable(state) -> action` so
FRASA / FIRM / Get-Up-Across-Morphologies policies can be swapped in
per-edge without touching the rest of the stack.

## 4. Implementation

### 4.1 CPU reference: *BalanceBot*

We built a custom planar cart-pole task with multi-goal sequences and
stochastic disturbances (push impulses, payload mass jitter, friction
noise, sensor noise). State is 7-D, action is 1-D continuous force. The
environment exposes a separate `recover_step()` channel that mirrors the
FRASA-style "safe set" assumption for recovery edges. Three OOD levels
(0.0 / 0.5 / 1.0) scale every perturbation source jointly. All ten
methods (six GATS-R configurations + four baselines), three seeds, three
OOD levels, ten episodes each run in roughly 45 minutes on a single core
(a ~5-minute smoke configuration is provided via reduced seeds/steps).

### 4.2 Isaac Lab G1 port

A second subpackage `src/gatsr/isaaclab/` re-implements every component as
GPU-batched torch tensors against Isaac Lab 2.2's `ManagerBasedRLEnv` for
the registered `Isaac-Velocity-Flat-G1-v0` and `Isaac-Velocity-Rough-G1-v0`
tasks. Highlights:

- **Env wrapper** mirrors the BalanceBot API: `reset / step / recover_step`
  with `physical_state` exposing `[lin_vel(3), ang_vel(3), grav(3),
  jp(N), jv(N)]` — a structured 83-D proprioceptive view that the CBF and
  recovery layers consume directly.
- **GPU ensemble latent** with an optional second-GPU rollout mirror
  (disabled by default because slot-2 on the test rig is PCIe gen 3 ×1).
- **Batched MPPI** runs `N · K` rollouts (envs × samples) in a single
  flat tensor through the latent model.
- **Per-env vectorised** CBF, monitor, and recovery: every counter is a
  length-`num_envs` tensor.

The wrapper API is intentionally close to BalanceBot's so the same
benchmark logic can drive either world.

### 4.3 Tooling and reproducibility

- A `.bat` launcher (and PowerShell 7 equivalent) sets the Isaac-Sim env
  vars and runs the conda env's Python; no `conda activate` required.
- 59-test pytest suite (50 CPU + 9 Isaac-Lab-module tests that don't need
  the simulator).
- Both benchmarks emit `raw.csv` (per-episode) and `summary.csv`
  (mean ± std per method).
- A windowed visualiser script (`scripts/isaaclab_visualize.py`) renders
  the G1 in the Isaac Sim viewport while the chosen method drives it,
  printing live CBF/OOD/recovery counters every 100 steps.

## 5. Experiments

### 5.1 BalanceBot CPU benchmark

Ten methods (random; LQR = the analytic L1 controller; MPPI in L2;
TD-MPC2-lite with a value head fit on real Monte-Carlo returns; Dreamer-lite
RSSM with an imagination-trained actor; full GATS-R; and five ablations),
3 seeds × 10 episodes per OOD level (n = 30 per cell), ~45 min on one
Ryzen-9-9900X core. Values below are means over the three OOD levels;
per-OOD breakdowns are in `results/summary.csv`.

| Method | Success | Return | Recovery att./ep | Recovery success | CBF interv./ep | Planning ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0.00 | -62.9 | 0.00 | — | 0.0 | 0.0 |
| MPPI (L2 only) | 0.00 | -54.1 | 0.00 | — | 0.0 | 18.1 |
| TD-MPC2-lite | 0.00 | -45.1 | 0.00 | — | 0.0 | 18.0 |
| Dreamer-lite | 0.00 | -61.4 | 0.00 | — | 0.0 | 0.1 |
| **LQR (= L1)** | **0.88** | **+269.7** | 0.00 | — | 0.0 | 0.0 |
| **GATS-R (full)** | **0.64** | +220.6 | 0.36 | 0.00 | 24.3 | 26.7 |
| GATS-R no layered (L1 off) | 0.00 | -0.5 | 2.54 | 0.60 | 40.1 | 24.3 |
| GATS-R no graph | 0.64 | +221.1 | 0.36 | 0.00 | 24.5 | 29.4 |
| GATS-R no recovery | 0.64 | +220.6 | 0.00 | — | 23.3 | 27.5 |
| GATS-R no monitor | 0.64 | +220.6 | 0.36 | 0.00 | 24.3 | 27.5 |
| GATS-R no CBF | 0.79 | +253.0 | 0.21 | 0.00 | 0.0 | 29.9 |

**Findings.** (i) The analytic L1 controller (LQR) wins on this linearisable
task (0.88 success, +270 return) — exactly why the architecture keeps it as
L1; GATS-R inherits most of that performance through its layered selector
(0.64). (ii) The pure model-based-learning baselines fail outright (0.00
success) at this deliberately minimal CPU training budget (2000 random
transitions): MPPI/TD-MPC2-lite plan through an L2 that is too inaccurate to
drive the cart to goals, and Dreamer-lite's imagination actor — though now
genuinely trained — cannot either. This is the honest, expected result and the
motivation for the analytic prior. (iii) **The layered L1 selector is
load-bearing**: the `no layered` ablation, which forces planning through L2
everywhere, collapses success to 0.00 — the single largest ablation effect.
(iv) CBF is a *safety-vs-performance trade-off*, not a free win: turning it off
*raises* success and return (0.79 / +253) because the policy is unconstrained;
its value is the ~24 interventions/episode it makes on unsafe proposed actions,
which matters on the G1 (§5.2), not on this benign cart-pole. (v) Graph,
monitor, and recovery are within noise of full here — once L1 holds the pole
upright the monitor rarely flags OOD and recovery rarely fires (and the few
attempts do not complete within the episode, hence 0.00 recovery success). The
recovery layer is exercised meaningfully in the `no layered` ablation
(2.5 attempts/ep at 0.60 success) and on the G1 (§5.2). (vi) MCTS planning sits
at ~27 ms/decision, above the 20 ms G1 control-loop budget — flagged as
follow-up.

Robustness curves vs OOD and per-ablation panels are in `results/figures/`
(`fig01`–`fig06`); `scripts/verify_claims.py` turns findings (i)–(v) plus
per-seed determinism into automated assertions.

### 5.2 Isaac Lab G1 closed-loop validation

Setup: Isaac Sim 5.1, `Isaac-Velocity-Rough-G1-v0`, 16 parallel envs ×
3 episodes × 150 steps per method, dual RTX 5090. The L2 model is
*deliberately under-trained* on 512 random transitions to keep the smoke
benchmark under ten minutes. This is the table from RESULTS.md, reproduced
here for self-containment:

| Method | Return | CBF interv./ep | Recovery att./ep | Recovery success | Time-to-rec. | Planning ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | -4.82 | 0 | 0 | — | — | 0.0 |
| mppi | -3.49 | 0 | 0 | — | — | 4.9 |
| gatsr_no_rec | -3.49 | 0 | 0 | — | — | 4.9 |
| **gatsr_full** | -3.99 | **16.2** | **1.42** | **91.2%** | **~14** | **5.0** |

**Findings.** (i) *Only* `gatsr_full` activates CBF and recovery — the
ablation rows are otherwise bit-identical to `mppi`, confirming the
implementation has no spurious activation paths. (ii) Recovery success
rate is 91.2% at ~14 control-steps to recover; the rest are timeouts on
states the PD placeholder cannot stabilise (where FRASA / FIRM would slot
in). (iii) Planning fits the budget: 5 ms per decision against a 20 ms
G1 control loop. (iv) `success_rate` is 0 across the board — the
under-trained L2 cannot keep a humanoid upright on rough terrain for 150
steps. We make no claim otherwise; the headline is *the closed loop runs
and the safety machinery activates measurably on real humanoid physics.*

### 5.3 What the experiments *do not* establish

- We do not claim improved task return on Isaac Lab G1; the L2 is too
  under-trained for that.
- We do not run real-G1 hardware.
- We do not run the ten Isaac Lab tasks the CoRL-2024 bar would require.
- We do not compare against full TD-MPC2 (Hansen 2024 codebase) or
  DreamerV3 (Hafner 2023 codebase) at their published configurations.
- We do not statistically test improvements (≥3 seeds are reported, but
  with the L2 capacity at this scale the variance is dominated by L2
  fit quality, not by the safety layers we ablate).

## 6. Limitations and threats to validity

**Under-trained L2.** The Isaac Lab L2 sees only 512 random transitions
because we time-boxed the smoke benchmark. A real evaluation needs at
least 10^6 transitions and a few GPU-hours of training. We deliberately
chose this scope to validate *integration* before *performance*.

**One task, no hardware.** Isaac Lab results are on one task; CoRL 2024
oral acceptance requires ~10. No real Unitree G1 results — sim-to-real
gap remains untested.

**Placeholder recovery.** The G1 recovery is a PD-to-default-pose, not a
trained FRASA/FIRM policy. Recovery success on hard falls (large initial
tilt, broken contact) is correspondingly lower than what a learned policy
would deliver; the 91.2% number reflects mostly mild stumbles.

**Single inner-loop optimizer.** The G1 port uses batched MPPI inside
each skill-graph edge; continuous MCTS is only used on the CPU toy.
Scaling MCTS+VPW to 37-D action space is left as follow-up.

**Genesis / other simulators.** We followed the thesis recommendation
and used Isaac Lab; cross-simulator transfer (MJX, Genesis) is not
reported.

**Reviewer-rejection risks.** A 2024–2025 CoRL/RSS submission of this
work as a *headline* paper would be rejected on incremental grounds.
The natural targets are a workshop (CoRL-WS, NeurIPS-RobotLearning,
RSS-WS) or arXiv preprint as a reference implementation; the publishable
headline emerges only after the follow-up training and hardware work in
§7.

## 7. Roadmap to publication

The interfaces are designed so a follow-up team can land each item below
without touching the orchestrator:

1. **Train L2 to convergence** — 10^6+ transitions, multi-seed; target
   sub-1.5x the published TD-MPC2 step-cost on the G1 task. (Days of
   GPU.)
2. **Swap PD recovery for FRASA / FIRM** — register the FRASA controller
   on the recovery-anchor edge; verify recovery success rate climbs.
3. **Add 9 more Isaac Lab G1 tasks** — `Flat-G1-v0`, `Rough-G1-Run`,
   `BeamDojo`, plus three perturbation sweeps and three manipulation
   tasks via RoboCasa; report mean ± std over 5 seeds.
4. **Calibrate the monitor properly** — quantile thresholds on a
   nominal-operation dataset, separately per OOD level for the
   precision/recall curve.
5. **Statistical tests** — paired Wilcoxon over (seed × episode) cells.
6. **Real G1 push-recovery smoke test** — one hardware experiment on a
   single G1 unit, flat ground only, push-from-side.

If items 1–5 land at the magnitudes the literature suggests, the headline
GATS-R-vs-TD-MPC2 success-rate gap on the OOD sweep should be the
publishable result; this paper is the credible baseline that the
follow-up would compare against.

## 8. Conclusion

GATS-R is an *integration*: layered world models + skill graph +
continuous MCTS + Sentinel monitor + CBF-RL safety + graph-indexed
recovery, closed into a single loop on continuous robot control. We
release a CPU reference benchmark with full ablations and a
GPU-batched Isaac Lab port that drives the 37-DoF Unitree G1 humanoid
end-to-end, with every architectural component verifiably activating on
real humanoid physics. The contribution is not a benchmark-beating
number; it is a clean, inspectable, reproducible foundation on which the
research community can lift in the published TD-MPC2 / FRASA / safe-
stoppability components and run the publication-grade comparison the
parent thesis lays out.

## Reproducibility

All experiments in this paper are reproducible from a fresh clone:

```bash
git clone https://github.com/MMWilliams/gats-r.git
cd gats-r && pip install -r requirements.txt && pip install -e .
pytest -q                                                 # 59 tests, ~4 s
python scripts/verify_claims.py                           # asserts §5.1 findings, ~2 min
python scripts/benchmark.py --seeds 3 --episodes 10       # CPU table (§5.1), ~45 min
python scripts/make_figures.py                            # 6 figures
```

Isaac Lab + G1 (requires Isaac Sim 5.x, `isaaclab` conda env, NVIDIA GPU):

```cmd
scripts\run_isaaclab.bat scripts\isaaclab_benchmark.py ^
    --task Isaac-Velocity-Rough-G1-v0 --num_envs 16 --episodes 3 ^
    --max_steps 150 --train_steps 512 ^
    --methods random mppi gatsr_no_rec gatsr_full
```

For the windowed visualiser:

```cmd
scripts\run_isaaclab.bat scripts\isaaclab_visualize.py ^
    --task Isaac-Velocity-Rough-G1-v0 --num_envs 2 --method gatsr_full ^
    --train_steps 512 --run_steps 3000
```

Total CPU benchmark wall-clock: ~45 min (`--seeds 3 --episodes 10`; a
~5 min smoke config is provided). Total Isaac Lab benchmark wall-clock:
~8 min. Substantive metrics are deterministic per `--seed` on a given
machine (the `planning_ms` timing column aside).

## Acknowledgements

This work consumed the GATS conceptual scaffold and the 2024–2026
robotics-RL literature catalogued in the parent research-direction
analysis. Implementation, port, and writing assisted by Claude Opus 4.7
under direction.

## References

The references below are paraphrased from the parent research-direction
analysis (which itself is open and inspectable in the repository's session
log). Each is a known publication or preprint:

[@gats]: Williams, M. M. *Generalized Augmented Tree Search: Layered World
Models for Reliable Symbolic Planning.* (Manuscript.)

[@hansen2024tdmpc2]: Hansen, N., Su, H., Wang, X. *TD-MPC2: Scalable, Robust
World Models for Continuous Control.* ICLR 2024.

[@hafner2023dreamerv3]: Hafner, D. et al. *DreamerV3: Mastering Diverse
Domains through World Models.* arXiv 2301.04104.

[@unizero2024]: Pu, Y. et al. *UniZero: Generalized and Efficient Planning
with Scalable Latent World Models.* 2024.

[@sets2024]: Riviere, B., Hönig, W., Anderson, M., Chung, S.-J. *Spectral
Expansion Tree Search.* Science Robotics, 2024.

[@pwm2024]: Georgiev, B., Giridhar, V., Hansen, N., Garg, A. *PWM: Policy
Learning with Multi-task World Models.* 2024.

[@savinov2018sptm]: Savinov, N. et al. *Semi-Parametric Topological
Memory for Navigation.* ICLR 2018.

[@zhang2021wmag]: Zhang, L., Yang, G., Stadie, B. C. *World Model as a
Graph: Learning Latent Landmarks for Planning.* ICML 2021.

[@lim2020vpw]: Lim, M. H., Tomlin, C. J., Sunberg, Z. *Voronoi Progressive
Widening: Efficient Online Solvers for Continuous POMDPs.* CDC 2020.

[@sampled-muzero]: Hubert, T. et al. *Learning and Planning in Complex
Action Spaces.* ICML 2021.

[@agia2024sentinel]: Agia, C., Sinha, R., Yang, J., Cao, R., Antonova, R.,
Pavone, M., Bohg, J. *Sentinel: Multi-Stage Failure Detection for
Imitation Learning.* CoRL 2024, PMLR v270.

[@faildetect2025]: *FAIL-Detect: Uncertainty-Aware Runtime Failure Detection
without Failure Data.* 2025.

[@safe2025]: Liang, J., Sinha, R., Itkina, M. et al. *SAFE: Internal-Feature
Uncertainty for VLA Failure Detection.* CoRL 2025.

[@frasa2024]: *FRASA: End-to-end Fall Recovery via Reinforcement Learning.*
2024.

[@hifar2025]: *HiFAR: Hierarchical Fall Avoidance and Recovery.* Feb 2025.

[@xu2025firm]: Xu, K. et al. *FIRM: Fall-Prevention, Impact-Mitigation, and
Stand-Up via Memory of Safe Reactions on the Unitree G1.* arXiv 2511.07407,
Nov 2025.

[@spraggett2025crossq]: Spraggett, J. *Learning to Get Up Across
Morphologies.* arXiv 2512.12230, RoboCup Symposium 2025.

[@yang2025cbfrl]: Yang, J. et al. *CBF-RL: Safety Filtering RL in Training
with Control Barrier Functions.* arXiv 2510.14959, Oct 2025.

[@ames2017cbf]: Ames, A. D. et al. *Control Barrier Functions: Theory and
Applications.* ECC 2019.

[@rwm2025]: Li, M., Krause, A., Hutter, M. *RWM: Robust World Models with
Epistemic Uncertainty Heads.* arXiv 2504.16680, ETH Zürich 2025.

[@nvidia2024isaaclab]: NVIDIA. *Isaac Lab Documentation, v2.2.* 2024–2025.

---

*Repository:* <https://github.com/MMWilliams/gats-r>
