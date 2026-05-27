"""Produce publication-style figures from a finished benchmark run.

Reads `results/raw.csv` and `results/summary.csv`, writes the following plots
under `results/figures/`:

    fig01_success_vs_ood.png     — robustness curve: success vs OOD level per method
    fig02_return_vs_ood.png      — long-horizon return curve per method
    fig03_safety_violations.png  — bar chart of safety violations per method
    fig04_recovery.png           — recovery success rate + time-to-recover
    fig05_planning_latency.png   — per-decision planning latency (ms) per method
    fig06_ablation.png           — GATS-R ablation panel: drop each component

Run with:
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GATSR_METHODS = {
    "gatsr_full": "GATS-R (full)",
    "gatsr_no_graph": "  - no graph",
    "gatsr_no_recovery": "  - no recovery",
    "gatsr_no_monitor": "  - no monitor",
    "gatsr_no_cbf": "  - no CBF",
}
BASELINE_METHODS = {
    "random": "Random",
    "lqr": "LQR (analytic)",
    "mppi": "MPPI (L2 only)",
    "td_mpc2_lite": "TD-MPC2-lite",
    "dreamer_lite": "Dreamer-lite",
    "gatsr_full": "GATS-R (ours)",
}
METHOD_COLORS = {
    "random": "#999999",
    "lqr": "#1f77b4",
    "mppi": "#ff7f0e",
    "td_mpc2_lite": "#2ca02c",
    "dreamer_lite": "#9467bd",
    "gatsr_full": "#d62728",
    "gatsr_no_graph": "#e377c2",
    "gatsr_no_recovery": "#8c564b",
    "gatsr_no_monitor": "#bcbd22",
    "gatsr_no_cbf": "#17becf",
}


def load_raw(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            for k in (
                "seed",
                "ood_level",
                "episode",
                "steps",
                "ep_return",
                "success",
                "failures_detected",
                "recoveries_attempted",
                "recoveries_succeeded",
                "safety_violations",
                "time_to_recover",
                "planning_ms",
            ):
                if k == "method":
                    continue
                if k in r and r[k] != "":
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def _aggregate(rows, methods, key, agg=np.mean):
    """Returns dict method -> { ood -> (mean, std) }."""
    out = {m: {} for m in methods}
    bucket = defaultdict(list)
    for r in rows:
        if r["method"] not in methods:
            continue
        bucket[(r["method"], float(r["ood_level"]))].append(float(r[key]))
    for (m, ood), vs in bucket.items():
        out[m][ood] = (float(np.mean(vs)), float(np.std(vs)))
    return out


def fig_success_vs_ood(rows, out: Path) -> None:
    methods = list(BASELINE_METHODS.keys())
    agg = _aggregate(rows, methods, "success")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        oods = sorted(agg[m].keys())
        means = [agg[m][o][0] for o in oods]
        stds = [agg[m][o][1] for o in oods]
        ax.errorbar(
            oods,
            means,
            yerr=stds,
            label=BASELINE_METHODS[m],
            color=METHOD_COLORS[m],
            marker="o",
            capsize=3,
        )
    ax.set_xlabel("OOD perturbation level")
    ax.set_ylabel("Episode success rate")
    ax.set_title("Robustness curve: success vs. OOD level")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_return_vs_ood(rows, out: Path) -> None:
    methods = list(BASELINE_METHODS.keys())
    agg = _aggregate(rows, methods, "ep_return")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        oods = sorted(agg[m].keys())
        means = [agg[m][o][0] for o in oods]
        stds = [agg[m][o][1] for o in oods]
        ax.errorbar(
            oods,
            means,
            yerr=stds,
            label=BASELINE_METHODS[m],
            color=METHOD_COLORS[m],
            marker="s",
            capsize=3,
        )
    ax.set_xlabel("OOD perturbation level")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Return vs. OOD level (mean ± std across seeds × episodes)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_safety_violations(rows, out: Path) -> None:
    methods = list(BASELINE_METHODS.keys())
    agg = _aggregate(rows, methods, "safety_violations")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    oods = sorted({float(r["ood_level"]) for r in rows})
    x = np.arange(len(methods))
    for i, ood in enumerate(oods):
        means = [agg[m].get(ood, (0.0, 0.0))[0] for m in methods]
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            label=f"OOD = {ood:.1f}",
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([BASELINE_METHODS[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Mean CBF interventions per episode")
    ax.set_title("Safety-filter activations by method × OOD level")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_recovery(rows, out: Path) -> None:
    methods = [m for m in BASELINE_METHODS if m.startswith("gatsr") or m == "lqr"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # success rate of recovery attempts
    rec_by_method = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # m -> ood -> [att, succ]
    ttr_by_method = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = r["method"]
        if m not in methods:
            continue
        ood = float(r["ood_level"])
        rec_by_method[m][ood][0] += float(r["recoveries_attempted"])
        rec_by_method[m][ood][1] += float(r["recoveries_succeeded"])
        if float(r["time_to_recover"]) > 0:
            ttr_by_method[m][ood].append(float(r["time_to_recover"]))

    oods = [0.0, 0.5, 1.0]
    x = np.arange(len(methods))
    width = 0.25
    for i, ood in enumerate(oods):
        rates = []
        for m in methods:
            att, succ = rec_by_method[m].get(ood, [0, 0])
            rates.append(succ / max(1, att))
        ax1.bar(x + (i - 1) * width, rates, width, label=f"OOD = {ood:.1f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([BASELINE_METHODS.get(m, m) for m in methods], rotation=20, ha="right")
    ax1.set_ylabel("Recovery success rate")
    ax1.set_title("Recovery success rate per method × OOD")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=8)

    for m in methods:
        all_ttrs = [v for ood_list in ttr_by_method[m].values() for v in ood_list]
        if all_ttrs:
            ax2.scatter([m] * len(all_ttrs), all_ttrs, alpha=0.5, color=METHOD_COLORS.get(m, "#444"))
            ax2.scatter([m], [np.mean(all_ttrs)], color="red", marker="x", s=80, zorder=10)
    ax2.set_ylabel("Time to recover (env steps)")
    ax2.set_title("Time to recover (× = mean)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_planning_latency(rows, out: Path) -> None:
    methods = list(BASELINE_METHODS.keys())
    agg = _aggregate(rows, methods, "planning_ms")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = [agg[m].get(0.5, (0.0, 0.0))[0] for m in methods]
    stds = [agg[m].get(0.5, (0.0, 0.0))[1] for m in methods]
    bars = ax.bar(
        [BASELINE_METHODS[m] for m in methods],
        means,
        yerr=stds,
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )
    ax.set_ylabel("Mean planning time per decision (ms)")
    ax.set_title("Per-decision compute (OOD=0.5)")
    ax.axhline(y=20.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="G1 ctrl loop = 20 ms")
    ax.legend(fontsize=8)
    ax.set_xticklabels([BASELINE_METHODS[m] for m in methods], rotation=20, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ablation(rows, out: Path) -> None:
    methods = list(GATSR_METHODS.keys())
    success = _aggregate(rows, methods, "success")
    ret = _aggregate(rows, methods, "ep_return")
    safety = _aggregate(rows, methods, "safety_violations")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    oods = [0.0, 0.5, 1.0]
    x = np.arange(len(methods))
    width = 0.25
    for ax, agg, title, ylabel in [
        (axes[0], success, "Success rate vs. ablation", "Success rate"),
        (axes[1], ret, "Return vs. ablation", "Mean return"),
        (axes[2], safety, "CBF interventions vs. ablation", "Mean interventions / episode"),
    ]:
        for i, ood in enumerate(oods):
            means = [agg[m].get(ood, (0.0, 0.0))[0] for m in methods]
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                label=f"OOD = {ood:.1f}",
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([GATSR_METHODS[m] for m in methods], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    args = p.parse_args()
    res = Path(args.results_dir)
    raw = res / "raw.csv"
    if not raw.exists():
        raise SystemExit(f"missing {raw}; run scripts/benchmark.py first")
    rows = load_raw(raw)
    figdir = res / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig_success_vs_ood(rows, figdir / "fig01_success_vs_ood.png")
    fig_return_vs_ood(rows, figdir / "fig02_return_vs_ood.png")
    fig_safety_violations(rows, figdir / "fig03_safety_violations.png")
    fig_recovery(rows, figdir / "fig04_recovery.png")
    fig_planning_latency(rows, figdir / "fig05_planning_latency.png")
    fig_ablation(rows, figdir / "fig06_ablation.png")
    print(f"[done] wrote figures to {figdir}")


if __name__ == "__main__":
    main()
