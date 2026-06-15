"""Verify the Isaac Lab + Unitree-G1 claims against produced data.

Reads `results/isaaclab_summary.csv` (produced by `scripts/isaaclab_benchmark.py`
on the rough-terrain G1 task with the documented command) and asserts the
headline numbers reported in RESULTS.md / paper/PAPER.md hold within tolerance.

Usage (after running the Isaac benchmark):
    python scripts/verify_isaaclab_claims.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "isaaclab_summary.csv"

# (column, expected, absolute tolerance) per method. Tolerances are generous
# enough to absorb GPU/driver nondeterminism but tight enough to catch a
# regression or a fabricated number.
EXPECTED = {
    "gatsr_full": {
        "return_mean": (-3.99, 0.4),
        "safety_violations_mean": (16.2, 2.0),
        "recoveries_attempted_mean": (1.42, 0.4),
        "recovery_success_rate": (0.912, 0.07),
    },
    "mppi": {"return_mean": (-3.49, 0.4), "safety_violations_mean": (0.0, 1e-6)},
    "gatsr_no_rec": {"return_mean": (-3.49, 0.4), "safety_violations_mean": (0.0, 1e-6)},
    "random": {"return_mean": (-4.82, 0.5), "safety_violations_mean": (0.0, 1e-6)},
}


def main() -> int:
    if not SUMMARY.exists():
        print(f"[verify-isaac] {SUMMARY} not found. Run scripts/isaaclab_benchmark.py first:")
        print("  scripts\\run_isaaclab.bat scripts\\isaaclab_benchmark.py "
              "--task Isaac-Velocity-Rough-G1-v0 --num_envs 16 --episodes 3 "
              "--max_steps 150 --train_steps 512 --methods random mppi gatsr_no_rec gatsr_full")
        return 2
    rows = {r["method"]: r for r in csv.DictReader(SUMMARY.open())}
    ok = True
    for method, checks in EXPECTED.items():
        if method not in rows:
            print(f"  [FAIL] {method}: missing from summary")
            ok = False
            continue
        for col, (exp, tol) in checks.items():
            got = float(rows[method][col])
            good = abs(got - exp) <= tol
            ok = ok and good
            print(f"  [{'PASS' if good else 'FAIL'}] {method}.{col}: got {got:.4f}, "
                  f"expected {exp} +/- {tol}")
    print(f"\n[verify-isaac] {'ALL VERIFIED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
