from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class EpisodeLog:
    method: str
    seed: int
    ood_level: float
    episode: int
    steps: int
    ep_return: float
    success: int
    failures_detected: int
    recoveries_attempted: int
    recoveries_succeeded: int
    safety_violations: int
    time_to_recover: float  # mean steps from fall to back-in-graph, -1 if N/A
    planning_ms: float  # mean per-decision planning latency


@dataclass
class BenchmarkLog:
    rows: list[EpisodeLog] = field(default_factory=list)

    def add(self, row: EpisodeLog) -> None:
        self.rows.append(row)

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("")
            return
        fields = list(asdict(self.rows[0]).keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(asdict(r))

    def extend(self, other: Iterable[EpisodeLog]) -> None:
        self.rows.extend(other)
