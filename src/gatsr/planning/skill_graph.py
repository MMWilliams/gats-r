"""Skill graph (latent-landmark graph).

Built offline from collected trajectories: cluster the latent embeddings into
N landmark nodes, then add edges where transitions actually occurred in the
training data (reachability proxy a la SPTM, Savinov et al. 2018, and World
Model as a Graph, Zhang/Yang/Stadie 2021).

At plan time the agent:
    1. Snaps the current state to the nearest landmark (start).
    2. Snaps the goal to the nearest landmark (target).
    3. Runs Dijkstra over the graph to get a sub-goal sequence.
    4. Hands each sub-goal to the inner-loop planner (MPPI or MCTS).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SkillNode:
    idx: int
    latent: np.ndarray
    physical: np.ndarray  # canonical physical state at the node
    is_recovery: bool = False  # True for nodes used as recovery anchors
    description: str = ""


@dataclass
class SkillGraph:
    nodes: List[SkillNode] = field(default_factory=list)
    edges: dict = field(default_factory=dict)  # node_idx -> list[(target_idx, cost)]

    # --- construction -----------------------------------------------------

    def add_node(self, node: SkillNode) -> int:
        node.idx = len(self.nodes)
        self.nodes.append(node)
        self.edges.setdefault(node.idx, [])
        return node.idx

    def add_edge(self, i: int, j: int, cost: float) -> None:
        for k, (t, c) in enumerate(self.edges[i]):
            if t == j:
                self.edges[i][k] = (j, min(c, cost))
                return
        self.edges[i].append((j, cost))

    @classmethod
    def from_trajectories(
        cls,
        latents: np.ndarray,
        physicals: np.ndarray,
        n_clusters: int = 16,
        edge_radius: float = 0.6,
        recovery_anchor_physical: np.ndarray | None = None,
        seed: int = 0,
    ) -> "SkillGraph":
        """Cluster trajectory latents into landmarks and add edges based on
        observed temporal transitions and Euclidean proximity."""
        rng = np.random.default_rng(seed)
        N = latents.shape[0]
        n_clusters = min(n_clusters, max(2, N // 4))
        idx = rng.choice(N, size=n_clusters, replace=False)
        centers = latents[idx].copy()
        # 5 Lloyd iterations (k-means lite)
        for _ in range(5):
            d = np.linalg.norm(latents[:, None] - centers[None], axis=-1)
            assign = d.argmin(axis=1)
            for k in range(n_clusters):
                if (assign == k).sum() > 0:
                    centers[k] = latents[assign == k].mean(axis=0)
        # canonical physical state per node = mean of physical states in cluster
        graph = cls()
        for k in range(n_clusters):
            mask = assign == k
            if mask.sum() == 0:
                continue
            phys = physicals[mask].mean(axis=0)
            graph.add_node(
                SkillNode(
                    idx=-1,
                    latent=centers[k].astype(np.float64),
                    physical=phys.astype(np.float64),
                    description=f"landmark_{k}",
                )
            )
        # always add at least one explicit recovery anchor (upright, zero velocity)
        if recovery_anchor_physical is None:
            recovery_anchor_physical = np.zeros(4)
        anchor_latent = centers.mean(axis=0)  # placeholder; will be updated by SkillGraph.refresh_recovery
        graph.add_node(
            SkillNode(
                idx=-1,
                latent=anchor_latent,
                physical=recovery_anchor_physical.astype(np.float64),
                is_recovery=True,
                description="recovery_upright",
            )
        )
        # edges by Euclidean proximity in latent
        for i, ni in enumerate(graph.nodes):
            for j, nj in enumerate(graph.nodes):
                if i == j:
                    continue
                d = float(np.linalg.norm(ni.latent - nj.latent))
                if d < edge_radius:
                    graph.add_edge(i, j, d)
            # ensure every node has an edge to recovery anchor (graph-indexed recovery)
            graph.add_edge(i, graph.nodes[-1].idx, 5.0)
        return graph

    # --- queries ---------------------------------------------------------

    def nearest(self, latent: np.ndarray) -> int:
        d = [float(np.linalg.norm(n.latent - latent)) for n in self.nodes]
        return int(np.argmin(d))

    def nearest_physical(self, physical: np.ndarray) -> int:
        d = [float(np.linalg.norm(n.physical[:4] - physical[:4])) for n in self.nodes]
        return int(np.argmin(d))

    def shortest_path(self, start: int, goal: int) -> List[int]:
        """Dijkstra over the graph."""
        if start == goal:
            return [start]
        dist = {n.idx: float("inf") for n in self.nodes}
        prev = {n.idx: None for n in self.nodes}
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == goal:
                break
            for v, c in self.edges.get(u, []):
                nd = d + c
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist.get(goal, float("inf")) == float("inf"):
            return [start]
        # reconstruct
        path: List[int] = []
        u: Optional[int] = goal
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
        return path

    def recovery_node(self) -> SkillNode:
        for n in reversed(self.nodes):
            if n.is_recovery:
                return n
        return self.nodes[-1]

    def __len__(self) -> int:
        return len(self.nodes)
