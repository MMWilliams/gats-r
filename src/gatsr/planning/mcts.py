"""Continuous MCTS with Voronoi Progressive Widening (VPW).

Implementation of the Lim et al. (2020) variant of MCTS for continuous action
spaces. The key insight: instead of expanding all actions, sample a new action
only when the number of children at a node satisfies
    n_children <= k * n_visits ** alpha
(Action Progressive Widening; Coulom 2007 / Couëtoux 2011), and bias the new
action toward the Voronoi cell with largest unexplored volume (VPW; Lim 2020).

In this toy domain the action space is 1-D so VPW reduces to *quantile gap
selection*; we keep the generic implementation so the planner ports unchanged
to higher-dim G1-style action spaces.

The MCTS uses a world model `rollout_fn` as the simulator — identical
interface to MPPI so the two can be swapped in/out of `HybridPlanner`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class MCTSConfig:
    horizon: int = 6  # depth of the tree
    n_simulations: int = 64  # total simulations per call
    action_dim: int = 1
    discount: float = 0.97
    c_puct: float = 1.4
    # Progressive widening parameters
    k_pw: float = 1.0
    alpha_pw: float = 0.5
    # Voronoi: number of candidate samples when expanding
    n_candidates: int = 8
    # Value bootstrap horizon
    bootstrap_depth: int = 4
    seed: int = 0


@dataclass
class _Node:
    state: np.ndarray
    depth: int
    parent: Optional["_Node"] = None
    parent_action: Optional[np.ndarray] = None
    children: List["_Node"] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    visit_counts: List[int] = field(default_factory=list)
    value_sums: List[float] = field(default_factory=list)
    n_visits: int = 0
    cum_value: float = 0.0


class ContinuousMCTS:
    """MCTS over a learned/given dynamics model for continuous action."""

    def __init__(
        self,
        cfg: MCTSConfig,
        step_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        value_fn: Callable[[np.ndarray], float] | None = None,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        action_sampler: Callable[[np.random.Generator, np.ndarray], np.ndarray] | None = None,
    ):
        """
        step_fn(state, action) -> (next_state, epistemic)
        value_fn(state) -> bootstrapped value (used at the leaf)
        reward_fn(s, a, s') -> immediate reward
        action_sampler(rng, state) -> candidate action (default: uniform in [-1,1])
        """
        self.cfg = cfg
        self.step = step_fn
        self.value_fn = value_fn if value_fn is not None else (lambda s: 0.0)
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward
        self.action_sampler = action_sampler if action_sampler is not None else self._default_sampler
        self.rng = np.random.default_rng(cfg.seed)

    # --- defaults --------------------------------------------------------

    def _default_reward(self, s: np.ndarray, a: np.ndarray, sp: np.ndarray) -> float:
        # uprightness reward (cosine of pole angle) is a good local proxy
        return float(np.cos(sp[2] if sp.shape[-1] >= 3 else 0.0))

    def _default_sampler(self, rng: np.random.Generator, state: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=self.cfg.action_dim)

    # --- core algorithm --------------------------------------------------

    def plan(self, root_state: np.ndarray) -> Tuple[np.ndarray, dict]:
        root = _Node(state=root_state.copy(), depth=0)
        for _ in range(self.cfg.n_simulations):
            self._simulate(root)
        # pick best mean-value action child of root
        if not root.children:
            return self.action_sampler(self.rng, root_state), {"nodes": 1, "depth": 0}
        best_idx = int(np.argmax(
            [vs / max(n, 1) for vs, n in zip(root.value_sums, root.visit_counts)]
        ))
        return root.actions[best_idx], {
            "nodes": self._count_nodes(root),
            "max_depth": self._max_depth(root),
            "root_visits": root.n_visits,
        }

    def _simulate(self, root: _Node) -> None:
        node = root
        path: List[Tuple[_Node, int]] = []  # (node, child_idx)
        depth = 0
        rewards: List[float] = []

        # Selection + expansion
        while True:
            # Decide whether to widen
            allowed = math.floor(self.cfg.k_pw * (node.n_visits + 1) ** self.cfg.alpha_pw)
            if len(node.children) <= allowed and depth < self.cfg.horizon:
                # Expand: sample a candidate action using Voronoi-style farthest-from-existing
                a = self._sample_action_voronoi(node)
                s_next, _eps = self.step(node.state, a)
                r = self.reward_fn(node.state, a, s_next)
                child = _Node(state=s_next, depth=node.depth + 1, parent=node, parent_action=a)
                node.children.append(child)
                node.actions.append(a)
                node.visit_counts.append(0)
                node.value_sums.append(0.0)
                # rollout from child to bootstrap value
                rewards.append(r)
                path.append((node, len(node.children) - 1))
                v = self._bootstrap_value(child)
                self._backup(path, rewards, v)
                return
            if depth >= self.cfg.horizon or not node.children:
                break
            # PUCT selection
            idx = self._select_child(node)
            path.append((node, idx))
            r = self.reward_fn(node.state, node.actions[idx], node.children[idx].state)
            rewards.append(r)
            node = node.children[idx]
            depth += 1

        # Leaf — bootstrap value
        v = self._bootstrap_value(node)
        self._backup(path, rewards, v)

    def _bootstrap_value(self, node: _Node) -> float:
        """Short rollout using random actions, then value_fn at the tail.

        This is the MuZero/AlphaZero pattern. With value_fn returning 0 (no
        critic available), the bootstrap is the cumulative reward over the
        random tail, which still helps shape the tree."""
        s = node.state.copy()
        ret = 0.0
        gamma = 1.0
        for _ in range(self.cfg.bootstrap_depth):
            a = self.action_sampler(self.rng, s)
            sp, _ = self.step(s, a)
            r = self.reward_fn(s, a, sp)
            ret += gamma * r
            gamma *= self.cfg.discount
            s = sp
        ret += gamma * float(self.value_fn(s))
        return ret

    def _backup(self, path: List[Tuple[_Node, int]], rewards: List[float], leaf_value: float) -> None:
        # Compute returns along the path (Bellman-style, discounted)
        # rewards[i] is reward for transition at level i
        g = leaf_value
        for (node, idx), r in zip(reversed(path), reversed(rewards)):
            g = r + self.cfg.discount * g
            node.visit_counts[idx] += 1
            node.value_sums[idx] += g
            node.n_visits += 1
            node.cum_value += g

    def _select_child(self, node: _Node) -> int:
        # PUCT-style UCB with uniform prior (we use uniform because we lack a learned policy head)
        N = max(1, sum(node.visit_counts))
        c = self.cfg.c_puct
        scores = []
        for vs, n in zip(node.value_sums, node.visit_counts):
            q = vs / max(n, 1)
            u = c * math.sqrt(math.log(N + 1) / (n + 1))
            scores.append(q + u)
        return int(np.argmax(scores))

    def _sample_action_voronoi(self, node: _Node) -> np.ndarray:
        """Sample several candidate actions and pick the one farthest from
        the existing children (Voronoi-cell volume proxy)."""
        if not node.actions:
            return self.action_sampler(self.rng, node.state)
        candidates = np.stack(
            [self.action_sampler(self.rng, node.state) for _ in range(self.cfg.n_candidates)],
            axis=0,
        )
        existing = np.stack(node.actions, axis=0)  # (k, A)
        # pairwise distances (n_candidates, k)
        d = np.linalg.norm(candidates[:, None, :] - existing[None, :, :], axis=-1)
        min_d = d.min(axis=1)
        return candidates[int(np.argmax(min_d))]

    def _count_nodes(self, node: _Node) -> int:
        n = 1
        for c in node.children:
            n += self._count_nodes(c)
        return n

    def _max_depth(self, node: _Node) -> int:
        if not node.children:
            return node.depth
        return max(self._max_depth(c) for c in node.children)
