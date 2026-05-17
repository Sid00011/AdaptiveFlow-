"""
Tabular Q-learning agent for per-task processor selection.

State (discrete, hand-crafted):
  - task_size_bucket  : 0 (small) | 1 (medium) | 2 (large)
  - locality_bucket   : 0 (none local) | 1 (some local) | 2 (best is local)
  - load_pattern      : 0 (balanced) | 1 (mild skew) | 2 (heavy skew)
  - best_node_type    : 0 (CPU) | 1 (GPU) | 2 (Edge) — type of EFT-minimizing candidate

Action: select a processor pid from the candidate set (masked).

Reward: assigned at end of DAG = -normalized_makespan + locality_bonus,
backpropagated to every (state, action) tuple of the episode (Monte Carlo).

Learning: epsilon-greedy with decaying epsilon. Q-update via incremental mean
(equivalent to Q-learning with alpha = 1/visits — variance-stable for
small state spaces, no hyperparameter tuning needed).
"""
from __future__ import annotations

import math
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cluster import Cluster, NodeType
from dag_generator import DataDAG, DataTask


State = Tuple[int, int, int, int]  # (size, locality, load, best_type)


@dataclass
class TrainingStats:
    episode: int = 0
    total_reward: float = 0.0
    last_makespan: float = 0.0
    exploration_rate: float = 1.0


class QLearningAgent:
    def __init__(
        self,
        n_proc: int,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        gamma: float = 0.95,
        seed: int = 0,
    ):
        self.n_proc = n_proc
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.q: Dict[State, Dict[int, float]] = defaultdict(
            lambda: {p: 0.0 for p in range(n_proc)}
        )
        self.visits: Dict[Tuple[State, int], int] = defaultdict(int)
        self.stats = TrainingStats()
        # rolling baseline for advantage estimation
        self._return_baseline: float = 0.0
        self._baseline_alpha = 0.1

    # --------------- state encoding ---------------
    def encode_state(self, cluster: Cluster, task: DataTask, options) -> State:
        # task size bucket relative to a fixed scale (sec)
        if task.base_compute < 3.0:
            size_b = 0
        elif task.base_compute < 8.0:
            size_b = 1
        else:
            size_b = 2

        local_count = sum(1 for (_, _, _, local) in options if local)
        if local_count == 0:
            loc_b = 0
        elif local_count >= len(options) - 1:
            loc_b = 2
        else:
            loc_b = 1

        loads = list(cluster.per_node_load().values())
        if loads and max(loads) > 0:
            mean = sum(loads) / len(loads)
            std = (sum((l - mean) ** 2 for l in loads) / len(loads)) ** 0.5
            cv = std / max(mean, 1e-6)
        else:
            cv = 0.0
        if cv < 0.15:
            load_b = 0
        elif cv < 0.45:
            load_b = 1
        else:
            load_b = 2

        # type of the EFT-best candidate
        best = min(options, key=lambda o: o[2])
        bt = cluster.nodes[best[0]].ntype
        type_b = {NodeType.CPU: 0, NodeType.GPU: 1, NodeType.EDGE: 2}[bt]

        return (size_b, loc_b, load_b, type_b)

    # --------------- action selection ---------------
    def act(self, state: State, valid_pids: List[int]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(valid_pids)
        qvals = self.q[state]
        # mask invalid actions
        best = max(valid_pids, key=lambda p: qvals[p])
        return best

    # --------------- learning ---------------
    def observe_episode(
        self,
        dag: DataDAG,
        transitions: List[Tuple[State, int, float]],
        placement,
    ):
        """
        Called by the RLScheduler after a DAG is fully scheduled.
        Computes a single episodic reward and updates Q-values via
        Monte-Carlo return.
        """
        if not transitions:
            return
        finishes = [r.finish for r in placement.values()]
        starts = [r.start for r in placement.values()]
        makespan = max(finishes) - min(starts)
        local_rate = sum(1 for r in placement.values() if r.data_local) / len(placement)

        # Reward: shorter makespan is better, with locality bonus.
        # Normalize by avg compute so reward scale is roughly O(1).
        avg_c = dag.average_compute()
        norm_make = makespan / max(avg_c * len(placement), 1e-6)
        reward = -norm_make + 0.3 * local_rate

        # Update rolling baseline
        self._return_baseline = (
            (1 - self._baseline_alpha) * self._return_baseline
            + self._baseline_alpha * reward
        )
        advantage = reward - self._return_baseline

        # Apply advantage to every (state, action) of the episode
        for (state, action, _eft) in transitions:
            self.visits[(state, action)] += 1
            n = self.visits[(state, action)]
            old = self.q[state][action]
            self.q[state][action] = old + (advantage - old) / n

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.stats.episode += 1
        self.stats.total_reward += reward
        self.stats.last_makespan = makespan
        self.stats.exploration_rate = self.epsilon

    # --------------- persistence ---------------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "q": dict(self.q),
                "visits": dict(self.visits),
                "epsilon": self.epsilon,
                "baseline": self._return_baseline,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k, v in data["q"].items():
            self.q[k] = v
        for k, v in data["visits"].items():
            self.visits[k] = v
        self.epsilon = data["epsilon"]
        self._return_baseline = data["baseline"]
