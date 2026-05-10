"""
Heterogeneous cluster model with data locality and failure support.

Extends the static HEFT cluster with:
  - Per-node data residency (which input partitions live on which node)
  - Bandwidth modeling for cross-node data transfer
  - Failure injection (node down for a recovery window)
  - Rolling load history for the autonomic controller
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class NodeType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    EDGE = "edge"


@dataclass
class ProcessorNode:
    pid: int
    ntype: NodeType
    speed: float           # relative compute speed
    memory_gb: float
    bandwidth_gbps: float  # link bandwidth from this node
    storage_gb: float = 100.0


@dataclass
class TaskRecord:
    task_id: str
    pid: int
    start: float
    finish: float
    data_local: bool
    failed: bool = False


class Cluster:
    """Runtime state of the cluster (mutable, advances with simulated time)."""

    def __init__(self, nodes: List[ProcessorNode], seed: int = 0):
        self.nodes = nodes
        self.n = len(nodes)
        # earliest time each processor is free
        self.avail: Dict[int, float] = {p.pid: 0.0 for p in nodes}
        # set of data partition IDs currently resident on each node
        self.data_residency: Dict[int, Set[str]] = {p.pid: set() for p in nodes}
        # rolling load history (used-time over the last window)
        self.busy_until: Dict[int, float] = {p.pid: 0.0 for p in nodes}
        # nodes currently down (pid -> recovery time)
        self.down_until: Dict[int, float] = {}
        # full schedule trace
        self.trace: List[TaskRecord] = []
        # rolling window of recent completion times per processor for MAPE-K
        self.recent_load: Dict[int, deque] = {p.pid: deque(maxlen=32) for p in nodes}
        self._rng_seed = seed

    # ---------- queries ----------
    def is_up(self, pid: int, now: float) -> bool:
        return self.down_until.get(pid, -1.0) <= now

    def free_at(self, pid: int, now: float) -> float:
        if not self.is_up(pid, now):
            return max(self.avail[pid], self.down_until[pid])
        return max(self.avail[pid], now)

    def comm_cost(self, src_pid: int, dst_pid: int, data_gb: float) -> float:
        if src_pid == dst_pid:
            return 0.0
        bw = min(self.nodes[src_pid].bandwidth_gbps, self.nodes[dst_pid].bandwidth_gbps)
        return data_gb / max(bw, 1e-6)

    def data_local(self, pid: int, partitions: List[str]) -> bool:
        if not partitions:
            return True
        return any(p in self.data_residency[pid] for p in partitions)

    # ---------- mutations ----------
    def commit(self, rec: TaskRecord, output_partition: Optional[str] = None):
        self.trace.append(rec)
        self.avail[rec.pid] = rec.finish
        self.busy_until[rec.pid] = rec.finish
        self.recent_load[rec.pid].append(rec.finish - rec.start)
        if output_partition is not None:
            self.data_residency[rec.pid].add(output_partition)

    def inject_failure(self, pid: int, now: float, recovery_seconds: float = 30.0):
        """Bring a node down. Any task running on it after `now` is interrupted."""
        self.down_until[pid] = now + recovery_seconds
        # data resident on a downed node is unavailable until recovery
        # (we keep the set; consumers check is_up before using locality)

    # ---------- metrics ----------
    def makespan(self) -> float:
        return max((r.finish for r in self.trace), default=0.0)

    def per_node_load(self) -> Dict[int, float]:
        return dict(self.busy_until)

    def load_imbalance(self) -> float:
        loads = list(self.busy_until.values())
        if not loads or max(loads) == 0:
            return 0.0
        mean = statistics.mean(loads)
        if mean == 0:
            return 0.0
        std = statistics.pstdev(loads)
        return std / mean  # coefficient of variation

    def rolling_imbalance(self) -> float:
        """Imbalance over the rolling window (used by MAPE-K controller)."""
        sums = [sum(q) if q else 0.0 for q in self.recent_load.values()]
        if not sums or max(sums) == 0:
            return 0.0
        mean = statistics.mean(sums)
        if mean == 0:
            return 0.0
        std = statistics.pstdev(sums)
        return std / mean

    def utilization(self, horizon: float) -> float:
        if horizon == 0:
            return 0.0
        total = sum(self.busy_until.values())
        return total / (self.n * horizon)


class ClusterFactory:
    """Cluster topologies that mirror common distributed-data setups."""

    @staticmethod
    def homogeneous_4(seed: int = 0) -> Cluster:
        nodes = [ProcessorNode(i, NodeType.CPU, 1.0, 16, 10.0) for i in range(4)]
        return Cluster(nodes, seed=seed)

    @staticmethod
    def cpu_gpu_mix(seed: int = 0) -> Cluster:
        nodes = [
            ProcessorNode(0, NodeType.CPU, 1.0, 16, 10.0),
            ProcessorNode(1, NodeType.CPU, 1.2, 16, 10.0),
            ProcessorNode(2, NodeType.GPU, 3.5, 32, 25.0),
            ProcessorNode(3, NodeType.GPU, 4.0, 32, 25.0),
        ]
        return Cluster(nodes, seed=seed)

    @staticmethod
    def edge_cloud(seed: int = 0) -> Cluster:
        # 2 edge nodes (slow, cheap bandwidth), 2 cloud nodes (fast, fat pipe)
        nodes = [
            ProcessorNode(0, NodeType.EDGE, 0.4, 4, 0.5),
            ProcessorNode(1, NodeType.EDGE, 0.5, 4, 0.5),
            ProcessorNode(2, NodeType.CPU, 2.0, 32, 20.0),
            ProcessorNode(3, NodeType.CPU, 2.2, 32, 20.0),
        ]
        return Cluster(nodes, seed=seed)

    @staticmethod
    def skewed_8(seed: int = 0) -> Cluster:
        """8-node cluster with skewed speeds — stresses the scheduler."""
        speeds = [1.0, 1.0, 1.5, 1.5, 2.0, 2.5, 3.0, 4.0]
        nodes = [
            ProcessorNode(i, NodeType.CPU if s < 2.5 else NodeType.GPU,
                          s, 16 + i, 5.0 + i)
            for i, s in enumerate(speeds)
        ]
        return Cluster(nodes, seed=seed)


def effective_compute(base_cost: float, node: ProcessorNode) -> float:
    """Wall-clock seconds to run a task of `base_cost` on `node`."""
    return base_cost / max(node.speed, 1e-6)
