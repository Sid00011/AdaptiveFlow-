"""
Schedulers for streaming data workflows.

All schedulers schedule one DataDAG against a live Cluster, respecting:
  - per-task heterogeneous compute cost
  - data locality bonus (zero comm cost when input partition is resident)
  - cross-node communication cost = data_gb / min(bandwidth)
  - current per-processor availability (no preemption)

Implemented:
  - HEFTLC      : Heterogeneous Earliest Finish Time with load-aware tie-break
                  (static baseline — port of the previous project's contribution)
  - DLS         : Data Locality Scheduling — locality-aware variant of HEFT-LC
  - RLScheduler : delegates the per-task placement decision to a learned policy

All schedulers return a list of TaskRecord; the simulator commits them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cluster import Cluster, TaskRecord, effective_compute
from dag_generator import DataDAG, DataTask


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def task_ready_time(
    cluster: Cluster, dag: DataDAG, tid: str,
    placement: Dict[str, TaskRecord], pid: int,
) -> Tuple[float, bool]:
    """
    Earliest start time of `tid` on `pid` given:
      - parents in this DAG already placed (in `placement`)
      - data locality from `cluster.data_residency`
    Returns (start_time, data_local_flag).
    """
    task = dag.tasks[tid]
    earliest = cluster.free_at(pid, 0.0)
    local = cluster.data_local(pid, task.input_partitions)
    for parent_tid in dag.parents(tid):
        prec = placement[parent_tid]
        if prec.pid == pid:
            arrive = prec.finish
        else:
            parent_task = dag.tasks[parent_tid]
            arrive = prec.finish + cluster.comm_cost(prec.pid, pid, parent_task.data_out_gb)
        earliest = max(earliest, arrive)
    # external locality penalty for partitions that aren't from in-DAG parents
    if not local and not dag.parents(tid):
        # synthetic input not produced by parent — pay a small fetch cost
        earliest += cluster.comm_cost(0, pid, task.data_in_gb) * 0.3
    return earliest, local


# --------------------------------------------------------------------------
# HEFT-LC (static baseline)
# --------------------------------------------------------------------------

class HEFTLC:
    """HEFT with load-aware tie-breaking. epsilon=0 recovers vanilla HEFT."""
    name = "HEFT-LC"

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def _upward_rank(self, cluster: Cluster, dag: DataDAG) -> Dict[str, float]:
        n = cluster.n
        rank: Dict[str, float] = {}
        for tid in reversed(dag.topo()):
            t = dag.tasks[tid]
            avg_cost = sum(t.per_proc_compute[p] / cluster.nodes[p].speed for p in range(n)) / n
            kids = dag.children(tid)
            if not kids:
                rank[tid] = avg_cost
            else:
                best = 0.0
                for c in kids:
                    avg_comm = t.data_out_gb / max(
                        sum(cluster.nodes[p].bandwidth_gbps for p in range(n)) / n, 1e-6
                    )
                    best = max(best, avg_comm + rank[c])
                rank[tid] = avg_cost + best
        return rank

    def schedule(self, cluster: Cluster, dag: DataDAG, now: float) -> List[TaskRecord]:
        rank = self._upward_rank(cluster, dag)
        order = sorted(dag.topo(), key=lambda t: -rank[t])
        placement: Dict[str, TaskRecord] = {}

        for tid in order:
            task = dag.tasks[tid]
            best_eft, best_pid, best_start, best_local = float("inf"), -1, 0.0, False
            candidates: List[Tuple[float, int, float, bool]] = []
            for p in range(cluster.n):
                if not cluster.is_up(p, now):
                    continue
                est, local = task_ready_time(cluster, dag, tid, placement, p)
                est = max(est, now)
                comp = task.per_proc_compute[p] / cluster.nodes[p].speed
                eft = est + comp
                candidates.append((eft, p, est, local))
                if eft < best_eft:
                    best_eft, best_pid, best_start, best_local = eft, p, est, local

            # load-aware tie-breaking
            tol = best_eft * (1.0 + self.epsilon)
            tied = [c for c in candidates if c[0] <= tol]
            if len(tied) > 1:
                loads = cluster.per_node_load()
                tied.sort(key=lambda c: (loads[c[1]], c[0]))
                best_eft, best_pid, best_start, best_local = tied[0]

            rec = TaskRecord(
                task_id=tid, pid=best_pid,
                start=best_start, finish=best_eft, data_local=best_local,
            )
            placement[tid] = rec
            cluster.commit(rec, output_partition=task.output_partition)

        return list(placement.values())


# --------------------------------------------------------------------------
# DLS — Data-Locality Scheduling
# --------------------------------------------------------------------------

class DLS:
    """HEFT-LC variant that prefers data-local placements when the locality
    bonus exceeds the speed penalty."""
    name = "DLS"

    def __init__(self, locality_weight: float = 1.5):
        self.locality_weight = locality_weight
        self._inner = HEFTLC(epsilon=0.05)

    def schedule(self, cluster: Cluster, dag: DataDAG, now: float) -> List[TaskRecord]:
        rank = self._inner._upward_rank(cluster, dag)
        order = sorted(dag.topo(), key=lambda t: -rank[t])
        placement: Dict[str, TaskRecord] = {}

        for tid in order:
            task = dag.tasks[tid]
            best_score, best_pid, best_start, best_eft, best_local = float("inf"), -1, 0.0, 0.0, False
            for p in range(cluster.n):
                if not cluster.is_up(p, now):
                    continue
                est, local = task_ready_time(cluster, dag, tid, placement, p)
                est = max(est, now)
                comp = task.per_proc_compute[p] / cluster.nodes[p].speed
                eft = est + comp
                # score = EFT, with locality bonus reducing effective cost
                score = eft - (self.locality_weight * task.data_in_gb if local else 0.0)
                if score < best_score:
                    best_score, best_pid, best_start, best_eft, best_local = (
                        score, p, est, eft, local
                    )
            rec = TaskRecord(
                task_id=tid, pid=best_pid,
                start=best_start, finish=best_eft, data_local=best_local,
            )
            placement[tid] = rec
            cluster.commit(rec, output_partition=task.output_partition)
        return list(placement.values())


# --------------------------------------------------------------------------
# RLScheduler — wraps a learned per-task policy
# --------------------------------------------------------------------------

class RLScheduler:
    """Reinforcement-learning scheduler. Picks placement via `agent.act(state)`.

    The agent observes a discretized state per task:
      (task_size_bucket, locality_pattern, load_pattern)
    and emits an action in {0..n_proc-1}.

    Reward (computed by the simulator after the DAG completes) is the
    normalized makespan improvement vs HEFT-LC.
    """
    name = "RL"

    def __init__(self, agent, fallback: Optional[HEFTLC] = None):
        self.agent = agent
        self.fallback = fallback or HEFTLC(epsilon=0.05)
        self._inner = HEFTLC(epsilon=0.0)  # for upward rank

    def schedule(self, cluster: Cluster, dag: DataDAG, now: float) -> List[TaskRecord]:
        rank = self._inner._upward_rank(cluster, dag)
        order = sorted(dag.topo(), key=lambda t: -rank[t])
        placement: Dict[str, TaskRecord] = {}
        states_actions: List[Tuple[tuple, int, float]] = []

        for tid in order:
            task = dag.tasks[tid]
            # Build candidate options (EFT, est, local) for every up processor
            options = []
            for p in range(cluster.n):
                if not cluster.is_up(p, now):
                    continue
                est, local = task_ready_time(cluster, dag, tid, placement, p)
                est = max(est, now)
                comp = task.per_proc_compute[p] / cluster.nodes[p].speed
                eft = est + comp
                options.append((p, est, eft, local))

            state = self.agent.encode_state(cluster, task, options)
            valid_pids = [opt[0] for opt in options]
            action = self.agent.act(state, valid_pids)
            # Find chosen option
            chosen = next(opt for opt in options if opt[0] == action)
            p, est, eft, local = chosen
            rec = TaskRecord(
                task_id=tid, pid=p, start=est, finish=eft, data_local=local,
            )
            placement[tid] = rec
            cluster.commit(rec, output_partition=task.output_partition)
            states_actions.append((state, action, eft))

        # Hand the transition log back to the agent for online learning
        self.agent.observe_episode(dag, states_actions, placement)
        return list(placement.values())
