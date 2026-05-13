"""
Event-driven simulator for streaming workflows.

Drives a `WorkflowStream` against a cluster + scheduler (and optional
autonomic controller). Returns per-DAG records: arrival, completion,
makespan, imbalance after, locality rate, scheduler decisions.

Failure model: at each DAG arrival, with probability `failure_rate`,
one random node is brought down for `failure_recovery` seconds. Tasks
already committed to that node finish (we model a soft failure that
prevents future scheduling but doesn't roll back, mirroring a node
being marked unavailable in Kubernetes/YARN).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from cluster import Cluster
from controller import AutonomicController
from dag_generator import DataDAG, WorkflowStream


@dataclass
class DAGResult:
    dag_id: str
    arrival: float
    start: float
    finish: float
    makespan: float
    n_tasks: int
    imbalance_after: float
    locality_rate: float
    epsilon_in_effect: float
    scheduler: str


@dataclass
class StreamReport:
    scheduler_name: str
    results: List[DAGResult] = field(default_factory=list)
    adaptations: List = field(default_factory=list)
    failures: List = field(default_factory=list)

    def mean_makespan(self) -> float:
        return sum(r.makespan for r in self.results) / max(len(self.results), 1)

    def mean_imbalance(self) -> float:
        return sum(r.imbalance_after for r in self.results) / max(len(self.results), 1)

    def mean_locality(self) -> float:
        return sum(r.locality_rate for r in self.results) / max(len(self.results), 1)

    def throughput(self, horizon: float) -> float:
        return len(self.results) / max(horizon, 1e-6)


def simulate(
    stream: WorkflowStream,
    cluster: Cluster,
    scheduler,
    controller: Optional[AutonomicController] = None,
    failure_rate: float = 0.0,
    failure_recovery: float = 30.0,
    seed: int = 0,
    verbose: bool = False,
) -> StreamReport:
    rng = random.Random(seed)
    report = StreamReport(scheduler_name=scheduler.name)

    for dag in stream:
        now = dag.arrival_time

        # Optional failure injection
        if failure_rate > 0 and rng.random() < failure_rate:
            victim = rng.randrange(cluster.n)
            cluster.inject_failure(victim, now, recovery_seconds=failure_recovery)
            report.failures.append((now, victim))

        # Schedule the DAG
        records = scheduler.schedule(cluster, dag, now)
        starts = [r.start for r in records]
        finishes = [r.finish for r in records]
        ms = max(finishes) - min(starts) if records else 0.0
        loc = sum(1 for r in records if r.data_local) / max(len(records), 1)

        eps = getattr(scheduler, "epsilon", 0.0)
        result = DAGResult(
            dag_id=dag.dag_id,
            arrival=now,
            start=min(starts) if starts else now,
            finish=max(finishes) if finishes else now,
            makespan=ms,
            n_tasks=len(records),
            imbalance_after=cluster.load_imbalance(),
            locality_rate=loc,
            epsilon_in_effect=eps,
            scheduler=scheduler.name,
        )
        report.results.append(result)

        # Feed the autonomic loop
        if controller is not None:
            controller.monitor(cluster, ms, loc, now)
            action = controller.tick()
            if action is not None:
                report.adaptations.append((now, action))
                if verbose:
                    print(f"  [t={now:7.1f}] {action}")

    return report
