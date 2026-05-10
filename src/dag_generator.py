"""
Data-aware DAG generator for streaming workflows.

A `DataDAG` is a DAG where each task carries:
  - per-processor compute cost (base seconds)
  - input data partition IDs (drive locality decisions)
  - output partition ID (becomes input to children)
  - data volume (GB) for communication cost

A `WorkflowStream` emits DataDAGs over time following a Poisson arrival process,
mimicking a multi-tenant data platform that receives jobs continuously.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import networkx as nx


@dataclass
class DataTask:
    tid: str
    base_compute: float                  # base seconds (divided by node speed)
    per_proc_compute: Dict[int, float]   # heterogeneous compute cost per pid
    input_partitions: List[str]          # data IDs this task reads
    output_partition: str                # data ID this task produces
    data_in_gb: float                    # input volume
    data_out_gb: float                   # output volume


class DataDAG:
    def __init__(self, dag_id: str):
        self.dag_id = dag_id
        self.g = nx.DiGraph()
        self.tasks: Dict[str, DataTask] = {}
        self.arrival_time: float = 0.0

    def add_task(self, t: DataTask):
        self.tasks[t.tid] = t
        self.g.add_node(t.tid)

    def add_edge(self, src: str, dst: str):
        self.g.add_edge(src, dst)

    def topo(self) -> List[str]:
        return list(nx.topological_sort(self.g))

    def parents(self, tid: str) -> List[str]:
        return list(self.g.predecessors(tid))

    def children(self, tid: str) -> List[str]:
        return list(self.g.successors(tid))

    def entry(self) -> List[str]:
        return [n for n in self.g if self.g.in_degree(n) == 0]

    def exit(self) -> List[str]:
        return [n for n in self.g if self.g.out_degree(n) == 0]

    def average_compute(self) -> float:
        return sum(t.base_compute for t in self.tasks.values()) / max(len(self.tasks), 1)


# ---------- Generators ----------

def _heterogeneous_costs(base: float, n_proc: int, beta: float, rng: random.Random) -> Dict[int, float]:
    """Per-processor cost: base * (1 + beta * U(-1, 1))."""
    return {p: max(0.1, base * (1.0 + beta * rng.uniform(-1.0, 1.0))) for p in range(n_proc)}


def generate_mapreduce_dag(
    dag_id: str, n_mappers: int, n_reducers: int, scale: float,
    n_proc: int, beta: float, seed: int,
) -> DataDAG:
    """
    MapReduce-shaped DAG: source -> M mappers -> R reducers -> sink.
    Mappers each read one input partition, reducers shuffle from all mappers.
    """
    rng = random.Random(seed)
    d = DataDAG(dag_id)

    src_compute = 1.0 * scale
    d.add_task(DataTask(
        tid=f"{dag_id}/src",
        base_compute=src_compute,
        per_proc_compute=_heterogeneous_costs(src_compute, n_proc, beta, rng),
        input_partitions=[],
        output_partition=f"{dag_id}/src.out",
        data_in_gb=0.0,
        data_out_gb=0.5 * scale,
    ))

    for i in range(n_mappers):
        c = 5.0 * scale * rng.uniform(0.8, 1.4)
        tid = f"{dag_id}/m{i}"
        d.add_task(DataTask(
            tid=tid, base_compute=c,
            per_proc_compute=_heterogeneous_costs(c, n_proc, beta, rng),
            input_partitions=[f"{dag_id}/src.out"],
            output_partition=f"{dag_id}/m{i}.out",
            data_in_gb=0.5 * scale / n_mappers,
            data_out_gb=1.0 * scale / n_mappers,
        ))
        d.add_edge(f"{dag_id}/src", tid)

    for j in range(n_reducers):
        c = 8.0 * scale * rng.uniform(0.8, 1.4)
        tid = f"{dag_id}/r{j}"
        d.add_task(DataTask(
            tid=tid, base_compute=c,
            per_proc_compute=_heterogeneous_costs(c, n_proc, beta, rng),
            input_partitions=[f"{dag_id}/m{i}.out" for i in range(n_mappers)],
            output_partition=f"{dag_id}/r{j}.out",
            data_in_gb=1.0 * scale,
            data_out_gb=0.3 * scale,
        ))
        for i in range(n_mappers):
            d.add_edge(f"{dag_id}/m{i}", tid)

    sink_compute = 1.0 * scale
    d.add_task(DataTask(
        tid=f"{dag_id}/sink", base_compute=sink_compute,
        per_proc_compute=_heterogeneous_costs(sink_compute, n_proc, beta, rng),
        input_partitions=[f"{dag_id}/r{j}.out" for j in range(n_reducers)],
        output_partition=f"{dag_id}/sink.out",
        data_in_gb=0.3 * scale * n_reducers,
        data_out_gb=0.1 * scale,
    ))
    for j in range(n_reducers):
        d.add_edge(f"{dag_id}/r{j}", f"{dag_id}/sink")

    return d


def generate_etl_dag(
    dag_id: str, n_stages: int, fanout: int, scale: float,
    n_proc: int, beta: float, seed: int,
) -> DataDAG:
    """Pipeline-shaped DAG with branching: extract -> transform^k -> load."""
    rng = random.Random(seed)
    d = DataDAG(dag_id)
    prev = [f"{dag_id}/extract"]
    c0 = 2.0 * scale
    d.add_task(DataTask(
        tid=prev[0], base_compute=c0,
        per_proc_compute=_heterogeneous_costs(c0, n_proc, beta, rng),
        input_partitions=[], output_partition=f"{prev[0]}.out",
        data_in_gb=0.0, data_out_gb=1.0 * scale,
    ))
    for k in range(n_stages):
        new_layer = []
        for i in range(fanout):
            tid = f"{dag_id}/t{k}_{i}"
            c = 3.0 * scale * rng.uniform(0.7, 1.3)
            d.add_task(DataTask(
                tid=tid, base_compute=c,
                per_proc_compute=_heterogeneous_costs(c, n_proc, beta, rng),
                input_partitions=[f"{p}.out" for p in prev],
                output_partition=f"{tid}.out",
                data_in_gb=0.8 * scale / fanout,
                data_out_gb=0.6 * scale / fanout,
            ))
            for p in prev:
                d.add_edge(p, tid)
            new_layer.append(tid)
        prev = new_layer
    sink = f"{dag_id}/load"
    cs = 1.5 * scale
    d.add_task(DataTask(
        tid=sink, base_compute=cs,
        per_proc_compute=_heterogeneous_costs(cs, n_proc, beta, rng),
        input_partitions=[f"{p}.out" for p in prev],
        output_partition=f"{sink}.out",
        data_in_gb=0.6 * scale * len(prev), data_out_gb=0.2 * scale,
    ))
    for p in prev:
        d.add_edge(p, sink)
    return d


def generate_random_dag(
    dag_id: str, n_tasks: int, edge_p: float, scale: float,
    n_proc: int, beta: float, seed: int,
) -> DataDAG:
    rng = random.Random(seed)
    d = DataDAG(dag_id)
    for i in range(n_tasks):
        c = scale * rng.uniform(1.0, 8.0)
        d.add_task(DataTask(
            tid=f"{dag_id}/n{i}", base_compute=c,
            per_proc_compute=_heterogeneous_costs(c, n_proc, beta, rng),
            input_partitions=[], output_partition=f"{dag_id}/n{i}.out",
            data_in_gb=scale * rng.uniform(0.1, 1.5),
            data_out_gb=scale * rng.uniform(0.1, 1.5),
        ))
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if rng.random() < edge_p:
                d.add_edge(f"{dag_id}/n{i}", f"{dag_id}/n{j}")
                # parent's output becomes child's input
                d.tasks[f"{dag_id}/n{j}"].input_partitions.append(f"{dag_id}/n{i}.out")
    return d


# ---------- Streaming workload ----------

@dataclass
class WorkflowStream:
    """Poisson stream of mixed DataDAGs over a horizon."""
    horizon: float
    arrival_rate: float        # DAGs per second
    n_proc: int
    seed: int = 0
    mix: Tuple[float, float, float] = (0.5, 0.3, 0.2)   # MR, ETL, Random

    def __iter__(self) -> Iterator[DataDAG]:
        rng = random.Random(self.seed)
        t = 0.0
        idx = 0
        while True:
            t += rng.expovariate(self.arrival_rate)
            if t >= self.horizon:
                return
            r = rng.random()
            if r < self.mix[0]:
                d = generate_mapreduce_dag(
                    f"job{idx:04d}", n_mappers=rng.randint(2, 5),
                    n_reducers=rng.randint(2, 4),
                    scale=rng.uniform(0.5, 2.0),
                    n_proc=self.n_proc, beta=0.4, seed=self.seed + idx,
                )
            elif r < self.mix[0] + self.mix[1]:
                d = generate_etl_dag(
                    f"job{idx:04d}", n_stages=rng.randint(2, 4),
                    fanout=rng.randint(2, 4),
                    scale=rng.uniform(0.5, 2.0),
                    n_proc=self.n_proc, beta=0.4, seed=self.seed + idx,
                )
            else:
                d = generate_random_dag(
                    f"job{idx:04d}", n_tasks=rng.randint(8, 16),
                    edge_p=0.3,
                    scale=rng.uniform(0.5, 2.0),
                    n_proc=self.n_proc, beta=0.4, seed=self.seed + idx,
                )
            d.arrival_time = t
            idx += 1
            yield d
