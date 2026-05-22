"""
Run the full benchmark suite and emit data/results.csv.

Scenarios:
  A. Scheduler comparison on a stable stream (HEFT-LC, DLS, RL)
  B. Effect of cluster type (homogeneous / cpu_gpu / edge_cloud / skewed_8)
  C. Failure injection: throughput and makespan vs failure rate
  D. Autonomic controller on/off: imbalance and adaptation trace over time
  E. RL learning curve: per-episode makespan
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from cluster import ClusterFactory
from controller import AutonomicController, ControllerConfig
from dag_generator import WorkflowStream
from rl_agent import QLearningAgent
from schedulers import DLS, HEFTLC, RLScheduler
from simulator import simulate


DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
OUT = DATA / "results.csv"
LEARN = DATA / "learning_curve.csv"
ADAPT = DATA / "adaptations.csv"


def cluster_for(name: str, seed: int):
    return {
        "homogeneous": ClusterFactory.homogeneous_4,
        "cpu_gpu": ClusterFactory.cpu_gpu_mix,
        "edge_cloud": ClusterFactory.edge_cloud,
        "skewed_8": ClusterFactory.skewed_8,
    }[name](seed=seed)


def scheduler_for(name: str, n_proc: int, seed: int):
    if name == "HEFT-LC":
        return HEFTLC(epsilon=0.05)
    if name == "DLS":
        return DLS(locality_weight=1.5)
    if name == "RL":
        agent = QLearningAgent(n_proc=n_proc, seed=seed)
        return RLScheduler(agent=agent)
    raise ValueError(name)


def run_scenario_A_B(writer):
    schedulers = ["HEFT-LC", "DLS", "RL"]
    cluster_types = ["homogeneous", "cpu_gpu", "edge_cloud", "skewed_8"]
    arrival_rates = [0.05, 0.1, 0.2]
    seeds = [1, 2, 3]

    n = sum(1 for _ in arrival_rates) * len(seeds) * len(cluster_types) * len(schedulers)
    print(f"[A+B] {n} stream runs")

    done = 0
    for cluster_name in cluster_types:
        for sched_name in schedulers:
            for rate in arrival_rates:
                for seed in seeds:
                    cluster = cluster_for(cluster_name, seed)
                    scheduler = scheduler_for(sched_name, cluster.n, seed)
                    stream = WorkflowStream(
                        horizon=600.0, arrival_rate=rate,
                        n_proc=cluster.n, seed=seed,
                    )
                    t0 = time.time()
                    report = simulate(stream, cluster, scheduler, seed=seed)
                    elapsed = time.time() - t0

                    if not report.results:
                        continue
                    writer.writerow({
                        "scenario": "A_compare",
                        "scheduler": sched_name,
                        "cluster": cluster_name,
                        "arrival_rate": rate,
                        "failure_rate": 0.0,
                        "controller": "off",
                        "seed": seed,
                        "n_dags": len(report.results),
                        "mean_makespan": report.mean_makespan(),
                        "mean_imbalance": report.mean_imbalance(),
                        "mean_locality": report.mean_locality(),
                        "throughput": report.throughput(600.0),
                        "n_failures": len(report.failures),
                        "n_adaptations": 0,
                        "wall_seconds": elapsed,
                    })
                    done += 1
                    if done % 10 == 0:
                        print(f"  A+B progress: {done}/{n}")


def run_scenario_C_failures(writer):
    print("[C] failure-injection sweep")
    schedulers = ["HEFT-LC", "DLS", "RL"]
    failure_rates = [0.0, 0.05, 0.1, 0.2]
    seeds = [1, 2, 3]

    for sched_name in schedulers:
        for fr in failure_rates:
            for seed in seeds:
                cluster = cluster_for("skewed_8", seed)
                scheduler = scheduler_for(sched_name, cluster.n, seed)
                stream = WorkflowStream(
                    horizon=400.0, arrival_rate=0.15,
                    n_proc=cluster.n, seed=seed,
                )
                report = simulate(
                    stream, cluster, scheduler,
                    failure_rate=fr, failure_recovery=25.0, seed=seed,
                )
                if not report.results:
                    continue
                writer.writerow({
                    "scenario": "C_failure",
                    "scheduler": sched_name,
                    "cluster": "skewed_8",
                    "arrival_rate": 0.15,
                    "failure_rate": fr,
                    "controller": "off",
                    "seed": seed,
                    "n_dags": len(report.results),
                    "mean_makespan": report.mean_makespan(),
                    "mean_imbalance": report.mean_imbalance(),
                    "mean_locality": report.mean_locality(),
                    "throughput": report.throughput(400.0),
                    "n_failures": len(report.failures),
                    "n_adaptations": 0,
                    "wall_seconds": 0.0,
                })


def run_scenario_D_autonomic(writer, adapt_writer):
    print("[D] autonomic controller on/off")
    seeds = [1, 2, 3, 4]
    cluster_types = ["homogeneous", "skewed_8"]

    for cluster_name in cluster_types:
        for mode in ["off", "on"]:
            for seed in seeds:
                cluster = cluster_for(cluster_name, seed)
                scheduler = HEFTLC(epsilon=0.05)
                controller = None
                if mode == "on":
                    controller = AutonomicController(
                        scheduler,
                        ControllerConfig(
                            window_size=10,
                            target_imbalance=0.20,
                            epsilon_step=0.02,
                        ),
                    )
                stream = WorkflowStream(
                    horizon=800.0, arrival_rate=0.12,
                    n_proc=cluster.n, seed=seed,
                )
                report = simulate(stream, cluster, scheduler, controller=controller, seed=seed)
                if not report.results:
                    continue
                writer.writerow({
                    "scenario": "D_autonomic",
                    "scheduler": "HEFT-LC",
                    "cluster": cluster_name,
                    "arrival_rate": 0.12,
                    "failure_rate": 0.0,
                    "controller": mode,
                    "seed": seed,
                    "n_dags": len(report.results),
                    "mean_makespan": report.mean_makespan(),
                    "mean_imbalance": report.mean_imbalance(),
                    "mean_locality": report.mean_locality(),
                    "throughput": report.throughput(800.0),
                    "n_failures": 0,
                    "n_adaptations": len(report.adaptations),
                    "wall_seconds": 0.0,
                })

                # write adaptation trace
                for (t, desc) in report.adaptations:
                    adapt_writer.writerow({
                        "cluster": cluster_name, "seed": seed,
                        "time": t, "action": desc,
                    })


def run_scenario_E_learning(learn_writer):
    """
    Each 'episode' is an independent short stream against a fresh cluster.
    The agent is persistent across episodes (learns across them).
    Reports the makespan ratio vs a HEFT-LC oracle on the same stream.
    """
    print("[E] RL learning curve")
    for seed in [1, 2, 3]:
        agent = QLearningAgent(n_proc=8, seed=seed)
        rl = RLScheduler(agent=agent)
        for ep in range(400):
            ep_seed = seed * 10_000 + ep
            # Baseline first (HEFT-LC on the same stream, fresh cluster)
            base_cluster = ClusterFactory.skewed_8(seed=ep_seed)
            base_stream = WorkflowStream(
                horizon=200.0, arrival_rate=0.15,
                n_proc=base_cluster.n, seed=ep_seed,
            )
            base = simulate(base_stream, base_cluster, HEFTLC(0.05), seed=ep_seed)

            # RL on a replay of the same stream
            rl_cluster = ClusterFactory.skewed_8(seed=ep_seed)
            rl_stream = WorkflowStream(
                horizon=200.0, arrival_rate=0.15,
                n_proc=rl_cluster.n, seed=ep_seed,
            )
            rl_report = simulate(rl_stream, rl_cluster, rl, seed=ep_seed)

            if not base.results or not rl_report.results:
                continue
            ratio = rl_report.mean_makespan() / max(base.mean_makespan(), 1e-6)
            learn_writer.writerow({
                "seed": seed, "episode": ep,
                "makespan": rl_report.mean_makespan(),
                "baseline": base.mean_makespan(),
                "ratio": ratio,
                "locality": rl_report.mean_locality(),
                "epsilon": agent.epsilon,
            })
        print(f"  seed {seed}: final eps={agent.epsilon:.3f}, Q states={len(agent.q)}")


def main():
    fieldnames = [
        "scenario", "scheduler", "cluster", "arrival_rate", "failure_rate",
        "controller", "seed", "n_dags",
        "mean_makespan", "mean_imbalance", "mean_locality",
        "throughput", "n_failures", "n_adaptations", "wall_seconds",
    ]
    with OUT.open("w", newline="") as f, \
         ADAPT.open("w", newline="") as fa, \
         LEARN.open("w", newline="") as fl:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        wa = csv.DictWriter(fa, fieldnames=["cluster", "seed", "time", "action"])
        wa.writeheader()
        wl = csv.DictWriter(fl, fieldnames=["seed", "episode", "makespan", "baseline", "ratio", "locality", "epsilon"])
        wl.writeheader()

        run_scenario_A_B(w)
        run_scenario_C_failures(w)
        run_scenario_D_autonomic(w, wa)
        run_scenario_E_learning(wl)

    print(f"\nWrote {OUT}, {ADAPT}, {LEARN}")


if __name__ == "__main__":
    main()
