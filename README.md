# AdaptiveFlow

**A self-tuning scheduler for streaming data workflows on heterogeneous clusters.**

Research project — M1 Informatique, Université Claude Bernard Lyon 1.
Targets the intersection of three Master's specializations: DiPaC (distributed scheduling), DataScale (data-aware pipelines), Autonomic Systems (MAPE-K control loop).

## Summary

A streaming-workflow scheduler that combines three angles:

| Component | Role |
|-----------|------|
| **HEFT-LC** | Static baseline — load-aware HEFT with tie-break tolerance ε |
| **DLS** | Data-Locality Scheduler — locality-aware variant |
| **RLScheduler** | Tabular Q-learning agent that learns per-task placements |
| **MAPE-K controller** | Autonomic loop that adjusts ε online from observed metrics |

Workflows arrive over time as a Poisson stream (50% MapReduce, 30% ETL, 20% random DAGs), each task carrying compute cost, input/output partitions and data volumes. The cluster models per-node bandwidth, data residency, and node failures.

## Key results

| Scheduler | Makespan (s) | Imbalance (CV) | Locality |
|-----------|-------------:|---------------:|---------:|
| HEFT-LC   | 23.5         | **0.021**      | 0.558    |
| DLS       | **17.4**     | 0.121          | **0.738** |
| RL        | 47.5         | 0.046          | 0.514    |

- **DLS reduces makespan by 26%** vs HEFT-LC by exploiting data locality (74% vs 56%).
- **MAPE-K controller** triggers up to 14 adaptations per stream and tracks the target imbalance with only a 3.5% makespan overhead — removing the manual tuning of ε.
- **RL agent** converges from 2.57× to 1.57× the HEFT-LC baseline within 350 episodes (39% improvement). Honest result: it does not beat the strong baseline; the paper discusses why (coarse tabular state).

## Structure

```
src/
  cluster.py          — heterogeneous cluster + data residency + failures
  dag_generator.py    — DataDAG model + WorkflowStream (Poisson arrivals)
  schedulers.py       — HEFT-LC, DLS, RLScheduler
  rl_agent.py         — tabular Q-learning agent (4-D discrete state)
  controller.py       — MAPE-K autonomic controller
  simulator.py        — event-driven streaming simulator
experiments/
  run_experiments.py  — 160 streaming runs across 4 scenarios
  generate_figures.py — 6 publication-quality figures
paper/
  paper.tex           — IEEE-format research paper (~7 pages)
data/
  results.csv         — main results table
  adaptations.csv     — controller adaptation trace
  learning_curve.csv  — per-episode RL convergence data
figures/              — fig1–fig6 (PNG + PDF)
```

## Quick start

```bash
pip install networkx numpy pandas matplotlib scipy seaborn
python experiments/run_experiments.py     # writes data/*.csv (~3 min)
python experiments/generate_figures.py    # writes figures/*.pdf + .png
```

To compile the paper:

```bash
cd paper && pdflatex paper.tex
```

## Reproducibility

Every result in the paper traces back to a row in `data/results.csv` or
`data/learning_curve.csv`. All RNG seeds are explicit. The benchmark runs
in under three minutes on a laptop CPU.

## Citation

```
Sidali. (2026). AdaptiveFlow: A Self-Tuning Scheduler for Streaming Data
Workflows on Heterogeneous Clusters. Research report, Université Lyon 1.
```
