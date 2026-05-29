"""
Generate publication-quality figures from data/results.csv and aux CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

COLORS = {"HEFT-LC": "#1f77b4", "DLS": "#2ca02c", "RL": "#d62728"}


def save(fig, name):
    out_png = FIG / f"{name}.png"
    out_pdf = FIG / f"{name}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name} + .pdf")


def fig1_scheduler_comparison():
    df = pd.read_csv(DATA / "results.csv")
    a = df[df.scenario == "A_compare"].copy()
    agg = a.groupby(["scheduler", "cluster"]).agg(
        makespan=("mean_makespan", "mean"),
        imbalance=("mean_imbalance", "mean"),
        locality=("mean_locality", "mean"),
    ).reset_index()

    clusters = ["homogeneous", "cpu_gpu", "edge_cloud", "skewed_8"]
    schedulers = ["HEFT-LC", "DLS", "RL"]
    x = np.arange(len(clusters))
    w = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, (metric, ax, title) in enumerate(zip(
        ["makespan", "imbalance", "locality"], axes,
        ["Mean Per-DAG Makespan (s)", "Mean Load Imbalance (CV)", "Data Locality Rate"],
    )):
        for j, s in enumerate(schedulers):
            vals = [
                agg[(agg.scheduler == s) & (agg.cluster == c)][metric].values[0]
                if not agg[(agg.scheduler == s) & (agg.cluster == c)].empty else 0.0
                for c in clusters
            ]
            ax.bar(x + (j - 1) * w, vals, w, label=s, color=COLORS[s])
        ax.set_xticks(x)
        ax.set_xticklabels(clusters, rotation=20)
        ax.set_title(title)
        if i == 0:
            ax.legend(loc="upper left")
    fig.suptitle("Scheduler comparison across cluster topologies", y=1.02)
    save(fig, "fig1_scheduler_comparison")


def fig2_failure_robustness():
    df = pd.read_csv(DATA / "results.csv")
    c = df[df.scenario == "C_failure"].copy()
    agg = c.groupby(["scheduler", "failure_rate"]).agg(
        makespan=("mean_makespan", "mean"),
        makespan_std=("mean_makespan", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in ["HEFT-LC", "DLS", "RL"]:
        sub = agg[agg.scheduler == s]
        ax.errorbar(sub.failure_rate, sub.makespan, yerr=sub.makespan_std,
                    marker="o", capsize=4, label=s, color=COLORS[s])
    ax.set_xlabel("Failure Rate per DAG arrival")
    ax.set_ylabel("Mean per-DAG Makespan (s)")
    ax.set_title("Robustness under node failures (8-node skewed cluster)")
    ax.legend()
    save(fig, "fig2_failure_robustness")


def fig3_autonomic_controller():
    df = pd.read_csv(DATA / "results.csv")
    d = df[df.scenario == "D_autonomic"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, metric, title in zip(
        axes, ["mean_makespan", "mean_imbalance"],
        ["Mean Makespan (s)", "Mean Load Imbalance (CV)"],
    ):
        clusters = ["homogeneous", "skewed_8"]
        x = np.arange(len(clusters))
        w = 0.35
        off = [d[(d.cluster == c) & (d.controller == "off")][metric].mean() for c in clusters]
        on  = [d[(d.cluster == c) & (d.controller == "on")][metric].mean() for c in clusters]
        ax.bar(x - w/2, off, w, label="static ε=0.05", color="#888888")
        ax.bar(x + w/2, on,  w, label="autonomic", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(clusters)
        ax.set_title(title); ax.legend()
    fig.suptitle("MAPE-K controller: self-tuning ε vs fixed ε", y=1.02)
    save(fig, "fig3_autonomic_controller")


def fig4_adaptation_trace():
    ad = pd.read_csv(DATA / "adaptations.csv")
    if ad.empty:
        return
    # parse epsilon out of "raise eps -> 0.090 (imb=0.49)"
    def eps(s):
        try:
            return float(s.split("->")[1].split("(")[0].strip())
        except Exception:
            return np.nan
    ad["eps"] = ad.action.apply(eps)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for (cluster, seed), sub in ad.groupby(["cluster", "seed"]):
        if cluster != "skewed_8":
            continue
        ax.step(sub["time"], sub["eps"], where="post", alpha=0.7,
                label=f"seed={seed}")
    ax.axhline(0.05, color="grey", linestyle="--", label="static default")
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("ε (HEFT-LC tie-break tolerance)")
    ax.set_title("Autonomic adaptation trace (skewed_8 cluster)")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig4_adaptation_trace")


def fig5_rl_learning_curve():
    lc = pd.read_csv(DATA / "learning_curve.csv")
    if lc.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))

    # Smoothed ratio over episodes
    ax = axes[0]
    for seed, sub in lc.groupby("seed"):
        smooth = sub.ratio.rolling(20, min_periods=1).mean()
        ax.plot(sub.episode, smooth, label=f"seed={seed}", alpha=0.85)
    ax.axhline(1.0, color="grey", linestyle="--", label="HEFT-LC parity")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean makespan ratio (RL / HEFT-LC)")
    ax.set_title("Q-learning convergence")
    ax.legend()

    # Locality rate over episodes
    ax = axes[1]
    for seed, sub in lc.groupby("seed"):
        smooth = sub.locality.rolling(20, min_periods=1).mean()
        ax.plot(sub.episode, smooth, label=f"seed={seed}", alpha=0.85)
    ax.set_xlabel("Episode"); ax.set_ylabel("Data locality rate")
    ax.set_title("Locality learned over time")
    ax.legend()
    fig.suptitle("RL agent learning curves", y=1.02)
    save(fig, "fig5_rl_learning_curve")


def fig7_shift_timeline():
    """Time-series of epsilon and makespan during a non-stationary stream."""
    st = pd.read_csv(DATA / "shift_timeline.csv")
    if st.empty:
        return
    sub = st[(st.cluster == "skewed_8") & (st.seed == 1)].copy()

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)

    # Top: epsilon trajectory for the autonomic config
    ax = axes[0]
    for sched, color in [("HEFT", "#9467bd"), ("HEFT-LC", "#888888"), ("HEFT-LC+ctrl", "#d62728")]:
        s = sub[sub.scheduler == sched].sort_values("arrival")
        ax.step(s["arrival"], s["epsilon"], where="post", label=sched,
                color=color, linewidth=1.6)
    # phase shading
    ax.axvspan(0, 300, alpha=0.08, color="green", label="light")
    ax.axvspan(300, 600, alpha=0.10, color="red", label="burst")
    ax.axvspan(600, 900, alpha=0.08, color="blue", label="mr")
    ax.set_ylabel(r"$\varepsilon$ in effect")
    ax.set_title("Online ε adaptation across workload phases")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # Bottom: smoothed makespan per scheduler
    ax = axes[1]
    for sched, color in [("HEFT", "#9467bd"), ("HEFT-LC", "#888888"), ("HEFT-LC+ctrl", "#d62728")]:
        s = sub[sub.scheduler == sched].sort_values("arrival")
        smooth = s["makespan"].rolling(8, min_periods=1).mean()
        ax.plot(s["arrival"], smooth, label=sched, color=color, alpha=0.9, linewidth=1.4)
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Per-DAG makespan (smoothed)")
    ax.set_title("Per-DAG makespan over the non-stationary stream")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig7_shift_timeline")


def fig6_pareto():
    df = pd.read_csv(DATA / "results.csv")
    a = df[df.scenario == "A_compare"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in ["HEFT-LC", "DLS", "RL"]:
        sub = a[a.scheduler == s]
        ax.scatter(sub.mean_imbalance, sub.mean_makespan,
                   alpha=0.6, s=60, label=s, color=COLORS[s], edgecolors="k", linewidths=0.4)
    ax.set_xlabel("Mean Load Imbalance (CV)")
    ax.set_ylabel("Mean Per-DAG Makespan (s)")
    ax.set_title("Makespan / imbalance Pareto front across all runs")
    ax.legend()
    save(fig, "fig6_pareto")


def summary_table():
    df = pd.read_csv(DATA / "results.csv")
    a = df[df.scenario == "A_compare"]
    out = a.groupby("scheduler").agg(
        makespan=("mean_makespan", "mean"),
        imbalance=("mean_imbalance", "mean"),
        locality=("mean_locality", "mean"),
        throughput=("throughput", "mean"),
    ).round(3)
    print("\n=== Summary Table (Scenario A, across all cluster types and arrival rates) ===")
    print(out.to_string())
    print(f"\nTotal runs: {len(df)}")
    print(f"  Scenario A (comparison): {len(a)}")
    print(f"  Scenario C (failures):   {len(df[df.scenario=='C_failure'])}")
    print(f"  Scenario D (autonomic):  {len(df[df.scenario=='D_autonomic'])}")


def main():
    print("Generating figures...")
    fig1_scheduler_comparison()
    fig2_failure_robustness()
    fig3_autonomic_controller()
    fig4_adaptation_trace()
    fig5_rl_learning_curve()
    fig6_pareto()
    fig7_shift_timeline()
    summary_table()
    print(f"\nAll figures saved to {FIG}")


if __name__ == "__main__":
    main()
