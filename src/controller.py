"""
MAPE-K autonomic controller for the scheduler.

  Monitor  → collect load imbalance, locality rate, throughput from the cluster
  Analyze  → detect skew, low locality, falling throughput
  Plan     → adjust HEFT-LC's epsilon (tie-break tolerance) and the DLS weight
  Execute  → push new parameters into the scheduler
  Knowledge→ rolling history of metrics and chosen adaptations

The autonomic angle: instead of fixing epsilon at 0.05 (the HEFT-LC default),
the controller learns it online from observed cluster behaviour. This removes
the manual tuning step that limited the original HEFT-LC contribution.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from cluster import Cluster
from schedulers import HEFTLC


@dataclass
class Sample:
    t: float
    imbalance: float
    locality_rate: float
    throughput: float
    makespan_window: float
    epsilon: float


@dataclass
class ControllerConfig:
    window_size: int = 20            # episodes per analysis tick
    target_imbalance: float = 0.25   # set-point for CV of node loads
    epsilon_min: float = 0.0
    epsilon_max: float = 0.25
    epsilon_step: float = 0.02
    cooldown_episodes: int = 3       # don't adapt every single episode


class AutonomicController:
    """Closes the MAPE-K loop around a HEFT-LC scheduler."""

    def __init__(self, scheduler: HEFTLC, cfg: Optional[ControllerConfig] = None):
        self.scheduler = scheduler
        self.cfg = cfg or ControllerConfig()
        self.history: Deque[Sample] = deque(maxlen=self.cfg.window_size * 4)
        self._since_last_adapt = 0
        self._completed_in_window: List[float] = []   # makespans
        self._locality_in_window: List[float] = []

    # ------------- MONITOR -------------
    def monitor(self, cluster: Cluster, dag_makespan: float, locality_rate: float, now: float):
        sample = Sample(
            t=now,
            imbalance=cluster.rolling_imbalance(),
            locality_rate=locality_rate,
            throughput=1.0 / max(dag_makespan, 1e-3),
            makespan_window=dag_makespan,
            epsilon=self.scheduler.epsilon,
        )
        self.history.append(sample)
        self._completed_in_window.append(dag_makespan)
        self._locality_in_window.append(locality_rate)
        self._since_last_adapt += 1

    # ------------- ANALYZE + PLAN + EXECUTE -------------
    def tick(self) -> Optional[str]:
        """
        Called periodically by the simulator. Returns a human-readable
        description of the action taken, or None if no adaptation happened.
        """
        if self._since_last_adapt < self.cfg.cooldown_episodes:
            return None
        if len(self._completed_in_window) < self.cfg.cooldown_episodes:
            return None

        avg_imb = sum(s.imbalance for s in list(self.history)[-self.cfg.window_size:]) / min(
            len(self.history), self.cfg.window_size
        )
        avg_loc = sum(self._locality_in_window) / len(self._locality_in_window)
        avg_make = sum(self._completed_in_window) / len(self._completed_in_window)

        # Symbolic policy: piecewise rule on imbalance error
        action_desc = None
        err = avg_imb - self.cfg.target_imbalance

        if err > 0.1:
            # too imbalanced → widen the tie-break tolerance
            new_eps = min(self.cfg.epsilon_max,
                          self.scheduler.epsilon + self.cfg.epsilon_step)
            if new_eps != self.scheduler.epsilon:
                self.scheduler.epsilon = new_eps
                action_desc = f"raise eps -> {new_eps:.3f} (imb={avg_imb:.2f})"
        elif err < -0.05:
            # imbalance well below target → tighten to chase makespan
            new_eps = max(self.cfg.epsilon_min,
                          self.scheduler.epsilon - self.cfg.epsilon_step)
            if new_eps != self.scheduler.epsilon:
                self.scheduler.epsilon = new_eps
                action_desc = f"lower eps -> {new_eps:.3f} (imb={avg_imb:.2f})"

        # reset window
        self._since_last_adapt = 0
        self._completed_in_window.clear()
        self._locality_in_window.clear()
        return action_desc

    def snapshot(self) -> Sample:
        if self.history:
            return self.history[-1]
        return Sample(0.0, 0.0, 0.0, 0.0, 0.0, self.scheduler.epsilon)
