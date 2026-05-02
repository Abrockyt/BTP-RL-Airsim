"""Terminal-only post simulation analyzer for UAV experiments.

The class in this module is intentionally lightweight during flight: update_step()
only stores raw samples in memory. All heavy calculations happen later in
generate_final_report(), which prints a comprehensive mathematical summary to
the terminal and returns the computed metrics as a dictionary.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np


class PostSimulationTerminalAnalyzer:
    """Collect raw UAV telemetry during flight and evaluate it after landing.

    Supported modes:
    - single: one drone, one stream of metrics
    - swarm: multiple drones, with optional swarm cohesion analysis
    - algo_comparison: compare metrics grouped by algorithm_name
    """

    VALID_MODES = {"single", "swarm", "algo_comparison"}

    def __init__(self, mode: str = "single") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")

        self.mode = mode
        self._samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    @staticmethod
    def _as_vector3(value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size != 3:
            raise ValueError("position and target_waypoint must be 3D vectors")
        return vector.astype(float, copy=True)

    def update_step(
        self,
        drone_id: str | int,
        position: np.ndarray,
        target_waypoint: np.ndarray,
        energy_level: float,
        inference_latency_ms: float,
        rl_step_reward: float,
        algorithm_name: str,
    ) -> None:
        """Store one raw telemetry sample.

        This method is intentionally non-verbose and performs no heavy math so it
        can be called inside a real-time AirSim loop without adding overhead.
        """

        drone_key = str(drone_id)
        sample = {
            "position": self._as_vector3(position),
            "target_waypoint": self._as_vector3(target_waypoint),
            "energy_level": float(energy_level),
            "inference_latency_ms": float(inference_latency_ms),
            "rl_step_reward": float(rl_step_reward),
            "algorithm_name": str(algorithm_name),
        }
        self._samples[drone_key].append(sample)

    @staticmethod
    def _per_drone_metrics(samples: List[Dict[str, Any]]) -> Dict[str, float]:
        positions = np.asarray([sample["position"] for sample in samples], dtype=float)
        targets = np.asarray([sample["target_waypoint"] for sample in samples], dtype=float)
        energy_levels = np.asarray([sample["energy_level"] for sample in samples], dtype=float)
        latencies = np.asarray([sample["inference_latency_ms"] for sample in samples], dtype=float)
        rewards = np.asarray([sample["rl_step_reward"] for sample in samples], dtype=float)

        if len(positions) > 1:
            step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = float(np.sum(step_distances))
        else:
            total_distance = 0.0

        if len(energy_levels) > 1:
            energy_drops = np.maximum(energy_levels[:-1] - energy_levels[1:], 0.0)
            total_energy_drained = float(np.sum(energy_drops))
        elif len(energy_levels) == 1:
            total_energy_drained = max(0.0, 100.0 - float(energy_levels[0]))
        else:
            total_energy_drained = 0.0

        cot = total_energy_drained / total_distance if total_distance > 1e-12 else float("inf")
        cte = float(np.mean(np.linalg.norm(positions - targets, axis=1))) if len(positions) else float("nan")
        mean_latency = float(np.mean(latencies)) if len(latencies) else float("nan")
        cumulative_reward = float(np.sum(rewards)) if len(rewards) else 0.0

        return {
            "total_distance_m": total_distance,
            "total_energy_drained": total_energy_drained,
            "cot": cot,
            "cte": cte,
            "mean_latency_ms": mean_latency,
            "cumulative_reward": cumulative_reward,
            "steps": float(len(samples)),
        }

    def _group_by_algorithm(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for drone_id, samples in self._samples.items():
            for sample in samples:
                grouped[sample["algorithm_name"]][drone_id].append(sample)
        return grouped

    @staticmethod
    def _format_value(value: float, unit: str = "") -> str:
        if value is None:
            return "N/A"
        if np.isnan(value):
            return "N/A"
        if np.isinf(value):
            return "inf"
        suffix = f" {unit}" if unit else ""
        return f"{value:.4f}{suffix}" if abs(value) < 100 else f"{value:.2f}{suffix}"

    def _swarm_cohesion_variance(self) -> Optional[float]:
        if self.mode != "swarm":
            return None

        if len(self._samples) < 2:
            return float("nan")

        max_steps = max(len(samples) for samples in self._samples.values())
        pairwise_distances: List[float] = []

        for step_index in range(max_steps):
            active_positions = []
            for samples in self._samples.values():
                if step_index < len(samples):
                    active_positions.append(samples[step_index]["position"])

            if len(active_positions) < 2:
                continue

            for left, right in combinations(active_positions, 2):
                pairwise_distances.append(float(np.linalg.norm(left - right)))

        if not pairwise_distances:
            return float("nan")

        return float(np.var(np.asarray(pairwise_distances, dtype=float)))

    def _print_block(self, title: str, metrics: Dict[str, float]) -> None:
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Total Distance Traveled: {self._format_value(metrics['total_distance_m'], 'm')}")
        print(f"Total Energy Drained:    {self._format_value(metrics['total_energy_drained'], 'Wh')}")
        print(f"Average CoT:             {self._format_value(metrics['cot'])}")
        print(f"Average CTE:             {self._format_value(metrics['cte'], 'm')}")
        print(f"Mean Inference Latency:  {self._format_value(metrics['mean_latency_ms'], 'ms')}")
        print(f"Avg Cumulative Reward:   {self._format_value(metrics['cumulative_reward'])}")
        print(f"Samples:                 {int(metrics['steps'])}")

    def generate_final_report(self) -> Dict[str, Any]:
        """Compute all derived metrics, print a terminal report, and return results."""

        report: Dict[str, Any] = {
            "mode": self.mode,
            "drones": {},
        }

        print("\n" + "=" * 72)
        print("POST-SIMULATION TERMINAL ANALYSIS")
        print("=" * 72)
        print(f"Mode: {self.mode}")
        print("All calculations below were generated after the simulation completed.")

        if self.mode == "algo_comparison":
            grouped = self._group_by_algorithm()
            report["algorithms"] = {}

            for algorithm_name, drone_groups in grouped.items():
                per_drone_metrics = {
                    drone_id: self._per_drone_metrics(samples)
                    for drone_id, samples in drone_groups.items()
                }

                valid_cot = [metrics["cot"] for metrics in per_drone_metrics.values() if np.isfinite(metrics["cot"])]
                cte_values = [metrics["cte"] for metrics in per_drone_metrics.values() if not np.isnan(metrics["cte"])]
                latency_values = []
                reward_values = [metrics["cumulative_reward"] for metrics in per_drone_metrics.values()]
                total_distance = float(np.sum([metrics["total_distance_m"] for metrics in per_drone_metrics.values()]))
                total_energy = float(np.sum([metrics["total_energy_drained"] for metrics in per_drone_metrics.values()]))

                for samples in drone_groups.values():
                    latency_values.extend(sample["inference_latency_ms"] for sample in samples)

                algo_metrics = {
                    "total_distance_m": total_distance,
                    "total_energy_drained": total_energy,
                    "cot": float(np.mean(valid_cot)) if valid_cot else float("nan"),
                    "cte": float(np.mean(cte_values)) if cte_values else float("nan"),
                    "mean_latency_ms": float(np.mean(latency_values)) if latency_values else float("nan"),
                    "cumulative_reward": float(np.mean(reward_values)) if reward_values else 0.0,
                    "steps": float(sum(len(samples) for samples in drone_groups.values())),
                }

                report["algorithms"][algorithm_name] = {
                    "per_drone": per_drone_metrics,
                    "summary": algo_metrics,
                }
                self._print_block(f"Algorithm: {algorithm_name}", algo_metrics)

            if self.mode == "swarm":
                cohesion_variance = self._swarm_cohesion_variance()
                report["swarm_cohesion_variance"] = cohesion_variance
                print(f"\nSwarm Cohesion Variance: {self._format_value(cohesion_variance)}")

            print("=" * 72)
            return report

        # single / swarm
        all_drone_metrics = {
            drone_id: self._per_drone_metrics(samples)
            for drone_id, samples in self._samples.items()
        }
        report["drones"] = all_drone_metrics

        if not all_drone_metrics:
            print("No samples were recorded.")
            print("=" * 72)
            return report

        for drone_id, metrics in all_drone_metrics.items():
            self._print_block(f"Drone: {drone_id}", metrics)

        aggregate = {
            "total_distance_m": float(np.sum([m["total_distance_m"] for m in all_drone_metrics.values()])),
            "total_energy_drained": float(np.sum([m["total_energy_drained"] for m in all_drone_metrics.values()])),
            "cot": float(np.mean([m["cot"] for m in all_drone_metrics.values() if np.isfinite(m["cot"]) ]))
            if any(np.isfinite(m["cot"]) for m in all_drone_metrics.values())
            else float("nan"),
            "cte": float(np.mean([m["cte"] for m in all_drone_metrics.values() if not np.isnan(m["cte"]) ]))
            if any(not np.isnan(m["cte"]) for m in all_drone_metrics.values())
            else float("nan"),
            "mean_latency_ms": float(np.mean([sample["inference_latency_ms"] for samples in self._samples.values() for sample in samples])),
            "cumulative_reward": float(np.mean([m["cumulative_reward"] for m in all_drone_metrics.values()])),
            "steps": float(sum(len(samples) for samples in self._samples.values())),
        }

        report["summary"] = aggregate
        print("\nAggregate Summary")
        print("-" * 17)
        print(f"Average CoT:             {self._format_value(aggregate['cot'])}")
        print(f"Average CTE:             {self._format_value(aggregate['cte'], 'm')}")
        print(f"Mean Inference Latency:  {self._format_value(aggregate['mean_latency_ms'], 'ms')}")
        print(f"Avg Cumulative Reward:   {self._format_value(aggregate['cumulative_reward'])}")

        if self.mode == "swarm":
            cohesion_variance = self._swarm_cohesion_variance()
            report["swarm_cohesion_variance"] = cohesion_variance
            print(f"Swarm Cohesion Variance: {self._format_value(cohesion_variance)}")

        print("=" * 72)
        return report


if __name__ == "__main__":
    analyzer = PostSimulationTerminalAnalyzer(mode="single")
    analyzer.update_step(
        drone_id="drone-1",
        position=np.array([0.0, 0.0, 0.0]),
        target_waypoint=np.array([10.0, 0.0, 0.0]),
        energy_level=100.0,
        inference_latency_ms=12.5,
        rl_step_reward=1.0,
        algorithm_name="Baseline-PPO",
    )
    analyzer.update_step(
        drone_id="drone-1",
        position=np.array([3.0, 4.0, 0.0]),
        target_waypoint=np.array([10.0, 0.0, 0.0]),
        energy_level=98.5,
        inference_latency_ms=11.2,
        rl_step_reward=1.4,
        algorithm_name="Baseline-PPO",
    )
    analyzer.generate_final_report()