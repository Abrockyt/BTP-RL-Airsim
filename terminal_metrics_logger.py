"""
A pure Python, terminal-only metrics logger for reinforcement learning simulations.
This module provides a class to silently collect telemetry during a run and
generate a comprehensive, well-formatted ASCII table of academic metrics
only after the simulation completes.
"""

import numpy as np
from collections import deque

class TerminalMetricsLogger:
    """
    Tracks advanced academic metrics for RL simulations with zero-overhead logging.
    """
    def __init__(self, max_steps=10000):
        """
        Initializes the logger with data storage.
        """
        self.positions = deque(maxlen=max_steps)
        self.target_waypoints = deque(maxlen=max_steps)
        self.energy_drained_steps = deque(maxlen=max_steps)
        self.inference_latencies_ms = deque(maxlen=max_steps)
        self.step_rewards = deque(maxlen=max_steps)

    def log_step(self, position, target_waypoint, energy_drained_this_step, inference_latency_ms, step_reward):
        """
        Appends raw data to memory. This is a low-overhead operation.
        """
        self.positions.append(np.array(position, dtype=np.float32))
        self.target_waypoints.append(np.array(target_waypoint, dtype=np.float32))
        self.energy_drained_steps.append(float(energy_drained_this_step))
        self.inference_latencies_ms.append(float(inference_latency_ms))
        self.step_rewards.append(float(step_reward))

    def generate_final_report(self):
        """
        Calculates and prints the final metrics in a formatted ASCII table.
        """
        if not self.positions:
            print("No data logged. Cannot generate report.")
            return

        # Convert to numpy arrays for vectorized calculations
        positions_arr = np.array(self.positions)
        target_waypoints_arr = np.array(self.target_waypoints)
        
        # 1. Total Distance Traveled
        distances = np.linalg.norm(np.diff(positions_arr, axis=0), axis=1)
        total_distance_traveled = np.sum(distances)

        # 2. Total Energy Drained
        total_energy_drained = np.sum(self.energy_drained_steps)

        # 3. Average Cost of Transport (CoT)
        if total_distance_traveled > 0:
            avg_cot = total_energy_drained / total_distance_traveled
        else:
            avg_cot = 0.0

        # 4. Average Cross-Track Error (CTE)
        cte_errors = np.linalg.norm(positions_arr - target_waypoints_arr, axis=1)
        avg_cte = np.mean(cte_errors)

        # 5. Mean Inference Latency
        mean_inference_latency = np.mean(self.inference_latencies_ms)

        # 6. Average Step Reward
        avg_step_reward = np.mean(self.step_rewards)

        # --- Print Report ---
        print("\n" + "="*60)
        print("           ACADEMIC PERFORMANCE METRICS REPORT")
        print("="*60)
        print(f"{'Metric':<30} | {'Value':<25}")
        print("-"*60)
        print(f"{'Average Cost of Transport (CoT)':<30} | {avg_cot:,.4f} Wh/m")
        print(f"{'Average Cross-Track Error (CTE)':<30} | {avg_cte:,.4f} m")
        print(f"{'Mean Inference Latency':<30} | {mean_inference_latency:,.4f} ms")
        print(f"{'Average Step Reward':<30} | {avg_step_reward:,.4f}")
        print("="*60)

if __name__ == '__main__':
    # Example Usage
    logger = TerminalMetricsLogger()
    
    # Simulate a few steps
    for i in range(10):
        logger.log_step(
            position=np.array([i, i*0.5, -10]),
            target_waypoint=np.array([i, i*0.5 + (np.random.rand()-0.5), -10]),
            energy_drained_this_step=0.05 + np.random.rand() * 0.01,
            inference_latency_ms=15 + np.random.rand() * 5,
            step_reward=1.0 - np.random.rand() * 0.2
        )
    
    logger.generate_final_report()
