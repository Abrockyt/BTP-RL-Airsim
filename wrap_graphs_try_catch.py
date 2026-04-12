import re
with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    text = f.read()

new_update = """    def update_graphs(self):
        try:
            # Separate history by flight mode
            off_history = [run for run in getattr(self, 'run_history', []) if run.get('flight_mode') == "NORMAL"]
            on_history = [run for run in getattr(self, 'run_history', []) if run.get('flight_mode') == "WIND"]

            def compute_metrics(history):
                if not history:
                    return [], [], []
                import numpy as np
                episodes = np.arange(1, len(history) + 1)
                success_flags = np.array([1 if r.get('goal_reached', False) else 0 for r in history])
                rewards = np.array([r.get('reward', 0) for r in history])

                success_rate = np.cumsum(success_flags) / episodes
                avg_reward = np.cumsum(rewards) / episodes
                return episodes, success_rate, avg_reward

            ep_off, sr_off, rew_off = compute_metrics(off_history)
            ep_on, sr_on, rew_on = compute_metrics(on_history)

            if len(ep_off) > 0:
                self.line_sr_off.set_data(ep_off, sr_off)
                self.line_rew_off.set_data(ep_off, rew_off)
                self.ax1.set_xlim(1, max(2, len(ep_off)))
                self.ax1.relim()
                self.ax1.autoscale_view(scalex=False, scaley=True)
                self.ax2.set_xlim(1, max(2, len(ep_off)))
                self.ax2.relim()
                self.ax2.autoscale_view(scalex=False, scaley=True)

            if len(ep_on) > 0:
                self.line_sr_on.set_data(ep_on, sr_on)
                self.line_rew_on.set_data(ep_on, rew_on)
                self.ax3.set_xlim(1, max(2, len(ep_on)))
                self.ax3.relim()
                self.ax3.autoscale_view(scalex=False, scaley=True)
                self.ax4.set_xlim(1, max(2, len(ep_on)))
                self.ax4.relim()
                self.ax4.autoscale_view(scalex=False, scaley=True)

            # Update Map
            if hasattr(self, 'map_goal_pos'):
                self.map_goal_pos.set_data([self.goal_x], [self.goal_y])

            if hasattr(self, 'metrics') and len(self.metrics.get('positions_x', [])) > 0:
                self.map_drone_pos.set_data([self.metrics['positions_x'][-1]], [self.metrics['positions_y'][-1]])
                self.map_drone_path.set_data(self.metrics['positions_x'], self.metrics['positions_y'])

            self.canvas.draw()
        except Exception as e:
            print(f"Graph update error completely bypassed: {e}")"""

text = re.sub(r'    def update_graphs\(self\):.*?self\.canvas\.draw\(\)', new_update, text, flags=re.DOTALL)

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(text)
