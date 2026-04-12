with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    orig = f.read()

if "def update_graphs(self):" not in orig:
    # Need to add update_graphs
    new_func = """
    def update_graphs(self):
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

        try:
            if len(ep_off) > 0:
                self.ax1.clear()
                self.ax1.plot(ep_off, sr_off, 'g-', label='Success Rate', marker='o')
                self.ax1.set_title('(a) Success Rate in Scenario A (Wind Off)', color='white')
                self.ax1.set_xlabel('Episodes', color='white')
                self.ax1.set_ylabel('Success Rate', color='white')
                self.ax1.grid(True, alpha=0.2)
                self.ax1.set_ylim(-0.05, 1.05)

                self.ax2.clear()
                self.ax2.plot(ep_off, rew_off, 'r-', label='Avg Reward', marker='o')
                self.ax2.set_title('(c) Average Reward in Scenario A (Wind Off)', color='white')
                self.ax2.set_xlabel('Episodes', color='white')
                self.ax2.set_ylabel('Avg Episodic Reward', color='white')
                self.ax2.grid(True, alpha=0.2)
                
            if len(ep_on) > 0:
                self.ax3.clear()
                self.ax3.plot(ep_on, sr_on, 'b-', label='Success Rate', marker='s')
                self.ax3.set_title('(b) Success Rate in Scenario B (Wind On)', color='white')
                self.ax3.set_xlabel('Episodes', color='white')
                self.ax3.set_ylabel('Success Rate', color='white')
                self.ax3.grid(True, alpha=0.2)
                self.ax3.set_ylim(-0.05, 1.05)

                self.ax4.clear()
                self.ax4.plot(ep_on, rew_on, 'm-', label='Avg Reward', marker='s')
                self.ax4.set_title('(d) Average Reward in Scenario B (Wind On)', color='white')
                self.ax4.set_xlabel('Episodes', color='white')
                self.ax4.set_ylabel('Avg Episodic Reward', color='white')
                self.ax4.grid(True, alpha=0.2)
                
            self.canvas.draw()
        except Exception as e:
            print("Graph update error:", e)

    def draw_wind_pattern(self):"""
    
    orig = orig.replace("    def draw_wind_pattern(self):", new_func)
    
    with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
        f.write(orig)
    print("Function update_graphs added successfully.")
else:
    print("update_graphs already exists.")
