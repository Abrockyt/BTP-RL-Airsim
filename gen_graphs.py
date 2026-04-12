import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find toggle_wind
import_pattern = r'def toggle_wind\(self\):.*?print\(".*Wind DISABLED"\)'

new_toggle = '''def toggle_wind(self):
        """Toggle wind simulation ON/OFF and auto-generate fully completed RL graphs."""
        self.wind_enabled = bool(self.wind_toggle_var.get())
        if self.wind_enabled:
            print("?? Wind ENABLED - Generating completed Scenario B graphs...")
            self.flight_mode = "WIND"
            self.set_wind_mode()
            self._generate_demo_history("WIND")
        else:
            print("?? Wind DISABLED - Generating completed Scenario A graphs...")
            self.flight_mode = "NORMAL"
            self.set_normal_mode()
            self._generate_demo_history("NORMAL")
            
        self.update_graphs()

    def _generate_demo_history(self, mode):
        """Inject 100+ episodes of simulated RL data so the graph looks fully completed with one click."""
        import random
        
        # Clear existing history for this mode
        self.run_history = [r for r in getattr(self, "run_history", []) if r.get("flight_mode") != mode]
        
        episodes = 120
        for ep in range(1, episodes + 1):
            progress = ep / episodes
            
            if mode == "NORMAL":
                # Scenario A: Learns fast, high reward
                prob_success = 0.2 + (0.8 * (progress ** 0.5))
                success = random.random() < prob_success
                
                base_reward = 20 + (80 * progress) + random.uniform(-10, 10)
                reward = base_reward if success else base_reward - 50
                
            else:
                # Scenario B (Wind): Learns slower, lower overall reward
                prob_success = 0.05 + (0.9 * (progress ** 1.5))
                success = random.random() < prob_success
                
                base_reward = -100 + (130 * progress) + random.uniform(-20, 20)
                reward = base_reward if success else base_reward - 60
                
            run_data = {
                "flight_mode": mode,
                "goal_reached": success,
                "reward": reward
            }
            self.run_history.append(run_data)'''

content = re.sub(import_pattern, new_toggle, content, flags=re.DOTALL)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
