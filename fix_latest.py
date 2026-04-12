import re

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Disable update_vision_display
new_uv = '''    def update_vision_display(self):
        pass

    def draw_wind_pattern(self):'''
text = re.sub(r'    def update_vision_display\(self\):.*?    def draw_wind_pattern\(self\):', new_uv, text, flags=re.DOTALL)

# 2. Disable update_rl_display
new_rl = '''    def update_rl_display(self):
        pass

    def _smooth_velocity_command(self,'''
text = re.sub(r'    def update_rl_display\(self\):.*?    def _smooth_velocity_command\(self,', new_rl, text, flags=re.DOTALL)

# 3. Disable _comparison_ttc_avoidance
new_ttc = '''    def _comparison_ttc_avoidance(self, client, vehicle_name, current_speed):
        return False

    def open_multidrone_window'''
text = re.sub(r'    def _comparison_ttc_avoidance\(self, client, vehicle_name, current_speed\):.*?    def open_multidrone_window', new_ttc, text, flags=re.DOTALL)

# 4. Update the plots to have marker='o'
text = text.replace("linewidth=2.5, label='Drone 1: MHA-PPO'", "linewidth=2.5, label='Drone 1: MHA-PPO', marker='o', markersize=3")
text = text.replace("linewidth=2.5, label='Drone 2: GNN'", "linewidth=2.5, label='Drone 2: GNN', marker='s', markersize=3")

# 5. Increase speed
text = text.replace("desired_speed = 9.5 if distance > 30.0 else (8.2 if distance > 18.0 else max(1.5, distance * 0.35))",
                    "desired_speed = 22.0 if distance > 25.0 else (15.0 if distance > 10.0 else max(2.5, distance * 0.4))")
text = text.replace("dt = 0.2", "dt = 0.05")
text = text.replace("duration=0.6", "duration=0.15")

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(text)
