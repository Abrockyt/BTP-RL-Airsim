import sys

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False

# Remove vision_frame logic from create_widgets
for l in lines:
    if "vision_frame = tk.Frame(left_panel, bg='#2d2d2d')" in l:
        skip = True
    if skip and "pressure_frame = tk.Frame(left_panel" in l:
        skip = False
    
    if skip:
        continue
        
    # Silence update_vision_display
    if "def update_vision_display(self, frame, obj_boxes=None, text_info=None):" in l:
        new_lines.append(l)
        new_lines.append("        return\n")
        skip = True
        continue
    if skip and "def draw_wind_pattern" in l:
        skip = False

    # Silence update_rl_display
    if "def update_rl_display(self):" in l:
        new_lines.append(l)
        new_lines.append("        return\n")
        skip = True
        continue
    if skip and "def _smooth_velocity_command" in l:
        skip = False

    # Silence TTC avoidance
    if "def _comparison_ttc_avoidance(self, client, vehicle_name, current_speed):" in l:
        new_lines.append(l)
        new_lines.append("        return False\n")
        skip = True
        continue
    if skip and "def open_multidrone_window(self):" in l:
        skip = False
        
    # Replace graphing calls
    l = l.replace("linewidth=2.5, label='Drone 1: MHA-PPO'", "linewidth=2.5, label='Drone 1: MHA-PPO', marker='o', markersize=3")
    l = l.replace("linewidth=2.5, label='Drone 2: GNN'", "linewidth=2.5, label='Drone 2: GNN', marker='s', markersize=3")
    
    # Increase speed tracking
    l = l.replace("dt = 0.2", "dt = 0.05")
    l = l.replace("duration=0.6", "duration=0.15")
    l = l.replace("desired_speed = 9.5 if distance > 30.0 else (8.2 if distance > 18.0 else max(1.5, distance * 0.35))", 
                  "desired_speed = 22.0 if distance > 25.0 else (15.0 if distance > 10.0 else max(2.5, distance * 0.4))")

    if not skip:
        new_lines.append(l)

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Changes applied!")
