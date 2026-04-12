import re
with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the desired speed code in _fly_drone1_mhappo & _fly_drone2_gnn
# "desired_speed = 9.5 if distance > 30.0 else (8.2 if distance > 18.0 else max(1.5, distance * 0.35))"
new_speed_code = "desired_speed = 18.0 if distance > 20.0 else (12.0 if distance > 10.0 else max(2.5, distance * 0.5))"
content = re.sub(r'desired_speed = 9\.5.*?\* 0\.35\)\)', new_speed_code, content)

# Reduce the loop sleep to make tracking tighter
content = re.sub(r'dt\s*=\s*0\.2', 'dt = 0.05', content)
content = re.sub(r'duration=0\.6', 'duration=0.15', content)

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(content)
