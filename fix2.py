import re
with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    content = f.read()

# Make ttc avoidance dummy
new_ttc = """
    def _comparison_ttc_avoidance(self, client, vehicle_name, current_speed):
        return False
"""
content = re.sub(r'    def _comparison_ttc_avoidance\(self, client, vehicle_name, current_speed\):.*?return True', new_ttc.strip() + "\n\n", content, flags=re.DOTALL)

# Speed up the drone up directly to the goal
new_speed_code = "desired_speed = 22.0 if distance > 25.0 else (15.0 if distance > 10.0 else max(2.5, distance * 0.4))"
content = re.sub(r'desired_speed = 9\.5 if distance.*?distance \* 0\.35\)\)', new_speed_code, content)

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(content)
