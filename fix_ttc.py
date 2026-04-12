import re

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    content = f.read()

new_avoidance = """
    def _comparison_ttc_avoidance(self, client, vehicle_name, current_speed):
        try:
            dist = client.getDistanceSensorData(distance_sensor_name="Distance", vehicle_name=vehicle_name)
            if dist and dist.distance < 4.0:
                client.moveByVelocityAsync(0.0, 0.0, -2.0, duration=0.5, vehicle_name=vehicle_name)
                return True
        except:
            pass
        return False
"""

content = re.sub(r'    def _comparison_ttc_avoidance\(self, client, vehicle_name, current_speed\):.*?return True\n', new_avoidance.strip("\n") + "\n\n", content, flags=re.DOTALL)

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(content)

