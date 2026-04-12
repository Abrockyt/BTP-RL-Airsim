import re
with open('iot_projet_gui.py', 'r', encoding='utf-8') as f: content = f.read()

# Remove the client allocation from _flight_loop
pattern = r"            # Re-use existing AirSim client.*?\n            safe_airsim_call\(self\.client\.enableApiControl, True, vehicle_name='Drone1'\)"
replacement = "            if getattr(self, 'client', None) is None:\n                self.client = airsim.MultirotorClient()\n                self.client.confirmConnection()\n            safe_airsim_call(self.client.enableApiControl, True, vehicle_name='Drone1')"
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f: f.write(content)
