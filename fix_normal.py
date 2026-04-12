import re
with open('iot_projet_gui.py', 'r', encoding='utf-8') as f: content = f.read()
pattern = r'(elif self\.flight_mode == "NORMAL":\s+# NORMAL MODE: Predictive TTC-based collision avoidance.*?)(\s+# Get current altitude)'
# replace everything with obstacle_avoided = False
replacement = r'elif self.flight_mode == "NORMAL":\n                    obstacle_avoided = False\n\n\2'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)
with open('iot_projet_gui.py', 'w', encoding='utf-8') as f: f.write(content)
