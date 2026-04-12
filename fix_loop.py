import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Don't turn off 'running' when goal is reached so we know we want to continue
pattern1 = r'self\.flight_active = False\s+self\.running = False\s+self\.smart_landing\(\)'
replacement1 = 'self.flight_active = False\n                    # self.running left True for continuous episodes\n                    self.smart_landing()'
content = re.sub(pattern1, replacement1, content)

# 2. Don't turn off 'running' when crashing either (optional, but let's do it so the graphs build automatically in all cases unless the user presses STOP). Just the end of the _flight_loop is fine

# Add the continuous loop trigger at the end of _flight_loop
pattern2 = r'(print\(f"   Best Time: \{self\.best_time:\.1f\}s"\)\s+print\(f"\{''=\''\*60\}\\n"\))'
replacement2 = r'\1\n\n            if getattr(self, "running", False):\n                print("?? Continuous Episode Mode: Starting next episode automatically in 4 seconds...")\n                self.window.after(4000, self.start_flight)'
content = re.sub(pattern2, replacement2, content)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
