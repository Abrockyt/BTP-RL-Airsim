import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix safe_airsim_call(self.client.moveByVelocityAsync, ...)
# Add vehicle_name="Drone1" if missing
def fix_call(match):
    func_str = match.group(1)
    args_str = match.group(2)
    if 'vehicle_name' not in args_str:
        return f"safe_airsim_call({func_str}, {args_str}, vehicle_name='Drone1')"
    return match.group(0)

new_text = re.sub(r'safe_airsim_call\((self\.client\.[A-Za-z]+)(, [^)]+)\)', fix_call, text)

if new_text != text:
    print('Updated safe_airsim_calls!')
    with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
