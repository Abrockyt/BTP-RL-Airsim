with open(r'C:\Users\abroc\AppData\Roaming\Code\User\History\6c13d48d\4z9D.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
old_func_match = re.search(r'    def update_graphs\(self\):.*?    def update_vision_display\(self\):', text, flags=re.DOTALL)
if old_func_match:
    old_func = old_func_match.group(0).replace('    def update_vision_display(self):', '')
    
    with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
        curr_text = f.read()
        
    curr_text = re.sub(r'    def update_graphs\(self\):.*?    def draw_wind_pattern', old_func + '    def draw_wind_pattern', curr_text, flags=re.DOTALL)
    
    with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
        f.write(curr_text)
    print("Graph logic restored!")
else:
    print("Could not find old update_graphs")
