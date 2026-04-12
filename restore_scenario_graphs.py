import re

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    text = f.read()

with open(r'C:\Users\abroc\AppData\Roaming\Code\User\History\6c13d48d\4z9D.py', 'r', encoding='utf-8') as f:
    backup_text = f.read()

init_match = re.search(r'    def init_graphs\(self\):.*?        self\.canvas\.mpl_connect\(\'button_press_event\', self\.on_map_click\)', backup_text, flags=re.DOTALL)
update_match = re.search(r'    def update_graphs\(self\):.*?(?=    def update_vision_display\(self\):)', backup_text, flags=re.DOTALL)

if init_match and update_match:
    old_init = init_match.group(0)
    old_update = update_match.group(0)
    
    # Replace init_graphs in current file
    text = re.sub(r'    def init_graphs\(self\):.*?        self\.canvas\.draw\(\)', old_init, text, flags=re.DOTALL)
    
    # Replace update_graphs in current file
    text = re.sub(r'    def update_graphs\(self\):.*?(?=    def draw_wind_pattern)', old_update, text, flags=re.DOTALL)
    
    with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Restored original Scenario graphs!")
else:
    print("Could not extract from backup.")

