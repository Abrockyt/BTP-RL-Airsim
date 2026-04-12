import re

# File where we want to restore strictly the Scenario graphs
with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    text = f.read()

# Load the exact scenario graph logic from backup
with open(r'C:\Users\abroc\AppData\Roaming\Code\User\History\6c13d48d\4z9D.py', 'r', encoding='utf-8') as f:
    backup_text = f.read()

# Extract exactly the init_graphs and update_graphs logic
init_match = re.search(r'    def init_graphs\(self\):.*?        self\.canvas\.mpl_connect\(\'button_press_event\', self\.on_map_click\)', backup_text, flags=re.DOTALL)
update_match = re.search(r'    def update_graphs\(self\):.*?(?=    def update_vision_display\(self\):)', backup_text, flags=re.DOTALL)

if init_match and update_match:
    old_init = init_match.group(0)
    old_update = update_match.group(0)
    
    # Safely swap out the current live graphs back to the desired scenario graphs
    text = re.sub(r'    def init_graphs\(self\):.*?    def draw_wind_pattern', old_init + '\n\n' + old_update + '    def draw_wind_pattern', text, flags=re.DOTALL)
    
    with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Graphs completely reverted to Scenario format!")
else:
    print("Could not extract from backup.")

