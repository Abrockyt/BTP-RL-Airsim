import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'def start_flight\(self\):.*?threading\.Thread\(target=self\._flight_loop, daemon=True\)\.start\(\)'

replacement = '''def start_flight(self):
        import random
        # Automatically generate random goal as requested
        self.goal_x = round(random.uniform(-60.0, 60.0), 1)
        self.goal_y = round(random.uniform(-60.0, 150.0), 1)
        
        # Update UI text boxes
        self.goal_x_entry.delete(0, "end")
        self.goal_y_entry.delete(0, "end")
        self.goal_x_entry.insert(0, str(self.goal_x))
        self.goal_y_entry.insert(0, str(self.goal_y))
        
        # Update current goal label
        if hasattr(self, "current_goal_label"):
            self.current_goal_label.config(text=f"Current: ({self.goal_x}, {self.goal_y})", fg="#4caf50")
            
        print(f"?? Random Goal Selected automatically: {self.goal_x}, {self.goal_y}")
        
        # Update graph map
        self.update_graphs()
        
        self.status_label.config(text="? Initializing...", fg="orange")
        self.window.update()
        import threading
        threading.Thread(target=self._flight_loop, daemon=True).start()'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
