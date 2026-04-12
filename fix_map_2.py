import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace everything from "# Update Map" down to the "self.canvas.draw()" inclusive
pattern = re.compile(r'        # Update Map\n(?:.*?)self\.canvas\.draw\(\)', re.DOTALL)

new_map_logic = """        # Update Map
        if hasattr(self, 'map_goal_pos'):
            self.map_goal_pos.set_data([self.goal_x], [self.goal_y])

        if len(self.metrics['positions_x']) > 0:
            self.map_drone_pos.set_data([self.metrics['positions_x'][-1]], [self.metrics['positions_y'][-1]])
            self.map_drone_path.set_data(self.metrics['positions_x'], self.metrics['positions_y'])

        self.canvas.draw()"""

text = pattern.sub(new_map_logic, text)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(text)
