import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Fix init_graphs map limits
text = text.replace(
    "self.map_ax.set_title('Mission Map (Click to set Goal)', color='white')",
    "self.map_ax.set_title('Mission Map (Click to set Goal)', color='white')\n        self.map_ax.set_xlim(-150, 150)\n        self.map_ax.set_ylim(-150, 150)"
)

# 2. Fix update_graphs map drawing logic
old_map_logic = """        # Update Map
        if len(self.metrics['positions_x']) > 0:
            self.map_drone_pos.set_data([self.metrics['positions_x'][-1]], [self.metrics['positions_y'][-1]])
            self.map_drone_path.set_data(self.metrics['positions_x'], self.metrics['positions_y'])
            self.map_ax.relim()
            self.map_ax.autoscale_view()

            # Set the goal pos
            try:
                if self.env and hasattr(self.env, 'goal_position'):
                    goal = self.env.goal_position
                    if goal is not None:
                        self.map_goal_pos.set_data([goal[0]], [goal[1]])        
            except:
                pass"""

new_map_logic = """        # Always draw the goal
        self.map_goal_pos.set_data([self.goal_x], [self.goal_y])

        # Update Map Drone Stream
        if len(self.metrics['positions_x']) > 0:
            self.map_drone_pos.set_data([self.metrics['positions_x'][-1]], [self.metrics['positions_y'][-1]])
            self.map_drone_path.set_data(self.metrics['positions_x'], self.metrics['positions_y'])"""

text = text.replace(old_map_logic, new_map_logic)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(text)
