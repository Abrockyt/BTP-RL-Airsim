import re

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace init_graphs
new_init = """    def init_graphs(self):
        # Initialize the 4 dynamic trajectory plots + 1 map subplot
        self.fig.clear()

        # Create 2x3 grid
        self.ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')  # Speed
        self.ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')  # Altitude
        self.ax3 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')  # Battery
        self.ax4 = self.fig.add_subplot(2, 3, 5, facecolor='#1e1e1e')  # Energy
        self.map_ax = self.fig.add_subplot(2, 3, (3, 6), facecolor='#1e1e1e') # Right side: Map

        # Configure common styling
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.map_ax]:
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', color='#4d4d4d', alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color('#4d4d4d')

        # Map styling
        self.map_ax.set_title('Mission Map (Click to set Goal)', color='white')
        self.map_ax.set_xlim(-150, 150)
        self.map_ax.set_ylim(-150, 150)

        # Set titles & labels
        self.ax1.set_title('Live Speed', color='white')
        self.ax1.set_ylabel('Speed (m/s)', color='white')
        self.ax1.set_xlabel('Time (s)', color='white')

        self.ax2.set_title('Live Altitude', color='white')
        self.ax2.set_ylabel('Altitude (m)', color='white')
        self.ax2.set_xlabel('Time (s)', color='white')

        self.ax3.set_title('Battery Drain', color='white')
        self.ax3.set_ylabel('Battery (%)', color='white')
        self.ax3.set_xlabel('Time (s)', color='white')
        self.ax3.set_ylim(90, 101)

        self.ax4.set_title('Total Energy Consumed', color='white')
        self.ax4.set_ylabel('Energy (Wh)', color='white')
        self.ax4.set_xlabel('Time (s)', color='white')

        # Adjust layout
        self.fig.tight_layout(pad=3.0)

        # Initialize empty lines
        self.line_ax1, = self.ax1.plot([], [], 'c-', linewidth=2.0)
        self.line_ax2, = self.ax2.plot([], [], 'y-', linewidth=2.0)
        self.line_ax3, = self.ax3.plot([], [], 'g-', linewidth=2.0)
        self.line_ax4, = self.ax4.plot([], [], 'm-', linewidth=2.0)

        self.map_goal_pos, = self.map_ax.plot([], [], 'r*', markersize=15, zorder=5)
        self.map_drone_pos, = self.map_ax.plot([], [], 'bo', markersize=8, zorder=4)
        self.map_drone_path, = self.map_ax.plot([], [], 'b--', alpha=0.6, zorder=3)

        self.canvas.draw()"""

text = re.sub(r'    def init_graphs\(self\):.*?        self\.canvas\.draw\(\)', new_init, text, count=1, flags=re.DOTALL)

# Replace update_graphs
new_update = """    def update_graphs(self):
        time_data = self.metrics.get('time', [])
        if len(time_data) > 0:
            # Ax1 = Speed
            speeds = self.metrics.get('speeds', [])
            self.line_ax1.set_data(time_data, speeds)
            self.ax1.set_xlim(0, max(5, time_data[-1]))
            if speeds: self.ax1.set_ylim(0, max(5.0, max(speeds)*1.2))

            # Ax2 = Altitude
            altitudes = self.metrics.get('altitudes', [])
            self.line_ax2.set_data(time_data, altitudes)
            self.ax2.set_xlim(0, max(5, time_data[-1]))
            if altitudes: self.ax2.set_ylim(0, max(10.0, max(altitudes)*1.2))

            # Ax3 = Battery
            battery = self.metrics.get('battery', [])
            self.line_ax3.set_data(time_data, battery)
            self.ax3.set_xlim(0, max(5, time_data[-1]))
            if battery: self.ax3.set_ylim(max(0, min(battery)-5), 105)

            # Ax4 = Energy
            energy = self.metrics.get('energy', [])
            self.line_ax4.set_data(time_data, energy)
            self.ax4.set_xlim(0, max(5, time_data[-1]))
            if energy: self.ax4.set_ylim(0, max(0.5, max(energy)*1.2))

        # Update Map
        if hasattr(self, 'map_goal_pos'):
            self.map_goal_pos.set_data([self.goal_x], [self.goal_y])

        if len(self.metrics['positions_x']) > 0:
            self.map_drone_pos.set_data([self.metrics['positions_x'][-1]], [self.metrics['positions_y'][-1]])
            self.map_drone_path.set_data(self.metrics['positions_x'], self.metrics['positions_y'])

        self.canvas.draw()"""

text = re.sub(r'    def update_graphs\(self\):.*?        self\.canvas\.draw\(\)', new_update, text, count=1, flags=re.DOTALL)

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(text)
