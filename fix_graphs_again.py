with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    text = f.read()

import re

new_update_graphs = """    def update_graphs(self):
        try:
            # Update Map safely
            if hasattr(self, 'map_goal_pos'):
                self.map_goal_pos.set_data([self.goal_x], [self.goal_y])

            if getattr(self, 'metrics', None) and len(self.metrics.get('positions_x', [])) > 0:
                self.map_drone_pos.set_data([self.metrics['positions_x'][-1]], [self.metrics['positions_y'][-1]])
                self.map_drone_path.set_data(self.metrics['positions_x'], self.metrics['positions_y'])

            # Update Telemetry (Live Speed, Altitude, Battery, Energy)
            if getattr(self, 'metrics', None) and len(self.metrics.get('time', [])) > 0:
                t = self.metrics['time']
                
                # Speed
                if hasattr(self, 'line_ax1'):
                    self.line_ax1.set_data(t, self.metrics['speeds'])
                    self.ax1.set_xlim(max(0, t[-1]-50), max(50, t[-1]))
                    self.ax1.set_ylim(-1, max(25, max(self.metrics['speeds'])+5))
                
                # Altitude
                if hasattr(self, 'line_ax2'):
                    self.line_ax2.set_data(t, [abs(a) for a in self.metrics['altitudes']])
                    self.ax2.set_xlim(max(0, t[-1]-50), max(50, t[-1]))
                    self.ax2.set_ylim(-1, 30)
                
                # Battery
                if hasattr(self, 'line_ax3'):
                    self.line_ax3.set_data(t, self.metrics['battery'])
                    self.ax3.set_xlim(max(0, t[-1]-50), max(50, t[-1]))
                    
                # Energy
                if hasattr(self, 'line_ax4'):
                    self.line_ax4.set_data(t, self.metrics['energy'])
                    self.ax4.set_xlim(max(0, t[-1]-50), max(50, t[-1]))
                    self.ax4.set_ylim(-0.01, max(1.0, max(self.metrics['energy'])+0.5))

            self.canvas.draw()
        except Exception as e:
            print(f"Graph update error: {e}")"""

text = re.sub(r'    def update_graphs\(self\):.*?self\.canvas\.draw\(\)', new_update_graphs, text, flags=re.DOTALL)

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(text)
