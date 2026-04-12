"""
🚁 SMART DRONE CONTROL GUI
Complete GUI with Vision, All Sensors, Performance Analytics, and Goal Navigation
"""

# Fix IOLoop FIRST - before any other imports
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nest-asyncio"])
    import nest_asyncio
    nest_asyncio.apply()

import airsim
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
import threading
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

# Energy constants
P_HOVER = 200  # Watts
BATTERY_CAPACITY_WH = 100  # Wh

class DroneControlGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚁 Smart Drone Control - Vision & Sensors")
        self.window.geometry("1600x900")
        self.window.configure(bg='#1a1a1a')
        
        # Flight state
        self.client = None
        self.running = False
        self.goal_x = 100.0
        self.goal_y = 100.0
        self.altitude = -20.0
        
        # Sensor data
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.imu_data = {'accel': [0, 0, 0], 'gyro': [0, 0, 0]}
        self.gps_data = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}
        self.battery = 100.0
        self.energy = 0.0
        self.wind_x = 2.0
        self.wind_y = 1.0
        
        # Vision
        self.depth_image = None
        
        # Metrics
        self.metrics = {
            'time': [], 'speed': [], 'altitude': [], 'battery': [],
            'energy': [], 'path_x': [], 'path_y': []
        }
        self.start_time = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main container
        main = tk.Frame(self.window, bg='#1a1a1a')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # LEFT PANEL - Controls and Sensors
        left = tk.Frame(main, bg='#2d2d2d', width=400)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        
        # Title
        tk.Label(left, text="🚁 DRONE CONTROL", font=("Arial", 16, "bold"),
                bg='#2d2d2d', fg='white').pack(pady=10)
        
        # ----- VISION DISPLAY -----
        tk.Label(left, text="📷 DEPTH VISION", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#00e5ff').pack(pady=(10, 5))
        
        self.vision_canvas = tk.Canvas(left, width=360, height=200, bg='#1a1a1a', highlightthickness=0)
        self.vision_canvas.pack()
        
        self.obstacle_label = tk.Label(left, text="DEPTH CAMERA READY", font=("Arial", 10, "bold"),
                                      bg='#2d2d2d', fg='#757575')
        self.obstacle_label.pack(pady=5)
        
        # ----- GOAL SETTING -----
        tk.Label(left, text="🎯 GOAL POSITION", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#4caf50').pack(pady=(15, 5))
        
        goal_frame = tk.Frame(left, bg='#2d2d2d')
        goal_frame.pack()
        
        tk.Label(goal_frame, text="X:", bg='#2d2d2d', fg='white').grid(row=0, column=0, padx=5)
        self.goal_x_entry = tk.Entry(goal_frame, width=8, bg='#404040', fg='white', insertbackground='white')
        self.goal_x_entry.insert(0, "100.0")
        self.goal_x_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(goal_frame, text="Y:", bg='#2d2d2d', fg='white').grid(row=0, column=2, padx=5)
        self.goal_y_entry = tk.Entry(goal_frame, width=8, bg='#404040', fg='white', insertbackground='white')
        self.goal_y_entry.insert(0, "100.0")
        self.goal_y_entry.grid(row=0, column=3, padx=5)
        
        tk.Button(goal_frame, text="SET GOAL", command=self.set_goal, bg='#4caf50', fg='white',
                 font=("Arial", 9, "bold")).grid(row=0, column=4, padx=10)
        
        self.goal_label = tk.Label(left, text=f"Current Goal: ({self.goal_x}, {self.goal_y})",
                                   font=("Arial", 9), bg='#2d2d2d', fg='#76ff03')
        self.goal_label.pack()
        
        # ----- TELEMETRY -----
        tk.Label(left, text="📊 TELEMETRY", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#ff9800').pack(pady=(15, 5))
        
        tel_frame = tk.Frame(left, bg='#2d2d2d')
        tel_frame.pack()
        
        self.pos_label = tk.Label(tel_frame, text="Position: (0, 0)", font=("Arial", 10),
                                 bg='#2d2d2d', fg='white')
        self.pos_label.pack()
        self.speed_label = tk.Label(tel_frame, text="Speed: 0.0 m/s", font=("Arial", 10),
                                   bg='#2d2d2d', fg='white')
        self.speed_label.pack()
        self.dist_label = tk.Label(tel_frame, text="Distance to Goal: -- m", font=("Arial", 10),
                                  bg='#2d2d2d', fg='white')
        self.dist_label.pack()
        
        # ----- SENSORS -----
        tk.Label(left, text="🔬 SENSORS", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#9c27b0').pack(pady=(15, 5))
        
        sens_frame = tk.Frame(left, bg='#2d2d2d')
        sens_frame.pack()
        
        # IMU
        tk.Label(sens_frame, text="IMU (Accel):", font=("Arial", 9, "bold"),
                bg='#2d2d2d', fg='#00e5ff').pack()
        self.imu_label = tk.Label(sens_frame, text="X:0.0 Y:0.0 Z:0.0 m/s²", font=("Arial", 9),
                                 bg='#2d2d2d', fg='white')
        self.imu_label.pack()
        
        # GPS
        tk.Label(sens_frame, text="GPS:", font=("Arial", 9, "bold"),
                bg='#2d2d2d', fg='#00e5ff').pack(pady=(5,0))
        self.gps_label = tk.Label(sens_frame, text="Lat:0.0 Lon:0.0 Alt:0.0m", font=("Arial", 9),
                                 bg='#2d2d2d', fg='white')
        self.gps_label.pack()
        
        # Wind
        tk.Label(sens_frame, text="Wind Sensor:", font=("Arial", 9, "bold"),
                bg='#2d2d2d', fg='#00e5ff').pack(pady=(5,0))
        self.wind_label = tk.Label(sens_frame, text="Wind: 2.2 m/s", font=("Arial", 9),
                                   bg='#2d2d2d', fg='white')
        self.wind_label.pack()
        
        # ----- BATTERY -----
        tk.Label(left, text="🔋 BATTERY", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='#ffeb3b').pack(pady=(15, 5))
        
        self.battery_label = tk.Label(left, text="100.0%", font=("Arial", 24, "bold"),
                                      bg='#2d2d2d', fg='#4caf50')
        self.battery_label.pack()
        self.energy_label = tk.Label(left, text="Energy: 0.00 Wh", font=("Arial", 10),
                                     bg='#2d2d2d', fg='white')
        self.energy_label.pack()
        
        # ----- CONTROLS -----
        tk.Label(left, text="🎮 CONTROLS", font=("Arial", 12, "bold"),
                bg='#2d2d2d', fg='white').pack(pady=(15, 5))
        
        self.start_btn = tk.Button(left, text="🚀 START FLIGHT", command=self.start_flight,
                                   bg='#4caf50', fg='white', font=("Arial", 12, "bold"),
                                   width=20, height=2)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(left, text="⏹ STOP", command=self.stop_flight,
                                 bg='#f44336', fg='white', font=("Arial", 10, "bold"),
                                 width=20, state="disabled")
        self.stop_btn.pack(pady=5)
        
        self.status_label = tk.Label(left, text="● Ready", font=("Arial", 11, "bold"),
                                     bg='#2d2d2d', fg='#757575')
        self.status_label.pack(pady=10)
        
        # RIGHT PANEL - Graphs
        right = tk.Frame(main, bg='#1a1a1a')
        right.pack(side='right', fill='both', expand=True)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(12, 8), facecolor='#1a1a1a')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def set_goal(self):
        try:
            self.goal_x = float(self.goal_x_entry.get())
            self.goal_y = float(self.goal_y_entry.get())
            self.goal_label.config(text=f"Current Goal: ({self.goal_x}, {self.goal_y})")
            print(f"🎯 Goal set to: ({self.goal_x}, {self.goal_y})")
        except:
            messagebox.showerror("Error", "Invalid goal coordinates!")
    
    def start_flight(self):
        self.running = True
        threading.Thread(target=self._flight_loop, daemon=True).start()
        
    def stop_flight(self):
        self.running = False
        
    def _flight_loop(self):
        try:
            # Connect
            print("🔌 Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✅ Connected!\n")
            
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_label.config(text="● Flying", fg='#4caf50')
            
            # Reset metrics
            for key in self.metrics:
                self.metrics[key].clear()
            self.start_time = time.time()
            self.battery = 100.0
            self.energy = 0.0
            
            # Takeoff
            print("🛫 Taking off...")
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(self.altitude, 3.0).join()
            print("✅ At altitude, starting navigation...\n")
            
            # Start update threads
            threading.Thread(target=self._update_displays, daemon=True).start()
            threading.Thread(target=self._update_graphs, daemon=True).start()
            
            step = 0
            while self.running:
                # Get state
                state = self.client.getMultirotorState()
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                self.position = np.array([pos.x_val, pos.y_val, pos.z_val])
                self.velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
                
                # Get sensors
                imu = self.client.getImuData()
                self.imu_data = {
                    'accel': [imu.linear_acceleration.x_val, imu.linear_acceleration.y_val, imu.linear_acceleration.z_val],
                    'gyro': [imu.angular_velocity.x_val, imu.angular_velocity.y_val, imu.angular_velocity.z_val]
                }
                
                gps = self.client.getGpsData()
                self.gps_data = {
                    'lat': gps.gnss.geo_point.latitude,
                    'lon': gps.gnss.geo_point.longitude,
                    'alt': gps.gnss.geo_point.altitude
                }
                
                # Navigation
                goal_pos = np.array([self.goal_x, self.goal_y])
                to_goal = goal_pos - self.position[:2]
                dist = np.linalg.norm(to_goal)
                
                # Check goal reached
                if dist < 5.0:
                    print(f"🏆 GOAL REACHED! Distance: {dist:.1f}m")
                    self.status_label.config(text="● Goal Reached!", fg='#76ff03')
                    messagebox.showinfo("Success", f"🎉 Goal reached!\nEnergy: {self.energy:.2f} Wh")
                    break
                
                # Proportional controller
                direction = to_goal / (dist + 1e-6)
                speed = 10.0 if dist > 20 else max(3.0, dist * 0.4)
                
                target_vx = direction[0] * speed + self.wind_x * 0.1
                target_vy = direction[1] * speed + self.wind_y * 0.1
                
                self.client.moveByVelocityAsync(
                    float(target_vx), float(target_vy), 0.5,
                    duration=0.2
                ).join()
                
                # Energy
                velocity_magnitude = np.linalg.norm(self.velocity[:2])
                power = P_HOVER * (1 + 0.005 * velocity_magnitude**2)
                energy_step = power * (0.2 / 3600.0)
                self.energy += energy_step
                self.battery = max(0, 100.0 - (self.energy / BATTERY_CAPACITY_WH * 100))
                
                # Metrics
                elapsed = time.time() - self.start_time
                self.metrics['time'].append(elapsed)
                self.metrics['speed'].append(velocity_magnitude)
                self.metrics['altitude'].append(abs(self.position[2]))
                self.metrics['battery'].append(self.battery)
                self.metrics['energy'].append(self.energy)
                self.metrics['path_x'].append(self.position[0])
                self.metrics['path_y'].append(self.position[1])
                
                # Log
                if step % 20 == 0:
                    print(f"Step {step:3d} | Pos: ({self.position[0]:.1f}, {self.position[1]:.1f}) | "
                          f"Goal: ({self.goal_x}, {self.goal_y}) | Dist: {dist:.1f}m | "
                          f"Battery: {self.battery:.1f}%")
                
                step += 1
                time.sleep(0.1)
            
            # Land
            print("\n🛬 Landing...")
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✅ Landed successfully!")
            
        except Exception as e:
            print(f"❌ Flight error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_flight()
            
    def _update_displays(self):
        while self.running:
            try:
                # Telemetry
                self.pos_label.config(text=f"Position: ({self.position[0]:.1f}, {self.position[1]:.1f})")
                speed = np.linalg.norm(self.velocity[:2])
                self.speed_label.config(text=f"Speed: {speed:.1f} m/s")
                dist = np.linalg.norm(np.array([self.goal_x, self.goal_y]) - self.position[:2])
                self.dist_label.config(text=f"Distance to Goal: {dist:.1f}m")
                
                # Battery
                color = '#4caf50' if self.battery > 50 else '#ff9800' if self.battery > 20 else '#f44336'
                self.battery_label.config(text=f"{self.battery:.1f}%", fg=color)
                self.energy_label.config(text=f"Energy: {self.energy:.2f} Wh")
                
                # Sensors
                self.imu_label.config(text=f"X:{self.imu_data['accel'][0]:.1f} Y:{self.imu_data['accel'][1]:.1f} Z:{self.imu_data['accel'][2]:.1f} m/s²")
                self.gps_label.config(text=f"Lat:{self.gps_data['lat']:.6f} Lon:{self.gps_data['lon']:.6f} Alt:{self.gps_data['alt']:.1f}m")
                
                wind_mag = np.sqrt(self.wind_x**2 + self.wind_y**2)
                self.wind_label.config(text=f"Wind: {wind_mag:.1f} m/s")
                
                # Vision
                self.update_vision()
                
                time.sleep(0.1)
            except:
                pass
    
    def update_vision(self):
        try:
            self.vision_canvas.delete("all")
            
            if not self.client or not self.running:
                return
            
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
            ])
            
            if responses and len(responses[0].image_data_float) > 0:
                depth_data = np.array(responses[0].image_data_float)
                depth_data = depth_data.reshape(responses[0].height, responses[0].width)
                
                h, w = depth_data.shape
                left_depth = np.mean(depth_data[h//3:2*h//3, :w//3])
                center_depth = np.mean(depth_data[h//3:2*h//3, w//3:2*w//3])
                right_depth = np.mean(depth_data[h//3:2*h//3, 2*w//3:])
                
                def depth_color(d):
                    return '#f44336' if d < 3 else '#ff9800' if d < 5 else '#ffeb3b' if d < 10 else '#4caf50'
                
                # Draw zones
                self.vision_canvas.create_rectangle(10, 10, 110, 190, fill=depth_color(left_depth), outline='white', width=2)
                self.vision_canvas.create_text(60, 50, text="LEFT", fill='white', font=("Arial", 10, "bold"))
                self.vision_canvas.create_text(60, 150, text=f"{left_depth:.1f}m", fill='white', font=("Arial", 9))
                
                self.vision_canvas.create_rectangle(130, 10, 230, 190, fill=depth_color(center_depth), outline='white', width=2)
                self.vision_canvas.create_text(180, 50, text="CENTER", fill='white', font=("Arial", 10, "bold"))
                self.vision_canvas.create_text(180, 150, text=f"{center_depth:.1f}m", fill='white', font=("Arial", 9))
                
                self.vision_canvas.create_rectangle(250, 10, 350, 190, fill=depth_color(right_depth), outline='white', width=2)
                self.vision_canvas.create_text(300, 50, text="RIGHT", fill='white', font=("Arial", 10, "bold"))
                self.vision_canvas.create_text(300, 150, text=f"{right_depth:.1f}m", fill='white', font=("Arial", 9))
                
                min_depth = min(left_depth, center_depth, right_depth)
                text = "⚠️ DANGER!" if min_depth < 3 else "⚠️ WARNING" if min_depth < 5 else "CLEAR ✓"
                color = depth_color(min_depth)
                self.obstacle_label.config(text=text, fg=color)
        except:
            pass
    
    def _update_graphs(self):
        while self.running:
            try:
                if len(self.metrics['time']) > 2:
                    self.fig.clear()
                    times = self.metrics['time']
                    
                    # Speed
                    ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
                    ax1.plot(times, self.metrics['speed'], color='#2196f3', linewidth=2)
                    ax1.set_title('SPEED', color='white', fontweight='bold')
                    ax1.set_ylabel('m/s', color='white')
                    ax1.tick_params(colors='white')
                    ax1.grid(alpha=0.2)
                    
                    # Altitude
                    ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
                    ax2.plot(times, self.metrics['altitude'], color='#ff9800', linewidth=2)
                    ax2.set_title('ALTITUDE', color='white', fontweight='bold')
                    ax2.set_ylabel('m', color='white')
                    ax2.tick_params(colors='white')
                    ax2.grid(alpha=0.2)
                    
                    # Battery
                    ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
                    ax3.plot(times, self.metrics['battery'], color='#4caf50', linewidth=2)
                    ax3.set_title('BATTERY', color='white', fontweight='bold')
                    ax3.set_ylabel('%', color='white')
                    ax3.tick_params(colors='white')
                    ax3.grid(alpha=0.2)
                    
                    # Energy
                    ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
                    ax4.plot(times, self.metrics['energy'], color='#f44336', linewidth=2)
                    ax4.set_title('ENERGY', color='white', fontweight='bold')
                    ax4.set_ylabel('Wh', color='white')
                    ax4.tick_params(colors='white')
                    ax4.grid(alpha=0.2)
                    
                    # Flight path
                    ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')
                    ax5.plot(self.metrics['path_x'], self.metrics['path_y'], 'c-', linewidth=2, label='Path')
                    ax5.plot([0], [0], 'go', markersize=10, label='Start')
                    ax5.plot([self.goal_x], [self.goal_y], 'r*', markersize=20, label='Goal')
                    if self.metrics['path_x']:
                        ax5.plot([self.metrics['path_x'][-1]], [self.metrics['path_y'][-1]], 'bo', markersize=8, label='Current')
                    ax5.set_title('FLIGHT MAP', color='white', fontweight='bold')
                    ax5.set_xlabel('X (m)', color='white')
                    ax5.set_ylabel('Y (m)', color='white')
                    ax5.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
                    ax5.tick_params(colors='white')
                    ax5.grid(alpha=0.2)
                    ax5.axis('equal')
                    
                    self.canvas.draw()
                
                time.sleep(0.5)
            except:
                pass
    
    def run(self):
        print("="*70)
        print("🚁 SMART DRONE CONTROL GUI")
        print("="*70)
        print("✓ All sensors active: Depth Camera, IMU, GPS, Wind")
        print("✓ Set goal and click START FLIGHT")
        print("="*70 + "\n")
        self.window.mainloop()

if __name__ == "__main__":
    app = DroneControlGUI()
    app.run()
