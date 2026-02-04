"""
Smart Drone Control with Interactive GUI - OPTIMIZED FOR STABILITY
Features: Adaptive speed, Smart collision recovery, Interactive map, Energy tracking
FIXES: Reduced API call frequency, better buffer management
"""

import sys
sys.path.append('.')
from smart_drone import (
    DronePilot, load_model, preprocess_image, 
    smart_speed_control, smart_collision_recovery
)

import airsim
import torch
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import os


class SmartDroneGUI:
    """Interactive map GUI for smart drone control with restart functionality"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚁 Smart Drone - Stable Flight Control")
        self.window.geometry("1500x900")
        
        # Make window visible
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.focus_force()
        
        # Map parameters
        self.map_size = 600
        self.map_range = 100
        self.scale = self.map_size / (2 * self.map_range)
        
        # State
        self.client = None
        self.model = None
        self.device = None
        self.drone_position = (0, 0)
        self.home_position = None
        self.goal_position = None
        self.running = False
        self.flight_active = False
        self.navigating_to_goal = False
        self.path_points = []
        
        # API call counter - to reduce frequency
        self.api_call_counter = 0
        
        # Performance metrics
        self.metrics = {
            'speeds': [], 
            'altitudes': [], 
            'collisions': 0, 
            'predictions': 0, 
            'warnings': 0,
            'energy': [],
            'timestamps': []
        }
        self.start_time = None
        
        # Battery simulation
        self.battery_percent = 100.0
        self.battery_capacity_wh = 100.0
        self.total_energy_consumed = 0.0
        
        # Graph components
        self.graph_figure = None
        self.graph_canvas = None
        
        os.makedirs('performance_graphs', exist_ok=True)
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = tk.Label(self.window, text="🚁 Smart Drone Control (Optimized)", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        subtitle = tk.Label(self.window, text="Stable Flight with Adaptive Speed & Collision Recovery", 
                           font=("Arial", 10), fg="gray")
        subtitle.pack()
        
        # Battery display
        self.battery_label = tk.Label(self.window, text="🔋 Battery: 100.0% | Energy: 0.0 Wh", 
                                     font=("Arial", 11, "bold"), fg="green")
        self.battery_label.pack(pady=5)
        
        # Info frame
        info_frame = tk.Frame(self.window)
        info_frame.pack(pady=10)
        
        self.status_label = tk.Label(info_frame, text="Status: Not connected", 
                                     font=("Arial", 11, "bold"))
        self.status_label.grid(row=0, column=0, padx=15)
        
        self.position_label = tk.Label(info_frame, text="Position: (0.0, 0.0)", 
                                       font=("Arial", 10))
        self.position_label.grid(row=0, column=1, padx=15)
        
        self.speed_label = tk.Label(info_frame, text="Speed: -- m/s", 
                                   font=("Arial", 10), fg="blue")
        self.speed_label.grid(row=0, column=2, padx=15)
        
        # Canvas for map
        map_graph_container = tk.Frame(self.window)
        map_graph_container.pack(pady=10, fill="both", expand=True)
        
        # Left side: Map
        map_frame = tk.Frame(map_graph_container)
        map_frame.pack(side="left", padx=10)
        
        tk.Label(map_frame, text="🗺️ Flight Map", font=("Arial", 12, "bold")).pack()
        
        self.canvas = tk.Canvas(map_frame, width=self.map_size, height=self.map_size,
                               bg="white", highlightthickness=2, highlightbackground="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_map_click)
        
        # Right side: Live Graphs
        graph_frame = tk.Frame(map_graph_container, bg="#e8f4f8", relief="groove", borderwidth=2)
        graph_frame.pack(side="right", padx=10, fill="both", expand=True)
        
        graph_header = tk.Frame(graph_frame, bg="#e8f4f8")
        graph_header.pack(fill="x", padx=10, pady=5)
        
        tk.Label(graph_header, text="📊 Live Performance Metrics", 
                font=("Arial", 12, "bold"), bg="#e8f4f8").pack(side="left")
        
        self.capture_btn = tk.Button(graph_header, text="📷 Capture", 
                                     command=self.capture_graph,
                                     bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                                     width=12, state="disabled")
        self.capture_btn.pack(side="right", padx=5)
        
        # Create matplotlib figure
        self.graph_figure = Figure(figsize=(7, 6), dpi=80)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.init_live_graph()
        
        # Control buttons
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=15)
        
        self.start_btn = tk.Button(btn_frame, text="▶ Start Flight", 
                                   command=self.start_flight, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                   width=15, height=2)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.restart_btn = tk.Button(btn_frame, text="🏠 Restart (Home)", 
                                     command=self.restart_to_home, 
                                     bg="#FF9800", fg="white", font=("Arial", 12, "bold"),
                                     width=15, height=2, state="disabled")
        self.restart_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop Flight", 
                                 command=self.stop_flight, 
                                 bg="#F44336", fg="white", font=("Arial", 12, "bold"),
                                 width=15, height=2, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Features info
        features_frame = tk.Frame(self.window, bg="#f0f0f0", relief="groove", borderwidth=2)
        features_frame.pack(pady=10, padx=20, fill="x")
        
        features_title = tk.Label(features_frame, text="✨ Smart Features:", 
                                 font=("Arial", 10, "bold"), bg="#f0f0f0")
        features_title.pack(anchor="w", padx=10, pady=5)
        
        features = [
            "🚀 Adaptive Speed: Fast (5-8 m/s) on straight paths, Slow (2-3 m/s) on turns",
            "🛡️  Smart Recovery: Obstacle avoidance by climbing, moving left/right",
            "🔮 Collision Prediction: Uses depth sensors to predict obstacles",
            "🎯 Click Map: Set goals and navigate autonomously",
            "🔋 Battery & Energy: Real-time battery tracking",
            "📊 Live Graphs: Speed, Altitude, Battery, Energy during flight",
            "⚡ OPTIMIZED: Reduced API calls for stable connection"
        ]
        
        for feature in features:
            tk.Label(features_frame, text=feature, font=("Arial", 9), 
                    bg="#f0f0f0", anchor="w").pack(anchor="w", padx=20, pady=2)
        
        self.draw_grid()
    
    def init_live_graph(self):
        self.graph_figure.clear()
        ax = self.graph_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Start flight to see live metrics...', 
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.graph_canvas.draw()
    
    def update_live_graph(self):
        if not self.running or not self.start_time:
            return
        
        try:
            self.graph_figure.clear()
            
            ax1 = self.graph_figure.add_subplot(411)
            ax2 = self.graph_figure.add_subplot(412)
            ax3 = self.graph_figure.add_subplot(413)
            ax4 = self.graph_figure.add_subplot(414)
            
            if len(self.metrics['speeds']) > 0:
                ax1.plot(self.metrics['speeds'], color='#2196F3', linewidth=2)
                ax1.set_title('Speed (m/s)', fontweight='bold', fontsize=10)
                ax1.set_ylabel('m/s', fontsize=9)
                ax1.grid(alpha=0.3)
            
            if len(self.metrics['altitudes']) > 0:
                alts = [abs(a) for a in self.metrics['altitudes']]
                ax2.plot(alts, color='#FF9800', linewidth=2)
                ax2.set_title('Altitude (m)', fontweight='bold', fontsize=10)
                ax2.set_ylabel('m', fontsize=9)
                ax2.grid(alpha=0.3)
            
            if len(self.metrics['energy']) > 0:
                battery_history = [100.0 - (e / self.battery_capacity_wh * 100) for e in self.metrics['energy']]
                ax3.plot(battery_history, color='#4CAF50', linewidth=2)
                ax3.set_title('Battery Level (%)', fontweight='bold', fontsize=10)
                ax3.set_ylabel('%', fontsize=9)
                ax3.grid(alpha=0.3)
                ax3.set_ylim([0, 105])
            
            if len(self.metrics['energy']) > 0:
                ax4.plot(self.metrics['energy'], color='#FF5722', linewidth=2)
                ax4.set_title('Total Energy Consumed (Wh)', fontweight='bold', fontsize=10)
                ax4.set_ylabel('Wh', fontsize=9)
                ax4.set_xlabel('Steps', fontsize=9)
                ax4.grid(alpha=0.3)
            
            self.graph_figure.tight_layout()
            self.graph_canvas.draw()
        except Exception as e:
            print(f"Graph error: {e}")
    
    def capture_graph(self):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_graphs/capture_{timestamp}.png'
            self.graph_figure.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📷 Graph saved: {filename}")
            messagebox.showinfo("Saved!", f"Graph captured:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save: {e}")
    
    def draw_grid(self):
        for i in range(-100, 101, 20):
            x = self.world_to_canvas(i, 0)[0]
            y = self.world_to_canvas(0, i)[1]
            self.canvas.create_line(x, 0, x, self.map_size, fill="lightgray", dash=(2, 2))
            self.canvas.create_line(0, y, self.map_size, y, fill="lightgray", dash=(2, 2))
        
        center = self.map_size // 2
        self.canvas.create_line(center, 0, center, self.map_size, fill="gray", width=2)
        self.canvas.create_line(0, center, self.map_size, center, fill="gray", width=2)
        
        self.canvas.create_text(center + 5, 10, text="N", font=("Arial", 10, "bold"))
        self.canvas.create_text(self.map_size - 10, center - 5, text="E", font=("Arial", 10, "bold"))
    
    def world_to_canvas(self, x, y):
        canvas_x = int(self.map_size / 2 + x * self.scale)
        canvas_y = int(self.map_size / 2 - y * self.scale)
        return canvas_x, canvas_y
    
    def canvas_to_world(self, canvas_x, canvas_y):
        x = (canvas_x - self.map_size / 2) / self.scale
        y = -(canvas_y - self.map_size / 2) / self.scale
        return x, y
    
    def on_map_click(self, event):
        if not self.running:
            messagebox.showwarning("Not Ready", "Start flight first!")
            return
        
        goal_x, goal_y = self.canvas_to_world(event.x, event.y)
        self.goal_position = (goal_x, goal_y)
        
        print(f"\n🎯 Goal set: ({goal_x:.1f}, {goal_y:.1f})")
        
        self.canvas.delete("goal")
        canvas_x, canvas_y = self.world_to_canvas(goal_x, goal_y)
        self.canvas.create_oval(
            canvas_x - 10, canvas_y - 10,
            canvas_x + 10, canvas_y + 10,
            fill='red', outline='darkred', width=3, tags="goal"
        )
        
        self.flight_active = False
        time.sleep(0.3)
        self.navigating_to_goal = True
        threading.Thread(target=self._navigate_to_goal, daemon=True).start()
    
    def calculate_energy_consumption(self, speed, altitude_change, is_maneuvering):
        base_power = 15.0
        speed_power = 2.0 * (speed ** 1.5)
        
        if altitude_change < 0:
            climb_power = abs(altitude_change) * 50.0
        else:
            climb_power = abs(altitude_change) * 5.0
        
        maneuver_power = 10.0 if is_maneuvering else 0.0
        total_power = base_power + speed_power + climb_power + maneuver_power
        energy_wh = total_power * (0.5 / 3600.0)  # 0.5s update rate
        
        return energy_wh
    
    def start_flight(self):
        self.status_label.config(text="Status: Initializing...", fg="orange")
        self.window.update()
        threading.Thread(target=self._initialize_and_fly, daemon=True).start()
    
    def _initialize_and_fly(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print("📂 Loading model...")
            self.model = load_model("smart_airsim_model .pth", self.device)
            
            print("🔌 Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            print("🛫 Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(2)
            
            self.client.moveToZAsync(-3.0, 2.0).join()
            time.sleep(1)
            
            home_state = self.client.getMultirotorState()
            self.home_position = (home_state.kinematics_estimated.position.x_val,
                                 home_state.kinematics_estimated.position.y_val)
            print(f"🏠 Home: ({self.home_position[0]:.1f}, {self.home_position[1]:.1f})")
            
            hc = self.world_to_canvas(self.home_position[0], self.home_position[1])
            self.canvas.create_oval(hc[0]-8, hc[1]-8, hc[0]+8, hc[1]+8,
                                   fill='green', outline='darkgreen', width=3, tags="home")
            
            self.metrics = {
                'speeds': [], 
                'altitudes': [], 
                'collisions': 0, 
                'predictions': 0, 
                'warnings': 0,
                'energy': [],
                'timestamps': []
            }
            self.battery_percent = 100.0
            self.total_energy_consumed = 0.0
            self.start_time = time.time()
            
            self.running = True
            self.flight_active = True
            self.status_label.config(text="Status: Flying ✓", fg="green")
            self.start_btn.config(state="disabled")
            self.restart_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.capture_btn.config(state="normal")
            
            self.update_live_graph()
            
            self.update_position_loop()
            self._flight_control_loop()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)[:30]}", fg="red")
            print(f"Init error: {e}")
            import traceback
            traceback.print_exc()
    
    def _flight_control_loop(self):
        COLLISION_COOLDOWN = 2.0
        STARTING_ALTITUDE = -3.0
        MIN_HEIGHT_ABOVE_GROUND = 2.5
        YAW_RATE_SCALE = 60.0
        ALTITUDE_KP = 0.8
        UPDATE_RATE = 0.5  # INCREASED to 0.5s for stability
        
        last_collision_time = 0
        step = 0
        
        print("\n🚀 SMART FLIGHT STARTED")
        
        try:
            with torch.no_grad():
                while self.flight_active and self.running:
                    current_time = time.time()
                    
                    # Get state
                    try:
                        state = self.client.getMultirotorState()
                        pos = state.kinematics_estimated.position
                        
                        if pos.z_val > -MIN_HEIGHT_ABOVE_GROUND:
                            print(f"⚠️  TOO LOW - CLIMBING!")
                            self.metrics['warnings'] += 1
                            self.client.moveToZAsync(-MIN_HEIGHT_ABOVE_GROUND - 1.0, velocity=2.0)
                            time.sleep(1)
                            continue
                    except Exception as e:
                        print(f"State error: {type(e).__name__}")
                        time.sleep(0.3)
                        continue
                    
                    # Get image (less frequently)
                    if step % 2 == 0:  # Every other step
                        try:
                            responses = self.client.simGetImages([
                                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                            ])
                            if not responses or len(responses[0].image_data_uint8) == 0:
                                time.sleep(0.3)
                                continue
                            
                            image_tensor = preprocess_image(responses[0])
                            image_tensor = image_tensor.to(self.device)
                            steering_angle = self.model(image_tensor).item()
                            forward_speed = smart_speed_control(steering_angle, base_slow=2.0, base_fast=5.0)
                        except Exception as e:
                            print(f"Image error: {type(e).__name__}")
                            forward_speed = 3.0
                            steering_angle = 0
                    
                    if self.navigating_to_goal and self.goal_position:
                        self.metrics['speeds'].append(forward_speed)
                        self.metrics['altitudes'].append(pos.z_val)
                        self.metrics['timestamps'].append(time.time() - self.start_time)
                    
                    self.speed_label.config(text=f"Speed: {forward_speed:.1f} m/s")
                    
                    # Control
                    yaw_rate = np.clip(steering_angle * YAW_RATE_SCALE, -60.0, 60.0)
                    altitude_error = STARTING_ALTITUDE - pos.z_val
                    base_vz = float(np.clip(altitude_error * ALTITUDE_KP, -1.5, 1.5))
                    
                    try:
                        self.client.moveByVelocityBodyFrameAsync(
                            forward_speed, 0, base_vz, duration=UPDATE_RATE + 0.2,
                            yaw_mode=airsim.YawMode(True, yaw_rate)
                        )
                    except Exception as e:
                        print(f"Movement error: {type(e).__name__}")
                    
                    # Energy calculation
                    if step > 0:
                        prev_alt = self.metrics['altitudes'][-1] if len(self.metrics['altitudes']) > 0 else pos.z_val
                        altitude_change = (pos.z_val - prev_alt) / UPDATE_RATE
                        
                        energy_step = self.calculate_energy_consumption(forward_speed, altitude_change, False)
                        self.total_energy_consumed += energy_step
                        self.battery_percent = max(0.0, 100.0 - (self.total_energy_consumed / self.battery_capacity_wh * 100))
                        
                        if self.navigating_to_goal:
                            self.metrics['energy'].append(self.total_energy_consumed)
                        
                        battery_color = "green" if self.battery_percent > 50 else "orange" if self.battery_percent > 20 else "red"
                        self.battery_label.config(
                            text=f"🔋 Battery: {self.battery_percent:.1f}% | Energy: {self.total_energy_consumed:.2f} Wh",
                            fg=battery_color
                        )
                    
                    # Update graph
                    if step % 20 == 0 and self.navigating_to_goal:
                        self.update_live_graph()
                    
                    step += 1
                    time.sleep(UPDATE_RATE)
                    
        except Exception as e:
            print(f"Flight error: {e}")
        finally:
            self.flight_active = False
    
    def update_position_loop(self):
        if not self.running:
            return
        
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            self.drone_position = (pos.x_val, pos.y_val)
            
            self.position_label.config(text=f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f})")
            
            self.canvas.delete("drone")
            cx, cy = self.world_to_canvas(pos.x_val, pos.y_val)
            self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6,
                                   fill="blue", outline="darkblue", width=2, tags="drone")
            
            if len(self.path_points) == 0 or \
               np.sqrt((pos.x_val - self.path_points[-1][0])**2 + 
                      (pos.y_val - self.path_points[-1][1])**2) > 0.5:
                self.path_points.append((pos.x_val, pos.y_val))
                
                if len(self.path_points) > 1:
                    self.canvas.delete("path")
                    for i in range(len(self.path_points) - 1):
                        x1, y1 = self.world_to_canvas(*self.path_points[i])
                        x2, y2 = self.world_to_canvas(*self.path_points[i+1])
                        self.canvas.create_line(x1, y1, x2, y2, fill="lightblue", width=2, tags="path")
        except:
            pass
        
        self.window.after(200, self.update_position_loop)
    
    def restart_to_home(self):
        if not self.running or self.home_position is None:
            messagebox.showwarning("Not Ready", "Start flight first!")
            return
        
        print(f"\n🏠 Returning to home: {self.home_position}")
        self.status_label.config(text="Status: Returning home...", fg="orange")
        
        self.flight_active = False
        time.sleep(0.5)
        threading.Thread(target=self._navigate_to_home, daemon=True).start()
    
    def _navigate_to_home(self):
        FORWARD_VELOCITY = 3.0
        TARGET_ALTITUDE = -3.0
        goal_threshold = 2.0
        
        try:
            for step in range(500):
                if not self.running:
                    break
                
                try:
                    state = self.client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    current_x, current_y, current_z = pos.x_val, pos.y_val, pos.z_val
                except:
                    time.sleep(0.3)
                    continue
                
                dx = self.home_position[0] - current_x
                dy = self.home_position[1] - current_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < goal_threshold:
                    print(f"✅ HOME REACHED!")
                    self.client.moveByVelocityAsync(0, 0, 0, 1)
                    self.status_label.config(text="Status: Home ✓", fg="green")
                    break
                
                goal_angle = np.arctan2(dy, dx)
                orientation = state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(orientation)[2]
                
                yaw_error = goal_angle - current_yaw
                while yaw_error > np.pi:
                    yaw_error -= 2 * np.pi
                while yaw_error < -np.pi:
                    yaw_error += 2 * np.pi
                
                yaw_rate = np.clip(yaw_error * 0.8, -1.0, 1.0)
                alignment = 1.0 - min(abs(yaw_error) / np.pi, 1.0)
                vx = FORWARD_VELOCITY * alignment
                
                altitude_error = TARGET_ALTITUDE - current_z
                vz = np.clip(altitude_error * 0.5, -1.0, 1.0)
                
                try:
                    self.client.moveByVelocityAsync(
                        vx * np.cos(current_yaw),
                        vx * np.sin(current_yaw),
                        vz,
                        duration=0.5,
                        yaw_mode=airsim.YawMode(True, yaw_rate * 50)
                    )
                except:
                    pass
                
                if step % 10 == 0:
                    print(f"  Returning... Dist: {distance:.1f}m")
                
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Navigation error: {e}")
    
    def _navigate_to_goal(self):
        if self.goal_position is None:
            return
        
        self.metrics = {
            'speeds': [], 
            'altitudes': [], 
            'collisions': 0, 
            'predictions': 0, 
            'warnings': 0,
            'energy': [],
            'timestamps': []
        }
        self.battery_percent = 100.0
        self.total_energy_consumed = 0.0
        self.start_time = time.time()
        self.init_live_graph()
        
        FORWARD_VELOCITY = 3.0
        TARGET_ALTITUDE = -3.0
        goal_threshold = 2.0
        
        goal_x, goal_y = self.goal_position
        print(f"\n🎯 Navigating to goal: ({goal_x:.1f}, {goal_y:.1f})")
        self.status_label.config(text=f"Status: → Goal ({goal_x:.0f}, {goal_y:.0f})", fg="blue")
        
        try:
            for step in range(1000):
                if not self.running or not self.navigating_to_goal:
                    break
                
                try:
                    state = self.client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    current_x, current_y, current_z = pos.x_val, pos.y_val, pos.z_val
                except:
                    time.sleep(0.3)
                    continue
                
                dx = goal_x - current_x
                dy = goal_y - current_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < goal_threshold:
                    print(f"✅ GOAL REACHED!")
                    self.client.moveByVelocityAsync(0, 0, 0, 1)
                    self.status_label.config(text="Status: Goal reached ✓", fg="green")
                    
                    if len(self.metrics['speeds']) > 0:
                        print(f"\n📊 Navigation Complete:")
                        print(f"   Steps: {len(self.metrics['speeds'])}")
                        print(f"   Avg Speed: {np.mean(self.metrics['speeds']):.2f} m/s")
                        self.update_live_graph()
                    
                    self.navigating_to_goal = False
                    break
                
                goal_angle = np.arctan2(dy, dx)
                orientation = state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(orientation)[2]
                
                yaw_error = goal_angle - current_yaw
                while yaw_error > np.pi:
                    yaw_error -= 2 * np.pi
                while yaw_error < -np.pi:
                    yaw_error += 2 * np.pi
                
                yaw_rate = np.clip(yaw_error * 0.8, -1.0, 1.0)
                alignment = 1.0 - min(abs(yaw_error) / np.pi, 1.0)
                vx = FORWARD_VELOCITY * alignment
                
                altitude_error = TARGET_ALTITUDE - current_z
                vz = np.clip(altitude_error * 0.5, -1.0, 1.0)
                
                self.metrics['speeds'].append(vx)
                self.metrics['altitudes'].append(current_z)
                self.metrics['timestamps'].append(time.time() - self.start_time)
                
                if step > 0:
                    prev_alt = self.metrics['altitudes'][-2] if len(self.metrics['altitudes']) > 1 else current_z
                    altitude_change = (current_z - prev_alt) / 0.5
                    
                    energy_step = self.calculate_energy_consumption(vx, altitude_change, False)
                    self.total_energy_consumed += energy_step
                    self.battery_percent = max(0.0, 100.0 - (self.total_energy_consumed / self.battery_capacity_wh * 100))
                    self.metrics['energy'].append(self.total_energy_consumed)
                    
                    battery_color = "green" if self.battery_percent > 50 else "orange" if self.battery_percent > 20 else "red"
                    self.battery_label.config(
                        text=f"🔋 Battery: {self.battery_percent:.1f}% | Energy: {self.total_energy_consumed:.2f} Wh",
                        fg=battery_color
                    )
                
                if step % 20 == 0:
                    self.update_live_graph()
                
                try:
                    self.client.moveByVelocityAsync(
                        vx * np.cos(current_yaw),
                        vx * np.sin(current_yaw),
                        vz,
                        duration=0.5,
                        yaw_mode=airsim.YawMode(True, yaw_rate * 50)
                    )
                except:
                    pass
                
                if step % 10 == 0:
                    print(f"  → Goal... Dist: {distance:.1f}m | Yaw err: {np.degrees(yaw_error):+.0f}°")
                
                time.sleep(0.5)
            
        except Exception as e:
            print(f"Goal navigation error: {e}")
        finally:
            self.navigating_to_goal = False
    
    def stop_flight(self):
        self.flight_active = False
        self.running = False
        self.navigating_to_goal = False
        
        if self.goal_position and len(self.metrics['speeds']) > 0:
            print(f"\n📊 Flight Summary:")
            print(f"   Total Steps: {len(self.metrics['speeds'])}")
            print(f"   Avg Speed: {np.mean(self.metrics['speeds']):.2f} m/s")
            print(f"   Energy Used: {self.total_energy_consumed:.2f} Wh")
            print(f"   Battery Remaining: {self.battery_percent:.1f}%")
            self.update_live_graph()
        
        if self.client:
            try:
                print("\n🛬 Landing...")
                self.client.moveByVelocityAsync(0, 0, 0, 1).join()
                self.client.landAsync().join()
                time.sleep(2)
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
                print("✓ Landed")
            except:
                pass
        
        self.status_label.config(text="Status: Stopped", fg="red")
        self.stop_btn.config(state="disabled")
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = SmartDroneGUI()
    gui.run()
