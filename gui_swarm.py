"""
GUI 3: MULTI-DRONE SWARM with Communication & Energy
Main drone (Drone1) navigates to goal while communicating with swarm
Shows drone positions, communication network, and energy consumption
"""

import airsim
import torch
import torch.nn as nn
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
import threading
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#GNN Actor for main drone
class GNN_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN_Layer, self).__init__()
        self.message_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, node_features, adj_matrix):
        messages = self.message_net(node_features)
        aggregated = torch.matmul(adj_matrix, messages)
        combined = torch.cat([node_features, aggregated], dim=2)
        updated = self.update_net(combined)
        return updated

class SwarmGNN_Actor(nn.Module):
    def __init__(self):
        super(SwarmGNN_Actor, self).__init__()
        self.gnn1 = GNN_Layer(6, 64)
        self.gnn2 = GNN_Layer(64, 64)
        self.action_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        self.collision_net = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, adj_matrix):
        h1 = self.gnn1(node_features, adj_matrix)
        h2 = self.gnn2(h1, adj_matrix)
        action = self.action_net(h2)
        collision = self.collision_net(h2)
        return action, collision

class SwarmGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🌐 MULTI-DRONE SWARM - Communication & Energy")
        self.window.geometry("1600x900")
        self.window.configure(bg='#1a1a1a')
        
        self.client = None
        self.actor = SwarmGNN_Actor()
        
        # Drone configuration
        self.main_drone = "Drone1"
        self.background_drones = [f"Drone{i}" for i in range(2, 7)]  # 5 background drones
        
        # Flight state
        self.running = False
        self.goal_x = 100.0
        self.goal_y = 100.0
        self.altitude = -20.0
        
        # Main drone state
        self.main_pos = np.array([0.0, 0.0, 0.0])
        self.main_vel = np.array([0.0, 0.0, 0.0])
        self.main_path = []
        self.main_energy = 0.0
        self.main_battery = 100.0
        
        # All drone positions and energies
        self.drone_positions = {}
        self.drone_energies = {}
        for d in [self.main_drone] + self.background_drones:
            self.drone_positions[d] = [0, 0]
            self.drone_energies[d] = 0.0
        
        # Communication
        self.communications = []
        self.comm_range = 30.0
        
        # Metrics
        self.metrics = {
            'time': [],
            'comm_count': [],
            'collision_risk': []
        }
        self.start_time = None
        self.run_number = 0
        self.stop_event = threading.Event()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header = tk.Frame(self.window, bg='#0d47a1', height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="🌐 MULTI-DRONE SWARM NAVIGATION", font=("Arial", 22, "bold"), 
                fg="white", bg='#0d47a1').pack(pady=5)
        tk.Label(header, text="GNN-Based Swarm with Real-Time Communication", font=("Arial", 11), 
                fg="#90caf9", bg='#0d47a1').pack()
        
        # Main container
        main = tk.Frame(self.window, bg='#1a1a1a')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel
        left = tk.Frame(main, bg='#2d2d2d', width=380, relief='raised', borderwidth=2)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        
        # Goal Setting
        tk.Label(left, text="🎯 GOAL POSITION", font=("Arial", 13, "bold"),
                fg="#ff9800", bg='#2d2d2d').pack(pady=15)
        
        tk.Label(left, text="💡 Click on map to set goal", font=("Arial", 9, "italic"),
                fg="#90caf9", bg='#2d2d2d').pack()
        
        goal_frame = tk.Frame(left, bg='#2d2d2d')
        goal_frame.pack(pady=10)
        
        tk.Label(goal_frame, text="X:", font=("Arial", 10), fg="white", bg='#2d2d2d').grid(row=0, column=0, padx=5)
        self.goal_x_entry = tk.Entry(goal_frame, width=10, font=("Arial", 10), bg='#1e1e1e', fg='white', insertbackground='white')
        self.goal_x_entry.grid(row=0, column=1, padx=5)
        self.goal_x_entry.insert(0, "100")
        
        tk.Label(goal_frame, text="Y:", font=("Arial", 10), fg="white", bg='#2d2d2d').grid(row=0, column=2, padx=5)
        self.goal_y_entry = tk.Entry(goal_frame, width=10, font=("Arial", 10), bg='#1e1e1e', fg='white', insertbackground='white')
        self.goal_y_entry.grid(row=0, column=3, padx=5)
        self.goal_y_entry.insert(0, "100")
        
        tk.Button(left, text="✓ SET GOAL", command=self.set_goal, bg="#2196f3", fg="white",
                 font=("Arial", 10, "bold"), width=18, cursor='hand2').pack(pady=5)
        
        # Swarm Status
        tk.Label(left, text="🌐 SWARM STATUS", font=("Arial", 13, "bold"),
                fg="#4caf50", bg='#2d2d2d').pack(pady=(20, 10))
        
        status_frame = tk.Frame(left, bg='#1e1e1e', relief='sunken', borderwidth=2)
        status_frame.pack(pady=5, padx=15, fill='x')
        
        self.swarm_status = tk.Label(status_frame, text="Active: 0/6 drones", font=("Arial", 10),
                                    fg="white", bg='#1e1e1e', anchor='w')
        self.swarm_status.pack(fill='x', pady=3, padx=10)
        
        self.comm_status = tk.Label(status_frame, text="Communications: 0", font=("Arial", 10),
                                   fg="white", bg='#1e1e1e', anchor='w')
        self.comm_status.pack(fill='x', pady=3, padx=10)
        
        self.range_status = tk.Label(status_frame, text="Comm Range: 30m", font=("Arial", 10),
                                    fg="white", bg='#1e1e1e', anchor='w')
        self.range_status.pack(fill='x', pady=3, padx=10)
        
        # Main Drone Telemetry
        tk.Label(left, text="🚁 MAIN DRONE (Drone1)", font=("Arial", 13, "bold"),
                fg="#ff9800", bg='#2d2d2d').pack(pady=(20, 10))
        
        telem_frame = tk.Frame(left, bg='#2d2d2d')
        telem_frame.pack(pady=5, fill='x', padx=20)
        
        self.pos_label = tk.Label(telem_frame, text="Pos: (0.0, 0.0)", font=("Arial", 9),
                                 fg="white", bg='#2d2d2d', anchor='w')
        self.pos_label.pack(fill='x', pady=2)
        
        self.speed_label = tk.Label(telem_frame, text="Speed: 0.0 m/s", font=("Arial", 9),
                                   fg="white", bg='#2d2d2d', anchor='w')
        self.speed_label.pack(fill='x', pady=2)
        
        self.dist_label = tk.Label(telem_frame, text="Goal Dist: --", font=("Arial", 9),
                                  fg="white", bg='#2d2d2d', anchor='w')
        self.dist_label.pack(fill='x', pady=2)
        
        self.collision_label = tk.Label(telem_frame, text="Collision Risk: 0%", font=("Arial", 9),
                                       fg="white", bg='#2d2d2d', anchor='w')
        self.collision_label.pack(fill='x', pady=2)
        
        # Battery
        tk.Label(left, text="🔋 BATTERY", font=("Arial", 12, "bold"),
                fg="#ff9800", bg='#2d2d2d').pack(pady=(15, 5))
        
        self.battery_label = tk.Label(left, text="100%", font=("Arial", 18, "bold"),
                                     fg="#4caf50", bg='#2d2d2d')
        self.battery_label.pack()
        
        self.energy_label = tk.Label(left, text="Energy: 0.0 Wh", font=("Arial", 9),
                                    fg="#90caf9", bg='#2d2d2d')
        self.energy_label.pack()
        
        # RL Stats
        tk.Label(left, text="🧠 RL LEARNING", font=("Arial", 12, "bold"),
                fg="#9c27b0", bg='#2d2d2d').pack(pady=(15, 5))
        
        self.run_label = tk.Label(left, text="Run: #0", font=("Arial", 10),
                                 fg="white", bg='#2d2d2d')
        self.run_label.pack()
        
        # Status
        self.status_label = tk.Label(left, text="● Ready", font=("Arial", 11),
                                    fg="#90caf9", bg='#2d2d2d')
        self.status_label.pack(pady=20)
        
        # Control buttons
        self.start_btn = tk.Button(left, text="🚀 START SWARM", command=self.start_swarm,
                                  bg="#4caf50", fg="white", font=("Arial", 12, "bold"),
                                  width=18, height=2, cursor='hand2')
        self.start_btn.pack(pady=10)
        
        self.stop_btn = tk.Button(left, text="⏹ STOP", command=self.stop_swarm,
                                 bg="#f44336", fg="white", font=("Arial", 12, "bold"),
                                 width=18, height=2, state="disabled", cursor='hand2')
        self.stop_btn.pack(pady=5)
        
        # Right panel - Graphs
        right = tk.Frame(main, bg='#2d2d2d', relief='raised', borderwidth=2)
        right.pack(side='right', fill='both', expand=True)
        
        tk.Label(right, text="📊 SWARM MAP & ANALYTICS", font=("Arial", 15, "bold"),
                fg="#4caf50", bg='#2d2d2d').pack(pady=10)
        
        self.fig = Figure(figsize=(11, 8), facecolor='#2d2d2d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        self.init_graphs()
    
    def init_graphs(self):
        self.fig.clear()
        
        # 3 subplots: map (large), communication graph, energy graph
        ax1 = self.fig.add_subplot(2, 2, (1, 3), facecolor='#1e1e1e')  # Large map
        ax2 = self.fig.add_subplot(2, 2, 2, facecolor='#1e1e1e')       # Communication
        ax3 = self.fig.add_subplot(2, 2, 4, facecolor='#1e1e1e')       # Energy
        
        # Map
        ax1.scatter(0, 0, color='#ff9800', s=300, marker='o', edgecolors='white', linewidths=3, label='Main Drone')
        ax1.scatter(self.goal_x, self.goal_y, color='#ff0000', s=500, marker='*', edgecolors='yellow', linewidths=3, label='Goal')
        ax1.set_title('SWARM MAP (Click to set goal)', fontweight='bold', color='white', fontsize=12)
        ax1.set_xlabel('X (m)', color='white', fontsize=10)
        ax1.set_ylabel('Y (m)', color='white', fontsize=10)
        ax1.grid(alpha=0.3, color='#90ee90', linestyle='--')
        ax1.legend(fontsize=9, facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
        ax1.tick_params(colors='white', labelsize=9)
        for spine in ax1.spines.values():
            spine.set_color('#404040')
        ax1.set_xlim([-20, max(self.goal_x + 20, 120)])
        ax1.set_ylim([-20, max(self.goal_y + 20, 120)])
        
        # Communication graph
        ax2.text(0.5, 0.5, 'Start flight...', ha='center', va='center',
                fontsize=10, color='#757575', transform=ax2.transAxes)
        ax2.set_title('COMMUNICATION', fontweight='bold', color='white', fontsize=10)
        ax2.tick_params(colors='white', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color('#404040')
        
        # Energy graph
        ax3.text(0.5, 0.5, 'Start flight...', ha='center', va='center',
                fontsize=10, color='#757575', transform=ax3.transAxes)
        ax3.set_title('ENERGY', fontweight='bold', color='white', fontsize=10)
        ax3.tick_params(colors='white', labelsize=8)
        for spine in ax3.spines.values():
            spine.set_color('#404040')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def set_goal(self):
        try:
            self.goal_x = float(self.goal_x_entry.get())
            self.goal_y = float(self.goal_y_entry.get())
            self.init_graphs()
            messagebox.showinfo("Goal Set", f"Goal: ({self.goal_x}, {self.goal_y})")
        except:
            messagebox.showerror("Error", "Invalid coordinates!")
    
    def on_map_click(self, event):
        if self.running or event.inaxes is None:
            return
        
        try:
            axes = self.fig.get_axes()
            if len(axes) >= 1 and event.inaxes == axes[0]:
                if event.xdata is not None and event.ydata is not None:
                    self.goal_x = round(event.xdata, 1)
                    self.goal_y = round(event.ydata, 1)
                    self.goal_x_entry.delete(0, tk.END)
                    self.goal_y_entry.delete(0, tk.END)
                    self.goal_x_entry.insert(0, str(self.goal_x))
                    self.goal_y_entry.insert(0, str(self.goal_y))
                    self.init_graphs()
        except:
            pass
    
    def start_swarm(self):
        self.status_label.config(text="● Initializing...", fg="orange")
        self.window.update()
        threading.Thread(target=self._swarm_loop, daemon=True).start()
    
    def stop_swarm(self):
        self.running = False
        self.stop_event.set()
        self.status_label.config(text="● Stopped", fg="#f44336")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        if self.client:
            try:
                for drone in [self.main_drone] + self.background_drones:
                    try:
                        self.client.landAsync(vehicle_name=drone)
                    except:
                        pass
            except:
                pass
    
    def _swarm_loop(self):
        try:
            # Connect
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_label.config(text="● Flying", fg="#4caf50")
            
            # Reset
            self.metrics = {'time': [], 'comm_count': [], 'collision_risk': []}
            self.start_time = time.time()
            self.main_energy = 0.0
            self.main_battery = 100.0
            self.main_path = []
            self.communications = []
            self.run_number += 1
            self.stop_event.clear()
            
            for d in [self.main_drone] + self.background_drones:
                self.drone_energies[d] = 0.0
            
            # Takeoff main drone
            self.client.enableApiControl(True, vehicle_name=self.main_drone)
            self.client.armDisarm(True, vehicle_name=self.main_drone)
            self.client.takeoffAsync(vehicle_name=self.main_drone).join()
            self.client.moveToZAsync(self.altitude, 3.0, vehicle_name=self.main_drone).join()
            
            # Start background drones
            for drone in self.background_drones:
                threading.Thread(target=self._background_drone_thread, args=(drone,), daemon=True).start()
                time.sleep(0.3)
            
            # Start update threads
            threading.Thread(target=self._update_displays, daemon=True).start()
            threading.Thread(target=self._update_graphs, daemon=True).start()
            
            # Main navigation loop
            while self.running:
                # Get main drone state
                state = self.client.getMultirotorState(vehicle_name=self.main_drone)
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                self.main_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                self.main_vel = np.array([vel.x_val, vel.y_val, vel.z_val])
                self.main_path.append(self.main_pos[:2].copy())
                self.drone_positions[self.main_drone] = self.main_pos[:2].tolist()
                
                # Calculate to goal
                goal_vec = np.array([self.goal_x, self.goal_y])
                to_goal = goal_vec - self.main_pos[:2]
                dist = np.linalg.norm(to_goal)
                
                # Check if reached
                if dist < 5.0:
                    self.status_label.config(text="● Goal Reached!", fg="#76ff03")
                    messagebox.showinfo("Success", f"🎉 Goal reached!\nRun #{self.run_number}\nEnergy: {self.main_energy:.2f} Wh")
                    break
                
                # Get all drone positions for GNN
                all_drones = [self.main_drone] + [d for d in self.background_drones if d in self.drone_positions]
                num_drones = len(all_drones)
                
                # Build adjacency matrix (communication within range)
                adj_matrix = np.zeros((num_drones, num_drones))
                comm_count = 0
                
                for i, d_i in enumerate(all_drones):
                    for j, d_j in enumerate(all_drones):
                        if i != j:
                            pos_i = np.array(self.drone_positions[d_i])
                            pos_j = np.array(self.drone_positions[d_j])
                            distance = np.linalg.norm(pos_i - pos_j)
                            
                            if distance < self.comm_range:
                                adj_matrix[i, j] = 1.0
                                if d_i == self.main_drone:
                                    comm_count += 1
                                    self.communications.append({
                                        'time': time.time(),
                                        'from': d_i,
                                        'to': d_j,
                                        'distance': distance
                                    })
                
                # Build node features
                node_features = []
                min_dist_to_obstacle = 999.0
                
                for drone in all_drones:
                    pos_drone = np.array(self.drone_positions[drone])
                    
                    goal_rel = goal_vec - pos_drone
                    
                    # Find nearest other drone
                    min_dist = 999.0
                    nearest_angle = 0.0
                    for other_drone in all_drones:
                        if other_drone != drone:
                            other_pos = np.array(self.drone_positions[other_drone])
                            diff = other_pos - pos_drone
                            d = np.linalg.norm(diff)
                            if d < min_dist:
                                min_dist = d
                                if d > 0:
                                    nearest_angle = np.arctan2(diff[1], diff[0])
                    
                    if drone == self.main_drone:
                        min_dist_to_obstacle = min_dist
                    
                    features = [
                        goal_rel[0] / 100.0,
                        goal_rel[1] / 100.0,
                        0.0,  # velocity (simplified)
                        0.0,
                        min_dist / 50.0,
                        nearest_angle / np.pi
                    ]
                    node_features.append(features)
                
                # GNN inference
                state_tensor = torch.FloatTensor(node_features).unsqueeze(0)
                adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)
                
                with torch.no_grad():
                    actions, collision_probs = self.actor(state_tensor, adj_tensor)
                    main_action = actions.squeeze(0)[0].numpy()
                    collision_risk = float(collision_probs.squeeze(0)[0].item())
                
                # Navigate
                direction = to_goal / (dist + 1e-6)
                
                gnn_weight = 0.2 + (collision_risk * 0.3)
                goal_weight = 1.0 - gnn_weight
                
                vx = float(direction[0] * goal_weight + main_action[0] * gnn_weight) * 5.0
                vy = float(direction[1] * goal_weight + main_action[1] * gnn_weight) * 5.0
                
                target_x = pos.x_val + vx * 0.5
                target_y = pos.y_val + vy * 0.5
                
                self.client.moveToPositionAsync(target_x, target_y, self.altitude, 5.0, vehicle_name=self.main_drone).join()
                
                # Update metrics
                elapsed = time.time() - self.start_time
                speed = np.linalg.norm(self.main_vel[:2])
                
                # Energy
                energy_step = (200 + speed ** 2 * 10) * 0.5 / 3600.0
                self.main_energy += energy_step
                self.main_battery = max(0, 100.0 - (self.main_energy / 100.0) * 100)
                self.drone_energies[self.main_drone] = self.main_energy
                
                self.metrics['time'].append(elapsed)
                self.metrics['comm_count'].append(comm_count)
                self.metrics['collision_risk'].append(collision_risk * 100)
                
                time.sleep(0.1)
            
            self.stop_swarm()
            
        except Exception as e:
            print(f"Swarm error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Swarm error:\n{str(e)}")
            self.stop_swarm()
    
    def _background_drone_thread(self, drone_name):
        """Background drone random movement"""
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            
            client.enableApiControl(True, vehicle_name=drone_name)
            client.armDisarm(True, vehicle_name=drone_name)
            client.takeoffAsync(vehicle_name=drone_name).join()
            client.moveToZAsync(self.altitude, 3.0, vehicle_name=drone_name).join()
            
            energy = 0.0
            
            while not self.stop_event.is_set():
                state = client.getMultirotorState(vehicle_name=drone_name)
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                self.drone_positions[drone_name] = [pos.x_val, pos.y_val]
                
                # Random movement
                dx = np.random.uniform(-30, 30)
                dy = np.random.uniform(-30, 30)
                target_x = pos.x_val + dx
                target_y = pos.y_val + dy
                
                client.moveToPositionAsync(target_x, target_y, self.altitude, np.random.uniform(3.0, 6.0), vehicle_name=drone_name).join()
                
                # Energy
                speed = np.sqrt(vel.x_val**2 + vel.y_val**2)
                energy_step = (200 + speed ** 2 * 10) * 0.5 / 3600.0
                energy += energy_step
                self.drone_energies[drone_name] = energy
                
                time.sleep(np.random.uniform(2.0, 4.0))
        except:
            pass
    
    def _update_displays(self):
        while self.running:
            try:
                # Swarm status
                active_drones = len([d for d in [self.main_drone] + self.background_drones if d in self.drone_positions])
                self.swarm_status.config(text=f"Active: {active_drones}/{len([self.main_drone] + self.background_drones)} drones")
                
                recent_comms = [c for c in self.communications if time.time() - c['time'] < 5.0]
                self.comm_status.config(text=f"Communications: {len(recent_comms)}")
                
                # Main drone
                self.pos_label.config(text=f"Pos: ({self.main_pos[0]:.1f}, {self.main_pos[1]:.1f})")
                self.speed_label.config(text=f"Speed: {np.linalg.norm(self.main_vel[:2]):.1f} m/s")
                
                dist = np.linalg.norm(np.array([self.goal_x, self.goal_y]) - self.main_pos[:2])
                self.dist_label.config(text=f"Goal Dist: {dist:.1f}m")
                
                if len(self.metrics['collision_risk']) > 0:
                    risk = self.metrics['collision_risk'][-1]
                    risk_color = "#4caf50" if risk < 30 else "#ff9800" if risk < 70 else "#f44336"
                    self.collision_label.config(text=f"Collision Risk: {risk:.0f}%", fg=risk_color)
                
                # Battery
                battery_color = "#4caf50" if self.main_battery > 50 else "#ff9800" if self.main_battery > 20 else "#f44336"
                self.battery_label.config(text=f"{self.main_battery:.1f}%", fg=battery_color)
                self.energy_label.config(text=f"Energy: {self.main_energy:.2f} Wh")
                
                # RL
                self.run_label.config(text=f"Run: #{self.run_number}")
                
                time.sleep(0.1)
            except:
                pass
    
    def _update_graphs(self):
        while self.running:
            try:
                if len(self.metrics['time']) > 2:
                    self.fig.clear()
                    times = self.metrics['time']
                    
                    # Swarm Map (large)
                    ax1 = self.fig.add_subplot(2, 2, (1, 3), facecolor='#1e1e1e')
                    
                    # Goal
                    ax1.scatter(self.goal_x, self.goal_y, color='#ff0000', s=500, marker='*', edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
                    
                    # Main drone path
                    if len(self.main_path) > 1:
                        path = np.array(self.main_path)
                        ax1.plot(path[:, 0], path[:, 1], color='#ff9800', linewidth=3, alpha=0.8, zorder=5)
                    
                    # Main drone
                    ax1.scatter(self.main_pos[0], self.main_pos[1], color='#ff9800', s=300, marker='o', 
                               edgecolors='white', linewidths=3, label='Main Drone', zorder=6)
                    
                    # Background drones
                    for drone in self.background_drones:
                        if drone in self.drone_positions:
                            pos = self.drone_positions[drone]
                            ax1.scatter(pos[0], pos[1], color='#2196f3', s=150, marker='o',
                                       edgecolors='white', linewidths=2, alpha=0.7, zorder=4)
                    
                    # Communication lines (recent)
                    recent_comms = [c for c in self.communications if time.time() - c['time'] < 3.0]
                    drawn_pairs = set()
                    for comm in recent_comms:
                        pair = tuple(sorted([comm['from'], comm['to']]))
                        if pair not in drawn_pairs and comm['from'] in self.drone_positions and comm['to'] in self.drone_positions:
                            pos_from = self.drone_positions[comm['from']]
                            pos_to = self.drone_positions[comm['to']]
                            ax1.plot([pos_from[0], pos_to[0]], [pos_from[1], pos_to[1]], 
                                    color='#00ffff', linewidth=1.5, alpha=0.4, zorder=3)
                            drawn_pairs.add(pair)
                    
                    ax1.set_title('SWARM MAP (Cyan lines = Communication)', fontweight='bold', color='white', fontsize=11)
                    ax1.set_xlabel('X (m)', color='white', fontsize=9)
                    ax1.set_ylabel('Y (m)', color='white', fontsize=9)
                    ax1.grid(alpha=0.3, color='#90ee90', linestyle='--')
                    ax1.legend(fontsize=8, facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
                    ax1.tick_params(colors='white', labelsize=9)
                    for spine in ax1.spines.values():
                        spine.set_color('#404040')
                    
                    # Communication Graph
                    ax2 = self.fig.add_subplot(2, 2, 2, facecolor='#1e1e1e')
                    ax2.plot(times, self.metrics['comm_count'], color='#00bcd4', linewidth=2.5)
                    ax2.set_title('COMMUNICATION COUNT', fontweight='bold', fontsize=10, color='white')
                    ax2.set_ylabel('Count', fontsize=8, color='white')
                    ax2.set_xlabel('Time (s)', fontsize=8, color='white')
                    ax2.grid(alpha=0.2, color='#404040')
                    ax2.tick_params(colors='white', labelsize=8)
                    for spine in ax2.spines.values():
                        spine.set_color('#404040')
                    
                    # Energy Graph (all drones)
                    ax3 = self.fig.add_subplot(2, 2, 4, facecolor='#1e1e1e')
                    
                    # Bar chart of energy consumption
                    drones = list(self.drone_energies.keys())
                    energies = [self.drone_energies[d] for d in drones]
                    colors = ['#ff9800' if d == self.main_drone else '#2196f3' for d in drones]
                    
                    ax3.bar(range(len(drones)), energies, color=colors, edgecolor='white', linewidth=1.5)
                    ax3.set_title('ENERGY CONSUMPTION', fontweight='bold', fontsize=10, color='white')
                    ax3.set_ylabel('Energy (Wh)', fontsize=8, color='white')
                    ax3.set_xlabel('Drone', fontsize=8, color='white')
                    ax3.set_xticks(range(len(drones)))
                    ax3.set_xticklabels([d.replace('Drone', 'D') for d in drones], rotation=45, fontsize=7)
                    ax3.grid(alpha=0.2, color='#404040', axis='y')
                    ax3.tick_params(colors='white', labelsize=8)
                    for spine in ax3.spines.values():
                        spine.set_color('#404040')
                    
                    self.fig.tight_layout()
                    self.canvas.draw()
                
                time.sleep(0.5)
            except:
                pass
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = SwarmGUI()
    app.run()
