"""
GUI 2: GNN vs MHA-PPO COMPARISON - Two drones race to goal
Drone1 = GNN, Drone2 = MHA-PPO
Simple comparison - both reach goal with RL and collision prediction
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

# GNN Actor
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

class GNN_Actor(nn.Module):
    def __init__(self):
        super(GNN_Actor, self).__init__()
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

# MHA-PPO Actor
class MHA_Actor(nn.Module):
    def __init__(self):
        super(MHA_Actor, self).__init__()
        self.embed = nn.Linear(8, 64)
        self.mha = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)
        self.collision_fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, state):
        x = self.embed(state)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.mha(x, x, x)
        x = attn_out.squeeze(1)
        h = self.relu(self.fc1(x))
        action = self.tanh(self.fc2(h))
        collision = self.sigmoid(self.collision_fc(h))
        return action, collision

class ComparisonGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🏆 ALGORITHM COMPARISON - GNN vs MHA-PPO")
        self.window.geometry("1600x900")
        self.window.configure(bg='#1a1a1a')
        
        self.client = None
        self.gnn_actor = GNN_Actor()
        self.mha_actor = MHA_Actor()
        
        # Flight state
        self.running = False
        self.goal_x = 100.0
        self.goal_y = 100.0
        self.altitude = -20.0
        
        # Drone 1 (GNN) - Orange
        self.drone1_pos = np.array([0.0, 0.0, 0.0])
        self.drone1_vel = np.array([0.0, 0.0, 0.0])
        self.drone1_path = []
        self.drone1_energy = 0.0
        self.drone1_battery = 100.0
        self.drone1_distance = 0.0
        self.drone1_finished = False
        self.drone1_finish_time = None
        
        # Drone 2 (MHA-PPO) - Green
        self.drone2_pos = np.array([0.0, 3.0, 0.0])
        self.drone2_vel = np.array([0.0, 0.0, 0.0])
        self.drone2_path = []
        self.drone2_energy = 0.0
        self.drone2_battery = 100.0
        self.drone2_distance = 0.0
        self.drone2_finished = False
        self.drone2_finish_time = None
        
        self.start_time = None
        self.run_number = 0
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header = tk.Frame(self.window, bg='#0d47a1', height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="🏆 ALGORITHM COMPARISON", font=("Arial", 22, "bold"), 
                fg="white", bg='#0d47a1').pack(pady=5)
        tk.Label(header, text="GNN vs MHA-PPO - Real-Time Comparison", font=("Arial", 11), 
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
        
        # Drone 1 (GNN) Stats
        drone1_frame = tk.LabelFrame(left, text="🟠 DRONE 1: GNN", font=("Arial", 12, "bold"),
                                    fg="#ff9800", bg='#2d2d2d', relief='ridge', borderwidth=3)
        drone1_frame.pack(pady=15, padx=15, fill='x')
        
        self.d1_status = tk.Label(drone1_frame, text="Status: Ready", font=("Arial", 10),
                                 fg="white", bg='#2d2d2d', anchor='w')
        self.d1_status.pack(fill='x', pady=3, padx=10)
        
        self.d1_pos = tk.Label(drone1_frame, text="Pos: (0.0, 0.0)", font=("Arial", 9),
                              fg="white", bg='#2d2d2d', anchor='w')
        self.d1_pos.pack(fill='x', pady=2, padx=10)
        
        self.d1_speed = tk.Label(drone1_frame, text="Speed: 0.0 m/s", font=("Arial", 9),
                                fg="white", bg='#2d2d2d', anchor='w')
        self.d1_speed.pack(fill='x', pady=2, padx=10)
        
        self.d1_dist = tk.Label(drone1_frame, text="Goal Dist: --", font=("Arial", 9),
                               fg="white", bg='#2d2d2d', anchor='w')
        self.d1_dist.pack(fill='x', pady=2, padx=10)
        
        self.d1_battery = tk.Label(drone1_frame, text="Battery: 100%", font=("Arial", 10, "bold"),
                                  fg="#4caf50", bg='#2d2d2d', anchor='w')
        self.d1_battery.pack(fill='x', pady=3, padx=10)
        
        self.d1_energy = tk.Label(drone1_frame, text="Energy: 0.0 Wh", font=("Arial", 9),
                                 fg="#90caf9", bg='#2d2d2d', anchor='w')
        self.d1_energy.pack(fill='x', pady=2, padx=10)
        
        # Drone 2 (MHA-PPO) Stats
        drone2_frame = tk.LabelFrame(left, text="🟢 DRONE 2: MHA-PPO", font=("Arial", 12, "bold"),
                                    fg="#4caf50", bg='#2d2d2d', relief='ridge', borderwidth=3)
        drone2_frame.pack(pady=15, padx=15, fill='x')
        
        self.d2_status = tk.Label(drone2_frame, text="Status: Ready", font=("Arial", 10),
                                 fg="white", bg='#2d2d2d', anchor='w')
        self.d2_status.pack(fill='x', pady=3, padx=10)
        
        self.d2_pos = tk.Label(drone2_frame, text="Pos: (0.0, 3.0)", font=("Arial", 9),
                              fg="white", bg='#2d2d2d', anchor='w')
        self.d2_pos.pack(fill='x', pady=2, padx=10)
        
        self.d2_speed = tk.Label(drone2_frame, text="Speed: 0.0 m/s", font=("Arial", 9),
                                fg="white", bg='#2d2d2d', anchor='w')
        self.d2_speed.pack(fill='x', pady=2, padx=10)
        
        self.d2_dist = tk.Label(drone2_frame, text="Goal Dist: --", font=("Arial", 9),
                               fg="white", bg='#2d2d2d', anchor='w')
        self.d2_dist.pack(fill='x', pady=2, padx=10)
        
        self.d2_battery = tk.Label(drone2_frame, text="Battery: 100%", font=("Arial", 10, "bold"),
                                  fg="#4caf50", bg='#2d2d2d', anchor='w')
        self.d2_battery.pack(fill='x', pady=3, padx=10)
        
        self.d2_energy = tk.Label(drone2_frame, text="Energy: 0.0 Wh", font=("Arial", 9),
                                 fg="#90caf9", bg='#2d2d2d', anchor='w')
        self.d2_energy.pack(fill='x', pady=2, padx=10)
        
        # Winner Display
        self.winner_frame = tk.Frame(left, bg='#1e1e1e', relief='sunken', borderwidth=3)
        self.winner_frame.pack(pady=20, padx=15, fill='x')
        
        tk.Label(self.winner_frame, text="🏁 RACE RESULTS", font=("Arial", 12, "bold"),
                fg="#ffeb3b", bg='#1e1e1e').pack(pady=5)
        
        self.winner_label = tk.Label(self.winner_frame, text="Start race to see results", 
                                    font=("Arial", 11), fg="white", bg='#1e1e1e')
        self.winner_label.pack(pady=5)
        
        self.time_label = tk.Label(self.winner_frame, text="", font=("Arial", 9),
                                  fg="#90caf9", bg='#1e1e1e')
        self.time_label.pack()
        
        self.energy_compare_label = tk.Label(self.winner_frame, text="", font=("Arial", 9),
                                           fg="#90caf9", bg='#1e1e1e')
        self.energy_compare_label.pack()
        
        # Run counter
        self.run_label = tk.Label(left, text="Run: #0", font=("Arial", 10),
                                 fg="white", bg='#2d2d2d')
        self.run_label.pack(pady=10)
        
        # Control buttons
        self.start_btn = tk.Button(left, text="🚀 START RACE", command=self.start_race,
                                  bg="#4caf50", fg="white", font=("Arial", 13, "bold"),
                                  width=18, height=2, cursor='hand2')
        self.start_btn.pack(pady=15)
        
        self.stop_btn = tk.Button(left, text="⏹ STOP", command=self.stop_race,
                                 bg="#f44336", fg="white", font=("Arial", 13, "bold"),
                                 width=18, height=2, state="disabled", cursor='hand2')
        self.stop_btn.pack(pady=5)
        
        # Right panel - Map
        right = tk.Frame(main, bg='#2d2d2d', relief='raised', borderwidth=2)
        right.pack(side='right', fill='both', expand=True)
        
        tk.Label(right, text="📍 LIVE RACE MAP", font=("Arial", 16, "bold"),
                fg="#4caf50", bg='#2d2d2d').pack(pady=15)
        
        self.fig = Figure(figsize=(12, 9), facecolor='#2d2d2d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=15, pady=15)
        
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        self.init_map()
    
    def init_map(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor='#1e1e1e')
        
        ax.scatter(0, 0, color='#ff9800', s=300, marker='o', edgecolors='white', linewidths=3, label='Drone1 (GNN)', zorder=5)
        ax.scatter(0, 3, color='#4caf50', s=300, marker='o', edgecolors='white', linewidths=3, label='Drone2 (MHA-PPO)', zorder=5)
        ax.scatter(self.goal_x, self.goal_y, color='#ff0000', s=500, marker='*', edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
        
        ax.set_title('RACE MAP (Click to set goal)', fontweight='bold', color='white', fontsize=14)
        ax.set_xlabel('X (m)', color='white', fontsize=11)
        ax.set_ylabel('Y (m)', color='white', fontsize=11)
        ax.grid(alpha=0.3, color='#90ee90', linestyle='--', linewidth=1)
        ax.legend(fontsize=10, facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white', loc='upper right')
        ax.tick_params(colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#404040')
        ax.set_xlim([-20, max(self.goal_x + 20, 120)])
        ax.set_ylim([-20, max(self.goal_y + 20, 120)])
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def set_goal(self):
        try:
            self.goal_x = float(self.goal_x_entry.get())
            self.goal_y = float(self.goal_y_entry.get())
            self.init_map()
            messagebox.showinfo("Goal Set", f"Goal: ({self.goal_x}, {self.goal_y})")
        except:
            messagebox.showerror("Error", "Invalid coordinates!")
    
    def on_map_click(self, event):
        if self.running or event.inaxes is None:
            return
        
        if event.xdata is not None and event.ydata is not None:
            self.goal_x = round(event.xdata, 1)
            self.goal_y = round(event.ydata, 1)
            self.goal_x_entry.delete(0, tk.END)
            self.goal_y_entry.delete(0, tk.END)
            self.goal_x_entry.insert(0, str(self.goal_x))
            self.goal_y_entry.insert(0, str(self.goal_y))
            self.init_map()
    
    def start_race(self):
        threading.Thread(target=self._race_loop, daemon=True).start()
    
    def stop_race(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        if self.client:
            try:
                self.client.landAsync(vehicle_name="Drone1")
                self.client.landAsync(vehicle_name="Drone2")
            except:
                pass
    
    def _race_loop(self):
        try:
            # Connect
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.run_number += 1
            self.run_label.config(text=f"Run: #{self.run_number}")
            
            # Reset
            self.drone1_path = []
            self.drone2_path = []
            self.drone1_energy = 0.0
            self.drone2_energy = 0.0
            self.drone1_battery = 100.0
            self.drone2_battery = 100.0
            self.drone1_finished = False
            self.drone2_finished = False
            self.drone1_finish_time = None
            self.drone2_finish_time = None
            
            self.winner_label.config(text="Race in progress...", fg="white")
            self.time_label.config(text="")
            self.energy_compare_label.config(text="")
            
            # Takeoff both
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.enableApiControl(True, vehicle_name="Drone2")
            self.client.armDisarm(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone2")
            self.client.takeoffAsync(vehicle_name="Drone1")
            self.client.takeoffAsync(vehicle_name="Drone2")
            time.sleep(3)
            self.client.moveToZAsync(self.altitude, 3.0, vehicle_name="Drone1")
            self.client.moveToZAsync(self.altitude, 3.0, vehicle_name="Drone2")
            time.sleep(3)
            
            self.start_time = time.time()
            
            # Start update threads
            threading.Thread(target=self._update_displays, daemon=True).start()
            threading.Thread(target=self._update_map, daemon=True).start()
            threading.Thread(target=self._drone1_navigation, daemon=True).start()
            threading.Thread(target=self._drone2_navigation, daemon=True).start()
            
            # Wait for both to finish
            while self.running:
                if self.drone1_finished and self.drone2_finished:
                    self.show_results()
                    break
                time.sleep(0.5)
            
            time.sleep(5)
            self.stop_race()
            
        except Exception as e:
            print(f"Race error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Race error:\n{str(e)}")
            self.stop_race()
    
    def _drone1_navigation(self):
        """GNN navigation for Drone1"""
        while self.running and not self.drone1_finished:
            try:
                state = self.client.getMultirotorState(vehicle_name="Drone1")
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                self.drone1_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                self.drone1_vel = np.array([vel.x_val, vel.y_val, vel.z_val])
                self.drone1_path.append(self.drone1_pos[:2].copy())
                
                goal_vec = np.array([self.goal_x, self.goal_y])
                to_goal = goal_vec - self.drone1_pos[:2]
                self.drone1_distance = np.linalg.norm(to_goal)
                
                if self.drone1_distance < 5.0:
                    self.drone1_finished = True
                    self.drone1_finish_time = time.time() - self.start_time
                    self.d1_status.config(text="Status: ✓ FINISHED", fg="#76ff03")
                    break
                
                # GNN features (2 nodes: drone1, drone2)
                state2 = self.client.getMultirotorState(vehicle_name="Drone2")
                pos2 = state2.kinematics_estimated.position
                drone2_pos = np.array([pos2.x_val, pos2.y_val])
                
                rel_pos = drone2_pos - self.drone1_pos[:2]
                distance_between = np.linalg.norm(rel_pos)
                
                # Node features
                node1 = [
                    to_goal[0] / 100.0,
                    to_goal[1] / 100.0,
                    self.drone1_vel[0] / 10.0,
                    self.drone1_vel[1] / 10.0,
                    distance_between / 50.0,
                    0.1
                ]
                
                node2 = [0.0, 0.0, 0.0, 0.0, distance_between / 50.0, 0.1]
                
                nodes = torch.FloatTensor([node1, node2]).unsqueeze(0)
                
                # Adjacency (connected if < 30m)
                adj = torch.FloatTensor([[1.0, 1.0 if distance_between < 30 else 0.0],
                                        [1.0 if distance_between < 30 else 0.0, 1.0]]).unsqueeze(0)
                
                with torch.no_grad():
                    action, collision = self.gnn_actor(nodes, adj)
                    action = action.squeeze(0)[0].numpy()
                
                direction = to_goal / (self.drone1_distance + 1e-6)
                vx = float(direction[0] * 0.7 + action[0] * 0.3) * 5.0
                vy = float(direction[1] * 0.7 + action[1] * 0.3) * 5.0
                
                target_x = pos.x_val + vx * 0.5
                target_y = pos.y_val + vy * 0.5
                
                self.client.moveToPositionAsync(target_x, target_y, self.altitude, 5.0, vehicle_name="Drone1").join()
                
                # Energy
                speed = np.linalg.norm(self.drone1_vel[:2])
                energy_step = (200 + speed ** 2 * 10) * 0.5 / 3600.0
                self.drone1_energy += energy_step
                self.drone1_battery = max(0, 100.0 - (self.drone1_energy / 100.0) * 100)
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Drone1 nav error: {e}")
                break
    
    def _drone2_navigation(self):
        """MHA-PPO navigation for Drone2"""
        while self.running and not self.drone2_finished:
            try:
                state = self.client.getMultirotorState(vehicle_name="Drone2")
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                self.drone2_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                self.drone2_vel = np.array([vel.x_val, vel.y_val, vel.z_val])
                self.drone2_path.append(self.drone2_pos[:2].copy())
                
                goal_vec = np.array([self.goal_x, self.goal_y])
                to_goal = goal_vec - self.drone2_pos[:2]
                self.drone2_distance = np.linalg.norm(to_goal)
                
                if self.drone2_distance < 5.0:
                    self.drone2_finished = True
                    self.drone2_finish_time = time.time() - self.start_time
                    self.d2_status.config(text="Status: ✓ FINISHED", fg="#76ff03")
                    break
                
                # MHA-PPO state
                mha_state = torch.FloatTensor([
                    to_goal[0] / 100.0,
                    to_goal[1] / 100.0,
                    self.drone2_vel[0] / 10.0,
                    self.drone2_vel[1] / 10.0,
                    2.0 / 5.0,  # wind_x
                    1.0 / 5.0,  # wind_y
                    self.drone2_battery / 100.0,
                    self.drone2_distance / 100.0
                ])
                
                with torch.no_grad():
                    action, collision = self.mha_actor(mha_state)
                    action = action.squeeze().numpy()
                
                direction = to_goal / (self.drone2_distance + 1e-6)
                vx = float(direction[0] * 0.7 + action[0] * 0.3) * 5.0
                vy = float(direction[1] * 0.7 + action[1] * 0.3) * 5.0
                
                target_x = pos.x_val + vx * 0.5
                target_y = pos.y_val + vy * 0.5
                
                self.client.moveToPositionAsync(target_x, target_y, self.altitude, 5.0, vehicle_name="Drone2").join()
                
                # Energy
                speed = np.linalg.norm(self.drone2_vel[:2])
                energy_step = (200 + speed ** 2 * 10) * 0.5 / 3600.0
                self.drone2_energy += energy_step
                self.drone2_battery = max(0, 100.0 - (self.drone2_energy / 100.0) * 100)
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Drone2 nav error: {e}")
                break
    
    def _update_displays(self):
        while self.running:
            try:
                # Drone1
                self.d1_pos.config(text=f"Pos: ({self.drone1_pos[0]:.1f}, {self.drone1_pos[1]:.1f})")
                self.d1_speed.config(text=f"Speed: {np.linalg.norm(self.drone1_vel[:2]):.1f} m/s")
                self.d1_dist.config(text=f"Goal Dist: {self.drone1_distance:.1f}m")
                
                battery_color1 = "#4caf50" if self.drone1_battery > 50 else "#ff9800" if self.drone1_battery > 20 else "#f44336"
                self.d1_battery.config(text=f"Battery: {self.drone1_battery:.1f}%", fg=battery_color1)
                self.d1_energy.config(text=f"Energy: {self.drone1_energy:.2f} Wh")
                
                # Drone2
                self.d2_pos.config(text=f"Pos: ({self.drone2_pos[0]:.1f}, {self.drone2_pos[1]:.1f})")
                self.d2_speed.config(text=f"Speed: {np.linalg.norm(self.drone2_vel[:2]):.1f} m/s")
                self.d2_dist.config(text=f"Goal Dist: {self.drone2_distance:.1f}m")
                
                battery_color2 = "#4caf50" if self.drone2_battery > 50 else "#ff9800" if self.drone2_battery > 20 else "#f44336"
                self.d2_battery.config(text=f"Battery: {self.drone2_battery:.1f}%", fg=battery_color2)
                self.d2_energy.config(text=f"Energy: {self.drone2_energy:.2f} Wh")
                
                time.sleep(0.1)
            except:
                pass
    
    def _update_map(self):
        while self.running:
            try:
                self.fig.clear()
                ax = self.fig.add_subplot(111, facecolor='#1e1e1e')
                
                # Goal
                ax.scatter(self.goal_x, self.goal_y, color='#ff0000', s=500, marker='*', edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
                
                # Start positions
                ax.scatter(0, 0, color='#ff9800', s=100, marker='x', linewidths=2, alpha=0.5)
                ax.scatter(0, 3, color='#4caf50', s=100, marker='x', linewidths=2, alpha=0.5)
                
                # Paths
                if len(self.drone1_path) > 1:
                    path1 = np.array(self.drone1_path)
                    ax.plot(path1[:, 0], path1[:, 1], color='#ff9800', linewidth=3, alpha=0.6, label='Drone1 (GNN)')
                
                if len(self.drone2_path) > 1:
                    path2 = np.array(self.drone2_path)
                    ax.plot(path2[:, 0], path2[:, 1], color='#4caf50', linewidth=3, alpha=0.6, label='Drone2 (MHA-PPO)')
                
                # Current positions
                if len(self.drone1_path) > 0:
                    ax.scatter(self.drone1_pos[0], self.drone1_pos[1], color='#ff9800', s=300, marker='o', 
                              edgecolors='white', linewidths=3, zorder=5)
                    ax.text(self.drone1_pos[0], self.drone1_pos[1] + 5, 'D1', fontsize=10, fontweight='bold',
                           color='white', ha='center', bbox=dict(boxstyle='round', facecolor='#ff9800', alpha=0.8))
                
                if len(self.drone2_path) > 0:
                    ax.scatter(self.drone2_pos[0], self.drone2_pos[1], color='#4caf50', s=300, marker='o',
                              edgecolors='white', linewidths=3, zorder=5)
                    ax.text(self.drone2_pos[0], self.drone2_pos[1] + 5, 'D2', fontsize=10, fontweight='bold',
                           color='white', ha='center', bbox=dict(boxstyle='round', facecolor='#4caf50', alpha=0.8))
                
                ax.set_title('RACE MAP', fontweight='bold', color='white', fontsize=14)
                ax.set_xlabel('X (m)', color='white', fontsize=11)
                ax.set_ylabel('Y (m)', color='white', fontsize=11)
                ax.grid(alpha=0.3, color='#90ee90', linestyle='--', linewidth=1)
                ax.legend(fontsize=10, facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white', loc='upper right')
                ax.tick_params(colors='white', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_color('#404040')
                
                self.fig.tight_layout()
                self.canvas.draw()
                
                time.sleep(0.5)
            except:
                pass
    
    def show_results(self):
        """Display race results"""
        if self.drone1_finish_time and self.drone2_finish_time:
            if self.drone1_finish_time < self.drone2_finish_time:
                winner = "🟠 DRONE 1 (GNN) WINS! 🏆"
                winner_color = "#ff9800"
            elif self.drone2_finish_time < self.drone1_finish_time:
                winner = "🟢 DRONE 2 (MHA-PPO) WINS! 🏆"
                winner_color = "#4caf50"
            else:
                winner = "🤝 TIE!"
                winner_color = "#ffeb3b"
            
            self.winner_label.config(text=winner, fg=winner_color)
            self.time_label.config(text=f"D1: {self.drone1_finish_time:.1f}s | D2: {self.drone2_finish_time:.1f}s")
            self.energy_compare_label.config(text=f"Energy - D1: {self.drone1_energy:.2f}Wh | D2: {self.drone2_energy:.2f}Wh")
            
            messagebox.showinfo("Race Finished!", 
                              f"{winner}\n\n"
                              f"Drone1 (GNN): {self.drone1_finish_time:.1f}s, {self.drone1_energy:.2f}Wh\n"
                              f"Drone2 (MHA-PPO): {self.drone2_finish_time:.1f}s, {self.drone2_energy:.2f}Wh")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ComparisonGUI()
    app.run()
