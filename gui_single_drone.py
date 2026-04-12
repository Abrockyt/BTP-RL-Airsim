"""
GUI 1: SINGLE DRONE - Vision, Map, Performance Analytics, Wind, Goal Setting
Simple and focused - ONE drone reaches goal with RL and collision prediction
"""

# Fix IOLoop conflict FIRST - before importing airsim
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("⚠️ Installing nest_asyncio to fix event loop conflicts...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nest-asyncio"])
    import nest_asyncio
    nest_asyncio.apply()

import airsim
import torch
import torch.nn as nn
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# GNN Layer for Graph Neural Network
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
        # Message passing
        messages = self.message_net(node_features)
        # Aggregate messages from neighbors
        aggregated = torch.matmul(adj_matrix, messages)
        # Update node features
        combined = torch.cat([node_features, aggregated], dim=2)
        updated = self.update_net(combined)
        return updated

# GNN Actor with Collision Prediction
class GNN_Actor(nn.Module):
    def __init__(self):
        super(GNN_Actor, self).__init__()
        self.gnn1 = GNN_Layer(6, 64)
        self.gnn2 = GNN_Layer(64, 64)
        
        # Action network
        self.action_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        
        # Collision prediction network
        self.collision_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, adj_matrix):
        h1 = self.gnn1(node_features, adj_matrix)
        h2 = self.gnn2(h1, adj_matrix)
        
        # Get action and collision prediction
        action = self.action_net(h2)
        collision_risk = self.collision_net(h2)
        
        return action, collision_risk

class SingleDroneGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚁 SINGLE DRONE - Vision & Performance")
        self.window.geometry("1600x900")
        self.window.configure(bg='#1a1a1a')
        
        self.client = None
        self.gnn_actor = GNN_Actor()
        
        # Try to load trained model
        try:
            self.gnn_actor.load_state_dict(torch.load("gnn_actor.pth", map_location='cpu'))
            print("✅ Loaded trained GNN model")
        except:
            print("⚠️ No trained model found, using random initialization")
        
        self.gnn_actor.eval()
        
        # Flight state
        self.running = False
        self.goal_x = 100.0
        self.goal_y = 100.0
        self.altitude = -20.0
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Metrics
        self.metrics = {
            'time': [],
            'speed': [],
            'altitude': [],
            'battery': [],
            'energy': [],
            'path_x': [],
            'path_y': [],
            'collision_risk': []
        }
        self.start_time = None
        self.battery = 100.0
        self.energy = 0.0
        
        # Wind
        self.wind_x = 2.0
        self.wind_y = 1.0
        self.wind_enabled = True
        self.heavy_wind_detected = False
        self.flight_mode = "NORMAL"  # NORMAL or WIND
        
        # RL tracking
        self.run_number = 0
        self.best_energy = float('inf')
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header = tk.Frame(self.window, bg='#0d47a1', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="🚁 SINGLE DRONE NAVIGATION", font=("Arial", 20, "bold"), 
                fg="white", bg='#0d47a1').pack(pady=15)
        
        # Main container
        main = tk.Frame(self.window, bg='#1a1a1a')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel container with scrollbar
        left_container = tk.Frame(main, bg='#2d2d2d', width=400, relief='raised', borderwidth=2)
        left_container.pack(side='left', fill='both', padx=(0, 10))
        left_container.pack_propagate(False)
        
        # Create canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_container, bg='#2d2d2d', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient='vertical', command=left_canvas.yview)
        
        # Create scrollable frame
        left = tk.Frame(left_canvas, bg='#2d2d2d')
        
        # Configure canvas
        left.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=left, anchor='nw')
        left_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side='right', fill='y')
        left_canvas.pack(side='left', fill='both', expand=True)
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Vision Display
        tk.Label(left, text="👁️ DEPTH VISION", font=("Arial", 12, "bold"), 
                fg="#4caf50", bg='#2d2d2d').pack(pady=10)
        
        self.vision_canvas = tk.Canvas(left, width=360, height=200, bg='black',
                                      highlightthickness=2, highlightbackground='#4caf50')
        self.vision_canvas.pack(pady=5)
        
        self.obstacle_label = tk.Label(left, text="CLEAR PATH", font=("Arial", 11, "bold"),
                                      fg="#4caf50", bg='#2d2d2d')
        self.obstacle_label.pack(pady=5)
        
        # Wind Display
        tk.Label(left, text="💨 WIND CONDITIONS", font=("Arial", 12, "bold"),
                fg="#2196f3", bg='#2d2d2d').pack(pady=(15, 5))
        
        self.wind_canvas = tk.Canvas(left, width=360, height=120, bg='#0d0d0d',
                                     highlightthickness=1, highlightbackground='#2196f3')
        self.wind_canvas.pack(pady=5)
        
        self.wind_label = tk.Label(left, text="Wind: 2.2 m/s", font=("Arial", 10),
                                   fg="white", bg='#2d2d2d')
        self.wind_label.pack()
        
        self.wind_status_label = tk.Label(left, text="Status: NORMAL ✓", 
                                          font=("Arial", 9, "bold"), fg="#4caf50", bg='#2d2d2d')
        self.wind_status_label.pack(pady=2)
        
        # Wind On/Off Toggle
        wind_toggle_frame = tk.Frame(left, bg='#2d2d2d')
        wind_toggle_frame.pack(pady=5)
        
        self.wind_var = tk.BooleanVar(value=True)
        self.wind_toggle = tk.Checkbutton(wind_toggle_frame, text="Wind Enabled",
                                         variable=self.wind_var, command=self.toggle_wind,
                                         font=("Arial", 9, "bold"), fg="white", bg='#2d2d2d',
                                         selectcolor='#1e1e1e', activebackground='#2d2d2d')
        self.wind_toggle.pack()
        
        # Flight Mode Buttons
        tk.Label(left, text="🎮 FLIGHT MODE", font=("Arial", 11, "bold"),
                fg="#ff9800", bg='#2d2d2d').pack(pady=(15, 5))
        
        mode_frame = tk.Frame(left, bg='#2d2d2d')
        mode_frame.pack(pady=5)
        
        self.normal_mode_btn = tk.Button(mode_frame, text="📷 NORMAL\n(Vision)",
                                         font=("Arial", 9, "bold"), bg='#4caf50', fg='white',
                                         relief='sunken', borderwidth=3, width=12, height=2,
                                         command=self.set_normal_mode)
        self.normal_mode_btn.grid(row=0, column=0, padx=5)
        
        self.wind_mode_btn = tk.Button(mode_frame, text="💨 WIND\n(Pressure)",
                                       font=("Arial", 9, "bold"), bg='#607d8b', fg='white',
                                       relief='raised', borderwidth=3, width=12, height=2,
                                       command=self.set_wind_mode)
        self.wind_mode_btn.grid(row=0, column=1, padx=5)
        
        self.mode_status_label = tk.Label(left, text="Mode: NORMAL (Vision-Based)",
                                          font=("Arial", 8, "italic"), fg="#4caf50", bg='#2d2d2d')
        self.mode_status_label.pack(pady=3)
        
        # Goal Setting
        tk.Label(left, text="🎯 GOAL POSITION", font=("Arial", 12, "bold"),
                fg="#ff9800", bg='#2d2d2d').pack(pady=(15, 5))
        
        tk.Label(left, text="💡 Click on map to set goal", font=("Arial", 9, "italic"),
                fg="#90caf9", bg='#2d2d2d').pack()
        
        goal_frame = tk.Frame(left, bg='#2d2d2d')
        goal_frame.pack(pady=5)
        
        tk.Label(goal_frame, text="X:", font=("Arial", 10), fg="white", bg='#2d2d2d').grid(row=0, column=0, padx=5)
        self.goal_x_entry = tk.Entry(goal_frame, width=8, font=("Arial", 10), bg='#1e1e1e', fg='white', insertbackground='white')
        self.goal_x_entry.grid(row=0, column=1, padx=5)
        self.goal_x_entry.insert(0, "100")
        
        tk.Label(goal_frame, text="Y:", font=("Arial", 10), fg="white", bg='#2d2d2d').grid(row=0, column=2, padx=5)
        self.goal_y_entry = tk.Entry(goal_frame, width=8, font=("Arial", 10), bg='#1e1e1e', fg='white', insertbackground='white')
        self.goal_y_entry.grid(row=0, column=3, padx=5)
        self.goal_y_entry.insert(0, "100")
        
        tk.Button(left, text="✓ SET GOAL", command=self.set_goal, bg="#2196f3", fg="white",
                 font=("Arial", 10, "bold"), width=15, cursor='hand2').pack(pady=5)
        
        self.goal_status = tk.Label(left, text=f"Current: (100.0, 100.0)", font=("Arial", 9),
                                   fg="#4caf50", bg='#2d2d2d')
        self.goal_status.pack()
        
        # Telemetry
        tk.Label(left, text="📊 TELEMETRY", font=("Arial", 12, "bold"),
                fg="#2196f3", bg='#2d2d2d').pack(pady=(15, 5))
        
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
        
        self.battery_label = tk.Label(left, text="100%", font=("Arial", 20, "bold"),
                                     fg="#4caf50", bg='#2d2d2d')
        self.battery_label.pack()
        
        self.energy_label = tk.Label(left, text="Energy: 0.0 Wh", font=("Arial", 9),
                                    fg="#90caf9", bg='#2d2d2d')
        self.energy_label.pack()
        
        # RL Stats
        tk.Label(left, text="🧠 GNN + RL LEARNING", font=("Arial", 12, "bold"),
                fg="#9c27b0", bg='#2d2d2d').pack(pady=(15, 5))
        
        tk.Label(left, text="Graph Neural Network with Collision Avoidance",
                font=("Arial", 8, "italic"), fg="#ce93d8", bg='#2d2d2d').pack()
        
        rl_frame = tk.Frame(left, bg='#2d2d2d')
        rl_frame.pack(pady=5, fill='x', padx=20)
        
        self.run_label = tk.Label(rl_frame, text="Run: #0", font=("Arial", 9, "bold"),
                                 fg="white", bg='#2d2d2d', anchor='w')
        self.run_label.pack(fill='x', pady=2)
        
        self.best_label = tk.Label(rl_frame, text="Best Energy: -- Wh", font=("Arial", 9),
                                  fg="#76ff03", bg='#2d2d2d', anchor='w')
        self.best_label.pack(fill='x', pady=2)
        
        self.collision_risk_label = tk.Label(rl_frame, text="Collision Risk: 0%", font=("Arial", 9),
                                            fg="#4caf50", bg='#2d2d2d', anchor='w')
        self.collision_risk_label.pack(fill='x', pady=2)
        
        self.gnn_status_label = tk.Label(rl_frame, text="GNN: Active ✓", font=("Arial", 9),
                                         fg="#00e5ff", bg='#2d2d2d', anchor='w')
        self.gnn_status_label.pack(fill='x', pady=2)
        
        # Status
        self.status_label = tk.Label(left, text="● Ready", font=("Arial", 11),
                                    fg="#90caf9", bg='#2d2d2d')
        self.status_label.pack(pady=15)
        
        # Control buttons
        self.start_btn = tk.Button(left, text="🚀 START FLIGHT", command=self.start_flight,
                                  bg="#4caf50", fg="white", font=("Arial", 12, "bold"),
                                  width=18, height=2, cursor='hand2')
        self.start_btn.pack(pady=8)
        
        self.land_btn = tk.Button(left, text="🛬 LAND", command=self.land_drone,
                                 bg="#ff9800", fg="white", font=("Arial", 12, "bold"),
                                 width=18, height=2, state="disabled", cursor='hand2')
        self.land_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(left, text="⏹ STOP", command=self.stop_flight,
                                 bg="#f44336", fg="white", font=("Arial", 12, "bold"),
                                 width=18, height=2, state="disabled", cursor='hand2')
        self.stop_btn.pack(pady=5)
        
        # Right panel - Graphs
        right = tk.Frame(main, bg='#2d2d2d', relief='raised', borderwidth=2)
        right.pack(side='right', fill='both', expand=True)
        
        tk.Label(right, text="📈 PERFORMANCE ANALYTICS & FLIGHT MAP", font=("Arial", 14, "bold"),
                fg="#4caf50", bg='#2d2d2d').pack(pady=10)
        
        self.fig = Figure(figsize=(11, 8), facecolor='#2d2d2d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        self.init_graphs()
    
    def init_graphs(self):
        self.fig.clear()
        
        # 5 subplots: speed, altitude, battery, energy, map
        ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
        ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
        ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
        ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
        ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_title('Start flight...', color='#757575', fontsize=10)
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#404040')
        
        # Map
        ax5.scatter(0, 0, color='#4caf50', s=200, marker='o', edgecolors='white', linewidths=2, label='Start')
        ax5.scatter(self.goal_x, self.goal_y, color='#ff0000', s=400, marker='*', edgecolors='yellow', linewidths=3, label='Goal')
        ax5.set_title('FLIGHT MAP (Click to set goal)', fontweight='bold', color='white', fontsize=11)
        ax5.set_xlabel('X (m)', color='white', fontsize=9)
        ax5.set_ylabel('Y (m)', color='white', fontsize=9)
        ax5.grid(alpha=0.3, color='#90ee90', linestyle='--')
        ax5.legend(fontsize=8, facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
        ax5.tick_params(colors='white', labelsize=8)
        for spine in ax5.spines.values():
            spine.set_color('#404040')
        ax5.set_xlim([-20, max(self.goal_x + 20, 120)])
        ax5.set_ylim([-20, max(self.goal_y + 20, 120)])
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def set_goal(self):
        try:
            self.goal_x = float(self.goal_x_entry.get())
            self.goal_y = float(self.goal_y_entry.get())
            self.goal_status.config(text=f"Current: ({self.goal_x}, {self.goal_y})")
            self.init_graphs()
            messagebox.showinfo("Goal Set", f"Goal: ({self.goal_x}, {self.goal_y})")
        except:
            messagebox.showerror("Error", "Invalid coordinates!")
    
    def on_map_click(self, event):
        if self.running or event.inaxes is None:
            return
        
        try:
            axes = self.fig.get_axes()
            if len(axes) >= 5 and event.inaxes == axes[4]:
                if event.xdata is not None and event.ydata is not None:
                    self.goal_x = round(event.xdata, 1)
                    self.goal_y = round(event.ydata, 1)
                    self.goal_x_entry.delete(0, tk.END)
                    self.goal_y_entry.delete(0, tk.END)
                    self.goal_x_entry.insert(0, str(self.goal_x))
                    self.goal_y_entry.insert(0, str(self.goal_y))
                    self.goal_status.config(text=f"Current: ({self.goal_x}, {self.goal_y})")
                    self.init_graphs()
        except:
            pass
    
    def start_flight(self):
        self.status_label.config(text="● Initializing...", fg="orange")
        self.window.update()
        threading.Thread(target=self._flight_loop, daemon=True).start()
    
    def stop_flight(self):
        self.running = False
        self.status_label.config(text="● Stopped", fg="#f44336")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.land_btn.config(state="disabled")
        
        if self.client:
            try:
                self.client.landAsync(vehicle_name="Drone1").join()
            except:
                pass
    
    def land_drone(self):
        """Land the drone"""
        if self.client:
            try:
                self.status_label.config(text="● Landing...", fg="#ff9800")
                self.client.landAsync(vehicle_name="Drone1").join()
                self.status_label.config(text="● Landed", fg="#4caf50")
                self.land_btn.config(state="disabled")
                self.start_btn.config(state="normal")
                messagebox.showinfo("Landed", "Drone landed successfully!")
            except Exception as e:
                messagebox.showerror("Landing Error", f"Failed to land: {str(e)}")
    
    def toggle_wind(self):
        """Toggle wind on/off"""
        self.wind_enabled = self.wind_var.get()
        if self.wind_enabled:
            self.wind_x = 2.0
            self.wind_y = 1.0
            self.wind_status_label.config(text="Status: WIND ON ✓", fg="#4caf50")
            print("💨 Wind ENABLED - Wind: 2.2 m/s")
        else:
            self.wind_x = 0.0
            self.wind_y = 0.0
            self.wind_status_label.config(text="Status: NO WIND ✗", fg="#757575")
            print("🚫 Wind DISABLED - Calm conditions")
        self.draw_wind()
    
    def set_normal_mode(self):
        """Set flight mode to NORMAL"""
        self.flight_mode = "NORMAL"
        self.heavy_wind_detected = False
        
        # Update button styles
        self.normal_mode_btn.config(relief='sunken', bg='#4caf50')
        self.wind_mode_btn.config(relief='raised', bg='#607d8b')
        
        # Update status
        self.mode_status_label.config(text="Mode: NORMAL (Vision-Based)", fg="#4caf50")
        self.wind_status_label.config(text="Status: NORMAL ✓", fg="#4caf50")
        
        print("📷 NORMAL MODE activated - Vision-based navigation")
    
    def set_wind_mode(self):
        """Set flight mode to WIND (Heavy wind situation)"""
        self.flight_mode = "WIND"
        self.heavy_wind_detected = True
        
        # Update button styles
        self.normal_mode_btn.config(relief='raised', bg='#607d8b')
        self.wind_mode_btn.config(relief='sunken', bg='#f44336')
        
        # Update status
        self.mode_status_label.config(text="Mode: WIND (Pressure-Based)", fg="#f44336")
        self.wind_status_label.config(text="Status: HEAVY WIND ⚠️", fg="#f44336")
        
        print("💨 WIND MODE activated - Heavy wind navigation")
    
    def _flight_loop(self):
        try:
            # Connect
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.land_btn.config(state="normal")
            self.status_label.config(text="● Flying", fg="#4caf50")
            
            # Reset metrics
            self.metrics = {
                'time': [],
                'speed': [],
                'altitude': [],
                'battery': [],
                'energy': [],
                'path_x': [],
                'path_y': [],
                'collision_risk': []
            }
            self.start_time = time.time()
            self.battery = 100.0
            self.energy = 0.0
            self.run_number += 1
            
            # Takeoff
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone1")
            self.client.takeoffAsync(vehicle_name="Drone1").join()
            self.client.moveToZAsync(self.altitude, 3.0, vehicle_name="Drone1").join()
            
            # Start update threads
            threading.Thread(target=self._update_displays, daemon=True).start()
            threading.Thread(target=self._update_graphs, daemon=True).start()
            
            # Navigation loop
            step = 0
            while self.running:
                # Get state
                state = self.client.getMultirotorState(vehicle_name="Drone1")
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                self.position = np.array([pos.x_val, pos.y_val, pos.z_val])
                self.velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
                
                # Calculate to goal
                goal_vec = np.array([self.goal_x, self.goal_y])
                current_pos = self.position[:2]
                to_goal = goal_vec - current_pos
                dist = np.linalg.norm(to_goal)
                
                # Check if reached
                if dist < 5.0:
                    self.status_label.config(text="● Goal Reached!", fg="#76ff03")
                    messagebox.showinfo("Success", f"🎉 Goal reached!\nRun #{self.run_number}\nEnergy: {self.energy:.2f} Wh")
                    
                    # Update RL - Save best performance
                    if self.energy < self.best_energy:
                        self.best_energy = self.energy
                        # Try to save model
                        try:
                            torch.save(self.gnn_actor.state_dict(), "gnn_actor_best.pth")
                            print(f"✅ New best! Saved model with energy: {self.energy:.2f} Wh")
                        except:
                            pass
                    
                    break
                
                # GNN Node Features (drone state as graph node)
                # Features: [goal_rel_x, goal_rel_y, vel_x, vel_y, wind_x, wind_y]
                node_features = torch.FloatTensor([[
                    to_goal[0] / 100.0,
                    to_goal[1] / 100.0,
                    self.velocity[0] / 10.0,
                    self.velocity[1] / 10.0,
                    self.wind_x / 5.0 if self.wind_enabled else 0.0,
                    self.wind_y / 5.0 if self.wind_enabled else 0.0
                ]]).unsqueeze(0)  # Shape: [1, 1, 6]
                
                # Adjacency matrix (self-connection for single drone)
                adj_matrix = torch.FloatTensor([[[1.0]]])  # Shape: [1, 1, 1] - Fixed: removed extra unsqueeze
                
                # Get GNN action and collision prediction
                with torch.no_grad():
                    gnn_action, collision_risk_tensor = self.gnn_actor(node_features, adj_matrix)
                    gnn_action = gnn_action.squeeze(0).squeeze(0).numpy()  # Shape: [2]
                    collision_risk = float(collision_risk_tensor.squeeze(0).squeeze(0).item())
                
                # Navigation strategy
                direction = to_goal / (dist + 1e-6)
                
                # Adaptive blending based on distance and collision risk
                if dist > 50:
                    # Far from goal - mostly direct navigation
                    goal_weight = 0.85
                    gnn_weight = 0.15
                elif dist > 20:
                    # Medium distance - balanced
                    goal_weight = 0.70
                    gnn_weight = 0.30
                else:
                    # Close to goal - more GNN for precision
                    goal_weight = 0.60
                    gnn_weight = 0.40
                
                # If collision risk high, trust GNN more
                if collision_risk > 0.5:
                    gnn_weight += 0.2
                    goal_weight -= 0.2
                
                # Blend actions
                vx = float(direction[0] * goal_weight + gnn_action[0] * gnn_weight) * 6.0
                vy = float(direction[1] * goal_weight + gnn_action[1] * gnn_weight) * 6.0
                
                # Apply wind effect if enabled and in wind mode
                if self.wind_enabled and self.heavy_wind_detected:
                    vx += self.wind_x * 0.3
                    vy += self.wind_y * 0.3
                
                # Calculate target position (look-ahead)
                lookahead = 0.8 if dist > 20 else 0.4
                target_x = pos.x_val + vx * lookahead
                target_y = pos.y_val + vy * lookahead
                
                # Move drone - use moveToPositionAsync with join for actual movement
                velocity = float(np.sqrt(vx**2 + vy**2))
                velocity = min(max(velocity, 2.0), 8.0)  # Clamp between 2-8 m/s
                
                self.client.moveToPositionAsync(
                    target_x, target_y, self.altitude,
                    velocity,
                    vehicle_name="Drone1"
                ).join()
                
                # Console logging every 10 steps
                if step % 10 == 0:
                    print(f"Step {step} | Pos: ({pos.x_val:.1f}, {pos.y_val:.1f}) | "
                          f"Goal: ({self.goal_x:.1f}, {self.goal_y:.1f}) | "
                          f"Dist: {dist:.1f}m | "
                          f"Collision: {collision_risk*100:.1f}% | "
                          f"Velocity: {velocity:.1f} m/s")
                
                # Calculate metrics
                elapsed = time.time() - self.start_time
                speed = np.linalg.norm(self.velocity[:2])
                alt = abs(self.position[2])
                
                # Reinforcement Learning: Calculate reward
                # Reward = -distance - energy_cost + speed_bonus - collision_penalty
                reward = -dist/100.0 - self.energy/1000.0 + speed/10.0 - collision_risk
                
                # Energy
                energy_step = (200 + speed ** 2 * 10) * 0.5 / 3600.0
                self.energy += energy_step
                self.battery = max(0, 100.0 - (self.energy / 100.0) * 100)
                
                self.metrics['time'].append(elapsed)
                self.metrics['speed'].append(speed)
                self.metrics['altitude'].append(alt)
                self.metrics['battery'].append(self.battery)
                self.metrics['energy'].append(self.energy)
                self.metrics['path_x'].append(self.position[0])
                self.metrics['path_y'].append(self.position[1])
                self.metrics['collision_risk'].append(collision_risk * 100)
                
                step += 1
                time.sleep(0.1)
            
            self.stop_flight()
            
        except Exception as e:
            print(f"Flight error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Flight error:\n{str(e)}")
            self.stop_flight()
    
    def _update_displays(self):
        while self.running:
            try:
                # Telemetry
                self.pos_label.config(text=f"Pos: ({self.position[0]:.1f}, {self.position[1]:.1f})")
                self.speed_label.config(text=f"Speed: {np.linalg.norm(self.velocity[:2]):.1f} m/s")
                
                dist = np.linalg.norm(np.array([self.goal_x, self.goal_y]) - self.position[:2])
                self.dist_label.config(text=f"Goal Dist: {dist:.1f}m")
                
                if len(self.metrics['collision_risk']) > 0:
                    risk = self.metrics['collision_risk'][-1]
                    risk_color = "#4caf50" if risk < 30 else "#ff9800" if risk < 70 else "#f44336"
                    self.collision_label.config(text=f"Collision Risk: {risk:.0f}%", fg=risk_color)
                
                # Battery
                battery_color = "#4caf50" if self.battery > 50 else "#ff9800" if self.battery > 20 else "#f44336"
                self.battery_label.config(text=f"{self.battery:.1f}%", fg=battery_color)
                self.energy_label.config(text=f"Energy: {self.energy:.2f} Wh")
                
                # RL
                self.run_label.config(text=f"Run: #{self.run_number}")
                if self.best_energy < float('inf'):
                    self.best_label.config(text=f"Best Energy: {self.best_energy:.2f} Wh")
                
                # GNN Collision Risk
                if len(self.metrics['collision_risk']) > 0:
                    risk = self.metrics['collision_risk'][-1]
                    risk_color = "#4caf50" if risk < 30 else "#ff9800" if risk < 70 else "#f44336"
                    self.collision_risk_label.config(text=f"Collision Risk: {risk:.0f}%", fg=risk_color)
                
                # GNN Status
                self.gnn_status_label.config(text="GNN: Active ✓", fg="#00e5ff")
                
                # Wind
                wind_mag = np.sqrt(self.wind_x**2 + self.wind_y**2)
                self.wind_label.config(text=f"Wind: {wind_mag:.1f} m/s")
                self.draw_wind()
                
                # Vision (simulated)
                self.update_vision()
                
                time.sleep(0.1)
            except:
                pass
    
    def draw_wind(self):
        try:
            self.wind_canvas.delete("all")
            w, h = 360, 120
            
            if not self.wind_enabled or (self.wind_x == 0 and self.wind_y == 0):
                # No wind - show calm
                self.wind_canvas.create_text(w//2, h//2, text="NO WIND - CALM", 
                                            fill="#757575", font=("Arial", 12, "bold"))
                self.wind_label.config(text="Wind: 0.0 m/s", fg="#757575")
                return
            
            wind_mag = np.sqrt(self.wind_x**2 + self.wind_y**2)
            
            # Color based on wind strength
            if self.flight_mode == "WIND" or self.heavy_wind_detected:
                color = "#f44336"  # Red for heavy wind
            elif wind_mag < 5:
                color = "#4caf50"  # Green for light wind
            else:
                color = "#ffeb3b"  # Yellow for moderate wind
            
            # Draw arrows
            for gx in range(60, w, 80):
                for gy in range(30, h, 40):
                    end_x = gx + self.wind_x * 10
                    end_y = gy - self.wind_y * 10
                    self.wind_canvas.create_line(gx, gy, end_x, end_y, fill=color, width=2, arrow=tk.LAST)
            
            self.wind_canvas.create_text(w//2, 15, text=f"{wind_mag:.1f} m/s", fill=color, font=("Arial", 10, "bold"))
            self.wind_label.config(text=f"Wind: {wind_mag:.1f} m/s", fg=color)
        except:
            pass
    
    def update_vision(self):
        """Get real depth vision from AirSim camera (like smart_drone_gui.py)"""
        try:
            self.vision_canvas.delete("all")
            
            if not self.client or not self.running:
                # Show default state when not flying
                fallback_color = "#2d2d2d"
                self.vision_canvas.create_rectangle(10, 10, 110, 190, fill=fallback_color, outline="#404040", width=2)
                self.vision_canvas.create_text(60, 100, text="LEFT\n---", fill="#757575", font=("Arial", 10, "bold"))
                self.vision_canvas.create_rectangle(130, 10, 230, 190, fill=fallback_color, outline="#404040", width=2)
                self.vision_canvas.create_text(180, 100, text="CENTER\n---", fill="#757575", font=("Arial", 10, "bold"))
                self.vision_canvas.create_rectangle(250, 10, 350, 190, fill=fallback_color, outline="#404040", width=2)
                self.vision_canvas.create_text(300, 100, text="RIGHT\n---", fill="#757575", font=("Arial", 10, "bold"))
                self.obstacle_label.config(text="DEPTH CAMERA READY", fg="#757575")
                return
            
            # Get depth image from AirSim front camera
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
            ], vehicle_name="Drone1")
            
            if responses and len(responses[0].image_data_float) > 0:
                # Convert depth data to numpy array
                depth_data = np.array(responses[0].image_data_float)
                depth_data = depth_data.reshape(responses[0].height, responses[0].width)
                
                h, w = depth_data.shape
                
                # Divide into 3 zones: LEFT, CENTER, RIGHT
                left_depth = np.mean(depth_data[h//3:2*h//3, :w//3])
                center_depth = np.mean(depth_data[h//3:2*h//3, w//3:2*w//3])
                right_depth = np.mean(depth_data[h//3:2*h//3, 2*w//3:])
                
                # Clip extreme values (100m max)
                left_depth = min(left_depth, 100.0)
                center_depth = min(center_depth, 100.0)
                right_depth = min(right_depth, 100.0)
                
                # Determine colors based on depth (distance to obstacles)
                def depth_to_color(depth):
                    if depth < 3.0:  # Very close - DANGER
                        return "#f44336"  # Red
                    elif depth < 5.0:  # Close - WARNING
                        return "#ff9800"  # Orange
                    elif depth < 10.0:  # Medium distance - CAUTION
                        return "#ffeb3b"  # Yellow
                    else:  # Far - CLEAR
                        return "#4caf50"  # Green
                
                left_color = depth_to_color(left_depth)
                center_color = depth_to_color(center_depth)
                right_color = depth_to_color(right_depth)
                
                # Draw LEFT zone
                self.vision_canvas.create_rectangle(10, 10, 110, 190, fill=left_color, outline="white", width=2)
                self.vision_canvas.create_text(60, 50, text="LEFT", fill="white", font=("Arial", 10, "bold"))
                self.vision_canvas.create_text(60, 150, text=f"{left_depth:.1f}m", fill="white", font=("Arial", 9))
                
                # Draw CENTER zone
                self.vision_canvas.create_rectangle(130, 10, 230, 190, fill=center_color, outline="white", width=2)
                self.vision_canvas.create_text(180, 50, text="CENTER", fill="white", font=("Arial", 10, "bold"))
                self.vision_canvas.create_text(180, 150, text=f"{center_depth:.1f}m", fill="white", font=("Arial", 9))
                
                # Draw RIGHT zone
                self.vision_canvas.create_rectangle(250, 10, 350, 190, fill=right_color, outline="white", width=2)
                self.vision_canvas.create_text(300, 50, text="RIGHT", fill="white", font=("Arial", 10, "bold"))
                self.vision_canvas.create_text(300, 150, text=f"{right_depth:.1f}m", fill="white", font=("Arial", 9))
                
                # Update obstacle status based on minimum depth
                min_depth = min(left_depth, center_depth, right_depth)
                if min_depth < 3.0:
                    text = "⚠️ DANGER - OBSTACLE CLOSE!"
                    color = "#f44336"
                elif min_depth < 5.0:
                    text = "⚠️ WARNING - OBSTACLE AHEAD"
                    color = "#ff9800"
                elif min_depth < 10.0:
                    text = "CAUTION - OBSTACLE DETECTED"
                    color = "#ffeb3b"
                else:
                    text = "CLEAR PATH ✓"
                    color = "#4caf50"
                
                self.obstacle_label.config(text=text, fg=color)
            else:
                # No depth data - show fallback
                fallback_color = "#607d8b"
                self.vision_canvas.create_rectangle(10, 10, 110, 190, fill=fallback_color, outline="white", width=2)
                self.vision_canvas.create_text(60, 100, text="LEFT\n??", fill="white", font=("Arial", 10, "bold"))
                self.vision_canvas.create_rectangle(130, 10, 230, 190, fill=fallback_color, outline="white", width=2)
                self.vision_canvas.create_text(180, 100, text="CENTER\n??", fill="white", font=("Arial", 10, "bold"))
                self.vision_canvas.create_rectangle(250, 10, 350, 190, fill=fallback_color, outline="white", width=2)
                self.vision_canvas.create_text(300, 100, text="RIGHT\n??", fill="white", font=("Arial", 10, "bold"))
                self.obstacle_label.config(text="NO CAMERA DATA", fg="#ff9800")
        except Exception as e:
            # Fallback on error - don't crash the GUI
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
                    ax1.set_title('SPEED', fontweight='bold', fontsize=10, color='white')
                    ax1.set_ylabel('m/s', fontsize=8, color='white')
                    ax1.grid(alpha=0.2, color='#404040')
                    ax1.tick_params(colors='white', labelsize=8)
                    for spine in ax1.spines.values():
                        spine.set_color('#404040')
                    
                    # Altitude
                    ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
                    ax2.plot(times, self.metrics['altitude'], color='#ff9800', linewidth=2)
                    ax2.set_title('ALTITUDE', fontweight='bold', fontsize=10, color='white')
                    ax2.set_ylabel('m', fontsize=8, color='white')
                    ax2.axhline(y=20, color='#76ff03', linestyle='--', alpha=0.5)
                    ax2.grid(alpha=0.2, color='#404040')
                    ax2.tick_params(colors='white', labelsize=8)
                    for spine in ax2.spines.values():
                        spine.set_color('#404040')
                    
                    # Battery
                    ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
                    battery_color = '#4caf50' if self.battery > 50 else '#ff9800' if self.battery > 20 else '#f44336'
                    ax3.plot(times, self.metrics['battery'], color=battery_color, linewidth=2.5)
                    ax3.set_title('BATTERY', fontweight='bold', fontsize=10, color='white')
                    ax3.set_ylabel('%', fontsize=8, color='white')
                    ax3.grid(alpha=0.2, color='#404040')
                    ax3.tick_params(colors='white', labelsize=8)
                    for spine in ax3.spines.values():
                        spine.set_color('#404040')
                    
                    # Energy
                    ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
                    ax4.plot(times, self.metrics['energy'], color='#e91e63', linewidth=2.5)
                    ax4.set_title('ENERGY', fontweight='bold', fontsize=10, color='white')
                    ax4.set_ylabel('Wh', fontsize=8, color='white')
                    ax4.set_xlabel('Time (s)', fontsize=8, color='white')
                    ax4.grid(alpha=0.2, color='#404040')
                    ax4.tick_params(colors='white', labelsize=8)
                    for spine in ax4.spines.values():
                        spine.set_color('#404040')
                    
                    # Flight Map
                    ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')
                    
                    ax5.scatter(0, 0, color='#4caf50', s=200, marker='o', edgecolors='white', linewidths=2, label='Start')
                    ax5.scatter(self.goal_x, self.goal_y, color='#ff0000', s=400, marker='*', edgecolors='yellow', linewidths=3, label='Goal')
                    
                    if len(self.metrics['path_x']) > 1:
                        ax5.plot(self.metrics['path_x'], self.metrics['path_y'], color='#00ffff', linewidth=3, alpha=0.8, zorder=5)
                    
                    if len(self.metrics['path_x']) > 0:
                        ax5.scatter(self.metrics['path_x'][-1], self.metrics['path_y'][-1], 
                                   color='#2196f3', s=200, marker='o', edgecolors='white', linewidths=2, label='Drone', zorder=6)
                    
                    ax5.set_title('FLIGHT MAP', fontweight='bold', fontsize=11, color='white')
                    ax5.set_xlabel('X (m)', fontsize=9, color='white')
                    ax5.set_ylabel('Y (m)', fontsize=9, color='white')
                    ax5.grid(alpha=0.3, color='#90ee90', linestyle='--')
                    ax5.legend(fontsize=8, facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
                    ax5.tick_params(colors='white', labelsize=8)
                    for spine in ax5.spines.values():
                        spine.set_color('#404040')
                    
                    self.fig.tight_layout()
                    self.canvas.draw()
                
                time.sleep(0.5)
            except:
                pass
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = SingleDroneGUI()
    app.run()
