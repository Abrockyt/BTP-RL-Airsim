"""
GNN SWARM GUI - Multi-Drone System with Collision Avoidance
Main drone uses GNN to navigate to goal while communicating with background drones
Background drones move randomly
Includes collision detection and GUI visualization
"""

import logging
import warnings

# Suppress logs
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('tornado').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

import airsim
import torch
import torch.nn as nn
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
from collections import deque

# Matplotlib for graphs
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================================================================
# GNN ARCHITECTURE WITH COLLISION DETECTION
# =============================================================================

class GNN_Layer(nn.Module):
    """Graph Neural Network Layer with message passing"""
    def __init__(self, input_dim, hidden_dim):
        super(GNN_Layer, self).__init__()
        # Message network: transforms node features for message passing
        self.message_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Update network: updates node features based on aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features: (batch, num_nodes, input_dim)
            adj_matrix: (batch, num_nodes, num_nodes) - adjacency matrix
        """
        # Generate messages from all nodes
        messages = self.message_net(node_features)  # (batch, num_nodes, hidden_dim)
        
        # Aggregate messages based on adjacency
        aggregated = torch.matmul(adj_matrix, messages)  # (batch, num_nodes, hidden_dim)
        
        # Combine original features with aggregated messages
        combined = torch.cat([node_features, aggregated], dim=2)
        
        # Update node features
        updated = self.update_net(combined)
        return updated


class CollisionGNN_Actor(nn.Module):
    """GNN Actor with Collision Detection and Avoidance"""
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=64):
        super(CollisionGNN_Actor, self).__init__()
        
        self.state_dim = state_dim  # [goal_x, goal_y, vel_x, vel_y, nearest_obs_dist, nearest_obs_angle]
        self.action_dim = action_dim
        
        # Two GNN layers for deep message passing
        self.gnn1 = GNN_Layer(state_dim, hidden_dim)
        self.gnn2 = GNN_Layer(hidden_dim, hidden_dim)
        
        # Collision prediction network
        self.collision_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Collision probability [0, 1]
        )
        
        # Action network
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
    
    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features: (batch, num_nodes, state_dim)
            adj_matrix: (batch, num_nodes, num_nodes)
        Returns:
            actions: (batch, num_nodes, action_dim)
            collision_probs: (batch, num_nodes, 1)
        """
        # GNN processing
        h1 = self.gnn1(node_features, adj_matrix)
        h2 = self.gnn2(h1, adj_matrix)
        
        # Predict collision probability
        collision_probs = self.collision_net(h2)
        
        # Generate actions
        actions = self.action_net(h2)
        
        return actions, collision_probs


# =============================================================================
# CONFIGURATION
# =============================================================================

MAIN_DRONE = "Drone1"
BACKGROUND_DRONES = [f"Drone{i}" for i in range(2, 11)]
GOAL = np.array([100.0, 100.0])
ALTITUDE = -20.0  # 20m above ground
COMM_RANGE = 30.0  # meters
COLLISION_THRESHOLD = 5.0  # meters - minimum safe distance
VELOCITY_SCALE = 6.0  # Maximum velocity m/s

# =============================================================================
# GNN SWARM CONTROLLER
# =============================================================================

class GNNSwarmController:
    """Main controller for GNN swarm system"""
    
    def __init__(self):
        self.client = None
        self.model = None
        self.stop_event = threading.Event()
        self.bg_threads = []
        
        # Telemetry data
        self.main_position = np.array([0.0, 0.0, 0.0])
        self.main_velocity = np.array([0.0, 0.0, 0.0])
        self.drone_positions = {}
        self.drone_velocities = {}
        self.goal_distance = 0.0
        self.comm_count = 0
        self.step_count = 0
        self.collision_risk = 0.0
        self.nearest_drone_dist = 999.0
        
        # History for graphs
        self.distance_history = deque(maxlen=100)
        self.collision_history = deque(maxlen=100)
        self.speed_history = deque(maxlen=100)
        self.comm_history = deque(maxlen=100)
        
        # Status
        self.is_running = False
        self.reached_goal = False
        
    def connect(self):
        """Connect to AirSim"""
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        return True
    
    def load_model(self):
        """Load GNN model"""
        self.model = CollisionGNN_Actor(state_dim=6, action_dim=2)
        try:
            self.model.load_state_dict(torch.load("gnn_collision_model.pth", map_location='cpu'))
            self.model.eval()
            return True, "Trained model loaded"
        except:
            self.model.eval()
            return False, "Using random weights (untrained)"
    
    def reset_simulation(self):
        """Reset AirSim simulation"""
        self.client.reset()
        time.sleep(2)
    
    def calculate_collision_features(self, my_pos, all_positions):
        """Calculate nearest obstacle distance and angle"""
        min_dist = 999.0
        nearest_angle = 0.0
        
        for drone_name, pos in all_positions.items():
            if drone_name == MAIN_DRONE:
                continue
            
            diff = pos - my_pos[:2]
            dist = np.linalg.norm(diff)
            
            if dist < min_dist:
                min_dist = dist
                if dist > 0:
                    nearest_angle = np.arctan2(diff[1], diff[0])
        
        self.nearest_drone_dist = min_dist
        return min_dist, nearest_angle
    
    def background_drone_thread(self, drone_name):
        """Background drone with random movement"""
        try:
            # Create separate client
            client = airsim.MultirotorClient()
            client.confirmConnection()
            
            # Takeoff
            client.enableApiControl(True, vehicle_name=drone_name)
            client.armDisarm(True, vehicle_name=drone_name)
            client.takeoffAsync(vehicle_name=drone_name).join()
            client.moveToZAsync(ALTITUDE, 3.0, vehicle_name=drone_name).join()
            
            # Random movement loop
            while not self.stop_event.is_set():
                # Get current position
                state = client.getMultirotorState(vehicle_name=drone_name)
                pos = state.kinematics_estimated.position
                
                # Random target
                dx = np.random.uniform(-40, 40)
                dy = np.random.uniform(-40, 40)
                target_x = pos.x_val + dx
                target_y = pos.y_val + dy
                
                # Move
                client.moveToPositionAsync(
                    target_x, target_y, ALTITUDE, 
                    np.random.uniform(3.0, 7.0),
                    vehicle_name=drone_name
                ).join()
                
                time.sleep(np.random.uniform(2.0, 5.0))
        
        except Exception as e:
            pass  # Silently handle background thread errors
    
    def main_drone_navigation(self):
        """Main drone GNN-based navigation with collision avoidance"""
        try:
            # Takeoff
            self.client.enableApiControl(True, vehicle_name=MAIN_DRONE)
            self.client.armDisarm(True, vehicle_name=MAIN_DRONE)
            self.client.takeoffAsync(vehicle_name=MAIN_DRONE).join()
            self.client.moveToZAsync(ALTITUDE, 3.0, vehicle_name=MAIN_DRONE).join()
            
            self.step_count = 0
            
            while not self.stop_event.is_set() and self.is_running:
                # ===== GET POSITIONS =====
                all_positions = {}
                all_velocities = {}
                
                # Main drone
                state_main = self.client.getMultirotorState(vehicle_name=MAIN_DRONE)
                pos_main = state_main.kinematics_estimated.position
                vel_main = state_main.kinematics_estimated.linear_velocity
                
                self.main_position = np.array([pos_main.x_val, pos_main.y_val, pos_main.z_val])
                self.main_velocity = np.array([vel_main.x_val, vel_main.y_val, vel_main.z_val])
                
                all_positions[MAIN_DRONE] = self.main_position[:2]
                all_velocities[MAIN_DRONE] = self.main_velocity[:2]
                
                # Background drones
                for drone in BACKGROUND_DRONES:
                    try:
                        state = self.client.getMultirotorState(vehicle_name=drone)
                        pos = state.kinematics_estimated.position
                        vel = state.kinematics_estimated.linear_velocity
                        all_positions[drone] = np.array([pos.x_val, pos.y_val])
                        all_velocities[drone] = np.array([vel.x_val, vel.y_val])
                    except:
                        pass
                
                self.drone_positions = all_positions
                self.drone_velocities = all_velocities
                
                # ===== BUILD GNN INPUT =====
                num_drones = len(all_positions)
                all_drones = list(all_positions.keys())
                
                # Build adjacency matrix (communication graph)
                adj_matrix = np.zeros((num_drones, num_drones))
                comm_count = 0
                
                for i, drone_i in enumerate(all_drones):
                    for j, drone_j in enumerate(all_drones):
                        if i != j:
                            dist = np.linalg.norm(all_positions[drone_i] - all_positions[drone_j])
                            if dist < COMM_RANGE:
                                adj_matrix[i, j] = 1.0
                                if drone_i == MAIN_DRONE:
                                    comm_count += 1
                
                self.comm_count = comm_count
                
                # Build node features [goal_rel_x, goal_rel_y, vel_x, vel_y, nearest_obs_dist, nearest_obs_angle]
                node_features = []
                for drone in all_drones:
                    pos = all_positions[drone]
                    vel = all_velocities[drone]
                    
                    # Goal relative position
                    goal_rel = GOAL - pos
                    
                    # Collision features
                    nearest_dist, nearest_angle = self.calculate_collision_features(
                        np.array([pos[0], pos[1], ALTITUDE]), all_positions
                    )
                    
                    features = [
                        goal_rel[0] / 100.0,  # Normalize
                        goal_rel[1] / 100.0,
                        vel[0] / 10.0,
                        vel[1] / 10.0,
                        nearest_dist / 50.0,
                        nearest_angle / np.pi
                    ]
                    node_features.append(features)
                
                # ===== GNN INFERENCE =====
                state_tensor = torch.FloatTensor(node_features).unsqueeze(0)
                adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)
                
                with torch.no_grad():
                    actions, collision_probs = self.model(state_tensor, adj_tensor)
                    main_action = actions.squeeze(0)[0].numpy()
                    self.collision_risk = float(collision_probs.squeeze(0)[0].item())
                
                # ===== EXECUTE ACTION =====
                goal_distance = np.linalg.norm(GOAL - all_positions[MAIN_DRONE])
                self.goal_distance = goal_distance
                
                # Check goal reached
                if goal_distance < 5.0:
                    self.reached_goal = True
                    self.is_running = False
                    break
                
                # Calculate direction to goal
                goal_direction = (GOAL - all_positions[MAIN_DRONE]) / (goal_distance + 1e-6)
                
                # Blend GNN action with goal-seeking
                # If collision risk high, trust GNN more for avoidance
                gnn_weight = 0.2 + (self.collision_risk * 0.3)  # 0.2 to 0.5
                goal_weight = 1.0 - gnn_weight
                
                vx = float(goal_direction[0] * goal_weight + main_action[0] * gnn_weight) * VELOCITY_SCALE
                vy = float(goal_direction[1] * goal_weight + main_action[1] * gnn_weight) * VELOCITY_SCALE
                
                # Calculate target position
                target_x = pos_main.x_val + vx * 0.5
                target_y = pos_main.y_val + vy * 0.5
                
                # Move
                self.client.moveToPositionAsync(
                    target_x, target_y, ALTITUDE,
                    VELOCITY_SCALE,
                    vehicle_name=MAIN_DRONE
                ).join()
                
                # Update history
                self.distance_history.append(goal_distance)
                self.collision_history.append(self.collision_risk * 100)
                self.speed_history.append(np.linalg.norm(self.main_velocity[:2]))
                self.comm_history.append(comm_count)
                
                self.step_count += 1
                time.sleep(0.1)
        
        except Exception as e:
            print(f"Navigation error: {e}")
            import traceback
            traceback.print_exc()
    
    def start_swarm(self):
        """Start the swarm system"""
        self.is_running = True
        self.reached_goal = False
        self.stop_event.clear()
        
        # Start background drones
        for drone in BACKGROUND_DRONES[:6]:  # Use 6 background drones
            thread = threading.Thread(
                target=self.background_drone_thread,
                args=(drone,),
                daemon=True
            )
            thread.start()
            self.bg_threads.append(thread)
            time.sleep(0.3)
        
        # Start main drone navigation
        nav_thread = threading.Thread(target=self.main_drone_navigation, daemon=True)
        nav_thread.start()
    
    def stop_swarm(self):
        """Stop the swarm system"""
        self.is_running = False
        self.stop_event.set()
        
        # Land all drones
        time.sleep(1)
        for drone in [MAIN_DRONE] + BACKGROUND_DRONES:
            try:
                self.client.landAsync(vehicle_name=drone)
            except:
                pass


# =============================================================================
# GUI APPLICATION
# =============================================================================

class GNNSwarmGUI:
    """GUI for GNN Swarm System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("GNN SWARM - Multi-Drone Navigation with Collision Avoidance")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        self.controller = GNNSwarmController()
        self.update_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI components"""
        
        # Header
        header = tk.Frame(self.root, bg='#2d2d2d', height=60)
        header.pack(fill='x', padx=10, pady=(10, 0))
        
        title_label = tk.Label(
            header,
            text="🚁 GNN SWARM SYSTEM - Collision-Aware Multi-Drone Navigation",
            font=('Segoe UI', 16, 'bold'),
            bg='#2d2d2d',
            fg='#00ff00'
        )
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls and Telemetry
        left_panel = tk.Frame(main_container, bg='#2d2d2d', width=400)
        left_panel.pack(side='left', fill='both', padx=(0, 5))
        
        # Control Panel
        control_frame = tk.LabelFrame(
            left_panel,
            text="🎮 CONTROL PANEL",
            font=('Segoe UI', 11, 'bold'),
            bg='#2d2d2d',
            fg='#00bfff',
            relief='groove',
            bd=2
        )
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Buttons
        self.connect_btn = tk.Button(
            control_frame,
            text="🔌 CONNECT TO AIRSIM",
            command=self.connect_airsim,
            bg='#0066cc',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            height=2
        )
        self.connect_btn.pack(fill='x', padx=10, pady=5)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶️ START MISSION",
            command=self.start_mission,
            bg='#00aa00',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            height=2,
            state='disabled'
        )
        self.start_btn.pack(fill='x', padx=10, pady=5)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹️ STOP MISSION",
            command=self.stop_mission,
            bg='#cc0000',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            height=2,
            state='disabled'
        )
        self.stop_btn.pack(fill='x', padx=10, pady=5)
        
        self.reset_btn = tk.Button(
            control_frame,
            text="🔄 RESET SIMULATION",
            command=self.reset_simulation,
            bg='#ff8800',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            height=2,
            state='disabled'
        )
        self.reset_btn.pack(fill='x', padx=10, pady=5)
        
        # Goal Settings
        goal_frame = tk.LabelFrame(
            left_panel,
            text="🎯 GOAL CONFIGURATION",
            font=('Segoe UI', 11, 'bold'),
            bg='#2d2d2d',
            fg='#00bfff',
            relief='groove',
            bd=2
        )
        goal_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(goal_frame, text="Goal X (m):", bg='#2d2d2d', fg='white').grid(row=0, column=0, padx=5, pady=5)
        self.goal_x_entry = tk.Entry(goal_frame, width=10)
        self.goal_x_entry.insert(0, "100.0")
        self.goal_x_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(goal_frame, text="Goal Y (m):", bg='#2d2d2d', fg='white').grid(row=1, column=0, padx=5, pady=5)
        self.goal_y_entry = tk.Entry(goal_frame, width=10)
        self.goal_y_entry.insert(0, "100.0")
        self.goal_y_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Telemetry
        telemetry_frame = tk.LabelFrame(
            left_panel,
            text="📊 TELEMETRY",
            font=('Segoe UI', 11, 'bold'),
            bg='#2d2d2d',
            fg='#00bfff',
            relief='groove',
            bd=2
        )
        telemetry_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.telemetry_text = tk.Text(
            telemetry_frame,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Consolas', 9),
            height=20,
            wrap='word'
        )
        self.telemetry_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right panel - Graphs
        right_panel = tk.Frame(main_container, bg='#2d2d2d')
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Create matplotlib figures
        self.fig = Figure(figsize=(10, 8), facecolor='#2d2d2d')
        
        # 4 subplots
        self.ax1 = self.fig.add_subplot(221, facecolor='#1e1e1e')  # Distance to goal
        self.ax2 = self.fig.add_subplot(222, facecolor='#1e1e1e')  # Collision risk
        self.ax3 = self.fig.add_subplot(223, facecolor='#1e1e1e')  # Speed
        self.ax4 = self.fig.add_subplot(224, facecolor='#1e1e1e')  # Communication
        
        # Configure axes
        for ax, title in zip(
            [self.ax1, self.ax2, self.ax3, self.ax4],
            ['Distance to Goal (m)', 'Collision Risk (%)', 'Speed (m/s)', 'Active Communications']
        ):
            ax.set_title(title, color='white', fontsize=10)
            ax.tick_params(colors='white', labelsize=8)
            ax.set_facecolor('#1e1e1e')
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=30)
        status_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.status_label = tk.Label(
            status_frame,
            text="⚪ Status: Ready to connect",
            font=('Segoe UI', 9),
            bg='#2d2d2d',
            fg='#ffff00',
            anchor='w'
        )
        self.status_label.pack(fill='x', padx=10, pady=5)
    
    def update_status(self, message, color='#ffff00'):
        """Update status bar"""
        self.status_label.config(text=message, fg=color)
    
    def connect_airsim(self):
        """Connect to AirSim"""
        self.update_status("⏳ Connecting to AirSim...", '#ffff00')
        
        def connect_thread():
            try:
                if self.controller.connect():
                    success, msg = self.controller.load_model()
                    
                    self.update_status(f"✅ Connected! Model: {msg}", '#00ff00')
                    self.connect_btn.config(state='disabled')
                    self.start_btn.config(state='normal')
                    self.reset_btn.config(state='normal')
                    messagebox.showinfo("Success", f"Connected to AirSim!\n{msg}")
                else:
                    self.update_status("❌ Connection failed", '#ff0000')
                    messagebox.showerror("Error", "Failed to connect to AirSim")
            except Exception as e:
                self.update_status(f"❌ Error: {str(e)}", '#ff0000')
                messagebox.showerror("Error", f"Connection error:\n{str(e)}")
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def start_mission(self):
        """Start the mission"""
        try:
            # Update goal
            global GOAL
            GOAL = np.array([float(self.goal_x_entry.get()), float(self.goal_y_entry.get())])
            
            self.update_status("🚀 Starting mission...", '#00ff00')
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            # Start controller
            self.controller.start_swarm()
            
            # Start UI updates
            self.update_running = True
            self.update_telemetry()
            self.update_graphs()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start mission:\n{str(e)}")
            self.update_status("❌ Mission start failed", '#ff0000')
    
    def stop_mission(self):
        """Stop the mission"""
        self.update_status("⏹️ Stopping mission...", '#ffff00')
        self.update_running = False
        self.controller.stop_swarm()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        if self.controller.reached_goal:
            messagebox.showinfo("Success", f"🎉 Goal Reached in {self.controller.step_count} steps!")
            self.update_status("✅ Goal reached!", '#00ff00')
        else:
            self.update_status("⏹️ Mission stopped", '#ffff00')
    
    def reset_simulation(self):
        """Reset AirSim"""
        self.update_status("🔄 Resetting simulation...", '#ffff00')
        
        def reset_thread():
            self.controller.reset_simulation()
            self.update_status("✅ Simulation reset", '#00ff00')
            messagebox.showinfo("Reset", "Simulation reset complete")
        
        threading.Thread(target=reset_thread, daemon=True).start()
    
    def update_telemetry(self):
        """Update telemetry display"""
        if not self.update_running:
            return
        
        try:
            c = self.controller
            
            telemetry = f"""
═══════════════════════════════════════
         🚁 MAIN DRONE STATUS
═══════════════════════════════════════
Position:    ({c.main_position[0]:.1f}, {c.main_position[1]:.1f}, {c.main_position[2]:.1f}) m
Velocity:    ({c.main_velocity[0]:.1f}, {c.main_velocity[1]:.1f}, {c.main_velocity[2]:.1f}) m/s
Speed:       {np.linalg.norm(c.main_velocity[:2]):.1f} m/s

Goal:        ({GOAL[0]:.1f}, {GOAL[1]:.1f})
Distance:    {c.goal_distance:.1f} m

═══════════════════════════════════════
         🛡️ COLLISION DETECTION
═══════════════════════════════════════
Risk Level:     {c.collision_risk*100:.1f}%
Nearest Drone:  {c.nearest_drone_dist:.1f} m
Safe Distance:  {COLLISION_THRESHOLD:.1f} m
Status:         {'⚠️ WARNING' if c.nearest_drone_dist < COLLISION_THRESHOLD else '✅ SAFE'}

═══════════════════════════════════════
         📡 COMMUNICATION
═══════════════════════════════════════
Active Links:   {c.comm_count}
Comm Range:     {COMM_RANGE:.1f} m
Total Drones:   {len(c.drone_positions)}

═══════════════════════════════════════
         📈 MISSION PROGRESS
═══════════════════════════════════════
Steps:          {c.step_count}
Status:         {'🎯 NAVIGATING' if c.is_running else '⏹️ STOPPED'}
"""
            
            self.telemetry_text.delete('1.0', 'end')
            self.telemetry_text.insert('1.0', telemetry)
            
            # Update status bar
            risk_emoji = "🟢" if c.collision_risk < 0.3 else "🟡" if c.collision_risk < 0.7 else "🔴"
            self.update_status(
                f"{risk_emoji} Distance: {c.goal_distance:.1f}m | Collision Risk: {c.collision_risk*100:.0f}% | Comm: {c.comm_count} | Step: {c.step_count}",
                '#00ff00' if c.collision_risk < 0.5 else '#ffff00'
            )
        
        except Exception as e:
            pass
        
        # Schedule next update
        if self.update_running:
            self.root.after(100, self.update_telemetry)
    
    def update_graphs(self):
        """Update real-time graphs"""
        if not self.update_running:
            return
        
        try:
            c = self.controller
            
            # Clear axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
            # Plot 1: Distance to goal
            if len(c.distance_history) > 0:
                self.ax1.plot(list(c.distance_history), color='#00ff00', linewidth=2)
                self.ax1.set_title('Distance to Goal (m)', color='white', fontsize=10)
                self.ax1.axhline(y=5.0, color='#00ffff', linestyle='--', linewidth=1, alpha=0.5, label='Goal Zone')
                self.ax1.legend(fontsize=8)
            
            # Plot 2: Collision risk
            if len(c.collision_history) > 0:
                colors = ['#00ff00' if r < 30 else '#ffff00' if r < 70 else '#ff0000' for r in c.collision_history]
                self.ax2.plot(list(c.collision_history), color='#ff0000', linewidth=2)
                self.ax2.axhline(y=30, color='#00ff00', linestyle='--', linewidth=1, alpha=0.5)
                self.ax2.axhline(y=70, color='#ff0000', linestyle='--', linewidth=1, alpha=0.5)
                self.ax2.set_title('Collision Risk (%)', color='white', fontsize=10)
                self.ax2.set_ylim(0, 100)
            
            # Plot 3: Speed
            if len(c.speed_history) > 0:
                self.ax3.plot(list(c.speed_history), color='#00bfff', linewidth=2)
                self.ax3.set_title('Speed (m/s)', color='white', fontsize=10)
            
            # Plot 4: Communication
            if len(c.comm_history) > 0:
                self.ax4.plot(list(c.comm_history), color='#ff00ff', linewidth=2)
                self.ax4.set_title('Active Communications', color='white', fontsize=10)
            
            # Configure all axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.tick_params(colors='white', labelsize=8)
                ax.set_facecolor('#1e1e1e')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#555555')
                ax.grid(True, alpha=0.2, color='#555555')
            
            self.canvas.draw()
        
        except Exception as e:
            pass
        
        # Schedule next update
        if self.update_running:
            self.root.after(500, self.update_graphs)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GNNSwarmGUI(root)
    root.mainloop()
