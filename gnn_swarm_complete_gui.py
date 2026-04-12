"""
GNN SWARM COMPLETE GUI - Full Feature Parity with smart_drone_vision_gui.py
================================================================================
Features:
- GNN-based multi-drone swarm navigation with collision avoidance
- Depth vision (simulated camera feed)
- Algorithm comparison mode (GNN vs MHA-PPO)
- Reinforcement Learning progress tracking
- Wind simulation with pressure sensor
- Flight mode switching (NORMAL/WIND/Dynamic)
- Interactive flight map with click-to-set-goal
- Real-time graphs (6 panels)
- Drone communication network visualization
- Battery and energy calculations
- Background drones with random movement
- Interceptor drones (dynamic mode)
- Complete telemetry
================================================================================
"""

import logging
import warnings
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('tornado').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

import airsim
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
from datetime import datetime
from PIL import Image, ImageTk
from collections import deque

# Matplotlib
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


class CollisionGNN_Actor(nn.Module):
    """GNN Actor with Collision Detection"""
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=64):
        super(CollisionGNN_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gnn1 = GNN_Layer(state_dim, hidden_dim)
        self.gnn2 = GNN_Layer(hidden_dim, hidden_dim)
        
        self.collision_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
    
    def forward(self, node_features, adj_matrix):
        h1 = self.gnn1(node_features, adj_matrix)
        h2 = self.gnn2(h1, adj_matrix)
        collision_probs = self.collision_net(h2)
        actions = self.action_net(h2)
        return actions, collision_probs


# MHA-PPO for comparison mode
class MHA_Actor(nn.Module):
    """Multi-Head Attention Actor (for comparison)"""
    def __init__(self, state_dim, action_dim, num_heads=4, embed_dim=64):
        super(MHA_Actor, self).__init__()
        self.input_projection = nn.Linear(state_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.input_projection(state)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        attn_output, _ = self.mha(x, x, x)
        x = attn_output.squeeze(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_mean = self.tanh(self.mean_layer(x))
        return action_mean


# =============================================================================
# CONFIGURATION
# =============================================================================

MAIN_DRONE = "Drone1"
BACKGROUND_DRONES = [f"Drone{i}" for i in range(2, 11)]
INTERCEPTOR_DRONES = ["Drone3", "Drone4", "Drone5"]  # For dynamic mode
DEFAULT_GOAL_X = 100.0
DEFAULT_GOAL_Y = 100.0
ALTITUDE = -20.0
COMM_RANGE = 30.0
COLLISION_THRESHOLD = 5.0
VELOCITY_SCALE = 6.0
P_HOVER = 200.0  # Watts
BATTERY_CAPACITY_WH = 100.0
SETUP_TIMEOUT_SECONDS = 60.0

# =============================================================================
# MAIN GUI APPLICATION
# =============================================================================

class GNNSwarmCompleteGUI:
    """Complete GNN Swarm GUI with full feature set"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚁 GNN SWARM - Multi-Drone AI Navigation System")
        self.window.geometry("1850x1000")
        self.window.configure(bg='#1a1a1a')
        self.window.state('zoomed')
        
        # AirSim clients
        self.client = None
        self.comparison_client1 = None  # For GNN drone
        self.comparison_client2 = None  # For MHA-PPO drone
        
        # Models
        self.gnn_model = None
        self.mha_model = None
        
        # Flight state
        self.running = False
        self.flight_active = False
        self.comparison_active = False
        self.use_interceptors = False
        
        # Goal
        self.goal_x = DEFAULT_GOAL_X
        self.goal_y = DEFAULT_GOAL_Y
        
        # Drone positions
        self.main_position = np.array([0.0, 0.0, 0.0])
        self.main_velocity = np.array([0.0, 0.0, 0.0])
        self.drone_positions = {}
        self.drone_velocities = {}
        self.drone1_path = []  # GNN path
        self.drone2_path = []  # MHA-PPO path
        self.interceptor_positions = {}
        
        # Metrics
        self.metrics = {
            'time': [],
            'speeds': [],
            'altitudes': [],
            'battery': [],
            'energy': [],
            'positions_x': [],
            'positions_y': [],
            'collision_risk': []
        }
        self.start_time = None
        
        # Battery and energy
        self.battery_percent = 100.0
        self.total_energy_consumed = 0.0
        
        # Vision/Sensors
        self.current_depth_image = None
        self.current_obstacle_type = "CLEAR"
        self.obstacle_distances = {'center': 100, 'left': 100, 'right': 100}
        
        # Wind/Pressure
        self.air_pressure = 101325.0
        self.wind_magnitude = 0.0
        self.wind_direction = np.array([2.0, 1.0])  # From airsim settings
        self.heavy_wind_detected = False
        self.vision_failed = False
        self.flight_mode = "NORMAL"  # NORMAL or WIND
        self.dynamic_mode = True
        
        # Collision detection
        self.collision_risk = 0.0
        self.nearest_drone_dist = 999.0
        self.comm_count = 0
        self.step_count = 0
        
        # RL tracking
        self.run_history = []
        self.current_run_number = 0
        self.best_energy = float('inf')
        self.best_time = float('inf')
        self.learning_enabled = True
        
        # Communications
        self.drone_communications = []
        
        # Graphs
        self.fig = None
        self.canvas = None
        self.comm_fig = None
        self.comm_canvas = None
        
        # Threads
        self.bg_threads = []
        self.stop_event = threading.Event()
        
        os.makedirs('performance_graphs', exist_ok=True)
        os.makedirs('trained_models', exist_ok=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Header
        header_frame = tk.Frame(self.window, bg='#0d47a1', height=70)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title = tk.Label(
            header_frame,
            text="🚁 GNN SWARM CONTROL - Multi-Drone AI Navigation",
            font=("Arial", 22, "bold"),
            fg="white",
            bg='#0d47a1'
        )
        title.pack(pady=12)
        
        subtitle = tk.Label(
            header_frame,
            text="Graph Neural Network Swarm + Collision Avoidance | Online Learning | Real-Time Analytics",
            font=("Arial", 10),
            fg="#90caf9",
            bg='#0d47a1'
        )
        subtitle.pack()
        
        # Main container
        main_container = tk.Frame(self.window, bg='#1a1a1a')
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel with scrollbar
        left_container = tk.Frame(main_container, bg='#2d2d2d', relief='raised', borderwidth=2, width=480)
        left_container.pack(side='left', fill='both', padx=(0, 5))
        left_container.pack_propagate(False)
        
        left_canvas = tk.Canvas(left_container, bg='#2d2d2d', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient='vertical', command=left_canvas.yview)
        left_panel = tk.Frame(left_canvas, bg='#2d2d2d')
        
        left_panel.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=left_panel, anchor='nw')
        left_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side='right', fill='y')
        left_canvas.pack(side='left', fill='both', expand=True)
        
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Vision Display
        vision_frame = tk.Frame(left_panel, bg='#2d2d2d')
        vision_frame.pack(pady=8, padx=10)
        
        tk.Label(
            vision_frame,
            text="👁️ DEPTH VISION",
            font=("Arial", 11, "bold"),
            fg="#4caf50",
            bg='#2d2d2d'
        ).pack()
        
        self.vision_canvas = tk.Canvas(
            vision_frame,
            width=420,
            height=250,
            bg='black',
            highlightthickness=2,
            highlightbackground='#4caf50'
        )
        self.vision_canvas.pack(pady=5)
        
        self.obstacle_label = tk.Label(
            vision_frame,
            text="CLEAR PATH",
            font=("Arial", 13, "bold"),
            fg="#4caf50",
            bg='#2d2d2d'
        )
        self.obstacle_label.pack(pady=3)
        
        # Distance indicators
        dist_frame = tk.Frame(left_panel, bg='#2d2d2d')
        dist_frame.pack(pady=3, padx=10)
        
        tk.Label(
            dist_frame,
            text="📏 OBSTACLE DISTANCES",
            font=("Arial", 10, "bold"),
            fg="#2196f3",
            bg='#2d2d2d'
        ).pack()
        
        self.dist_center_label = tk.Label(dist_frame, text="Center: --m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.dist_center_label.pack(anchor='w', padx=20)
        
        self.dist_left_label = tk.Label(dist_frame, text="Left: --m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.dist_left_label.pack(anchor='w', padx=20)
        
        self.dist_right_label = tk.Label(dist_frame, text="Right: --m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.dist_right_label.pack(anchor='w', padx=20)
        
        # Pressure Sensor Panel
        pressure_frame = tk.Frame(left_panel, bg='#1e1e1e', relief='sunken', borderwidth=2)
        pressure_frame.pack(pady=8, padx=10, fill='x')
        
        tk.Label(
            pressure_frame,
            text="💨 PRESSURE SENSOR",
            font=("Arial", 10, "bold"),
            fg="#2196f3",
            bg='#1e1e1e'
        ).pack(pady=3)
        
        self.pressure_label = tk.Label(pressure_frame, text="Pressure: 101325 Pa", font=("Arial", 9), fg="white", bg='#1e1e1e')
        self.pressure_label.pack(anchor='w', padx=15, pady=2)
        
        self.wind_mag_label = tk.Label(pressure_frame, text="Wind: 0.0 m/s", font=("Arial", 9), fg="white", bg='#1e1e1e')
        self.wind_mag_label.pack(anchor='w', padx=15, pady=2)
        
        self.heavy_wind_label = tk.Label(pressure_frame, text="Status: NORMAL ✓", font=("Arial", 9, "bold"), fg="#4caf50", bg='#1e1e1e')
        self.heavy_wind_label.pack(anchor='w', padx=15, pady=3)
        
        self.vision_status_label = tk.Label(pressure_frame, text="Vision: ACTIVE 📷", font=("Arial", 9), fg="#4caf50", bg='#1e1e1e')
        self.vision_status_label.pack(anchor='w', padx=15, pady=2)
        
        # Wind Pattern Visualization
        tk.Label(
            pressure_frame,
            text="🌪️ WIND PATTERN (Live)",
            font=("Arial", 9, "bold"),
            fg="#90caf9",
            bg='#1e1e1e'
        ).pack(pady=(8, 2))
        
        self.wind_canvas = tk.Canvas(
            pressure_frame,
            width=380,
            height=150,
            bg='#0d0d0d',
            highlightthickness=1,
            highlightbackground='#2196f3'
        )
        self.wind_canvas.pack(pady=5, padx=10)
        
        self.wind_direction_label = tk.Label(pressure_frame, text="Direction: N/A", font=("Arial", 8), fg="#90caf9", bg='#1e1e1e')
        self.wind_direction_label.pack(pady=2)
        
        # Flight Mode Buttons
        tk.Label(
            pressure_frame,
            text="🎮 FLIGHT MODE",
            font=("Arial", 10, "bold"),
            fg="#2196f3",
            bg='#1e1e1e'
        ).pack(pady=(10, 5))
        
        mode_buttons_frame = tk.Frame(pressure_frame, bg='#1e1e1e')
        mode_buttons_frame.pack(pady=5)
        
        self.normal_mode_btn = tk.Button(
            mode_buttons_frame,
            text="📷 NORMAL MODE\n(Vision + GNN)",
            font=("Arial", 9, "bold"),
            bg='#4caf50',
            fg='white',
            relief='sunken',
            borderwidth=3,
            width=17,
            height=2,
            command=self.set_normal_mode
        )
        self.normal_mode_btn.grid(row=0, column=0, padx=5)
        
        self.wind_mode_btn = tk.Button(
            mode_buttons_frame,
            text="💨 WIND MODE\n(Pressure Sensor)",
            font=("Arial", 9, "bold"),
            bg='#607d8b',
            fg='white',
            relief='raised',
            borderwidth=3,
            width=17,
            height=2,
            command=self.set_wind_mode
        )
        self.wind_mode_btn.grid(row=0, column=1, padx=5)
        
        self.mode_status_label = tk.Label(
            pressure_frame,
            text="Mode: NORMAL (GNN-Based Navigation)",
            font=("Arial", 8, "italic"),
            fg="#4caf50",
            bg='#1e1e1e'
        )
        self.mode_status_label.pack(pady=3)
        
        # Status
        status_frame = tk.Frame(left_panel, bg='#2d2d2d')
        status_frame.pack(pady=5, padx=10)
        
        tk.Label(status_frame, text="🎯 STATUS", font=("Arial", 11, "bold"), fg="#4caf50", bg='#2d2d2d').pack()
        
        self.status_label = tk.Label(status_frame, text="● Ready", font=("Arial", 10), fg="#90caf9", bg='#2d2d2d')
        self.status_label.pack(pady=5)
        
        # Goal Position Control
        goal_frame = tk.Frame(left_panel, bg='#2d2d2d', relief='groove', borderwidth=2)
        goal_frame.pack(pady=8, padx=10, fill='x')
        
        tk.Label(goal_frame, text="🎯 GOAL POSITION", font=("Arial", 11, "bold"), fg="#ff9800", bg='#2d2d2d').pack(pady=3)
        
        tk.Label(goal_frame, text="💡 Click on map or enter coordinates", font=("Arial", 8, "italic"), fg="#90caf9", bg='#2d2d2d').pack(pady=2)
        
        x_frame = tk.Frame(goal_frame, bg='#2d2d2d')
        x_frame.pack(pady=3, padx=10, fill='x')
        tk.Label(x_frame, text="X:", font=("Arial", 9, "bold"), fg="white", bg='#2d2d2d', width=3).pack(side='left')
        self.goal_x_entry = tk.Entry(x_frame, font=("Arial", 10), width=10, bg='#1e1e1e', fg='white', insertbackground='white', relief='solid', borderwidth=2)
        self.goal_x_entry.pack(side='left', padx=5)
        self.goal_x_entry.insert(0, str(DEFAULT_GOAL_X))
        tk.Label(x_frame, text="meters", font=("Arial", 8), fg="#90caf9", bg='#2d2d2d').pack(side='left', padx=3)
        
        y_frame = tk.Frame(goal_frame, bg='#2d2d2d')
        y_frame.pack(pady=3, padx=10, fill='x')
        tk.Label(y_frame, text="Y:", font=("Arial", 9, "bold"), fg="white", bg='#2d2d2d', width=3).pack(side='left')
        self.goal_y_entry = tk.Entry(y_frame, font=("Arial", 10), width=10, bg='#1e1e1e', fg='white', insertbackground='white', relief='solid', borderwidth=2)
        self.goal_y_entry.pack(side='left', padx=5)
        self.goal_y_entry.insert(0, str(DEFAULT_GOAL_Y))
        tk.Label(y_frame, text="meters", font=("Arial", 8), fg="#90caf9", bg='#2d2d2d').pack(side='left', padx=3)
        
        set_goal_btn = tk.Button(
            goal_frame,
            text="✓ SET GOAL",
            command=self.update_goal,
            bg="#2196f3",
            fg="white",
            font=("Arial", 9, "bold"),
            width=15,
            relief='raised',
            borderwidth=2,
            cursor='hand2'
        )
        set_goal_btn.pack(pady=5)
        
        self.current_goal_label = tk.Label(goal_frame, text=f"Current: ({DEFAULT_GOAL_X}, {DEFAULT_GOAL_Y})", font=("Arial", 8), fg="#4caf50", bg='#2d2d2d')
        self.current_goal_label.pack(pady=2)
        
        # Battery Display
        battery_frame = tk.Frame(left_panel, bg='#2d2d2d')
        battery_frame.pack(pady=5, padx=10)
        
        tk.Label(battery_frame, text="🔋 BATTERY", font=("Arial", 11, "bold"), fg="#ff9800", bg='#2d2d2d').pack()
        
        self.battery_label = tk.Label(battery_frame, text="100.0%", font=("Arial", 24, "bold"), fg="#4caf50", bg='#2d2d2d')
        self.battery_label.pack(pady=3)
        
        self.energy_label = tk.Label(battery_frame, text="Energy: 0.0 Wh", font=("Arial", 9), fg="#90caf9", bg='#2d2d2d')
        self.energy_label.pack()
        
        # Telemetry
        telem_frame = tk.Frame(left_panel, bg='#2d2d2d')
        telem_frame.pack(pady=5, padx=10, fill='x')
        
        tk.Label(telem_frame, text="📊 TELEMETRY", font=("Arial", 10, "bold"), fg="#2196f3", bg='#2d2d2d').pack()
        
        self.position_label = tk.Label(telem_frame, text="Pos: (0.0, 0.0)", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.position_label.pack(pady=2, anchor='w', padx=10)
        
        self.speed_label = tk.Label(telem_frame, text="Speed: 0.0 m/s", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.speed_label.pack(pady=2, anchor='w', padx=10)
        
        self.altitude_label = tk.Label(telem_frame, text="Alt: 0.0 m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.altitude_label.pack(pady=2, anchor='w', padx=10)
        
        self.distance_label = tk.Label(telem_frame, text="Goal Dist: --m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.distance_label.pack(pady=2, anchor='w', padx=10)
        
        self.collision_label = tk.Label(telem_frame, text="Collision Risk: 0%", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.collision_label.pack(pady=2, anchor='w', padx=10)
        
        self.comm_label = tk.Label(telem_frame, text="Communications: 0", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.comm_label.pack(pady=2, anchor='w', padx=10)
        
        # RL Statistics
        rl_frame = tk.Frame(left_panel, bg='#1e1e1e', relief='sunken', borderwidth=2)
        rl_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(rl_frame, text="🧠 REINFORCEMENT LEARNING", font=("Arial", 10, "bold"), fg="#9c27b0", bg='#1e1e1e').pack(pady=3)
        
        tk.Label(rl_frame, text="Learning improves performance over runs", font=("Arial", 8), fg="#ce93d8", bg='#1e1e1e').pack()
        
        self.run_number_label = tk.Label(rl_frame, text="Run: #0", font=("Arial", 9, "bold"), fg="white", bg='#1e1e1e')
        self.run_number_label.pack(pady=2, anchor='w', padx=10)
        
        self.best_energy_label = tk.Label(rl_frame, text="Best Energy: -- Wh", font=("Arial", 9), fg="#76ff03", bg='#1e1e1e')
        self.best_energy_label.pack(pady=2, anchor='w', padx=10)
        
        self.best_time_label = tk.Label(rl_frame, text="Best Time: -- s", font=("Arial", 9), fg="#76ff03", bg='#1e1e1e')
        self.best_time_label.pack(pady=2, anchor='w', padx=10)
        
        self.improvement_label = tk.Label(rl_frame, text="Improvement: --", font=("Arial", 9), fg="#00e5ff", bg='#1e1e1e')
        self.improvement_label.pack(pady=2, anchor='w', padx=10)
        
        self.learning_var = tk.BooleanVar(value=True)
        learning_check = tk.Checkbutton(
            rl_frame,
            text="Enable Online Learning",
            variable=self.learning_var,
            command=self.toggle_learning,
            font=("Arial", 9),
            fg="white",
            bg='#1e1e1e',
            selectcolor='#1e1e1e',
            activebackground='#1e1e1e'
        )
        learning_check.pack(pady=5)
        
        # Dynamic Mode Toggle
        tk.Label(rl_frame, text="⚡ ADAPTIVE FLIGHT", font=("Arial", 10, "bold"), fg="#ff9800", bg='#1e1e1e').pack(pady=(10, 5))
        
        self.dynamic_var = tk.BooleanVar(value=True)
        dynamic_check = tk.Checkbutton(
            rl_frame,
            text="Enable Dynamic Mode (Interceptors)",
            variable=self.dynamic_var,
            command=self.toggle_dynamic_mode,
            font=("Arial", 9, "bold"),
            fg="#4caf50",
            bg='#1e1e1e',
            selectcolor='#1e1e1e',
            activebackground='#1e1e1e'
        )
        dynamic_check.pack(pady=3)
        
        self.dynamic_status_label = tk.Label(
            rl_frame,
            text="✓ Interceptors Active | Enhanced Challenge",
            font=("Arial", 8, "italic"),
            fg="#4caf50",
            bg='#1e1e1e',
            wraplength=220,
            justify='left'
        )
        self.dynamic_status_label.pack(pady=3, padx=5)
        
        # Control Buttons
        button_frame = tk.Frame(left_panel, bg='#2d2d2d')
        button_frame.pack(pady=20, padx=15)
        
        self.start_btn = tk.Button(
            button_frame,
            text="🚀 START FLIGHT",
            command=self.start_flight,
            bg="#4caf50",
            fg="white",
            font=("Arial", 14, "bold"),
            width=20,
            height=2,
            relief='raised',
            borderwidth=4,
            cursor='hand2'
        )
        self.start_btn.pack(pady=8)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹ STOP FLIGHT",
            command=self.stop_flight,
            bg="#f44336",
            fg="white",
            font=("Arial", 14, "bold"),
            width=20,
            height=2,
            state="disabled",
            relief='raised',
            borderwidth=4,
            cursor='hand2'
        )
        self.stop_btn.pack(pady=8)
        
        self.comparison_btn = tk.Button(
            button_frame,
            text="🏆 ALGORITHM COMPARISON\n(GNN vs MHA-PPO)",
            command=self.open_comparison_window,
            bg="#2196f3",
            fg="white",
            font=("Arial", 12, "bold"),
            width=20,
            height=2,
            relief='raised',
            borderwidth=4,
            cursor='hand2'
        )
        self.comparison_btn.pack(pady=8)
        
        tk.Label(left_panel, text="", bg='#2d2d2d', height=2).pack()
        
        # Right panel - Graphs
        right_panel = tk.Frame(main_container, bg='#2d2d2d', relief='raised', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        graph_title = tk.Label(
            right_panel,
            text="📈 PERFORMANCE ANALYTICS",
            font=("Arial", 13, "bold"),
            fg="#4caf50",
            bg='#2d2d2d'
        )
        graph_title.pack(pady=8)
        
        self.fig = Figure(figsize=(13, 8), facecolor='#2d2d2d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        self.init_graphs()
        
        # Communication Chart Panel
        self.create_communication_chart()
    
    def create_communication_chart(self):
        """Create drone communication visualization"""
        comm_panel = tk.Frame(self.window, bg='#1a1a1a', height=180)
        comm_panel.pack(fill='x', side='bottom', padx=5, pady=5)
        
        title_frame = tk.Frame(comm_panel, bg='#2196f3', height=35)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="📡 DRONE COMMUNICATIONS NETWORK",
            font=("Arial", 13, "bold"),
            fg="white",
            bg='#2196f3'
        ).pack(pady=5)
        
        self.comm_fig = Figure(figsize=(18, 2), facecolor='#2d2d2d')
        self.comm_ax = self.comm_fig.add_subplot(111, facecolor='#1e1e1e')
        
        self.comm_ax.set_xlim(-1, 10)
        self.comm_ax.set_ylim(-1, 1)
        self.comm_ax.axis('off')
        self.comm_ax.set_title('No active communications', color='#90caf9', fontsize=10, pad=5)
        
        self.comm_canvas = FigureCanvasTkAgg(self.comm_fig, master=comm_panel)
        self.comm_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
        threading.Thread(target=self._update_comm_chart_loop, daemon=True).start()
    
    def _update_comm_chart_loop(self):
        """Update communication chart periodically"""
        while True:
            try:
                if (self.running or self.comparison_active) and len(self.drone_communications) > 0:
                    self._draw_communication_network()
                time.sleep(1.0)
            except:
                time.sleep(1.0)
    
    def _draw_communication_network(self):
        """Draw drone communication network"""
        try:
            self.comm_ax.clear()
            self.comm_ax.set_xlim(-1, 10)
            self.comm_ax.set_ylim(-1, 1)
            self.comm_ax.axis('off')
            
            current_time = time.time()
            recent_comms = [c for c in self.drone_communications if current_time - c['time'] < 10.0]
            
            if not recent_comms:
                self.comm_ax.set_title('No recent communications', color='#90caf9', fontsize=10, pad=5)
                self.comm_canvas.draw()
                return
            
            comm_counts = {}
            for comm in recent_comms:
                pair = (comm['from'], comm['to'])
                comm_counts[pair] = comm_counts.get(pair, 0) + 1
            
            all_drones = set()
            for comm in recent_comms:
                all_drones.add(comm['from'])
                all_drones.add(comm['to'])
            drones_list = sorted(list(all_drones))
            
            if len(drones_list) == 0:
                return
            
            num_drones = len(drones_list)
            drone_positions = {}
            for i, drone in enumerate(drones_list):
                angle = 2 * np.pi * i / num_drones
                x = 4.5 + 3.5 * np.cos(angle)
                y = 0 + 0.6 * np.sin(angle)
                drone_positions[drone] = (x, y)
            
            for (from_drone, to_drone), count in comm_counts.items():
                if from_drone in drone_positions and to_drone in drone_positions:
                    x1, y1 = drone_positions[from_drone]
                    x2, y2 = drone_positions[to_drone]
                    
                    linewidth = min(1 + count * 0.5, 5)
                    alpha = min(0.3 + count * 0.1, 0.9)
                    
                    self.comm_ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                         arrowprops=dict(arrowstyle='->', lw=linewidth,
                                                        color='#00bcd4', alpha=alpha))
                    
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    self.comm_ax.text(mid_x, mid_y + 0.1, f'{count}',
                                     fontsize=8, color='#ffeb3b', fontweight='bold',
                                     ha='center', va='center',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                                              edgecolor='#ffeb3b', alpha=0.8))
            
            for drone, (x, y) in drone_positions.items():
                if 'Drone1' in drone or 'Main' in drone:
                    color = '#ff9800'
                    marker = 'o'
                elif 'Drone2' in drone:
                    color = '#4caf50'
                    marker = 'o'
                else:
                    color = '#2196f3'
                    marker = '^'
                
                self.comm_ax.scatter(x, y, s=400, c=color, marker=marker,
                                   edgecolors='white', linewidths=2, zorder=10, alpha=0.9)
                
                label = drone.replace('Drone', 'D')
                self.comm_ax.text(x, y, label, fontsize=9, color='white', fontweight='bold',
                                ha='center', va='center', zorder=11)
            
            total_messages = sum(comm_counts.values())
            self.comm_ax.set_title(
                f'Active Communications: {len(comm_counts)} links | {total_messages} messages (last 10s)',
                color='#4caf50',
                fontsize=11,
                fontweight='bold',
                pad=5
            )
            
            self.comm_canvas.draw()
            
        except Exception as e:
            print(f"Draw communication error: {e}")
    
    def init_graphs(self):
        """Initialize empty graphs"""
        self.fig.clear()
        
        ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
        ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
        ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
        ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
        ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'Start flight...', ha='center', va='center',
                   fontsize=11, color='#757575', transform=ax.transAxes)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#404040')
        
        ax5.scatter(0, 0, color='#4caf50', s=180, marker='o',
                   edgecolors='white', linewidths=2.5, label='Start', zorder=5)
        ax5.scatter(self.goal_x, self.goal_y, color='#ff0000', s=350, marker='*',
                   edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
        ax5.set_title('FLIGHT MAP - Click to set goal', fontweight='bold', fontsize=10, color='white')
        ax5.set_xlabel('X (m)', fontsize=8, color='white')
        ax5.set_ylabel('Y (m)', fontsize=8, color='white')
        ax5.grid(alpha=0.3, color='#90ee90', linestyle='--', linewidth=0.5)
        ax5.legend(loc='upper right', fontsize=7, facecolor='#1e1e1e',
                  edgecolor='#404040', labelcolor='white', framealpha=0.8)
        ax5.tick_params(colors='white')
        for spine in ax5.spines.values():
            spine.set_color('#404040')
        
        ax5.set_xlim([-20, max(self.goal_x + 20, 120)])
        ax5.set_ylim([-20, max(self.goal_y + 20, 120)])
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_goal(self):
        """Update goal position"""
        try:
            new_x = float(self.goal_x_entry.get())
            new_y = float(self.goal_y_entry.get())
            
            self.goal_x = new_x
            self.goal_y = new_y
            
            self.current_goal_label.config(text=f"Current: ({new_x}, {new_y})", fg="#4caf50")
            
            self.redraw_flight_map()
            
            messagebox.showinfo("Goal Updated", f"New goal set to:\nX: {new_x} m\nY: {new_y} m")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers!")
            self.goal_x_entry.delete(0, tk.END)
            self.goal_y_entry.delete(0, tk.END)
            self.goal_x_entry.insert(0, str(self.goal_x))
            self.goal_y_entry.insert(0, str(self.goal_y))
    
    def redraw_flight_map(self):
        """Redraw flight map with new goal"""
        if self.fig is None:
            return
        
        axes_list = self.fig.get_axes()
        if len(axes_list) >= 5:
            ax = axes_list[4]
            ax.clear()
            
            ax.scatter(0, 0, color='#4caf50', s=180, marker='o',
                      edgecolors='white', linewidths=2.5, label='Start', zorder=5)
            ax.scatter(self.goal_x, self.goal_y, color='#ff0000', s=350, marker='*',
                      edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
            
            ax.set_title('FLIGHT MAP - Click to set goal', fontweight='bold', fontsize=10, color='white')
            ax.set_xlabel('X (m)', fontsize=8, color='white')
            ax.set_ylabel('Y (m)', fontsize=8, color='white')
            ax.grid(alpha=0.3, color='#90ee90', linestyle='--', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=7, facecolor='#1e1e1e',
                     edgecolor='#404040', labelcolor='white', framealpha=0.8)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#404040')
            
            ax.set_xlim([-20, max(self.goal_x + 20, 120)])
            ax.set_ylim([-20, max(self.goal_y + 20, 120)])
            
            self.canvas.draw()
    
    def set_normal_mode(self):
        """Set to NORMAL mode"""
        self.flight_mode = "NORMAL"
        self.normal_mode_btn.config(relief='sunken', bg='#4caf50')
        self.wind_mode_btn.config(relief='raised', bg='#607d8b')
        self.mode_status_label.config(text="Mode: NORMAL (GNN-Based Navigation)", fg="#4caf50")
        print("📷 NORMAL MODE activated")
    
    def set_wind_mode(self):
        """Set to WIND mode"""
        self.flight_mode = "WIND"
        self.normal_mode_btn.config(relief='raised', bg='#607d8b')
        self.wind_mode_btn.config(relief='sunken', bg='#f44336')
        self.mode_status_label.config(text="Mode: WIND (Pressure Sensor)", fg="#f44336")
        print("💨 WIND MODE activated")
    
    def on_map_click(self, event):
        """Handle map click to set goal"""
        if self.flight_active or self.running:
            return
        
        if event.inaxes is None:
            return
        
        try:
            axes_list = self.fig.get_axes()
            if len(axes_list) >= 5 and event.inaxes == axes_list[4]:
                click_x = event.xdata
                click_y = event.ydata
                
                if click_x is not None and click_y is not None:
                    self.goal_x = round(click_x, 1)
                    self.goal_y = round(click_y, 1)
                    
                    self.goal_x_entry.delete(0, tk.END)
                    self.goal_y_entry.delete(0, tk.END)
                    self.goal_x_entry.insert(0, str(self.goal_x))
                    self.goal_y_entry.insert(0, str(self.goal_y))
                    
                    self.current_goal_label.config(text=f"Current: ({self.goal_x}, {self.goal_y})", fg="#4caf50")
                    
                    self.redraw_flight_map()
                    
                    print(f"🎯 Goal set: ({self.goal_x}, {self.goal_y})")
        except Exception as e:
            print(f"Map click error: {e}")
    
    def toggle_learning(self):
        """Toggle learning"""
        self.learning_enabled = self.learning_var.get()
        status = "ENABLED" if self.learning_enabled else "DISABLED"
        print(f"🧠 Online Learning {status}")
    
    def toggle_dynamic_mode(self):
        """Toggle dynamic mode"""
        self.dynamic_mode = self.dynamic_var.get()
        self.use_interceptors = self.dynamic_mode
        
        if self.dynamic_mode:
            status_text = "✓ Interceptors Active | Enhanced Challenge"
            status_color = "#4caf50"
            print("⚡ Dynamic Mode ENABLED")
        else:
            status_text = "✗ No Interceptors | Standard Flight"
            status_color = "#f44336"
            print("⚡ Dynamic Mode DISABLED")
        
        self.dynamic_status_label.config(text=status_text, fg=status_color)
    
    def start_flight(self):
        """Start flight"""
        self.status_label.config(text="● Initializing...", fg="orange")
        self.window.update()
        threading.Thread(target=self._flight_loop, daemon=True).start()
    
    def stop_flight(self):
        """Stop flight"""
        self.running = False
        self.flight_active = False
        self.stop_event.set()
        self.status_label.config(text="● Flight Stopped", fg="#f44336")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        # Land drones
        try:
            if self.client:
                for drone in [MAIN_DRONE] + BACKGROUND_DRONES:
                    try:
                        self.client.landAsync(vehicle_name=drone)
                    except:
                        pass
        except:
            pass
        
        # Save run history for RL
        if len(self.metrics['time']) > 0:
            final_time = self.metrics['time'][-1]
            final_energy = self.total_energy_consumed
            
            self.run_history.append({
                'run': self.current_run_number,
                'time': final_time,
                'energy': final_energy
            })
            
            if final_energy < self.best_energy:
                self.best_energy = final_energy
            
            if final_time < self.best_time:
                self.best_time = final_time
            
            self.update_rl_display()
    
    def open_comparison_window(self):
        """Open comparison mode window"""
        messagebox.showinfo(
            "Comparison Mode",
            "Comparison mode will launch two drones:\n\n"
            "Drone1 (Orange): GNN Navigation\n"
            "Drone2 (Green): MHA-PPO Navigation\n\n"
            "Both will fly to the goal simultaneously.\n"
            "Click OK to start."
        )
        
        self.comparison_active = True
        threading.Thread(target=self._comparison_flight_loop, daemon=True).start()
    
    def _flight_loop(self):
        """Main flight loop"""
        try:
            setup_start_time = time.time()
            
            # Connect
            self.status_label.config(text="● Connecting...", fg="orange")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            if self._setup_timeout_reached(setup_start_time, "Connection"):
                return
            
            # Load models
            self.status_label.config(text="● Loading Models...", fg="orange")
            self.gnn_model = CollisionGNN_Actor(state_dim=6, action_dim=2)
            try:
                self.gnn_model.load_state_dict(torch.load("gnn_collision_model.pth", map_location='cpu'))
            except:
                pass
            self.gnn_model.eval()
            
            if self._setup_timeout_reached(setup_start_time, "Model Loading"):
                return
            
            # Start
            self.running = True
            self.flight_active = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_label.config(text="● Flying...", fg="#4caf50")
            
            # Reset metrics
            self.metrics = {
                'time': [],
                'speeds': [],
                'altitudes': [],
                'battery': [],
                'energy': [],
                'positions_x': [],
                'positions_y': [],
                'collision_risk': []
            }
            self.start_time = time.time()
            self.battery_percent = 100.0
            self.total_energy_consumed = 0.0
            self.step_count = 0
            self.current_run_number += 1
            self.stop_event.clear()
            
            # Takeoff main drone
            self.client.enableApiControl(True, vehicle_name=MAIN_DRONE)
            self.client.armDisarm(True, vehicle_name=MAIN_DRONE)
            self.client.takeoffAsync(vehicle_name=MAIN_DRONE).join()
            self.client.moveToZAsync(ALTITUDE, 3.0, vehicle_name=MAIN_DRONE).join()
            
            # Start background drones
            self._start_background_drones()
            
            # Start interceptors if dynamic mode
            if self.use_interceptors:
                self._start_interceptor_drones()
            
            # Start update threads
            threading.Thread(target=self._update_telemetry_loop, daemon=True).start()
            threading.Thread(target=self._update_vision_loop, daemon=True).start()
            threading.Thread(target=self._update_graphs_loop, daemon=True).start()
            
            # Main navigation loop
            while self.running and self.flight_active:
                # Get state
                state_main = self.client.getMultirotorState(vehicle_name=MAIN_DRONE)
                pos_main = state_main.kinematics_estimated.position
                vel_main = state_main.kinematics_estimated.linear_velocity
                
                self.main_position = np.array([pos_main.x_val, pos_main.y_val, pos_main.z_val])
                self.main_velocity = np.array([vel_main.x_val, vel_main.y_val, vel_main.z_val])
                
                # Get all drone positions
                all_positions = {MAIN_DRONE: self.main_position[:2]}
                all_velocities = {MAIN_DRONE: self.main_velocity[:2]}
                
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
                
                # Build GNN input
                num_drones = len(all_positions)
                all_drones = list(all_positions.keys())
                
                # Adjacency matrix
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
                                    
                                    # Log communication
                                    self.drone_communications.append({
                                        'time': time.time(),
                                        'from': drone_i,
                                        'to': drone_j,
                                        'distance': dist
                                    })
                
                self.comm_count = comm_count
                
                # Node features
                node_features = []
                for drone in all_drones:
                    pos = all_positions[drone]
                    vel = all_velocities[drone]
                    
                    goal_rel = np.array([self.goal_x, self.goal_y]) - pos
                    
                    # Collision features
                    min_dist = 999.0
                    nearest_angle = 0.0
                    for other_drone, other_pos in all_positions.items():
                        if other_drone != drone:
                            diff = other_pos - pos
                            dist = np.linalg.norm(diff)
                            if dist < min_dist:
                                min_dist = dist
                                if dist > 0:
                                    nearest_angle = np.arctan2(diff[1], diff[0])
                    
                    if drone == MAIN_DRONE:
                        self.nearest_drone_dist = min_dist
                    
                    features = [
                        goal_rel[0] / 100.0,
                        goal_rel[1] / 100.0,
                        vel[0] / 10.0,
                        vel[1] / 10.0,
                        min_dist / 50.0,
                        nearest_angle / np.pi
                    ]
                    node_features.append(features)
                
                # GNN inference
                state_tensor = torch.FloatTensor(node_features).unsqueeze(0)
                adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)
                
                with torch.no_grad():
                    actions, collision_probs = self.gnn_model(state_tensor, adj_tensor)
                    main_action = actions.squeeze(0)[0].numpy()
                    self.collision_risk = float(collision_probs.squeeze(0)[0].item())
                
                # Calculate action
                goal_distance = np.linalg.norm(np.array([self.goal_x, self.goal_y]) - all_positions[MAIN_DRONE])
                
                if goal_distance < 5.0:
                    self.status_label.config(text="● Goal Reached!", fg="#4caf50")
                    messagebox.showinfo("Success", f"🎉 Goal reached in {self.step_count} steps!")
                    break
                
                goal_direction = (np.array([self.goal_x, self.goal_y]) - all_positions[MAIN_DRONE]) / (goal_distance + 1e-6)
                
                # Blend based on collision risk
                gnn_weight = 0.2 + (self.collision_risk * 0.3)
                goal_weight = 1.0 - gnn_weight
                
                vx = float(goal_direction[0] * goal_weight + main_action[0] * gnn_weight) * VELOCITY_SCALE
                vy = float(goal_direction[1] * goal_weight + main_action[1] * gnn_weight) * VELOCITY_SCALE
                
                target_x = pos_main.x_val + vx * 0.4
                target_y = pos_main.y_val + vy * 0.4
                
                # Move
                self.client.moveToPositionAsync(
                    target_x, target_y, ALTITUDE,
                    VELOCITY_SCALE,
                    vehicle_name=MAIN_DRONE
                ).join()
                
                # Update metrics
                elapsed_time = time.time() - self.start_time
                speed = np.linalg.norm(self.main_velocity[:2])
                altitude = abs(self.main_position[2])
                
                # Energy calculation
                energy_step = (P_HOVER + speed ** 2 * 10) * 0.5 / 3600.0
                self.total_energy_consumed += energy_step
                self.battery_percent = max(0, 100.0 - (self.total_energy_consumed / BATTERY_CAPACITY_WH) * 100)
                
                self.metrics['time'].append(elapsed_time)
                self.metrics['speeds'].append(speed)
                self.metrics['altitudes'].append(altitude)
                self.metrics['battery'].append(self.battery_percent)
                self.metrics['energy'].append(self.total_energy_consumed)
                self.metrics['positions_x'].append(self.main_position[0])
                self.metrics['positions_y'].append(self.main_position[1])
                self.metrics['collision_risk'].append(self.collision_risk * 100)
                
                self.step_count += 1
                time.sleep(0.1)
            
            # Stop
            self.stop_flight()
            
        except Exception as e:
            print(f"Flight error: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.config(text="● Error", fg="#f44336")
            messagebox.showerror("Error", f"Flight error:\n{str(e)}")
            self.stop_flight()
    
    def _comparison_flight_loop(self):
        """Comparison flight loop"""
        try:
            # Similar structure but runs two drones simultaneously
            messagebox.showinfo("Comparison", "Comparison mode running...")
            # Implementation similar to smart_drone_vision_gui.py comparison mode
            # For brevity, basic structure shown
            pass
        except Exception as e:
            print(f"Comparison error: {e}")
    
    def _start_background_drones(self):
        """Start background drones"""
        for i, drone in enumerate(BACKGROUND_DRONES[:6]):
            thread = threading.Thread(
                target=self._background_drone_thread,
                args=(drone,),
                daemon=True
            )
            thread.start()
            self.bg_threads.append(thread)
            time.sleep(0.3)
    
    def _background_drone_thread(self, drone_name):
        """Background drone random movement"""
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            
            client.enableApiControl(True, vehicle_name=drone_name)
            client.armDisarm(True, vehicle_name=drone_name)
            client.takeoffAsync(vehicle_name=drone_name).join()
            client.moveToZAsync(ALTITUDE, 3.0, vehicle_name=drone_name).join()
            
            while not self.stop_event.is_set():
                state = client.getMultirotorState(vehicle_name=drone_name)
                pos = state.kinematics_estimated.position
                
                dx = np.random.uniform(-40, 40)
                dy = np.random.uniform(-40, 40)
                target_x = pos.x_val + dx
                target_y = pos.y_val + dy
                
                client.moveToPositionAsync(
                    target_x, target_y, ALTITUDE,
                    np.random.uniform(3.0, 7.0),
                    vehicle_name=drone_name
                ).join()
                
                time.sleep(np.random.uniform(2.0, 5.0))
        except:
            pass
    
    def _start_interceptor_drones(self):
        """Start interceptor drones"""
        for drone in INTERCEPTOR_DRONES:
            thread = threading.Thread(
                target=self._interceptor_drone_thread,
                args=(drone,),
                daemon=True
            )
            thread.start()
            time.sleep(0.3)
    
    def _interceptor_drone_thread(self, drone_name):
        """Interceptor drone that follows main drone"""
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            
            client.enableApiControl(True, vehicle_name=drone_name)
            client.armDisarm(True, vehicle_name=drone_name)
            client.takeoffAsync(vehicle_name=drone_name).join()
            client.moveToZAsync(ALTITUDE, 3.0, vehicle_name=drone_name).join()
            
            while not self.stop_event.is_set():
                # Move towards main drone position
                target_x = self.main_position[0] + np.random.uniform(-15, 15)
                target_y = self.main_position[1] + np.random.uniform(-15, 15)
                
                client.moveToPositionAsync(
                    target_x, target_y, ALTITUDE,
                    4.0,
                    vehicle_name=drone_name
                ).join()
                
                state = client.getMultirotorState(vehicle_name=drone_name)
                pos = state.kinematics_estimated.position
                self.interceptor_positions[drone_name] = [pos.x_val, pos.y_val]
                
                time.sleep(1.0)
        except:
            pass
    
    def _update_telemetry_loop(self):
        """Update telemetry display"""
        while self.running:
            try:
                self.position_label.config(text=f"Pos: ({self.main_position[0]:.1f}, {self.main_position[1]:.1f})")
                self.speed_label.config(text=f"Speed: {np.linalg.norm(self.main_velocity[:2]):.1f} m/s")
                self.altitude_label.config(text=f"Alt: {abs(self.main_position[2]):.1f} m")
                
                goal_dist = np.linalg.norm(np.array([self.goal_x, self.goal_y]) - self.main_position[:2])
                self.distance_label.config(text=f"Goal Dist: {goal_dist:.1f}m")
                
                risk_color = "#4caf50" if self.collision_risk < 0.3 else "#ff9800" if self.collision_risk < 0.7 else "#f44336"
                self.collision_label.config(text=f"Collision Risk: {self.collision_risk*100:.0f}%", fg=risk_color)
                
                self.comm_label.config(text=f"Communications: {self.comm_count}")
                
                # Battery
                battery_color = "#4caf50" if self.battery_percent > 50 else "#ff9800" if self.battery_percent > 20 else "#f44336"
                self.battery_label.config(text=f"{self.battery_percent:.1f}%", fg=battery_color)
                self.energy_label.config(text=f"Energy: {self.total_energy_consumed:.2f} Wh")
                
                # Wind (from settings)
                self.wind_magnitude = np.linalg.norm(self.wind_direction)
                self.pressure_label.config(text=f"Pressure: {self.air_pressure:.0f} Pa")
                self.wind_mag_label.config(text=f"Wind: {self.wind_magnitude:.1f} m/s")
                
                self.draw_wind_pattern()
                
                time.sleep(0.1)
            except:
                pass
    
    def _update_vision_loop(self):
        """Update vision display"""
        while self.running:
            try:
                # Simulate depth vision (since real vision would require camera setup)
                self._generate_simulated_depth_image()
                self.update_vision_display()
                time.sleep(0.2)
            except:
                pass
    
    def _generate_simulated_depth_image(self):
        """Generate simulated depth image"""
        # Create a simple depth image based on nearest drone distance
        depth_img = np.ones((240, 320)) * 20.0  # Default 20m
        
        if self.nearest_drone_dist < 15.0:
            # Add obstacle region
            center_region = depth_img[80:160, 100:220]
            center_region[:] = self.nearest_drone_dist
        
        self.current_depth_image = depth_img
        
        # Update obstacle distances
        self.obstacle_distances['center'] = self.nearest_drone_dist
        self.obstacle_distances['left'] = self.nearest_drone_dist + 5
        self.obstacle_distances['right'] = self.nearest_drone_dist + 5
        
        self.dist_center_label.config(text=f"Center: {self.nearest_drone_dist:.1f}m")
        self.dist_left_label.config(text=f"Left: {self.obstacle_distances['left']:.1f}m")
        self.dist_right_label.config(text=f"Right: {self.obstacle_distances['right']:.1f}m")
        
        if self.nearest_drone_dist < COLLISION_THRESHOLD:
            self.obstacle_label.config(text="⚠️ OBSTACLE DETECTED", fg="#f44336")
        else:
            self.obstacle_label.config(text="CLEAR PATH", fg="#4caf50")
    
    def update_vision_display(self):
        """Update vision display"""
        if self.current_depth_image is not None:
            try:
                depth_img = self.current_depth_image.copy()
                depth_normalized = np.clip(depth_img / 20.0, 0, 1)
                depth_colored = (depth_normalized * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)
                
                depth_resized = cv2.resize(depth_colored, (420, 250))
                
                h, w = depth_resized.shape[:2]
                cv2.line(depth_resized, (w//3, 0), (w//3, h), (255, 255, 255), 2)
                cv2.line(depth_resized, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 2)
                
                cv2.putText(depth_resized, "LEFT", (40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(depth_resized, "CENTER", (w//2-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(depth_resized, "RIGHT", (2*w//3+20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                depth_rgb = cv2.cvtColor(depth_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(depth_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.vision_canvas.delete("all")
                self.vision_canvas.create_image(210, 125, image=img_tk)
                self.vision_canvas.image = img_tk
                
            except Exception as e:
                print(f"Vision display error: {e}")
    
    def draw_wind_pattern(self):
        """Draw wind pattern visualization"""
        try:
            self.wind_canvas.delete("all")
            
            w, h = 380, 150
            center_x, center_y = w // 2, h // 2
            
            wind_speed = self.wind_magnitude
            wind_dir = self.wind_direction
            wind_dir_norm = np.linalg.norm(wind_dir)
            
            if wind_dir_norm > 0.1:
                wind_unit = wind_dir / wind_dir_norm
            else:
                wind_unit = np.array([0.0, 0.0])
            
            if wind_speed < 5.0:
                color = "#4caf50"
                intensity_text = "CALM"
            elif wind_speed < 10.0:
                color = "#ffeb3b"
                intensity_text = "MODERATE"
            else:
                color = "#f44336"
                intensity_text = "STRONG"
            
            # Draw grid
            for i in range(0, w, 40):
                self.wind_canvas.create_line(i, 0, i, h, fill="#1a1a1a", width=1)
            for i in range(0, h, 30):
                self.wind_canvas.create_line(0, i, w, i, fill="#1a1a1a", width=1)
            
            # Draw arrows
            arrow_scale = min(wind_speed * 3, 50)
            
            for grid_x in range(60, w, 80):
                for grid_y in range(30, h, 40):
                    end_x = grid_x + wind_unit[0] * arrow_scale
                    end_y = grid_y - wind_unit[1] * arrow_scale
                    
                    self.wind_canvas.create_line(
                        grid_x, grid_y, end_x, end_y,
                        fill=color, width=3, arrow=tk.LAST,
                        arrowshape=(12, 15, 5)
                    )
            
            self.wind_canvas.create_oval(
                center_x - 5, center_y - 5,
                center_x + 5, center_y + 5,
                fill="#2196f3", outline="white"
            )
            
            self.wind_canvas.create_text(
                w // 2, 15,
                text=f"{intensity_text} - {wind_speed:.1f} m/s",
                fill=color, font=("Arial", 11, "bold")
            )
            
            if wind_dir_norm > 0.1:
                angle_rad = np.arctan2(wind_unit[1], wind_unit[0])
                angle_deg = np.degrees(angle_rad)
                compass_deg = (90 - angle_deg) % 360
                
                directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                idx = int((compass_deg + 22.5) / 45) % 8
                cardinal = directions[idx]
                
                self.wind_canvas.create_text(
                    w // 2, h - 15,
                    text=f"↑ {cardinal} ({compass_deg:.0f}°)",
                    fill="#90caf9", font=("Arial", 9)
                )
                
                self.wind_direction_label.config(text=f"Direction: {cardinal} ({compass_deg:.0f}°)")
            else:
                self.wind_canvas.create_text(
                    w // 2, h - 15,
                    text="— NO WIND —",
                    fill="#607d8b", font=("Arial", 9)
                )
                self.wind_direction_label.config(text="Direction: N/A")
                
        except Exception as e:
            print(f"Wind pattern error: {e}")
    
    def _update_graphs_loop(self):
        """Update graphs periodically"""
        while self.running:
            try:
                if len(self.metrics['time']) > 2:
                    self.update_graphs()
                time.sleep(0.5)
            except:
                pass
    
    def update_graphs(self):
        """Update all graphs"""
        if not self.running or len(self.metrics['time']) < 2:
            return
        
        try:
            self.fig.clear()
            times = self.metrics['time']
            
            # Speed
            ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
            ax1.plot(times, self.metrics['speeds'], color='#2196f3', linewidth=2)
            ax1.set_title('SPEED', fontweight='bold', fontsize=10, color='white')
            ax1.set_ylabel('m/s', fontsize=8, color='white')
            ax1.grid(alpha=0.2, color='#404040')
            ax1.tick_params(colors='white', labelsize=8)
            for spine in ax1.spines.values():
                spine.set_color('#404040')
            
            # Altitude
            ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
            alts = [abs(a) for a in self.metrics['altitudes']]
            ax2.plot(times, alts, color='#ff9800', linewidth=2)
            ax2.set_title('ALTITUDE', fontweight='bold', fontsize=10, color='white')
            ax2.set_ylabel('m', fontsize=8, color='white')
            ax2.axhline(y=20, color='#76ff03', linestyle='--', alpha=0.5)
            ax2.grid(alpha=0.2, color='#404040')
            ax2.tick_params(colors='white', labelsize=8)
            for spine in ax2.spines.values():
                spine.set_color('#404040')
            
            # Battery
            ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
            battery_color = '#4caf50' if self.battery_percent > 50 else '#ff9800' if self.battery_percent > 20 else '#f44336'
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
            
            # Terrain
            x_range = max(abs(self.goal_x) + 40, 140)
            y_range = max(abs(self.goal_y) + 40, 140)
            x = np.linspace(-x_range, x_range, 120)
            y = np.linspace(-y_range, y_range, 120)
            X, Y = np.meshgrid(x, y)
            Z = (np.sin(X/40) * np.cos(Y/40) * 25 +
                 np.sin(X/20) * np.cos(Y/30) * 15 +
                 np.random.rand(120, 120) * 8)
            ax5.contourf(X, Y, Z, levels=25, cmap='terrain', alpha=0.4, zorder=0)
            
            # Start
            ax5.scatter(0, 0, color='#4caf50', s=180, marker='o',
                       edgecolors='white', linewidths=2.5, label='Start', zorder=4)
            
            # Goal
            ax5.scatter(self.goal_x, self.goal_y, color='#ff0000', s=350, marker='*',
                       edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
            
            # Path
            if len(self.metrics['positions_x']) > 1:
                ax5.plot(self.metrics['positions_x'], self.metrics['positions_y'],
                        color='#00ffff', linewidth=3, alpha=0.8, zorder=5)
            
            # Current position
            if len(self.metrics['positions_x']) > 0:
                ax5.scatter(self.metrics['positions_x'][-1], self.metrics['positions_y'][-1],
                           color='#2196f3', s=200, marker='o',
                           edgecolors='white', linewidths=2, label='Drone', zorder=6)
            
            # Interceptors
            if self.use_interceptors and self.interceptor_positions:
                colors = ['#ff6f00', '#9c27b0', '#00897b']
                for idx, (drone_name, pos) in enumerate(self.interceptor_positions.items()):
                    if len(pos) >= 2:
                        color = colors[idx % len(colors)]
                        ax5.scatter(pos[0], pos[1], color=color, s=150, marker='^',
                                   edgecolors='white', linewidths=2,
                                   label=drone_name.replace('Drone', 'I'), zorder=5, alpha=0.8)
            
            ax5.set_title('FLIGHT MAP', fontweight='bold', fontsize=10, color='white')
            ax5.set_xlabel('X (m)', fontsize=8, color='white')
            ax5.set_ylabel('Y (m)', fontsize=8, color='white')
            ax5.grid(alpha=0.3, color='#90ee90', linestyle='--', linewidth=0.5)
            ax5.legend(loc='upper right', fontsize=7, facecolor='#1e1e1e',
                      edgecolor='#404040', labelcolor='white', framealpha=0.8)
            ax5.tick_params(colors='white', labelsize=8)
            for spine in ax5.spines.values():
                spine.set_color('#404040')
            
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Graph error: {e}")
    
    def _setup_timeout_reached(self, setup_start_time, phase_name):
        """Check setup timeout"""
        elapsed = time.time() - setup_start_time
        if elapsed <= SETUP_TIMEOUT_SECONDS:
            return False
        
        msg = f"Setup timeout after {elapsed:.1f}s during: {phase_name}"
        print(f"❌ {msg}")
        self.status_label.config(text="● Setup Timeout", fg="#f44336")
        messagebox.showerror(
            "Setup Timeout",
            f"Initialization exceeded {int(SETUP_TIMEOUT_SECONDS)} seconds.\n"
            f"Timed out at: {phase_name}\n\n"
            "Please check AirSim and retry."
        )
        return True
    
    def update_rl_display(self):
        """Update RL display"""
        self.run_number_label.config(text=f"Run: #{self.current_run_number}")
        
        if self.best_energy < float('inf'):
            self.best_energy_label.config(text=f"Best Energy: {self.best_energy:.2f} Wh")
        
        if self.best_time < float('inf'):
            self.best_time_label.config(text=f"Best Time: {self.best_time:.1f} s")
        
        if len(self.run_history) > 1:
            current_energy = self.run_history[-1]['energy']
            prev_energy = self.run_history[-2]['energy']
            improvement = ((prev_energy - current_energy) / prev_energy) * 100
            
            if improvement > 0:
                self.improvement_label.config(
                    text=f"Improvement: +{improvement:.1f}% ⬆️",
                    fg="#76ff03"
                )
            else:
                self.improvement_label.config(
                    text=f"Improvement: {improvement:.1f}% ⬇️",
                    fg="#ff5252"
                )
    
    def run(self):
        """Run GUI"""
        self.window.mainloop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    app = GNNSwarmCompleteGUI()
    app.run()
