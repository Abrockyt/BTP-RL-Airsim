"""
Energy-Saving Drone Control Center
Complete standalone script with PPO Agent, GUI, and live performance tracking
"""

import airsim
import torch
import torch.nn as nn
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
import threading
import os
from datetime import datetime
from tkinter import messagebox

# Matplotlib for graphs
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================================================================
# BRAIN ARCHITECTURE (Multi-Head Attention Actor)
# =============================================================================

class MHA_Actor(nn.Module):
    """Multi-Head Attention Actor Network for PPO"""
    def __init__(self, state_dim, action_dim, num_heads=4, hidden_dim=128):
        super(MHA_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        # Embed state
        x = self.embedding(state)
        
        # Add sequence dimension for attention (batch, seq_len, features)
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # Squeeze sequence dimension
        x = attn_output.squeeze(1)
        
        # Output action
        x = self.tanh(self.fc1(x))
        action = self.tanh(self.fc2(x))
        
        return action


class PPO_Agent:
    """PPO Agent with MHA Actor"""
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4):
        self.gamma = gamma
        self.K_epochs = K_epochs
        
        self.actor = MHA_Actor(state_dim, action_dim)
        if lr > 0:
            self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
    
    def select_action(self, state):
        """Select action using the trained actor"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()
        
        # Handle both single and batch outputs
        if len(action.shape) > 1:
            action = action[0]
        
        return action


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "energy_saving_brain.pth"
GOAL_POS = np.array([100.0, 100.0])  # Target goal position
P_HOVER = 200.0  # Base hover power (Watts)
BATTERY_CAPACITY_WH = 100.0  # Battery capacity in Watt-hours

# =============================================================================
# GUI CLASS
# =============================================================================

class EnergyDroneControlCenter:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("⚡ Energy-Saving Drone Control Center")
        self.window.geometry("1600x900")
        self.window.configure(bg='#1e1e1e')
        
        # AirSim client
        self.client = None
        self.agent = None
        
        # State
        self.drone_position = np.array([0.0, 0.0])
        self.home_position = None
        self.running = False
        self.flight_active = False
        
        # Performance metrics
        self.metrics = {
            'time': [],
            'speeds': [], 
            'altitudes': [], 
            'battery': [],
            'energy': [],
            'positions_x': [],
            'positions_y': []
        }
        self.start_time = None
        
        # Battery simulation
        self.battery_percent = 100.0
        self.total_energy_consumed = 0.0
        
        # Graph components
        self.fig = None
        self.canvas = None
        
        # Create graphs folder
        os.makedirs('performance_graphs', exist_ok=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.window, bg='#0d47a1', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="⚡ ENERGY-SAVING DRONE CONTROL CENTER", 
                        font=("Arial", 24, "bold"), fg="white", bg='#0d47a1')
        title.pack(pady=20)
        
        subtitle = tk.Label(header_frame, text="PPO Multi-Head Attention Agent | Real-Time Battery Physics | Live Performance Tracking", 
                           font=("Arial", 11), fg="#90caf9", bg='#0d47a1')
        subtitle.pack()
        
        # Main content frame
        content_frame = tk.Frame(self.window, bg='#1e1e1e')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls and Info
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief='raised', borderwidth=2)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Status section
        status_frame = tk.Frame(left_panel, bg='#2d2d2d')
        status_frame.pack(pady=20, padx=20)
        
        tk.Label(status_frame, text="🎯 FLIGHT STATUS", 
                font=("Arial", 12, "bold"), fg="#4caf50", bg='#2d2d2d').pack()
        
        self.status_label = tk.Label(status_frame, text="● Ready to Deploy", 
                                     font=("Arial", 11), fg="#90caf9", bg='#2d2d2d')
        self.status_label.pack(pady=10)
        
        # Battery display
        battery_frame = tk.Frame(left_panel, bg='#2d2d2d')
        battery_frame.pack(pady=10, padx=20)
        
        tk.Label(battery_frame, text="🔋 POWER SYSTEM", 
                font=("Arial", 12, "bold"), fg="#ff9800", bg='#2d2d2d').pack()
        
        self.battery_label = tk.Label(battery_frame, text="100.0%", 
                                      font=("Arial", 32, "bold"), fg="#4caf50", bg='#2d2d2d')
        self.battery_label.pack(pady=5)
        
        self.energy_label = tk.Label(battery_frame, text="Energy: 0.0 Wh", 
                                     font=("Arial", 10), fg="#90caf9", bg='#2d2d2d')
        self.energy_label.pack()
        
        # Info display
        info_frame = tk.Frame(left_panel, bg='#2d2d2d')
        info_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(info_frame, text="📊 TELEMETRY", 
                font=("Arial", 12, "bold"), fg="#2196f3", bg='#2d2d2d').pack()
        
        self.position_label = tk.Label(info_frame, text="Position: (0.0, 0.0)", 
                                      font=("Arial", 10), fg="white", bg='#2d2d2d')
        self.position_label.pack(pady=5, anchor='w')
        
        self.speed_label = tk.Label(info_frame, text="Speed: 0.0 m/s", 
                                   font=("Arial", 10), fg="white", bg='#2d2d2d')
        self.speed_label.pack(pady=5, anchor='w')
        
        self.altitude_label = tk.Label(info_frame, text="Altitude: 0.0 m", 
                                      font=("Arial", 10), fg="white", bg='#2d2d2d')
        self.altitude_label.pack(pady=5, anchor='w')
        
        self.distance_label = tk.Label(info_frame, text="Distance to Goal: -- m", 
                                      font=("Arial", 10), fg="white", bg='#2d2d2d')
        self.distance_label.pack(pady=5, anchor='w')
        
        # Goal info
        goal_frame = tk.Frame(left_panel, bg='#2d2d2d')
        goal_frame.pack(pady=10, padx=20)
        
        tk.Label(goal_frame, text="🎯 TARGET", 
                font=("Arial", 12, "bold"), fg="#f44336", bg='#2d2d2d').pack()
        
        tk.Label(goal_frame, text=f"({GOAL_POS[0]:.0f}, {GOAL_POS[1]:.0f})", 
                font=("Arial", 14, "bold"), fg="#ff5252", bg='#2d2d2d').pack(pady=5)
        
        # Control buttons
        button_frame = tk.Frame(left_panel, bg='#2d2d2d')
        button_frame.pack(pady=20, padx=20)
        
        self.start_btn = tk.Button(button_frame, text="🚀 START FLIGHT", 
                                   command=self.start_flight, 
                                   bg="#4caf50", fg="white", font=("Arial", 13, "bold"),
                                   width=18, height=2, relief='raised', borderwidth=3)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏹ STOP & LAND", 
                                 command=self.stop_flight, 
                                 bg="#f44336", fg="white", font=("Arial", 13, "bold"),
                                 width=18, height=2, state="disabled", relief='raised', borderwidth=3)
        self.stop_btn.pack(pady=5)
        
        self.capture_btn = tk.Button(button_frame, text="📷 CAPTURE GRAPH", 
                                     command=self.capture_graph,
                                     bg="#9c27b0", fg="white", font=("Arial", 11, "bold"),
                                     width=18, height=1, state="disabled", relief='raised', borderwidth=2)
        self.capture_btn.pack(pady=5)
        
        # Right panel - Graphs
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', relief='raised', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        graph_title = tk.Label(right_panel, text="📈 LIVE PERFORMANCE ANALYTICS", 
                              font=("Arial", 14, "bold"), fg="#4caf50", bg='#2d2d2d')
        graph_title.pack(pady=10)
        
        # Create matplotlib figure with 5 subplots (4 metrics + 1 map)
        self.fig = Figure(figsize=(12, 8), facecolor='#2d2d2d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        self.init_graphs()
        
        # Footer
        footer = tk.Label(self.window, text="AI-Powered Energy Optimization | RTX 3050 Accelerated", 
                         font=("Arial", 9), fg="#757575", bg='#1e1e1e')
        footer.pack(pady=5)
    
    def init_graphs(self):
        """Initialize empty graphs"""
        self.fig.clear()
        
        # Create subplots: 2x3 grid
        ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
        ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
        ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
        ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
        ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')  # Flight map (larger)
        
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.text(0.5, 0.5, 'Waiting for flight data...', 
                   ha='center', va='center', fontsize=12, color='#757575',
                   transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#404040')
            ax.spines['top'].set_color('#404040')
            ax.spines['left'].set_color('#404040')
            ax.spines['right'].set_color('#404040')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_graphs(self):
        """Update all performance graphs"""
        if not self.running or len(self.metrics['time']) < 2:
            return
        
        try:
            self.fig.clear()
            
            times = self.metrics['time']
            
            # Subplot 1: Speed
            ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
            ax1.plot(times, self.metrics['speeds'], color='#2196f3', linewidth=2, label='Speed')
            ax1.set_title('SPEED (m/s)', fontweight='bold', fontsize=11, color='white')
            ax1.set_ylabel('m/s', fontsize=9, color='white')
            ax1.grid(alpha=0.2, color='#404040')
            ax1.axhline(y=5.0, color='#ff5252', linestyle='--', alpha=0.5, linewidth=1.5, label='Safety Limit')
            ax1.tick_params(colors='white')
            ax1.spines['bottom'].set_color('#404040')
            ax1.spines['top'].set_color('#404040')
            ax1.spines['left'].set_color('#404040')
            ax1.spines['right'].set_color('#404040')
            
            # Subplot 2: Altitude
            ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
            alts = [abs(a) for a in self.metrics['altitudes']]
            ax2.plot(times, alts, color='#ff9800', linewidth=2)
            ax2.set_title('ALTITUDE (m)', fontweight='bold', fontsize=11, color='white')
            ax2.set_ylabel('meters', fontsize=9, color='white')
            ax2.grid(alpha=0.2, color='#404040')
            ax2.tick_params(colors='white')
            ax2.spines['bottom'].set_color('#404040')
            ax2.spines['top'].set_color('#404040')
            ax2.spines['left'].set_color('#404040')
            ax2.spines['right'].set_color('#404040')
            
            # Subplot 3: Battery
            ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
            battery_color = '#4caf50' if self.battery_percent > 50 else '#ff9800' if self.battery_percent > 20 else '#f44336'
            ax3.plot(times, self.metrics['battery'], color=battery_color, linewidth=2.5)
            ax3.set_title('BATTERY LEVEL (%)', fontweight='bold', fontsize=11, color='white')
            ax3.set_ylabel('%', fontsize=9, color='white')
            ax3.grid(alpha=0.2, color='#404040')
            ax3.axhline(y=20, color='#f44336', linestyle='--', alpha=0.7, linewidth=2, label='Critical')
            ax3.fill_between(times, 0, 20, color='#f44336', alpha=0.15)
            ax3.set_ylim([0, 105])
            ax3.tick_params(colors='white')
            ax3.spines['bottom'].set_color('#404040')
            ax3.spines['top'].set_color('#404040')
            ax3.spines['left'].set_color('#404040')
            ax3.spines['right'].set_color('#404040')
            
            # Subplot 4: Energy Consumption
            ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
            ax4.plot(times, self.metrics['energy'], color='#e91e63', linewidth=2)
            ax4.set_title('ENERGY CONSUMED (Wh)', fontweight='bold', fontsize=11, color='white')
            ax4.set_ylabel('Watt-hours', fontsize=9, color='white')
            ax4.set_xlabel('Time (s)', fontsize=9, color='white')
            ax4.grid(alpha=0.2, color='#404040')
            ax4.tick_params(colors='white')
            ax4.spines['bottom'].set_color('#404040')
            ax4.spines['top'].set_color('#404040')
            ax4.spines['left'].set_color('#404040')
            ax4.spines['right'].set_color('#404040')
            
            # Subplot 5: Flight Map
            ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')
            
            # Plot trajectory
            if len(self.metrics['positions_x']) > 1:
                ax5.plot(self.metrics['positions_x'], self.metrics['positions_y'], 
                        color='#64b5f6', linewidth=2, alpha=0.6, label='Path')
            
            # Current position (blue dot)
            if len(self.metrics['positions_x']) > 0:
                ax5.scatter(self.metrics['positions_x'][-1], self.metrics['positions_y'][-1], 
                           color='#2196f3', s=200, marker='o', edgecolors='white', linewidths=2, 
                           label='Drone', zorder=5)
            
            # Home (green)
            if self.home_position is not None:
                ax5.scatter(self.home_position[0], self.home_position[1], 
                           color='#4caf50', s=150, marker='s', edgecolors='white', linewidths=2,
                           label='Home', zorder=5)
            
            # Goal (red)
            ax5.scatter(GOAL_POS[0], GOAL_POS[1], 
                       color='#f44336', s=200, marker='*', edgecolors='white', linewidths=2,
                       label='Goal', zorder=5)
            
            ax5.set_title('FLIGHT MAP', fontweight='bold', fontsize=11, color='white')
            ax5.set_xlabel('X Position (m)', fontsize=9, color='white')
            ax5.set_ylabel('Y Position (m)', fontsize=9, color='white')
            ax5.grid(alpha=0.2, color='#404040')
            ax5.legend(loc='upper left', fontsize=8, facecolor='#2d2d2d', edgecolor='#404040', labelcolor='white')
            ax5.tick_params(colors='white')
            ax5.spines['bottom'].set_color('#404040')
            ax5.spines['top'].set_color('#404040')
            ax5.spines['left'].set_color('#404040')
            ax5.spines['right'].set_color('#404040')
            ax5.set_aspect('equal', adjustable='box')
            
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Graph update error: {e}")
    
    def capture_graph(self):
        """Save current graph as PNG"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_graphs/energy_flight_{timestamp}.png'
            self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#2d2d2d')
            print(f"📷 Graph saved: {filename}")
            messagebox.showinfo("Saved!", f"Graph captured:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save: {e}")
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚁 Smart Drone - Energy-Efficient PPO Flight")
        self.window.geometry("1500x950")
        
        # AirSim client
        self.client = None
        self.agent = None
        
        # State
        self.drone_position = (0, 0)
        self.home_position = None
        self.goal_position = GOAL_POS
        self.running = False
        self.flight_active = False
        self.navigating_to_goal = False
        
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
        
        # Create graphs folder
        os.makedirs('performance_graphs', exist_ok=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = tk.Label(self.window, text="🚁 Smart Drone - Energy-Efficient PPO Flight", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        subtitle = tk.Label(self.window, text="PPO Agent + Real-time Energy Tracking + Live Performance Graphs", 
                           font=("Arial", 10), fg="gray")
        subtitle.pack()
        
        # Battery display
        self.battery_label = tk.Label(self.window, text="🔋 Battery: 100.0% | Energy: 0.0 Wh", 
                                     font=("Arial", 11, "bold"), fg="green")
        self.battery_label.pack(pady=5)
        
        # Status
        self.status_label = tk.Label(self.window, text="Status: Ready to fly", 
                                     font=("Arial", 12, "bold"), fg="blue")
        self.status_label.pack(pady=5)
        
        # Position and speed labels
        info_frame = tk.Frame(self.window)
        info_frame.pack(pady=5)
        
        self.position_label = tk.Label(info_frame, text="Position: (0, 0)", 
                                      font=("Arial", 10))
        self.position_label.pack(side="left", padx=20)
        
        self.speed_label = tk.Label(info_frame, text="Speed: 0.0 m/s", 
                                   font=("Arial", 10))
        self.speed_label.pack(side="left", padx=20)
        
        self.goal_label = tk.Label(info_frame, text=f"Goal: ({GOAL_POS[0]:.0f}, {GOAL_POS[1]:.0f})", 
                                  font=("Arial", 10), fg="red")
        self.goal_label.pack(side="left", padx=20)
        
        # Map and Graph Container
        map_graph_container = tk.Frame(self.window)
        map_graph_container.pack(pady=10, fill="both", expand=True)
        
        # Left: Map
        map_frame = tk.Frame(map_graph_container, relief="groove", borderwidth=2)
        map_frame.pack(side="left", padx=10, fill="both", expand=True)
        
        map_label = tk.Label(map_frame, text="🗺️ Flight Map", font=("Arial", 11, "bold"))
        map_label.pack()
        
        self.canvas = tk.Canvas(map_frame, width=600, height=600, bg='white')
        self.canvas.pack()
        
        # Right: Live Graphs
        graph_frame = tk.Frame(map_graph_container, relief="groove", borderwidth=2)
        graph_frame.pack(side="right", padx=10, fill="both", expand=True)
        
        graph_header = tk.Frame(graph_frame)
        graph_header.pack(pady=5)
        
        graph_title = tk.Label(graph_header, text="📊 Live Performance Graphs", 
                              font=("Arial", 11, "bold"))
        graph_title.pack(side="left", padx=10)
        
        self.capture_btn = tk.Button(graph_header, text="📷 Capture", 
                                     command=self.capture_graph,
                                     bg="#9C27B0", fg="white", font=("Arial", 9, "bold"),
                                     state="disabled")
        self.capture_btn.pack(side="right", padx=10)
        
        # Matplotlib figure
        self.graph_figure = Figure(figsize=(7, 6), dpi=80)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.init_live_graph()
        
        # Draw grid
        self.draw_grid()
        
        # Buttons
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="🚀 Start Flight", 
                                   command=self.start_flight, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                   width=15, height=2)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop Flight", 
                                 command=self.stop_flight, 
                                 bg="#F44336", fg="white", font=("Arial", 12, "bold"),
                                 width=15, height=2, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Features info
        features_frame = tk.Frame(self.window, bg="#f0f0f0", relief="groove", borderwidth=2)
        features_frame.pack(pady=10, padx=20, fill="x")
        
        features_title = tk.Label(features_frame, text="✨ Smart Features:", 
                                 font=("Arial", 10, "bold"), bg="#f0f0f0")
        features_title.pack(anchor="w", padx=10, pady=5)
        
        features = [
            "🧠 PPO Agent: Multi-Head Attention-based reinforcement learning for optimal flight",
            "⚡ Energy Optimization: AI minimizes power consumption while reaching goal",
            "🔋 Real Battery Drain: Physics-based battery simulation (200W base + speed penalty)",
            "📊 Live Graphs: Speed, Altitude, Battery %, Energy tracked in real-time",
            f"🎯 Goal: Navigate to ({GOAL_POS[0]:.0f}, {GOAL_POS[1]:.0f}) with minimal energy use",
            "📷 Capture: Save graph screenshots to performance_graphs folder"
        ]
        
        for feat in features:
            lbl = tk.Label(features_frame, text=feat, font=("Arial", 9), bg="#f0f0f0", anchor="w")
            lbl.pack(anchor="w", padx=20, pady=2)
    
    def init_live_graph(self):
        """Initialize empty graph"""
        self.graph_figure.clear()
        ax = self.graph_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Start flight to see live metrics...', 
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.graph_canvas.draw()
    
    def update_live_graph(self):
        """Update live performance graph"""
        if not self.running or not self.start_time:
            return
        
        try:
            self.graph_figure.clear()
            
            # Create 4 subplots
            ax1 = self.graph_figure.add_subplot(411)
            ax2 = self.graph_figure.add_subplot(412)
            ax3 = self.graph_figure.add_subplot(413)
            ax4 = self.graph_figure.add_subplot(414)
            
            # Plot 1: Speed
            if len(self.metrics['speeds']) > 0:
                ax1.plot(self.metrics['speeds'], color='#2196F3', linewidth=2)
                ax1.set_title('Speed (m/s)', fontweight='bold', fontsize=10)
                ax1.set_ylabel('m/s', fontsize=9)
                ax1.grid(alpha=0.3)
                ax1.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Max Safe')
            
            # Plot 2: Altitude
            if len(self.metrics['altitudes']) > 0:
                alts = [abs(a) for a in self.metrics['altitudes']]
                ax2.plot(alts, color='#FF9800', linewidth=2)
                ax2.set_title('Altitude (m)', fontweight='bold', fontsize=10)
                ax2.set_ylabel('m', fontsize=9)
                ax2.grid(alpha=0.3)
            
            # Plot 3: Battery %
            if len(self.metrics['timestamps']) > 0:
                battery_history = [100.0 - (e / self.battery_capacity_wh * 100) for e in self.metrics['energy']]
                ax3.plot(battery_history, color='#4CAF50', linewidth=2)
                ax3.set_title('Battery Level (%)', fontweight='bold', fontsize=10)
                ax3.set_ylabel('%', fontsize=9)
                ax3.grid(alpha=0.3)
                ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax3.fill_between(range(len(battery_history)), 0, 20, color='red', alpha=0.1)
                ax3.set_ylim([0, 105])
            
            # Plot 4: Energy Consumption
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
        """Capture and save current graph"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_graphs/capture_{timestamp}.png'
            self.graph_figure.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📷 Graph saved: {filename}")
            messagebox.showinfo("Saved!", f"Graph captured:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save: {e}")
    
    def draw_grid(self):
        """Draw grid on map"""
        for i in range(-100, 101, 20):
            x = self.world_to_canvas(i, 0)[0]
            self.canvas.create_line(x, 0, x, 600, fill='lightgray', width=1)
            y = self.world_to_canvas(0, i)[1]
            self.canvas.create_line(0, y, 600, y, fill='lightgray', width=1)
        
        # Draw goal marker
        gx, gy = self.world_to_canvas(GOAL_POS[0], GOAL_POS[1])
        self.canvas.create_oval(gx-10, gy-10, gx+10, gy+10, 
                               fill='red', outline='darkred', width=3, tags="goal")
        self.canvas.create_text(gx, gy-20, text="🎯 GOAL", fill="red", font=("Arial", 10, "bold"))
    
    def world_to_canvas(self, x, y):
        """Convert world coordinates to canvas coordinates"""
        scale = 3
        canvas_x = 300 + x * scale
        canvas_y = 300 - y * scale
        return (canvas_x, canvas_y)
    
    def calculate_energy_consumption(self, speed, altitude_change, dt):
        """Calculate realistic energy consumption based on flight parameters"""
        # Base power consumption (hovering)
        base_power = P_HOVER  # 200W for hovering
        
        # Speed-based power (quadratic relationship)
        speed_power = P_HOVER * 0.005 * (speed ** 2)
        
        # Altitude change power
        if altitude_change < 0:  # Climbing
            climb_power = abs(altitude_change) * 50.0
        else:
            climb_power = abs(altitude_change) * 5.0
        
        # Total power in Watts
        total_power = base_power + speed_power + climb_power
        
        # Convert to Wh
        energy_wh = total_power * (dt / 3600.0)
        
        return energy_wh
    
    def start_flight(self):
        self.status_label.config(text="Status: Initializing...", fg="orange")
        self.window.update()
        threading.Thread(target=self._initialize_and_fly, daemon=True).start()
    
    def _initialize_and_fly(self):
        try:
            # Load PPO Agent
            print("🧠 Loading PPO Agent...")
            self.agent = PPO_Agent(state_dim=7, action_dim=3, lr=0, gamma=0, K_epochs=0)
            
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                if 'actor_state_dict' in checkpoint:
                    self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                else:
                    self.agent.actor.load_state_dict(checkpoint)
                print("✓ Brain loaded successfully")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.status_label.config(text=f"Error: Model load failed", fg="red")
                return
            
            self.agent.actor.eval()
            
            # Connect to AirSim
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
            
            # Store home
            home_state = self.client.getMultirotorState()
            self.home_position = (home_state.kinematics_estimated.position.x_val,
                                 home_state.kinematics_estimated.position.y_val)
            print(f"🏠 Home: ({self.home_position[0]:.1f}, {self.home_position[1]:.1f})")
            
            # Draw home marker
            hc = self.world_to_canvas(self.home_position[0], self.home_position[1])
            self.canvas.create_oval(hc[0]-8, hc[1]-8, hc[0]+8, hc[1]+8,
                                   fill='green', outline='darkgreen', width=3, tags="home")
            
            # Reset metrics and battery
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
            self.navigating_to_goal = True
            self.status_label.config(text="Status: Flying to Goal ✓", fg="green")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.capture_btn.config(state="normal")
            
            self.update_position_loop()
            self._ppo_flight_loop()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)[:30]}", fg="red")
            print(f"Init error: {e}")
            import traceback
            traceback.print_exc()
    
    def _ppo_flight_loop(self):
        """Main flight loop using PPO agent"""
        dt = 0.1
        step = 0
        prev_altitude = -3.0
        
        print(f"\n🎯 PPO Flight Started - Target: ({GOAL_POS[0]:.0f}, {GOAL_POS[1]:.0f})")
        
        try:
            while self.flight_active and self.running and self.navigating_to_goal:
                loop_start = time.time()
                
                # Get current state
                try:
                    state_obj = self.client.getMultirotorState()
                    pos = state_obj.kinematics_estimated.position
                    vel = state_obj.kinematics_estimated.linear_velocity
                    
                    current_pos = np.array([pos.x_val, pos.y_val])
                    dist_vector = GOAL_POS - current_pos
                    distance = np.linalg.norm(dist_vector)
                    
                    velocity = np.array([vel.x_val, vel.y_val])
                    
                    # Wind estimation (simplified)
                    wind = velocity * 0.1
                    
                    # Battery level (normalized 0-1)
                    battery_normalized = self.battery_percent / 100.0
                    
                    # Construct state vector
                    state_vec = np.concatenate([
                        dist_vector,
                        velocity,
                        wind,
                        [battery_normalized]
                    ]).astype(np.float32)
                    
                except Exception as e:
                    print(f"State error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Check if goal reached
                if distance < 5.0:
                    print(f"🏆 GOAL REACHED! Distance: {distance:.2f}m")
                    self.status_label.config(text="Status: Goal Reached ✓", fg="green")
                    
                    # Final summary
                    if len(self.metrics['speeds']) > 0:
                        print(f"\n📊 Flight Summary:")
                        print(f"   Steps: {len(self.metrics['speeds'])}")
                        print(f"   Avg Speed: {np.mean(self.metrics['speeds']):.2f} m/s")
                        print(f"   Energy Used: {self.total_energy_consumed:.2f} Wh")
                        print(f"   Battery: {self.battery_percent:.1f}%")
                        self.update_live_graph()
                    
                    self.navigating_to_goal = False
                    break
                
                # Check battery
                if self.battery_percent <= 0:
                    print("🪫 Battery Depleted!")
                    self.status_label.config(text="Status: Battery Empty", fg="red")
                    break
                
                # AI decides action
                action = self.agent.select_action(state_vec)
                
                # Scale actions
                target_vx = np.clip(action[0] * 10.0, -5, 5)
                target_vy = np.clip(action[1] * 10.0, -5, 5)
                yaw_rate = action[2]
                
                # Execute move
                try:
                    self.client.moveByVelocityAsync(
                        float(target_vx),
                        float(target_vy),
                        0,
                        duration=0.1
                    )
                except Exception as e:
                    print(f"Move error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Calculate metrics
                speed = np.linalg.norm([target_vx, target_vy])
                altitude_change = (pos.z_val - prev_altitude) / dt
                prev_altitude = pos.z_val
                
                # Energy consumption
                energy_step = self.calculate_energy_consumption(speed, altitude_change, dt)
                self.total_energy_consumed += energy_step
                self.battery_percent = max(0.0, 100.0 - (self.total_energy_consumed / self.battery_capacity_wh * 100))
                
                # Track metrics
                self.metrics['speeds'].append(speed)
                self.metrics['altitudes'].append(pos.z_val)
                self.metrics['energy'].append(self.total_energy_consumed)
                self.metrics['timestamps'].append(time.time() - self.start_time)
                
                # Update battery display
                battery_color = "green" if self.battery_percent > 50 else "orange" if self.battery_percent > 20 else "red"
                battery_emoji = "🔋" if self.battery_percent > 20 else "⚠️"
                self.battery_label.config(
                    text=f"{battery_emoji} Battery: {self.battery_percent:.1f}% | Energy: {self.total_energy_consumed:.2f} Wh",
                    fg=battery_color
                )
                
                # Update graphs every 50 steps
                if step % 50 == 0:
                    self.update_live_graph()
                
                # Logging
                if step % 20 == 0:
                    print(f"Step {step:4d} | 🔋 {self.battery_percent:.1f}% | 📏 {distance:.1f}m | 💨 {speed:.1f} m/s")
                
                step += 1
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except Exception as e:
            print(f"Flight error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.flight_active = False
    
    def update_position_loop(self):
        if not self.running:
            return
        
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            
            self.drone_position = (pos.x_val, pos.y_val)
            
            # Update position label
            self.position_label.config(text=f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f})")
            
            # Draw drone on canvas
            self.canvas.delete("drone")
            cx, cy = self.world_to_canvas(pos.x_val, pos.y_val)
            self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, 
                                   fill='blue', outline='darkblue', width=2, tags="drone")
            
        except:
            pass
        
        if self.running:
            self.window.after(100, self.update_position_loop)
    
    def stop_flight(self):
        self.flight_active = False
        self.running = False
        self.navigating_to_goal = False
        
        # Final graph update
        if len(self.metrics['speeds']) > 0:
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
