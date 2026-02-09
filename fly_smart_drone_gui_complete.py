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
        title.pack(pady=15)
        
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
    
    def start_flight(self):
        """Start the flight in a separate thread"""
        self.status_label.config(text="● Initializing Systems...", fg="orange")
        self.window.update()
        threading.Thread(target=self._initialize_and_fly, daemon=True).start()
    
    def _initialize_and_fly(self):
        """Initialize AirSim, load model, and start flight"""
        try:
            # Load PPO Agent
            print("🧠 Loading PPO Agent with Multi-Head Attention...")
            self.agent = PPO_Agent(state_dim=7, action_dim=3, lr=0, gamma=0, K_epochs=0)
            
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                if 'actor_state_dict' in checkpoint:
                    self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                else:
                    self.agent.actor.load_state_dict(checkpoint)
                device_name = "CUDA (RTX 3050)" if torch.cuda.is_available() else "CPU"
                print(f"✓ Brain loaded successfully on {device_name}")
                self.status_label.config(text=f"● Brain Loaded ({device_name})", fg="#4caf50")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.status_label.config(text="● Model Load Failed!", fg="red")
                messagebox.showerror("Error", f"Failed to load model:\n{e}")
                return
            
            self.agent.actor.eval()
            
            # Connect to AirSim
            print("🔌 Connecting to AirSim...")
            self.status_label.config(text="● Connecting to AirSim...", fg="orange")
            self.window.update()
            
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            print("🛫 Taking off...")
            self.status_label.config(text="● Taking Off...", fg="orange")
            self.window.update()
            
            self.client.takeoffAsync().join()
            time.sleep(2)
            
            self.client.moveToZAsync(-3.0, 2.0).join()
            time.sleep(1)
            
            # Store home position
            home_state = self.client.getMultirotorState()
            self.home_position = np.array([
                home_state.kinematics_estimated.position.x_val,
                home_state.kinematics_estimated.position.y_val
            ])
            print(f"🏠 Home: ({self.home_position[0]:.1f}, {self.home_position[1]:.1f})")
            
            # Reset metrics
            self.metrics = {
                'time': [],
                'speeds': [], 
                'altitudes': [], 
                'battery': [],
                'energy': [],
                'positions_x': [],
                'positions_y': []
            }
            self.battery_percent = 100.0
            self.total_energy_consumed = 0.0
            self.start_time = time.time()
            
            self.running = True
            self.flight_active = True
            
            self.status_label.config(text="● Flying to Goal!", fg="#4caf50")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.capture_btn.config(state="normal")
            
            # Start telemetry updates
            self.update_telemetry_loop()
            
            # Start flight loop
            self._ppo_flight_loop()
            
        except Exception as e:
            self.status_label.config(text="● Error!", fg="red")
            print(f"Initialization error: {e}")
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()
    
    def _ppo_flight_loop(self):
        """Main flight loop using PPO agent"""
        dt = 0.1
        step = 0
        prev_altitude = -3.0
        
        print(f"\n🎯 PPO Energy-Efficient Flight Started")
        print(f"Goal: ({GOAL_POS[0]:.0f}, {GOAL_POS[1]:.0f})")
        print("=" * 60)
        
        try:
            while self.flight_active and self.running:
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
                    
                    # Wind estimation (simplified - in real scenario, calculate from drift)
                    wind = velocity * 0.1
                    
                    # Battery level (normalized 0-1)
                    battery_normalized = self.battery_percent / 100.0
                    
                    # Construct 7-value state vector
                    state_vec = np.concatenate([
                        dist_vector,      # 2 values: distance to goal in X, Y
                        velocity,         # 2 values: current velocity X, Y
                        wind,             # 2 values: wind/drift estimation X, Y
                        [battery_normalized]  # 1 value: battery level
                    ]).astype(np.float32)
                    
                except Exception as e:
                    print(f"State error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Check if goal reached
                if distance < 5.0:
                    print(f"\n🏆 GOAL REACHED! Distance: {distance:.2f}m")
                    self.status_label.config(text="● Goal Reached! ✓", fg="#4caf50")
                    
                    # Final summary
                    if len(self.metrics['speeds']) > 0:
                        print(f"\n{'='*60}")
                        print(f"📊 FLIGHT SUMMARY:")
                        print(f"{'='*60}")
                        print(f"   Total Steps: {len(self.metrics['speeds'])}")
                        print(f"   Avg Speed: {np.mean(self.metrics['speeds']):.2f} m/s")
                        print(f"   Max Speed: {np.max(self.metrics['speeds']):.2f} m/s")
                        print(f"   Energy Used: {self.total_energy_consumed:.2f} Wh")
                        print(f"   Battery Remaining: {self.battery_percent:.1f}%")
                        print(f"   Flight Time: {time.time() - self.start_time:.1f} seconds")
                        print(f"{'='*60}")
                        self.update_graphs()
                    
                    self.flight_active = False
                    break
                
                # Check battery
                if self.battery_percent <= 0:
                    print("\n🪫 Battery Depleted! Emergency landing required.")
                    self.status_label.config(text="● Battery Empty!", fg="red")
                    break
                
                # AI decides action
                action = self.agent.select_action(state_vec)
                
                # Scale actions to real-world units
                target_vx = np.clip(action[0] * 10.0, -5, 5)  # Max 5 m/s for safety
                target_vy = np.clip(action[1] * 10.0, -5, 5)
                yaw_rate = action[2]  # Yaw control (not used in this simple version)
                
                # Execute move
                try:
                    self.client.moveByVelocityAsync(
                        float(target_vx),
                        float(target_vy),
                        0,  # Keep altitude stable
                        duration=0.15
                    )
                except Exception as e:
                    print(f"Move error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Calculate metrics
                speed = np.linalg.norm([target_vx, target_vy])
                altitude_change = (pos.z_val - prev_altitude) / dt
                prev_altitude = pos.z_val
                
                # Energy consumption (Physics: Power = P_hover * (1 + 0.005 * speed^2))
                power_watts = P_HOVER * (1 + 0.005 * speed**2)
                energy_step_wh = power_watts * (dt / 3600.0)  # Convert to Wh
                self.total_energy_consumed += energy_step_wh
                
                # Update battery percentage
                self.battery_percent = max(0.0, 100.0 - (self.total_energy_consumed / BATTERY_CAPACITY_WH * 100))
                
                # Track metrics
                elapsed_time = time.time() - self.start_time
                self.metrics['time'].append(elapsed_time)
                self.metrics['speeds'].append(speed)
                self.metrics['altitudes'].append(pos.z_val)
                self.metrics['battery'].append(self.battery_percent)
                self.metrics['energy'].append(self.total_energy_consumed)
                self.metrics['positions_x'].append(pos.x_val)
                self.metrics['positions_y'].append(pos.y_val)
                
                # Update graphs every 50 steps
                if step % 50 == 0:
                    self.update_graphs()
                
                # Console logging
                if step % 20 == 0:
                    print(f"Step {step:4d} | 🔋 {self.battery_percent:5.1f}% | "
                          f"📏 {distance:6.1f}m | 💨 {speed:4.1f} m/s | "
                          f"⚡ {self.total_energy_consumed:5.2f} Wh")
                
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
            print("\n✓ Flight loop ended")
    
    def update_telemetry_loop(self):
        """Update telemetry displays in GUI"""
        if not self.running:
            return
        
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            # Update position
            self.drone_position = np.array([pos.x_val, pos.y_val])
            self.position_label.config(text=f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f})")
            
            # Update speed
            speed = np.linalg.norm([vel.x_val, vel.y_val])
            self.speed_label.config(text=f"Speed: {speed:.2f} m/s")
            
            # Update altitude
            altitude = abs(pos.z_val)
            self.altitude_label.config(text=f"Altitude: {altitude:.1f} m")
            
            # Update distance to goal
            distance = np.linalg.norm(GOAL_POS - self.drone_position)
            self.distance_label.config(text=f"Distance to Goal: {distance:.1f} m")
            
            # Update battery display
            battery_color = "#4caf50" if self.battery_percent > 50 else "#ff9800" if self.battery_percent > 20 else "#f44336"
            self.battery_label.config(text=f"{self.battery_percent:.1f}%", fg=battery_color)
            self.energy_label.config(text=f"Energy: {self.total_energy_consumed:.2f} Wh")
            
        except:
            pass
        
        if self.running:
            self.window.after(100, self.update_telemetry_loop)
    
    def stop_flight(self):
        """Stop flight and land"""
        self.flight_active = False
        self.running = False
        
        # Final graph update
        if len(self.metrics['time']) > 0:
            self.update_graphs()
        
        if self.client:
            try:
                print("\n🛬 Landing...")
                self.status_label.config(text="● Landing...", fg="orange")
                self.client.moveByVelocityAsync(0, 0, 0, 1).join()
                self.client.landAsync().join()
                time.sleep(2)
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
                print("✓ Landed safely")
                self.status_label.config(text="● Landed", fg="gray")
            except Exception as e:
                print(f"Landing error: {e}")
        
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
    
    def run(self):
        """Run the GUI main loop"""
        self.window.mainloop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("⚡ ENERGY-SAVING DRONE CONTROL CENTER")
    print("="*60)
    print("Initializing GUI...")
    
    app = EnergyDroneControlCenter()
    app.run()
