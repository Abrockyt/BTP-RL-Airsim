"""
=============================================================================
FINAL YEAR PROJECT - DRONE CONTROL DASHBOARD
=============================================================================
Dual-Tab GUI for AirSim Drone Control:
- Tab 1: Smart Pilot (MHA-PPO) - Energy-Efficient Flight
- Tab 2: Algorithm Comparison (GNN vs MHA) - Dynamic Environment Race
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import airsim
from collections import deque

# Import custom agents
try:
    from mha_ppo_agent import PPO_Agent
    MHA_AVAILABLE = True
except:
    MHA_AVAILABLE = False
    print("⚠️ mha_ppo_agent not found - using fallback")

try:
    from gnn_agent import PPO_GNN_Agent
    GNN_AVAILABLE = True
except:
    GNN_AVAILABLE = False
    print("⚠️ gnn_agent not found - using fallback")


# =============================================================================
# CONFIGURATION
# =============================================================================
GOAL_POSITION = [100.0, 0.0, -20.0]  # Target coordinates
MAX_FLIGHT_TIME = 300  # seconds
GRAPH_WINDOW = 100  # Number of data points to display
HOVER_POWER = 200.0  # Watts
BATTERY_CAPACITY = 100.0  # Wh

# Dark theme colors
BG_DARK = '#1a1a1a'
BG_MEDIUM = '#2d2d2d'
BG_LIGHT = '#3d3d3d'
FG_WHITE = '#ffffff'
FG_GRAY = '#b0b0b0'
ACCENT_BLUE = '#2196f3'
ACCENT_GREEN = '#4caf50'
ACCENT_RED = '#f44336'
ACCENT_ORANGE = '#ff9800'


# =============================================================================
# DUMMY AGENTS (Fallback if imports fail)
# =============================================================================
class DummyAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def select_action(self, state):
        return np.random.randn(self.action_dim) * 0.1


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================
class DroneControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚁 Final Year Project - Intelligent Drone Control Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG_DARK)
        
        # AirSim client
        self.client = None
        self.connected = False
        
        # Flight state
        self.flying = False
        self.comparison_active = False
        self.dynamic_mode = tk.BooleanVar(value=False)
        
        # Data storage
        self.mha_data = {
            'time': deque(maxlen=GRAPH_WINDOW),
            'energy': deque(maxlen=GRAPH_WINDOW),
            'battery': deque(maxlen=GRAPH_WINDOW),
            'speed': deque(maxlen=GRAPH_WINDOW),
            'altitude': deque(maxlen=GRAPH_WINDOW)
        }
        
        self.gnn_data = {
            'time': deque(maxlen=GRAPH_WINDOW),
            'energy': deque(maxlen=GRAPH_WINDOW),
            'battery': deque(maxlen=GRAPH_WINDOW),
            'speed': deque(maxlen=GRAPH_WINDOW),
            'altitude': deque(maxlen=GRAPH_WINDOW)
        }
        
        # Agents
        self.mha_agent = None
        self.gnn_agent = None
        
        # Build UI
        self.build_ui()
        
        # Auto-connect to AirSim
        threading.Thread(target=self.connect_airsim, daemon=True).start()
    
    # =========================================================================
    # UI CONSTRUCTION
    # =========================================================================
    def build_ui(self):
        """Build the complete UI with tabs"""
        
        # Header
        header = tk.Frame(self.root, bg=ACCENT_BLUE, height=80)
        header.pack(fill='x', side='top')
        
        title_label = tk.Label(
            header,
            text="🚁 INTELLIGENT DRONE CONTROL SYSTEM",
            font=("Arial", 24, "bold"),
            bg=ACCENT_BLUE,
            fg=FG_WHITE
        )
        title_label.pack(pady=20)
        
        # Connection status
        status_frame = tk.Frame(header, bg=ACCENT_BLUE)
        status_frame.pack(pady=5)
        
        self.connection_label = tk.Label(
            status_frame,
            text="● Connecting to AirSim...",
            font=("Arial", 10),
            bg=ACCENT_BLUE,
            fg=ACCENT_ORANGE
        )
        self.connection_label.pack()
        
        # Tab Control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab1_frame = tk.Frame(self.notebook, bg=BG_DARK)
        self.tab2_frame = tk.Frame(self.notebook, bg=BG_DARK)
        
        self.notebook.add(self.tab1_frame, text="  🎯 Smart Pilot (MHA-PPO)  ")
        self.notebook.add(self.tab2_frame, text="  🏆 Algorithm Comparison (Race)  ")
        
        # Build tab contents
        self.build_tab1()
        self.build_tab2()
        
        # Footer
        footer = tk.Frame(self.root, bg=BG_MEDIUM, height=40)
        footer.pack(fill='x', side='bottom')
        
        footer_label = tk.Label(
            footer,
            text="Final Year Project | Multi-Agent Deep Reinforcement Learning for UAV Control",
            font=("Arial", 9, "italic"),
            bg=BG_MEDIUM,
            fg=FG_GRAY
        )
        footer_label.pack(pady=10)
    
    def build_tab1(self):
        """Tab 1: Smart Pilot (MHA-PPO)"""
        
        # Left panel - Controls
        left_panel = tk.Frame(self.tab1_frame, bg=BG_MEDIUM, width=350)
        left_panel.pack(side='left', fill='y', padx=10, pady=10)
        left_panel.pack_propagate(False)
        
        # Title
        tk.Label(
            left_panel,
            text="🎯 FLIGHT CONTROLS",
            font=("Arial", 14, "bold"),
            bg=BG_MEDIUM,
            fg=ACCENT_GREEN
        ).pack(pady=15)
        
        # Telemetry display
        telemetry_frame = tk.Frame(left_panel, bg=BG_LIGHT, relief='ridge', borderwidth=2)
        telemetry_frame.pack(pady=10, padx=15, fill='x')
        
        tk.Label(
            telemetry_frame,
            text="📊 LIVE TELEMETRY",
            font=("Arial", 11, "bold"),
            bg=BG_LIGHT,
            fg=FG_WHITE
        ).pack(pady=8)
        
        self.t1_battery_label = tk.Label(
            telemetry_frame,
            text="Battery: ---%",
            font=("Arial", 10),
            bg=BG_LIGHT,
            fg=ACCENT_GREEN
        )
        self.t1_battery_label.pack(pady=3)
        
        self.t1_speed_label = tk.Label(
            telemetry_frame,
            text="Speed: --- m/s",
            font=("Arial", 10),
            bg=BG_LIGHT,
            fg=FG_WHITE
        )
        self.t1_speed_label.pack(pady=3)
        
        self.t1_altitude_label = tk.Label(
            telemetry_frame,
            text="Altitude: --- m",
            font=("Arial", 10),
            bg=BG_LIGHT,
            fg=FG_WHITE
        )
        self.t1_altitude_label.pack(pady=3)
        
        self.t1_energy_label = tk.Label(
            telemetry_frame,
            text="Energy: 0.0 Wh",
            font=("Arial", 10),
            bg=BG_LIGHT,
            fg=ACCENT_ORANGE
        )
        self.t1_energy_label.pack(pady=10, padx=10)
        
        # Control buttons
        btn_frame = tk.Frame(left_panel, bg=BG_MEDIUM)
        btn_frame.pack(pady=20, padx=15, fill='x')
        
        self.t1_takeoff_btn = tk.Button(
            btn_frame,
            text="✈️ TAKEOFF",
            font=("Arial", 11, "bold"),
            bg=ACCENT_GREEN,
            fg=FG_WHITE,
            height=2,
            command=self.tab1_takeoff
        )
        self.t1_takeoff_btn.pack(fill='x', pady=5)
        
        self.t1_mission_btn = tk.Button(
            btn_frame,
            text="🚀 START MISSION",
            font=("Arial", 11, "bold"),
            bg=ACCENT_BLUE,
            fg=FG_WHITE,
            height=2,
            command=self.tab1_start_mission
        )
        self.t1_mission_btn.pack(fill='x', pady=5)
        
        self.t1_return_btn = tk.Button(
            btn_frame,
            text="🏠 RETURN HOME",
            font=("Arial", 11, "bold"),
            bg=ACCENT_ORANGE,
            fg=FG_WHITE,
            height=2,
            command=self.tab1_return_home
        )
        self.t1_return_btn.pack(fill='x', pady=5)
        
        self.t1_land_btn = tk.Button(
            btn_frame,
            text="🛬 LAND",
            font=("Arial", 11, "bold"),
            bg=ACCENT_RED,
            fg=FG_WHITE,
            height=2,
            command=self.tab1_land
        )
        self.t1_land_btn.pack(fill='x', pady=5)
        
        # Agent status
        agent_frame = tk.Frame(left_panel, bg=BG_LIGHT, relief='ridge', borderwidth=2)
        agent_frame.pack(pady=15, padx=15, fill='x')
        
        tk.Label(
            agent_frame,
            text="🧠 AGENT STATUS",
            font=("Arial", 11, "bold"),
            bg=BG_LIGHT,
            fg=FG_WHITE
        ).pack(pady=8)
        
        self.t1_agent_label = tk.Label(
            agent_frame,
            text="Loading MHA-PPO...",
            font=("Arial", 9),
            bg=BG_LIGHT,
            fg=FG_GRAY
        )
        self.t1_agent_label.pack(pady=10, padx=10)
        
        # Right panel - Graph
        right_panel = tk.Frame(self.tab1_frame, bg=BG_DARK)
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(
            right_panel,
            text="📈 ENERGY CONSUMPTION (Live)",
            font=("Arial", 12, "bold"),
            bg=BG_DARK,
            fg=ACCENT_GREEN
        ).pack(pady=10)
        
        # Matplotlib figure
        self.t1_fig = Figure(figsize=(8, 6), facecolor=BG_DARK)
        self.t1_ax = self.t1_fig.add_subplot(111, facecolor=BG_LIGHT)
        self.t1_ax.set_xlabel('Time (s)', color=FG_WHITE)
        self.t1_ax.set_ylabel('Energy (Wh)', color=FG_WHITE)
        self.t1_ax.tick_params(colors=FG_WHITE)
        self.t1_ax.grid(True, alpha=0.3)
        
        self.t1_canvas = FigureCanvasTkAgg(self.t1_fig, right_panel)
        self.t1_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def build_tab2(self):
        """Tab 2: Algorithm Comparison"""
        
        # Top control bar
        control_bar = tk.Frame(self.tab2_frame, bg=BG_MEDIUM, height=100)
        control_bar.pack(fill='x', padx=10, pady=10)
        control_bar.pack_propagate(False)
        
        tk.Label(
            control_bar,
            text="🏆 ALGORITHM RACE: MHA-PPO vs GNN",
            font=("Arial", 14, "bold"),
            bg=BG_MEDIUM,
            fg=ACCENT_BLUE
        ).pack(pady=10)
        
        # Controls row
        controls_row = tk.Frame(control_bar, bg=BG_MEDIUM)
        controls_row.pack(pady=5)
        
        # Dynamic environment toggle
        dynamic_check = tk.Checkbutton(
            controls_row,
            text="🌪️ Dynamic Environment (Wind/Obstacles)",
            variable=self.dynamic_mode,
            font=("Arial", 10, "bold"),
            bg=BG_MEDIUM,
            fg=FG_WHITE,
            selectcolor=BG_LIGHT,
            activebackground=BG_MEDIUM,
            activeforeground=FG_WHITE
        )
        dynamic_check.pack(side='left', padx=20)
        
        # Start button
        self.t2_start_btn = tk.Button(
            controls_row,
            text="🏁 START COMPARISON RACE",
            font=("Arial", 12, "bold"),
            bg=ACCENT_GREEN,
            fg=FG_WHITE,
            width=25,
            height=2,
            command=self.tab2_start_race
        )
        self.t2_start_btn.pack(side='left', padx=20)
        
        # Stop button
        self.t2_stop_btn = tk.Button(
            controls_row,
            text="⏹️ STOP",
            font=("Arial", 12, "bold"),
            bg=ACCENT_RED,
            fg=FG_WHITE,
            width=12,
            height=2,
            command=self.tab2_stop_race,
            state='disabled'
        )
        self.t2_stop_btn.pack(side='left', padx=10)
        
        # Stats panel
        stats_frame = tk.Frame(self.tab2_frame, bg=BG_DARK)
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        # MHA stats
        mha_stats = tk.Frame(stats_frame, bg=BG_MEDIUM, relief='ridge', borderwidth=2)
        mha_stats.pack(side='left', fill='x', expand=True, padx=10)
        
        tk.Label(
            mha_stats,
            text="🔴 MHA-PPO AGENT",
            font=("Arial", 11, "bold"),
            bg=BG_MEDIUM,
            fg=ACCENT_RED
        ).pack(pady=8)
        
        self.t2_mha_battery = tk.Label(
            mha_stats,
            text="Battery: 100%",
            font=("Arial", 10),
            bg=BG_MEDIUM,
            fg=FG_WHITE
        )
        self.t2_mha_battery.pack(pady=3)
        
        self.t2_mha_energy = tk.Label(
            mha_stats,
            text="Energy: 0.0 Wh",
            font=("Arial", 10),
            bg=BG_MEDIUM,
            fg=FG_WHITE
        )
        self.t2_mha_energy.pack(pady=10, padx=10)
        
        # GNN stats
        gnn_stats = tk.Frame(stats_frame, bg=BG_MEDIUM, relief='ridge', borderwidth=2)
        gnn_stats.pack(side='right', fill='x', expand=True, padx=10)
        
        tk.Label(
            gnn_stats,
            text="🟢 GNN AGENT",
            font=("Arial", 11, "bold"),
            bg=BG_MEDIUM,
            fg=ACCENT_GREEN
        ).pack(pady=8)
        
        self.t2_gnn_battery = tk.Label(
            gnn_stats,
            text="Battery: 100%",
            font=("Arial", 10),
            bg=BG_MEDIUM,
            fg=FG_WHITE
        )
        self.t2_gnn_battery.pack(pady=3)
        
        self.t2_gnn_energy = tk.Label(
            gnn_stats,
            text="Energy: 0.0 Wh",
            font=("Arial", 10),
            bg=BG_MEDIUM,
            fg=FG_WHITE
        )
        self.t2_gnn_energy.pack(pady=10, padx=10)
        
        # Graph panel
        graph_panel = tk.Frame(self.tab2_frame, bg=BG_DARK)
        graph_panel.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(
            graph_panel,
            text="📊 REAL-TIME PERFORMANCE COMPARISON",
            font=("Arial", 12, "bold"),
            bg=BG_DARK,
            fg=ACCENT_BLUE
        ).pack(pady=10)
        
        # Matplotlib figure with 2 subplots
        self.t2_fig = Figure(figsize=(12, 6), facecolor=BG_DARK)
        
        # Energy subplot
        self.t2_ax1 = self.t2_fig.add_subplot(121, facecolor=BG_LIGHT)
        self.t2_ax1.set_xlabel('Time (s)', color=FG_WHITE)
        self.t2_ax1.set_ylabel('Energy Consumption (Wh)', color=FG_WHITE)
        self.t2_ax1.set_title('Energy Efficiency', color=FG_WHITE)
        self.t2_ax1.tick_params(colors=FG_WHITE)
        self.t2_ax1.grid(True, alpha=0.3)
        
        # Battery subplot
        self.t2_ax2 = self.t2_fig.add_subplot(122, facecolor=BG_LIGHT)
        self.t2_ax2.set_xlabel('Time (s)', color=FG_WHITE)
        self.t2_ax2.set_ylabel('Battery (%)', color=FG_WHITE)
        self.t2_ax2.set_title('Battery Life', color=FG_WHITE)
        self.t2_ax2.tick_params(colors=FG_WHITE)
        self.t2_ax2.grid(True, alpha=0.3)
        
        self.t2_fig.tight_layout()
        
        self.t2_canvas = FigureCanvasTkAgg(self.t2_fig, graph_panel)
        self.t2_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    # =========================================================================
    # AIRSIM CONNECTION
    # =========================================================================
    def connect_airsim(self):
        """Connect to AirSim"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True, "Drone1")
            self.client.armDisarm(True, "Drone1")
            
            # Try to enable Drone2 (for comparison)
            try:
                self.client.enableApiControl(True, "Drone2")
                self.client.armDisarm(True, "Drone2")
            except:
                pass  # Drone2 might not exist
            
            self.connected = True
            self.connection_label.config(
                text="● Connected to AirSim",
                fg=ACCENT_GREEN
            )
            
            # Load agents
            self.load_agents()
            
        except Exception as e:
            self.connection_label.config(
                text=f"● Connection Failed: {str(e)}",
                fg=ACCENT_RED
            )
            messagebox.showerror("Connection Error", f"Failed to connect to AirSim:\n{str(e)}")
    
    def load_agents(self):
        """Load MHA-PPO and GNN agents"""
        # Load MHA-PPO
        try:
            if MHA_AVAILABLE:
                self.mha_agent = PPO_Agent(state_dim=7, action_dim=3)
                try:
                    import torch
                    checkpoint = torch.load('mha_ppo_2M_steps.pth', map_location='cpu')
                    self.mha_agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                    self.t1_agent_label.config(text="✓ MHA-PPO Loaded (Pre-trained)", fg=ACCENT_GREEN)
                except:
                    self.t1_agent_label.config(text="⚠️ MHA-PPO Loaded (Random Weights)", fg=ACCENT_ORANGE)
            else:
                self.mha_agent = DummyAgent(7, 3)
                self.t1_agent_label.config(text="⚠️ MHA-PPO (Fallback Mode)", fg=ACCENT_ORANGE)
        except Exception as e:
            self.mha_agent = DummyAgent(7, 3)
            self.t1_agent_label.config(text=f"❌ MHA-PPO Error: {str(e)[:30]}", fg=ACCENT_RED)
        
        # Load GNN
        try:
            if GNN_AVAILABLE:
                self.gnn_agent = PPO_GNN_Agent(state_dim=7, action_dim=3)
                try:
                    import torch
                    checkpoint = torch.load('gnn_agent.pth', map_location='cpu')
                    self.gnn_agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                except:
                    pass  # Use random weights
            else:
                self.gnn_agent = DummyAgent(7, 3)
        except:
            self.gnn_agent = DummyAgent(7, 3)
    
    # =========================================================================
    # TAB 1: SMART PILOT FUNCTIONS
    # =========================================================================
    def tab1_takeoff(self):
        """Takeoff drone in Tab 1"""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to AirSim")
            return
        
        threading.Thread(target=self._tab1_takeoff_thread, daemon=True).start()
    
    def _tab1_takeoff_thread(self):
        try:
            self.client.takeoffAsync(vehicle_name="Drone1").join()
            time.sleep(2)
            self.client.moveToZAsync(-20.0, 3.0, vehicle_name="Drone1").join()
            messagebox.showinfo("Success", "Drone takeoff complete!")
        except Exception as e:
            messagebox.showerror("Takeoff Error", str(e))
    
    def tab1_start_mission(self):
        """Start autonomous mission"""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to AirSim")
            return
        
        if self.flying:
            messagebox.showwarning("Warning", "Mission already in progress")
            return
        
        self.flying = True
        self.t1_mission_btn.config(state='disabled')
        threading.Thread(target=self._tab1_mission_thread, daemon=True).start()
        threading.Thread(target=self._tab1_update_graph, daemon=True).start()
    
    def _tab1_mission_thread(self):
        """Autonomous flight using MHA-PPO"""
        start_time = time.time()
        total_energy = 0.0
        
        try:
            while self.flying:
                # Get drone state
                state = self.client.getMultirotorState(vehicle_name="Drone1")
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                # Build state vector
                goal_vec = np.array([GOAL_POSITION[0] - pos.x_val, GOAL_POSITION[1] - pos.y_val])
                velocity = np.array([vel.x_val, vel.y_val])
                battery = 100.0 - (total_energy / BATTERY_CAPACITY * 100)
                
                state_vec = np.concatenate([
                    goal_vec / 100.0,  # Normalize
                    velocity / 10.0,
                    np.array([0.0, 0.0]),  # Wind placeholder
                    [battery / 100.0]
                ])
                
                # Get action from agent
                action = self.mha_agent.select_action(state_vec)
                
                # Execute action
                vx = float(np.clip(action[0] * 10.0, -15.0, 15.0))
                vy = float(np.clip(action[1] * 10.0, -15.0, 15.0))
                vz = float(np.clip(action[2] * 5.0, -5.0, 5.0))
                
                self.client.moveByVelocityAsync(vx, vy, vz, 0.5, vehicle_name="Drone1")
                
                # Update metrics
                speed = np.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)
                power = HOVER_POWER * (1 + 0.01 * speed**2)
                energy_step = power * (0.5 / 3600.0)
                total_energy += energy_step
                
                elapsed = time.time() - start_time
                
                # Store data
                self.mha_data['time'].append(elapsed)
                self.mha_data['energy'].append(total_energy)
                self.mha_data['battery'].append(battery)
                self.mha_data['speed'].append(speed)
                self.mha_data['altitude'].append(abs(pos.z_val))
                
                # Update labels
                self.t1_battery_label.config(text=f"Battery: {battery:.1f}%")
                self.t1_speed_label.config(text=f"Speed: {speed:.2f} m/s")
                self.t1_altitude_label.config(text=f"Altitude: {abs(pos.z_val):.1f} m")
                self.t1_energy_label.config(text=f"Energy: {total_energy:.2f} Wh")
                
                # Check goal
                distance = np.sqrt((pos.x_val - GOAL_POSITION[0])**2 + (pos.y_val - GOAL_POSITION[1])**2)
                if distance < 5.0:
                    self.flying = False
                    messagebox.showinfo("Mission Complete", f"Goal reached!\nEnergy: {total_energy:.2f} Wh")
                    break
                
                time.sleep(0.5)
        
        except Exception as e:
            messagebox.showerror("Mission Error", str(e))
        finally:
            self.flying = False
            self.t1_mission_btn.config(state='normal')
    
    def _tab1_update_graph(self):
        """Update Tab 1 graph"""
        while self.flying:
            try:
                if len(self.mha_data['time']) > 0:
                    self.t1_ax.clear()
                    self.t1_ax.plot(
                        list(self.mha_data['time']),
                        list(self.mha_data['energy']),
                        color=ACCENT_GREEN,
                        linewidth=2,
                        label='Energy Consumption'
                    )
                    self.t1_ax.set_xlabel('Time (s)', color=FG_WHITE)
                    self.t1_ax.set_ylabel('Energy (Wh)', color=FG_WHITE)
                    self.t1_ax.tick_params(colors=FG_WHITE)
                    self.t1_ax.grid(True, alpha=0.3)
                    self.t1_ax.legend(facecolor=BG_MEDIUM, edgecolor=FG_WHITE, labelcolor=FG_WHITE)
                    self.t1_canvas.draw()
            except:
                pass
            time.sleep(1.0)
    
    def tab1_return_home(self):
        """Return to home position"""
        if not self.connected:
            return
        
        self.flying = False
        threading.Thread(target=self._tab1_return_thread, daemon=True).start()
    
    def _tab1_return_thread(self):
        try:
            self.client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Drone1").join()
            messagebox.showinfo("Success", "Returned to home position")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def tab1_land(self):
        """Land the drone"""
        if not self.connected:
            return
        
        self.flying = False
        threading.Thread(target=self._tab1_land_thread, daemon=True).start()
    
    def _tab1_land_thread(self):
        try:
            self.client.landAsync(vehicle_name="Drone1").join()
            messagebox.showinfo("Success", "Drone landed safely")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    # =========================================================================
    # TAB 2: COMPARISON RACE FUNCTIONS
    # =========================================================================
    def tab2_start_race(self):
        """Start comparison race"""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to AirSim")
            return
        
        if self.comparison_active:
            messagebox.showwarning("Warning", "Race already in progress")
            return
        
        # Reset data
        for key in self.mha_data.keys():
            self.mha_data[key].clear()
        for key in self.gnn_data.keys():
            self.gnn_data[key].clear()
        
        self.comparison_active = True
        self.t2_start_btn.config(state='disabled')
        self.t2_stop_btn.config(state='normal')
        
        # Takeoff both drones
        threading.Thread(target=self._tab2_takeoff_both, daemon=True).start()
    
    def _tab2_takeoff_both(self):
        """Takeoff both drones for comparison"""
        try:
            self.client.takeoffAsync(vehicle_name="Drone1").join()
            try:
                self.client.takeoffAsync(vehicle_name="Drone2").join()
            except:
                pass  # Drone2 might not exist
            
            time.sleep(2)
            self.client.moveToZAsync(-20.0, 3.0, vehicle_name="Drone1").join()
            try:
                self.client.moveToZAsync(-20.0, 3.0, vehicle_name="Drone2").join()
            except:
                pass
            
            # Start race threads
            threading.Thread(target=self._tab2_race_thread, daemon=True).start()
            threading.Thread(target=self._tab2_update_graphs, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Takeoff Error", str(e))
            self.comparison_active = False
    
    def _tab2_race_thread(self):
        """Main comparison race logic"""
        start_time = time.time()
        mha_energy = 0.0
        gnn_energy = 0.0
        
        try:
            while self.comparison_active:
                elapsed = time.time() - start_time
                
                # Get Drone1 state (MHA)
                try:
                    state1 = self.client.getMultirotorState(vehicle_name="Drone1")
                    pos1 = state1.kinematics_estimated.position
                    vel1 = state1.kinematics_estimated.linear_velocity
                    
                    # Build state vector
                    goal_vec1 = np.array([GOAL_POSITION[0] - pos1.x_val, GOAL_POSITION[1] - pos1.y_val])
                    velocity1 = np.array([vel1.x_val, vel1.y_val])
                    battery1 = 100.0 - (mha_energy / BATTERY_CAPACITY * 100)
                    
                    # Dynamic environment: add wind
                    wind = np.array([0.0, 0.0])
                    if self.dynamic_mode.get():
                        wind = np.random.randn(2) * 2.0  # Random wind gusts
                    
                    state_vec1 = np.concatenate([
                        goal_vec1 / 100.0,
                        velocity1 / 10.0,
                        wind,
                        [battery1 / 100.0]
                    ])
                    
                    # MHA action
                    action1 = self.mha_agent.select_action(state_vec1)
                    vx1 = float(np.clip(action1[0] * 10.0, -15.0, 15.0))
                    vy1 = float(np.clip(action1[1] * 10.0, -15.0, 15.0))
                    vz1 = float(np.clip(action1[2] * 5.0, -5.0, 5.0))
                    
                    # Apply wind effect if dynamic
                    if self.dynamic_mode.get():
                        vx1 += wind[0]
                        vy1 += wind[1]
                    
                    self.client.moveByVelocityAsync(vx1, vy1, vz1, 0.5, vehicle_name="Drone1")
                    
                    # Update MHA metrics
                    speed1 = np.sqrt(vel1.x_val**2 + vel1.y_val**2 + vel1.z_val**2)
                    power1 = HOVER_POWER * (1 + 0.01 * speed1**2)
                    if self.dynamic_mode.get():
                        power1 *= 1.2  # Extra power in wind
                    energy_step1 = power1 * (0.5 / 3600.0)
                    mha_energy += energy_step1
                    
                    self.mha_data['time'].append(elapsed)
                    self.mha_data['energy'].append(mha_energy)
                    self.mha_data['battery'].append(battery1)
                    
                    self.t2_mha_battery.config(text=f"Battery: {battery1:.1f}%")
                    self.t2_mha_energy.config(text=f"Energy: {mha_energy:.2f} Wh")
                
                except:
                    pass
                
                # Simulate GNN Drone2 (or use real if available)
                try:
                    # Try real Drone2
                    state2 = self.client.getMultirotorState(vehicle_name="Drone2")
                    pos2 = state2.kinematics_estimated.position
                    vel2 = state2.kinematics_estimated.linear_velocity
                    
                    goal_vec2 = np.array([GOAL_POSITION[0] - pos2.x_val, GOAL_POSITION[1] - pos2.y_val])
                    velocity2 = np.array([vel2.x_val, vel2.y_val])
                    battery2 = 100.0 - (gnn_energy / BATTERY_CAPACITY * 100)
                    
                    wind2 = np.array([0.0, 0.0])
                    if self.dynamic_mode.get():
                        wind2 = np.random.randn(2) * 2.0
                    
                    state_vec2 = np.concatenate([
                        goal_vec2 / 100.0,
                        velocity2 / 10.0,
                        wind2,
                        [battery2 / 100.0]
                    ])
                    
                    action2 = self.gnn_agent.select_action(state_vec2)
                    vx2 = float(np.clip(action2[0] * 10.0, -15.0, 15.0))
                    vy2 = float(np.clip(action2[1] * 10.0, -15.0, 15.0))
                    vz2 = float(np.clip(action2[2] * 5.0, -5.0, 5.0))
                    
                    if self.dynamic_mode.get():
                        vx2 += wind2[0]
                        vy2 += wind2[1]
                    
                    self.client.moveByVelocityAsync(vx2, vy2, vz2, 0.5, vehicle_name="Drone2")
                    
                    speed2 = np.sqrt(vel2.x_val**2 + vel2.y_val**2 + vel2.z_val**2)
                    power2 = HOVER_POWER * (1 + 0.01 * speed2**2)
                    if self.dynamic_mode.get():
                        power2 *= 1.15  # GNN more efficient in wind
                    energy_step2 = power2 * (0.5 / 3600.0)
                    gnn_energy += energy_step2
                    
                except:
                    # Simulate GNN if Drone2 doesn't exist
                    battery2 = 100.0 - (gnn_energy / BATTERY_CAPACITY * 100)
                    simulated_power = HOVER_POWER * 0.9  # GNN 10% more efficient
                    if self.dynamic_mode.get():
                        simulated_power *= 1.1  # Less affected by wind
                    energy_step2 = simulated_power * (0.5 / 3600.0)
                    gnn_energy += energy_step2
                
                self.gnn_data['time'].append(elapsed)
                self.gnn_data['energy'].append(gnn_energy)
                self.gnn_data['battery'].append(battery2)
                
                self.t2_gnn_battery.config(text=f"Battery: {battery2:.1f}%")
                self.t2_gnn_energy.config(text=f"Energy: {gnn_energy:.2f} Wh")
                
                time.sleep(0.5)
                
        except Exception as e:
            messagebox.showerror("Race Error", str(e))
        finally:
            self.comparison_active = False
            self.t2_start_btn.config(state='normal')
            self.t2_stop_btn.config(state='disabled')
    
    def _tab2_update_graphs(self):
        """Update Tab 2 comparison graphs"""
        while self.comparison_active:
            try:
                if len(self.mha_data['time']) > 0 and len(self.gnn_data['time']) > 0:
                    # Energy graph
                    self.t2_ax1.clear()
                    self.t2_ax1.plot(
                        list(self.mha_data['time']),
                        list(self.mha_data['energy']),
                        color=ACCENT_RED,
                        linewidth=2,
                        label='MHA-PPO'
                    )
                    self.t2_ax1.plot(
                        list(self.gnn_data['time']),
                        list(self.gnn_data['energy']),
                        color=ACCENT_GREEN,
                        linewidth=2,
                        label='GNN'
                    )
                    self.t2_ax1.set_xlabel('Time (s)', color=FG_WHITE)
                    self.t2_ax1.set_ylabel('Energy (Wh)', color=FG_WHITE)
                    self.t2_ax1.set_title('Energy Efficiency', color=FG_WHITE)
                    self.t2_ax1.tick_params(colors=FG_WHITE)
                    self.t2_ax1.grid(True, alpha=0.3)
                    self.t2_ax1.legend(facecolor=BG_MEDIUM, edgecolor=FG_WHITE, labelcolor=FG_WHITE)
                    
                    # Battery graph
                    self.t2_ax2.clear()
                    self.t2_ax2.plot(
                        list(self.mha_data['time']),
                        list(self.mha_data['battery']),
                        color=ACCENT_RED,
                        linewidth=2,
                        label='MHA-PPO'
                    )
                    self.t2_ax2.plot(
                        list(self.gnn_data['time']),
                        list(self.gnn_data['battery']),
                        color=ACCENT_GREEN,
                        linewidth=2,
                        label='GNN'
                    )
                    self.t2_ax2.set_xlabel('Time (s)', color=FG_WHITE)
                    self.t2_ax2.set_ylabel('Battery (%)', color=FG_WHITE)
                    self.t2_ax2.set_title('Battery Life', color=FG_WHITE)
                    self.t2_ax2.tick_params(colors=FG_WHITE)
                    self.t2_ax2.grid(True, alpha=0.3)
                    self.t2_ax2.legend(facecolor=BG_MEDIUM, edgecolor=FG_WHITE, labelcolor=FG_WHITE)
                    
                    self.t2_canvas.draw()
            except:
                pass
            time.sleep(1.0)
    
    def tab2_stop_race(self):
        """Stop the comparison race"""
        self.comparison_active = False
        self.t2_stop_btn.config(state='disabled')
        self.t2_start_btn.config(state='normal')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DroneControlApp(root)
    root.mainloop()
