"""
Smart Drone Vision GUI - PPO + Computer Vision with Live Interface
Complete GUI system with obstacle detection visualization and performance tracking
"""

import airsim
import cv2
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
from PIL import Image, ImageTk

# Fix IOLoop conflict with GUI
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # nest_asyncio not required if not available

# Matplotlib for graphs
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================================================================
# BRAIN ARCHITECTURE
# =============================================================================

class MHA_Actor(nn.Module):
    """Multi-Head Attention Actor Network for PPO"""
    def __init__(self, state_dim, action_dim, num_heads=4, hidden_dim=128):
        super(MHA_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.embedding(state)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        
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
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()
        
        if len(action.shape) > 1:
            action = action[0]
        
        return action


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "energy_saving_brain.pth"
DEFAULT_GOAL_X = 100.0
DEFAULT_GOAL_Y = 100.0
P_HOVER = 200.0
BATTERY_CAPACITY_WH = 100.0

# =============================================================================
# GUI CLASS
# =============================================================================

class SmartVisionDroneGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("👁️ Smart Vision Drone - AI + Computer Vision")
        self.window.geometry("1850x1000")
        self.window.configure(bg='#1a1a1a')
        self.window.state('zoomed')  # Maximize window
        
        # AirSim client
        self.client = None
        self.agent = None
        
        # Goal position (editable)
        self.goal_x = DEFAULT_GOAL_X
        self.goal_y = DEFAULT_GOAL_Y
        
        # State
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
            'positions_y': [],
            'obstacles': []
        }
        self.start_time = None
        
        # Battery
        self.battery_percent = 100.0
        self.total_energy_consumed = 0.0
        
        # Vision data
        self.current_depth_image = None
        self.current_obstacle_type = "CLEAR"
        self.obstacle_distances = {'center': 100, 'left': 100, 'right': 100, 'top': 100}
        
        # Graph components
        self.fig = None
        self.canvas = None
        
        # Map background for satellite view
        self.map_background = None
        self.map_extent = [-150, 150, -150, 150]
        
        # Ground height tracking
        self.ground_height = 0.0
        
        os.makedirs('performance_graphs', exist_ok=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.window, bg='#0d47a1', height=70)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="👁️ SMART VISION DRONE CONTROL", 
                        font=("Arial", 22, "bold"), fg="white", bg='#0d47a1')
        title.pack(pady=12)
        
        subtitle = tk.Label(header_frame, text="PPO AI + Computer Vision Obstacle Avoidance | Real-Time Depth Sensing", 
                           font=("Arial", 10), fg="#90caf9", bg='#0d47a1')
        subtitle.pack()
        
        # Main container
        main_container = tk.Frame(self.window, bg='#1a1a1a')
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel container with scrollbar
        left_container = tk.Frame(main_container, bg='#2d2d2d', relief='raised', borderwidth=2, width=480)
        left_container.pack(side='left', fill='both', padx=(0, 5))
        left_container.pack_propagate(False)
        
        # Create canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_container, bg='#2d2d2d', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient='vertical', command=left_canvas.yview)
        
        # Create scrollable frame
        left_panel = tk.Frame(left_canvas, bg='#2d2d2d')
        
        # Configure canvas
        left_panel.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=left_panel, anchor='nw')
        left_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side='right', fill='y')
        left_canvas.pack(side='left', fill='both', expand=True)
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Vision Display
        vision_frame = tk.Frame(left_panel, bg='#2d2d2d')
        vision_frame.pack(pady=8, padx=10)
        
        tk.Label(vision_frame, text="👁️ DEPTH VISION", 
                font=("Arial", 11, "bold"), fg="#4caf50", bg='#2d2d2d').pack()
        
        self.vision_canvas = tk.Canvas(vision_frame, width=420, height=250, bg='black', highlightthickness=2, highlightbackground='#4caf50')
        self.vision_canvas.pack(pady=5)
        
        # Obstacle status
        self.obstacle_label = tk.Label(vision_frame, text="CLEAR PATH", 
                                       font=("Arial", 13, "bold"), fg="#4caf50", bg='#2d2d2d')
        self.obstacle_label.pack(pady=3)
        
        # Distance indicators
        dist_frame = tk.Frame(left_panel, bg='#2d2d2d')
        dist_frame.pack(pady=3, padx=10)
        
        tk.Label(dist_frame, text="📏 OBSTACLE DISTANCES", 
                font=("Arial", 10, "bold"), fg="#2196f3", bg='#2d2d2d').pack()
        
        self.dist_center_label = tk.Label(dist_frame, text="Center: --m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.dist_center_label.pack(anchor='w', padx=20)
        
        self.dist_left_label = tk.Label(dist_frame, text="Left: --m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.dist_left_label.pack(anchor='w', padx=20)
        
        self.dist_right_label = tk.Label(dist_frame, text="Right: --m", font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.dist_right_label.pack(anchor='w', padx=20)
        
        # Status section
        status_frame = tk.Frame(left_panel, bg='#2d2d2d')
        status_frame.pack(pady=5, padx=10)
        
        tk.Label(status_frame, text="🎯 STATUS", 
                font=("Arial", 11, "bold"), fg="#4caf50", bg='#2d2d2d').pack()
        
        self.status_label = tk.Label(status_frame, text="● Ready", 
                                     font=("Arial", 10), fg="#90caf9", bg='#2d2d2d')
        self.status_label.pack(pady=5)
        
        # Goal Position Control
        goal_frame = tk.Frame(left_panel, bg='#2d2d2d', relief='groove', borderwidth=2)
        goal_frame.pack(pady=8, padx=10, fill='x')
        
        tk.Label(goal_frame, text="🎯 GOAL POSITION", 
                font=("Arial", 11, "bold"), fg="#ff9800", bg='#2d2d2d').pack(pady=3)
        
        # Instruction
        tk.Label(goal_frame, text="💡 Click on map or enter coordinates", 
                font=("Arial", 8, "italic"), fg="#90caf9", bg='#2d2d2d').pack(pady=2)
        
        # Goal X
        x_frame = tk.Frame(goal_frame, bg='#2d2d2d')
        x_frame.pack(pady=3, padx=10, fill='x')
        tk.Label(x_frame, text="X:", font=("Arial", 9, "bold"), fg="white", bg='#2d2d2d', width=3).pack(side='left')
        self.goal_x_entry = tk.Entry(x_frame, font=("Arial", 10), width=10, bg='#1e1e1e', fg='white', 
                                     insertbackground='white', relief='solid', borderwidth=2)
        self.goal_x_entry.pack(side='left', padx=5)
        self.goal_x_entry.insert(0, str(DEFAULT_GOAL_X))
        tk.Label(x_frame, text="meters", font=("Arial", 8), fg="#90caf9", bg='#2d2d2d').pack(side='left', padx=3)
        
        # Goal Y
        y_frame = tk.Frame(goal_frame, bg='#2d2d2d')
        y_frame.pack(pady=3, padx=10, fill='x')
        tk.Label(y_frame, text="Y:", font=("Arial", 9, "bold"), fg="white", bg='#2d2d2d', width=3).pack(side='left')
        self.goal_y_entry = tk.Entry(y_frame, font=("Arial", 10), width=10, bg='#1e1e1e', fg='white', 
                                     insertbackground='white', relief='solid', borderwidth=2)
        self.goal_y_entry.pack(side='left', padx=5)
        self.goal_y_entry.insert(0, str(DEFAULT_GOAL_Y))
        tk.Label(y_frame, text="meters", font=("Arial", 8), fg="#90caf9", bg='#2d2d2d').pack(side='left', padx=3)
        
        # Set Goal Button
        set_goal_btn = tk.Button(goal_frame, text="✓ SET GOAL", 
                                command=self.update_goal, 
                                bg="#2196f3", fg="white", font=("Arial", 9, "bold"),
                                width=15, relief='raised', borderwidth=2, cursor='hand2')
        set_goal_btn.pack(pady=5)
        
        self.current_goal_label = tk.Label(goal_frame, text=f"Current: ({DEFAULT_GOAL_X}, {DEFAULT_GOAL_Y})", 
                                          font=("Arial", 8), fg="#4caf50", bg='#2d2d2d')
        self.current_goal_label.pack(pady=2)
        
        # Battery display
        battery_frame = tk.Frame(left_panel, bg='#2d2d2d')
        battery_frame.pack(pady=5, padx=10)
        
        tk.Label(battery_frame, text="🔋 BATTERY", 
                font=("Arial", 11, "bold"), fg="#ff9800", bg='#2d2d2d').pack()
        
        self.battery_label = tk.Label(battery_frame, text="100.0%", 
                                      font=("Arial", 24, "bold"), fg="#4caf50", bg='#2d2d2d')
        self.battery_label.pack(pady=3)
        
        self.energy_label = tk.Label(battery_frame, text="Energy: 0.0 Wh", 
                                     font=("Arial", 9), fg="#90caf9", bg='#2d2d2d')
        self.energy_label.pack()
        
        # Telemetry
        telem_frame = tk.Frame(left_panel, bg='#2d2d2d')
        telem_frame.pack(pady=5, padx=10, fill='x')
        
        tk.Label(telem_frame, text="📊 TELEMETRY", 
                font=("Arial", 10, "bold"), fg="#2196f3", bg='#2d2d2d').pack()
        
        self.position_label = tk.Label(telem_frame, text="Pos: (0.0, 0.0)", 
                                      font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.position_label.pack(pady=2, anchor='w', padx=10)
        
        self.speed_label = tk.Label(telem_frame, text="Speed: 0.0 m/s", 
                                   font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.speed_label.pack(pady=2, anchor='w', padx=10)
        
        self.altitude_label = tk.Label(telem_frame, text="Alt: 0.0 m", 
                                      font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.altitude_label.pack(pady=2, anchor='w', padx=10)
        
        self.distance_label = tk.Label(telem_frame, text="Goal Dist: --m", 
                                      font=("Arial", 9), fg="white", bg='#2d2d2d')
        self.distance_label.pack(pady=2, anchor='w', padx=10)
        
        # Control buttons - LARGER AND MORE VISIBLE
        button_frame = tk.Frame(left_panel, bg='#2d2d2d')
        button_frame.pack(pady=20, padx=15)
        
        self.start_btn = tk.Button(button_frame, text="🚀 START FLIGHT", 
                                   command=self.start_flight, 
                                   bg="#4caf50", fg="white", font=("Arial", 14, "bold"),
                                   width=20, height=2, relief='raised', borderwidth=4,
                                   cursor='hand2')
        self.start_btn.pack(pady=8)
        
        self.stop_btn = tk.Button(button_frame, text="⏹ STOP FLIGHT", 
                                 command=self.stop_flight, 
                                 bg="#f44336", fg="white", font=("Arial", 14, "bold"),
                                 width=20, height=2, state="disabled", relief='raised', borderwidth=4,
                                 cursor='hand2')
        self.stop_btn.pack(pady=8)
        
        # Add some padding at bottom so buttons are visible when scrolling
        tk.Label(left_panel, text="", bg='#2d2d2d', height=2).pack()
        
        # Right panel - Graphs
        right_panel = tk.Frame(main_container, bg='#2d2d2d', relief='raised', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        graph_title = tk.Label(right_panel, text="📈 PERFORMANCE ANALYTICS", 
                              font=("Arial", 13, "bold"), fg="#4caf50", bg='#2d2d2d')
        graph_title.pack(pady=8)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(13, 8), facecolor='#2d2d2d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Connect click event for goal setting
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        self.init_graphs()
    
    def init_graphs(self):
        """Initialize empty graphs"""
        self.fig.clear()
        
        ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
        ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
        ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
        ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
        ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'Start flight...', 
                   ha='center', va='center', fontsize=11, color='#757575',
                   transform=ax.transAxes)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#404040')
        
        # Initialize flight map with goal position
        ax5.scatter(0, 0, color='#4caf50', s=180, marker='o', 
                   edgecolors='white', linewidths=2.5, label='Start', zorder=5)
        ax5.scatter(self.goal_x, self.goal_y, 
                   color='#ff0000', s=350, marker='*', 
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
        
        # Set reasonable initial limits
        ax5.set_xlim([-20, max(self.goal_x + 20, 120)])
        ax5.set_ylim([-20, max(self.goal_y + 20, 120)])
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_graphs(self):
        """Update performance graphs"""
        if not self.running:
            return
            
        if len(self.metrics['time']) < 2:
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
            ax3.axhline(y=20, color='#f44336', linestyle='--', alpha=0.7, linewidth=2)
            ax3.set_ylim([0, 105])
            ax3.tick_params(colors='white', labelsize=8)
            for spine in ax3.spines.values():
                spine.set_color('#404040')
            
            # Energy
            ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
            ax4.plot(times, self.metrics['energy'], color='#e91e63', linewidth=2)
            ax4.set_title('ENERGY', fontweight='bold', fontsize=10, color='white')
            ax4.set_ylabel('Wh', fontsize=8, color='white')
            ax4.set_xlabel('Time (s)', fontsize=8, color='white')
            ax4.grid(alpha=0.2, color='#404040')
            ax4.tick_params(colors='white', labelsize=8)
            for spine in ax4.spines.values():
                spine.set_color('#404040')
            
            # Flight Map
            ax5 = self.fig.add_subplot(2, 3, (5, 6), facecolor='#1e1e1e')
            
            # Draw satellite-style terrain background
            x_range = max(abs(self.goal_x) + 40, 140)
            y_range = max(abs(self.goal_y) + 40, 140)
            x = np.linspace(-x_range, x_range, 120)
            y = np.linspace(-y_range, y_range, 120)
            X, Y = np.meshgrid(x, y)
            Z = (np.sin(X/40) * np.cos(Y/40) * 25 + 
                 np.sin(X/20) * np.cos(Y/30) * 15 + 
                 np.random.rand(120, 120) * 8)
            ax5.contourf(X, Y, Z, levels=25, cmap='terrain', alpha=0.4, zorder=0)
            
            # Draw map background if available
            if self.map_background is not None:
                ax5.imshow(self.map_background, extent=self.map_extent, 
                          aspect='auto', alpha=0.7, zorder=0)
            
            # Draw start position (green circle)
            ax5.scatter(0, 0, color='#4caf50', s=180, marker='o', 
                       edgecolors='white', linewidths=2.5, label='Start', zorder=4)
            
            # Draw GOAL STAR FIRST (bright red with yellow border - ALWAYS VISIBLE)
            ax5.scatter(self.goal_x, self.goal_y, 
                       color='#ff0000', s=350, marker='*', 
                       edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
            
            # Draw flight path (cyan trail)
            if len(self.metrics['positions_x']) > 1:
                ax5.plot(self.metrics['positions_x'], self.metrics['positions_y'], 
                        color='#00ffff', linewidth=3, alpha=0.8, zorder=5)
            
            # Draw current drone position (blue circle)
            if len(self.metrics['positions_x']) > 0:
                ax5.scatter(self.metrics['positions_x'][-1], self.metrics['positions_y'][-1], 
                           color='#2196f3', s=200, marker='o', 
                           edgecolors='white', linewidths=2, label='Drone', zorder=6)
            
            map_title = 'FLIGHT MAP' if self.running else 'FLIGHT MAP - Click to set goal'
            ax5.set_title(map_title, fontweight='bold', fontsize=10, color='white')
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
    
    def update_vision_display(self):
        """Update depth vision visualization"""
        if self.current_depth_image is not None:
            try:
                # Convert depth image to displayable format
                depth_img = self.current_depth_image.copy()
                
                # Normalize and colorize
                depth_normalized = np.clip(depth_img / 20.0, 0, 1)
                depth_colored = (depth_normalized * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)
                
                # Resize for display
                depth_resized = cv2.resize(depth_colored, (420, 250))
                
                # Draw region boundaries
                h, w = depth_resized.shape[:2]
                cv2.line(depth_resized, (w//3, 0), (w//3, h), (255, 255, 255), 2)
                cv2.line(depth_resized, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 2)
                cv2.line(depth_resized, (0, h//3), (w, h//3), (255, 255, 255), 2)
                
                # Labels
                cv2.putText(depth_resized, "LEFT", (40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(depth_resized, "CENTER", (w//2-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(depth_resized, "RIGHT", (2*w//3+20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Convert to PhotoImage
                depth_rgb = cv2.cvtColor(depth_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(depth_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                # Update canvas
                self.vision_canvas.delete("all")
                self.vision_canvas.create_image(210, 125, image=img_tk)
                self.vision_canvas.image = img_tk  # Keep reference
                
            except Exception as e:
                print(f"Vision display error: {e}")
    
    def start_flight(self):
        self.status_label.config(text="● Initializing...", fg="orange")
        self.window.update()
        threading.Thread(target=self._flight_loop, daemon=True).start()
    
    def update_goal(self):
        """Update goal position from entry fields"""
        try:
            new_x = float(self.goal_x_entry.get())
            new_y = float(self.goal_y_entry.get())
            
            self.goal_x = new_x
            self.goal_y = new_y
            
            self.current_goal_label.config(text=f"Current: ({new_x}, {new_y})", fg="#4caf50")
            
            # REDRAW MAP TO SHOW NEW GOAL
            self.redraw_flight_map()
            
            messagebox.showinfo("Goal Updated", f"New goal set to:\nX: {new_x} m\nY: {new_y} m")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for X and Y coordinates!")
            self.goal_x_entry.delete(0, tk.END)
            self.goal_y_entry.delete(0, tk.END)
            self.goal_x_entry.insert(0, str(self.goal_x))
            self.goal_y_entry.insert(0, str(self.goal_y))
    
    def on_map_click(self, event):
        """Handle click on flight map to set goal"""
        # Only allow setting goal when not flying
        if self.flight_active or self.running:
            return
        
        # Check if click is on the flight map (subplot 5-6)
        if event.inaxes is None:
            return
        
        # Get the axes and check if it's the flight map (bottom right subplot)
        try:
            axes_list = self.fig.get_axes()
            if len(axes_list) >= 5 and event.inaxes == axes_list[4]:  # Flight map is the 5th subplot
                # Get clicked coordinates
                click_x = event.xdata
                click_y = event.ydata
                
                if click_x is not None and click_y is not None:
                    # Update goal
                    self.goal_x = round(click_x, 1)
                    self.goal_y = round(click_y, 1)
                    
                    # Update entry fields
                    self.goal_x_entry.delete(0, tk.END)
                    self.goal_y_entry.delete(0, tk.END)
                    self.goal_x_entry.insert(0, str(self.goal_x))
                    self.goal_y_entry.insert(0, str(self.goal_y))
                    
                    # Update current goal label
                    self.current_goal_label.config(text=f"Current: ({self.goal_x}, {self.goal_y})", fg="#4caf50")
                    
                    # REDRAW THE MAP WITH NEW GOAL POSITION
                    self.redraw_flight_map()
                    
                    print(f"🎯 Goal set by map click: ({self.goal_x}, {self.goal_y})")
        except Exception as e:
            print(f"Map click error: {e}")
    
    def redraw_flight_map(self):
        """Redraw only the flight map with updated goal position"""
        try:
            axes_list = self.fig.get_axes()
            if len(axes_list) >= 5:
                ax5 = axes_list[4]  # Flight map
                ax5.clear()
                
                # Draw start position
                ax5.scatter(0, 0, color='#4caf50', s=150, marker='o', 
                           edgecolors='white', linewidths=2, label='Start', zorder=5)
                
                # Draw GOAL STAR (BRIGHT RED with YELLOW border - HIGHEST ZORDER)
                ax5.scatter(self.goal_x, self.goal_y, 
                           color='#ff0000', s=350, marker='*', 
                           edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
                
                # Draw existing path if any
                if len(self.metrics['positions_x']) > 1:
                    ax5.plot(self.metrics['positions_x'], self.metrics['positions_y'], 
                            color='#64b5f6', linewidth=2, alpha=0.6)
                
                if len(self.metrics['positions_x']) > 0:
                    ax5.scatter(self.metrics['positions_x'][-1], self.metrics['positions_y'][-1], 
                               color='#2196f3', s=200, marker='o', 
                               edgecolors='white', linewidths=2, zorder=5)
                
                # Set title
                map_title = 'FLIGHT MAP' if self.running else 'FLIGHT MAP - Click to set goal'
                ax5.set_title(map_title, fontweight='bold', fontsize=10, color='white')
                ax5.set_xlabel('X (m)', fontsize=8, color='white')
                ax5.set_ylabel('Y (m)', fontsize=8, color='white')
                ax5.grid(alpha=0.2, color='#404040')
                ax5.legend(loc='upper right', fontsize=8, facecolor='#2d2d2d', 
                          edgecolor='#404040', labelcolor='white')
                ax5.tick_params(colors='white')
                for spine in ax5.spines.values():
                    spine.set_color('#404040')
                
                # Set limits to show goal
                x_max = max(abs(self.goal_x) + 20, 120)
                y_max = max(abs(self.goal_y) + 20, 120)
                ax5.set_xlim([-20, x_max])
                ax5.set_ylim([-20, y_max])
                
                self.canvas.draw()
        except Exception as e:
            print(f"Redraw map error: {e}")
    
    def _flight_loop(self):
        """Main flight loop with vision and AI"""
        try:
            # Initialize
            print("🧠 Loading PPO Agent...")
            self.agent = PPO_Agent(state_dim=7, action_dim=3, lr=0, gamma=0, K_epochs=0)
            
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                if 'actor_state_dict' in checkpoint:
                    self.agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                else:
                    self.agent.actor.load_state_dict(checkpoint, strict=False)
                print("✓ Brain loaded")
            except:
                print("⚠️ Using random policy")
            
            self.agent.actor.eval()
            
            # Connect AirSim
            print("🔌 Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            print("🛫 Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(1.5)
            self.client.moveToZAsync(-20.0, 3.0).join()  # Climb to 20m altitude
            time.sleep(0.5)
            
            # Reset metrics
            self.metrics = {
                'time': [], 'speeds': [], 'altitudes': [], 
                'battery': [], 'energy': [], 
                'positions_x': [], 'positions_y': [], 'obstacles': []
            }
            self.battery_percent = 100.0
            self.total_energy_consumed = 0.0
            self.start_time = time.time()
            
            # Get initial position and ground height
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            self.ground_height = abs(pos.z_val) - 20.0  # Calculate ground offset
            print(f"📍 Ground height: {self.ground_height:.1f}m")
            
            # Initialize graphs with first data point BEFORE flight starts
            print("📊 Initializing performance graphs...")
            self.metrics['time'].append(0.0)
            self.metrics['speeds'].append(0.0)
            self.metrics['altitudes'].append(abs(pos.z_val))
            self.metrics['battery'].append(100.0)
            self.metrics['energy'].append(0.0)
            self.metrics['positions_x'].append(pos.x_val)
            self.metrics['positions_y'].append(pos.y_val)
            self.update_graphs()
            print("✓ Graphs ready - starting flight!")
            
            self.running = True
            self.flight_active = True
            self.status_label.config(text="● Flying!", fg="#4caf50")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.goal_x_entry.config(state="disabled")
            self.goal_y_entry.config(state="disabled")
            
            # Start telemetry updates
            self.update_telemetry_loop()
            
            # Flight loop
            dt = 0.1
            step = 0
            
            while self.flight_active and self.running:
                loop_start = time.time()
                
                # Get vision data
                vision_data = self.get_depth_perception()
                
                if vision_data:
                    self.current_depth_image = vision_data['raw_image']
                    obstacle_info = vision_data['obstacle']
                    self.current_obstacle_type = obstacle_info['type']
                    
                    # Update distance labels
                    self.obstacle_distances = vision_data['distances']
                    
                    # Update vision display
                    self.update_vision_display()
                
                # Get state
                state_vec, current_pos, velocity = self.get_state_vector()
                
                if state_vec is None:
                    continue
                
                goal_pos = np.array([self.goal_x, self.goal_y])
                distance = np.linalg.norm(goal_pos - current_pos)
                
                # Check goal
                if distance < 5.0:
                    print(f"🏆 GOAL REACHED! Distance: {distance:.1f}m")
                    self.status_label.config(text="● Goal Reached!", fg="#4caf50")
                    break
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    print(f"Step {step:4d} | Pos: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | Goal: ({self.goal_x}, {self.goal_y}) | Dist: {distance:.1f}m | Battery: {self.battery_percent:.1f}%")
                
                # Execute obstacle avoidance or navigate to goal
                obstacle_avoided = False
                if vision_data and self.execute_evasive_maneuver(vision_data['obstacle']):
                    obstacle_avoided = True
                    # Still update metrics during obstacle avoidance
                    target_vx = 0.0
                    target_vy = 0.0
                
                # Get current altitude and ground-relative height
                state = self.client.getMultirotorState()
                current_altitude = abs(state.kinematics_estimated.position.z_val)
                height_above_ground = current_altitude - self.ground_height
                
                # PROPORTIONAL CONTROLLER - Navigate to goal
                if not obstacle_avoided:
                    direction_to_goal = goal_pos - current_pos
                    distance_to_goal = np.linalg.norm(direction_to_goal)
                    
                    if distance_to_goal > 0.1:
                        direction_normalized = direction_to_goal / distance_to_goal
                        # Speed: maintain steady speed, slow down near goal
                        if distance_to_goal > 30.0:
                            desired_speed = 12.0  # Steady cruise speed
                        else:
                            desired_speed = max(4.0, distance_to_goal * 0.4)  # Slow approach
                        target_vx = direction_normalized[0] * desired_speed
                        target_vy = direction_normalized[1] * desired_speed
                    else:
                        target_vx = 0.0
                        target_vy = 0.0
                    
                    # Maintain 20m height ABOVE GROUND (not absolute altitude)
                    target_height_above_ground = 20.0
                    height_error = target_height_above_ground - height_above_ground
                    target_vz = np.clip(height_error * 0.6, -4.0, 4.0)  # Proportional altitude control
                    
                    self.client.moveByVelocityAsync(float(target_vx), float(target_vy), float(-target_vz), duration=0.15)
                
                # Update metrics
                speed = np.linalg.norm([target_vx, target_vy])
                power_watts = P_HOVER * (1 + 0.005 * speed**2)
                energy_step = power_watts * (dt / 3600.0)
                self.total_energy_consumed += energy_step
                self.battery_percent = max(0, 100 - (self.total_energy_consumed / BATTERY_CAPACITY_WH * 100))
                
                elapsed_time = time.time() - self.start_time
                self.metrics['time'].append(elapsed_time)
                self.metrics['speeds'].append(speed)
                self.metrics['altitudes'].append(current_altitude)  # Use current_altitude from above
                self.metrics['battery'].append(self.battery_percent)
                self.metrics['energy'].append(self.total_energy_consumed)
                self.metrics['positions_x'].append(current_pos[0])  # Use current_pos from above
                self.metrics['positions_y'].append(current_pos[1])
                
                if step % 50 == 0:
                    try:
                        self.update_graphs()
                        if step % 100 == 0:
                            print(f"  📊 Graphs updated at step {step}")
                    except Exception as graph_err:
                        print(f"Graph update error at step {step}: {graph_err}")
                
                step += 1
                
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except Exception as e:
            print(f"Flight error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.flight_active = False
    
    def get_depth_perception(self):
        """Get depth image and analyze obstacles"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ])
            
            if not responses:
                return None
            
            img1d = np.array(responses[0].image_data_float, dtype=np.float32)
            if len(img1d) == 0:
                return None
            
            img2d = img1d.reshape(responses[0].height, responses[0].width)
            img2d = np.nan_to_num(img2d, nan=100.0, posinf=100.0, neginf=100.0)
            
            h, w = img2d.shape
            center = img2d[h//3:2*h//3, w//3:2*w//3]
            left = img2d[h//3:2*h//3, :w//3]
            right = img2d[h//3:2*h//3, 2*w//3:]
            top = img2d[:h//3, :]
            
            def safe_mean(region):
                valid = region[(region > 0.1) & (region < 100)]
                return np.mean(valid) if len(valid) > 0 else 100.0
            
            distances = {
                'center': safe_mean(center),
                'left': safe_mean(left),
                'right': safe_mean(right),
                'top': safe_mean(top)
            }
            
            obstacle_info = self.classify_obstacle(distances)
            
            return {
                'distances': distances,
                'obstacle': obstacle_info,
                'raw_image': img2d
            }
        except:
            return None
    
    def classify_obstacle(self, distances):
        """Classify obstacle type"""
        DANGER = 15.0  # Increased from 5.0 to 15.0 for earlier detection
        
        center_blocked = distances['center'] < DANGER
        left_blocked = distances['left'] < DANGER
        right_blocked = distances['right'] < DANGER
        
        if center_blocked and left_blocked and right_blocked:
            return {'type': 'HOUSE', 'action': 'CLIMB', 'message': '⚠️ HOUSE - CLIMBING!'}
        elif center_blocked and (not left_blocked or not right_blocked):
            swerve = 'LEFT' if distances['left'] > distances['right'] else 'RIGHT'
            return {'type': 'POLE', 'action': f'SWERVE_{swerve}', 'message': f'⚠️ POLE - SWERVING {swerve}!'}
        elif left_blocked or right_blocked:
            side = 'RIGHT' if left_blocked else 'LEFT'
            return {'type': 'GAP', 'action': f'NAV_{side}', 'message': f'⚠️ GAP - {side}!'}
        else:
            return {'type': 'CLEAR', 'action': 'AI', 'message': '✓ AI Control'}
    
    def execute_evasive_maneuver(self, obstacle_info):
        """Execute obstacle avoidance"""
        action = obstacle_info['action']
        
        if action == 'CLIMB':
            # Climb higher and faster to clear buildings
            self.client.moveByVelocityAsync(2.0, 0, -8.0, duration=2.0)
            return True
        elif action.startswith('SWERVE_'):
            # More aggressive swerve to avoid poles/trees
            direction = 1.0 if 'RIGHT' in action else -1.0
            self.client.moveByVelocityAsync(4.0, direction * 15.0, -2.0, duration=1.0)
            return True
        elif action.startswith('NAV_'):
            # Navigate around gaps with stronger correction
            direction = 1.0 if 'RIGHT' in action else -1.0
            self.client.moveByVelocityAsync(5.0, direction * 8.0, -2.0, duration=0.8)
            return True
        
        return False
    
    def get_state_vector(self):
        """Get state for PPO agent"""
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            current_pos = np.array([pos.x_val, pos.y_val])
            goal_pos = np.array([self.goal_x, self.goal_y])
            dist_vector = goal_pos - current_pos
            velocity = np.array([vel.x_val, vel.y_val])
            wind = velocity * 0.1
            
            state_vec = np.concatenate([
                dist_vector, velocity, wind, [self.battery_percent / 100.0]
            ]).astype(np.float32)
            
            return state_vec, current_pos, velocity
        except:
            return None, None, None
    
    def update_telemetry_loop(self):
        """Update GUI telemetry"""
        if not self.running:
            return
        
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            # Update labels
            self.position_label.config(text=f"Pos: ({pos.x_val:.1f}, {pos.y_val:.1f})")
            speed = np.linalg.norm([vel.x_val, vel.y_val])
            self.speed_label.config(text=f"Speed: {speed:.2f} m/s")
            
            # Show height above ground
            height_above_ground = abs(pos.z_val) - self.ground_height
            self.altitude_label.config(text=f"Alt: {height_above_ground:.1f} m AGL")
            
            goal_pos = np.array([self.goal_x, self.goal_y])
            distance = np.linalg.norm(goal_pos - np.array([pos.x_val, pos.y_val]))
            self.distance_label.config(text=f"Goal Dist: {distance:.1f} m")
            
            # Update battery
            battery_color = "#4caf50" if self.battery_percent > 50 else "#ff9800" if self.battery_percent > 20 else "#f44336"
            self.battery_label.config(text=f"{self.battery_percent:.1f}%", fg=battery_color)
            self.energy_label.config(text=f"Energy: {self.total_energy_consumed:.2f} Wh")
            
            # Update obstacle status
            colors = {'CLEAR': '#4caf50', 'POLE': '#ff9800', 'HOUSE': '#f44336', 'GAP': '#2196f3'}
            self.obstacle_label.config(
                text=self.current_obstacle_type, 
                fg=colors.get(self.current_obstacle_type, '#4caf50')
            )
            
            # Update distance indicators
            self.dist_center_label.config(text=f"Center: {self.obstacle_distances['center']:.1f}m")
            self.dist_left_label.config(text=f"Left: {self.obstacle_distances['left']:.1f}m")
            self.dist_right_label.config(text=f"Right: {self.obstacle_distances['right']:.1f}m")
            
        except:
            pass
        
        if self.running:
            self.window.after(100, self.update_telemetry_loop)
    
    def smart_landing(self):
        """Smart landing with obstacle avoidance"""
        try:
            print("\n🛬 Smart Landing Sequence...")
            self.status_label.config(text="● Smart Landing...", fg="orange")
            
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            current_height = abs(pos.z_val) - self.ground_height
            
            print(f"  📍 Current position: ({pos.x_val:.1f}, {pos.y_val:.1f}), Height: {current_height:.1f}m AGL")
            
            # Phase 1: Gradual descent with obstacle detection
            descent_step = 0
            safe_descent_height = 5.0  # Stop descent at 5m above ground
            
            while current_height > safe_descent_height:
                # Check for obstacles below
                vision_data = self.get_depth_perception()
                
                if vision_data:
                    distances = vision_data['distances']
                    center_dist = distances['center']
                    
                    # Check if obstacle directly below
                    if center_dist < 10.0:
                        print(f"  ⚠️ Obstacle detected below at {center_dist:.1f}m!")
                        print("  ↔️ Moving sideways to clear area...")
                        
                        # Find clear direction
                        left_clear = distances['left'] > 15.0
                        right_clear = distances['right'] > 15.0
                        
                        if left_clear:
                            print("  ← Moving LEFT to avoid obstacle")
                            self.client.moveByVelocityAsync(0, -6.0, 0, duration=2.0).join()
                        elif right_clear:
                            print("  → Moving RIGHT to avoid obstacle")
                            self.client.moveByVelocityAsync(0, 6.0, 0, duration=2.0).join()
                        else:
                            print("  ← Moving BACK to avoid obstacle")
                            self.client.moveByVelocityAsync(-6.0, 0, 0, duration=2.0).join()
                        
                        time.sleep(0.5)
                        continue
                
                # Descend gradually
                descent_step += 1
                if descent_step % 3 == 0:
                    print(f"  ⬇️ Descending... Height: {current_height:.1f}m AGL")
                
                self.client.moveByVelocityAsync(0, 0, 4.0, duration=0.5).join()
                time.sleep(0.3)
                
                # Update current height
                state = self.client.getMultirotorState()
                pos = state.kinematics_estimated.position
                current_height = abs(pos.z_val) - self.ground_height
            
            print(f"  ✓ Clear descent to {current_height:.1f}m AGL")
            
            # Phase 2: Navigate to goal position at low altitude
            goal_pos = np.array([self.goal_x, self.goal_y])
            current_pos = np.array([pos.x_val, pos.y_val])
            distance_to_goal = np.linalg.norm(goal_pos - current_pos)
            
            if distance_to_goal > 3.0:
                print(f"  🎯 Moving to goal position ({self.goal_x:.1f}, {self.goal_y:.1f})...")
                print(f"     Distance: {distance_to_goal:.1f}m")
                
                # Move horizontally to goal while maintaining low altitude
                move_steps = 0
                max_move_steps = 50
                
                while distance_to_goal > 2.0 and move_steps < max_move_steps:
                    direction = (goal_pos - current_pos) / distance_to_goal
                    speed = min(5.0, distance_to_goal * 0.5)
                    
                    # Move towards goal at safe height
                    self.client.moveByVelocityAsync(
                        float(direction[0] * speed),
                        float(direction[1] * speed),
                        0,  # Maintain current altitude
                        duration=0.5
                    ).join()
                    
                    # Update position
                    state = self.client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    current_pos = np.array([pos.x_val, pos.y_val])
                    distance_to_goal = np.linalg.norm(goal_pos - current_pos)
                    
                    move_steps += 1
                    if move_steps % 5 == 0:
                        print(f"     Distance to goal: {distance_to_goal:.1f}m")
                
                print("  ✓ Reached goal position")
            
            # Phase 3: Final landing
            print("  🛬 Final landing...")
            self.client.moveByVelocityAsync(0, 0, 0, 0.5).join()  # Stop
            self.client.landAsync().join()
            time.sleep(0.5)
            
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✓ Landed safely")
            self.status_label.config(text="● Landed", fg="gray")
            
        except Exception as e:
            print(f"Landing error: {e}")
            import traceback
            traceback.print_exc()
            # Emergency land
            try:
                self.client.landAsync().join()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except:
                pass
    
    def stop_flight(self):
        """Stop flight and initiate smart landing"""
        self.flight_active = False
        self.running = False
        
        if self.client:
            self.smart_landing()
        
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self.goal_x_entry.config(state="normal")
        self.goal_y_entry.config(state="normal")
    
    def run(self):
        """Run GUI"""
        self.window.mainloop()


if __name__ == "__main__":
    print("="*70)
    print("👁️ SMART VISION DRONE CONTROL CENTER")
    print("="*70)
    
    app = SmartVisionDroneGUI()
    app.run()
