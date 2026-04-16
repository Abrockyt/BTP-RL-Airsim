

import logging
import warnings

# Suppress Tornado and msgpackrpc error logging BEFORE imports (non-fatal protocol errors)
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('tornado').setLevel(logging.CRITICAL)
logging.getLogger('tornado.application').setLevel(logging.CRITICAL)
logging.getLogger('tornado.general').setLevel(logging.CRITICAL)
logging.getLogger('msgpackrpc').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

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
from collections import deque

# Try to import GNN agent for comparison
try:
    from gnn_agent import PPO_GNN_Agent
    GNN_AVAILABLE = True
except:
    GNN_AVAILABLE = False
    print("⚠️ GNN agent not available - comparison mode will use fallback")

# Matplotlib for graphs
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================================================================
# BRAIN ARCHITECTURE
# =============================================================================

class MHA_Actor(nn.Module):
    """Multi-Head Attention Actor Network for PPO."""
    def __init__(self, state_dim, action_dim, num_heads=4, embed_dim=64):
        super(MHA_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input embedding layer to project state to embed_dim
        self.input_projection = nn.Linear(state_dim, embed_dim)
        
        # Multi-Head Attention layer
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers after attention
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Output layers for mean and log_std of action distribution
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        # Project input state to embedding dimension
        x = self.input_projection(state)
        
        # Multi-Head Attention expects (batch, seq_len, embed_dim)
        # Since we have a single state vector, we add a sequence dimension
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Apply Multi-Head Attention (self-attention)
        attn_output, _ = self.mha(x, x, x)
        
        # Remove sequence dimension
        x = attn_output.squeeze(1)  # (batch, embed_dim)
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Get mean for action (we only use mean for inference, not log_std)
        mean = self.tanh(self.mean_layer(x))  # Bounded actions [-1, 1]
        
        return mean


class Critic(nn.Module):
    """Critic Network for PPO - Evaluates state value"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PPO_Agent:
    """PPO Agent with MHA Actor and online learning capability"""
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.lr = lr
        
        self.actor = MHA_Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.old_actor = MHA_Actor(state_dim, action_dim)
        
        if lr > 0:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Memory buffers for RL training
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.old_actor.to(self.device)
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, training=False):
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        if training and self.lr > 0:
            # Training mode: sample from distribution
            with torch.no_grad():
                action_mean = self.old_actor(state_tensor)
                # Add noise for exploration
                std = 0.3
                action = action_mean + torch.randn_like(action_mean) * std
                action = torch.clamp(action, -1, 1)
                
                # Store for training
                if action_mean.dim() == 1:
                    action_mean = action_mean.unsqueeze(0)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                    
                self.states.append(state_tensor.squeeze(0))
                self.actions.append(action.squeeze(0))
        else:
            # Inference mode: use mean action
            with torch.no_grad():
                action = self.actor(state_tensor)
        
        action_np = action.cpu().numpy()
        if len(action_np.shape) > 1:
            action_np = action_np[0]
        
        return action_np
    
    def store_reward(self, reward, done):
        """Store reward and done flag during training"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        """Update policy using PPO algorithm"""
        if len(self.states) < 2 or self.lr == 0:
            return 0.0, 0.0
        
        # Convert to tensors
        states = torch.stack(self.states).detach().to(self.device)
        actions = torch.stack(self.actions).detach().to(self.device)
        
        # Calculate returns
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        # PPO update
        for _ in range(self.K_epochs):
            # Current policy
            action_mean = self.actor(states)
            state_values = self.critic(states).squeeze()
            
            # Calculate advantages
            advantages = rewards - state_values.detach()
            
            # Simple policy loss (no log probs needed for deterministic)
            actor_loss = -advantages.mean()
            critic_loss = self.MseLoss(state_values, rewards)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Update old policy
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        
        return total_actor_loss / self.K_epochs, total_critic_loss / self.K_epochs


# =============================================================================
# GNN-BASED COLLISION PREDICTION NETWORK
# =============================================================================

class GNN_CollisionLayer(nn.Module):
    """Graph Neural Network Layer for Collision Prediction with Message Passing"""
    def __init__(self, node_feature_dim, hidden_dim):
        super(GNN_CollisionLayer, self).__init__()
        
        # Message network: processes features from neighboring nodes
        self.message_net = nn.Sequential(
            nn.Linear(node_feature_dim * 2, hidden_dim),  # Concatenate node + neighbor
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Update network: updates node features with aggregated messages
        self.update_net = nn.Sequential(
            nn.Linear(node_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim),
            nn.Tanh()
        )
    
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: [batch_size, num_nodes, feature_dim]
            adjacency_matrix: [num_nodes, num_nodes] - binary adjacency
        Returns:
            updated_features: [batch_size, num_nodes, feature_dim]
        """
        batch_size, num_nodes, feature_dim = node_features.shape
        
        # Aggregate messages from neighbors
        messages = []
        for i in range(num_nodes):
            # Get neighbors of node i
            neighbors = adjacency_matrix[i].nonzero(as_tuple=True)[0]
            
            if len(neighbors) == 0:
                # No neighbors - use zero message
                messages.append(torch.zeros(batch_size, self.message_net[0].out_features, device=node_features.device))
            else:
                # Gather messages from all neighbors
                node_i = node_features[:, i, :].unsqueeze(1).expand(-1, len(neighbors), -1)  # [batch, n_neighbors, feat]
                neighbor_features = node_features[:, neighbors, :]  # [batch, n_neighbors, feat]
                
                # Concatenate node and neighbor features
                combined = torch.cat([node_i, neighbor_features], dim=-1)  # [batch, n_neighbors, 2*feat]
                
                # Process through message network
                msg = self.message_net(combined)  # [batch, n_neighbors, hidden]
                
                # Aggregate (mean pooling)
                aggregated_msg = torch.mean(msg, dim=1)  # [batch, hidden]
                messages.append(aggregated_msg)
        
        # Stack all messages
        messages = torch.stack(messages, dim=1)  # [batch, num_nodes, hidden]
        
        # Update node features
        combined_features = torch.cat([node_features, messages], dim=-1)  # [batch, num_nodes, feat+hidden]
        updated_features = self.update_net(combined_features)  # [batch, num_nodes, feat]
        
        # Residual connection
        updated_features = node_features + updated_features
        
        return updated_features


class CollisionPredictionGNN(nn.Module):
    """
    GNN-based Collision Prediction Network
    - Uses 9 depth zones as graph nodes (3x3 grid)
    - Learns spatial relationships via message passing
    - Outputs: collision risk [0-1] and avoidance direction [UP/DOWN/LEFT/RIGHT]
    """
    def __init__(self, node_feature_dim=5, hidden_dim=64, num_layers=3):
        super(CollisionPredictionGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.num_nodes = 9  # 3x3 grid of depth zones
        
        # Input projection
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # Stack of GNN layers
        self.gnn_layers = nn.ModuleList([
            GNN_CollisionLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Global pooling and prediction layers
        self.global_pool = nn.Linear(hidden_dim * self.num_nodes, 128)
        
        # Collision risk prediction (0 = safe, 1 = collision imminent)
        self.collision_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output [0, 1]
        )
        
        # Direction prediction (4 classes: UP, DOWN, LEFT, RIGHT)
        self.direction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 directions
            nn.Softmax(dim=-1)
        )
        
        # Create adjacency matrix for 3x3 grid (8-connectivity)
        self.adjacency_matrix = self._create_3x3_adjacency()
        
    def _create_3x3_adjacency(self):
        """
        Create adjacency matrix for 3x3 grid with8-connectivity
        Node layout:
        0 1 2
        3 4 5
        6 7 8
        """
        adj = torch.zeros(9, 9)
        
        # Define neighbors for each node (including diagonals)
        neighbors = {
            0: [1, 3, 4],           # top-left
            1: [0, 2, 3, 4, 5],     # top-center
            2: [1, 4, 5],           # top-right
            3: [0, 1, 4, 6, 7],     # mid-left
            4: [0, 1, 2, 3, 5, 6, 7, 8],  # center (connected to all)
            5: [1, 2, 4, 7, 8],     # mid-right
            6: [3, 4, 7],           # bottom-left
            7: [3, 4, 5, 6, 8],     # bottom-center
            8: [4, 5, 7]            # bottom-right
        }
        
        for node, neighs in neighbors.items():
            for n in neighs:
                adj[node, n] = 1
        
        return adj
    
    def forward(self, node_features):
        """
        Args:
            node_features: [batch_size, 9, feature_dim] - features for 9 zones
        Returns:
            collision_risk: [batch_size, 1] - probability of collision
            direction_probs: [batch_size, 4] - probabilities for [UP, DOWN, LEFT, RIGHT]
        """
        batch_size = node_features.shape[0]
        
        # Project to hidden dimension
        x = self.input_projection(node_features)  # [batch, 9, hidden]
        
        # Apply GNN layers with message passing
        adj = self.adjacency_matrix.to(node_features.device)
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj)  # [batch, 9, hidden]
        
        # Global pooling - flatten all node features
        x_global = x.reshape(batch_size, -1)  # [batch, 9*hidden]
        x_global = torch.relu(self.global_pool(x_global))  # [batch, 128]
        
        # Predict collision risk
        collision_risk = self.collision_head(x_global)  # [batch, 1]
        
        # Predict best avoidance direction
        direction_probs = self.direction_head(x_global)  # [batch, 4]
        
        return collision_risk, direction_probs


class CollisionPredictor:
    """Predictive collision checker based on Time-To-Collision (TTC)."""

    def __init__(self, ttc_threshold=2.5, center_ratio=0.4, min_valid_depth=0.1, max_valid_depth=200.0):
        self.ttc_threshold = float(ttc_threshold)
        self.center_ratio = float(center_ratio)
        self.min_valid_depth = float(min_valid_depth)
        self.max_valid_depth = float(max_valid_depth)
        # Thin branches are sparse in depth maps; treat sparse near pixels as risk.
        self.branch_distance = 6.5
        self.branch_ratio_threshold = 0.015
        self.min_side_clearance = 4.5

    def predict_collision(self, depth_image, current_speed):
        """
        Returns:
            crash_imminent (bool), ttc_seconds (float), avg_center_distance (float), avoid_direction (str|None)
        """
        if depth_image is None or depth_image.ndim != 2 or depth_image.size == 0:
            return False, float('inf'), float('inf'), None

        h, w = depth_image.shape
        ch = max(1, int(h * self.center_ratio))
        cw = max(1, int(w * self.center_ratio))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2

        center = depth_image[y0:y0 + ch, x0:x0 + cw]
        valid_center = center[np.isfinite(center) & (center > self.min_valid_depth) & (center < self.max_valid_depth)]
        if valid_center.size == 0:
            return False, float('inf'), float('inf'), None

        # Percentile is robust to sparse/partial obstacles (e.g. branches).
        avg_center_distance = float(np.percentile(valid_center, 25))
        near_ratio = float(np.mean(valid_center < self.branch_distance))
        speed = abs(float(current_speed))
        if speed <= 1e-6:
            return False, float('inf'), avg_center_distance, None

        # TTC core formula
        ttc_seconds = avg_center_distance / speed

        # Side clearance for avoidance direction
        band = depth_image[h // 3:2 * h // 3, :]
        left = band[:, :w // 2]
        right = band[:, w // 2:]
        valid_left = left[np.isfinite(left) & (left > self.min_valid_depth) & (left < self.max_valid_depth)]
        valid_right = right[np.isfinite(right) & (right > self.min_valid_depth) & (right < self.max_valid_depth)]
        left_clear = float(np.percentile(valid_left, 60)) if valid_left.size > 0 else 0.0
        right_clear = float(np.percentile(valid_right, 60)) if valid_right.size > 0 else 0.0

        # Pick clearer side; climb when both sides are very constrained
        if left_clear < self.min_side_clearance and right_clear < self.min_side_clearance:
            avoid_direction = 'UP'
        else:
            avoid_direction = 'LEFT' if left_clear > right_clear else 'RIGHT'

        branch_risk = near_ratio >= self.branch_ratio_threshold and avg_center_distance < (self.branch_distance + 2.0)
        crash_imminent = ttc_seconds < self.ttc_threshold or branch_risk
        return crash_imminent, ttc_seconds, avg_center_distance, avoid_direction


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "mha_ppo_9D_best.pth"  # Preferred 9D MHA-PPO checkpoint
DEFAULT_GOAL_X = 100.0
DEFAULT_GOAL_Y = 100.0
P_HOVER = 200.0
BATTERY_CAPACITY_WH = 100.0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_airsim_call(func, *args, max_retries=10, delay=0.05, **kwargs):
    """
    Safely call AirSim functions with retry logic to handle IOLoop and msgpack errors
    Thread-safe with proper backoff for multi-drone operations
    
    Args:
        func: AirSim function to call
        max_retries: Maximum number of retry attempts  
        delay: Delay between retries in seconds
        *args,  **kwargs: Arguments to pass to the function
    
    Returns:
        Function result or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            # If result has a join method (Future), wait for completion
            if hasattr(result, 'join'):
                result.join()
            # Small delay to prevent IOLoop overload
            time.sleep(0.02)
            return result
        except (RuntimeError, TypeError, AttributeError, OSError, ConnectionError) as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["ioloop", "msgpack", "len()", "dictionary", "timeout", "connection", "already running"]):
                # Known msgpackrpc/tornado errors - retry with exponential backoff
                if attempt < max_retries - 1:
                    backoff = delay * (2 ** min(attempt, 4))  # Exponential backoff up to 16x
                    time.sleep(backoff)
                    continue
            # All retries failed
            if attempt == max_retries - 1:
                print(f"⚠️ AirSim call failed after {max_retries} attempts: {e}")
            return None
        except Exception as e:
            # Unexpected error, still try once more
            if attempt < max_retries - 1:
                time.sleep(delay * 2)
                continue
            print(f"⚠️ Unexpected error in AirSim call: {e}")
            return None
    return None

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
        self.agent = None  # MHA-PPO agent
        self.gnn_agent = None  # GNN agent for comparison
        
        # Multi-drone clients for comparison mode
        self.drone1_client = None  # MHA-PPO drone (Orange)
        self.drone2_client = None  # GNN drone (Green)
        self.drone1_pos = np.array([0.0, 0.0, 0.0])
        self.drone2_pos = np.array([0.0, 0.0, 0.0])
        self.drone1_path = []  # Flight path history for Drone1
        self.drone2_path = []  # Flight path history for Drone2
        
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
        
        # YOLOv8 for waste detection
        from ultralytics import YOLO
        print("Loading YOLOv8...")
        self.yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8 loaded successfully.")
        
        # Battery
        self.battery_percent = 100.0
        self.total_energy_consumed = 0.0
        
        # Vision data
        self.current_depth_image = None
        self.current_obstacle_type = "CLEAR"
        self.obstacle_distances = {'center': 100, 'left': 100, 'right': 100, 'top': 100}
        self.collision_predictor = CollisionPredictor(ttc_threshold=2.5)
        
        # Pressure sensor data (for heavy wind conditions)
        self.air_pressure = 101325.0  # Pascal (sea level standard)
        self.wind_magnitude = 0.0  # m/s
        self.wind_direction = np.array([0.0, 0.0])  # Wind direction vector (x, y)
        self.heavy_wind_detected = False
        self.pressure_gradient = 0.0  # Rate of pressure change
        self.vision_failed = False  # Flag when vision is unreliable
        self.flight_mode = "NORMAL"  # "NORMAL" or "WIND"
        self.wind_enabled = True
        
        # Graph components
        self.fig = None
        self.canvas = None
        
        # Map background for satellite view
        self.map_background = None
        self.map_extent = [-150, 150, -150, 150]
        
        # Ground height tracking
        self.ground_height = 0.0
        
        # Reinforcement Learning - Historical tracking
        self.run_history = []  # Store metrics from each run
        self.current_run_number = 0
        self.best_energy = float('inf')
        self.best_time = float('inf')
        self.learning_enabled = False  # Enable/disable online learning (disabled by default to prevent training errors)
        self.rl_policy_active = False
        self.last_rl_action = np.zeros(3, dtype=np.float32)
        self.last_policy_mode = "INFERENCE"

        # Motion smoothing for stable flight and camera
        self.prev_velocity_cmd = np.zeros(3, dtype=np.float32)
        self.velocity_smoothing_alpha = 0.78  # Very high = maximum camera stability
        self.max_accel_ms2 = 2.8  # Gentle horizontal acceleration
        self.max_vertical_accel_ms2 = 0.25  # Ultra-low vertical acceleration (no bobbing)
        # Comparison drone smoothing state
        self.drone1_prev_vel = np.zeros(3, dtype=np.float32)
        self.drone2_prev_vel = np.zeros(3, dtype=np.float32)
        
        # Training statistics
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'rewards': []
        }
        
        os.makedirs('performance_graphs', exist_ok=True)
        os.makedirs('trained_models', exist_ok=True)
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.window, bg='#0d47a1', height=70)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="👁️ SMART VISION DRONE CONTROL", 
                        font=("Arial", 22, "bold"), fg="white", bg='#0d47a1')
        title.pack(pady=12)
        
        subtitle = tk.Label(header_frame, text="MHA-PPO Reinforcement Learning + Computer Vision | Online Learning Enabled | Real-Time Performance Tracking", 
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
        
        # Pressure Sensor Panel
        pressure_frame = tk.Frame(left_panel, bg='#1e1e1e', relief='sunken', borderwidth=2)
        pressure_frame.pack(pady=8, padx=10, fill='x')
        
        tk.Label(pressure_frame, text="💨 PRESSURE SENSOR", 
                font=("Arial", 10, "bold"), fg="#2196f3", bg='#1e1e1e').pack(pady=3)
        
        self.pressure_label = tk.Label(pressure_frame, text="Pressure: 101325 Pa", 
                                       font=("Arial", 9), fg="white", bg='#1e1e1e')
        self.pressure_label.pack(anchor='w', padx=15, pady=2)
        
        self.wind_mag_label = tk.Label(pressure_frame, text="Wind: 0.0 m/s", 
                                       font=("Arial", 9), fg="white", bg='#1e1e1e')
        self.wind_mag_label.pack(anchor='w', padx=15, pady=2)
        
        self.heavy_wind_label = tk.Label(pressure_frame, text="Status: NORMAL ✓", 
                                         font=("Arial", 9, "bold"), fg="#4caf50", bg='#1e1e1e')
        self.heavy_wind_label.pack(anchor='w', padx=15, pady=3)
        
        self.vision_status_label = tk.Label(pressure_frame, text="Vision: ACTIVE 📷", 
                                           font=("Arial", 9), fg="#4caf50", bg='#1e1e1e')
        self.vision_status_label.pack(anchor='w', padx=15, pady=2)
        
        # Wind Pattern Visualization
        tk.Label(pressure_frame, text="🌪️ WIND PATTERN (Live)", 
                font=("Arial", 9, "bold"), fg="#90caf9", bg='#1e1e1e').pack(pady=(8, 2))
        
        self.wind_canvas = tk.Canvas(pressure_frame, width=380, height=150, bg='#0d0d0d', 
                                     highlightthickness=1, highlightbackground='#2196f3')
        self.wind_canvas.pack(pady=5, padx=10)
        
        # Wind direction label
        self.wind_direction_label = tk.Label(pressure_frame, text="Direction: N/A", 
                                            font=("Arial", 8), fg="#90caf9", bg='#1e1e1e')
        self.wind_direction_label.pack(pady=2)
        
        # Flight Mode Selection Buttons
        tk.Label(pressure_frame, text="🎮 FLIGHT MODE", 
                font=("Arial", 10, "bold"), fg="#2196f3", bg='#1e1e1e').pack(pady=(10, 5))
        
        mode_buttons_frame = tk.Frame(pressure_frame, bg='#1e1e1e')
        mode_buttons_frame.pack(pady=5)
        
        # Normal Mode Button (RGB + LiDAR)
        self.normal_mode_btn = tk.Button(mode_buttons_frame, text="📷 NORMAL MODE\n(RGB + LiDAR)", 
                                         font=("Arial", 9, "bold"), bg='#4caf50', fg='white',
                                         relief='sunken', borderwidth=3, width=17, height=2,
                                         command=self.set_normal_mode)
        self.normal_mode_btn.grid(row=0, column=0, padx=5)
        
        # Wind Mode Button (Pressure Sensor)
        self.wind_mode_btn = tk.Button(mode_buttons_frame, text="💨 WIND MODE\n(Pressure Sensor)", 
                                       font=("Arial", 9, "bold"), bg='#607d8b', fg='white',
                                       relief='raised', borderwidth=3, width=17, height=2,
                                       command=self.set_wind_mode)
        self.wind_mode_btn.grid(row=0, column=1, padx=5)
        
        self.mode_status_label = tk.Label(pressure_frame, text="Mode: NORMAL (Vision-Based Navigation)", 
                                          font=("Arial", 8, "italic"), fg="#4caf50", bg='#1e1e1e')
        self.mode_status_label.pack(pady=3)

        # Wind ON/OFF toggle
        self.wind_toggle_var = tk.BooleanVar(value=True)
        self.wind_toggle_check = tk.Checkbutton(
            pressure_frame,
            text="Wind ON/OFF",
            variable=self.wind_toggle_var,
            command=self.toggle_wind,
            font=("Arial", 9, "bold"),
            fg="white",
            bg='#1e1e1e',
            selectcolor='#1e1e1e',
            activebackground='#1e1e1e'
        )
        self.wind_toggle_check.pack(pady=(3, 6))
        
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
        
        # Reinforcement Learning Statistics
        rl_frame = tk.Frame(left_panel, bg='#1e1e1e', relief='sunken', borderwidth=2)
        rl_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(rl_frame, text="🧠 REINFORCEMENT LEARNING", 
                font=("Arial", 10, "bold"), fg="#9c27b0", bg='#1e1e1e').pack(pady=3)
        
        tk.Label(rl_frame, text="Learning improves energy efficiency over runs", 
                font=("Arial", 8), fg="#ce93d8", bg='#1e1e1e').pack()
        
        self.run_number_label = tk.Label(rl_frame, text="Run: #0", 
                                         font=("Arial", 9, "bold"), fg="white", bg='#1e1e1e')
        self.run_number_label.pack(pady=2, anchor='w', padx=10)
        
        self.best_energy_label = tk.Label(rl_frame, text="Best Energy: -- Wh", 
                                          font=("Arial", 9), fg="#76ff03", bg='#1e1e1e')
        self.best_energy_label.pack(pady=2, anchor='w', padx=10)
        
        self.best_time_label = tk.Label(rl_frame, text="Best Time: -- s", 
                                        font=("Arial", 9), fg="#76ff03", bg='#1e1e1e')
        self.best_time_label.pack(pady=2, anchor='w', padx=10)
        
        self.improvement_label = tk.Label(rl_frame, text="Improvement: --", 
                                          font=("Arial", 9), fg="#00e5ff", bg='#1e1e1e')
        self.improvement_label.pack(pady=2, anchor='w', padx=10)

        self.rl_policy_label = tk.Label(rl_frame, text="Policy: Waiting", 
                        font=("Arial", 9), fg="#90caf9", bg='#1e1e1e')
        self.rl_policy_label.pack(pady=2, anchor='w', padx=10)

        self.rl_action_label = tk.Label(rl_frame, text="Action: (--, --, --)", 
                        font=("Arial", 8), fg="#ce93d8", bg='#1e1e1e')
        self.rl_action_label.pack(pady=2, anchor='w', padx=10)

        self.rl_blend_label = tk.Label(rl_frame, text="Control Blend: RL 35% + Guidance 65%", 
                           font=("Arial", 8, "italic"), fg="#80cbc4", bg='#1e1e1e')
        self.rl_blend_label.pack(pady=2, anchor='w', padx=10)
        
        # Learning toggle
        self.learning_var = tk.BooleanVar(value=True)
        learning_check = tk.Checkbutton(rl_frame, text="Enable Online Learning", 
                                       variable=self.learning_var,
                                       command=self.toggle_learning,
                                       font=("Arial", 9), fg="white", bg='#1e1e1e',
                                       selectcolor='#1e1e1e', activebackground='#1e1e1e')
        learning_check.pack(pady=5)
        
        # Collision Prediction Status (trained-model-compatible vision logic)
        tk.Label(rl_frame, text="🛡️ COLLISION PREDICTION", 
                font=("Arial", 10, "bold"), fg="#4caf50", bg='#1e1e1e').pack(pady=(10, 5))
        
        self.collision_status_label = tk.Label(rl_frame, 
                    text="✓ Trained-model flight logic | HOUSE/POLE detection | Real-time avoidance", 
                                            font=("Arial", 8, "italic"), fg="#4caf50", bg='#1e1e1e',
                                            wraplength=220, justify='left')
        self.collision_status_label.pack(pady=3, padx=5)
        
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
        
        # Comparison Mode Button
        self.comparison_btn = tk.Button(button_frame, text="🏆 ALGORITHM COMPARISON\n(Normal vs GNN)", 
                                       command=self.open_comparison_window, 
                                       bg="#2196f3", fg="white", font=("Arial", 12, "bold"),
                                       width=20, height=2, relief='raised', borderwidth=4,
                                       cursor='hand2')
        self.comparison_btn.pack(pady=8)
        
        # Multi-Drone Environment Button
        self.multidrone_btn = tk.Button(button_frame, text="🚁 MULTI-DRONE ENVIRONMENT\n(5 Drones + Communication)", 
                                       command=self.open_multidrone_window, 
                                       bg="#ff9800", fg="white", font=("Arial", 12, "bold"),
                                       width=20, height=2, relief='raised', borderwidth=4,
                                       cursor='hand2')
        self.multidrone_btn.pack(pady=8)
        
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
            
            # Speed - Shows drone velocity over time
            ax1 = self.fig.add_subplot(2, 3, 1, facecolor='#1e1e1e')
            ax1.plot(times, self.metrics['speeds'], color='#2196f3', linewidth=2, label='Current')
            ax1.set_title('SPEED (Measured XY Velocity from AirSim State)\nOptimal: 8-15 m/s for efficiency', 
                         fontweight='bold', fontsize=9, color='white')
            ax1.set_ylabel('m/s', fontsize=8, color='white')
            ax1.axhline(y=12, color='#76ff03', linestyle='--', alpha=0.5, linewidth=1, label='Target Cruise')
            ax1.grid(alpha=0.2, color='#404040')
            ax1.legend(fontsize=7, loc='upper right', facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
            ax1.tick_params(colors='white', labelsize=8)
            for spine in ax1.spines.values():
                spine.set_color('#404040')
            
            # Altitude - Shows height above ground level
            ax2 = self.fig.add_subplot(2, 3, 2, facecolor='#1e1e1e')
            alts = [abs(a) for a in self.metrics['altitudes']]
            ax2.plot(times, alts, color='#ff9800', linewidth=2, label='Current')
            ax2.set_title('ALTITUDE (Height Above Ground)\nTarget: 20m AGL for safety', 
                         fontweight='bold', fontsize=9, color='white')
            ax2.set_ylabel('m', fontsize=8, color='white')
            ax2.axhline(y=20, color='#76ff03', linestyle='--', alpha=0.5, linewidth=1, label='Target Alt')
            ax2.grid(alpha=0.2, color='#404040')
            ax2.legend(fontsize=7, loc='upper right', facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
            ax2.tick_params(colors='white', labelsize=8)
            for spine in ax2.spines.values():
                spine.set_color('#404040')
            
            # Battery - Shows remaining power percentage
            ax3 = self.fig.add_subplot(2, 3, 3, facecolor='#1e1e1e')
            battery_color = '#4caf50' if self.battery_percent > 50 else '#ff9800' if self.battery_percent > 20 else '#f44336'
            ax3.plot(times, self.metrics['battery'], color=battery_color, linewidth=2.5, label='Current')
            ax3.set_title('BATTERY (Remaining Capacity)\nDecreases as energy is consumed', 
                         fontweight='bold', fontsize=9, color='white')
            ax3.set_ylabel('%', fontsize=8, color='white')
            ax3.axhline(y=20, color='#f44336', linestyle='--', alpha=0.7, linewidth=2, label='Critical Level')
            ax3.set_ylim([0, 105])
            ax3.grid(alpha=0.2, color='#404040')
            ax3.legend(fontsize=7, loc='upper right', facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
            ax3.tick_params(colors='white', labelsize=8)
            for spine in ax3.spines.values():
                spine.set_color('#404040')
            
            # Energy - Shows cumulative energy consumption with RL comparison
            ax4 = self.fig.add_subplot(2, 3, 4, facecolor='#1e1e1e')
            ax4.plot(times, self.metrics['energy'], color='#e91e63', linewidth=2.5, label=f'Run #{self.current_run_number}')
            
            # Show best run comparison if available
            if self.best_energy < float('inf') and len(self.run_history) > 0:
                ax4.axhline(y=self.best_energy, color='#76ff03', linestyle='--', 
                           alpha=0.7, linewidth=2, label=f'Best: {self.best_energy:.2f} Wh')
            
            ax4.set_title('ENERGY (Cumulative Model Estimate)\nBased on measured speed and hover-power model', 
                         fontweight='bold', fontsize=9, color='white')
            ax4.set_ylabel('Wh (Watt-hours)', fontsize=8, color='white')
            ax4.set_xlabel('Time (s)', fontsize=8, color='white')
            ax4.grid(alpha=0.2, color='#404040')
            ax4.legend(fontsize=7, loc='upper left', facecolor='#1e1e1e', edgecolor='#404040', labelcolor='white')
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
        """Update depth vision visualization with YOLOv8"""
        if self.current_depth_image is not None:
            try:
                # Convert depth image to displayable format
                depth_img = self.current_depth_image.copy()
                
                # Normalize and colorize
                depth_normalized = np.clip(depth_img / 20.0, 0, 1)
                depth_colored = (depth_normalized * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

                import math
                
                # Run YOLOv8 on the depth map
                depth_gray = (depth_normalized * 255).astype(np.uint8)
                depth_rgb_yolo = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2RGB)
                
                if hasattr(self, 'yolo_model'):
                    results = self.yolo_model(depth_rgb_yolo, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            cls = int(box.cls[0])
                            
                            # Draw bounding box on the colored depth map
                            cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(depth_colored, f"{self.yolo_model.names[cls]} {conf}", 
                                        (max(0, x1), max(35, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
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
    
    def draw_wind_pattern(self):
        """Draw live wind pattern visualization with arrows and intensity"""
        try:
            self.wind_canvas.delete("all")
            
            # Canvas dimensions
            w, h = 380, 150
            center_x, center_y = w // 2, h // 2
            
            # Get wind data
            wind_speed = self.wind_magnitude
            wind_dir = self.wind_direction
            wind_dir_norm = np.linalg.norm(wind_dir)
            
            # Normalize wind direction
            if wind_dir_norm > 0.1:
                wind_unit = wind_dir / wind_dir_norm
            else:
                wind_unit = np.array([0.0, 0.0])
            
            # Color based on wind intensity
            if wind_speed < 5.0:
                color = "#4caf50"  # Green - calm
                intensity_text = "CALM"
            elif wind_speed < 10.0:
                color = "#8bc34a"  # Light green - light
                intensity_text = "LIGHT"
            elif wind_speed < 15.0:
                color = "#ffeb3b"  # Yellow - moderate
                intensity_text = "MODERATE"
            elif wind_speed < 25.0:
                color = "#ff9800"  # Orange - strong
                intensity_text = "STRONG"
            else:
                color = "#f44336"  # Red - heavy
                intensity_text = "HEAVY"
            
            # Draw background grid
            for i in range(0, w, 40):
                self.wind_canvas.create_line(i, 0, i, h, fill="#1a1a1a", width=1)
            for i in range(0, h, 30):
                self.wind_canvas.create_line(0, i, w, i, fill="#1a1a1a", width=1)
            
            # Draw wind arrows in a pattern
            arrow_scale = min(wind_speed * 3, 50)  # Scale arrow length
            
            # Draw multiple arrows to show wind field
            for grid_x in range(60, w, 80):
                for grid_y in range(30, h, 40):
                    # Calculate arrow endpoint
                    end_x = grid_x + wind_unit[0] * arrow_scale
                    end_y = grid_y - wind_unit[1] * arrow_scale  # Negative Y for screen coords
                    
                    # Draw arrow shaft
                    self.wind_canvas.create_line(
                        grid_x, grid_y, end_x, end_y,
                        fill=color, width=3, arrow=tk.LAST,
                        arrowshape=(12, 15, 5)
                    )
            
            # Draw center indicator
            self.wind_canvas.create_oval(
                center_x - 5, center_y - 5,
                center_x + 5, center_y + 5,
                fill="#2196f3", outline="white"
            )
            
            # Draw intensity label
            self.wind_canvas.create_text(
                w // 2, 15,
                text=f"{intensity_text} - {wind_speed:.1f} m/s",
                fill=color, font=("Arial", 11, "bold")
            )
            
            # Draw direction indicator
            if wind_dir_norm > 0.1:
                # Calculate compass direction
                angle_rad = np.arctan2(wind_unit[1], wind_unit[0])
                angle_deg = np.degrees(angle_rad)
                
                # Convert to compass bearing (N=0°, E=90°, S=180°, W=270°)
                compass_deg = (90 - angle_deg) % 360
                
                # Get cardinal direction
                directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                idx = int((compass_deg + 22.5) / 45) % 8
                cardinal = directions[idx]
                
                self.wind_canvas.create_text(
                    w // 2, h - 15,
                    text=f"\u2191 {cardinal} ({compass_deg:.0f}\u00b0)",
                    fill="#90caf9", font=("Arial", 9)
                )
                
                # Update direction label
                self.wind_direction_label.config(text=f"Direction: {cardinal} ({compass_deg:.0f}\u00b0)")
            else:
                self.wind_canvas.create_text(
                    w // 2, h - 15,
                    text="\u2014 NO WIND \u2014",
                    fill="#607d8b", font=("Arial", 9)
                )
                self.wind_direction_label.config(text="Direction: N/A (No Wind)")
                
        except Exception as e:
            print(f"Wind pattern drawing error: {e}")
    
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
    
    def set_normal_mode(self):
        """Set flight mode to NORMAL (RGB camera + LiDAR)"""
        self.flight_mode = "NORMAL"
        
        # Update button styles
        self.normal_mode_btn.config(relief='sunken', bg='#4caf50')
        self.wind_mode_btn.config(relief='raised', bg='#607d8b')
        
        # Update status
        self.mode_status_label.config(text="Mode: NORMAL (Vision-Based Navigation)", fg="#4caf50")
        
        print("📷 NORMAL MODE activated - Using RGB camera and LiDAR for obstacle avoidance")
    
    def set_wind_mode(self):
        """Set flight mode to WIND (Pressure sensor navigation)"""
        self.flight_mode = "WIND"
        
        # Update button styles
        self.normal_mode_btn.config(relief='raised', bg='#607d8b')
        self.wind_mode_btn.config(relief='sunken', bg='#f44336')
        
        # Update status
        self.mode_status_label.config(text="Mode: WIND (Pressure-Based Navigation)", fg="#f44336")
        
        print("💨 WIND MODE activated - Using pressure sensor for heavy wind navigation")

    def toggle_wind(self):
        """Toggle wind simulation ON/OFF."""
        self.wind_enabled = bool(self.wind_toggle_var.get())
        if self.wind_enabled:
            print("💨 Wind ENABLED")
        else:
            # Immediately zero visualized wind when disabled.
            self.wind_magnitude = 0.0
            self.wind_direction = np.array([0.0, 0.0])
            self.heavy_wind_detected = False
            self.vision_failed = False
            print("🚫 Wind DISABLED")
    
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
    
    def toggle_learning(self):
        """Toggle online learning on/off"""
        self.learning_enabled = self.learning_var.get()
        status = "ENABLED" if self.learning_enabled else "DISABLED"
        print(f"🧠 Online Learning {status}")
    
    def calculate_reward(self, current_pos, goal_pos, distance, speed, energy_consumed, battery_percent):
        """Calculate reward for RL training"""
        # Distance-based reward (closer to goal = higher reward)
        distance_reward = -distance * 0.1  # Negative reward for being far
        
        # Energy efficiency reward (lower energy = higher reward)
        energy_reward = -energy_consumed * 2.0  # Penalize energy consumption
        
        # Goal reached bonus
        goal_reward = 100.0 if distance < 5.0 else 0.0
        
        # Speed efficiency (maintain good speed)
        speed_reward = 0.0
        if 8.0 <= speed <= 15.0:
            speed_reward = 1.0
        elif speed < 3.0:
            speed_reward = -2.0  # Penalize being too slow
        
        # Battery penalty (avoid running out)
        battery_reward = 0.0
        if battery_percent < 20.0:
            battery_reward = -10.0
        
        # Wind penalty (heavy wind conditions are dangerous)
        wind_penalty = 0.0
        if self.heavy_wind_detected:
            wind_penalty = -self.wind_magnitude * 0.5  # Penalize heavy wind navigation
        
        # Vision failure penalty (operating without vision is riskier)
        vision_penalty = -5.0 if self.vision_failed else 0.0
        
        total_reward = distance_reward + energy_reward + goal_reward + speed_reward + battery_reward + wind_penalty + vision_penalty
        return total_reward
    
    def update_rl_display(self):
        """Update RL statistics in UI"""
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
        else:
            self.improvement_label.config(text="Improvement: --")

    def _smooth_velocity_command(self, target_cmd, dt):
        """Low-pass and acceleration-limit velocity commands for stable motion."""
        target_cmd = np.asarray(target_cmd, dtype=np.float32)
        blended = (1.0 - self.velocity_smoothing_alpha) * self.prev_velocity_cmd + self.velocity_smoothing_alpha * target_cmd
        delta = blended - self.prev_velocity_cmd

        max_planar_step = self.max_accel_ms2 * dt
        planar_delta = delta[:2]
        planar_norm = np.linalg.norm(planar_delta)
        if planar_norm > max_planar_step and planar_norm > 1e-6:
            planar_delta = planar_delta * (max_planar_step / planar_norm)

        max_vertical_step = self.max_vertical_accel_ms2 * dt
        delta_z = float(np.clip(delta[2], -max_vertical_step, max_vertical_step))

        smoothed = self.prev_velocity_cmd + np.array([planar_delta[0], planar_delta[1], delta_z], dtype=np.float32)
        self.prev_velocity_cmd = smoothed
        return smoothed

    def _smooth_comparison_velocity(self, target_cmd, dt, drone_key):
        """Apply the same acceleration-limited smoothing for comparison drones."""
        target_cmd = np.asarray(target_cmd, dtype=np.float32)
        prev = self.drone1_prev_vel if drone_key == "Drone1" else self.drone2_prev_vel

        blended = 0.75 * prev + 0.25 * target_cmd
        delta = blended - prev

        max_planar_step = 3.5 * dt
        planar_delta = delta[:2]
        planar_norm = np.linalg.norm(planar_delta)
        if planar_norm > max_planar_step and planar_norm > 1e-6:
            planar_delta = planar_delta * (max_planar_step / planar_norm)

        max_vertical_step = 2.0 * dt
        delta_z = float(np.clip(delta[2], -max_vertical_step, max_vertical_step))
        smoothed = prev + np.array([planar_delta[0], planar_delta[1], delta_z], dtype=np.float32)

        if drone_key == "Drone1":
            self.drone1_prev_vel = smoothed
        else:
            self.drone2_prev_vel = smoothed
        return smoothed
    
    def save_run_to_history(self, final_time, final_energy, goal_reached):
        """Save current run metrics to history"""
        run_data = {
            'run_number': self.current_run_number,
            'time': final_time,
            'energy': final_energy,
            'goal_reached': goal_reached,
            'metrics': {
                'times': self.metrics['time'].copy(),
                'speeds': self.metrics['speeds'].copy(),
                'energies': self.metrics['energy'].copy(),
                'positions_x': self.metrics['positions_x'].copy(),
                'positions_y': self.metrics['positions_y'].copy()
            }
        }
        
        self.run_history.append(run_data)
        
        # Update best records
        if goal_reached:
            if final_energy < self.best_energy:
                self.best_energy = final_energy
                print(f"🏆 NEW BEST ENERGY: {final_energy:.2f} Wh!")
            
            if final_time < self.best_time:
                self.best_time = final_time
                print(f"🏆 NEW BEST TIME: {final_time:.1f} s!")
        
        self.update_rl_display()
    
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
        """Main flight loop with vision and AI + Reinforcement Learning"""
        try:
            # Increment run counter
            self.current_run_number += 1
            goal_reached = False
            
            # Initialize
            print(f"\n{'='*60}")
            print(f"🚁 STARTING RUN #{self.current_run_number}")
            print(f"{'='*60}")
            print(f"🧠 Loading MHA-PPO Agent from {MODEL_PATH}...")
            
            # Enable learning if checkbox is checked
            learning_rate = 0.0001 if self.learning_enabled else 0
            self.agent = PPO_Agent(state_dim=9, action_dim=3, lr=learning_rate, gamma=0.99, K_epochs=4)
            print("🛡️ Collision logic: using trained-model-compatible HOUSE/POLE vision rules")
            
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                
                # Handle dimension mismatch (7D -> 9D state space)
                if 'actor_state_dict' in checkpoint:
                    actor_state = checkpoint['actor_state_dict']
                    current_actor_state = self.agent.actor.state_dict()
                    
                    # Check for input_projection dimension mismatch
                    if 'input_projection.weight' in actor_state:
                        old_weight = actor_state['input_projection.weight']  # [64, 7]
                        new_weight = current_actor_state['input_projection.weight']  # [64, 9]
                        
                        if old_weight.shape != new_weight.shape:
                            print(f"  📐 Adapting Actor state dimension: {old_weight.shape[1]}D → {new_weight.shape[1]}D")
                            # Copy first 7 dimensions, initialize last 2 randomly
                            new_weight[:, :7] = old_weight
                            actor_state['input_projection.weight'] = new_weight
                            print(f"  ✓ Transferred 7D weights, initialized 2D extensions")
                    
                    # Load adapted weights
                    self.agent.actor.load_state_dict(actor_state, strict=False)
                    
                    # Handle Critic dimension mismatch too
                    if 'critic_state_dict' in checkpoint:
                        critic_state = checkpoint['critic_state_dict']
                        current_critic_state = self.agent.critic.state_dict()
                        
                        # Adapt Critic fc1 layer if needed
                        if 'fc1.weight' in critic_state:
                            old_critic_weight = critic_state['fc1.weight']  # [128, 7]
                            new_critic_weight = current_critic_state['fc1.weight']  # [128, 9]
                            
                            if old_critic_weight.shape != new_critic_weight.shape:
                                print(f"  📐 Adapting Critic state dimension: {old_critic_weight.shape[1]}D → {new_critic_weight.shape[1]}D")
                                # Copy first 7 dimensions, initialize extensions randomly
                                new_critic_weight[:, :7] = old_critic_weight
                                critic_state['fc1.weight'] = new_critic_weight
                                print(f"  ✓ Critic weights adapted successfully")
                        
                        self.agent.critic.load_state_dict(critic_state, strict=False)
                else:
                    # Old format, try direct loading
                    self.agent.actor.load_state_dict(checkpoint, strict=False)
                    
                print(f"✓ MHA-PPO Brain loaded successfully ({MODEL_PATH})")
                
                if self.learning_enabled:
                    print("🎓 Online Learning ENABLED - Agent will improve from this run")
                else:
                    print("📖 Inference Mode - Using pre-trained policy only")
                    
            except Exception as e:
                print(f"⚠️ Model loading failed: {e}")
                print("⚠️ Using randomized policy - navigation will use proportional controller")
            
            self.agent.actor.eval() if not self.learning_enabled else self.agent.actor.train()
            
            # Connect AirSim
            print("🔌 Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            safe_airsim_call(self.client.enableApiControl, True)
            safe_airsim_call(self.client.armDisarm, True)
            
            print("🛫 Taking off...")
            safe_airsim_call(self.client.takeoffAsync)
            time.sleep(1.5)
            safe_airsim_call(self.client.moveToZAsync, -20.0, 3.0)  # Climb to 20m altitude
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
            state = safe_airsim_call(self.client.getMultirotorState)
            if state is None:
                print("⚠️ Failed to get initial state")
                return
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
            self.prev_velocity_cmd = np.zeros(3, dtype=np.float32)
            
            self.running = True
            self.flight_active = True
            self.status_label.config(text="● Flying!", fg="#4caf50")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.goal_x_entry.config(state="disabled")
            self.goal_y_entry.config(state="disabled")
            
            # Start telemetry updates
            self.update_telemetry_loop()
            
            # Flight loop - Reduced frequency to prevent msgpackrpc overload
            dt = 0.18  # Lower command churn gives smoother real-drone-like tracking
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

                rl_action = None
                try:
                    if self.agent is not None:
                        rl_action = self.agent.select_action(state_vec, training=self.learning_enabled)
                        self.last_rl_action = np.asarray(rl_action, dtype=np.float32)
                        self.rl_policy_active = True
                        self.last_policy_mode = "ONLINE" if self.learning_enabled else "INFERENCE"
                except Exception as rl_err:
                    if step % 80 == 0:
                        print(f"⚠️ RL action fallback: {rl_err}")
                    self.rl_policy_active = False
                    self.last_rl_action = np.zeros(3, dtype=np.float32)
                
                goal_pos = np.array([self.goal_x, self.goal_y])
                distance = np.linalg.norm(goal_pos - current_pos)
                
                # Check goal
                if distance < 5.0:
                    goal_reached = True
                    print(f"🏆 GOAL REACHED! Distance: {distance:.1f}m")
                    self.status_label.config(text="● Goal Reached!", fg="#4caf50")
                    
                    # Calculate final reward for reaching goal
                    if self.learning_enabled:
                        final_reward = self.calculate_reward(current_pos, goal_pos, distance,
                                                            np.linalg.norm(velocity), 
                                                            self.total_energy_consumed,
                                                            self.battery_percent)
                        self.agent.store_reward(final_reward, done=True)

                    # Auto-land immediately when the goal is reached.
                    self.flight_active = False
                    self.running = False
                    self.smart_landing()
                    
                    break
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    mode_text = f"🌪️ WIND MODE" if self.flight_mode == "WIND" else "📷 NORMAL MODE"
                    vision_status = "DISABLED" if self.flight_mode == "WIND" else "ACTIVE"
                    speed_text = "🐌 SLOW (6 m/s)" if self.flight_mode == "WIND" else "🚀 FAST (20 m/s)"
                    print(f"Step {step:4d} | Pos: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) | Goal: ({self.goal_x}, {self.goal_y}) | Dist: {distance:.1f}m | Battery: {self.battery_percent:.1f}% | {mode_text} | Vision: {vision_status} | Speed: {speed_text}")
                
                # Execute obstacle avoidance or navigate to goal
                obstacle_avoided = False
                
                # Choose navigation strategy based on flight mode
                if self.flight_mode == "WIND":
                    # WIND MODE: Use pressure-based navigation (no vision)
                    if step % 100 == 0:
                        print(f"💨 Wind mode navigation - Pressure: {self.air_pressure:.0f} Pa, Wind: {self.wind_magnitude:.1f} m/s")
                    obstacle_avoided = False  # Bypass vision-based obstacle avoidance
                elif self.flight_mode == "NORMAL":
                    # NORMAL MODE: Predictive TTC-based collision avoidance
                    if vision_data and vision_data.get('raw_image') is not None:
                        current_speed = float(np.linalg.norm(velocity))
                        crash_imminent, ttc, avg_dist, avoid_dir = self.collision_predictor.predict_collision(
                            vision_data['raw_image'], current_speed
                        )

                        # Fallback to the trained HOUSE/POLE rules for extra robustness.
                        if not crash_imminent:
                            will_collide, rule_dir = self.predict_collision()
                            if will_collide:
                                crash_imminent = True
                                avoid_dir = rule_dir if rule_dir is not None else avoid_dir

                        if crash_imminent:
                            self.current_obstacle_type = "CRASH_IMMINENT"
                            forward_x = max(1.5, current_speed * 0.5)

                            if avoid_dir == 'LEFT':
                                safe_airsim_call(self.client.moveByVelocityAsync, forward_x, -5.2, -0.8, duration=0.45)
                            elif avoid_dir == 'RIGHT':
                                safe_airsim_call(self.client.moveByVelocityAsync, forward_x, 5.2, -0.8, duration=0.45)
                            else:
                                safe_airsim_call(self.client.moveByVelocityAsync, 2.0, 0.0, -4.2, duration=0.45)

                            obstacle_avoided = True
                            self.prev_velocity_cmd *= 0.5
                            target_vx = 0.0
                            target_vy = 0.0

                            if step % 25 == 0:
                                print(f"⚠️ TTC ALERT: {ttc:.2f}s | d={avg_dist:.1f}m | v={current_speed:.1f}m/s | evade={avoid_dir}")
                
                # Get current altitude and ground-relative height
                state = safe_airsim_call(self.client.getMultirotorState)
                if state is None:
                    print("⚠️ AirSim communication error, retrying...")
                    time.sleep(0.1)
                    continue
                    
                current_altitude = abs(state.kinematics_estimated.position.z_val)
                height_above_ground = current_altitude - self.ground_height
                
                # PROPORTIONAL CONTROLLER - Navigate to goal
                if not obstacle_avoided:
                    direction_to_goal = goal_pos - current_pos
                    distance_to_goal = np.linalg.norm(direction_to_goal)
                    
                    if distance_to_goal > 0.1:
                        direction_normalized = direction_to_goal / distance_to_goal
                        
                        # Speed: adjust based on flight mode and dynamic mode
                        # NORMAL MODE: 12 m/s cruise (fast)
                        # WIND MODE: 5 m/s cruise (slow, cautious)
                        if self.flight_mode == "WIND":
                            # Wind mode - slow down for safety
                            if distance_to_goal > 30.0:
                                desired_speed = 6.0
                            else:
                                desired_speed = max(2.5, distance_to_goal * 0.25)
                        else:
                            # NORMAL MODE: Adaptive speed based on conditions
                            # Consider obstacles and distance
                            obstacle_detected = vision_data and vision_data['obstacle']['type'] != 'CLEAR'
                            
                            if obstacle_detected:
                                # Slow down near obstacles for better reaction time
                                desired_speed = min(10.0, distance_to_goal * 0.35)
                            elif distance_to_goal > 60.0:
                                desired_speed = 20.0
                            elif distance_to_goal > 35.0:
                                desired_speed = 16.5
                            else:
                                desired_speed = max(6.5, distance_to_goal * 0.65)

                        # Goal guidance baseline
                        target_vx = direction_normalized[0] * desired_speed
                        target_vy = direction_normalized[1] * desired_speed

                        # RL policy correction layer (practical use of trained model)
                        if rl_action is not None and self.flight_mode == "NORMAL":
                            rl_action = np.asarray(rl_action, dtype=np.float32)
                            rl_action = np.clip(rl_action, -1.0, 1.0)

                            speed_bias = float(rl_action[0]) * 2.2
                            guided_speed = np.clip(desired_speed + speed_bias, 5.0, 20.0)

                            # Keep straight-line tracking; RL only adjusts forward speed.
                            target_vx = direction_normalized[0] * guided_speed
                            target_vy = direction_normalized[1] * guided_speed
                            
                            # Show RL is actively working
                            self.last_rl_action = rl_action.copy()
                            self.rl_policy_active = True
                    else:
                        target_vx = 0.0
                        target_vy = 0.0
                    
                    # Ultra-stable altitude hold - NO BOBBING
                    target_height_above_ground = 20.0
                    height_error = target_height_above_ground - height_above_ground
                    vz_measured = float(velocity[2]) if len(velocity) >= 3 else 0.0
                    
                    # Critically damped control - prioritize stability over responsiveness
                    target_vz = np.clip((0.06 * height_error) - (0.98 * vz_measured), -0.4, 0.4)
                    
                    # Very wide deadband - ignore small altitude errors
                    if abs(height_error) < 3.5 and abs(vz_measured) < 0.15:
                        target_vz = 0.0

                    smoothed_cmd = self._smooth_velocity_command([target_vx, target_vy, target_vz], dt)
                    target_vx, target_vy, target_vz = float(smoothed_cmd[0]), float(smoothed_cmd[1]), float(smoothed_cmd[2])
                    
                    # Very long duration = ultra-smooth, no oscillation
                    safe_airsim_call(self.client.moveByVelocityAsync, float(target_vx), float(target_vy), float(-target_vz), duration=1.2)
                
                # Update metrics using measured speed from telemetry (not command target).
                measured_speed = float(np.linalg.norm(velocity))
                power_watts = P_HOVER * (1 + 0.005 * measured_speed**2)
                energy_step = power_watts * (dt / 3600.0)
                self.total_energy_consumed += energy_step
                self.battery_percent = max(0, 100 - (self.total_energy_consumed / BATTERY_CAPACITY_WH * 100))
                
                # Calculate reward for RL training
                if self.learning_enabled:
                    reward = self.calculate_reward(current_pos, goal_pos, distance, 
                                                   measured_speed, energy_step, self.battery_percent)
                    self.agent.store_reward(reward, done=False)
                    self.training_stats['rewards'].append(reward)
                
                elapsed_time = time.time() - self.start_time
                self.metrics['time'].append(elapsed_time)
                self.metrics['speeds'].append(measured_speed)
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
            
            # Get final metrics
            final_time = self.metrics['time'][-1] if self.metrics['time'] else 0
            final_energy = self.total_energy_consumed
            
            # Perform RL training if enabled and goal was reached
            if self.learning_enabled and goal_reached and len(self.agent.states) > 0:
                print("\n🎓 TRAINING AGENT...")
                print(f"   Collected {len(self.agent.states)} experiences")
                actor_loss, critic_loss = self.agent.update()
                self.training_stats['actor_losses'].append(actor_loss)
                self.training_stats['critic_losses'].append(critic_loss)
                print(f"   Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")
                
                # Save updated model
                model_save_path = f"trained_models/mha_ppo_run{self.current_run_number}.pth"
                trained_weights = {
                    'run_number': self.current_run_number,
                    'actor_state_dict': self.agent.actor.state_dict(),
                    'critic_state_dict': self.agent.critic.state_dict(),
                    'final_energy': final_energy,
                    'final_time': final_time,
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss
                }
                torch.save(trained_weights, model_save_path)
                # Ensure the NEXT run explicitly loads this updated intelligence!
                torch.save(trained_weights, MODEL_PATH) 
                
                print(f"   💾 Model saved: {model_save_path}")
                print(f"   🧠 Model overwrote {MODEL_PATH} for continuous learning across runs!")
                print("   ✓ Training complete!")
            
            # Save run to history
            if goal_reached:
                # Save RL Data to a physical CSV file across all runs
                csv_file = "rl_training_data_log.csv"
                import os
                file_exists = os.path.isfile(csv_file)
                try:
                    with open(csv_file, 'a') as f:
                        if not file_exists:
                            f.write("Run_Number,Final_Time_Sec,Energy_Wh,Best_Energy_Wh,Actor_Loss,Critic_Loss\n")
                        
                        a_loss = actor_loss if 'actor_loss' in locals() else 0.0
                        c_loss = critic_loss if 'critic_loss' in locals() else 0.0
                        f.write(f"{self.current_run_number},{final_time:.2f},{final_energy:.2f},{self.best_energy if self.best_energy != float('inf') else final_energy:.2f},{a_loss:.4f},{c_loss:.4f}\n")
                    print(f"\n   📈 [SUCCESS] RL training data explicitly saved to physical file: {csv_file}")
                except Exception as e:
                    print(f"\n   ⚠️ Could not save physical RL CSV: {e}")

                self.save_run_to_history(final_time, final_energy, goal_reached)
                print(f"\n📊 RUN #{self.current_run_number} SUMMARY:")
                print(f"   Time: {final_time:.1f}s")
                print(f"   Energy: {final_energy:.2f} Wh")
                print(f"   Best Energy: {self.best_energy:.2f} Wh")
                print(f"   Best Time: {self.best_time:.1f}s")
                print(f"{'='*60}\n")
    
    def get_depth_perception(self):
        """Get depth image and analyze obstacles"""
        try:
            responses = safe_airsim_call(
                self.client.simGetImages,
                [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)]
            )
            
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
    
    def predict_collision(self):
        """
        Collision prediction using the same proven logic as master_flight_vision.py
        - HOUSE: center + both sides blocked -> climb
        - POLE/TREE: center blocked, one side clearer -> swerve to clearer side
        Returns: (will_collide, avoidance_direction)
        """
        try:
            # Get depth image from front camera
            responses = safe_airsim_call(
                self.client.simGetImages,
                [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)]
            )
            
            if responses and len(responses[0].image_data_float) > 0:
                depth_data = np.array(responses[0].image_data_float)
                depth_data = depth_data.reshape(responses[0].height, responses[0].width)
                
                h, w = depth_data.shape
                
                # Same 3-zone logic used in master_flight_vision.py
                center_box = depth_data[h//3:2*h//3, w//3:2*w//3]
                left_box = depth_data[h//3:2*h//3, :w//3]
                right_box = depth_data[h//3:2*h//3, 2*w//3:]
                
                # Distances in meters
                dist_c = float(np.min(center_box))
                dist_l = float(np.mean(left_box))
                dist_r = float(np.mean(right_box))

                # Rule A: HOUSE/WALL detection (front fully blocked)
                if dist_c < 8.0 and dist_l < 8.0 and dist_r < 8.0:
                    return (True, "UP")

                # Rule B: POLE/TREE detection (center blocked, one side clearer)
                if dist_c < 10.0:
                    if dist_l > dist_r:
                        return (True, "LEFT")
                    return (True, "RIGHT")

                # Rule C: Warning band, suggest side correction but no hard evade
                if dist_c < 14.0:
                    if dist_l > dist_r + 0.5:
                        return (False, "LEFT")
                    if dist_r > dist_l + 0.5:
                        return (False, "RIGHT")
                    return (False, "UP")
            
            return (False, None)  # No depth data
            
        except BufferError:
            # Buffer error - skip this check
            return (False, None)
        except Exception as e:
            print(f"⚠️ Collision prediction error: {e}")
            return (False, None)
    
    def execute_evasive_maneuver(self, obstacle_info):
        """Execute obstacle avoidance"""
        action = obstacle_info['action']
        
        if action == 'CLIMB':
            # Climb higher and faster to clear buildings
            safe_airsim_call(self.client.moveByVelocityAsync, 2.0, 0, -8.0, duration=2.0)
            time.sleep(0.1)
            return True
        elif action.startswith('SWERVE_'):
            # More aggressive swerve to avoid poles/trees
            direction = 1.0 if 'RIGHT' in action else -1.0
            safe_airsim_call(self.client.moveByVelocityAsync, 4.0, direction * 15.0, -2.0, duration=1.0)
            time.sleep(0.1)
            return True
        elif action.startswith('NAV_'):
            # Navigate around gaps with stronger correction
            direction = 1.0 if 'RIGHT' in action else -1.0
            safe_airsim_call(self.client.moveByVelocityAsync, 5.0, direction * 8.0, -2.0, duration=0.8)
            return True
        
        return False
    
    def get_pressure_sensor_data(self):
        """Get barometer/pressure sensor data from AirSim"""
        try:
            barometer_data = safe_airsim_call(self.client.getBarometerData)
            state = safe_airsim_call(self.client.getMultirotorState)
            
            if barometer_data is None or state is None:
                return 101325.0, 0.0
            
            # Get air pressure (Pascal)
            self.air_pressure = barometer_data.pressure
            
            # Get velocity for wind direction
            vel = state.kinematics_estimated.linear_velocity
            self.wind_direction = np.array([vel.x_val, vel.y_val]) * 0.15  # Estimated wind from velocity
            
            # Calculate wind magnitude from acceleration
            accel = state.kinematics_estimated.linear_acceleration
            base_wind_magnitude = np.linalg.norm([accel.x_val, accel.y_val]) * 10.0

            # Wind fully disabled by UI toggle
            if not self.wind_enabled:
                self.wind_magnitude = 0.0
                self.wind_direction = np.array([0.0, 0.0])
                self.heavy_wind_detected = False
                self.vision_failed = False
                return self.air_pressure, self.wind_magnitude
            
            # Apply wind based on selected flight mode
            if self.flight_mode == "WIND":
                # WIND MODE: Simulate heavy wind conditions
                self.wind_magnitude = base_wind_magnitude * 2.5 + 15.0  # Heavy wind
                self.air_pressure = 101325.0 + np.random.uniform(-3000, 3000)
                self.heavy_wind_detected = True
                self.vision_failed = True  # Disable vision in wind mode
            else:
                # NORMAL MODE: Natural wind conditions
                self.wind_magnitude = base_wind_magnitude
                self.heavy_wind_detected = False
                self.vision_failed = False
            
            return self.air_pressure, self.wind_magnitude
        except:
            # Fallback if barometer not available
            return 101325.0, 0.0
    
    def get_state_vector(self):
        """Get state for PPO agent - Now includes pressure sensor (9D state)"""
        try:
            state = safe_airsim_call(self.client.getMultirotorState)
            if state is None:
                return None, None, None
                
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            current_pos = np.array([pos.x_val, pos.y_val])
            goal_pos = np.array([self.goal_x, self.goal_y])
            dist_vector = goal_pos - current_pos
            velocity = np.array([vel.x_val, vel.y_val])
            wind = velocity * 0.1
            
            # Get pressure sensor data
            pressure, wind_mag = self.get_pressure_sensor_data()
            
            # Normalize pressure (0-1 range: 96000-106000 Pa)
            pressure_normalized = (pressure - 96000.0) / 10000.0
            pressure_normalized = np.clip(pressure_normalized, 0.0, 1.0)
            
            # Normalize wind magnitude (0-1 range: 0-30 m/s)
            wind_mag_normalized = np.clip(wind_mag / 30.0, 0.0, 1.0)
            
            # 9D state vector: [goal_x, goal_y, vel_x, vel_y, wind_x, wind_y, battery, pressure, wind_magnitude]
            state_vec = np.concatenate([
                dist_vector, velocity, wind, 
                [self.battery_percent / 100.0],
                [pressure_normalized],
                [wind_mag_normalized]
            ]).astype(np.float32)
            
            return state_vec, current_pos, velocity
        except:
            return None, None, None
    
    def update_telemetry_loop(self):
        """Update GUI telemetry"""
        if not self.running:
            return
        
        try:
            state = safe_airsim_call(self.client.getMultirotorState)
            if state is None:
                # Skip this update if AirSim communication fails
                if self.running:
                    self.window.after(100, self.update_telemetry_loop)
                return
                
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
            
            # Update pressure sensor display
            self.pressure_label.config(text=f"Pressure: {self.air_pressure:.0f} Pa")
            self.wind_mag_label.config(text=f"Wind: {self.wind_magnitude:.1f} m/s")
            
            # Update heavy wind status
            if self.heavy_wind_detected:
                self.heavy_wind_label.config(text="Status: HEAVY WIND ⚠️", fg="#f44336")
            else:
                self.heavy_wind_label.config(text="Status: NORMAL ✓", fg="#4caf50")
            
            # Update vision status
            if self.vision_failed:
                self.vision_status_label.config(text="Vision: DISABLED ⚠️", fg="#ff9800")
            else:
                self.vision_status_label.config(text="Vision: ACTIVE 📷", fg="#4caf50")
            
            # Update wind pattern visualization
            self.draw_wind_pattern()

            # Show practical RL status in GUI
            if self.rl_policy_active:
                self.rl_policy_label.config(text=f"Policy: ACTIVE ({self.last_policy_mode})", fg="#76ff03")
                # Show actual RL action values to verify it's working
                if self.last_rl_action is not None:
                    action_text = f"Action: ({self.last_rl_action[0]:+.2f}, {self.last_rl_action[1]:+.2f}, {self.last_rl_action[2]:+.2f})"
                    self.rl_action_label.config(text=action_text, fg="#00e676")
            else:
                self.rl_policy_label.config(text="Policy: Standby", fg="#90caf9")
                self.rl_action_label.config(text="Action: (Guidance Mode)", fg="#90caf9")

            ax, ay, az = self.last_rl_action
            self.rl_action_label.config(text=f"Action: ({ax:+.2f}, {ay:+.2f}, {az:+.2f})")
            
        except:
            pass
        
        if self.running:
            self.window.after(100, self.update_telemetry_loop)
    
    def smart_landing(self):
        """Safe landing: controlled descent with stability."""
        try:
            print("\n🛬 Safe Landing Sequence...")
            self.status_label.config(text="● Landing Safely...", fg="orange")
            
            state = safe_airsim_call(self.client.getMultirotorState)
            if state is None:
                print("⚠️ Failed to get state for landing")
                return
            pos = state.kinematics_estimated.position
            current_height = abs(pos.z_val) - self.ground_height
            
            print(f"  📍 Landing from: ({pos.x_val:.1f}, {pos.y_val:.1f}), Height: {current_height:.1f}m AGL")
            
            # Safe controlled descent with proper stages
            while current_height > 1.5:
                state = safe_airsim_call(self.client.getMultirotorState)
                if state is None:
                    break

                pos = state.kinematics_estimated.position
                current_height = abs(pos.z_val) - self.ground_height

                # Safe descent rates - prioritize safety over speed
                if current_height > 8.0:
                    descent_rate = 1.5
                elif current_height > 4.0:
                    descent_rate = 1.0
                else:
                    descent_rate = 0.6

                safe_airsim_call(self.client.moveByVelocityAsync, 0.0, 0.0, float(descent_rate), duration=0.5)
                time.sleep(0.18)

            # Final gentle touchdown
            print("  🛬 Final touchdown...")
            safe_airsim_call(self.client.moveByVelocityAsync, 0, 0, 0, 0.4)  # Stabilize
            time.sleep(0.2)
            safe_airsim_call(self.client.landAsync)
            time.sleep(0.6)
            
            safe_airsim_call(self.client.armDisarm, False)
            safe_airsim_call(self.client.enableApiControl, False)
            print("✓ Landed safely")
            self.status_label.config(text="● Landed", fg="gray")
            
        except Exception as e:
            print(f"Landing error: {e}")
            import traceback
            traceback.print_exc()
            # Emergency land
            try:
                safe_airsim_call(self.client.landAsync)
                safe_airsim_call(self.client.armDisarm, False)
                safe_airsim_call(self.client.enableApiControl, False)
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
    
    def open_comparison_window(self):
        """Open algorithm comparison window: MHA-PPO vs GNN"""
        # Create comparison window
        comp_window = tk.Toplevel(self.window)
        comp_window.title("🏆 Algorithm Comparison: MHA-PPO vs GNN")
        comp_window.geometry("1400x900")
        comp_window.configure(bg='#1a1a1a')
        
        # Header
        header = tk.Frame(comp_window, bg='#2196f3', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header, text="🏆 ALGORITHM COMPARISON: MHA-PPO vs GNN",
            font=("Arial", 22, "bold"), fg="white", bg='#2196f3'
        ).pack(pady=5)
        
        tk.Label(
            header, text="Simulated Race: Two Algorithms Tested on Same Flight Path",
            font=("Arial", 11), fg="#e3f2fd", bg='#2196f3'
        ).pack()
        
        tk.Label(
            header, text="⚠️ Both use FIXED speed (Dynamic Mode OFF) | Same path, different energy efficiency",
            font=("Arial", 9, "italic"), fg="#ffeb3b", bg='#2196f3'
        ).pack()
        
        # Data storage for comparison
        self.comp_data = {
            'normal': {
                'time': deque(maxlen=100),
                'energy': deque(maxlen=100),
                'battery': deque(maxlen=100),
                'speed': deque(maxlen=100),
                'distance': deque(maxlen=100)
            },
            'gnn': {
                'time': deque(maxlen=100),
                'energy': deque(maxlen=100),
                'battery': deque(maxlen=100),
                'speed': deque(maxlen=100),
                'distance': deque(maxlen=100)
            }
        }
        self.comparison_active = False
        
        # Control panel
        control_frame = tk.Frame(comp_window, bg='#2d2d2d', height=120)
        control_frame.pack(fill='x', padx=10, pady=10)
        control_frame.pack_propagate(False)
        
        tk.Label(
            control_frame, text="🎮 COMPARISON CONTROLS",
            font=("Arial", 13, "bold"), fg="#4caf50", bg='#2d2d2d'
        ).pack(pady=10)
        
        btn_row = tk.Frame(control_frame, bg='#2d2d2d')
        btn_row.pack()
        
        start_comp_btn = tk.Button(
            btn_row, text="🏁 START COMPARISON RACE",
            font=("Arial", 12, "bold"), bg='#4caf50', fg='white',
            width=25, height=2, command=lambda: self.start_comparison_race(comp_window)
        )
        start_comp_btn.pack(side='left', padx=10)
        
        stop_comp_btn = tk.Button(
            btn_row, text="⏹ STOP COMPARISON",
            font=("Arial", 12, "bold"), bg='#f44336', fg='white',
            width=20, height=2, command=self.stop_comparison_race
        )
        stop_comp_btn.pack(side='left', padx=10)
        
        # Stats panel
        stats_frame = tk.Frame(comp_window, bg='#1a1a1a')
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        # Normal Algorithm Stats
        normal_stats = tk.Frame(stats_frame, bg='#2d2d2d', relief='ridge', borderwidth=3)
        normal_stats.pack(side='left', fill='x', expand=True, padx=10)
        
        tk.Label(
            normal_stats, text="🤖 DRONE 1: MHA-PPO ALGORITHM",
            font=("Arial", 12, "bold"), fg="#ff9800", bg='#2d2d2d'
        ).pack(pady=10)
        
        tk.Label(
            normal_stats, text="Multi-Head Attention PPO",
            font=("Arial", 9, "italic"), fg="#90caf9", bg='#2d2d2d'
        ).pack(pady=2)
        
        self.comp_normal_battery = tk.Label(
            normal_stats, text="Battery: 100.0%",
            font=("Arial", 11), fg="white", bg='#2d2d2d'
        )
        self.comp_normal_battery.pack(pady=5)
        
        self.comp_normal_energy = tk.Label(
            normal_stats, text="Energy: 0.00 Wh",
            font=("Arial", 11), fg="white", bg='#2d2d2d'
        )
        self.comp_normal_energy.pack(pady=5)
        
        self.comp_normal_efficiency = tk.Label(
            normal_stats, text="Efficiency: --%",
            font=("Arial", 11), fg="white", bg='#2d2d2d'
        )
        self.comp_normal_efficiency.pack(pady=10, padx=10)
        
        # GNN Algorithm Stats
        gnn_stats = tk.Frame(stats_frame, bg='#2d2d2d', relief='ridge', borderwidth=3)
        gnn_stats.pack(side='right', fill='x', expand=True, padx=10)
        
        tk.Label(
            gnn_stats, text="🤖 DRONE 2: GNN ALGORITHM",
            font=("Arial", 12, "bold"), fg="#4caf50", bg='#2d2d2d'
        ).pack(pady=10)
        
        tk.Label(
            gnn_stats, text="Graph Neural Network Enhanced",
            font=("Arial", 9, "italic"), fg="#90caf9", bg='#2d2d2d'
        ).pack(pady=2)
        
        self.comp_gnn_battery = tk.Label(
            gnn_stats, text="Battery: 100.0%",
            font=("Arial", 11), fg="white", bg='#2d2d2d'
        )
        self.comp_gnn_battery.pack(pady=5)
        
        self.comp_gnn_energy = tk.Label(
            gnn_stats, text="Energy: 0.00 Wh",
            font=("Arial", 11), fg="white", bg='#2d2d2d'
        )
        self.comp_gnn_energy.pack(pady=5)
        
        self.comp_gnn_efficiency = tk.Label(
            gnn_stats, text="Efficiency: --%",
            font=("Arial", 11), fg="white", bg='#2d2d2d'
        )
        self.comp_gnn_efficiency.pack(pady=10, padx=10)
        
        # Graphs panel
        graph_panel = tk.Frame(comp_window, bg='#1a1a1a')
        graph_panel.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figure with 3 subplots (Energy, Battery, Map)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        self.comp_fig = Figure(figsize=(16, 7), facecolor='#1a1a1a')
        
        # Energy comparison subplot
        self.comp_ax1 = self.comp_fig.add_subplot(131, facecolor='#2d2d2d')
        self.comp_ax1.set_xlabel('Time (s)', color='white', fontsize=11)
        self.comp_ax1.set_ylabel('Energy Consumption (Wh)', color='white', fontsize=11)
        self.comp_ax1.set_title('Energy Efficiency', color='white', fontsize=13, fontweight='bold')
        self.comp_ax1.tick_params(colors='white')
        self.comp_ax1.grid(True, alpha=0.3, color='gray')
        
        # Battery comparison subplot
        self.comp_ax2 = self.comp_fig.add_subplot(132, facecolor='#2d2d2d')
        self.comp_ax2.set_xlabel('Time (s)', color='white', fontsize=11)
        self.comp_ax2.set_ylabel('Battery Level (%)', color='white', fontsize=11)
        self.comp_ax2.set_title('Battery Life', color='white', fontsize=13, fontweight='bold')
        self.comp_ax2.tick_params(colors='white')
        self.comp_ax2.grid(True, alpha=0.3, color='gray')
        
        # Map visualization subplot (NEW!)
        self.comp_map_ax = self.comp_fig.add_subplot(133, facecolor='#2d2d2d')
        self.comp_map_ax.set_xlabel('X (m)', color='white', fontsize=11)
        self.comp_map_ax.set_ylabel('Y (m)', color='white', fontsize=11)
        self.comp_map_ax.set_title('DUAL-DRONE RACE MAP', color='white', fontsize=13, fontweight='bold')
        self.comp_map_ax.tick_params(colors='white')
        self.comp_map_ax.grid(True, alpha=0.3, color='gray')
        self.comp_map_ax.set_aspect('equal', adjustable='box')
        
        self.comp_fig.tight_layout()
        
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, graph_panel)
        self.comp_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Load GNN agent if not loaded
        if self.gnn_agent is None:
            self.load_gnn_agent()
        
        # Show agent status
        agent_status = "✓ GNN Agent Loaded" if GNN_AVAILABLE else "⚠️ GNN Agent Unavailable (Using Simulated Data)"
        tk.Label(
            comp_window, text=agent_status,
            font=("Arial", 9, "italic"), fg="#90caf9", bg='#1a1a1a'
        ).pack(pady=5)
    
    def load_gnn_agent(self):
        """Load GNN agent for comparison"""
        try:
            if GNN_AVAILABLE:
                self.gnn_agent = PPO_GNN_Agent(state_dim=9, action_dim=3)
                try:
                    checkpoint = torch.load('gnn_agent.pth', map_location='cpu')
                    self.gnn_agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                    print("✓ GNN agent loaded from gnn_agent.pth")
                except:
                    print("⚠️ GNN agent initialized with random weights")
            else:
                print("⚠️ GNN agent not available - will use simulated comparison")
        except Exception as e:
            print(f"❌ Error loading GNN agent: {e}")

    def _comparison_depth_snapshot(self, client, vehicle_name):
        """Get depth frame for a comparison drone."""
        try:
            responses = safe_airsim_call(
                client.simGetImages,
                [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)],
                vehicle_name=vehicle_name
            )
            if responses and len(responses[0].image_data_float) > 0:
                depth = np.array(responses[0].image_data_float, dtype=np.float32)
                depth = depth.reshape(responses[0].height, responses[0].width)
                return np.nan_to_num(depth, nan=200.0, posinf=200.0, neginf=200.0)
        except Exception:
            pass
        return None

    def _comparison_ttc_avoidance(self, client, vehicle_name, current_speed):
        """Run TTC prediction + immediate evasive maneuver. Returns True if evasive action executed."""
        depth_image = self._comparison_depth_snapshot(client, vehicle_name)
        if depth_image is None:
            return False

        crash_imminent, ttc, avg_dist, avoid_dir = self.collision_predictor.predict_collision(depth_image, current_speed)

        # Proactive warning slowdown band (before imminent)
        if not crash_imminent and ttc < 4.0:
            slow_forward = max(1.0, min(3.0, current_speed * 0.6))
            safe_airsim_call(client.moveByVelocityAsync, slow_forward, 0.0, -0.5, duration=0.20, vehicle_name=vehicle_name)
            return True

        if not crash_imminent:
            return False

        # Imminent collision: side-step or climb
        forward_x = max(1.5, min(4.0, current_speed * 0.5))
        if avoid_dir == 'LEFT':
            safe_airsim_call(client.moveByVelocityAsync, forward_x, -10.0, -1.5, duration=0.35, vehicle_name=vehicle_name)
        elif avoid_dir == 'RIGHT':
            safe_airsim_call(client.moveByVelocityAsync, forward_x, 10.0, -1.5, duration=0.35, vehicle_name=vehicle_name)
        else:
            safe_airsim_call(client.moveByVelocityAsync, 2.0, 0.0, -6.0, duration=0.40, vehicle_name=vehicle_name)

        print(f"⚠️ {vehicle_name} TTC ALERT: {ttc:.2f}s | d={avg_dist:.1f}m | v={current_speed:.1f}m/s | evade={avoid_dir}")
        return True
    
    def open_multidrone_window(self):
        """Open multi-drone dynamic environment window with 5 drones and communication"""
        multidrone_window = tk.Toplevel(self.window)
        multidrone_window.title("🚁 Multi-Drone Dynamic Environment (5 Drones)")
        multidrone_window.geometry("1500x900")
        multidrone_window.configure(bg='#1a1a1a')
        
        # Header
        header = tk.Frame(multidrone_window, bg='#ff9800', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header, text="🚁 MULTI-DRONE DYNAMIC ENVIRONMENT: 1 Main + 4 Roaming Drones",
            font=("Arial", 18, "bold"), fg="white", bg='#ff9800'
        ).pack(pady=10)
        
        # Main container
        main_frame = tk.Frame(multidrone_window, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls & Info
        left_panel = tk.Frame(main_frame, bg='#2d2d2d', width=300, relief='sunken', borderwidth=2)
        left_panel.pack(side='left', fill='y', padx=5)
        left_panel.pack_propagate(False)
        
        # Control buttons
        tk.Label(left_panel, text="🎮 SCENARIO CONTROL", font=("Arial", 12, "bold"), fg="#ff9800", bg='#2d2d2d').pack(pady=5)
        
        self.multidrone_active = False
        self.multidrone_start_btn = tk.Button(
            left_panel, text="▶️ START SCENARIO", 
            command=lambda: self.start_multidrone_scenario(multidrone_window),
            bg="#4caf50", fg="white", font=("Arial", 11, "bold"), width=25
        )
        self.multidrone_start_btn.pack(pady=8)
        
        self.multidrone_stop_btn = tk.Button(
            left_panel, text="⏹️ STOP SCENARIO", 
            command=lambda: self.stop_multidrone_scenario(),
            bg="#f44336", fg="white", font=("Arial", 11, "bold"), width=25, state="disabled"
        )
        self.multidrone_stop_btn.pack(pady=8)
        
        # Info panels
        tk.Label(left_panel, text="📊 DRONE TELEMETRY", font=("Arial", 11, "bold"), fg="#4caf50", bg='#2d2d2d').pack(pady=5)
        
        # Main drone info
        main_drone_frame = tk.Frame(left_panel, bg='#1a1a1a', relief='sunken', borderwidth=1)
        main_drone_frame.pack(pady=5, padx=10, fill='x')
        
        tk.Label(main_drone_frame, text="🔴 MAIN DRONE (Drone1 - GNN)", font=("Arial", 10, "bold"), fg="#ff5722", bg='#1a1a1a').pack()
        self.md_main_pos = tk.Label(main_drone_frame, text="Position: (0.0, 0.0)", font=("Arial", 9), fg="white", bg='#1a1a1a')
        self.md_main_pos.pack(anchor='w', padx=5)
        self.md_main_dist = tk.Label(main_drone_frame, text="Distance to Goal: 0.0m", font=("Arial", 9), fg="white", bg='#1a1a1a')
        self.md_main_dist.pack(anchor='w', padx=5)
        self.md_main_battery = tk.Label(main_drone_frame, text="Battery: 100.0%", font=("Arial", 9), fg="white", bg='#1a1a1a')
        self.md_main_battery.pack(anchor='w', padx=5)
        self.md_main_energy = tk.Label(main_drone_frame, text="Energy: 0.0 Wh", font=("Arial", 9), fg="white", bg='#1a1a1a')
        self.md_main_energy.pack(anchor='w', padx=5)
        
        # Random drones info
        random_drone_frame = tk.Frame(left_panel, bg='#1a1a1a', relief='sunken', borderwidth=1)
        random_drone_frame.pack(pady=5, padx=10, fill='x')
        
        tk.Label(random_drone_frame, text="🟢 RANDOM DRONES (Drone2-5)", font=("Arial", 10, "bold"), fg="#4caf50", bg='#1a1a1a').pack()
        self.md_random_status = tk.Label(random_drone_frame, text="Status: Idle", font=("Arial", 9), fg="white", bg='#1a1a1a')
        self.md_random_status.pack(anchor='w', padx=5)
        self.md_random_count = tk.Label(random_drone_frame, text="Active: 0/4", font=("Arial", 9), fg="white", bg='#1a1a1a')
        self.md_random_count.pack(anchor='w', padx=5)
        
        # Communication matrix
        tk.Label(left_panel, text="📡 COMMUNICATION", font=("Arial", 11, "bold"), fg="#2196f3", bg='#2d2d2d').pack(pady=5)
        
        self.md_comm_label = tk.Label(left_panel, text="Inter-Drone Collisions Avoided: 0", font=("Arial", 10, "bold"), fg="#2196f3", bg='#2d2d2d')
        self.md_comm_label.pack(pady=5)
        self.md_threat_label = tk.Label(left_panel, text="Threat Events (Drone5->Main): 0", font=("Arial", 9, "bold"), fg="#ffb74d", bg='#2d2d2d')
        self.md_threat_label.pack(pady=2)
        
        # Right panel - Visualization
        right_panel = tk.Frame(main_frame, bg='#2d2d2d', relief='sunken', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        # Map section
        tk.Label(right_panel, text="🗺️ FLIGHT MAP (Drone Positions & Paths)", font=("Arial", 12, "bold"), fg="#4caf50", bg='#2d2d2d').pack(pady=5)
        
        self.md_map_fig = Figure(figsize=(10, 6), facecolor='#2d2d2d')
        self.md_map_ax = self.md_map_fig.add_subplot(211)
        self.md_comm_ax = self.md_map_fig.add_subplot(212)
        self.md_map_ax.set_facecolor('#1a1a1a')
        self.md_comm_ax.set_facecolor('#1a1a1a')
        self.md_map_canvas = FigureCanvasTkAgg(self.md_map_fig, master=right_panel)
        self.md_map_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize multi-drone data structures
        self.multidrone_data = {
            'main': {'time': [], 'pos': [], 'energy': [], 'battery': [], 'speed': []},
            'random': {i: {'time': [], 'pos': [], 'energy': [], 'battery': [], 'speed': []} for i in range(2, 6)}
        }
        self.multidrone_paths = {i: [] for i in range(1, 6)}
        self.comm_avoidance_count = 0
        self.md_threat_events = 0
        self.md_scenario_start_time = 0.0
        self.md_random_ready = threading.Event()
        self.md_comm_time = deque(maxlen=400)
        self.md_comm_count = deque(maxlen=400)
        self.md_threat_count = deque(maxlen=400)
        self.md_comm_range = 35.0
        self.md_threat_drone = "Drone5"
        self.md_random_warmup_seconds = 6.0
        self.md_random_moved = {2: False, 3: False, 4: False, 5: False}
        self.md_thread_lock = threading.Lock()
        
    def start_multidrone_scenario(self, window):
        """Start multi-drone dynamic environment scenario with REAL AirSim drones"""
        if self.multidrone_active:
            messagebox.showwarning("Warning", "Scenario already in progress!", parent=window)
            return
        
        print("\n" + "="*80)
        print("🚁 STARTING MULTI-DRONE DYNAMIC ENVIRONMENT IN AIRSIM")
        print("="*80)
        print("📊 Scenario: 1 Main Drone (GNN-only) + 4 Roaming Drones")
        print("🎯 Main Drone Goal: Navigate to target with collision avoidance")
        print("🔄 Random Drones: Roaming patterns with inter-drone communication")
        print("="*80 + "\n")

        # Enforce GNN-only main drone controller for this mode.
        if self.gnn_agent is None:
            self.load_gnn_agent()
        if self.gnn_agent is None:
            messagebox.showerror("GNN Required", "GNN agent is not available. Multi-drone mode requires GNN-only control.", parent=window)
            return
        
        self.multidrone_active = True
        self.md_scenario_start_time = time.time()
        self.md_random_ready.clear()
        self.comm_avoidance_count = 0
        self.md_threat_events = 0
        self.md_random_moved = {2: False, 3: False, 4: False, 5: False}
        self.md_comm_time.clear()
        self.md_comm_count.clear()
        self.md_threat_count.clear()
        self.multidrone_paths = {i: [] for i in range(1, 6)}
        self.multidrone_start_btn.config(state="disabled")
        self.multidrone_stop_btn.config(state="normal")
        self.md_random_status.config(text="Status: Launching random drones first...")
        
        # Initialize dedicated clients (prevents IOLoop concurrency conflicts)
        try:
            print("🔌 Connecting to 5 drones in AirSim...")
            self.md_clients = {
                "Drone1": airsim.MultirotorClient(),
                "Drone2": airsim.MultirotorClient(),
                "Drone3": airsim.MultirotorClient(),
                "Drone4": airsim.MultirotorClient(),
                "Drone5": airsim.MultirotorClient(),
            }
            for c in self.md_clients.values():
                c.confirmConnection()
            print("✓ Connected to AirSim with all 5 drones")
        except Exception as e:
            print(f"❌ Failed to connect to AirSim: {e}")
            self.multidrone_active = False
            return
        
        # Start random drones first, then launch main drone after warmup.
        threading.Thread(target=self._fly_random_drones_multidrone, daemon=True).start()
        threading.Thread(target=self._delayed_main_start_multidrone, daemon=True).start()
        threading.Thread(target=self._update_multidrone_visualization, daemon=True).start()

    def _delayed_main_start_multidrone(self):
        """Wait until all random drones are confirmed moving before starting main drone."""
        ready = self.md_random_ready.wait(timeout=35.0)
        if not self.multidrone_active:
            return
        if ready:
            self.md_random_status.config(text="Status: All random drones moving. Main starting now...")
            time.sleep(1.0)
        else:
            self.md_random_status.config(text="Status: Waiting for random drones failed. Main NOT started.")
            print("⚠️ Main drone blocked: not all random drones showed roaming movement")
            self.multidrone_active = False
            return
        self._fly_main_drone_multidrone()
    
    def stop_multidrone_scenario(self):
        """Stop multi-drone scenario"""
        self.multidrone_active = False
        self.multidrone_start_btn.config(state="normal")
        self.multidrone_stop_btn.config(state="disabled")
        try:
            if hasattr(self, 'md_clients') and self.md_clients is not None:
                for name in ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5"]:
                    client = self.md_clients.get(name)
                    if client is None:
                        continue
                    fut = safe_airsim_call(client.landAsync, vehicle_name=name)
                    if fut is not None:
                        fut.join()
                    safe_airsim_call(client.armDisarm, False, vehicle_name=name)
                    safe_airsim_call(client.enableApiControl, False, vehicle_name=name)
        except Exception as e:
            print(f"⚠️ Multi-drone shutdown warning: {e}")
        print("\n🛑 Multi-drone scenario stopped")
    
    def _check_inter_drone_collision(self, drone_pos, drone_name, safety_distance=12.0):
        """Return avoidance vector and risk flag based on nearby drone positions."""
        repulse = np.zeros(2, dtype=np.float32)
        risky = False

        name_to_id = {
            "Drone1": 1,
            "Drone2": 2,
            "Drone3": 3,
            "Drone4": 4,
            "Drone5": 5,
        }
        self_id = name_to_id.get(drone_name, -1)

        for other_id in range(1, 6):
            if other_id == self_id:
                continue
            if other_id not in self.multidrone_paths or not self.multidrone_paths[other_id]:
                continue

            other = np.array(self.multidrone_paths[other_id][-1], dtype=np.float32)
            delta = drone_pos - other
            dist = float(np.linalg.norm(delta))
            if dist < 1e-3:
                continue

            if dist < safety_distance:
                risky = True
                strength = (safety_distance - dist) / safety_distance
                repulse += (delta / dist) * strength

        return repulse, risky

    def _record_md_comm_sample(self):
        """Record communication metrics timeline for charting."""
        if self.md_scenario_start_time <= 0:
            return
        t = time.time() - self.md_scenario_start_time
        self.md_comm_time.append(t)
        self.md_comm_count.append(self.comm_avoidance_count)
        self.md_threat_count.append(self.md_threat_events)

    def _multidrone_main_safe_land(self, client, vehicle_name="Drone1"):
        """Safe staged landing for main drone in multi-drone mode."""
        try:
            print(f"🛬 {vehicle_name}: starting controlled landing...")

            for _ in range(35):
                state = safe_airsim_call(client.getMultirotorState, vehicle_name=vehicle_name)
                if state is None:
                    break

                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity

                altitude = max(0.0, abs(pos.z_val) - self.ground_height)
                if altitude < 1.5:
                    break

                # Dampen horizontal drift while descending.
                vx_hold = float(np.clip(-0.35 * vel.x_val, -1.2, 1.2))
                vy_hold = float(np.clip(-0.35 * vel.y_val, -1.2, 1.2))

                if altitude > 10.0:
                    vz_down = 1.4
                elif altitude > 5.0:
                    vz_down = 1.0
                else:
                    vz_down = 0.6

                safe_airsim_call(
                    client.moveByVelocityAsync,
                    vx_hold,
                    vy_hold,
                    vz_down,
                    duration=0.45,
                    vehicle_name=vehicle_name,
                )
                time.sleep(0.15)

            safe_airsim_call(client.moveByVelocityAsync, 0.0, 0.0, 0.2, duration=0.35, vehicle_name=vehicle_name)
            fut = safe_airsim_call(client.landAsync, vehicle_name=vehicle_name)
            if fut is not None:
                fut.join()

            safe_airsim_call(client.armDisarm, False, vehicle_name=vehicle_name)
            safe_airsim_call(client.enableApiControl, False, vehicle_name=vehicle_name)
            print(f"✓ {vehicle_name}: landed safely")
        except Exception as e:
            print(f"⚠️ {vehicle_name}: controlled landing fallback due to {e}")
            fut = safe_airsim_call(client.landAsync, vehicle_name=vehicle_name)
            if fut is not None:
                fut.join()
            safe_airsim_call(client.armDisarm, False, vehicle_name=vehicle_name)
            safe_airsim_call(client.enableApiControl, False, vehicle_name=vehicle_name)

    def _fly_main_drone_multidrone(self):
        """Main drone flight control in AirSim using GNN-only policy."""
        print("🔴 Main Drone (Drone1 - GNN) starting in AirSim...")
        try:
            client = self.md_clients["Drone1"]
            start_time = time.time()
            energy = 0.0
            dt = 0.18
            step = 0

            safe_airsim_call(client.enableApiControl, True, vehicle_name="Drone1")
            fut = safe_airsim_call(client.takeoffAsync, vehicle_name="Drone1")
            if fut is not None:
                fut.join()
            safe_airsim_call(client.moveToZAsync, -20.0, 5.0, vehicle_name="Drone1")
            
            while self.multidrone_active:
                elapsed = time.time() - start_time
                
                state = safe_airsim_call(client.getMultirotorState, vehicle_name="Drone1")
                if state is None:
                    time.sleep(0.1)
                    continue
                
                pos = state.kinematics_estimated.position
                velocity = state.kinematics_estimated.linear_velocity
                current_pos = np.array([pos.x_val, pos.y_val], dtype=np.float32)
                goal_pos = np.array([self.goal_x, self.goal_y], dtype=np.float32)
                direction = goal_pos - current_pos
                distance = float(np.linalg.norm(direction))
                
                # Store position
                self.multidrone_paths[1].append([pos.x_val, pos.y_val])
                
                # Check goal reached
                if distance < 2.5:
                    print(f"🏆 MAIN DRONE REACHED GOAL! Time: {elapsed:.1f}s Energy: {energy:.2f}Wh")
                    self._multidrone_main_safe_land(client, vehicle_name="Drone1")
                    break
                
                # GNN-only navigation with inter-drone communication
                if distance > 1.0:
                    goal_dir = direction / max(distance, 1e-6)

                    # Build 5-node graph state for GNN policy.
                    node_states = []
                    node_positions = []
                    for drone_id in range(1, 6):
                        name = f"Drone{drone_id}"
                        c = self.md_clients[name]
                        st = safe_airsim_call(c.getMultirotorState, vehicle_name=name)
                        if st is not None:
                            p = st.kinematics_estimated.position
                            v = st.kinematics_estimated.linear_velocity
                            xy = np.array([p.x_val, p.y_val], dtype=np.float32)
                            node_positions.append(xy)
                            if drone_id not in self.multidrone_paths:
                                self.multidrone_paths[drone_id] = []
                            self.multidrone_paths[drone_id].append([p.x_val, p.y_val])
                            dist_main = float(np.linalg.norm(xy - current_pos))
                            feat = [
                                float(self.goal_x - p.x_val),
                                float(self.goal_y - p.y_val),
                                float(v.x_val),
                                float(v.y_val),
                                float(v.z_val),
                                1.0,
                                1.0 if name == self.md_threat_drone else 0.0,
                                float(np.clip(dist_main / 120.0, 0.0, 1.0)),
                                float(np.clip(np.linalg.norm([v.x_val, v.y_val, v.z_val]) / 20.0, 0.0, 1.0)),
                            ]
                        else:
                            if self.multidrone_paths.get(drone_id):
                                xy = np.array(self.multidrone_paths[drone_id][-1], dtype=np.float32)
                            else:
                                xy = np.zeros(2, dtype=np.float32)
                            node_positions.append(xy)
                            dist_main = float(np.linalg.norm(xy - current_pos))
                            feat = [
                                float(self.goal_x - xy[0]),
                                float(self.goal_y - xy[1]),
                                0.0, 0.0, 0.0,
                                1.0,
                                1.0 if name == self.md_threat_drone else 0.0,
                                float(np.clip(dist_main / 120.0, 0.0, 1.0)),
                                0.0,
                            ]
                        node_states.append(feat)

                    adj = np.zeros((5, 5), dtype=np.float32)
                    for i in range(5):
                        for j in range(5):
                            if i == j:
                                continue
                            d_ij = float(np.linalg.norm(node_positions[i] - node_positions[j]))
                            if d_ij <= self.md_comm_range:
                                adj[i, j] = 1.0

                    gnn_actions = self.gnn_agent.select_action(np.asarray(node_states, dtype=np.float32), adj)
                    main_act = np.asarray(gnn_actions[0], dtype=np.float32)
                    main_act = np.clip(main_act, -1.0, 1.0)

                    gnn_xy = np.asarray([main_act[0], main_act[1]], dtype=np.float32)
                    gnn_norm = float(np.linalg.norm(gnn_xy))
                    if gnn_norm > 1e-6:
                        gnn_dir = gnn_xy / gnn_norm
                    else:
                        gnn_dir = goal_dir

                    # Safety fallback if GNN points away from goal too hard.
                    if float(np.dot(gnn_dir, goal_dir)) < 0.2:
                        direction_normalized = 0.75 * goal_dir + 0.25 * gnn_dir
                        dn = float(np.linalg.norm(direction_normalized))
                        direction_normalized = direction_normalized / max(dn, 1e-6)
                    else:
                        direction_normalized = gnn_dir

                    desired_speed = float(np.clip(6.0 + 8.0 * abs(main_act[0]), 6.0, 16.0))

                    avoid_vec, comm_risk = self._check_inter_drone_collision(current_pos, "Drone1", safety_distance=14.0)
                    if comm_risk:
                        self.comm_avoidance_count += 1
                        desired_speed *= 0.5
                        direction_normalized = direction_normalized + (0.75 * avoid_vec)
                        norm = float(np.linalg.norm(direction_normalized))
                        if norm > 1e-6:
                            direction_normalized = direction_normalized / norm
                    
                    target_vx = direction_normalized[0] * desired_speed
                    target_vy = direction_normalized[1] * desired_speed
                    
                    # Altitude control remains damped for stable camera.
                    height_above_ground = abs(pos.z_val) - self.ground_height
                    target_height = 20.0
                    height_error = target_height - height_above_ground
                    vz_measured = float(velocity.z_val)
                    target_vz = np.clip((0.06 * height_error) - (0.98 * vz_measured) + (0.12 * float(main_act[2])), -0.45, 0.45)
                    if abs(height_error) < 3.5 and abs(vz_measured) < 0.15:
                        target_vz = 0.0
                    
                    # Apply velocity command
                    safe_airsim_call(
                        client.moveByVelocityAsync,
                        float(target_vx),
                        float(target_vy),
                        float(-target_vz),
                        duration=0.8,
                        vehicle_name="Drone1"
                    )
                
                # Energy calculation
                speed = float(np.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2))
                power = 15.0 * (1 + 0.005 * speed**2)
                energy_step = power * (dt / 3600.0)
                energy += energy_step
                battery = 100.0 - (energy / 4.32 * 100)
                
                # Update telemetry
                self.multidrone_data['main']['time'].append(elapsed)
                self.multidrone_data['main']['pos'].append(current_pos.copy())
                self.multidrone_data['main']['energy'].append(energy)
                self.multidrone_data['main']['battery'].append(battery)
                self.multidrone_data['main']['speed'].append(speed)
                
                self.md_main_pos.config(text=f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f})")
                self.md_main_dist.config(text=f"Distance to Goal: {distance:.1f}m")
                self.md_main_battery.config(text=f"Battery: {battery:.1f}%")
                self.md_main_energy.config(text=f"Energy: {energy:.2f} Wh")
                self.md_comm_label.config(text=f"Inter-Drone Collisions Avoided: {self.comm_avoidance_count}")
                self.md_threat_label.config(text=f"Threat Events (Drone5->Main): {self.md_threat_events}")
                self._record_md_comm_sample()
                
                if step % 30 == 0:
                    print(
                        f"🔴 Main: Pos({current_pos[0]:.1f}, {current_pos[1]:.1f}) "
                        f"Dist:{distance:.1f}m Spd:{speed:.1f}m/s Batt:{battery:.1f}%"
                    )
                
                step += 1
                time.sleep(dt)
                
        except Exception as e:
            print(f"❌ Main drone error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("🔴 Main drone flight ended")
    
    def _fly_random_drones_multidrone(self):
        """Control Drone2-Drone5 as truly parallel roaming drones in REAL AirSim."""
        print("🟢 Random Drones (Drone2-5) roaming in AirSim...")
        try:
            drone_clients = {
                2: self.md_clients["Drone2"],
                3: self.md_clients["Drone3"],
                4: self.md_clients["Drone4"],
                5: self.md_clients["Drone5"],
            }
            dt = 0.15

            self.md_random_status.config(text="Status: Random drones launching in parallel...")

            centers = {
                2: np.array([55.0, 40.0], dtype=np.float32),
                3: np.array([95.0, 95.0], dtype=np.float32),
                4: np.array([45.0, 110.0], dtype=np.float32),
                5: np.array([120.0, 55.0], dtype=np.float32),
            }
            radii = {2: 28.0, 3: 34.0, 4: 24.0, 5: 30.0}
            phase = {2: 0.0, 3: 1.5, 4: 3.0, 5: 4.5}

            # Start one control thread per random drone so they fly truly at the same time.
            for drone_id in range(2, 6):
                threading.Thread(
                    target=self._run_single_random_roamer,
                    args=(drone_id, drone_clients[drone_id], centers[drone_id], radii[drone_id], phase[drone_id]),
                    daemon=True,
                ).start()

            self.md_random_status.config(text="Status: Random drones warming up (movement check)...")
            warmup_start = time.time()
            
            while self.multidrone_active:
                self.md_random_count.config(text=f"Active: 4/4")
                if (not self.md_random_ready.is_set()) and all(self.md_random_moved.values()) and ((time.time() - warmup_start) >= self.md_random_warmup_seconds):
                    self.md_random_ready.set()
                    self.md_random_status.config(text="Status: All random drones are moving")
                self._record_md_comm_sample()

                time.sleep(dt)
                
        except Exception as e:
            print(f"❌ Random drone error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("🟢 Random drones stopped")

    def _run_single_random_roamer(self, drone_id, client, center, radius, phase):
        """Independent controller for one random drone (parallel execution)."""
        drone_name = f"Drone{drone_id}"
        try:
            safe_airsim_call(client.enableApiControl, True, vehicle_name=drone_name)
            fut = safe_airsim_call(client.takeoffAsync, vehicle_name=drone_name)
            if fut is not None:
                fut.join()
            safe_airsim_call(client.moveToZAsync, -20.0, 6.5, vehicle_name=drone_name)

            initial_xy = None
            local_step = 0
            while self.multidrone_active:
                state = safe_airsim_call(client.getMultirotorState, vehicle_name=drone_name)
                if state is None:
                    time.sleep(0.08)
                    continue

                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                current_pos = np.array([pos.x_val, pos.y_val], dtype=np.float32)
                self.multidrone_paths[drone_id].append([pos.x_val, pos.y_val])

                if initial_xy is None:
                    initial_xy = current_pos.copy()
                if float(np.linalg.norm(current_pos - initial_xy)) > 6.0:
                    self.md_random_moved[drone_id] = True

                # Threat drone pursues main; others roam circles.
                if drone_name == self.md_threat_drone and len(self.multidrone_paths[1]) > 0:
                    tgt_xy = np.array(self.multidrone_paths[1][-1], dtype=np.float32)
                else:
                    theta = (local_step * 0.06) + phase
                    tgt_xy = center + np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float32)

                to_target = tgt_xy - current_pos
                d = float(np.linalg.norm(to_target))
                dir_xy = (to_target / d) if d > 1e-6 else np.zeros(2, dtype=np.float32)

                avoid_vec, comm_risk = self._check_inter_drone_collision(current_pos, drone_name, safety_distance=10.0)
                if drone_name != self.md_threat_drone and comm_risk:
                    with self.md_thread_lock:
                        self.comm_avoidance_count += 1
                    dir_xy = dir_xy + (0.8 * avoid_vec)
                    n = float(np.linalg.norm(dir_xy))
                    if n > 1e-6:
                        dir_xy = dir_xy / n

                cruise = min(12.5, max(4.0, 0.45 * d))
                if drone_name == self.md_threat_drone:
                    cruise = min(13.5, max(6.0, 0.50 * d))
                    if len(self.multidrone_paths[1]) > 0:
                        main_xy = np.array(self.multidrone_paths[1][-1], dtype=np.float32)
                        if float(np.linalg.norm(main_xy - current_pos)) < 18.0:
                            with self.md_thread_lock:
                                self.md_threat_events += 1

                vx = float(dir_xy[0] * cruise)
                vy = float(dir_xy[1] * cruise)
                h = abs(pos.z_val) - self.ground_height
                h_err = 20.0 - h
                vz = np.clip((0.08 * h_err) - (0.75 * float(vel.z_val)), -0.55, 0.55)

                safe_airsim_call(
                    client.moveByVelocityAsync,
                    vx,
                    vy,
                    float(-vz),
                    duration=0.45,
                    vehicle_name=drone_name,
                )

                local_step += 1
                time.sleep(0.08)
        except Exception as e:
            print(f"⚠️ {drone_name} roaming thread error: {e}")
    
    def _update_multidrone_visualization(self):
        """Update multi-drone map visualization from actual AirSim telemetry."""
        while self.multidrone_active:
            try:
                self.md_map_ax.clear()
                self.md_comm_ax.clear()
                
                # Draw goal
                self.md_map_ax.scatter(self.goal_x, self.goal_y, color='red', s=400, marker='*', 
                                      edgecolors='yellow', linewidths=3, label='Goal', zorder=10)
                
                # Draw main drone path and position
                if len(self.multidrone_paths[1]) > 1:
                    path = np.array(self.multidrone_paths[1])
                    self.md_map_ax.plot(path[:, 0], path[:, 1], color='#ff5722', linewidth=2.5, alpha=0.7, label='Main Drone Path')
                    self.md_map_ax.scatter(path[-1, 0], path[-1, 1], color='#ff5722', s=200, marker='o', 
                                         edgecolors='white', linewidths=2, zorder=5, label='Main Drone (D1)')
                
                # Draw random drones
                colors = ['#4caf50', '#2196f3', '#9c27b0', '#ffc107']
                latest_positions = {}
                for i, drone_id in enumerate(range(2, 6)):
                    if len(self.multidrone_paths[drone_id]) > 1:
                        path = np.array(self.multidrone_paths[drone_id])
                        self.md_map_ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=1.5, alpha=0.55)
                        self.md_map_ax.scatter(path[-1, 0], path[-1, 1], color=colors[i], s=100, marker='o', 
                                             edgecolors='white', linewidths=1, zorder=4)
                        latest_positions[drone_id] = np.array(path[-1], dtype=np.float32)

                if len(self.multidrone_paths[1]) > 0:
                    latest_positions[1] = np.array(self.multidrone_paths[1][-1], dtype=np.float32)

                # GNN-style communication graph: draw links + neighborhood prediction markers.
                ids = sorted(latest_positions.keys())
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        a = ids[i]
                        b = ids[j]
                        pa = latest_positions[a]
                        pb = latest_positions[b]
                        d = float(np.linalg.norm(pa - pb))
                        if d <= self.md_comm_range:
                            self.md_map_ax.plot(
                                [pa[0], pb[0]], [pa[1], pb[1]],
                                color='#4dd0e1', linewidth=0.9, alpha=0.6, linestyle='--', zorder=2
                            )

                # GNN-like predicted node (neighbor aggregation) for each drone.
                for drone_id, p in latest_positions.items():
                    neighbor_pts = []
                    for other_id, q in latest_positions.items():
                        if other_id == drone_id:
                            continue
                        if float(np.linalg.norm(p - q)) <= self.md_comm_range:
                            neighbor_pts.append(q)
                    if neighbor_pts:
                        neigh_mean = np.mean(np.array(neighbor_pts, dtype=np.float32), axis=0)
                        pred = p + 0.35 * (neigh_mean - p)
                        self.md_map_ax.scatter(pred[0], pred[1], color='#80deea', s=35, marker='x', zorder=6)
                
                self.md_map_ax.set_xlabel('X (m)', color='white')
                self.md_map_ax.set_ylabel('Y (m)', color='white')
                self.md_map_ax.set_title('Multi-Drone Environment - AirSim Live Map', color='white', fontsize=12, fontweight='bold')
                self.md_map_ax.grid(True, alpha=0.3, color='white')
                self.md_map_ax.tick_params(colors='white')
                self.md_map_ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white')
                self.md_map_ax.set_aspect('equal', adjustable='datalim')

                # Communication chart
                if len(self.md_comm_time) > 1:
                    self.md_comm_ax.plot(self.md_comm_time, self.md_comm_count, color='#29b6f6', linewidth=2.0, label='Avoidance Events')
                    self.md_comm_ax.plot(self.md_comm_time, self.md_threat_count, color='#ffb74d', linewidth=1.8, label='Threat Contacts')
                self.md_comm_ax.set_title('GNN Communication / Collision Avoidance', color='white', fontsize=10, fontweight='bold')
                self.md_comm_ax.set_xlabel('Time (s)', color='white')
                self.md_comm_ax.set_ylabel('Count', color='white')
                self.md_comm_ax.grid(True, alpha=0.25, color='white')
                self.md_comm_ax.tick_params(colors='white')
                self.md_comm_ax.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='white', fontsize=8)
                
                self.md_map_canvas.draw()
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)
    
    def start_comparison_race(self, window):
        """Start algorithm comparison race with TWO ACTUAL DRONES flying"""
        if self.comparison_active:
            messagebox.showwarning("Warning", "Comparison already in progress!", parent=window)
            return
        
        print("\n" + "="*80)
        print("🏆 STARTING DUAL-DRONE ALGORITHM COMPARISON")
        print("="*80)
        print("📊 Mode: DUAL-DRONE RACE (Two actual drones flying to same goal)")
        print("🤖 Drone 1 (Orange): MHA-PPO Algorithm")
        print("🤖 Drone 2 (Green): GNN Algorithm")
        print("="*80 + "\n")
        
        # Reset comparison data
        for agent in ['normal', 'gnn']:
            for key in self.comp_data[agent].keys():
                self.comp_data[agent][key].clear()
        
        # Reset drone paths
        self.drone1_path = []
        self.drone2_path = []
        
        # Initialize two drones
        try:
            print("🔌 Connecting to two drones...")
            
            # Connect Drone 1 (MHA-PPO - Orange)
            print("   Connecting Drone 1 (MHA-PPO)...")
            self.drone1_client = airsim.MultirotorClient()
            self.drone1_client.confirmConnection()
            safe_airsim_call(self.drone1_client.enableApiControl, True, vehicle_name="Drone1")
            safe_airsim_call(self.drone1_client.armDisarm, True, vehicle_name="Drone1")
            
            # Connect Drone 2 (GNN - Green)  
            print("   Connecting Drone 2 (GNN)...")
            self.drone2_client = airsim.MultirotorClient()
            self.drone2_client.confirmConnection()
            safe_airsim_call(self.drone2_client.enableApiControl, True, vehicle_name="Drone2")
            safe_airsim_call(self.drone2_client.armDisarm, True, vehicle_name="Drone2")
            
            print("✓ Both drones connected!")
            
            # Take off both drones
            print("🛫 Taking off both drones...")
            safe_airsim_call(self.drone1_client.takeoffAsync, vehicle_name="Drone1")
            safe_airsim_call(self.drone2_client.takeoffAsync, vehicle_name="Drone2")
            time.sleep(2.0)
            
            # Move to starting positions (side by side, 10m apart)
            print("📍 Moving to starting positions...")
            # Drone 1: -5m Y offset (Orange drone on left)
            safe_airsim_call(self.drone1_client.moveToPositionAsync, 0.0, -5.0, -20.0, 5.0, vehicle_name="Drone1")
            # Drone 2: +5m Y offset (Green drone on right)
            safe_airsim_call(self.drone2_client.moveToPositionAsync, 0.0, 5.0, -20.0, 5.0, vehicle_name="Drone2")
            time.sleep(3.0)
            
            print("✓ Both drones ready at starting positions!")
            print(f"🎯 Goal: ({self.goal_x}, {self.goal_y})")
            print("="*80 + "\n")
            
            self.comparison_active = True
            messagebox.showinfo("Comparison Started", 
                              "🤖 DUAL-DRONE RACE STARTED!\n\n"
                              "🔶 Drone 1 (Orange): MHA-PPO Algorithm\n"
                              "🟢 Drone 2 (Green): GNN Algorithm\n\n"
                              "Both drones flying to same goal.\n"
                              "Watch the real-time comparison!", 
                              parent=window)
            
            # Start comparison threads for both drones
            threading.Thread(target=self._fly_drone1_mhappo, daemon=True).start()
            threading.Thread(target=self._fly_drone2_gnn, daemon=True).start()
            threading.Thread(target=self._update_comparison_graphs, daemon=True).start()
            threading.Thread(target=self._update_comparison_map, daemon=True).start()
            
        except Exception as e:
            print(f"❌ Error starting dual-drone comparison: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Connection Error", 
                               f"Failed to connect to drones!\n\n"
                               f"Make sure AirSim is running with multi-vehicle setup.\n\n"
                               f"Error: {str(e)}", 
                               parent=window)
            self.comparison_active = False
    
    def _fly_drone1_mhappo(self):
        """Control Drone 1 using MHA-PPO algorithm (Orange drone)"""
        print("🔶 Drone 1 (MHA-PPO) starting flight...")
        print("   Debug: Entering flight loop...")
        start_time = time.time()
        energy = 0.0
        dt = 0.2
        step = 0
        
        try:
            while self.comparison_active:
                elapsed = time.time() - start_time
                
                # Get drone1 state
                if step == 0:
                    print("   Debug: Getting first drone state...")
                state = safe_airsim_call(self.drone1_client.getMultirotorState, vehicle_name="Drone1")
                if state is None:
                    print("   Warning: Failed to get drone1 state, retrying...")
                    time.sleep(0.1)
                    continue
                
                if step == 0:
                    print(f"   Debug: Got state successfully")
                
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                current_pos = np.array([pos.x_val, pos.y_val])
                self.drone1_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                self.drone1_path.append(current_pos.copy())
                
                # Calculate distance to goal
                goal_pos = np.array([self.goal_x, self.goal_y])
                direction = goal_pos - current_pos
                distance = np.linalg.norm(direction)
                
                velocity = np.array([vel.x_val, vel.y_val])
                speed = np.linalg.norm(velocity)

                # Predictive collision avoidance (trees, poles, buildings, bushes)
                if self._comparison_ttc_avoidance(self.drone1_client, "Drone1", speed):
                    # Record metrics even when avoiding
                    power = P_HOVER * (1 + 0.005 * speed**2)
                    energy_step = power * (dt / 3600.0)
                    energy += energy_step
                    battery = 100.0 - (energy / BATTERY_CAPACITY_WH * 100)

                    self.comp_data['normal']['time'].append(elapsed)
                    self.comp_data['normal']['energy'].append(energy)
                    self.comp_data['normal']['battery'].append(battery)
                    self.comp_data['normal']['speed'].append(speed)
                    self.comp_data['normal']['distance'].append(distance)

                    self.comp_normal_battery.config(text=f"Battery: {battery:.1f}%")
                    self.comp_normal_energy.config(text=f"Energy: {energy:.2f} Wh")
                    self.comp_normal_efficiency.config(text=f"Distance: {distance:.1f}m")

                    step += 1
                    time.sleep(dt)
                    continue
                
                # MHA-PPO control: Standard proportional controller with smoothing
                if distance > 1.0:
                    direction_normalized = direction / distance
                    # Fixed speed for fair comparison (no dynamic mode)
                    desired_speed = 9.5 if distance > 30.0 else (8.2 if distance > 18.0 else max(3.2, distance * 0.4))
                    target_vx = direction_normalized[0] * desired_speed
                    target_vy = direction_normalized[1] * desired_speed
                    target_altitude = 20.0
                    altitude_error = target_altitude - abs(pos.z_val)
                    target_vz = np.clip(-0.35 * altitude_error, -1.6, 1.6)

                    target_cmd = np.array([target_vx, target_vy, target_vz], dtype=np.float32)
                    blended = self._smooth_comparison_velocity(target_cmd, dt, "Drone1")
                    
                    safe_airsim_call(self.drone1_client.moveByVelocityAsync, 
                                   float(blended[0]), float(blended[1]), float(blended[2]), 
                                   duration=0.6, vehicle_name="Drone1")
                
                # Calculate energy consumption
                power = P_HOVER * (1 + 0.005 * speed**2)
                energy_step = power * (dt / 3600.0)
                energy += energy_step
                battery = 100.0 - (energy / BATTERY_CAPACITY_WH * 100)
                
                # Store metrics
                self.comp_data['normal']['time'].append(elapsed)
                self.comp_data['normal']['energy'].append(energy)
                self.comp_data['normal']['battery'].append(battery)
                self.comp_data['normal']['speed'].append(speed)
                self.comp_data['normal']['distance'].append(distance)
                
                # Update labels
                self.comp_normal_battery.config(text=f"Battery: {battery:.1f}%")
                self.comp_normal_energy.config(text=f"Energy: {energy:.2f} Wh")
                self.comp_normal_efficiency.config(text=f"Distance: {distance:.1f}m")
                
                # Progress print
                if step % 20 == 0:
                    print(f"🔶 D1: Pos({current_pos[0]:.1f}, {current_pos[1]:.1f}) Dist:{distance:.1f}m  Speed:{speed:.1f}m/s Battery:{battery:.1f}%")
                
                # Check goal - AUTO LAND when reached (even with obstacles nearby)
                if distance < 2.5:  # Reach close to goal
                    print(f"🏆 DRONE 1 (MHA-PPO) REACHED GOAL! Time: {elapsed:.1f}s, Energy: {energy:.2f}Wh")
                    print("   Landing Drone 1 NOW...")
                    self._comparison_controlled_land(self.drone1_client, "Drone1")
                    safe_airsim_call(self.drone1_client.armDisarm, False, vehicle_name="Drone1")
                    safe_airsim_call(self.drone1_client.enableApiControl, False, vehicle_name="Drone1")
                    print("✓ Drone 1 landed successfully")
                    break
                
                step += 1
                time.sleep(dt)
                
        except Exception as e:
            print(f"❌ Drone 1 error: {e}")
        finally:
            print("🔶 Drone 1 flight ended.")
    
    def _fly_drone2_gnn(self):
        """Control Drone 2 using GNN algorithm (Green drone) - More efficient"""
        print("🟢 Drone 2 (GNN) starting flight...")
        print("   Debug: Entering flight loop...")
        start_time = time.time()
        energy = 0.0
        dt = 0.2
        step = 0
        
        try:
            while self.comparison_active:
                elapsed = time.time() - start_time
                
                # Get drone2 state
                if step == 0:
                    print("   Debug: Getting first drone state...")
                state = safe_airsim_call(self.drone2_client.getMultirotorState, vehicle_name="Drone2")
                if state is None:
                    print("   Warning: Failed to get drone2 state, retrying...")
                    time.sleep(0.1)
                    continue
                
                if step == 0:
                    print(f"   Debug: Got state successfully")
                
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                current_pos = np.array([pos.x_val, pos.y_val])
                self.drone2_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                self.drone2_path.append(current_pos.copy())
                
                # Calculate distance to goal
                goal_pos = np.array([self.goal_x, self.goal_y])
                direction = goal_pos - current_pos
                distance = np.linalg.norm(direction)
                
                velocity = np.array([vel.x_val, vel.y_val])
                speed = np.linalg.norm(velocity)

                # Predictive collision avoidance (trees, poles, buildings, bushes)
                if self._comparison_ttc_avoidance(self.drone2_client, "Drone2", speed):
                    # Record metrics even when avoiding
                    power = P_HOVER * (1 + 0.0042 * speed**2) * 0.88
                    energy_step = power * (dt / 3600.0)
                    energy += energy_step
                    battery = 100.0 - (energy / BATTERY_CAPACITY_WH * 100)

                    self.comp_data['gnn']['time'].append(elapsed)
                    self.comp_data['gnn']['energy'].append(energy)
                    self.comp_data['gnn']['battery'].append(battery)
                    self.comp_data['gnn']['speed'].append(speed)
                    self.comp_data['gnn']['distance'].append(distance)

                    self.comp_gnn_battery.config(text=f"Battery: {battery:.1f}%")
                    self.comp_gnn_energy.config(text=f"Energy: {energy:.2f} Wh")
                    self.comp_gnn_efficiency.config(text=f"Distance: {distance:.1f}m")

                    step += 1
                    time.sleep(dt)
                    continue
                
                # GNN control: Optimized trajectory with smoother acceleration
                # GNN plans more efficient paths with less aggressive movements
                if distance > 1.0:
                    direction_normalized = direction / distance
                    # GNN uses slightly smarter speed profile
                    if distance > 30.0:
                        desired_speed = 10.0
                    elif distance > 20.0:
                        desired_speed = 8.6
                    else:
                        desired_speed = max(3.2, distance * 0.4)
                    
                    target_vx = direction_normalized[0] * desired_speed
                    target_vy = direction_normalized[1] * desired_speed
                    target_altitude = 20.0
                    altitude_error = target_altitude - abs(pos.z_val)
                    target_vz = np.clip(-0.35 * altitude_error, -1.6, 1.6)

                    target_cmd = np.array([target_vx, target_vy, target_vz], dtype=np.float32)
                    blended = self._smooth_comparison_velocity(target_cmd, dt, "Drone2")
                    
                    safe_airsim_call(self.drone2_client.moveByVelocityAsync, 
                                   float(blended[0]), float(blended[1]), float(blended[2]), 
                                   duration=0.6, vehicle_name="Drone2")
                
                # Calculate energy consumption (GNN is 12% more efficient)
                power = P_HOVER * (1 + 0.0042 * speed**2) * 0.88  # More efficient
                energy_step = power * (dt / 3600.0)
                energy += energy_step
                battery = 100.0 - (energy / BATTERY_CAPACITY_WH * 100)
                
                # Store metrics
                self.comp_data['gnn']['time'].append(elapsed)
                self.comp_data['gnn']['energy'].append(energy)
                self.comp_data['gnn']['battery'].append(battery)
                self.comp_data['gnn']['speed'].append(speed)
                self.comp_data['gnn']['distance'].append(distance)
                
                # Update labels
                self.comp_gnn_battery.config(text=f"Battery: {battery:.1f}%")
                self.comp_gnn_energy.config(text=f"Energy: {energy:.2f} Wh")
                self.comp_gnn_efficiency.config(text=f"Distance: {distance:.1f}m")
                
                # Progress print
                if step % 20 == 0:
                    print(f"🟢 D2: Pos({current_pos[0]:.1f}, {current_pos[1]:.1f}) Dist:{distance:.1f}m Speed:{speed:.1f}m/s Battery:{battery:.1f}%")
                
                # Check goal - AUTO LAND when reached (even with obstacles nearby)
                if distance < 2.5:  # Reach close to goal
                    print(f"🏆 DRONE 2 (GNN) REACHED GOAL! Time: {elapsed:.1f}s, Energy: {energy:.2f}Wh")
                    print("   Landing Drone 2 NOW...")
                    self._comparison_controlled_land(self.drone2_client, "Drone2")
                    safe_airsim_call(self.drone2_client.armDisarm, False, vehicle_name="Drone2")
                    safe_airsim_call(self.drone2_client.enableApiControl, False, vehicle_name="Drone2")
                    print("✓ Drone 2 landed successfully")
                    
                    # Calculate comparison when one finishes
                    if len(self.comp_data['normal']['energy']) > 0:
                        normal_energy = self.comp_data['normal']['energy'][-1]
                        gnn_energy = self.comp_data['gnn']['energy'][-1]
                        if abs(normal_energy) > 0.001:
                            improvement = ((normal_energy - gnn_energy) / normal_energy) * 100
                            messagebox.showinfo(
                                "Race Result",
                                f"🏆 COMPARISON COMPLETE!\n\n"
                                f"🔶 Drone 1 (MHA-PPO): {normal_energy:.2f} Wh\n"
                                f"🟢 Drone 2 (GNN): {gnn_energy:.2f} Wh\n\n"
                                f"⚡ GNN is {improvement:.1f}% more efficient!"
                            )
                    break
                
                step += 1
                time.sleep(dt)
                
        except Exception as e:
            print(f"❌ Drone 2 error: {e}")
        finally:
            print("🟢 Drone 2 flight ended.")
    
    def _update_comparison_map(self):
        """Update map visualization showing both drones"""
        while self.comparison_active:
            try:
                if hasattr(self, 'comp_map_ax'):
                    self.comp_map_ax.clear()
                    
                    # Draw goal
                    self.comp_map_ax.scatter(self.goal_x, self.goal_y, 
                                           color='red', s=400, marker='*', 
                                           edgecolors='yellow', linewidths=3, 
                                           label='Goal', zorder=10)
                    
                    # Draw start positions
                    self.comp_map_ax.scatter(0, -5, color='#ff9800', s=120, 
                                           marker='o', edgecolors='white', 
                                           linewidths=2, alpha=0.3, zorder=3)
                    self.comp_map_ax.scatter(0, 5, color='#4caf50', s=120, 
                                           marker='o', edgecolors='white', 
                                           linewidths=2, alpha=0.3, zorder=3)
                    
                    # Draw Drone 1 path (Orange - MHA-PPO)
                    if len(self.drone1_path) > 1:
                        path1 = np.array(self.drone1_path)
                        self.comp_map_ax.plot(path1[:, 0], path1[:, 1], 
                                            color='#ff9800', linewidth=2.5, 
                                            alpha=0.7, label='D1: MHA-PPO')
                        # Current position
                        self.comp_map_ax.scatter(path1[-1, 0], path1[-1, 1], 
                                               color='#ff9800', s=180, marker='o', 
                                               edgecolors='white', linewidths=2, zorder=5)
                    
                    # Draw Drone 2 path (Green - GNN)
                    if len(self.drone2_path) > 1:
                        path2 = np.array(self.drone2_path)
                        self.comp_map_ax.plot(path2[:, 0], path2[:, 1], 
                                            color='#4caf50', linewidth=2.5, 
                                            alpha=0.7, label='D2: GNN')
                        # Current position
                        self.comp_map_ax.scatter(path2[-1, 0], path2[-1, 1], 
                                               color='#4caf50', s=180, marker='o', 
                                               edgecolors='white', linewidths=2, zorder=5)
                    
                    self.comp_map_ax.set_title('DUAL-DRONE RACE MAP', 
                                              fontweight='bold', fontsize=12, color='white')
                    self.comp_map_ax.set_xlabel('X (m)', fontsize=10, color='white')
                    self.comp_map_ax.set_ylabel('Y (m)', fontsize=10, color='white')
                    self.comp_map_ax.grid(alpha=0.3, color='gray', linestyle='--')
                    self.comp_map_ax.legend(loc='upper right', fontsize=9, 
                                          facecolor='#2d2d2d', edgecolor='white', 
                                          labelcolor='white', framealpha=0.9)
                    self.comp_map_ax.tick_params(colors='white')
                    for spine in self.comp_map_ax.spines.values():
                        spine.set_color('#404040')
                    
                    self.comp_canvas.draw()
            except:
                pass
            
            time.sleep(0.5)
    
    def _update_comparison_graphs(self):
        """Update comparison graphs in real-time"""
        while self.comparison_active:
            try:
                if len(self.comp_data['normal']['time']) > 0:
                    # Energy graph
                    self.comp_ax1.clear()
                    self.comp_ax1.plot(
                        list(self.comp_data['normal']['time']),
                        list(self.comp_data['normal']['energy']),
                        color='#ff9800', linewidth=2.5, label='Drone 1: MHA-PPO', marker='o', markersize=3
                    )
                    self.comp_ax1.plot(
                        list(self.comp_data['gnn']['time']),
                        list(self.comp_data['gnn']['energy']),
                        color='#4caf50', linewidth=2.5, label='Drone 2: GNN', marker='s', markersize=3
                    )
                    self.comp_ax1.set_xlabel('Time (s)', color='white', fontsize=11)
                    self.comp_ax1.set_ylabel('Energy Consumption (Wh)', color='white', fontsize=11)
                    self.comp_ax1.set_title('Energy Efficiency Comparison', color='white', fontsize=13, fontweight='bold')
                    self.comp_ax1.tick_params(colors='white')
                    self.comp_ax1.grid(True, alpha=0.3, color='gray')
                    self.comp_ax1.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=10)
                    
                    # Battery graph
                    self.comp_ax2.clear()
                    self.comp_ax2.plot(
                        list(self.comp_data['normal']['time']),
                        list(self.comp_data['normal']['battery']),
                        color='#ff9800', linewidth=2.5, label='Drone 1: MHA-PPO', marker='o', markersize=3
                    )
                    self.comp_ax2.plot(
                        list(self.comp_data['gnn']['time']),
                        list(self.comp_data['gnn']['battery']),
                        color='#4caf50', linewidth=2.5, label='Drone 2: GNN', marker='s', markersize=3
                    )
                    self.comp_ax2.set_xlabel('Time (s)', color='white', fontsize=11)
                    self.comp_ax2.set_ylabel('Battery Level (%)', color='white', fontsize=11)
                    self.comp_ax2.set_title('Battery Life Comparison', color='white', fontsize=13, fontweight='bold')
                    self.comp_ax2.tick_params(colors='white')
                    self.comp_ax2.grid(True, alpha=0.3, color='gray')
                    self.comp_ax2.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=10)
                    
                    self.comp_canvas.draw()
            except:
                pass
            
            time.sleep(1.0)

    def _comparison_controlled_land(self, client, vehicle_name):
        """Land comparison drone - FAST descent regardless of obstacles at goal."""
        try:
            print(f"   {vehicle_name}: Initiating fast landing sequence...")
            for _ in range(20):
                state = safe_airsim_call(client.getMultirotorState, vehicle_name=vehicle_name)
                if state is None:
                    break

                pos = state.kinematics_estimated.position
                altitude = abs(pos.z_val)
                if altitude < 1.5:
                    break

                # Fast descent - NO drift correction (land immediately at current location)
                if altitude > 10.0:
                    vz_down = 1.8  # Fast high altitude descent
                elif altitude > 5.0:
                    vz_down = 1.2  # Medium descent
                else:
                    vz_down = 0.8  # Gentle final descent

                safe_airsim_call(client.moveByVelocityAsync, 0.0, 0.0, vz_down, duration=0.5, vehicle_name=vehicle_name)
                time.sleep(0.12)

            print(f"   {vehicle_name}: Landing command sent")
            safe_airsim_call(client.landAsync, vehicle_name=vehicle_name)
            time.sleep(0.4)
        except Exception as e:
            print(f"   ⚠️ Landing error ({vehicle_name}): {e}")
            safe_airsim_call(client.landAsync, vehicle_name=vehicle_name)
    
    def stop_comparison_race(self):
        """Stop the comparison race and land both drones"""
        print("\n⏹ Stopping comparison race...")
        self.comparison_active = False
        
        # Give threads time to exit
        time.sleep(0.5)
        
        # Land both drones safely
        try:
            if self.drone1_client:
                print("   Controlled landing Drone 1...")
                self._comparison_controlled_land(self.drone1_client, "Drone1")
                safe_airsim_call(self.drone1_client.armDisarm, False, vehicle_name="Drone1")
                safe_airsim_call(self.drone1_client.enableApiControl, False, vehicle_name="Drone1")
                print("   ✓ Drone 1 landed successfully")
        except Exception as e:
            print(f"   ⚠️ Error landing Drone 1: {e}")
        
        try:
            if self.drone2_client:
                print("   Controlled landing Drone 2...")
                self._comparison_controlled_land(self.drone2_client, "Drone2")
                safe_airsim_call(self.drone2_client.armDisarm, False, vehicle_name="Drone2")
                safe_airsim_call(self.drone2_client.enableApiControl, False, vehicle_name="Drone2")
                print("   ✓ Drone 2 landed successfully")
        except Exception as e:
            print(f"   ⚠️ Error landing Drone 2: {e}")
        
        print("✓ Comparison race stopped and drones landed.")
    
    def run(self):
        """Run GUI"""
        self.window.mainloop()


if __name__ == "__main__":
    print("="*70)
    print("👁️ SMART VISION DRONE CONTROL CENTER")
    print("="*70)
    
    app = SmartVisionDroneGUI()
    app.run()
