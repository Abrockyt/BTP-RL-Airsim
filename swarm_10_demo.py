import airsim
import torch
import torch.nn as nn
import numpy as np
import time
import math

# =========================================================
# 1. GNN ARCHITECTURE (Must match your trained file)
# =========================================================
class GNN_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN_Layer, self).__init__()
        self.message_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.update_net = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.ReLU())
    def forward(self, node_features, adj_matrix):
        messages = self.message_net(node_features)
        aggregated = torch.matmul(adj_matrix, messages)
        combined = torch.cat([node_features, aggregated], dim=2)
        return self.update_net(combined)

class GNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GNNActor, self).__init__()
        self.embedding = nn.Linear(state_dim, hidden_dim)
        self.gnn1 = GNN_Layer(hidden_dim, hidden_dim)
        self.gnn2 = GNN_Layer(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
    def forward(self, state, adj):
        x = torch.relu(self.embedding(state))
        x = torch.relu(self.gnn1(x, adj))
        x = self.gnn2(x, adj)
        return self.tanh(self.mean_layer(x)), None

# =========================================================
# 2. CONFIGURATION
# =========================================================
NUM_DRONES = 10
MODEL_PATH = "gnn_swarm_brain.pth" # Ensure you downloaded this from Kaggle
RADIUS = 20.0 # Must match the settings.json radius roughly

# Generate Names and Goals
drone_names = [f"Drone{i}" for i in range(1, NUM_DRONES + 1)]
goals = {}

# Goals: Fly to the EXACT opposite side of the circle
for i in range(NUM_DRONES):
    angle = (2 * math.pi * i) / NUM_DRONES
    # Current Pos approx: [R*cos(a), R*sin(a)]
    # Goal Pos: [-R*cos(a), -R*sin(a)]
    goals[drone_names[i]] = np.array([-RADIUS * math.cos(angle), -RADIUS * math.sin(angle)])

# =========================================================
# 3. CONNECT AND TAKEOFF
# =========================================================
print(f"🚀 Connecting to {NUM_DRONES} Drones...")
client = airsim.MultirotorClient()
client.confirmConnection()

for name in drone_names:
    client.enableApiControl(True, name)
    client.armDisarm(True, name)
    client.takeoffAsync(vehicle_name=name)

# Wait for takeoff to finish roughly
time.sleep(3)

# Move to flight altitude
print("⬆️ Ascending to altitude...")
for name in drone_names:
    client.moveToZAsync(-3, 1, vehicle_name=name)
time.sleep(3)

# =========================================================
# 4. LOAD BRAIN
# =========================================================
print("🧠 Loading GNN Brain...")
model = GNNActor(state_dim=4, action_dim=2)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() 
    print("✅ Brain Loaded!")
except:
    print("⚠️ Model not found/matching. Using random weights.")

# =========================================================
# 5. MAIN LOOP
# =========================================================
print("⚔️ BEGINNING SWARM CROSSOVER...")

while True:
    # A. Gather State from ALL Drones
    positions = []
    velocities = []
    
    for name in drone_names:
        state = client.getMultirotorState(vehicle_name=name)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        positions.append(np.array([pos.x_val, pos.y_val]))
        velocities.append(np.array([vel.x_val, vel.y_val]))
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # B. Build Graph Inputs
    node_features = []
    for i in range(NUM_DRONES):
        # Feature: [RelGoalX, RelGoalY, VelX, VelY]
        rel_goal = (goals[drone_names[i]] - positions[i]) / 50.0 # Normalize
        norm_vel = velocities[i] / 5.0
        node_features.append(np.concatenate([rel_goal, norm_vel]))
        
    state_tensor = torch.FloatTensor(np.array(node_features)).unsqueeze(0) # [1, 10, 4]
    
    # Adjacency Matrix (Fully Connected for now, or based on distance)
    # Let's use distance-based connection (Vision Range = 15m)
    adj_matrix = np.eye(NUM_DRONES)
    for i in range(NUM_DRONES):
        for j in range(NUM_DRONES):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 15.0: # If neighbor is close
                adj_matrix[i, j] = 1.0
                
    adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)
    
    # C. GNN Inference
    with torch.no_grad():
        actions_mean, _ = model(state_tensor, adj_tensor)
        actions = actions_mean.squeeze(0).numpy() # [10, 2]
        
    # D. Execute
    for i, name in enumerate(drone_names):
        vx = float(actions[i][0] * 2.0) # Speed scaling
        vy = float(actions[i][1] * 2.0)
        client.moveByVelocityAsync(vx, vy, 0, 0.1, vehicle_name=name)
        
    print(f"Flying... Avg Sep: {np.mean(adj_matrix)*10:.1f}m")
    time.sleep(0.05)