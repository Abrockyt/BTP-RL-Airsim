import airsim
import torch
import torch.nn as nn
import numpy as np
import time

# =========================================================
# 1. DEFINE BRAIN ARCHITECTURE (Must match training)
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
# 2. SETUP AIRSIM & LOAD BRAIN
# =========================================================
# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone1")
client.enableApiControl(True, "Drone2")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Drone2")

# Takeoff
print("🛫 Swarm Taking Off...")
f1 = client.takeoffAsync(vehicle_name="Drone1")
f2 = client.takeoffAsync(vehicle_name="Drone2")
f1.join()
f2.join()

client.moveToZAsync(-3, 1, "Drone1")
client.moveToZAsync(-3, 1, "Drone2").join()

# Load GNN Brain
print("🧠 Loading GNN Brain...")
model = GNNActor(state_dim=4, action_dim=2)
try:
    # Use load_state_dict to load the trained weights
    # We navigate carefully to handle different save formats
    state_dict = torch.load("gnn_swarm_brain.pth")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model Loaded!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Using random weights (drones might crash).")

# =========================================================
# 3. SWARM EXECUTION LOOP
# =========================================================
# Goals: Drone 1 goes to (10, 0), Drone 2 goes to (0, 0)
GOALS = { "Drone1": np.array([10.0, 0.0]), "Drone2": np.array([0.0, 0.0]) }

print("🚀 Executing Swarm Maneuver...")

while True:
    # 1. Get States
    s1 = client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated
    s2 = client.getMultirotorState(vehicle_name="Drone2").kinematics_estimated
    
    pos1 = np.array([s1.position.x_val, s1.position.y_val])
    pos2 = np.array([s2.position.x_val, s2.position.y_val])
    vel1 = np.array([s1.linear_velocity.x_val, s1.linear_velocity.y_val])
    vel2 = np.array([s2.linear_velocity.x_val, s2.linear_velocity.y_val])
    
    # 2. Prepare Graph Input [Batch=1, Nodes=2, Features=4]
    # State: [Goal_X, Goal_Y, Vel_X, Vel_Y]
    node1 = np.concatenate([(GOALS["Drone1"] - pos1)/50.0, vel1/5.0])
    node2 = np.concatenate([(GOALS["Drone2"] - pos2)/50.0, vel2/5.0])
    
    state_tensor = torch.FloatTensor([node1, node2]).unsqueeze(0)
    
    # Adjacency Matrix (They are close, so they are connected)
    adj_tensor = torch.FloatTensor([[1, 1], [1, 1]]).unsqueeze(0)
    
    # 3. GNN Prediction
    with torch.no_grad():
        action_mean, _ = model(state_tensor, adj_tensor)
        actions = action_mean.squeeze(0).numpy()
        
    # 4. Apply Action
    v1 = actions[0] * 3.0 # Scale to max speed 3 m/s
    v2 = actions[1] * 3.0
    
    client.moveByVelocityAsync(float(v1[0]), float(v1[1]), 0, 0.1, vehicle_name="Drone1")
    client.moveByVelocityAsync(float(v2[0]), float(v2[1]), 0, 0.1, vehicle_name="Drone2")
    
    # Distance Check
    dist = np.linalg.norm(pos1 - pos2)
    print(f"Separation: {dist:.2f}m")
    
    if dist < 1.0:
        print("💥 CRASH DETECTED!")
        break
        
    time.sleep(0.05)