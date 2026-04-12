"""
SIMPLE GNN SWARM SYSTEM
- Background drones move randomly
- Main drone (Drone1) uses GNN to reach goal while communicating with others
- Clean, fast, no complexity
"""
import airsim
import torch
import torch.nn as nn
import numpy as np
import time
import threading

# ============================================================================
# GNN ARCHITECTURE
# ============================================================================
class GNN_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN_Layer, self).__init__()
        self.message_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, node_features, adj_matrix):
        # Message passing
        messages = self.message_net(node_features)
        aggregated = torch.matmul(adj_matrix, messages)
        combined = torch.cat([node_features, aggregated], dim=2)
        return self.update_net(combined)

class GNN_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GNN_Actor, self).__init__()
        self.embedding = nn.Linear(state_dim, hidden_dim)
        self.gnn1 = GNN_Layer(hidden_dim, hidden_dim)
        self.gnn2 = GNN_Layer(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, state, adj):
        x = torch.relu(self.embedding(state))
        x = torch.relu(self.gnn1(x, adj))
        x = self.gnn2(x, adj)
        return self.tanh(self.mean_layer(x))

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_DRONES = 10
MAIN_DRONE = "Drone1"
BACKGROUND_DRONES = [f"Drone{i}" for i in range(2, NUM_DRONES + 1)]
GOAL = np.array([100.0, 100.0])  # Goal position
ALTITUDE = -20.0  # 20m above ground
COMM_RANGE = 30.0  # Communication range in meters

# ============================================================================
# RANDOM MOVEMENT FOR BACKGROUND DRONES
# ============================================================================
def random_movement_thread(drone_name, stop_event):
    """Make drone move randomly"""
    print(f"🎲 {drone_name} starting random movement...")
    
    try:
        # Create separate client for this thread
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        # Takeoff
        client.enableApiControl(True, vehicle_name=drone_name)
        client.armDisarm(True, vehicle_name=drone_name)
        client.takeoffAsync(vehicle_name=drone_name).join()
        client.moveToZAsync(ALTITUDE, 3.0, vehicle_name=drone_name).join()
        
        # Random movement loop
        while not stop_event.is_set():
            # Get current position
            state = client.getMultirotorState(vehicle_name=drone_name)
            pos = state.kinematics_estimated.position
            
            # Random target within 50m radius
            dx = np.random.uniform(-50, 50)
            dy = np.random.uniform(-50, 50)
            target_x = pos.x_val + dx
            target_y = pos.y_val + dy
            
            # Move to random position
            client.moveToPositionAsync(
                target_x, target_y, ALTITUDE, 
                np.random.uniform(3.0, 8.0),  # Random speed
                vehicle_name=drone_name
            )
            
            time.sleep(np.random.uniform(3.0, 8.0))  # Random wait
            
    except Exception as e:
        print(f"⚠️ {drone_name} error: {e}")

# ============================================================================
# GNN-BASED MAIN DRONE
# ============================================================================
def gnn_main_drone(client, model, stop_event):
    """Main drone uses GNN to navigate and communicate"""
    print(f"🧠 {MAIN_DRONE} starting GNN navigation...")
    
    try:
        # Takeoff
        client.enableApiControl(True, vehicle_name=MAIN_DRONE)
        client.armDisarm(True, vehicle_name=MAIN_DRONE)
        client.takeoffAsync(vehicle_name=MAIN_DRONE).join()
        client.moveToZAsync(ALTITUDE, 3.0, vehicle_name=MAIN_DRONE).join()
        
        print(f"🎯 Goal: ({GOAL[0]}, {GOAL[1]})")
        
        step = 0
        while not stop_event.is_set():
            # ============ GET ALL DRONE POSITIONS ============
            positions = {}
            velocities = {}
            
            # Main drone
            state_main = client.getMultirotorState(vehicle_name=MAIN_DRONE)
            pos_main = state_main.kinematics_estimated.position
            vel_main = state_main.kinematics_estimated.linear_velocity
            positions[MAIN_DRONE] = np.array([pos_main.x_val, pos_main.y_val])
            velocities[MAIN_DRONE] = np.array([vel_main.x_val, vel_main.y_val])
            
            # Background drones
            for drone in BACKGROUND_DRONES:
                try:
                    state = client.getMultirotorState(vehicle_name=drone)
                    pos = state.kinematics_estimated.position
                    vel = state.kinematics_estimated.linear_velocity
                    positions[drone] = np.array([pos.x_val, pos.y_val])
                    velocities[drone] = np.array([vel.x_val, vel.y_val])
                except:
                    pass  # Drone might not be active
            
            # ============ BUILD GNN INPUT ============
            all_drones = [MAIN_DRONE] + list(positions.keys())[1:]
            num_active = len(all_drones)
            
            # Node features: [goal_rel_x, goal_rel_y, vel_x, vel_y]
            node_features = []
            for drone in all_drones:
                if drone == MAIN_DRONE:
                    goal_rel = (GOAL - positions[drone]) / 100.0  # Normalize
                else:
                    goal_rel = np.zeros(2)  # Background drones don't have goals
                
                vel_norm = velocities[drone] / 10.0
                feature = np.concatenate([goal_rel, vel_norm])
                node_features.append(feature)
            
            state_tensor = torch.FloatTensor(node_features).unsqueeze(0)
            
            # Adjacency matrix (communication graph based on distance)
            adj_matrix = np.eye(num_active)
            comm_count = 0
            for i, drone_i in enumerate(all_drones):
                for j, drone_j in enumerate(all_drones):
                    if i != j:
                        dist = np.linalg.norm(positions[drone_i] - positions[drone_j])
                        if dist < COMM_RANGE:
                            adj_matrix[i, j] = 1.0
                            if i == 0:  # Main drone communicating
                                comm_count += 1
            
            adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0)
            
            # ============ GNN INFERENCE ============
            with torch.no_grad():
                actions = model(state_tensor, adj_tensor)
                main_action = actions.squeeze(0)[0].numpy()  # First drone is main
            
            # ============ EXECUTE ACTION ============
            distance_to_goal = np.linalg.norm(GOAL - positions[MAIN_DRONE])
            
            if distance_to_goal < 5.0:
                print(f"\n🎉 GOAL REACHED in {step} steps!")
                break
            
            # Calculate direction to goal (simple goal-seeking bias)
            goal_direction = (GOAL - positions[MAIN_DRONE]) / (distance_to_goal + 1e-6)
            
            # Blend GNN action with goal-seeking (90% goal, 10% GNN communication)
            vx = float(goal_direction[0] * 0.9 + main_action[0] * 0.1) * 5.0  
            vy = float(goal_direction[1] * 0.9 + main_action[1] * 0.1) * 5.0
            
            # Calculate next position (2m step toward goal)
            target_x = pos_main.x_val + vx * 0.4
            target_y = pos_main.y_val + vy * 0.4
            
            # Move to position
            client.moveToPositionAsync(
                target_x, target_y, ALTITUDE,
                5.0,  # velocity
                vehicle_name=MAIN_DRONE
            ).join()
            
            # ============ STATUS UPDATE ============
            if step % 5 == 0:
                print(f"Step {step:3d} | "
                      f"Pos: ({positions[MAIN_DRONE][0]:6.1f}, {positions[MAIN_DRONE][1]:6.1f}) | "
                      f"Goal: {distance_to_goal:6.1f}m | "
                      f"Comm: {comm_count} drones | "
                      f"Active: {num_active}")
            
            step += 1
            
    except Exception as e:
        print(f"❌ Main drone error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("SIMPLE GNN SWARM SYSTEM")
    print("="*70)
    print(f"Main Drone: {MAIN_DRONE} (GNN navigation)")
    print(f"Background: {len(BACKGROUND_DRONES)} drones (random movement)")
    print(f"Goal: ({GOAL[0]}, {GOAL[1]})")
    print("="*70)
    
    # Connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("\n✓ Connected to AirSim")
    
    # Load GNN model
    print("🧠 Loading GNN model...")
    model = GNN_Actor(state_dim=4, action_dim=2)
    try:
        model.load_state_dict(torch.load("gnn_agent.pth", map_location='cpu'))
        print("✓ GNN model loaded")
    except:
        print("⚠️ No trained model found, using random weights")
    model.eval()
    
    # Stop event for threads
    stop_event = threading.Event()
    
    # Start background drones (random movement)
    print(f"\n🎲 Starting {len(BACKGROUND_DRONES)} background drones...")
    bg_threads = []
    for drone in BACKGROUND_DRONES[:6]:  # Use only 6 background drones
        thread = threading.Thread(
            target=random_movement_thread,
            args=(drone, stop_event),
            daemon=True
        )
        thread.start()
        bg_threads.append(thread)
        time.sleep(0.5)  # Stagger starts
    
    print("✓ Background drones active\n")
    
    # Wait for drones to stabilize
    time.sleep(3)
    
    # Start main drone (GNN)
    print("🚀 Starting main drone with GNN...\n")
    try:
        gnn_main_drone(client, model, stop_event)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
    
    # Stop all
    print("\n🛬 Landing all drones...")
    stop_event.set()
    
    # Land all
    for drone in [MAIN_DRONE] + BACKGROUND_DRONES[:6]:
        try:
            client.landAsync(vehicle_name=drone)
        except:
            pass
    
    time.sleep(3)
    
    # Cleanup
    for drone in [MAIN_DRONE] + BACKGROUND_DRONES[:6]:
        try:
            client.armDisarm(False, vehicle_name=drone)
            client.enableApiControl(False, vehicle_name=drone)
        except:
            pass
    
    print("✓ Complete")
