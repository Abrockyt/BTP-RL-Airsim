import airsim
import torch
import numpy as np
import time
import math

# Import the brain structure
from mha_ppo_agent import PPO_Agent

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_PATH = "energy_saving_brain.pth" # The file you downloaded
GOAL_POS = np.array([100.0, 100.0])    # Must match your training goal roughly
P_HOVER = 200.0                        # Physics constant

# =========================================================
# 1. SETUP CONNECTION
# =========================================================
print("🔌 Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("🛫 Taking off...")
client.takeoffAsync().join()

# =========================================================
# 2. LOAD THE BRAIN
# =========================================================
# We use state_dim=7 because that's what we trained on Kaggle
# (Dist_X, Dist_Y, Vel_X, Vel_Y, Wind_X, Wind_Y, Battery)
agent = PPO_Agent(state_dim=7, action_dim=3, lr=0, gamma=0, K_epochs=0)

try:
    # Load the weights
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Handle dictionary mismatch if necessary
    if 'actor_state_dict' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    else:
        agent.actor.load_state_dict(checkpoint)
    print("🧠 BRAIN LOADED: Ready for Energy-Efficient Flight!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you put 'energy_saving_brain.pth' in this folder?")
    exit()

agent.actor.eval() # Set to evaluation mode (no training)

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def get_state(client, goal, battery_level):
    # 1. Get Kinematics
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    
    # 2. Calculate Relative Distance
    # In AirSim, Z is negative for height (NED coordinates)
    current_pos = np.array([pos.x_val, pos.y_val])
    dist_vector = goal - current_pos
    
    # 3. Velocity
    velocity = np.array([vel.x_val, vel.y_val])
    
    # 4. Wind Estimation (Simulated for consistency)
    # In a real scenario, you'd calculate this via (Expected_Pos - Actual_Pos)
    # For this demo, we assume the drone sensors detect slight drift
    wind_x = vel.x_val - 0.0 # Simplified drift
    wind_y = vel.y_val - 0.0
    wind = np.array([wind_x, wind_y]) * 0.1 # Scaling factor
    
    # 5. Construct Vector (7 Values)
    state_vec = np.concatenate([
        dist_vector, # 2
        velocity,    # 2
        wind,        # 2
        [battery_level] # 1
    ]).astype(np.float32)
    
    return state_vec, current_pos

# =========================================================
# 4. FLIGHT LOOP
# =========================================================
battery = 1.0 # Start at 100%
dt = 0.1
print(f"🎯 Target Goal: {GOAL_POS}")

while True:
    start_time = time.time()
    
    # A. Get Data
    state_vec, current_pos = get_state(client, GOAL_POS, battery)
    dist = np.linalg.norm(GOAL_POS - current_pos)
    
    # B. AI Decides Action
    # The agent returns [Vel_X, Vel_Y, Yaw_Rate] normalized -1 to 1
    action = agent.select_action(state_vec)
    
    # Scale actions back to real world units
    target_vx = np.clip(action[0] * 10.0, -5, 5) # Max 5 m/s safety limit
    target_vy = np.clip(action[1] * 10.0, -5, 5)
    yaw_rate = action[2]
    
    # C. Execute Move
    # moveByVelocityBodyFrameAsync flies relative to the drone's nose
    # We use world frame here so we convert velocity
    client.moveByVelocityAsync(
        float(target_vx), 
        float(target_vy), 
        0, # Keep altitude 0 (relative) or fix Z
        duration=0.1
    ).join()
    
    # D. Simulate Battery Drain (Physics)
    speed = np.linalg.norm([target_vx, target_vy])
    power_watts = P_HOVER * (1 + 0.005 * speed**2)
    battery -= (power_watts * dt / 50000.0) # Drain battery
    
    # E. Logging
    print(f"🔋 Bat: {battery*100:.1f}% | 📏 Dist: {dist:.1f}m | 💨 Speed: {speed:.1f} m/s")
    
    if dist < 5.0:
        print("🏆 GOAL REACHED! The AI successfully navigated.")
        break
        
    if battery <= 0:
        print("🪫 Battery Depleted. Landing.")
        client.landAsync().join()
        break
        
    # Maintain loop rate
    elapsed = time.time() - start_time
    if elapsed < dt:
        time.sleep(dt - elapsed)