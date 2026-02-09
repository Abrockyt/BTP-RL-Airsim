import airsim
import cv2
import numpy as np
import torch
import time
import math

# Import your Custom Brain
from mha_ppo_agent import PPO_Agent

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_PATH = "mha_ppo_1M_steps.pth"  # The file you are training on Kaggle
GOAL_POS = np.array([120.0, 120.0])  # Long distance goal
CRUISE_SPEED = 8.0                   # Fast flight (m/s)
MIN_HEIGHT = -2.0                    # Don't hit bushes (negative is up)

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Load the Trained Brain
print("🧠 Loading 1-Million-Step Energy Brain...")
agent = PPO_Agent(state_dim=7, action_dim=3, lr=0, gamma=0, K_epochs=0)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # Handle different save formats
    if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    else:
        agent.actor.load_state_dict(checkpoint)
    agent.actor.eval()
    print("✅ Brain Loaded! Energy Saving Mode: ON")
except:
    print("⚠️ Warning: Model not found. Using random weights for testing.")

print("🛫 Taking Off...")
client.takeoffAsync().join()
client.moveToZAsync(MIN_HEIGHT, 1).join() # Go to safe height

# =========================================================
# HELPER: COMPUTER VISION "EYES"
# =========================================================
def get_vision_prediction():
    # 1. Get Depth Image
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)])
    if not responses: return "SAFE"
    
    depth_img = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
    
    # 2. Process Regions (Top, Center, Left, Right)
    h, w = depth_img.shape
    center_box = depth_img[h//3:2*h//3, w//3:2*w//3]
    left_box = depth_img[h//3:2*h//3, 0:w//3]
    right_box = depth_img[h//3:2*h//3, 2*w//3:w]
    
    # Calculate average distance in each zone
    # AirSim depth is in meters. 100.0 means 'far away'
    dist_c = np.min(center_box) # Closest object in center
    dist_l = np.mean(left_box)
    dist_r = np.mean(right_box)
    
    # 3. COLLISION PREDICTION LOGIC
    # Rule A: HOUSE DETECTION (Wall ahead)
    if dist_c < 8.0 and dist_l < 8.0 and dist_r < 8.0:
        return "HOUSE"
    
    # Rule B: POLE/TREE DETECTION (Center blocked, sides open)
    if dist_c < 10.0:
        if dist_l > dist_r: return "POLE_RIGHT" # Left is open, go Left (stored as 'Right' obstacle)
        else: return "POLE_LEFT"  # Right is open, go Right
        
    return "SAFE"

# =========================================================
# MAIN FLIGHT LOOP
# =========================================================
battery = 1.0
print(f"🚀 Flying to Goal: {GOAL_POS}")

while True:
    loop_start = time.time()
    
    # 1. SENSE: What do we see?
    obstacle_type = get_vision_prediction()
    
    # 2. REACT: Emergency Reflexes (Priority 1)
    if obstacle_type == "HOUSE":
        print("⚠️ HOUSE DETECTED! CLIMBING UP!")
        # Fly Up Fast (Z decreases to go up)
        client.moveByVelocityBodyFrameAsync(2, 0, -3, 0.5).join()
        continue # Skip RL this step
        
    elif obstacle_type == "POLE_LEFT":
        print("⚠️ POLE DETECTED! SWERVING RIGHT!")
        client.moveByVelocityBodyFrameAsync(4, 4, 0, 0.5).join()
        continue
        
    elif obstacle_type == "POLE_RIGHT":
        print("⚠️ POLE DETECTED! SWERVING LEFT!")
        client.moveByVelocityBodyFrameAsync(4, -4, 0, 0.5).join()
        continue

    # 3. NAVIGATE: Energy Saving Brain (Priority 2)
    # Get State for RL
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    
    # Construct 7-value state vector
    current_pos = np.array([pos.x_val, pos.y_val])
    dist_vector = GOAL_POS - current_pos
    velocity = np.array([vel.x_val, vel.y_val])
    wind = np.array([0.5, -0.5]) # Simulated wind annoyance
    
    state_vec = np.concatenate([dist_vector, velocity, wind, [battery]]).astype(np.float32)
    
    # AI Decides
    action = agent.select_action(state_vec) # Returns [Vx, Vy, Yaw]
    
    # Scale Action (The AI outputs -1 to 1, we scale to Cruise Speed)
    target_vx = action[0] * CRUISE_SPEED
    target_vy = action[1] * CRUISE_SPEED
    
    # Execute Smooth Move
    client.moveByVelocityAsync(float(target_vx), float(target_vy), 0, 0.1).join()
    
    # Physics Calculation (Battery Drain)
    speed = np.linalg.norm([vel.x_val, vel.y_val])
    power = 200 * (1 + 0.005 * speed**2)
    battery -= power * 0.00001
    
    print(f"✅ Path Clear | Bat: {battery*100:.1f}% | Speed: {speed:.1f} m/s")
    
    # Goal Check
    if np.linalg.norm(dist_vector) < 5.0:
        print("🏆 GOAL REACHED! LANDING.")
        client.landAsync().join()
        break