"""
AirSim Drone Autonomous Flight using Car Steering Model
Converts steering predictions to drone lateral velocity control
"""

import airsim
import torch
import torch.nn as nn
import numpy as np
import cv2
import time


def load_model(model_path, device):
    """Load the model - checkpoint was saved as bare Sequential"""
    model = nn.Sequential(
        nn.Conv2d(3, 24, 5, 2),      # 0
        nn.ReLU(),                    # 1
        nn.Conv2d(24, 36, 5, 2),     # 2
        nn.ReLU(),                    # 3
        nn.Conv2d(36, 48, 5, 2),     # 4
        nn.ReLU(),                    # 5
        nn.Conv2d(48, 64, 3),        # 6
        nn.ReLU(),                    # 7
        nn.Flatten(),                 # 8
        nn.Linear(3840, 100),        # 9
        nn.ReLU(),                    # 10
        nn.Linear(100, 50),          # 11
        nn.ReLU(),                    # 12
        nn.Linear(50, 1)             # 13
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_response):
    """
    Preprocess AirSim image for model input
    Resize to (66, 200) and normalize
    """
    # Get numpy array from AirSim response
    img1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(image_response.height, image_response.width, 3)
    
    # Resize to model input size (66, 200)
    img_resized = cv2.resize(img_rgb, (200, 66))
    
    # Convert to float and normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def main():
    print("="*70)
    print("🚁 AIRSIM DRONE AUTONOMOUS FLIGHT")
    print("="*70)
    
    # Load the trained model
    model_path = "my_airsim_model.pth"
    print(f"\n📂 Loading model from: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    model = load_model(model_path, device)
    print("✓ Model loaded successfully")
    
    # Connect to AirSim
    print("\n🔌 Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✓ Connected to AirSim Multirotor")
    
    # Takeoff
    print("\n🛫 Taking off...")
    client.takeoffAsync().join()
    print("✓ Takeoff complete")
    
    # Hover and stabilize
    print("⏸️  Hovering for 2 seconds to stabilize...")
    time.sleep(2)
    
    # Flight parameters
    FORWARD_VELOCITY = 2.0  # m/s forward speed
    TARGET_ALTITUDE = -5.0  # Target altitude in NED (5m above ground)
    STEERING_SCALE = 3.0    # Scale steering output to lateral velocity
    ALTITUDE_KP = 0.5       # Proportional gain for altitude control
    
    print(f"\n🚀 Starting autonomous flight...")
    print(f"   Forward velocity: {FORWARD_VELOCITY} m/s")
    print(f"   Target altitude: {-TARGET_ALTITUDE} m above ground")
    print(f"   Steering scale: {STEERING_SCALE}x")
    print("Press Ctrl+C to land and stop")
    print("-"*70)
    
    last_position = None
    stuck_counter = 0
    last_collision_time = 0
    collision_cooldown = 3.0  # 3 seconds before checking collisions again
    
    try:
        step = 0
        with torch.no_grad():
            while True:
                # Get current state
                current_state = client.getMultirotorState()
                current_position = current_state.kinematics_estimated.position
                
                # Check for collision (with cooldown to prevent repeated triggers)
                current_time = time.time()
                collision_info = client.simGetCollisionInfo()
                
                if collision_info.has_collided and (current_time - last_collision_time) > collision_cooldown:
                    print(f"\n⚠️  COLLISION DETECTED at step {step}!")
                    last_collision_time = current_time
                    
                    # Multi-step recovery sequence
                    print("   → Step 1: Stopping...")
                    client.moveByVelocityAsync(0, 0, 0, 0.5).join()
                    time.sleep(0.5)
                    
                    print("   → Step 2: Moving backward...")
                    client.moveByVelocityBodyFrameAsync(-1.5, 0, 0, 2.0).join()
                    time.sleep(0.5)
                    
                    print("   → Step 3: Climbing to safe altitude...")
                    client.moveToZAsync(TARGET_ALTITUDE, 2.0).join()
                    time.sleep(1.0)
                    
                    print("   → Step 4: Clearing collision state...")
                    # Reset collision by getting fresh collision info
                    for _ in range(5):
                        client.simGetCollisionInfo()
                        time.sleep(0.1)
                    
                    print("   ✓ Recovery complete, resuming flight\n")
                    step += 1
                    continue
                
                # Detect if stuck (position hasn't changed)
                if last_position is not None:
                    distance_moved = np.sqrt(
                        (current_position.x_val - last_position.x_val)**2 +
                        (current_position.y_val - last_position.y_val)**2
                    )
                    if distance_moved < 0.1:  # Less than 10cm movement
                        stuck_counter += 1
                        if stuck_counter > 50:  # Stuck for 5 seconds
                            print(f"\n⚠️  STUCK DETECTED at step {step}! Moving to safe position...")
                            # Move up and forward
                            client.moveToZAsync(TARGET_ALTITUDE, 2.0).join()
                            time.sleep(1)
                            stuck_counter = 0
                            print("   ✓ Unstuck, resuming flight")
                            step += 1
                            continue
                    else:
                        stuck_counter = 0
                
                last_position = current_position
                
                # Get image from front camera
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                ])
                
                if responses and len(responses[0].image_data_uint8) > 0:
                    # Preprocess image
                    image_tensor = preprocess_image(responses[0])
                    image_tensor = image_tensor.to(device)
                    
                    # Get steering prediction from model
                    steering_output = model(image_tensor).item()
                    
                    # Convert steering to lateral velocity (vy)
                    vy = float(np.clip(steering_output * STEERING_SCALE, -3.0, 3.0))
                    
                    # Altitude control - maintain target altitude
                    current_altitude = current_position.z_val
                    altitude_error = TARGET_ALTITUDE - current_altitude
                    vz = float(np.clip(altitude_error * ALTITUDE_KP, -1.0, 1.0))
                    
                    # Send velocity command in body frame
                    client.moveByVelocityBodyFrameAsync(
                        FORWARD_VELOCITY,  # vx - forward
                        vy,                # vy - lateral (steering)
                        vz,                # vz - altitude control
                        duration=0.1       # Command duration
                    )
                    
                    # Print status every 10 steps
                    if step % 10 == 0:
                        altitude_m = -current_altitude
                        print(f"Step {step:4d} | "
                              f"Pos: ({current_position.x_val:6.2f}, {current_position.y_val:6.2f}) | "
                              f"Alt: {altitude_m:5.2f}m | "
                              f"Steering: {steering_output:+.3f} → Vy: {vy:+.3f}")
                    
                    step += 1
                    time.sleep(0.05)
                else:
                    print("⚠️  No image received, retrying...")
                    time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Land safely
        print("\n🛬 Landing...")
        client.landAsync().join()
        print("✓ Landed")
        
        # Cleanup
        client.armDisarm(False)
        client.enableApiControl(False)
        print("✓ API control disabled")
        print("="*70)


if __name__ == "__main__":
    import os
    if not os.path.exists("my_airsim_model.pth"):
        print("❌ Error: my_airsim_model.pth not found in current directory!")
        print("Please make sure the model file is in the same folder as this script.")
        exit(1)
    
    main()
