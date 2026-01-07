"""
Smart Drone Control with PyTorch Model
Uses trained steering model with adaptive speed and intelligent collision recovery
Includes interactive map GUI with restart functionality
"""

import airsim
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import messagebox
import threading


class DronePilot(nn.Module):
    """Model architecture - MUST MATCH training structure"""
    def __init__(self):
        super(DronePilot, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3)  # Required layer!
        )
        self.classifier = nn.Sequential(
            nn.Linear(3840, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path, device):
    """Load the trained model"""
    print(f"📂 Loading model from: {model_path}")
    model = DronePilot()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"🖥️  Using device: {device}")
    print("✓ Model loaded successfully")
    return model


def preprocess_image(image_response):
    """Preprocess AirSim image for model input"""
    # Convert to numpy array
    img1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(image_response.height, image_response.width, 3)
    
    # Resize to model input size (66, 200)
    img_resized = cv2.resize(img_rgb, (200, 66))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
    
    return img_tensor.unsqueeze(0)  # Add batch dimension


def smart_speed_control(steering_angle, base_slow=2.0, base_fast=5.0, threshold=0.15):
    """
    Smart speed control: slow down for sharp turns, speed up for straight paths
    
    Args:
        steering_angle: Model output steering angle
        base_slow: Speed for sharp turns (m/s)
        base_fast: Speed for straight driving (m/s)
        threshold: Angle threshold to switch between fast/slow
    
    Returns:
        forward_speed: Appropriate forward velocity
    """
    abs_angle = abs(steering_angle)
    
    if abs_angle > threshold:
        # Sharp turn - slow down
        return base_slow
    else:
        # Straight or gentle turn - speed up
        # Interpolate between slow and fast based on angle
        speed_factor = 1.0 - (abs_angle / threshold)
        return base_slow + (base_fast - base_slow) * speed_factor


def collision_recovery(client, altitude_gain=2.0):
    """
    DEPRECATED: Simple collision recovery - use smart_collision_recovery instead
    Intelligent collision recovery - climb over obstacle
    
    Args:
        client: AirSim MultirotorClient
        altitude_gain: How much to climb (meters)
    """
    print("⚠️  COLLISION DETECTED!")
    print("  → Step 1: Stopping...")
    client.moveByVelocityAsync(0, 0, 0, 0.5).join()
    
    print(f"  → Step 2: Climbing {altitude_gain}m to clear obstacle...")
    current_state = client.getMultirotorState()
    current_z = current_state.kinematics_estimated.position.z_val
    target_z = current_z - altitude_gain  # NED: negative is up
    client.moveToZAsync(target_z, 2.0).join()
    
    print("  → Step 3: Backing up slightly...")
    client.moveByVelocityBodyFrameAsync(-1.0, 0, 0, 1.0).join()
    
    print("  → Step 4: Clearing collision state...")
    for _ in range(5):
        client.simGetCollisionInfo()
        time.sleep(0.1)
    
    print("✓ Recovery complete - resuming flight")


def analyze_obstacle(client):
    """
    Analyze obstacle using depth camera to determine best recovery direction
    
    Returns:
        str: Recovery direction - "UP", "LEFT", "RIGHT", or "BACK"
    """
    try:
        # Get depth image from front camera
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
        ])
        
        if responses and len(responses[0].image_data_float) > 0:
            depth_data = np.array(responses[0].image_data_float)
            depth_data = depth_data.reshape(responses[0].height, responses[0].width)
            
            # Analyze depth regions (divide into zones)
            h, w = depth_data.shape
            
            # Check center-front (likely collision point)
            center_depth = np.mean(depth_data[h//3:2*h//3, w//3:2*w//3])
            
            # Check left and right sides
            left_depth = np.mean(depth_data[h//3:2*h//3, :w//3])
            right_depth = np.mean(depth_data[h//3:2*h//3, 2*w//3:])
            
            # Check upper area (sky/clearance above)
            upper_depth = np.mean(depth_data[:h//3, :])
            
            # Check lower area (ground/bushes)
            lower_depth = np.mean(depth_data[2*h//3:, :])
            
            # Decision logic
            # If upper area is clear (high depth) and center blocked → climb (building/wall)
            if upper_depth > center_depth * 1.5 and upper_depth > 10:
                return "UP"
            
            # If lower area blocked (bushes/ground) → go up or sideways, NOT down
            if lower_depth < 3:
                if left_depth > right_depth and left_depth > center_depth:
                    return "LEFT"
                elif right_depth > left_depth and right_depth > center_depth:
                    return "RIGHT"
                else:
                    return "UP"  # Climb if both sides blocked
            
            # If left is more clear than right → go left (tree on right)
            if left_depth > right_depth * 1.3 and left_depth > center_depth:
                return "LEFT"
            
            # If right is more clear than left → go right (tree on left)
            if right_depth > left_depth * 1.3 and right_depth > center_depth:
                return "RIGHT"
            
        # Default: back up if can't determine
        return "BACK"
        
    except Exception as e:
        # If depth analysis fails, default to backing up
        return "BACK"


def smart_collision_recovery(client):
    """
    Smart collision recovery based on obstacle analysis
    - Houses/walls: Climb over
    - Trees/poles: Move left or right
    - Bushes/ground: Never go down, go sideways or up
    """
    print("⚠️  COLLISION DETECTED! Analyzing obstacle...")
    
    # Stop immediately
    client.moveByVelocityAsync(0, 0, 0, 0.2)
    time.sleep(0.3)
    
    # Analyze obstacle direction
    recovery_direction = analyze_obstacle(client)
    
    if recovery_direction == "UP":
        print("  → Large obstacle (building/wall) - climbing over")
        client.moveByVelocityAsync(0, 0, -2.5, 1.5)  # Fast climb
        time.sleep(1.6)
    elif recovery_direction == "LEFT":
        print("  → Side obstacle (tree/pole) - moving left")
        client.moveByVelocityAsync(-1.5, -2.5, -0.5, 1.2)  # Back-left + slight up
        time.sleep(1.3)
    elif recovery_direction == "RIGHT":
        print("  → Side obstacle (tree/pole) - moving right")
        client.moveByVelocityAsync(-1.5, 2.5, -0.5, 1.2)  # Back-right + slight up
        time.sleep(1.3)
    else:  # BACK
        print("  → Ground/complex obstacle - reversing and climbing")
        client.moveByVelocityAsync(-2.5, 0, -1.5, 1.2)  # Fast back + up
        time.sleep(1.3)
    
    # Clear collision state
    for _ in range(3):
        try:
            client.simGetCollisionInfo()
        except:
            pass
        time.sleep(0.1)
    
    print("  ✓ Recovery complete")


def main():
    # Configuration
    MODEL_PATH = "smart_airsim_model .pth"  # Note: space before .pth
    STARTING_ALTITUDE = -3.0  # Car-like view (3m above ground)
    YAW_RATE_SCALE = 60.0  # Steering to yaw rate conversion (deg/s)
    ALTITUDE_KP = 0.8  # Altitude control gain
    COLLISION_COOLDOWN = 3.0  # Seconds between collision recoveries
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Connect to AirSim
    print("\n🔌 Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")
    print(f"Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)")
    
    # Enable API control and arm
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Takeoff
    print("\n🛫 Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Move to starting altitude
    print(f"📏 Moving to altitude: {-STARTING_ALTITUDE}m above ground...")
    client.moveToZAsync(STARTING_ALTITUDE, 2.0).join()
    time.sleep(1)
    
    print("\n" + "="*60)
    print("🚀 SMART AUTONOMOUS FLIGHT STARTED")
    print("="*60)
    print(f"⚡ Adaptive Speed: 2-5 m/s based on steering")
    print(f"🎯 Target Altitude: {-STARTING_ALTITUDE}m")
    print(f"🛡️  Collision Recovery: Climb over obstacles")
    print("Press Ctrl+C to stop\n")
    
    step = 0
    last_collision_time = 0
    
    try:
        with torch.no_grad():
            while True:
                current_time = time.time()
                
                # Collision check with smart recovery
                collision_info = client.simGetCollisionInfo()
                if collision_info.has_collided and (current_time - last_collision_time) > COLLISION_COOLDOWN:
                    last_collision_time = current_time
                    smart_collision_recovery(client)
                    continue
                
                # Get current state
                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                current_altitude = -pos.z_val  # Convert NED to altitude above ground
                
                # Get camera image
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                ])
                
                if not responses or len(responses[0].image_data_uint8) == 0:
                    print("⚠️  No image received, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Preprocess and predict
                image_tensor = preprocess_image(responses[0])
                image_tensor = image_tensor.to(device)
                
                steering_angle = model(image_tensor).item()
                
                # Smart speed control
                forward_speed = smart_speed_control(steering_angle)
                
                # Convert steering to yaw rate
                yaw_rate = np.clip(steering_angle * YAW_RATE_SCALE, -60.0, 60.0)  # deg/s
                
                # Altitude control (maintain target altitude)
                altitude_error = STARTING_ALTITUDE - pos.z_val
                vz = float(np.clip(altitude_error * ALTITUDE_KP, -1.5, 1.5))
                
                # Send control command (body frame for forward, yaw mode for turning)
                client.moveByVelocityBodyFrameAsync(
                    forward_speed,  # vx - forward
                    0,              # vy - lateral (0 for car-like)
                    vz,             # vz - altitude control
                    duration=0.1,
                    yaw_mode=airsim.YawMode(True, yaw_rate)  # is_rate=True
                )
                
                # Display status every 10 steps
                if step % 10 == 0:
                    speed_label = "🐢 SLOW" if forward_speed < 3.5 else "🚀 FAST"
                    print(f"Step {step:4d} | Pos: ({pos.x_val:6.1f}, {pos.y_val:6.1f}) | "
                          f"Alt: {current_altitude:4.1f}m | Steering: {steering_angle:+6.3f} → "
                          f"Yaw: {yaw_rate:+5.1f}°/s | Speed: {forward_speed:4.1f} m/s {speed_label}")
                
                step += 1
                time.sleep(0.05)  # 20 Hz control loop
                
    except KeyboardInterrupt:
        print("\n\n🛑 Flight interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Land safely
        print("\n🛬 Landing...")
        try:
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            client.landAsync().join()
            time.sleep(2)
        except:
            pass
        
        # Cleanup
        client.armDisarm(False)
        client.enableApiControl(False)
        print("✓ Disconnected safely")


if __name__ == "__main__":
    main()
