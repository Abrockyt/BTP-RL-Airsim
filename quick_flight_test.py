"""
Quick Flight Test - No Vision, Just Direct Movement
Tests if the basic movement logic works without vision/GUI overhead
"""
import airsim
import numpy as np
import time

print("="*60)
print("QUICK FLIGHT TEST - Direct Movement Only")
print("="*60)

# Connect
print("\n1. Connecting to Drone1...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name="Drone1")
client.armDisarm(True, vehicle_name="Drone1")
print("✓ Connected and armed")

# Takeoff
print("\n2. Taking off...")
client.takeoffAsync(vehicle_name="Drone1").join()
client.moveToZAsync(-20.0, 5.0, vehicle_name="Drone1").join()
time.sleep(1)
print("✓ At 20m altitude")

# Get initial position
state = client.getMultirotorState(vehicle_name="Drone1")
pos = state.kinematics_estimated.position
print(f"✓ Starting position: ({pos.x_val:.1f}, {pos.y_val:.1f})")

# Set goal
GOAL_X = 50.0
GOAL_Y = 50.0
print(f"\n3. Flying to goal ({GOAL_X}, {GOAL_Y})...")

# Flight loop
for step in range(200):
    # Get current position
    state = client.getMultirotorState(vehicle_name="Drone1")
    if state is None:
        print(f"   Step {step}: ❌ Failed to get state")
        time.sleep(0.15)
        continue
        
    pos = state.kinematics_estimated.position
    current_pos = np.array([pos.x_val, pos.y_val])
    goal_pos = np.array([GOAL_X, GOAL_Y])
    distance = np.linalg.norm(goal_pos - current_pos)
    
    # Print progress every 10 steps
    if step % 10 == 0:
        print(f"   Step {step:3d}: Pos=({pos.x_val:5.1f}, {pos.y_val:5.1f})  Dist={distance:5.1f}m")
    
    # Check if reached
    if distance < 3.0:
        print(f"\n✓ GOAL REACHED at step {step}!")
        break
    
    # Calculate movement
    direction = (goal_pos - current_pos) / distance
    speed = min(8.0, max(3.0, distance * 0.4))
    waypoint_dist = speed * 0.15
    target_x = pos.x_val + (direction[0] * waypoint_dist)
    target_y = pos.y_val + (direction[1] * waypoint_dist)
    
    # Move
    client.moveToPositionAsync(
        float(target_x),
        float(target_y),
        -20.0,
        float(speed),
        duration=0.15,
        vehicle_name="Drone1"
    )
    
    time.sleep(0.15)

# Land
print("\n4. Landing...")
client.landAsync(vehicle_name="Drone1").join()
client.armDisarm(False, vehicle_name="Drone1")
client.enableApiControl(False, vehicle_name="Drone1")

print("\n" + "="*60)
print("✓ QUICK FLIGHT TEST COMPLETE")
print("="*60)
print("\nIf this worked but GUI doesn't, the issue is in:")
print("  - Vision processing (get_depth_perception)")
print("  - GUI updates (update_vision_display)")
print("  - Telemetry loop interaction")
