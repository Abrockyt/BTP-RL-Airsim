"""
Minimal test: Does Drone1 actually move?
This will make Drone1 fly to position (20, 20) at 20m altitude
"""
import airsim
import time
import numpy as np

print("="*60)
print("DRONE1 MOVEMENT TEST")
print("="*60)

# Connect
print("\n1. Connecting...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("✓ Connected")

# Enable control
print("\n2. Taking control of Drone1...")
client.enableApiControl(True, vehicle_name="Drone1")
client.armDisarm(True, vehicle_name="Drone1")
print("✓ Drone1 armed")

# Takeoff
print("\n3. Taking off...")
takeoff = client.takeoffAsync(vehicle_name="Drone1")
takeoff.join()
print("✓ Airborne")

# Get initial position
state = client.getMultirotorState(vehicle_name="Drone1")
pos = state.kinematics_estimated.position
print(f"\n4. Initial position: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})")

# Move to altitude
print("\n5. Climbing to 20m altitude...")
client.moveToZAsync(-20.0, 5.0, vehicle_name="Drone1").join()
time.sleep(1)

state = client.getMultirotorState(vehicle_name="Drone1")
pos = state.kinematics_estimated.position
print(f"   Current position: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})")

# Move to goal
GOAL_X = 20.0
GOAL_Y = 20.0
print(f"\n6. Moving to goal ({GOAL_X}, {GOAL_Y})...")

for step in range(50):
    state = client.getMultirotorState(vehicle_name="Drone1")
    pos = state.kinematics_estimated.position
    
    current_pos = np.array([pos.x_val, pos.y_val])
    goal_pos = np.array([GOAL_X, GOAL_Y])
    distance = np.linalg.norm(goal_pos - current_pos)
    
    if step % 5 == 0:
        print(f"   Step {step:2d}: Pos=({pos.x_val:5.1f}, {pos.y_val:5.1f})  Distance={distance:5.1f}m")
    
    if distance < 2.0:
        print(f"\n✓ GOAL REACHED at step {step}!")
        break
    
    # Calculate direction
    direction = (goal_pos - current_pos) / distance
    speed = min(8.0, distance * 0.4)
    
    # Calculate waypoint
    waypoint_dist = speed * 0.15
    target_x = pos.x_val + (direction[0] * waypoint_dist)
    target_y = pos.y_val + (direction[1] * waypoint_dist)
    
    # Move
    client.moveToPositionAsync(
        float(target_x), 
        float(target_y), 
        -20.0,  # Keep at 20m altitude
        float(speed), 
        duration=0.15,
        vehicle_name="Drone1"
    )
    
    time.sleep(0.15)

# Land
print("\n7. Landing...")
client.landAsync(vehicle_name="Drone1").join()
client.armDisarm(False, vehicle_name="Drone1")
client.enableApiControl(False, vehicle_name="Drone1")
print("✓ Test complete")

print("\n" + "="*60)
print("If you see position changing above, movement is working!")
print("If position stays at (0, 0), check:")
print("  1. AirSim settings.json has Drone1 configured")
print("  2. Unreal Engine is focused (not minimized)")
print("  3. Physics simulation is enabled")
print("="*60)
