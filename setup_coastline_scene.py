import airsim
import random
import math
import time

def setup_coastline_environment():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    vehicle_name = "Drone1"
    client.enableApiControl(True, vehicle_name=vehicle_name)
    
    print("1. Teleporting Drone1 to the ocean (X: 200, Y: 150, Z: -20)...")
    water_pose = airsim.Pose(airsim.Vector3r(200.0, 150.0, -8.0), airsim.to_quaternion(0.0, 0.0, 0.0))
    client.simSetVehiclePose(water_pose, True, vehicle_name=vehicle_name)
    
    print("2. Taking off to establish physics...")
    client.takeoffAsync(vehicle_name=vehicle_name)
    
    print("3. Pitching camera completely downward (-90 degrees)...")
    down_pitch_rad = math.radians(-90.0)
    client.simSetCameraPose(
        "0",
        airsim.Pose(airsim.Vector3r(0.0, 0.0, 0.0), airsim.to_quaternion(down_pitch_rad, 0.0, 0.0)),
        vehicle_name=vehicle_name,
    )
    
    print("4. Spawning 30 pieces of floating waste on the water surface...")
    for i in range(30):
        obj_name = f"CoastWaste_{i}"
        x = 200.0 + random.uniform(-30.0, 30.0)
        y = 150.0 + random.uniform(-30.0, 30.0)
        z = 0.0  # Sea level
        pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0.0, 0.0, 0.0))
        
        assets = ["SimpleCube", "Cube", "SM_Cube"] if i % 2 == 0 else ["SimpleSphere", "Sphere", "SM_Sphere"]
        
        for asset in assets:
            try:
                client.simSpawnObject(obj_name, asset, pose, airsim.Vector3r(1.0, 1.0, 1.0), False)
                break
            except Exception:
                continue
                
        # Scale it down to be realistic
        try:
            client.simSetObjectScale(obj_name, airsim.Vector3r(3.0, 3.0, 3.0))
        except Exception:
            pass

    print("Setup complete! The Coastline is ready for GUI testing.")

if __name__ == "__main__":
    setup_coastline_environment()
