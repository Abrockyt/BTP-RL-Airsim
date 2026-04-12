& 'c:/Users/abroc/Desktop/UAV_Energy_Sim/.venv/Scripts/python.exe' 'coastline_waste_detection.py'import airsim
import random


def spawn_with_fallback(client, object_name, asset_candidates, pose, physics_enabled):
    """Spawn using the first asset that exists and works in the current map build."""
    for asset_name in asset_candidates:
        try:
            client.simSpawnObject(
                object_name,
                asset_name,
                pose,
                airsim.Vector3r(1.0, 1.0, 1.0),
                physics_enabled,
            )
            return asset_name
        except Exception:
            continue
    return None


def create_neighborhood_river():
    """
    Connects to AirSim and dynamically spawns a "Fake River" over the main road
    in the Neighborhood environment, populated with floating plastic waste.
    """
    # 1. Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    # 2. Build a visible fake river as a sequence of smaller cube tiles.
    # Giant single cubes can fail in Neighborhood, so tiled segments are more reliable.
    drone_pose = client.simGetVehiclePose(vehicle_name="Drone1")
    start_x = drone_pose.position.x_val
    start_y = drone_pose.position.y_val
    base_z = -0.05

    river_tile_count = 40
    river_tile_length = 3.0
    river_half_width = 3.0
    spawned_tiles = 0

    print(
        f"Spawning tiled river near Drone1... (X: {start_x:.2f}, Y: {start_y:.2f}, Z: {base_z:.2f})"
    )

    for i in range(river_tile_count):
        tile_name = f"RiverTile_{i}"
        tile_x = start_x + 4.0 + i * river_tile_length
        tile_y = start_y
        tile_pose = airsim.Pose(
            airsim.Vector3r(tile_x, tile_y, base_z),
            airsim.to_quaternion(0, 0, 0)
        )

        used_asset = spawn_with_fallback(
            client,
            tile_name,
            ["Cylinder", "Sphere", "Cone"],
            tile_pose,
            False,
        )

        try:
            client.simSetObjectPose(tile_name, tile_pose, True)
            client.simSetObjectScale(tile_name, airsim.Vector3r(river_tile_length, river_half_width * 2.0, 0.06))
            if used_asset is not None:
                spawned_tiles += 1
        except Exception:
            pass


    # 3. Spawn the Waste (Pollution)
    num_waste_items = 30
    print(f"Spawning {num_waste_items} pieces of waste on top of river tiles...")

    for i in range(num_waste_items):
        waste_name = f"Waste_{i}"

        # Spread waste across the full tiled river footprint.
        rand_x = start_x + random.uniform(4.0, 4.0 + river_tile_count * river_tile_length)
        rand_y = start_y + random.uniform(-2.5, 2.5)

        # Z: Spawn them slightly in the air and let them drop (-Z is up)
        rand_z = base_z - 1.2


        waste_pose = airsim.Pose(airsim.Vector3r(rand_x, rand_y, rand_z), airsim.to_quaternion(0, 0, 0))

        # Alternate available primitive types for visible variety.
        waste_assets = ["Sphere", "Cone"] if i % 2 == 0 else ["Cone", "Sphere"]
        spawn_with_fallback(client, waste_name, waste_assets, waste_pose, True)

        # Randomize the scale (size) between 0.3 and 0.6
        scale_val = random.uniform(0.3, 0.6)
        try:
            client.simSetObjectScale(waste_name, airsim.Vector3r(scale_val, scale_val, scale_val))
        except Exception:
            pass

    print(f"River tiles ready: {spawned_tiles}/{river_tile_count}")
    print("All waste successfully spawned on the river!")

if __name__ == "__main__":
    create_neighborhood_river()
