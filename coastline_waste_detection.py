import math
import random
import time

import airsim
import cv2
import numpy as np
import torch


# -----------------------------
# Scenario configuration
# -----------------------------
VEHICLE_NAME = "Drone1"
CAMERA_NAME = "0"
TARGET_X = 200.0
TARGET_Y = 150.0
TARGET_Z = -20.0  # NED frame: negative Z is up
SEA_LEVEL_Z = 0.0
WASTE_COUNT = 30
WASTE_RADIUS_XY = 30.0
DRONE_FORWARD_SPEED = 3.0


def spawn_object_with_fallback(client, object_name, preferred_assets, pose, physics_enabled=False):
    """
    Try multiple asset names so the script works across different AirSim map builds.
    Returns the asset name used, or None if all attempts fail.
    """
    for asset in preferred_assets:
        try:
            client.simSpawnObject(
                object_name,
                asset,
                pose,
                airsim.Vector3r(1.0, 1.0, 1.0),
                physics_enabled,
            )
            return asset
        except Exception:
            continue
    return None


def create_marine_waste_field(client, center_x, center_y):
    """
    Spawn small waste pieces (alternating cube/sphere style) around the target water area.
    """
    spawned = 0
    for i in range(WASTE_COUNT):
        name = f"MarineWaste_{i:02d}"

        x = center_x + random.uniform(-WASTE_RADIUS_XY, WASTE_RADIUS_XY)
        y = center_y + random.uniform(-WASTE_RADIUS_XY, WASTE_RADIUS_XY)
        z = SEA_LEVEL_Z
        pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0.0, 0.0, 0.0))

        # Request type as SimpleCube/SimpleSphere first, then map-safe fallbacks.
        asset_candidates = ["SimpleCube", "Cube", "SM_Cube"] if i % 2 == 0 else ["SimpleSphere", "Sphere", "SM_Sphere"]
        used_asset = spawn_object_with_fallback(client, name, asset_candidates, pose, physics_enabled=False)

        if used_asset is not None:
            try:
                client.simSetObjectScale(name, airsim.Vector3r(0.3, 0.3, 0.3))
                spawned += 1
            except Exception:
                pass

    print(f"Spawned waste objects: {spawned}/{WASTE_COUNT}")


def main():
    # 1) Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)
    print(f"Connected and API control enabled for {VEHICLE_NAME}.")

    # Make sure drone takes off so its physics engine wakes up properly before moving
    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()

    # 2) Teleport drone over water in Coastline
    water_pose = airsim.Pose(
        airsim.Vector3r(TARGET_X, TARGET_Y, TARGET_Z),
        airsim.to_quaternion(0.0, 0.0, 0.0),
    )
    client.simSetVehiclePose(water_pose, True, vehicle_name=VEHICLE_NAME)
    print(f"Teleported {VEHICLE_NAME} to X={TARGET_X}, Y={TARGET_Y}, Z={TARGET_Z}")

    # Point camera downward (nadir view) so water surface and waste are visible.
    down_pitch_rad = math.radians(-90.0)
    client.simSetCameraPose(
        CAMERA_NAME,
        airsim.Pose(airsim.Vector3r(0.0, 0.0, 0.0), airsim.to_quaternion(down_pitch_rad, 0.0, 0.0)),
        vehicle_name=VEHICLE_NAME,
    )

    # 3) Spawn fake marine waste under/around teleported location
    create_marine_waste_field(client, TARGET_X, TARGET_Y)

    # 4) Load YOLOv5s
    print("Loading YOLOv5s model...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.eval()
    print("YOLOv5s loaded.")

    # Start slow forward motion
    client.moveByVelocityAsync(
        DRONE_FORWARD_SPEED,
        0.0,
        0.0,
        300,
        vehicle_name=VEHICLE_NAME,
    )
    print(f"Drone moving forward at {DRONE_FORWARD_SPEED} m/s. Press 'q' to quit.")

    try:
        while True:
            response = client.simGetImage(CAMERA_NAME, airsim.ImageType.Scene, vehicle_name=VEHICLE_NAME)
            if response is None:
                time.sleep(0.05)
                continue

            # Decode compressed PNG image from AirSim.
            png = np.frombuffer(response, dtype=np.uint8)
            frame_bgr = cv2.imdecode(png, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                time.sleep(0.05)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # YOLO inference
            results = model(frame_rgb, size=640)
            detections = results.xyxy[0].cpu().numpy()

            # Draw boxes and labels
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_name = model.names[int(cls_id)]
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if len(detections) > 0:
                state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
                pos = state.kinematics_estimated.position
                print(f"🚨 Waste Detected at [{pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f}]")

            cv2.imshow("Marine Waste Detection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Safe shutdown
        client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
        client.landAsync(vehicle_name=VEHICLE_NAME).join()
        client.armDisarm(False, vehicle_name=VEHICLE_NAME)
        client.enableApiControl(False, vehicle_name=VEHICLE_NAME)
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
