import time
import math
import cv2
import numpy as np
import airsim
import threading

class ActiveVisionWasteDetector:
    """
    INNOVATIVE NOVEL APPROACH: Entropy-Driven Active Vision + Depth-Gated Fusion
    
    This algorithm calculates the Shannon Entropy of the YOLOv8 confidence distribution.
    If the drone spots something but the AI is uncertain (high entropy/low confidence), 
    the RL/Active Vision module dynamically drops the drone's altitude to get a closer 
    look and uses Depth-Planar geometry fusion to confirm the object.
    
    This overcomes traditional fixed-altitude surveying where small waste is missed or misclassified.
    """
    
    def __init__(self):
        self.vehicle_name = "Drone1"
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.is_running = True
        
        # Load Model
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            print("YOLOv8 Loaded for Active Vision.")
        except Exception as e:
            print(f"Failed to load YOLOv8: {e}")
            self.model = None

        self.waste_database = []
        self.base_altitude = -12.0 # High altitude survey
        self.inspection_altitude = -4.0 # Low altitude confirmation
        
    def calculate_entropy(self, confidences):
        """ Calculate Shannon entropy of detection confidences to measure uncertainty """
        if len(confidences) == 0:
            return 0.0
        # Normalize confidences to sum to 1 (probability distribution)
        probs = np.array(confidences) / np.sum(confidences)
        entropy = -np.sum(probs * np.log2(probs + 1e-6))
        return entropy

    def active_vision_loop(self):
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        
        print(f"Ascending to Survey Altitude: {-self.base_altitude}m")
        self.client.moveToZAsync(self.base_altitude, 3.0, vehicle_name=self.vehicle_name).join()
        
        print("Starting Entropy-Driven Patrol...")
        
        # Move forward slowly
        self.client.moveByVelocityZAsync(2.0, 0, self.base_altitude, 3000, 
                                         drivetrain=airsim.DrivetrainType.ForwardOnly, 
                                         yaw_mode=airsim.YawMode(False, 0),
                                         vehicle_name=self.vehicle_name)
        
        while self.is_running:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            current_x = state.kinematics_estimated.position.x_val
            current_y = state.kinematics_estimated.position.y_val
            current_z = state.kinematics_estimated.position.z_val
            
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
            ], vehicle_name=self.vehicle_name)
            
            if len(responses) == 2:
                # 1. Process RGB
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_bgr = img_1d.reshape(responses[0].height, responses[0].width, 3).copy()
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # 2. YOLOv8 Inference
                confidences = []
                boxes_to_draw = []
                if self.model:
                    results = self.model(img_rgb, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            confidences.append(conf)
                            boxes_to_draw.append(box.xyxy[0].cpu().numpy().astype(int))
                            
                # 3. Entropy & Uncertainty Evaluation
                entropy = self.calculate_entropy(confidences)
                max_conf = max(confidences) if confidences else 0.0
                
                # Draw boxes
                for (x1, y1, x2, y2), conf in zip(boxes_to_draw, confidences):
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(img_bgr, f"{conf:.2f}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                cv2.putText(img_bgr, f"Entropy: {entropy:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # 4. ACTIVE VISION DECISION MAKER
                # If we see something but confidence is moderate (uncertainty is high) -> Inspect!
                if 0.3 < max_conf < 0.75 or (entropy > 0.5 and len(confidences) > 1):
                    print(f"High Uncertainty Detected! (Entropy: {entropy:.2f}, Max Conf: {max_conf:.2f}) -> Initiating Active Inspection.")
                    
                    # Pause forward movement
                    self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.vehicle_name).join()
                    
                    # Drop altitude for high-res geometric confirmation
                    self.client.moveToZAsync(self.inspection_altitude, 2.0, vehicle_name=self.vehicle_name).join()
                    time.sleep(1.0) # Stabilize camera
                    
                    # Gather Depth Data to confirm object geometry
                    depth_resp = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)], vehicle_name=self.vehicle_name)[0]
                    depth_array = np.array(depth_resp.image_data_float, dtype=np.float32).reshape(depth_resp.height, depth_resp.width)
                    
                    water_depth = np.median(depth_array)
                    depth_mask = np.where((depth_array < (water_depth - 0.15)) & (depth_array > 0.5), 255, 0).astype(np.uint8)
                    contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    confirmed_objects = 0
                    for cnt in contours:
                        if 50 < cv2.contourArea(cnt) < 50000:
                            confirmed_objects += 1
                            
                    if confirmed_objects > 0:
                        print(f"ACTIVE VISION SUCCESS: Confirmed {confirmed_objects} waste items via Depth geometry at close range!")
                        self.waste_database.append((current_x, current_y, confirmed_objects))
                    else:
                        print("False Alarm: Depth sensor rejected the detection.")
                        
                    # Return to patrol altitude and resume
                    print("Returning to survey altitude...")
                    self.client.moveToZAsync(self.base_altitude, 2.0, vehicle_name=self.vehicle_name).join()
                    self.client.moveByVelocityZAsync(2.0, 0, self.base_altitude, 3000, 
                                                     drivetrain=airsim.DrivetrainType.ForwardOnly, 
                                                     yaw_mode=airsim.YawMode(False, 0),
                                                     vehicle_name=self.vehicle_name)
                    
                cv2.imshow("Active Vision UAV Feed", img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break
                    
            time.sleep(0.1)

        # Shutdown
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        cv2.destroyAllWindows()
        print(f"Mission Complete. Discovered {sum(w[2] for w in self.waste_database)} verified waste items.")

if __name__ == "__main__":
    detector = ActiveVisionWasteDetector()
    try:
        detector.active_vision_loop()
    except KeyboardInterrupt:
        detector.is_running = False
