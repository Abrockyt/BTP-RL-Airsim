"""
Smart Drone Control with Interactive GUI
Features: Adaptive speed (2-5 m/s), Smart collision recovery, Interactive map, Restart to home
"""

import sys
sys.path.append('.')
from smart_drone import (
    DronePilot, load_model, preprocess_image, 
    smart_speed_control, smart_collision_recovery
)

import airsim
import torch
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
import threading


class SmartDroneGUI:
    """Interactive map GUI for smart drone control with restart functionality"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚁 Smart Drone - Adaptive Speed + Smart Collision Recovery")
        self.window.geometry("900x800")
        
        # Make window visible
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.focus_force()
        
        # Map parameters
        self.map_size = 600
        self.map_range = 100
        self.scale = self.map_size / (2 * self.map_range)
        
        # State
        self.client = None
        self.model = None
        self.device = None
        self.drone_position = (0, 0)
        self.home_position = None
        self.goal_position = None  # User-selected goal
        self.running = False
        self.flight_active = False
        self.navigating_to_goal = False
        self.path_points = []
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = tk.Label(self.window, text="🚁 Smart Drone Control", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        subtitle = tk.Label(self.window, text="Adaptive Speed (2-5 m/s) + Smart Collision Recovery", 
                           font=("Arial", 10), fg="gray")
        subtitle.pack()
        
        # Info frame
        info_frame = tk.Frame(self.window)
        info_frame.pack(pady=10)
        
        self.status_label = tk.Label(info_frame, text="Status: Not connected", 
                                     font=("Arial", 11, "bold"))
        self.status_label.grid(row=0, column=0, padx=15)
        
        self.position_label = tk.Label(info_frame, text="Position: (0.0, 0.0)", 
                                       font=("Arial", 10))
        self.position_label.grid(row=0, column=1, padx=15)
        
        self.speed_label = tk.Label(info_frame, text="Speed: -- m/s", 
                                   font=("Arial", 10), fg="blue")
        self.speed_label.grid(row=0, column=2, padx=15)
        
        # Canvas for map
        self.canvas = tk.Canvas(self.window, width=self.map_size, height=self.map_size,
                               bg="white", highlightthickness=2, highlightbackground="black")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_map_click)  # Click to set goal
        
        # Control buttons
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=15)
        
        self.start_btn = tk.Button(btn_frame, text="▶ Start Flight", 
                                   command=self.start_flight, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                   width=15, height=2)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.restart_btn = tk.Button(btn_frame, text="🏠 Restart (Home)", 
                                     command=self.restart_to_home, 
                                     bg="#FF9800", fg="white", font=("Arial", 12, "bold"),
                                     width=15, height=2, state="disabled")
        self.restart_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop Flight", 
                                 command=self.stop_flight, 
                                 bg="#F44336", fg="white", font=("Arial", 12, "bold"),
                                 width=15, height=2, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Features info
        features_frame = tk.Frame(self.window, bg="#f0f0f0", relief="groove", borderwidth=2)
        features_frame.pack(pady=10, padx=20, fill="x")
        
        features_title = tk.Label(features_frame, text="✨ Smart Features:", 
                                 font=("Arial", 10, "bold"), bg="#f0f0f0")
        features_title.pack(anchor="w", padx=10, pady=5)
        
        features = [
            "🚀 Adaptive Speed: Fast (5 m/s) on straight paths, Slow (2 m/s) on turns",
            "🛡️  Smart Recovery: Buildings→UP | Trees→LEFT/RIGHT | Bushes→never down",
            "🔮 Collision Prediction: Uses depth sensors to avoid obstacles BEFORE collision",
            "🎯 Click Map: Click anywhere on map to set goal and navigate autonomously",
            "🏠 Restart Button: Returns to origin with same smart collision avoidance",
            "🗺️  Real-time Map: Blue=drone, Green=home, Red=goal, Light blue=path"
        ]
        
        for feature in features:
            tk.Label(features_frame, text=feature, font=("Arial", 9), 
                    bg="#f0f0f0", anchor="w").pack(anchor="w", padx=20, pady=2)
        
        # Draw grid
        self.draw_grid()
    
    def draw_grid(self):
        """Draw grid on map"""
        for i in range(-100, 101, 20):
            x = self.world_to_canvas(i, 0)[0]
            y = self.world_to_canvas(0, i)[1]
            self.canvas.create_line(x, 0, x, self.map_size, fill="lightgray", dash=(2, 2))
            self.canvas.create_line(0, y, self.map_size, y, fill="lightgray", dash=(2, 2))
        
        center = self.map_size // 2
        self.canvas.create_line(center, 0, center, self.map_size, fill="gray", width=2)
        self.canvas.create_line(0, center, self.map_size, center, fill="gray", width=2)
        
        self.canvas.create_text(center + 5, 10, text="N", font=("Arial", 10, "bold"))
        self.canvas.create_text(self.map_size - 10, center - 5, text="E", font=("Arial", 10, "bold"))
    
    def world_to_canvas(self, x, y):
        canvas_x = int(self.map_size / 2 + x * self.scale)
        canvas_y = int(self.map_size / 2 - y * self.scale)
        return canvas_x, canvas_y
    
    def canvas_to_world(self, canvas_x, canvas_y):
        """Convert canvas coordinates to world coordinates"""
        x = (canvas_x - self.map_size / 2) / self.scale
        y = -(canvas_y - self.map_size / 2) / self.scale
        return x, y
    
    def on_map_click(self, event):
        """Handle map click to set goal"""
        if not self.running:
            messagebox.showwarning("Not Ready", "Start flight first!")
            return
        
        # Convert click to world coordinates
        goal_x, goal_y = self.canvas_to_world(event.x, event.y)
        self.goal_position = (goal_x, goal_y)
        
        print(f"\n🎯 Goal set: ({goal_x:.1f}, {goal_y:.1f})")
        
        # Clear old goal marker
        self.canvas.delete("goal")
        
        # Draw new goal marker (red circle)
        canvas_x, canvas_y = self.world_to_canvas(goal_x, goal_y)
        self.canvas.create_oval(
            canvas_x - 10, canvas_y - 10,
            canvas_x + 10, canvas_y + 10,
            fill='red', outline='darkred', width=3, tags="goal"
        )
        
        # Stop model-based flight and start goal navigation
        self.flight_active = False
        time.sleep(0.3)
        self.navigating_to_goal = True
        threading.Thread(target=self._navigate_to_goal, daemon=True).start()
    
    def predict_collision(self):
        """
        Predict collision using depth sensors BEFORE it happens
        Returns: (will_collide, avoidance_direction)
        """
        try:
            # Get depth image from front camera
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
            ])
            
            if responses and len(responses[0].image_data_float) > 0:
                depth_data = np.array(responses[0].image_data_float)
                depth_data = depth_data.reshape(responses[0].height, responses[0].width)
                
                h, w = depth_data.shape
                
                # Check zones
                center_depth = np.mean(depth_data[h//3:2*h//3, w//3:2*w//3])
                left_depth = np.mean(depth_data[h//3:2*h//3, :w//3])
                right_depth = np.mean(depth_data[h//3:2*h//3, 2*w//3:])
                upper_depth = np.mean(depth_data[:h//3, :])
                lower_depth = np.mean(depth_data[2*h//3:, :])
                
                # Collision thresholds (in meters)
                DANGER_DISTANCE = 3.0  # Predict collision if obstacle < 3m ahead
                WARNING_DISTANCE = 5.0  # Start adjusting if < 5m
                
                # Check if obstacle is close
                if center_depth < DANGER_DISTANCE:
                    # Determine best avoidance direction
                    if upper_depth > center_depth * 1.5 and upper_depth > 8:
                        return (True, "UP")  # Building/wall - go over
                    elif lower_depth < 2.5:
                        # Ground/bushes - never go down
                        if left_depth > right_depth and left_depth > center_depth:
                            return (True, "LEFT")
                        elif right_depth > left_depth and right_depth > center_depth:
                            return (True, "RIGHT")
                        else:
                            return (True, "UP")
                    elif left_depth > right_depth * 1.2 and left_depth > center_depth:
                        return (True, "LEFT")  # Tree on right
                    elif right_depth > left_depth * 1.2 and right_depth > center_depth:
                        return (True, "RIGHT")  # Tree on left
                    else:
                        return (True, "UP")  # Default - climb
                
                # Gentle adjustment for warning zone
                elif center_depth < WARNING_DISTANCE:
                    if left_depth > right_depth * 1.1:
                        return (False, "LEFT")  # Slight left adjustment
                    elif right_depth > left_depth * 1.1:
                        return (False, "RIGHT")  # Slight right adjustment
            
            return (False, None)  # No collision predicted
            
        except BufferError:
            # Buffer error - skip this check
            return (False, None)
        except Exception as e:
            return (False, None)
    
    def predictive_avoidance(self, base_vx, base_vy, base_vz):
        """
        Adjust velocity based on collision prediction
        Returns: (adjusted_vx, adjusted_vy, adjusted_vz)
        """
        will_collide, direction = self.predict_collision()
        
        if will_collide:
            print(f"🔮 COLLISION PREDICTED! Avoiding: {direction}")
            
            if direction == "UP":
                # Climb to avoid
                return (base_vx * 0.3, base_vy, -2.0)  # Slow forward, climb fast
            elif direction == "LEFT":
                # Move left
                return (base_vx * 0.5, -2.0, base_vz - 0.5)  # Slow forward, left, slight up
            elif direction == "RIGHT":
                # Move right
                return (base_vx * 0.5, 2.0, base_vz - 0.5)  # Slow forward, right, slight up
        
        elif direction in ["LEFT", "RIGHT"]:
            # Gentle adjustment for warning zone
            lateral_adjust = -1.0 if direction == "LEFT" else 1.0
            return (base_vx * 0.8, lateral_adjust, base_vz)
        
        # No adjustment needed
        return (base_vx, base_vy, base_vz)
    
    def start_flight(self):
        self.status_label.config(text="Status: Initializing...", fg="orange")
        self.window.update()
        threading.Thread(target=self._initialize_and_fly, daemon=True).start()
    
    def _initialize_and_fly(self):
        try:
            # Setup
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print("📂 Loading model...")
            self.model = load_model("smart_airsim_model .pth", self.device)
            
            print("🔌 Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            print("🛫 Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(2)
            
            self.client.moveToZAsync(-3.0, 2.0).join()
            time.sleep(1)
            
            # Store home
            home_state = self.client.getMultirotorState()
            self.home_position = (home_state.kinematics_estimated.position.x_val,
                                 home_state.kinematics_estimated.position.y_val)
            print(f"🏠 Home: ({self.home_position[0]:.1f}, {self.home_position[1]:.1f})")
            
            # Draw home marker
            hc = self.world_to_canvas(self.home_position[0], self.home_position[1])
            self.canvas.create_oval(hc[0]-8, hc[1]-8, hc[0]+8, hc[1]+8,
                                   fill='green', outline='darkgreen', width=3, tags="home")
            
            self.running = True
            self.flight_active = True
            self.status_label.config(text="Status: Flying ✓", fg="green")
            self.start_btn.config(state="disabled")
            self.restart_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            
            self.update_position_loop()
            self._flight_control_loop()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)[:30]}", fg="red")
            print(f"Init error: {e}")
            import traceback
            traceback.print_exc()
    
    def _flight_control_loop(self):
        COLLISION_COOLDOWN = 2.0
        STARTING_ALTITUDE = -3.0
        YAW_RATE_SCALE = 60.0
        ALTITUDE_KP = 0.8
        UPDATE_RATE = 0.1  # Slower updates to prevent buffer overflow
        
        last_collision_time = 0
        step = 0
        
        print("\n🚀 SMART FLIGHT STARTED WITH COLLISION PREDICTION")
        
        try:
            with torch.no_grad():
                while self.flight_active and self.running:
                    current_time = time.time()
                    
                    # Actual collision check (backup safety)
                    try:
                        collision_info = self.client.simGetCollisionInfo()
                        if collision_info.has_collided and (current_time - last_collision_time) > COLLISION_COOLDOWN:
                            print("⚠️  Actual collision occurred (prediction missed)")
                            last_collision_time = current_time
                            smart_collision_recovery(self.client)
                            continue
                    except BufferError:
                        time.sleep(0.15)
                        continue
                    except:
                        pass
                    
                    # Get state with error handling
                    try:
                        state = self.client.getMultirotorState()
                        pos = state.kinematics_estimated.position
                    except BufferError:
                        time.sleep(0.15)
                        continue
                    except:
                        time.sleep(0.1)
                        continue
                    
                    # Get image with error handling
                    try:
                        responses = self.client.simGetImages([
                            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                        ])
                        if not responses or len(responses[0].image_data_uint8) == 0:
                            time.sleep(0.1)
                            continue
                    except BufferError:
                        time.sleep(0.15)
                        continue
                    except:
                        time.sleep(0.1)
                        continue
                    
                    # Predict
                    image_tensor = preprocess_image(responses[0])
                    image_tensor = image_tensor.to(self.device)
                    steering_angle = self.model(image_tensor).item()
                    
                    # Smart speed
                    forward_speed = smart_speed_control(steering_angle)
                    
                    # Update GUI
                    speed_emoji = "🐢 SLOW" if forward_speed < 3.5 else "🚀 FAST"
                    self.speed_label.config(text=f"Speed: {forward_speed:.1f} m/s {speed_emoji}")
                    
                    # Base control
                    yaw_rate = np.clip(steering_angle * YAW_RATE_SCALE, -60.0, 60.0)
                    altitude_error = STARTING_ALTITUDE - pos.z_val
                    base_vz = float(np.clip(altitude_error * ALTITUDE_KP, -1.5, 1.5))
                    
                    # PREDICTIVE COLLISION AVOIDANCE - adjust velocities BEFORE collision
                    adjusted_vx, adjusted_vy, adjusted_vz = self.predictive_avoidance(
                        forward_speed, 0, base_vz
                    )
                    
                    # Send control command with predicted adjustments
                    try:
                        if adjusted_vy != 0 or adjusted_vx != forward_speed:
                            # Using predictive avoidance - use body frame velocities
                            self.client.moveByVelocityBodyFrameAsync(
                                adjusted_vx, adjusted_vy, adjusted_vz, duration=UPDATE_RATE + 0.05,
                                yaw_mode=airsim.YawMode(True, yaw_rate)
                            )
                        else:
                            # Normal flight - use model steering
                            self.client.moveByVelocityBodyFrameAsync(
                                forward_speed, 0, base_vz, duration=UPDATE_RATE + 0.05,
                                yaw_mode=airsim.YawMode(True, yaw_rate)
                            )
                    except BufferError:
                        time.sleep(0.15)
                        continue
                    except:
                        pass
                    
                    if step % 20 == 0:
                        label = "SLOW" if forward_speed < 3.5 else "FAST"
                        print(f"Step {step:4d} | Steer: {steering_angle:+.3f} → {forward_speed:.1f} m/s ({label})")
                    
                    step += 1
                    time.sleep(UPDATE_RATE)
                    
        except Exception as e:
            print(f"Flight error: {e}")
        finally:
            self.flight_active = False
    
    def update_position_loop(self):
        if not self.running:
            return
        
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            self.drone_position = (pos.x_val, pos.y_val)
            
            self.position_label.config(text=f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f})")
            
            # Draw drone
            self.canvas.delete("drone")
            cx, cy = self.world_to_canvas(pos.x_val, pos.y_val)
            self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6,
                                   fill="blue", outline="darkblue", width=2, tags="drone")
            
            # Path
            if len(self.path_points) == 0 or \
               np.sqrt((pos.x_val - self.path_points[-1][0])**2 + 
                      (pos.y_val - self.path_points[-1][1])**2) > 0.5:
                self.path_points.append((pos.x_val, pos.y_val))
                
                if len(self.path_points) > 1:
                    self.canvas.delete("path")
                    for i in range(len(self.path_points) - 1):
                        x1, y1 = self.world_to_canvas(*self.path_points[i])
                        x2, y2 = self.world_to_canvas(*self.path_points[i+1])
                        self.canvas.create_line(x1, y1, x2, y2, 
                                               fill="lightblue", width=2, tags="path")
        except BufferError:
            # Skip this update if buffer error
            pass
        except:
            pass
        
        self.window.after(150, self.update_position_loop)  # Slower updates (150ms)
    
    def restart_to_home(self):
        if not self.running or self.home_position is None:
            messagebox.showwarning("Not Ready", "Start flight first!")
            return
        
        print(f"\n🏠 Returning to home: {self.home_position}")
        self.status_label.config(text="Status: Returning home...", fg="orange")
        
        self.flight_active = False
        time.sleep(0.5)
        threading.Thread(target=self._navigate_to_home, daemon=True).start()
    
    def _navigate_to_home(self):
        COLLISION_COOLDOWN = 2.0
        FORWARD_VELOCITY = 3.0
        TARGET_ALTITUDE = -3.0
        
        last_collision_time = 0
        goal_threshold = 2.0
        
        try:
            for step in range(500):
                if not self.running:
                    break
                
                current_time = time.time()
                
                try:
                    state = self.client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    current_x, current_y, current_z = pos.x_val, pos.y_val, pos.z_val
                except:
                    time.sleep(0.1)
                    continue
                
                dx = self.home_position[0] - current_x
                dy = self.home_position[1] - current_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < goal_threshold:
                    print(f"✅ HOME REACHED! Distance: {distance:.2f}m")
                    self.client.moveByVelocityAsync(0, 0, 0, 1)
                    self.status_label.config(text="Status: Home ✓", fg="green")
                    break
                
                # PREDICTIVE collision avoidance during navigation
                will_collide, avoid_direction = self.predict_collision()
                
                if will_collide:
                    print(f"🔮 Predicted collision while returning home - avoiding: {avoid_direction}")
                    
                    if avoid_direction == "UP":
                        self.client.moveByVelocityAsync(0, 0, -2.5, 1.5)
                        time.sleep(1.6)
                    elif avoid_direction == "LEFT":
                        self.client.moveByVelocityAsync(-1.5, -2.5, -0.5, 1.2)
                        time.sleep(1.3)
                    elif avoid_direction == "RIGHT":
                        self.client.moveByVelocityAsync(-1.5, 2.5, -0.5, 1.2)
                        time.sleep(1.3)
                    else:
                        self.client.moveByVelocityAsync(-2.5, 0, -1.5, 1.2)
                        time.sleep(1.3)
                    continue
                
                # Check actual collision (backup)
                try:
                    collision_info = self.client.simGetCollisionInfo()
                    if collision_info.has_collided and (current_time - last_collision_time) > COLLISION_COOLDOWN:
                        last_collision_time = current_time
                        smart_collision_recovery(self.client)
                        continue
                except:
                    pass
                
                goal_angle = np.arctan2(dy, dx)
                orientation = state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(orientation)[2]
                
                yaw_error = goal_angle - current_yaw
                while yaw_error > np.pi:
                    yaw_error -= 2 * np.pi
                while yaw_error < -np.pi:
                    yaw_error += 2 * np.pi
                
                yaw_rate = np.clip(yaw_error * 0.8, -1.0, 1.0)
                alignment = 1.0 - min(abs(yaw_error) / np.pi, 1.0)
                vx = FORWARD_VELOCITY * alignment
                
                altitude_error = TARGET_ALTITUDE - current_z
                vz = np.clip(altitude_error * 0.5, -1.0, 1.0)
                
                try:
                    self.client.moveByVelocityAsync(
                        vx * np.cos(current_yaw),
                        vx * np.sin(current_yaw),
                        vz,
                        duration=0.25,
                        yaw_mode=airsim.YawMode(True, yaw_rate * 50)
                    )
                except BufferError:
                    time.sleep(0.2)
                    continue
                except:
                    pass
                
                if step % 10 == 0:
                    print(f"  Returning... Dist: {distance:.1f}m")
                
                time.sleep(0.25)  # Slower updates to prevent buffer overflow
                
        except Exception as e:
            print(f"Navigation error: {e}")
    
    def _navigate_to_goal(self):
        """Navigate to user-selected goal with predictive collision avoidance"""
        if self.goal_position is None:
            return
        
        COLLISION_COOLDOWN = 2.0
        FORWARD_VELOCITY = 3.0
        TARGET_ALTITUDE = -3.0
        
        last_collision_time = 0
        goal_threshold = 2.0
        
        goal_x, goal_y = self.goal_position
        print(f"\n🎯 Navigating to goal: ({goal_x:.1f}, {goal_y:.1f})")
        self.status_label.config(text=f"Status: → Goal ({goal_x:.0f}, {goal_y:.0f})", fg="blue")
        
        try:
            for step in range(1000):
                if not self.running or not self.navigating_to_goal:
                    break
                
                current_time = time.time()
                
                try:
                    state = self.client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    current_x, current_y, current_z = pos.x_val, pos.y_val, pos.z_val
                except:
                    time.sleep(0.1)
                    continue
                
                dx = goal_x - current_x
                dy = goal_y - current_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < goal_threshold:
                    print(f"✅ GOAL REACHED! Distance: {distance:.2f}m")
                    self.client.moveByVelocityAsync(0, 0, 0, 1)
                    self.status_label.config(text="Status: Goal reached ✓", fg="green")
                    self.navigating_to_goal = False
                    break
                
                # PREDICTIVE collision avoidance
                will_collide, avoid_direction = self.predict_collision()
                
                if will_collide:
                    print(f"🔮 Predicted collision - avoiding: {avoid_direction}")
                    
                    if avoid_direction == "UP":
                        self.client.moveByVelocityAsync(0, 0, -2.5, 1.5)
                        time.sleep(1.6)
                    elif avoid_direction == "LEFT":
                        self.client.moveByVelocityAsync(-1.5, -2.5, -0.5, 1.2)
                        time.sleep(1.3)
                    elif avoid_direction == "RIGHT":
                        self.client.moveByVelocityAsync(-1.5, 2.5, -0.5, 1.2)
                        time.sleep(1.3)
                    else:
                        self.client.moveByVelocityAsync(-2.5, 0, -1.5, 1.2)
                        time.sleep(1.3)
                    continue
                
                # Check actual collision (backup safety)
                try:
                    collision_info = self.client.simGetCollisionInfo()
                    if collision_info.has_collided and (current_time - last_collision_time) > COLLISION_COOLDOWN:
                        print("⚠️  Actual collision (prediction missed)")
                        last_collision_time = current_time
                        smart_collision_recovery(self.client)
                        continue
                except:
                    pass
                
                # Calculate heading to goal
                goal_angle = np.arctan2(dy, dx)
                orientation = state.kinematics_estimated.orientation
                current_yaw = airsim.to_eularian_angles(orientation)[2]
                
                yaw_error = goal_angle - current_yaw
                while yaw_error > np.pi:
                    yaw_error -= 2 * np.pi
                while yaw_error < -np.pi:
                    yaw_error += 2 * np.pi
                
                # Control
                yaw_rate = np.clip(yaw_error * 0.8, -1.0, 1.0)
                alignment = 1.0 - min(abs(yaw_error) / np.pi, 1.0)
                vx = FORWARD_VELOCITY * alignment
                
                altitude_error = TARGET_ALTITUDE - current_z
                vz = np.clip(altitude_error * 0.5, -1.0, 1.0)
                
                try:
                    self.client.moveByVelocityAsync(
                        vx * np.cos(current_yaw),
                        vx * np.sin(current_yaw),
                        vz,
                        duration=0.25,
                        yaw_mode=airsim.YawMode(True, yaw_rate * 50)
                    )
                except BufferError:
                    time.sleep(0.2)
                    continue
                except:
                    pass
                
                if step % 10 == 0:
                    print(f"  → Goal... Dist: {distance:.1f}m | Yaw err: {np.degrees(yaw_error):+.0f}°")
                
                time.sleep(0.25)  # Slower updates to prevent buffer overflow
            
            if step >= 999:
                print("⏱️  Navigation timeout")
                self.status_label.config(text="Status: Timeout", fg="orange")
                
        except Exception as e:
            print(f"Goal navigation error: {e}")
        finally:
            self.navigating_to_goal = False
    
    def stop_flight(self):
        self.flight_active = False
        self.running = False
        
        if self.client:
            try:
                print("\n🛬 Landing...")
                self.client.moveByVelocityAsync(0, 0, 0, 1).join()
                self.client.landAsync().join()
                time.sleep(2)
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
                print("✓ Landed")
            except:
                pass
        
        self.status_label.config(text="Status: Stopped", fg="red")
        self.stop_btn.config(state="disabled")
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = SmartDroneGUI()
    gui.run()
