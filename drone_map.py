"""
Interactive Map for Drone Navigation with Model-Based Control
Click on the map to set a goal, drone navigates using trained steering model
"""

import airsim
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import ttk
import threading


def load_model(model_path, device):
    """Load the model - checkpoint was saved as bare Sequential"""
    model = nn.Sequential(
        nn.Conv2d(3, 24, 5, 2), nn.ReLU(),
        nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
        nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
        nn.Conv2d(48, 64, 3), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3840, 100), nn.ReLU(),
        nn.Linear(100, 50), nn.ReLU(),
        nn.Linear(50, 1)
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_response):
    """Preprocess AirSim image for model input"""
    img1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(image_response.height, image_response.width, 3)
    img_resized = cv2.resize(img_rgb, (200, 66))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
    return img_tensor.unsqueeze(0)


class DroneMapGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚁 Drone Navigation Map - Click to Set Goal")
        self.window.geometry("900x750")
        
        # Make window appear on top and focused
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.focus_force()
        
        # Map parameters
        self.map_size = 600
        self.map_range = 100  # ±100m
        self.scale = self.map_size / (2 * self.map_range)
        
        # AirSim & Model
        self.client = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Navigation state
        self.goal_position = None
        self.drone_position = (0, 0)
        self.navigating = False
        self.path_points = []
        self.home_position = None  # Store takeoff position
        self.running = False
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.window, text="🚁 Drone Navigation Map", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Info frame
        info_frame = tk.Frame(self.window)
        info_frame.pack(pady=5)
        
        self.status_label = tk.Label(info_frame, text="Status: Not connected", 
                                     font=("Arial", 10))
        self.status_label.grid(row=0, column=0, padx=10)
        
        self.position_label = tk.Label(info_frame, text="Position: (0.0, 0.0)", 
                                       font=("Arial", 10))
        self.position_label.grid(row=0, column=1, padx=10)
        
        # Canvas for map
        self.canvas = tk.Canvas(self.window, width=self.map_size, height=self.map_size,
                               bg="white", highlightthickness=2, highlightbackground="black")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_map_click)
        
        # Control buttons
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        self.connect_btn = tk.Button(btn_frame, text="Connect & Start", 
                                     command=self.connect_airsim, 
                                     bg="green", fg="white", font=("Arial", 12, "bold"),
                                     width=15, height=2)
        self.connect_btn.grid(row=0, column=0, padx=5)
        
        self.restart_btn = tk.Button(btn_frame, text="Restart (Home)", 
                                     command=self.restart_drone, 
                                     bg="orange", fg="white", font=("Arial", 12, "bold"),
                                     width=15, height=2, state="disabled")
        self.restart_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop Navigation", 
                                 command=self.stop_navigation, 
                                 bg="red", fg="white", font=("Arial", 12, "bold"),
                                 width=15, height=2, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Instructions
        instructions = tk.Label(self.window, 
                               text="Click anywhere on the map to set a goal. Drone will navigate autonomously.",
                               font=("Arial", 9), fg="gray")
        instructions.pack(pady=5)
        
        # Draw grid
        self.draw_grid()
        
    def draw_grid(self):
        """Draw grid lines on map"""
        # Grid lines every 20m
        for i in range(-100, 101, 20):
            x = self.world_to_canvas(i, 0)[0]
            y = self.world_to_canvas(0, i)[1]
            # Vertical lines
            self.canvas.create_line(x, 0, x, self.map_size, fill="lightgray", dash=(2, 2))
            # Horizontal lines
            self.canvas.create_line(0, y, self.map_size, y, fill="lightgray", dash=(2, 2))
        
        # Center axes
        center = self.map_size // 2
        self.canvas.create_line(center, 0, center, self.map_size, fill="gray", width=2)
        self.canvas.create_line(0, center, self.map_size, center, fill="gray", width=2)
        
        # Labels
        self.canvas.create_text(center + 5, 10, text="N", font=("Arial", 10, "bold"))
        self.canvas.create_text(self.map_size - 10, center - 5, text="E", font=("Arial", 10, "bold"))
        
    def world_to_canvas(self, x, y):
        """Convert world coordinates to canvas coordinates"""
        canvas_x = int(self.map_size / 2 + x * self.scale)
        canvas_y = int(self.map_size / 2 - y * self.scale)  # Flip Y
        return canvas_x, canvas_y
    
    def canvas_to_world(self, canvas_x, canvas_y):
        """Convert canvas coordinates to world coordinates"""
        x = (canvas_x - self.map_size / 2) / self.scale
        y = -(canvas_y - self.map_size / 2) / self.scale  # Flip Y
        return x, y
    
    def connect_airsim(self):
        """Connect to AirSim and load model"""
        try:
            self.status_label.config(text="Status: Connecting...")
            self.window.update()
            
            # Load model
            print("Loading model...")
            self.model = load_model("my_airsim_model.pth", self.device)
            
            # Connect to AirSim
            print("Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            # Takeoff
            print("Taking off...")
            self.client.takeoffAsync().join()
            time.sleep(2)
            
            # Store home position
            home_state = self.client.getMultirotorState()
            self.home_position = (home_state.kinematics_estimated.position.x_val,
                                 home_state.kinematics_estimated.position.y_val)
            print(f"Home position set: ({self.home_position[0]:.1f}, {self.home_position[1]:.1f})")
            
            self.running = True
            self.status_label.config(text="Status: Connected ✓", fg="green")
            self.connect_btn.config(state="disabled")
            self.restart_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            
            # Start position update loop
            self.update_position_loop()
            
            print("✓ Ready! Click on the map to set a goal.")
            
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {e}", fg="red")
            print(f"Connection error: {e}")
    
    def on_map_click(self, event):
        """Handle map click to set goal"""
        if self.client is None:
            print("Please connect to AirSim first!")
            return
        
        # Convert click to world coordinates
        goal_x, goal_y = self.canvas_to_world(event.x, event.y)
        self.goal_position = (goal_x, goal_y)
        
        print(f"\n🎯 Goal set: ({goal_x:.1f}, {goal_y:.1f})")
        
        # Clear previous goal marker
        self.canvas.delete("goal")
        
        # Draw new goal
        canvas_x, canvas_y = self.world_to_canvas(goal_x, goal_y)
        self.canvas.create_oval(canvas_x-8, canvas_y-8, canvas_x+8, canvas_y+8,
                               fill="red", outline="darkred", width=2, tags="goal")
        self.canvas.create_text(canvas_x, canvas_y-15, text="GOAL",
                               font=("Arial", 9, "bold"), fill="red", tags="goal")
        
        # Start navigation in separate thread
        if not self.navigating:
            nav_thread = threading.Thread(target=self.navigate_to_goal, daemon=True)
            nav_thread.start()
    
    def navigate_to_goal(self):
        """Navigate to goal using direct position control"""
        if self.goal_position is None or self.navigating:
            return
        
        self.navigating = True
        self.path_points = []
        
        goal_x, goal_y = self.goal_position
        max_steps = 1000
        goal_threshold = 2.0
        
        # Flight parameters
        FORWARD_VELOCITY = 3.0
        TARGET_ALTITUDE = -5.0
        YAW_RATE_SCALE = 0.8
        UPDATE_INTERVAL = 0.2  # Slower updates to prevent buffer overflow
        
        last_collision_time = 0
        collision_cooldown = 2.0
        
        print(f"🚀 Navigating to ({goal_x:.1f}, {goal_y:.1f})...")
        
        try:
            for step in range(max_steps):
                if not self.navigating:
                    break
                
                current_time = time.time()
                
                # Get current position with error handling
                try:
                    state = self.client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    current_x, current_y, current_z = pos.x_val, pos.y_val, pos.z_val
                except BufferError:
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"State read error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Calculate distance and angle to goal
                dx = goal_x - current_x
                dy = goal_y - current_y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Check if goal reached
                if distance < goal_threshold:
                    print(f"✅ GOAL REACHED! Final distance: {distance:.2f}m")
                    try:
                        self.client.moveByVelocityAsync(0, 0, 0, 1)
                    except:
                        pass
                    self.navigating = False
                    break
                
                # Check for collision with smart recovery
                try:
                    collision_info = self.client.simGetCollisionInfo()
                    has_collided = collision_info.has_collided
                except:
                    has_collided = False
                
                if has_collided and (current_time - last_collision_time) > collision_cooldown:
                    last_collision_time = current_time
                    
                    try:
                        # Smart collision recovery based on depth sensing
                        print(f"⚠️  Collision detected! Analyzing obstacle...")
                        
                        # Stop immediately
                        self.client.moveByVelocityAsync(0, 0, 0, 0.2)
                        time.sleep(0.3)
                        
                        # Get depth images to detect obstacle type/position
                        recovery_direction = self._analyze_obstacle_and_recover()
                        
                        if recovery_direction == "UP":
                            print("  → Large obstacle (building/wall) - climbing over")
                            self.client.moveByVelocityAsync(0, 0, -2.5, 1.5)  # Fast climb
                            time.sleep(1.6)
                        elif recovery_direction == "LEFT":
                            print("  → Side obstacle (tree/pole) - moving left")
                            self.client.moveByVelocityAsync(-1.5, -2.5, -0.5, 1.2)  # Back-left + slight up
                            time.sleep(1.3)
                        elif recovery_direction == "RIGHT":
                            print("  → Side obstacle (tree/pole) - moving right")
                            self.client.moveByVelocityAsync(-1.5, 2.5, -0.5, 1.2)  # Back-right + slight up
                            time.sleep(1.3)
                        else:  # BACK
                            print("  → Ground/complex obstacle - reversing and climbing")
                            self.client.moveByVelocityAsync(-2.5, 0, -1.5, 1.2)  # Fast back + up
                            time.sleep(1.3)
                        
                        # Clear collision state
                        for _ in range(3):
                            try:
                                self.client.simGetCollisionInfo()
                            except:
                                pass
                            time.sleep(0.1)
                        
                        print("  ✓ Recovery complete")
                    except Exception as e:
                        print(f"Recovery error: {e}")
                    
                    continue
                
                # Calculate desired heading to goal
                goal_angle = np.arctan2(dy, dx)
                
                # Get current yaw with error handling
                try:
                    orientation = state.kinematics_estimated.orientation
                    current_yaw = airsim.to_eularian_angles(orientation)[2]
                except:
                    current_yaw = 0
                
                # Calculate yaw error
                yaw_error = goal_angle - current_yaw
                while yaw_error > np.pi:
                    yaw_error -= 2 * np.pi
                while yaw_error < -np.pi:
                    yaw_error += 2 * np.pi
                
                # Calculate velocities
                yaw_rate = np.clip(yaw_error * YAW_RATE_SCALE, -1.0, 1.0)
                alignment = 1.0 - min(abs(yaw_error) / np.pi, 1.0)
                vx = FORWARD_VELOCITY * alignment
                
                # Altitude control
                altitude_error = TARGET_ALTITUDE - current_z
                vz = np.clip(altitude_error * 0.5, -1.0, 1.0)
                
                # Send velocity command with error handling
                try:
                    self.client.moveByVelocityAsync(
                        vx * np.cos(current_yaw),
                        vx * np.sin(current_yaw),
                        vz,
                        duration=UPDATE_INTERVAL + 0.1,
                        yaw_mode=airsim.YawMode(True, yaw_rate * 50)
                    )
                except BufferError:
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"Command error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Print progress
                if step % 20 == 0:
                    print(f"  Step {step:3d} | Dist: {distance:5.1f}m | Yaw err: {np.degrees(yaw_error):+6.1f}° | Vx: {vx:4.1f}")
                
                time.sleep(UPDATE_INTERVAL)
            
            if self.navigating and step >= max_steps - 1:
                try:
                    final_state = self.client.getMultirotorState()
                    final_pos = final_state.kinematics_estimated.position
                    final_dist = np.sqrt((goal_x - final_pos.x_val)**2 + (goal_y - final_pos.y_val)**2)
                    print(f"⏱️  Timeout. Final distance: {final_dist:.2f}m")
                except:
                    print(f"⏱️  Timeout.")
                
        except Exception as e:
            print(f"Navigation error: {e}")
        
        finally:
            self.navigating = False
            # Ensure stopped
            try:
                self.client.moveByVelocityAsync(0, 0, 0, 1)
            except:
                pass
    
    def update_position_loop(self):
        """Update drone position on map"""
        if self.client is None:
            return
        
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            self.drone_position = (pos.x_val, pos.y_val)
            
            # Update labels
            self.position_label.config(text=f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f})")
            
            # Draw drone on map
            self.canvas.delete("drone")
            canvas_x, canvas_y = self.world_to_canvas(pos.x_val, pos.y_val)
            self.canvas.create_oval(canvas_x-6, canvas_y-6, canvas_x+6, canvas_y+6,
                                   fill="blue", outline="darkblue", width=2, tags="drone")
            
            # Add to path if moved significantly
            if len(self.path_points) == 0 or \
               np.sqrt((pos.x_val - self.path_points[-1][0])**2 + 
                      (pos.y_val - self.path_points[-1][1])**2) > 0.5:
                self.path_points.append((pos.x_val, pos.y_val))
                
                # Draw path
                if len(self.path_points) > 1:
                    self.canvas.delete("path")
                    for i in range(len(self.path_points) - 1):
                        x1, y1 = self.world_to_canvas(*self.path_points[i])
                        x2, y2 = self.world_to_canvas(*self.path_points[i+1])
                        self.canvas.create_line(x1, y1, x2, y2, 
                                               fill="lightblue", width=2, tags="path")
            
        except Exception as e:
            print(f"Update error: {e}")
        
        # Schedule next update
        self.window.after(100, self.update_position_loop)
    
    def _analyze_obstacle_and_recover(self):
        """Analyze obstacle using depth camera to determine best recovery direction"""
        try:
            # Get depth image from front camera
            responses = self.client.simGetImages([
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
    
    def restart_drone(self):
        """Return drone to home/origin position"""
        if not self.running or self.home_position is None:
            print("Please connect to AirSim first!")
            return
        
        if self.navigating:
            self.navigating = False
            time.sleep(0.5)
        
        print(f"\n🏠 Returning to home position: ({self.home_position[0]:.1f}, {self.home_position[1]:.1f})")
        self.status_label.config(text="Status: Returning to home...")
        
        # Set home as the new goal
        self.goal_position = self.home_position
        
        # Clear current goal marker and draw home marker (green circle)
        self.canvas.delete("goal")
        home_canvas = self.world_to_canvas(self.home_position[0], self.home_position[1])
        self.canvas.create_oval(
            home_canvas[0] - 10, home_canvas[1] - 10,
            home_canvas[0] + 10, home_canvas[1] + 10,
            fill='green', outline='darkgreen', width=3, tags="goal"
        )
        
        # Start navigation to home
        threading.Thread(target=self.navigate_to_goal, daemon=True).start()
    
    def stop_navigation(self):
        """Stop current navigation"""
        self.navigating = False
        if self.client:
            self.client.moveByVelocityAsync(0, 0, 0, 1).join()
        print("Navigation stopped")
    
    def run(self):
        """Run the GUI"""
        self.window.mainloop()
        
        # Cleanup on exit
        if self.client:
            try:
                self.client.landAsync().join()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except:
                pass


if __name__ == "__main__":
    import os
    if not os.path.exists("my_airsim_model.pth"):
        print("❌ Error: my_airsim_model.pth not found!")
        exit(1)
    
    app = DroneMapGUI()
    app.run()
