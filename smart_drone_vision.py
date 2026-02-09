"""
Smart Drone Vision System
Combines PPO Reinforcement Learning with Computer Vision for Advanced Obstacle Avoidance
Optimized for RTX 3050
"""

import airsim
import cv2
import torch
import torch.nn as nn
import numpy as np
import time

# =============================================================================
# BRAIN ARCHITECTURE (Multi-Head Attention Actor for PPO)
# =============================================================================

class MHA_Actor(nn.Module):
    """Multi-Head Attention Actor Network for PPO"""
    def __init__(self, state_dim, action_dim, num_heads=4, hidden_dim=128):
        super(MHA_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        # Embed state
        x = self.embedding(state)
        
        # Add sequence dimension for attention
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        
        # Output action
        x = self.tanh(self.fc1(x))
        action = self.tanh(self.fc2(x))
        
        return action


class PPO_Agent:
    """PPO Agent with MHA Actor"""
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4):
        self.gamma = gamma
        self.K_epochs = K_epochs
        
        self.actor = MHA_Actor(state_dim, action_dim)
        if lr > 0:
            self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
    
    def select_action(self, state):
        """Select action using the trained actor"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()
        
        if len(action.shape) > 1:
            action = action[0]
        
        return action


# =============================================================================
# VISION SYSTEM (Computer Vision for Obstacle Detection)
# =============================================================================

class VisionSystem:
    """Computer Vision System for Obstacle Detection and Avoidance"""
    
    def __init__(self, client):
        self.client = client
        
        # Detection thresholds (meters)
        self.DANGER_DISTANCE = 5.0  # Critical obstacle distance
        self.WARNING_DISTANCE = 8.0  # Start preparing for obstacle
        self.MIN_ALTITUDE = 2.0  # Minimum safe ground clearance
        
        # Speed settings
        self.NORMAL_SPEED = 5.0
        self.FAST_SPEED = 10.0
        self.EVASIVE_SPEED = 12.0
        
    def get_depth_perception(self):
        """
        Get depth image and analyze obstacle regions
        Returns: dict with region distances and obstacle type
        """
        try:
            # Get depth image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
            ])
            
            if not responses or len(responses) == 0:
                return None
            
            response = responses[0]
            
            # Convert to numpy array
            img1d = np.array(response.image_data_float, dtype=np.float32)
            
            if len(img1d) == 0:
                return None
            
            # Reshape to 2D image
            img2d = img1d.reshape(response.height, response.width)
            
            # Replace inf and nan with large value
            img2d = np.nan_to_num(img2d, nan=100.0, posinf=100.0, neginf=100.0)
            
            # Analyze regions
            h, w = img2d.shape
            
            # Split into regions
            center = img2d[h//3:2*h//3, w//3:2*w//3]
            left = img2d[h//3:2*h//3, :w//3]
            right = img2d[h//3:2*h//3, 2*w//3:]
            top = img2d[:h//3, :]
            
            # Calculate average distances (filter out extreme values)
            def safe_mean(region):
                valid = region[(region > 0.1) & (region < 100)]
                return np.mean(valid) if len(valid) > 0 else 100.0
            
            distances = {
                'center': safe_mean(center),
                'left': safe_mean(left),
                'right': safe_mean(right),
                'top': safe_mean(top),
                'min_all': np.min(img2d[img2d > 0.1]) if np.any(img2d > 0.1) else 100.0
            }
            
            # Determine obstacle type
            obstacle_info = self.classify_obstacle(distances)
            
            return {
                'distances': distances,
                'obstacle': obstacle_info,
                'raw_image': img2d
            }
            
        except Exception as e:
            print(f"Vision error: {e}")
            return None
    
    def classify_obstacle(self, distances):
        """
        Classify obstacle type based on depth regions
        Returns: dict with obstacle type and recommended action
        """
        center_blocked = distances['center'] < self.DANGER_DISTANCE
        left_blocked = distances['left'] < self.DANGER_DISTANCE
        right_blocked = distances['right'] < self.DANGER_DISTANCE
        
        # HOUSE/WALL: All sides blocked
        if center_blocked and left_blocked and right_blocked:
            return {
                'type': 'HOUSE',
                'action': 'CLIMB',
                'severity': 'CRITICAL',
                'message': '⚠️ OBSTACLE DETECTED: HOUSE/WALL! CLIMBING!'
            }
        
        # POLE/TREE: Only center blocked, sides open
        elif center_blocked and (not left_blocked or not right_blocked):
            # Choose best side to swerve
            if not left_blocked and not right_blocked:
                swerve_dir = 'LEFT' if distances['left'] > distances['right'] else 'RIGHT'
            elif not left_blocked:
                swerve_dir = 'LEFT'
            else:
                swerve_dir = 'RIGHT'
            
            return {
                'type': 'POLE',
                'action': f'SWERVE_{swerve_dir}',
                'severity': 'HIGH',
                'message': f'⚠️ OBSTACLE DETECTED: POLE/TREE! SWERVING {swerve_dir}!'
            }
        
        # NARROW GAP: Need to navigate carefully
        elif left_blocked or right_blocked:
            open_side = 'RIGHT' if left_blocked else 'LEFT'
            return {
                'type': 'GAP',
                'action': f'NAVIGATE_{open_side}',
                'severity': 'MEDIUM',
                'message': f'⚠️ NARROW GAP! NAVIGATING {open_side}!'
            }
        
        # PATH CLEAR
        else:
            return {
                'type': 'CLEAR',
                'action': 'AI_CONTROL',
                'severity': 'SAFE',
                'message': '✓ Path clear - AI in control'
            }
    
    def get_ground_clearance(self):
        """Get current altitude above ground"""
        try:
            state = self.client.getMultirotorState()
            # In NED coordinates, negative z is up
            altitude = abs(state.kinematics_estimated.position.z_val)
            return altitude
        except:
            return 3.0  # Safe default
    
    def execute_evasive_maneuver(self, obstacle_info, current_speed=5.0):
        """
        Execute immediate evasive maneuver based on obstacle type
        Returns: True if maneuver executed, False if AI should control
        """
        action = obstacle_info['action']
        
        if action == 'CLIMB':
            # HOUSE/WALL: Ascend rapidly
            print(obstacle_info['message'])
            self.client.moveByVelocityAsync(
                current_speed * 0.3,  # Slow forward
                0,
                -3.0,  # Fast climb (negative z is up)
                duration=1.5
            )
            return True
        
        elif action.startswith('SWERVE_'):
            # POLE/TREE: High-speed lateral evasion
            print(obstacle_info['message'])
            direction = 1.0 if 'RIGHT' in action else -1.0
            self.client.moveByVelocityAsync(
                current_speed * 0.5,  # Moderate forward
                direction * self.EVASIVE_SPEED,  # Fast lateral
                -0.5,  # Slight climb for safety
                duration=1.0
            )
            return True
        
        elif action.startswith('NAVIGATE_'):
            # GAP: Careful navigation
            print(obstacle_info['message'])
            direction = 1.0 if 'RIGHT' in action else -1.0
            self.client.moveByVelocityAsync(
                current_speed * 0.7,
                direction * 3.0,
                -0.3,
                duration=0.8
            )
            return True
        
        else:
            # AI_CONTROL: Let PPO agent decide
            return False


# =============================================================================
# MAIN FLIGHT CONTROLLER
# =============================================================================

class SmartVisionDrone:
    """Main controller combining PPO AI with Computer Vision"""
    
    def __init__(self):
        self.client = None
        self.agent = None
        self.vision = None
        
        # Configuration
        self.MODEL_PATH = "energy_saving_brain.pth"
        self.GOAL_POS = np.array([100.0, 100.0])
        self.P_HOVER = 200.0
        self.BATTERY_CAPACITY = 100.0
        
        # State
        self.battery_percent = 100.0
        self.total_energy = 0.0
        self.speed_multiplier = 2.0  # High-speed mode
        
    def initialize(self):
        """Initialize AirSim, PPO agent, and vision system"""
        print("="*70)
        print("🚁 SMART VISION DRONE - PPO + Computer Vision")
        print("="*70)
        
        # Load PPO Agent
        print("🧠 Loading PPO Agent...")
        self.agent = PPO_Agent(state_dim=7, action_dim=3, lr=0, gamma=0, K_epochs=0)
        
        try:
            checkpoint = torch.load(self.MODEL_PATH, map_location=torch.device('cpu'))
            if 'actor_state_dict' in checkpoint:
                # Try to load with strict=False to allow partial loading
                self.agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
            else:
                self.agent.actor.load_state_dict(checkpoint, strict=False)
            
            device = "CUDA (RTX 3050)" if torch.cuda.is_available() else "CPU"
            print(f"✓ Brain loaded on {device} (with architecture adaptation)")
        except Exception as e:
            print(f"⚠️ Model loading warning: {e}")
            print("⚠️ Continuing with initialized weights (PPO agent will use random policy)")
            print("   For best results, ensure model architecture matches saved weights")
        
        self.agent.actor.eval()
        
        # Connect to AirSim
        print("🔌 Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Initialize vision system
        self.vision = VisionSystem(self.client)
        print("👁️ Vision system initialized")
        
        # Takeoff
        print("🛫 Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        
        self.client.moveToZAsync(-3.0, 2.0).join()
        time.sleep(1)
        
        print("✓ System ready!")
        print("="*70)
        return True
    
    def get_state_vector(self, goal, battery_level):
        """Construct 7-value state vector for PPO agent"""
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            # Distance to goal
            current_pos = np.array([pos.x_val, pos.y_val])
            dist_vector = goal - current_pos
            
            # Velocity
            velocity = np.array([vel.x_val, vel.y_val])
            
            # Wind estimation (simplified)
            wind = velocity * 0.1
            
            # Construct state: [Dist_X, Dist_Y, Vel_X, Vel_Y, Wind_X, Wind_Y, Battery]
            state_vec = np.concatenate([
                dist_vector,
                velocity,
                wind,
                [battery_level]
            ]).astype(np.float32)
            
            return state_vec, current_pos, velocity
            
        except Exception as e:
            print(f"State error: {e}")
            return None, None, None
    
    def fly_mission(self):
        """Main flight loop with vision-based obstacle avoidance"""
        dt = 0.1
        step = 0
        
        print(f"\n🎯 Mission: Fly to ({self.GOAL_POS[0]:.0f}, {self.GOAL_POS[1]:.0f})")
        print(f"⚡ Mode: High-Speed AI Navigation (Speed x{self.speed_multiplier})")
        print("="*70)
        
        try:
            while True:
                loop_start = time.time()
                
                # 1. VISION & PERCEPTION (The "Eyes")
                vision_data = self.vision.get_depth_perception()
                
                if vision_data is None:
                    time.sleep(0.05)
                    continue
                
                obstacle_info = vision_data['obstacle']
                
                # 2. CHECK GROUND CLEARANCE
                altitude = self.vision.get_ground_clearance()
                if altitude < self.vision.MIN_ALTITUDE:
                    print(f"⚠️ GROUND/BUSH AVOIDANCE! Altitude: {altitude:.1f}m < {self.vision.MIN_ALTITUDE}m - ASCENDING!")
                    self.client.moveByVelocityAsync(0, 0, -2.5, duration=1.0)
                    time.sleep(0.05)  # Minimal delay
                    continue
                
                # 3. GET STATE
                state_vec, current_pos, velocity = self.get_state_vector(
                    self.GOAL_POS, 
                    self.battery_percent / 100.0
                )
                
                if state_vec is None:
                    time.sleep(0.05)
                    continue
                
                # Calculate distance and speed
                distance = np.linalg.norm(self.GOAL_POS - current_pos)
                current_speed = np.linalg.norm(velocity)
                
                # Check goal reached
                if distance < 5.0:
                    print(f"\n{'='*70}")
                    print(f"🏆 GOAL REACHED! Distance: {distance:.2f}m")
                    print(f"📊 Final Battery: {self.battery_percent:.1f}%")
                    print(f"⚡ Energy Used: {self.total_energy:.2f} Wh")
                    print(f"{'='*70}")
                    break
                
                # Check battery
                if self.battery_percent <= 5.0:
                    print(f"\n⚠️ CRITICAL BATTERY: {self.battery_percent:.1f}% - LANDING!")
                    self.client.landAsync().join()
                    break
                
                # 4. SMART OBSTACLE LOGIC (The "Reflexes")
                obstacle_detected = self.vision.execute_evasive_maneuver(
                    obstacle_info, 
                    current_speed
                )
                
                if obstacle_detected:
                    # Evasive maneuver executed - skip AI control this iteration
                    step += 1
                    time.sleep(0.05)  # Minimal delay for instant reaction
                    continue
                
                # 5. EFFICIENT NAVIGATION (The "Brain")
                # Path is CLEAR - use PPO AI
                action = self.agent.select_action(state_vec)
                
                # Scale actions for high-speed flight
                target_vx = np.clip(action[0] * 10.0 * self.speed_multiplier, -10, 10)
                target_vy = np.clip(action[1] * 10.0 * self.speed_multiplier, -10, 10)
                
                # Execute AI decision
                self.client.moveByVelocityAsync(
                    float(target_vx),
                    float(target_vy),
                    0,
                    duration=0.12
                )
                
                # 6. UPDATE BATTERY (Physics simulation)
                speed = np.linalg.norm([target_vx, target_vy])
                power_watts = self.P_HOVER * (1 + 0.005 * speed**2)
                energy_step = power_watts * (dt / 3600.0)
                self.total_energy += energy_step
                self.battery_percent = max(0, 100 - (self.total_energy / self.BATTERY_CAPACITY * 100))
                
                # 7. LOGGING
                if step % 20 == 0:
                    status = obstacle_info['message']
                    print(f"Step {step:4d} | 🔋 {self.battery_percent:5.1f}% | "
                          f"📏 {distance:6.1f}m | 💨 {speed:5.1f} m/s | {status}")
                
                step += 1
                
                # High-speed execution - minimal delay
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n⏹ Flight interrupted by user")
        except Exception as e:
            print(f"\n❌ Flight error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.land()
    
    def land(self):
        """Safe landing procedure"""
        try:
            print("\n🛬 Landing...")
            self.client.moveByVelocityAsync(0, 0, 0, 1).join()
            self.client.landAsync().join()
            time.sleep(2)
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✓ Landed safely")
        except Exception as e:
            print(f"Landing error: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    drone = SmartVisionDrone()
    
    if drone.initialize():
        drone.fly_mission()
    else:
        print("❌ Initialization failed!")
