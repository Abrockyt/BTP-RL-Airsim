"""
Custom Gymnasium Environment for AirSim Drone Navigation
Implements 3D flight learning in Neighborhood environment with altitude constraints
Trains drone to fly at "medium level" (2-10m altitude) avoiding obstacles
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import time


class AirSimDroneEnv(gym.Env):
    """
    Custom Environment for training 3D drone navigation in AirSim Neighborhood
    
    Observation Space (6D): [distance_to_goal, angle_to_goal, current_altitude, lidar_left, lidar_center, lidar_right]
    Action Space (3D): Continuous velocity control [vx, vy, vz]
    - vx, vy: [-5.0, 5.0] m/s (forward/backward, left/right)
    - vz: [-2.0, 2.0] m/s (up/down - slower vertical movement)
    
    Altitude Constraint: "Medium Level" = -2m to -10m (2-10m above ground in NED)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, goal_position=(30.0, 0.0), randomize_training=True):
        super(AirSimDroneEnv, self).__init__()
        
        # Environment Configuration
        self.goal_position = np.array([goal_position[0], goal_position[1], -5.0])  # Default medium altitude
        self.start_position = np.array([0.0, 0.0, -5.0])  # Start at medium level
        self.lidar_name = "LidarSensor1"
        self.randomize_training = randomize_training
        
        # ALTITUDE CONSTRAINTS - "Medium Level"
        self.altitude_min = -10.0  # Maximum altitude (10m high in NED)
        self.altitude_max = -2.0   # Minimum altitude (2m high in NED)
        self.safe_altitude = -5.0  # Safe medium altitude for takeoff/reset
        
        # Training scenarios - Progressive difficulty with varied obstacles
        self.training_scenarios = [
            # Easy: Short, relatively clear paths
            {'start': (0.0, 0.0), 'goal': (15.0, 0.0), 'difficulty': 'easy'},
            {'start': (0.0, 0.0), 'goal': (10.0, 5.0), 'difficulty': 'easy'},
            {'start': (5.0, 0.0), 'goal': (20.0, 0.0), 'difficulty': 'easy'},
            {'start': (0.0, 5.0), 'goal': (15.0, 5.0), 'difficulty': 'easy'},
            
            # Medium: Navigate between houses/trees
            {'start': (5.0, 5.0), 'goal': (25.0, 5.0), 'difficulty': 'medium'},
            {'start': (10.0, -5.0), 'goal': (30.0, -5.0), 'difficulty': 'medium'},
            {'start': (0.0, 0.0), 'goal': (30.0, 0.0), 'difficulty': 'medium'},
            {'start': (8.0, 3.0), 'goal': (25.0, 8.0), 'difficulty': 'medium'},
            {'start': (12.0, -3.0), 'goal': (28.0, 3.0), 'difficulty': 'medium'},
            
            # Hard: Complex navigation through neighborhood
            {'start': (10.0, -8.0), 'goal': (35.0, -8.0), 'difficulty': 'hard'},
            {'start': (8.0, -5.0), 'goal': (30.0, 10.0), 'difficulty': 'hard'},
            {'start': (6.0, -10.0), 'goal': (22.0, 8.0), 'difficulty': 'hard'},
            {'start': (15.0, 3.0), 'goal': (15.0, 25.0), 'difficulty': 'hard'},
            {'start': (2.0, 12.0), 'goal': (28.0, -6.0), 'difficulty': 'hard'},
            
            # Extreme: Very long distances with multiple obstacles
            {'start': (0.0, 0.0), 'goal': (40.0, 0.0), 'difficulty': 'extreme'},
            {'start': (5.0, -10.0), 'goal': (35.0, 15.0), 'difficulty': 'extreme'},
            {'start': (0.0, 0.0), 'goal': (35.0, 20.0), 'difficulty': 'extreme'},
            {'start': (10.0, -12.0), 'goal': (40.0, 10.0), 'difficulty': 'extreme'},
        ]
        
        # Episode parameters
        self.max_steps = 300  # Reduced from 400 for faster episodes and crash prevention
        self.current_step = 0
        self.goal_threshold = 2.0  # meters - distance to consider goal reached
        self.collision_threshold = 2.5  # meters - minimum safe distance
        self.episode_count = 0  # Track episodes for curriculum
        
        # Smoothness tracking
        self.last_action = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self.last_position = self.start_position.copy()
        
        # Define Action Space: 3D Continuous velocity control [vx, vy, vz]
        # vx, vy: [-2.0, 2.0] m/s for horizontal movement (further reduced to prevent UE4 crashes)
        # vz: [-1.0, 1.0] m/s for vertical movement (reduced for maximum stability)
        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # Define Observation Space (6D): [dist_to_goal_norm, angle_to_goal, altitude_norm, lidar_left, lidar_center, lidar_right]
        # altitude_norm: current altitude normalized to [0, 1] range (altitude_min to altitude_max)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, np.pi, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )
        
        # AirSim Client
        self.client = None
        self._connect_airsim()
    
    def _connect_airsim(self):
        """Connect to AirSim and initialize drone"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("✓ Connected to AirSim")
        except Exception as e:
            print(f"❌ Failed to connect to AirSim: {e}")
            raise
    
    def _get_lidar_data(self):
        """
        Get LiDAR data and segment into Left, Center, Right sectors
        Returns: (left_min, center_min, right_min) normalized by 30m
        """
        try:
            lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
            
            if len(lidar_data.point_cloud) < 3:
                return 1.0, 1.0, 1.0  # No obstacles = maximum distance
            
            # Convert to numpy array
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape((-1, 3))
            
            # Filter points at flight level (±3m vertical tolerance)
            flight_level_mask = (points[:, 2] > -3.0) & (points[:, 2] < 3.0)
            points = points[flight_level_mask]
            
            if len(points) == 0:
                return 1.0, 1.0, 1.0
            
            # Calculate distances and angles
            distances = np.linalg.norm(points[:, :2], axis=1)  # XY distance only
            angles = np.arctan2(points[:, 1], points[:, 0])
            
            # Sector division
            left_mask = (angles > np.pi/6) & (angles <= np.pi)
            center_mask = (angles >= -np.pi/6) & (angles <= np.pi/6)
            right_mask = (angles < -np.pi/6) & (angles >= -np.pi)
            
            # Get minimum distance in each sector
            left_min = distances[left_mask].min() if np.any(left_mask) else 30.0
            center_min = distances[center_mask].min() if np.any(center_mask) else 30.0
            right_min = distances[right_mask].min() if np.any(right_mask) else 30.0
            
            # Normalize by 30m (max sensing range)
            return (
                min(left_min / 30.0, 1.0),
                min(center_min / 30.0, 1.0),
                min(right_min / 30.0, 1.0)
            )
            
        except Exception as e:
            print(f"LiDAR error: {e}")
            return 1.0, 1.0, 1.0
    
    def _get_observation(self):
        """
        Get current observation vector (6D)
        Returns: [distance_to_goal_norm, angle_to_goal, altitude_norm, lidar_left, lidar_center, lidar_right]
        
        altitude_norm: Normalized altitude in range [0, 1]
        - 0.0 = altitude_max (-2m, 2m high - too low!)
        - 1.0 = altitude_min (-10m, 10m high - too high!)
        - 0.5 = safe_altitude (-5m, 5m high - perfect!)
        """
        # Get current position
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Calculate goal vector
        goal_vector = self.goal_position[:2] - current_pos[:2]  # XY only
        distance_to_goal = np.linalg.norm(goal_vector)
        angle_to_goal = np.arctan2(goal_vector[1], goal_vector[0])
        
        # Normalize distance by 100m
        distance_norm = min(distance_to_goal / 100.0, 1.0)
        
        # Normalize current altitude to [0, 1] range
        # altitude_min (-10m) -> 1.0, altitude_max (-2m) -> 0.0
        altitude_norm = (current_pos[2] - self.altitude_max) / (self.altitude_min - self.altitude_max)
        altitude_norm = np.clip(altitude_norm, 0.0, 1.0)
        
        # Get LiDAR data
        lidar_left, lidar_center, lidar_right = self._get_lidar_data()
        
        obs = np.array([
            distance_norm,
            angle_to_goal,
            altitude_norm,  # Normalized altitude (0=too low, 1=too high, 0.5=perfect)
            lidar_left,
            lidar_center,
            lidar_right
        ], dtype=np.float32)
        
        return obs, current_pos, distance_to_goal
    
    def _calculate_reward(self, current_pos, distance_to_goal, action, lidar_center, lidar_left, lidar_right):
        """
        Calculate reward for 3D navigation with "medium level" altitude constraint
        
        Components:
        1. Progress reward: Getting closer to goal
        2. Smoothness penalty: Penalize jerky movements
        3. **ALTITUDE PENALTY**: Flying too high (Z < -10) or too low (Z > -2)
        4. **OBSTACLE AVOIDANCE BONUS**: Reward for navigating near obstacles safely
        5. Collision penalty: Hit obstacle
        6. Goal bonus: Reached target
        7. Time penalty: Encourage efficiency
        """
        reward = 0.0
        done = False
        info = {}
        
        # 1. PROGRESS REWARD - Moving closer to goal
        last_distance = np.linalg.norm(self.last_position[:2] - self.goal_position[:2])
        progress = last_distance - distance_to_goal
        reward += progress * 15.0  # Increased from 10 to encourage goal-seeking
        
        # 2. SMOOTHNESS PENALTY - Penalize jerky movements
        velocity_change = np.linalg.norm(action - self.last_action)
        smoothness_penalty = -velocity_change * 0.3  # Reduced to allow more aggressive maneuvering
        reward += smoothness_penalty
        
        # 3. **ALTITUDE PENALTY** - "Medium Level" Constraint
        current_altitude = current_pos[2]
        
        if current_altitude < self.altitude_min:  # Too high (Z < -10m)
            altitude_penalty = -1.0
            reward += altitude_penalty
        
        if current_altitude > self.altitude_max:  # Too low (Z > -2m)
            altitude_penalty = -1.0
            reward += altitude_penalty
        
        # Bonus for staying in safe zone
        if self.altitude_min <= current_altitude <= self.altitude_max:
            reward += 0.15
        
        # 4. **OBSTACLE AVOIDANCE BONUS** - Reward for navigating near obstacles safely
        # Encourage getting close to obstacles without collision
        min_clearance = min(lidar_left, lidar_center, lidar_right)
        if 0.15 < min_clearance < 0.4:  # Between 4.5m and 12m - navigating near obstacles
            reward += 0.5  # Bonus for navigating in tight spaces
        elif min_clearance < 0.15:  # Too close to obstacle
            reward -= 0.5  # Small penalty for being too risky
        
        # Bonus for balanced obstacle avoidance (using all sensors)
        if lidar_center < 0.5:  # Obstacle ahead
            # Reward if using left or right path
            if lidar_left > lidar_center or lidar_right > lidar_center:
                reward += 0.3  # Bonus for smart path selection
        
        # 5. FORWARD MOTION BONUS - Encourage movement toward goal
        forward_velocity = action[0]  # X velocity
        if forward_velocity > 0 and distance_to_goal > 5.0:  # Only when far from goal
            reward += 0.3
        
        # 6. COLLISION DETECTION
        if lidar_center < 0.08:  # ~2.5m (normalized by 30m)
            reward -= 50.0
            done = True
            info['result'] = 'collision'
            print("❌ COLLISION!")
        
        # 7. GOAL SUCCESS - Big reward for reaching goal
        if distance_to_goal < self.goal_threshold:
            reward += 150.0  # Increased from 100
            done = True
            info['result'] = 'success'
            print(f"✅ GOAL REACHED! Distance: {distance_to_goal:.2f}m, Altitude: {-current_pos[2]:.1f}m")
        
        # 8. TIMEOUT
        if self.current_step >= self.max_steps:
            done = True
            info['result'] = 'timeout'
            print(f"⏱️  Timeout. Final distance: {distance_to_goal:.2f}m, Altitude: {-current_pos[2]:.1f}m")
        
        # 9. TIME PENALTY - Encourage faster completion
        reward -= 0.01
        
        # Update tracking
        self.last_action = action.copy()
        self.last_position = current_pos.copy()
        
        return reward, done, info
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to starting state
        Ensures drone starts at safe medium altitude (-5m)
        """
        super().reset(seed=seed)
        
        # Randomize scenario during training with curriculum learning
        if self.randomize_training:
            # Progressive curriculum: Easy → Medium → Hard → Extreme
            if self.episode_count < 40:  # First 40 episodes: only easy
                easy_scenarios = [s for s in self.training_scenarios if s.get('difficulty') == 'easy']
                scenario = easy_scenarios[np.random.randint(0, len(easy_scenarios))]
            elif self.episode_count < 120:  # Next 80: easy + medium
                em_scenarios = [s for s in self.training_scenarios if s.get('difficulty') in ['easy', 'medium']]
                scenario = em_scenarios[np.random.randint(0, len(em_scenarios))]
            elif self.episode_count < 250:  # Next 130: easy + medium + hard
                emh_scenarios = [s for s in self.training_scenarios if s.get('difficulty') in ['easy', 'medium', 'hard']]
                scenario = emh_scenarios[np.random.randint(0, len(emh_scenarios))]
            else:  # After 250: all scenarios including extreme
                scenario = self.training_scenarios[np.random.randint(0, len(self.training_scenarios))]
            
            # Set start/goal at safe medium altitude
            self.start_position = np.array([scenario['start'][0], scenario['start'][1], self.safe_altitude])
            self.goal_position = np.array([scenario['goal'][0], scenario['goal'][1], self.safe_altitude])
            print(f"\n🎯 Ep {self.episode_count} [{scenario.get('difficulty', 'N/A')}]: {scenario['start']} → {scenario['goal']} @ {-self.safe_altitude:.0f}m altitude")
            self.episode_count += 1
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff to safe medium altitude (-5m)
        print(f"  ↑ Taking off to medium level ({-self.safe_altitude:.0f}m)...")
        self.client.takeoffAsync().join()
        
        # Move to start position at safe altitude
        self.client.moveToPositionAsync(
            self.start_position[0],
            self.start_position[1],
            self.safe_altitude,  # Ensure we start at safe medium altitude
            5.0
        ).join()
        
        time.sleep(0.5)  # Stabilize
        
        # Verify altitude
        state = self.client.getMultirotorState()
        actual_z = state.kinematics_estimated.position.z_val
        print(f"  ✓ Drone ready at altitude: {-actual_z:.1f}m (target: {-self.safe_altitude:.0f}m)")
        
        # Reset episode variables
        self.current_step = 0
        self.last_action = np.array([0.0, 0.0, 0.0])  # vx, vy, vz
        self.last_position = self.start_position.copy()
        
        # Get initial observation
        obs, _, _ = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        """
        Execute 3D action and return next state
        
        Args:
            action: [vx, vy, vz] velocity command
            - vx, vy: [-5.0, 5.0] m/s horizontal
            - vz: [-2.0, 2.0] m/s vertical (slower)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid ranges
        # vx, vy: [-2, 2], vz: [-1, 1] (ultra-conservative to prevent UE4 crashes)
        action[0] = np.clip(action[0], -2.0, 2.0)
        action[1] = np.clip(action[1], -2.0, 2.0)
        action[2] = np.clip(action[2], -1.0, 1.0)  # Slower vertical movement
        
        # Execute 3D velocity command with crash recovery
        try:
            self.client.moveByVelocityAsync(
                float(action[0]),  # vx - forward/backward
                float(action[1]),  # vy - left/right  
                float(action[2]),  # vz - up/down
                0.2  # Increased duration to reduce UE4 load
            ).join()
            time.sleep(0.02)  # Small delay to prevent overwhelming UE4
        except Exception as e:
            print(f"⚠️ Command failed: {e}")
            try:
                self._connect_airsim()
            except:
                pass
        
        # Get new observation
        obs, current_pos, distance_to_goal = self._get_observation()
        
        # Calculate reward with all LiDAR data
        reward, done, info = self._calculate_reward(
            current_pos,
            distance_to_goal,
            action,
            obs[4],  # lidar_center
            obs[3],  # lidar_left
            obs[5]   # lidar_right
        )
        
        self.current_step += 1
        
        # Gymnasium API uses 'terminated' and 'truncated' instead of 'done'
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up AirSim connection"""
        if self.client:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✓ Environment closed")


# Test the environment
if __name__ == "__main__":
    print("="*60)
    print("Testing AirSim Gymnasium Environment")
    print("="*60)
    
    env = AirSimDroneEnv()
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few random steps
    print("\nTesting random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Obs={obs}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("\n✓ Environment test complete!")
