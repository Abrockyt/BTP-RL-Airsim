import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO  # The AI Algorithm
import time

# --- CUSTOM GYM ENVIRONMENT (The Bridge) ---
class AirSimDroneEnv(gym.Env):
    def __init__(self):
        super(AirSimDroneEnv, self).__init__()
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # ACTION SPACE: What can the drone do?
        # 0: Hover, 1: Forward, 2: Right, 3: Backward, 4: Left
        self.action_space = spaces.Discrete(5)
        
        # OBSERVATION SPACE: What does the drone "see"?
        # We track: [X_pos, Y_pos, Z_pos, Battery_Level]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.start_battery = 1000.0
        self.current_battery = self.start_battery

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        
        # Move to safe altitude
        self.client.moveToZAsync(-5, 2).join()
        
        self.current_battery = self.start_battery
        return self._get_obs(), {}

    def _get_obs(self):
        # Get real state from AirSim
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        # Return [X, Y, Z, Battery]
        return np.array([pos.x_val, pos.y_val, pos.z_val, self.current_battery], dtype=np.float32)

    def step(self, action):
        # 1. TAKE ACTION
        # Velocity control: vx, vy, vz, duration
        speed = 5
        vx, vy = 0, 0
        
        if action == 1: vx = speed   # Forward
        if action == 2: vy = speed   # Right
        if action == 3: vx = -speed  # Backward
        if action == 4: vy = -speed  # Left
        
        # Send command to AirSim (0.5 second duration per step)
        self.client.moveByVelocityZAsync(vx, vy, -5, 0.5).join()
        
        # 2. CALCULATE ENERGY DRAIN (Simulation)
        # Moving costs more energy than hovering (action 0)
        energy_cost = 0.5 if action == 0 else 2.0 
        self.current_battery -= energy_cost
        
        # 3. CALCULATE REWARD
        obs = self._get_obs()
        distance_from_start = np.linalg.norm(obs[0:2]) # Distance from (0,0)
        
        # REWARD FORMULA: 
        # + Reward for traveling far
        # - Penalty for using battery
        reward = (distance_from_start * 1.0) - (energy_cost * 0.5)
        
        # 4. CHECK IF DONE
        terminated = False
        if self.client.simGetCollisionInfo().has_collided:
            reward = -100  # Big penalty for crash
            terminated = True
        if self.current_battery <= 0:
            terminated = True # End if battery dies
            
        return obs, reward, terminated, False, {}

# --- MAIN TRAINING LOOP ---
# Create the environment
env = AirSimDroneEnv()

# Initialize the AI Agent (PPO is a standard, powerful algorithm)
model = PPO("MlpPolicy", env, verbose=1)

print("Starting training... (Press Ctrl+C to stop)")
# Train for 10,000 steps (takes about 5-10 mins)
model.learn(total_timesteps=10000)

print("Training finished! Saving model...")
model.save("drone_energy_policy")
