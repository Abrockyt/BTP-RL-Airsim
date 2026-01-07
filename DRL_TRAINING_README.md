# 🚁 Deep Reinforcement Learning for Smooth Drone Navigation

Complete PPO-based training system for learning human-like flight in AirSim.

## 📋 Requirements

```bash
pip install gymnasium stable-baselines3 airsim numpy matplotlib
```

## 🎯 System Overview

### **Goal**: Train drone to fly from (0,0) to (30,0) with smooth, curved obstacle avoidance

### **Key Features**:
- ✅ Continuous action space: Velocity control (vx, vy)
- ✅ Smoothness reward: Penalizes jerky movements
- ✅ Progress tracking: Rewards getting closer to goal
- ✅ Collision avoidance: -100 penalty for crashes
- ✅ Success bonus: +100 for reaching goal

---

## 📁 Files

### 1. `airsim_env.py` - Custom Gymnasium Environment
- **Observation Space** (5 values):
  - Distance to goal (normalized by 100m)
  - Angle to goal (radians)
  - LiDAR left sector (normalized)
  - LiDAR center sector (normalized)
  - LiDAR right sector (normalized)

- **Action Space** (2 values):
  - vx: Forward/backward velocity [-5.0, 5.0] m/s
  - vy: Left/right velocity [-5.0, 5.0] m/s
  - Altitude: Fixed at -6m

- **Reward Function**:
  ```python
  reward = (progress × 10.0)          # Getting closer to goal
         - (velocity_change × 0.5)    # Smoothness penalty (jerk)
         + (forward_bonus × 0.1)      # Encourage movement
         - 100 (if collision)          # Crash penalty
         + 100 (if goal reached)       # Success bonus
         - 0.01 (per step)            # Time penalty
  ```

### 2. `train_pilot.py` - PPO Training Script
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy**: MLP Neural Network
- **Timesteps**: 100,000 (default, configurable)
- **Hyperparameters**:
  - Learning rate: 3e-4
  - Batch size: 64
  - Entropy coefficient: 0.01 (exploration)
  - Clip range: 0.2 (stability)

---

## 🚀 Quick Start

### **Step 1: Test the Environment**
```bash
python airsim_env.py
```
This will test the environment with random actions to verify AirSim connection.

### **Step 2: Train the Model**
```bash
python train_pilot.py --mode train --timesteps 100000
```

**What happens during training:**
- Drone learns from trial and error
- Progress displayed every 10 episodes
- Checkpoints saved every 10,000 steps
- Final model saved as `smooth_drone_policy.zip`

**Training output example:**
```
📊 Episode 10 | Avg Reward (last 10): -45.23 | Success: 0 | Collision: 3 | Timeout: 7
📊 Episode 20 | Avg Reward (last 10): -12.45 | Success: 1 | Collision: 2 | Timeout: 7
📊 Episode 30 | Avg Reward (last 10): 23.67 | Success: 4 | Collision: 1 | Timeout: 5
...
```

### **Step 3: Test the Trained Model**
```bash
python train_pilot.py --mode test --model smooth_drone_policy
```

This runs the trained policy for 5 episodes and shows performance.

---

## 📊 Training Tips

### **Monitor Progress**:
1. **TensorBoard** (optional):
   ```bash
   tensorboard --logdir trained_models/smooth_pilot_*/tensorboard/
   ```

2. **Training Plot**: Automatically generated as `training_progress.png`

### **Adjust Training Duration**:
```bash
# Quick test (10K steps)
python train_pilot.py --timesteps 10000

# Standard training (100K steps)
python train_pilot.py --timesteps 100000

# Extended training (500K steps)
python train_pilot.py --timesteps 500000
```

### **Resume Training**:
```python
# In train_pilot.py, replace model initialization with:
model = PPO.load("smooth_drone_policy", env=env)
model.learn(total_timesteps=50000)  # Continue training
```

---

## 🎮 Expected Behavior

### **Early Training (0-20K steps)**:
- Random movements
- Frequent collisions
- Learning basic controls

### **Mid Training (20K-60K steps)**:
- Starts avoiding obstacles
- Some successful goal reaches
- Still jerky movements

### **Late Training (60K-100K+ steps)**:
- Smooth curved paths
- Consistent goal reaching
- Human-like flight patterns

---

## 🔧 Hyperparameter Tuning

### **To encourage more smoothness**:
```python
# In airsim_env.py, increase smoothness penalty:
smoothness_penalty = -velocity_change * 1.0  # From 0.5
```

### **To learn faster**:
```python
# In train_pilot.py:
learning_rate=5e-4,  # From 3e-4
ent_coef=0.02,       # From 0.01 (more exploration)
```

### **To be more cautious**:
```python
# In airsim_env.py:
self.collision_threshold = 2.0  # From 1.5
reward -= 200.0  # Increase collision penalty
```

---

## 📈 Success Metrics

**Good Training Signs**:
- ✅ Success rate > 70% by 100K steps
- ✅ Average reward trending upward
- ✅ Collision rate decreasing
- ✅ Smooth velocity profiles (low jerk)

**Poor Training Signs**:
- ❌ No improvement after 50K steps
- ❌ Success rate stuck at 0%
- ❌ Reward not increasing

**If training fails**: Increase exploration (ent_coef), reduce learning rate, or simplify environment.

---

## 🎯 Using the Trained Model

### **In Your Own Code**:
```python
from stable_baselines3 import PPO
from airsim_env import AirSimDroneEnv

# Load trained model
model = PPO.load("smooth_drone_policy")

# Create environment
env = AirSimDroneEnv(goal_position=(30.0, 0.0))

# Run policy
obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break

env.close()
```

### **Deploy Different Goals**:
```python
# Test with different goal positions
env = AirSimDroneEnv(goal_position=(50.0, 20.0), fixed_altitude=-8.0)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No lidar found" | Check `airsim_settings.json` has `"LidarSensor1"` |
| Training too slow | Reduce `n_steps` to 1024 in PPO config |
| Drone crashes immediately | Increase collision penalty, reduce max velocity |
| No learning progress | Check reward function, increase timesteps |
| "Connection refused" | Ensure AirSim/Unreal is running |

---

## 📚 Learn More

- **PPO Algorithm**: [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- **Gymnasium**: [Gymnasium Documentation](https://gymnasium.farama.org/)
- **AirSim**: [Microsoft AirSim Docs](https://microsoft.github.io/AirSim/)

---

## 🎉 Next Steps

1. ✅ Train basic model (100K steps)
2. ✅ Test and visualize results
3. 🔄 Fine-tune hyperparameters
4. 🚀 Extend to dynamic goals
5. 🎯 Add obstacle complexity
6. 🧠 Try other algorithms (SAC, TD3)

---

**Happy Training! 🚁🤖**
