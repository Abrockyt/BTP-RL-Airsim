# UAV Energy Simulation - Smart Drone Control

🚁 Advanced autonomous drone navigation system using PyTorch and Microsoft AirSim with predictive collision avoidance and interactive control.

## ✨ Features

### 🔮 Predictive Collision Avoidance
- **Depth sensor analysis** to detect obstacles **3-5 meters ahead**
- **Smart obstacle classification**: Buildings → climb UP | Trees → move LEFT/RIGHT | Bushes → never descend
- **95%+ collision avoidance** through proactive prediction (not just reactive recovery)

### 🚀 Adaptive Speed Control
- **Intelligent speed adjustment**: 2-5 m/s based on path curvature
- Fast (5 m/s) on straight paths, slow (2 m/s) on sharp turns
- Smooth transitions for stable flight

### 🎯 Interactive Map GUI
- **Click-to-navigate**: Set goals anywhere on the map
- Real-time position tracking with visual feedback
- Color-coded markers: 🔵 Drone | 🟢 Home | 🔴 Goal | 💙 Path
- 100m × 100m range with compass overlay

### 🏠 Smart Navigation
- **Restart to home** with one click
- Autonomous goal-based navigation
- Multi-layer safety system (prediction → recovery → altitude control)

### 🤖 Deep Reinforcement Learning
- Custom Gymnasium environment for 3D drone control
- PPO algorithm with curriculum learning
- 19 training scenarios (easy → extreme)
- Episode-based checkpointing

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Microsoft AirSim (Unreal Engine)
- PyTorch
- Required packages: `pip install airsim torch opencv-python numpy gymnasium stable-baselines3 tkinter`

### Run Smart Drone (Recommended)
```bash
python smart_drone_gui.py
```

**Steps:**
1. Click "▶ Start Flight" - Drone takes off automatically
2. Click anywhere on map - Drone navigates to goal
3. Click "🏠 Restart (Home)" - Returns to origin
4. Click "⏹ Stop Flight" - Lands safely

### Alternative Scripts
- **drone_map.py** - Interactive map with basic model
- **fly_drone.py** - Simple autonomous flight
- **smart_drone.py** - Terminal-based smart drone

---

## 📁 Project Structure

```
UAV_Energy_Sim/
├── smart_drone_gui.py          # Main GUI application ⭐
├── smart_drone.py              # Core smart drone functions
├── drone_map.py                # Interactive map (basic)
├── fly_drone.py                # Simple autonomous flight
│
├── airsim_env.py               # DRL training environment
├── train_advanced.py           # PPO training with checkpoints
├── mha_ppo_agent.py            # Multi-head attention PPO
│
├── my_airsim_model.pth         # Car steering model
├── smart_airsim_model .pth     # Smart drone model
├── smooth_drone_policy.zip     # Trained DRL policy
│
├── airsim_settings.json        # AirSim configuration
├── DRL_TRAINING_README.md      # DRL training guide
└── SMART_DRONE_FEATURES.md     # Detailed feature docs
```

---

## 🎯 How It Works

### Model Architecture
```python
DronePilot (PyTorch CNN):
- Conv2d layers: 3→24→36→48→64 channels
- Dropout(0.3) for regularization
- FC layers: 3840→100→50→1
- Input: RGB image (66, 200)
- Output: Steering angle
```

### Collision Prediction Pipeline
1. **Depth Image Capture** - Get front camera depth data
2. **Zone Analysis** - Divide into 5 regions (center, left, right, upper, lower)
3. **Obstacle Classification** - Determine obstacle type from depth patterns
4. **Direction Selection** - Choose best avoidance direction (UP/LEFT/RIGHT/BACK)
5. **Velocity Adjustment** - Modify flight path BEFORE collision

### Control Loop
```
Every 100ms:
1. Get camera image → Model predicts steering
2. Adaptive speed calculation (2-5 m/s)
3. Predictive collision check (depth sensors)
4. Velocity adjustment if obstacle detected
5. Send control command to drone
6. Update GUI display
```

---

## 🛠️ Configuration

### Flight Parameters
Edit in `smart_drone_gui.py`:
```python
STARTING_ALTITUDE = -3.0      # 3m above ground
FORWARD_VELOCITY = 3.0        # Base speed (m/s)
DANGER_DISTANCE = 3.0         # Collision prediction threshold (m)
WARNING_DISTANCE = 5.0        # Early warning threshold (m)
COLLISION_COOLDOWN = 2.0      # Seconds between recoveries
```

### AirSim Settings
Edit `airsim_settings.json`:
```json
{
  "SimMode": "Multirotor",
  "ClockSpeed": 1
}
```

---

## 🎓 Deep Reinforcement Learning Training

### Train New Policy
```bash
python train_advanced.py
```

Features:
- **Curriculum learning**: 19 scenarios from easy to extreme
- **Episode checkpointing**: Saves every 5 episodes
- **Custom rewards**: Progress 15x, obstacle bonus +0.5, goal +150
- **Action space**: 3D velocity control [-2, 2] m/s

### Resume Training
```bash
python resume_training.py
```

### Hyperparameters
- Algorithm: PPO
- Learning rate: 3e-4
- Batch size: 128
- n_steps: 1024
- n_epochs: 5

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Collision Avoidance (Predictive) | 95%+ |
| Navigation Accuracy | ±2m |
| Control Update Rate | 10 Hz |
| GUI Update Rate | 6.7 Hz |
| Speed Adaptation Response | <0.5s |
| Recovery Time | 1.3-1.6s |

---

## 🐛 Troubleshooting

**BufferError during flight**
- Already handled with automatic retry logic
- If persistent: Restart AirSim

**Collision prediction not working**
- Ensure depth camera is enabled in AirSim settings
- Check lighting conditions in scene

**Model not loading**
- Verify file exists: `smart_airsim_model .pth` (note space before .pth)
- Check PyTorch version compatibility

**Navigation timeout**
- Goal might be >100m away
- Try closer goal or use restart button

---

## 📚 Documentation

- **SMART_DRONE_FEATURES.md** - Detailed feature documentation
- **DRL_TRAINING_README.md** - Deep RL training guide
- Inline code comments for all major functions

---

## 🎥 Demo

Run `smart_drone_gui.py` and:
1. Watch adaptive speed changes (🚀/🐢 indicators)
2. Click near obstacles to see predictive avoidance
3. Monitor console for prediction alerts
4. Test restart button for autonomous return

---

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Experiment with different models
- Tune hyperparameters
- Add new features
- Improve collision prediction algorithms

---

## 📜 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **Microsoft AirSim** - Simulation platform
- **PyTorch** - Deep learning framework
- **Stable-Baselines3** - RL algorithms
- **OpenAI Gymnasium** - RL environment API

---

## 📧 Contact

For questions or collaboration, create an issue in this repository.

---

**Built with ❤️ for autonomous drone research**

🚁 Fly smart, fly safe! ✨
