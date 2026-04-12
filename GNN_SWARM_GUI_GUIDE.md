# 🚁 GNN SWARM GUI - User Guide

## Overview
Advanced multi-drone system with:
- ✅ **GNN-based navigation** - Graph Neural Network for swarm intelligence
- ✅ **Collision detection and avoidance** - Real-time collision risk prediction
- ✅ **Professional GUI** - Real-time telemetry and graphs
- ✅ **Background drones** - 6 drones moving randomly
- ✅ **Goal navigation** - Main drone reaches goal perfectly

---

## 🚀 Quick Start

### 1. Start AirSim/Unreal Engine
Make sure your AirSim environment is running with 10-drone configuration.

### 2. Launch GUI
```powershell
python gnn_swarm_gui.py
```

### 3. Connect to AirSim
1. Click **"🔌 CONNECT TO AIRSIM"** button
2. Wait for connection confirmation
3. Model will load (uses random weights if no trained model found)

### 4. Configure Goal (Optional)
- **Goal X**: 100.0 (default) - X coordinate in meters
- **Goal Y**: 100.0 (default) - Y coordinate in meters
- Change these before starting mission

### 5. Start Mission
1. Click **"▶️ START MISSION"**
2. Watch the drones:
   - **Background drones (2-7)**: Move randomly
   - **Main drone (Drone1)**: Navigates to goal using GNN
3. Monitor real-time graphs and telemetry

### 6. Stop Mission
- Click **"⏹️ STOP MISSION"** to land all drones
- Or wait for automatic completion when goal is reached

---

## 📊 GUI Layout

### Left Panel - Control & Telemetry

#### Control Panel
- **Connect to AirSim**: Establish connection
- **Start Mission**: Begin drone navigation
- **Stop Mission**: Emergency stop and land all drones
- **Reset Simulation**: Reset AirSim environment

#### Goal Configuration
- Set target coordinates (X, Y) in meters
- Default: (100, 100)

#### Telemetry Display
Real-time data showing:
- **Main Drone Status**:
  - Position (X, Y, Z)
  - Velocity (vX, vY, vZ)
  - Current speed
  - Distance to goal

- **Collision Detection**:
  - Risk level (0-100%)
  - Nearest drone distance
  - Safety status (SAFE/WARNING)

- **Communication**:
  - Active communication links
  - Communication range (30m)
  - Total active drones

- **Mission Progress**:
  - Step count
  - Current status

### Right Panel - Real-Time Graphs

#### Distance to Goal (Top-Left)
- Green line shows distance over time
- Cyan dashed line shows goal zone (5m)
- Should decrease steadily as drone approaches

#### Collision Risk (Top-Right)
- Red line shows collision probability
- Green zone (0-30%): Safe
- Yellow zone (30-70%): Caution
- Red zone (70-100%): High risk

#### Speed (Bottom-Left)
- Blue line shows drone speed
- Shows adjustment based on obstacles

#### Active Communications (Bottom-Right)
- Purple line shows number of drones in range
- Communication range: 30 meters
- More drones = better swarm intelligence

---

## 🎯 How It Works

### GNN Architecture
```
Node Features (6D):
├── Goal relative X, Y (normalized)
├── Velocity X, Y (normalized)
├── Nearest obstacle distance
└── Nearest obstacle angle

GNN Layers:
├── Layer 1: Message passing between drones
├── Layer 2: Deep feature extraction
├── Collision Network: Predicts risk (0-1)
└── Action Network: Outputs velocities (-1 to 1)
```

### Collision Avoidance
1. **Detection**: Measures distance to all nearby drones
2. **Risk Calculation**: GNN predicts collision probability
3. **Avoidance**: Blends goal-seeking with collision avoidance
   - Low risk (0-30%): 80% goal-seeking, 20% GNN
   - High risk (70-100%): 50% goal-seeking, 50% GNN

### Communication Graph
- Drones within 30m can communicate
- Adjacency matrix connects nearby drones
- GNN uses graph structure for decisions

### Goal Navigation
- Main drone moves toward goal
- Adjusts path based on obstacles
- Reaches goal when within 5m

---

## 🎮 Usage Scenarios

### Standard Mission
1. Default goal (100, 100)
2. Click Connect → Start Mission
3. Watch drone navigate through swarm
4. Mission completes automatically

### Custom Goal
1. Enter desired coordinates
2. Example: X=50, Y=150
3. Start mission
4. Drone will navigate to new target

### Testing Collision Avoidance
1. Set goal near starting position initially
2. Watch collision risk spike when near drones
3. Observe avoidance maneuvers in real-time

### Reset After Mission
1. Click "Stop Mission" if not auto-stopped
2. Click "Reset Simulation"
3. Configure new goal
4. Start again

---

## 📈 Understanding the Telemetry

### Position Indicators
- **(0, 0, -20)**: Near start position, at altitude
- **(50, 50, -20)**: Halfway to default goal
- **(100, 100, -20)**: At default goal

### Velocity Indicators
- **Speed 0-2 m/s**: Slow, cautious (near obstacles)
- **Speed 3-5 m/s**: Normal navigation
- **Speed 6+ m/s**: Fast, clear path

### Collision Risk Levels
- **0-20%**: 🟢 Safe zone
- **20-50%**: 🟡 Monitor situation
- **50-70%**: 🟠 Caution advised
- **70-100%**: 🔴 Active avoidance

### Communication Count
- **0-2 drones**: Isolated, limited swarm intelligence
- **3-5 drones**: Good coverage
- **6+ drones**: Full swarm coordination

---

## 🔧 Troubleshooting

### GUI Won't Launch
```powershell
# Check dependencies
pip install torch numpy matplotlib airsim pillow

# Run
python gnn_swarm_gui.py
```

### Connection Fails
- ✅ Is AirSim/Unreal running?
- ✅ Check settings file copied to:
  `C:\Users\[YOUR_USER]\Documents\AirSim\settings.json`
- ✅ Try reset: Close AirSim, restart, reconnect

### Drones Not Moving
- Check console for errors
- Reset simulation
- Verify all 10 drones in settings.json

### Collision Risk Always High
- Normal if many drones nearby
- Watch for actual avoidance behavior
- Risk should decrease when path clears

### Goal Not Reached
- Check goal coordinates are reasonable (0-200 range)
- Ensure sufficient time (complex paths take longer)
- Monitor distance graph - should decrease

---

## 🎓 Advanced Features

### Model Training (Future)
To train the GNN model:
1. Collect flight data
2. Train collision prediction network
3. Save as `gnn_collision_model.pth`
4. Model will auto-load on next run

### Adjusting Parameters
Edit `gnn_swarm_gui.py`:
- `COMM_RANGE = 30.0` - Communication distance
- `COLLISION_THRESHOLD = 5.0` - Minimum safe distance
- `VELOCITY_SCALE = 6.0` - Maximum speed
- `ALTITUDE = -20.0` - Flight altitude

### Adding More Background Drones
In `start_swarm()` method:
```python
for drone in BACKGROUND_DRONES[:9]:  # Use 9 instead of 6
```

---

## 📝 Comparison: Simple vs GUI Version

| Feature | simple_gnn_swarm.py | gnn_swarm_gui.py |
|---------|---------------------|------------------|
| GUI | ❌ Terminal only | ✅ Full GUI |
| Graphs | ❌ None | ✅ 4 real-time plots |
| Collision Detection | ❌ Basic | ✅ AI-predicted |
| Telemetry | ⚠️ Basic text | ✅ Formatted display |
| Control | ⚠️ Code only | ✅ Buttons |
| Goal Setting | ⚠️ Edit code | ✅ GUI input |
| Status Feedback | ⚠️ Console | ✅ Visual indicators |
| Lines of Code | 270 | 800+ |

---

## 🎉 What You'll See

### In AirSim Window
- 7 drones flying (1 main + 6 background)
- Background drones moving randomly
- Main drone (Drone1) navigating to goal
- Smooth avoidance maneuvers when near others

### In GUI Window
- Real-time position updates
- Graphs updating every 0.5 seconds
- Collision risk changing with proximity
- Communication count varying as drones move
- Distance decreasing toward goal
- Status indicators showing mission progress

### When Goal Reached
- Popup: "🎉 Goal Reached in X steps!"
- Mission stops automatically
- All drones land
- Ready for next mission

---

## 🚀 Quick Commands Reference

```powershell
# Start GUI
python gnn_swarm_gui.py

# Check if AirSim ready
python test_gnn_ready.py

# Reset AirSim from terminal
python -c "import airsim; c = airsim.MultirotorClient(); c.confirmConnection(); c.reset(); print('Reset complete')"

# Run simple version (no GUI)
python simple_gnn_swarm.py
```

---

## 💡 Tips for Best Results

1. **Start with default goal** (100, 100) for first test
2. **Watch collision graph** - should spike near obstacles
3. **Monitor communication count** - shows swarm coordination
4. **Let mission complete** - don't stop early to see full behavior
5. **Reset between runs** - ensures clean starting state
6. **Adjust goal gradually** - test with 50, then 100, then 150
7. **Check distance graph** - should show smooth decrease
8. **Note step count** - typical runs: 50-150 steps depending on goal

---

## 📞 Support

**System Requirements:**
- Windows 10/11
- Python 3.8+
- AirSim/Unreal Engine
- 8GB+ RAM recommended

**File Locations:**
- Main GUI: `gnn_swarm_gui.py`
- Simple version: `simple_gnn_swarm.py`
- Settings: `airsim_settings_dual_drone.json`
- Model (optional): `gnn_collision_model.pth`

**Next Steps:**
1. Test with default settings
2. Try custom goals
3. Monitor collision avoidance
4. Experiment with different scenarios

---

**Enjoy your GNN swarm system! 🚁✨**
