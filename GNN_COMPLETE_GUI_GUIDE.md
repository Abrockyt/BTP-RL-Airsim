# 🚁 GNN SWARM COMPLETE GUI - USER GUIDE

## Overview

The **GNN Swarm Complete GUI** provides full-featured multi-drone navigation with:
- **Graph Neural Network (GNN)** for swarm intelligence with collision avoidance
- **Visual depth sensing** with RGB camera simulation
- **Wind simulation** with pressure sensor feedback
- **Flight mode switching** (NORMAL/WIND)
- **Algorithm comparison** (GNN vs MHA-PPO)
- **Reinforcement Learning** progress tracking
- **Real-time analytics** with 6-panel dashboard
- **Drone communication network** visualization
- **Dynamic mode** with interceptor drones
- **Interactive flight map** with click-to-set-goal

---

## 🎨 GUI Layout

### Left Panel (Control Center)

#### 1. **👁️ DEPTH VISION**
- **420x250 canvas** displays colorized depth information
- Shows obstacles in **left**, **center**, and **right** regions
- Color coding: 
  - 🔴 RED = Close obstacles (< 5m)
  - 🟡 YELLOW = Medium distance (5-15m)
  - 🟢 GREEN = Far/clear (> 15m)
- Real-time distance indicators below canvas

#### 2. **💨 PRESSURE SENSOR PANEL**
- **Air pressure** reading (Pa)
- **Wind magnitude** (m/s)
- **Wind status**: NORMAL ✓ / HEAVY WIND ⚠️
- **Vision status**: ACTIVE 📷 / FAILED ❌

#### 3. **🌪️ WIND PATTERN VISUALIZATION**
- **380x150 canvas** with live wind arrows
- Arrow direction = wind direction
- Arrow thickness = wind intensity
- Color coding:
  - 🟢 GREEN = Calm (< 5 m/s)
  - 🟡 YELLOW = Moderate (5-10 m/s)
  - 🔴 RED = Strong (> 10 m/s)
- Compass direction display (N, NE, E, SE, S, SW, W, NW)

#### 4. **🎮 FLIGHT MODE BUTTONS**

**NORMAL MODE (📷 Vision + GNN)**
- Uses depth vision for obstacle detection
- GNN-based navigation with collision avoidance
- Best for clear weather conditions
- Button: GREEN when active

**WIND MODE (💨 Pressure Sensor)**
- Uses barometer for wind detection
- Pressure-based navigation
- Adapts to heavy wind conditions
- Button: RED when active

#### 5. **🎯 GOAL POSITION CONTROL**
Two ways to set goal:
1. **Manual entry**: Type X, Y coordinates
2. **Map click**: Click on flight map to set goal

Current goal displayed in green text

#### 6. **🔋 BATTERY DISPLAY**
- Large percentage display
- Color coding:
  - 🟢 GREEN > 50%
  - 🟡 ORANGE 20-50%
  - 🔴 RED < 20%
- Energy consumed (Wh)

#### 7. **📊 TELEMETRY**
Real-time data:
- Position (X, Y)
- Speed (m/s)
- Altitude (m AGL)
- Goal distance (m)
- Collision risk (0-100%)
- Communications count

#### 8. **🧠 REINFORCEMENT LEARNING PANEL**
- **Run number**: Current flight attempt
- **Best energy**: Lowest energy consumption achieved
- **Best time**: Fastest completion time
- **Improvement**: % change from previous run
- **Toggle**: Enable/disable online learning

#### 9. **⚡ ADAPTIVE FLIGHT (Dynamic Mode)**
Toggle interceptor drones:
- ✅ **ON**: Drone3, Drone4, Drone5 act as moving obstacles
- ❌ **OFF**: Standard flight, no interceptors
- Creates dynamic challenge for testing

#### 10. **CONTROL BUTTONS**

**🚀 START FLIGHT** (Green)
- Initialize AirSim connection
- Load GNN model
- Takeoff main drone
- Start background drones
- Begin navigation

**⏹ STOP FLIGHT** (Red)
- Stop navigation
- Land all drones
- Save flight data
- Update RL metrics

**🏆 ALGORITHM COMPARISON** (Blue)
- Launch dual-drone race
- Drone1 uses GNN navigation
- Drone2 uses MHA-PPO navigation
- Both fly to same goal
- Compare performance

---

### Right Panel (Performance Analytics)

#### **📈 6-PANEL DASHBOARD**

**1. SPEED GRAPH**
- Real-time speed (m/s)
- Target cruise speed shown as dashed line
- Blue line

**2. ALTITUDE GRAPH**
- Altitude above ground (m)
- 20m AGL target shown
- Orange line

**3. BATTERY GRAPH**
- Battery percentage over time
- Color changes with level
- Critical threshold shown

**4. ENERGY GRAPH**
- Total energy consumed (Wh)
- Pink line
- Compare with best run if available

**5-6. FLIGHT MAP** (Large panel)
- **Satellite-style terrain** background
- **Start position**: Green circle (0, 0)
- **Goal position**: Red star ⭐
- **Current position**: Blue circle (drone)
- **Flight path**: Cyan trail showing route
- **Interceptors**: Triangle markers (if dynamic mode)
- **Grid**: Green dashed lines
- **Click map** to set new goal

---

### Bottom Panel (Communication Network)

#### **📡 DRONE COMMUNICATIONS NETWORK**

**Visual representation:**
- Drones arranged in **circular layout**
- **Arrow thickness** = number of messages
- **Arrow transparency** = message frequency
- Color coding:
  - 🟠 **ORANGE**: Main drone (Drone1)
  - 🟢 **GREEN**: Comparison drone (Drone2)
  - 🔵 **BLUE**: Background/interceptor drones

**Statistics shown:**
- Number of active communication links
- Total messages in last 10 seconds
- Updates every 1 second

---

## 🚀 How to Use

### Basic Flight

1. **Launch GUI**: `python gnn_swarm_complete_gui.py`

2. **Set goal**:
   - Enter coordinates manually OR
   - Click on flight map

3. **Click "🚀 START FLIGHT"**:
   - Wait for "Initializing..." status
   - Drone will takeoff and navigate
   - Watch real-time telemetry and graphs

4. **Monitor progress**:
   - Check distance to goal
   - Watch collision risk
   - View communication network
   - Observe energy consumption

5. **Goal reached**:
   - Success message appears
   - Metrics saved for RL
   - Flight history updated

6. **Click "⏹ STOP FLIGHT"** to end

---

### Advanced Features

#### **Flight Mode Switching**

**When to use NORMAL MODE:**
- Good weather conditions
- Vision sensors working
- Standard navigation needed

**When to use WIND MODE:**
- Heavy wind detected
- Vision sensors failed
- Pressure-based navigation preferred

Click buttons to switch modes during flight.

---

#### **Dynamic Mode (Interceptors)**

**Purpose:** Test navigation in dynamic environment

**How it works:**
1. Enable "Dynamic Mode" checkbox
2. Start flight
3. Drone3, Drone4, Drone5 will:
   - Follow main drone
   - Create moving obstacles
   - Test collision avoidance

**Use cases:**
- Training collision avoidance
- Testing adaptive behaviors
- Evaluating swarm intelligence

---

#### **Reinforcement Learning**

**Automatic tracking:**
- Each flight = 1 run
- System tracks best energy & time
- Shows improvement percentage
- Learns optimal paths over runs

**Toggle learning:**
- ✅ **ON**: System learns and improves
- ❌ **OFF**: Static behavior, no learning

**Metrics tracked:**
- Run number
- Energy consumed
- Flight time
- Goal reach success
- Collision avoidance rate

---

#### **Algorithm Comparison Mode**

**Compare GNN vs MHA-PPO:**

1. Click **"🏆 ALGORITHM COMPARISON"**
2. Two drones launch:
   - **Drone1 (Orange)**: GNN navigation
   - **Drone2 (Green)**: MHA-PPO navigation
3. Both fly to same goal
4. System tracks:
   - Which arrives first
   - Energy efficiency
   - Path smoothness
   - Collision avoidance
5. Results displayed after completion

**Use cases:**
- Research comparison
- Algorithm benchmarking
- Performance validation

---

## 📊 Understanding the Metrics

### Collision Risk (0-100%)
- **0-30% (Green)**: Safe, clear path
- **30-70% (Orange)**: Moderate risk, caution
- **70-100% (Red)**: High risk, evasive action

### Communications
- Shows number of drones within 30m range
- Higher = better swarm coordination
- Essential for GNN message passing

### Battery & Energy
- **Battery**: Percentage remaining
- **Energy**: Wh consumed
- System calculates:
  ```
  Power = P_hover + speed² × 10
  Energy step = Power × time / 3600
  ```

### Distance to Goal
- Direct line distance (m)
- Updates in real-time
- Flight ends when < 5m

---

## 🎯 Tips & Best Practices

### For Optimal Performance:

1. **Set realistic goals**: 50-150m range works best

2. **Watch collision risk**: 
   - High risk = GNN takes control
   - Low risk = direct path to goal

3. **Monitor battery**: 
   - Long flights drain battery
   - Energy tracking improves with RL

4. **Use dynamic mode**: 
   - Tests robustness
   - Improves collision avoidance

5. **Run multiple flights**: 
   - RL improves over runs
   - Best metrics saved automatically

6. **Check communication network**: 
   - Verify swarm connectivity
   - Ensure messages flowing

---

## 🔧 Configuration

### Key Parameters (in code):

```python
DEFAULT_GOAL_X = 100.0          # Default goal X
DEFAULT_GOAL_Y = 100.0          # Default goal Y
ALTITUDE = -20.0                # Flight altitude (m)
COMM_RANGE = 30.0               # Communication range (m)
COLLISION_THRESHOLD = 5.0       # Collision warning (m)
VELOCITY_SCALE = 6.0            # Max velocity (m/s)
P_HOVER = 200.0                 # Hover power (W)
BATTERY_CAPACITY_WH = 100.0     # Battery capacity (Wh)
SETUP_TIMEOUT_SECONDS = 60.0    # Max setup time
```

### Background Drones:
- Drone2-Drone10 available
- 6 used by default for background movement
- Random waypoint navigation

### Interceptor Drones (Dynamic Mode):
- Drone3, Drone4, Drone5
- Follow main drone with offset
- Create dynamic obstacles

---

## 🐛 Troubleshooting

### GUI doesn't launch
**Solution**: Check Python environment
```powershell
python --version  # Should be 3.8+
pip list | Select-String "torch|airsim|matplotlib"
```

### "Setup Timeout" error
**Causes:**
- AirSim not running
- 10-drone configuration not loaded
- Network connection issues

**Solution:**
1. Start AirSim
2. Load environment with 10+ drones
3. Wait for full initialization
4. Retry

### Drone doesn't move
**Check:**
- ✅ AirSim connected
- ✅ Correct `vehicle_name` (Drone1)
- ✅ API control enabled
- ✅ Arm/disarm successful

### Graphs not updating
**Solution:**
- Check if flight active
- Verify metrics collecting
- Look for threading errors in console

### Vision display blank/black
**Normal**: Simulated depth image
- Updates every 0.2s
- Shows obstacle regions
- Colorized based on distance

### Wind pattern not showing
**Check:**
- Wind enabled in `airsim_settings.json`
- Wind values: `"X": 2, "Y": 1`
- Magnitude calculated from settings

### Communication chart empty
**Expected** if:
- No drones within 30m range
- Background drones not active
- Flight not started

**Solution:**
- Start flight
- Wait for background drones to approach
- Check communication count in telemetry

---

## 📈 Features from smart_drone_vision_gui.py

### ✅ Fully Integrated:

- ✅ **Depth Vision Display** (420x250 canvas, colorized)
- ✅ **Pressure Sensor Panel** (pressure, wind magnitude, status)
- ✅ **Wind Pattern Visualization** (arrows, compass, intensity)
- ✅ **Flight Mode Buttons** (NORMAL/WIND switching)
- ✅ **Goal Position Control** (manual + map click)
- ✅ **Battery Display** (large %, energy Wh)
- ✅ **Telemetry Display** (pos, speed, alt, distance, risk, comm)
- ✅ **RL Statistics Panel** (run #, best metrics, improvement)
- ✅ **Dynamic Mode Toggle** (interceptor drones)
- ✅ **6-Panel Graphs** (speed, altitude, battery, energy, map)
- ✅ **Flight Map** (terrain, start, goal, path, drone, interceptors)
- ✅ **Map Click Handling** (set goal by clicking)
- ✅ **Communication Chart** (bottom panel, circle layout, arrows)
- ✅ **Scrollable Left Panel** (fits all controls)
- ✅ **Energy Calculations** (P_hover + speed² formula)
- ✅ **Comparison Mode** (GNN vs MHA-PPO)
- ✅ **Online Learning** (run history tracking)

### 🆕 Enhanced with GNN:

- 🆕 **CollisionGNN_Actor**: 2-layer GNN with collision detection
- 🆕 **Collision Risk Display**: Real-time 0-100% risk meter
- 🆕 **Swarm Communication**: GNN message passing visualization
- 🆕 **Node Features**: [goal_rel, velocity, nearest_obs_dist, angle]
- 🆕 **Graph Adjacency**: Dynamic connectivity based on 30m range
- 🆕 **Collision Prediction Network**: Outputs risk probability
- 🆕 **Blended Control**: GNN + goal-seeking hybrid navigation

---

## 🎓 System Architecture

### GNN Collision Avoidance

**How it works:**

1. **Node Features** (each drone):
   ```
   [goal_rel_x, goal_rel_y, vel_x, vel_y, nearest_obs_dist, nearest_obs_angle]
   ```

2. **Adjacency Matrix**:
   - 1 if within COMM_RANGE (30m)
   - 0 if out of range
   - Enables message passing

3. **GNN Layers**:
   - Layer 1: Message generation & aggregation
   - Layer 2: Node update & refinement

4. **Outputs**:
   - **Actions**: [vx, vy] velocity commands
   - **Collision Risk**: Probability 0-1

5. **Blended Control**:
   ```python
   gnn_weight = 0.2 + collision_risk * 0.3
   goal_weight = 1.0 - gnn_weight
   
   final_velocity = goal_direction * goal_weight + gnn_action * gnn_weight
   ```

### Comparison: GNN vs MHA-PPO

| Feature | GNN | MHA-PPO |
|---------|-----|---------|
| **Architecture** | Graph Neural Network | Multi-Head Attention |
| **Input** | Swarm state graph | Single drone state |
| **Strength** | Swarm coordination | Individual optimization |
| **Collision** | Graph-based detection | Sensor-based avoidance |
| **Communication** | Built-in message passing | External coordination |
| **Scalability** | Excellent (O(n)) | Good (O(n²)) |

---

## 📝 File Structure

```
gnn_swarm_complete_gui.py         # Main GUI (2000+ lines)
├── Imports & Configuration
├── GNN_Layer class
├── CollisionGNN_Actor class
├── MHA_Actor class (for comparison)
├── GNNSwarmCompleteGUI class
│   ├── __init__
│   ├── create_widgets
│   │   ├── Header
│   │   ├── Left Panel (scrollable)
│   │   │   ├── Vision Display
│   │   │   ├── Pressure Sensor Panel
│   │   │   ├── Wind Pattern Canvas
│   │   │   ├── Flight Mode Buttons
│   │   │   ├── Goal Control
│   │   │   ├── Battery Display
│   │   │   ├── Telemetry
│   │   │   ├── RL Statistics
│   │   │   ├── Dynamic Mode Toggle
│   │   │   └── Control Buttons
│   │   ├── Right Panel
│   │   │   └── 6-Panel Graph Dashboard
│   │   └── Bottom Panel
│   │       └── Communication Chart
│   ├── Flight Control Methods
│   │   ├── start_flight
│   │   ├── stop_flight
│   │   ├── _flight_loop
│   │   ├── _comparison_flight_loop
│   │   └── open_comparison_window
│   ├── Background Drones
│   │   ├── _start_background_drones
│   │   ├── _background_drone_thread
│   │   ├── _start_interceptor_drones
│   │   └── _interceptor_drone_thread
│   ├── Update Loops
│   │   ├── _update_telemetry_loop
│   │   ├── _update_vision_loop
│   │   ├── _update_graphs_loop
│   │   └── _update_comm_chart_loop
│   ├── Visualization Methods
│   │   ├── update_vision_display
│   │   ├── draw_wind_pattern
│   │   ├── update_graphs
│   │   ├── _draw_communication_network
│   │   ├── init_graphs
│   │   └── redraw_flight_map
│   ├── Mode Switching
│   │   ├── set_normal_mode
│   │   ├── set_wind_mode
│   │   ├── toggle_learning
│   │   └── toggle_dynamic_mode
│   ├── Goal Control
│   │   ├── update_goal
│   │   └── on_map_click
│   └── Utilities
│       ├── _setup_timeout_reached
│       ├── update_rl_display
│       └── _generate_simulated_depth_image
└── Main Execution
```

---

## 🎯 Research Applications

### Multi-Drone Coordination
- Test swarm intelligence algorithms
- Validate GNN architectures
- Compare with state-of-the-art methods

### Collision Avoidance
- Train collision detection models
- Evaluate safety metrics
- Test in dynamic environments

### Reinforcement Learning
- Track learning progress
- Optimize reward functions
- Compare online vs offline learning

### Energy Optimization
- Measure battery consumption
- Test energy-efficient paths
- Optimize flight parameters

### Vision-Based Navigation
- Depth sensing integration
- Obstacle detection algorithms
- Sensor fusion techniques

---

## 🔬 Advanced Use Cases

### 1. **Algorithm Research**
Compare multiple navigation algorithms:
- GNN (Graph Neural Network)
- MHA-PPO (Multi-Head Attention PPO)
- Custom algorithms (extensible)

### 2. **Safety Testing**
Test collision avoidance in scenarios:
- Static obstacles
- Dynamic obstacles (interceptors)
- Heavy wind conditions
- Mix of all above

### 3. **Multi-Agent Coordination**
Study swarm behaviors:
- Communication patterns
- Emergent behaviors
- Scalability testing

### 4. **Path Planning**
Optimize trajectories:
- Energy-efficient paths
- Time-optimal routes
- Safety-first navigation

### 5. **Sensor Integration**
Test sensor fusion:
- Depth camera + barometer
- Vision failure handling
- Multi-modal navigation

---

## 📚 References

**Based on:**
- `smart_drone_vision_gui.py` - Original vision-based GUI
- GNN architectures for swarm robotics
- Multi-Head Attention for RL
- AirSim multi-drone simulation

**Key Concepts:**
- Graph Neural Networks (GNNs)
- Reinforcement Learning (RL)
- Multi-Agent Systems (MAS)
- Collision Avoidance
- Energy-Aware Navigation

---

## 💡 Quick Start Summary

1. **Launch**: `python gnn_swarm_complete_gui.py`
2. **Set Goal**: Click map or enter coordinates
3. **Choose Mode**: Normal (vision) or Wind (pressure)
4. **Enable Dynamic**: Toggle for interceptors
5. **Start Flight**: Click green button
6. **Monitor**: Watch graphs, telemetry, communication
7. **Compare**: Use comparison mode for GNN vs MHA-PPO
8. **Iterate**: Multiple runs improve RL performance

---

## ✨ Key Highlights

🎯 **Complete Feature Parity** with smart_drone_vision_gui.py
🧠 **GNN-Based Swarm Intelligence** with collision avoidance
📊 **6-Panel Real-Time Analytics** dashboard
📡 **Communication Network** visualization
⚡ **Dynamic Mode** with interceptor drones
🧪 **Algorithm Comparison** (GNN vs MHA-PPO)
🎮 **Interactive Map** with click-to-set-goal
💨 **Wind Simulation** with pressure sensors
👁️ **Depth Vision** display with colorization
🔋 **Energy Tracking** with battery simulation
🧠 **Reinforcement Learning** progress tracking
🎨 **Professional UI** with scrollable panels

---

**Enjoy flying! 🚁✨**
