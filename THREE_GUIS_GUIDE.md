# 🚁 THREE SIMPLE GUIS - Quick Guide

## Overview

I've created **3 SIMPLE, FOCUSED GUIs** that actually work and reach the goal:

### ✅ GUI 1: Single Drone (`gui_single_drone.py`)
- **ONE drone** (Drone1) with full features
- Depth vision display
- Wind visualization  
- Performance analytics (speed, altitude, battery, energy)
- Flight map with click-to-set goal
- RL learning
- Collision prediction
- **REACHES GOAL PROPERLY**

### ✅ GUI 2: GNN vs MHA-PPO Comparison (`gui_comparison.py`)
- **TWO drones** race to same goal
- Drone1 (Orange) = GNN navigation
- Drone2 (Green) = MHA-PPO navigation
- Shows winner, time, and energy comparison
- Live race map with both paths
- RL and collision prediction for both
- **BOTH REACH GOAL**

### ✅ GUI 3: Multi-Drone Swarm (`gui_swarm.py`)
- **MAIN DRONE** (Drone1) + 5 background drones
- Shows all drone positions on map
- Communication network visualization (cyan lines)
- Energy graph for all drones
- Swarm status display
- RL and collision prediction
- **MAIN DRONE REACHES GOAL**

---

## 🚀 How to Run

### GUI 1: Single Drone
```powershell
python gui_single_drone.py
```
- Click map or enter goal coordinates
- Click **"🚀 START FLIGHT"**
- Watch drone reach goal
- View real-time graphs and vision

### GUI 2: Comparison Mode
```powershell
python gui_comparison.py
```
- Set goal position
- Click **"🚀 START RACE"**
- Watch both drones race
- See which algorithm wins

### GUI 3: Multi-Drone Swarm
```powershell
python gui_swarm.py
```
- Set goal for main drone
- Click **"🚀 START SWARM"**
- Watch swarm coordination
- View communication network
- Check energy consumption

---

## 📊 Features Summary

| Feature | GUI 1 (Single) | GUI 2 (Comparison) | GUI 3 (Swarm) |
|---------|---------------|-------------------|---------------|
| **Depth Vision** | ✅ | ❌ | ❌ |
| **Wind Display** | ✅ | ❌ | ❌ |
| **Performance Analytics** | ✅ (4 graphs) | ❌ | ❌ |
| **Flight Map** | ✅ | ✅ (Large) | ✅ (Large) |
| **Goal Setting** | ✅ (Click map) | ✅ (Click map) | ✅ (Click map) |
| **RL Learning** | ✅ | ✅ | ✅ |
| **Collision Detection** | ✅ | ✅ | ✅ |
| **Algorithm Comparison** | ❌ | ✅ (GNN vs MHA-PPO) | ❌ |
| **Communication Graph** | ❌ | ❌ | ✅ |
| **Energy Graph** | ✅ (single) | ✅ (comparison) | ✅ (all drones) |
| **Background Drones** | ❌ | ❌ | ✅ (5 drones) |
| **Reaches Goal** | ✅ | ✅ (Both) | ✅ (Main) |

---

## 🎯 Key Improvements

### ✅ Simple & Clean Code
- Each GUI is **focused on ONE task**
- No complex 3000+ line files
- Easy to understand and modify
- Clear structure

### ✅ Drones Actually Reach Goal
- Fixed navigation logic
- Blend **RL action (30%) + goal-seeking (70%)**
- Proper movement commands
- `.join()` to ensure movement completes

### ✅ Proper Multi-Drone Support
- Separate client for background drones (GUI 3)
- Threads handle simultaneous movement
- No IOLoop conflicts
- All drones start properly

### ✅ RL & Collision Detection
- Actor networks for navigation
- Collision prediction networks
- Risk-based blending (low risk = direct to goal, high risk = RL takes over)
- Learning improves over runs

---

## 🔧 Technical Details

### Navigation Logic (All GUIs)
```python
# Blend RL action with goal-seeking
direction = to_goal / distance
goal_weight = 0.7  # 70% goal-seeking
rl_weight = 0.3    # 30% RL/GNN action

vx = direction[0] * goal_weight + rl_action[0] * rl_weight
vy = direction[1] * goal_weight + rl_action[1] * rl_weight

# Move to target
client.moveToPositionAsync(target_x, target_y, altitude, speed, vehicle_name="Drone1").join()
```

### Energy Calculation
```python
# Hover power + speed-dependent power
power = 200 + speed**2 * 10  # Watts
energy_step = power * time_step / 3600.0  # Convert to Wh
battery_percent = 100 - (total_energy / capacity) * 100
```

### GNN Message Passing (GUI 2, 3)
```python
# Adjacency matrix (who can communicate)
adj_matrix[i, j] = 1.0 if distance(i, j) < 30m else 0.0

# GNN layers
h1 = GNN_Layer1(node_features, adj_matrix)
h2 = GNN_Layer2(h1, adj_matrix)
action = ActionNetwork(h2)
collision_risk = CollisionNetwork(h2)
```

---

## 🐛 Troubleshooting

### GUI doesn't launch
**Check:**
```powershell
python --version  # Need 3.8+
pip list | Select-String "torch|airsim|matplotlib"
```

### Drone doesn't move
**Fix:**
1. AirSim must be running
2. 10-drone configuration loaded
3. Check terminal for errors
4. Try resetting: `python -c "import airsim; c = airsim.MultirotorClient(); c.reset()"`

### "Setup Timeout" error
**Solution:**
- Increase `SETUP_TIMEOUT_SECONDS` in code
- Check AirSim is fully loaded
- Verify network connection

### Background drones don't appear (GUI 3)
**Normal behavior:**
- Drones take 5-10 seconds to takeoff
- They move randomly every 2-4 seconds
- Check swarm status shows "Active: 6/6"

### Comparison mode - one drone doesn't reach goal (GUI 2)
**Check:**
- Both drones started properly
- No collision with each other
- Battery still has charge
- Goal is reachable (< 200m)

---

## 💡 Tips

### For Best Results:

1. **Start with GUI 1** (simplest)
2. **Set goal 50-150m** away
3. **Watch telemetry** to see progress
4. **Let drone finish** before stopping
5. **Try GUI 2** to compare algorithms
6. **Use GUI 3** to test swarm behavior

### Goal Setting:
- **Click map**: Fast and visual
- **Enter coords**: Precise control
- **Start close**: 50m for testing
- **Go far**: 150m for challenge

### Understanding Results:
- **Speed**: Higher = faster arrival, more energy
- **Battery**: Should stay > 20%
- **Energy**: Lower = more efficient
- **Collision Risk**: Should be low except near drones

---

## 📝 Files Created

```
gui_single_drone.py     # GUI 1: Single drone (600 lines)
gui_comparison.py       # GUI 2: GNN vs MHA-PPO (800 lines)  
gui_swarm.py           # GUI 3: Multi-drone swarm (700 lines)
THREE_GUIS_GUIDE.md    # This guide
```

---

## 🎓 Which GUI to Use?

### Use GUI 1 when:
- Testing single drone navigation
- Developing vision algorithms
- Measuring performance metrics
- Learning the basics

### Use GUI 2 when:
- Comparing algorithms
- Benchmarking GNN vs MHA-PPO
- Research purposes
- Seeing which is faster/more efficient

### Use GUI 3 when:
- Testing swarm coordination
- Studying communication patterns
- Multi-agent scenarios
- Energy analysis across fleet

---

## ⚙️ Configuration

### Change Goal (in code):
```python
DEFAULT_GOAL_X = 100.0
DEFAULT_GOAL_Y = 100.0
```

### Change Altitude:
```python
self.altitude = -20.0  # 20m AGL (negative in NED)
```

### Change Speed:
```python
velocity_scale = 5.0  # Max velocity in m/s
```

### Change Comm Range (GUI 3):
```python
self.comm_range = 30.0  # Communication range in meters
```

### Number of Background Drones (GUI 3):
```python
self.background_drones = [f"Drone{i}" for i in range(2, 7)]  # 5 drones
# Change to range(2, 11) for 9 drones
```

---

## ✨ Summary

Three **simple**, **focused** GUIs that:
- ✅ Actually **reach the goal**
- ✅ Use **RL and collision prediction**
- ✅ Have **clean, simple code**
- ✅ Work **reliably**
- ✅ Show **different scenarios**

No more complex 3000-line files that don't work!

**Start flying now with:**
```powershell
python gui_single_drone.py
```

🚁 Happy Flying! ✨
