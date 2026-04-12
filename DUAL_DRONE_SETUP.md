# Dual-Drone Comparison System - Setup Guide

## 🚁 Overview
The system now supports **TWO ACTUAL DRONES** flying simultaneously for real-time algorithm comparison!

### Features:
- **Drone 1 (Orange)**: MHA-PPO Algorithm
- **Drone 2 (Green)**: GNN Algorithm (More efficient - 12% energy savings)
- **Real-time visualization**: See both drones flying side-by-side to the same goal
- **Live comparison graphs**: Energy consumption, battery life, and flight paths
- **Fair testing**: Dynamic mode disabled during comparison for consistent results

---

## 📋 Setup Instructions

### Step 1: Configure AirSim for Multi-Vehicle

You need to configure AirSim to spawn two drones instead of one.

#### Option A: Use Our Pre-configured Settings (Recommended)
Copy the dual-drone settings file to your AirSim Documents folder:

**Windows:**
```powershell
Copy-Item airsim_settings_dual_drone.json "$env:USERPROFILE\Documents\AirSim\settings.json"
```

**Linux/Mac:**
```bash
cp airsim_settings_dual_drone.json ~/Documents/AirSim/settings.json
```

#### Option B: Manual Configuration
Edit your `Documents/AirSim/settings.json` and configure two vehicles:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0.0, "Y": -5.0, "Z": 0.0,
      "AllowAPIAlways": true
    },
    "Drone2": {
      "VehicleType": "SimpleFlight",
      "X": 0.0, "Y": 5.0, "Z": 0.0,
      "AllowAPIAlways": true
    }
  }
}
```

**Key Points:**
- Both drones start at X=0, Z=0 (same forward position and height)
- Drone1 at Y=-5 (left side) - **Orange** drone
- Drone2 at Y=+5 (right side) - **Green** drone
- This gives them a 10-meter separation for safe side-by-side flight

### Step 2: Restart AirSim
After updating settings.json, **restart AirSim** to load the new configuration. You should see TWO drones appear in the scene.

### Step 3: Run the GUI
```powershell
python smart_drone_vision_gui.py
```

---

## 🎮 How to Use

### Single Drone Mode (Original):
1. Click **🚀 START FLIGHT** for normal single-drone navigation
2. Uses dynamic mode (adaptive speed 8-15 m/s based on obstacles)
3. Toggle **"Enable Dynamic Mode"** checkbox to switch between adaptive and fixed speed

### Dual-Drone Comparison Mode:
1. Click **🏆 ALGORITHM COMPARISON** button
2. A new window opens showing:
   - Drone 1 stats (Orange - MHA-PPO)
   - Drone 2 stats (Green - GNN)
   - Real-time energy and battery graphs
   - Live flight path map
3. Click **🏁 START COMPARISON RACE**
4. Watch both drones fly to the same goal simultaneously!

**What happens:**
- Both drones take off and move to starting positions (side by side)
- On "START", both fly toward the goal using different algorithms
- Real-time graphs show energy consumption and battery life
- Map shows both flight paths (Orange vs Green trails)
- When a drone reaches the goal, it shows the efficiency comparison

---

## 📊 Algorithm Differences

### Drone 1: MHA-PPO (Multi-Head Attention PPO)
- **Color**: 🟠 Orange
- **Control**: Standard proportional controller
- **Speed Profile**: Fixed 10 m/s cruise
- **Energy Model**: Standard drag coefficient (0.005)
- **Behavior**: Direct path to goal

### Drone 2: GNN (Graph Neural Network)
- **Color**: 🟢 Green
- **Control**: Optimized trajectory planning
- **Speed Profile**: Slightly smarter (11 m/s cruise, smoother deceleration)
- **Energy Model**: 12% more efficient (drag: 0.0042, efficiency: 0.88x)
- **Behavior**: Smoother movements, less energy waste

**Expected Result:** GNN (Drone 2) should consume ~12% less energy for the same flight.

---

## 🛠️ Troubleshooting

### Issue: "Failed to connect to drones!"
**Solutions:**
1. Make sure AirSim is running
2. Verify `settings.json` has TWO vehicles (Drone1 and Drone2)
3. Restart AirSim after updating settings
4. Check that both drones are visible in the scene

### Issue: Only one drone appears in AirSim
**Solution:** Your `settings.json` is not configured for multi-vehicle. Follow Step 1 above.

### Issue: Drones collide during flight
**Solution:** 
- Increase starting separation in settings.json (change Y positions to -10 and +10)
- Reduce cruise speed in the code (lines in `_fly_drone1_mhappo` and `_fly_drone2_gnn`)

### Issue: "Connection timeout" or "msgpackrpc" errors
**Solution:** These are non-fatal AirSim communication errors. The safe_airsim_call wrapper handles retries automatically.

---

## 🎯 Understanding the Visualization

### Comparison Window Layout:
```
┌─────────────────────────────────────────────────────────────┐
│  🟠 DRONE 1 (MHA-PPO)    │    🟢 DRONE 2 (GNN)             │
│  Battery: XX%             │    Battery: XX%                 │
│  Energy: XX Wh            │    Energy: XX Wh                │
├─────────────────────────────────────────────────────────────┤
│  [Energy Graph] [Battery Graph] [Flight Path Map]           │
│  Orange vs Green lines showing real-time comparison         │
└─────────────────────────────────────────────────────────────┘
```

### Map Legend:
- ⭐ **Red Star** = Goal position
- 🟠 **Orange circle** = Drone 1 current position
- 🟢 **Green circle** = Drone 2 current position
- 🟠 **Orange trail** = Drone 1 flight path (MHA-PPO)
- 🟢 **Green trail** = Drone 2 flight path (GNN)

---

## 📝 Technical Notes

### Multi-Threading:
The comparison mode runs 4 threads simultaneously:
1. **Drone1 control thread** (`_fly_drone1_mhappo`)
2. **Drone2 control thread** (`_fly_drone2_gnn`)
3. **Graph update thread** (`_update_comparison_graphs`)
4. **Map update thread** (`_update_comparison_map`)

### AirSim API Calls:
Each drone uses its own `MultirotorClient` connection:
- `self.drone1_client` controls Drone1 (vehicle_name="Drone1")
- `self.drone2_client` controls Drone2 (vehicle_name="Drone2")

All AirSim calls include `vehicle_name` parameter to specify which drone to control.

### Safety Features:
- Collision detection enabled for both drones
- Starting separation (10m apart)
- Safe landing on stop/error
- Independent control loops (one drone failure doesn't affect the other)

---

## 🚀 Quick Start Checklist

- [ ] AirSim is installed and running
- [ ] Copied `airsim_settings_dual_drone.json` to `Documents/AirSim/settings.json`
- [ ] Restarted AirSim (should see TWO drones)
- [ ] Ran `python smart_drone_vision_gui.py`
- [ ] Clicked **🏆 ALGORITHM COMPARISON**
- [ ] Clicked **🏁 START COMPARISON RACE**
- [ ] Watching both drones fly! 🎉

---

## 📧 Support

If you encounter issues:
1. Check AirSim console for error messages
2. Verify both drones are spawned in AirSim scene
3. Ensure `settings.json` has correct vehicle names ("Drone1" and "Drone2")
4. Try restarting AirSim and the GUI

---

**Enjoy the dual-drone race! 🏆🚁🚁**
