# SIMPLE GNN SWARM - QUICK START

## What This Does
✅ **Background drones (Drone2-7):** Move randomly  
✅ **Main drone (Drone1):** Uses GNN to reach goal while communicating with others  
✅ **Simple:** No complex modes, just works  

---

## Setup (1 minute)

### 1. Copy AirSim settings
```powershell
Copy-Item .\airsim_settings_dual_drone.json "$env:USERPROFILE\Documents\AirSim\settings.json" -Force
```

### 2. Start AirSim
- Open Unreal Engine / AirSim
- Wait for map to load
- You should see 10 drones

### 3. Run the system
```powershell
python simple_gnn_swarm.py
```

---

## What You'll See

**Terminal output:**
```
Step   0 | Pos: (   0.0,    0.0) | Goal:  141.4m | Comm: 3 drones | Active: 7
Step   5 | Pos: (  12.3,   15.7) | Goal:  127.8m | Comm: 4 drones | Active: 7
Step  10 | Pos: (  25.8,   31.2) | Goal:  112.5m | Comm: 2 drones | Active: 7
...
🎉 GOAL REACHED in 45 steps!
```

**In AirSim:**
- 6 drones moving randomly (background)
- 1 main drone (Drone1) flying straight to goal
- Main drone changes path when near other drones (GNN communication)

---

## How It Works

### GNN Communication
- Main drone detects other drones within 30m
- GNN processes:
  - Main drone's goal direction
  - Positions of nearby drones
  - Velocities of nearby drones
- Outputs: Movement commands that avoid collisions while reaching goal

### Node Features (per drone)
```
[goal_rel_x, goal_rel_y, velocity_x, velocity_y]
```

### Adjacency Matrix
```
Communication = 1 if distance < 30m, else 0
```

---

## Modify Goal

Edit line 23 in `simple_gnn_swarm.py`:
```python
GOAL = np.array([100.0, 100.0])  # Change coordinates
```

---

## Troubleshooting

### Error: "TransportError: Retry connection"
**Fix:** Start AirSim/Unreal Engine first

### Drones don't move
**Fix:** Make sure settings.json copied correctly:
```powershell
Get-Content "$env:USERPROFILE\Documents\AirSim\settings.json" | Select-String "Drone1"
```

### Only main drone appears
**Fix:** Restart AirSim after copying settings

---

## Compared to Old System

| Old System | New System |
|------------|------------|
| 3000+ lines | 270 lines |
| Multiple modes | One simple mode |
| Complex GUI | Terminal only |
| Many features | Does one thing well |
| Hard to modify | Easy to change |

---

## Next Steps

Once this works, you can:
1. Train the GNN model: `python train_gnn_swarm.py` (I can create this)
2. Add more drones: Edit NUM_DRONES variable
3. Change communication range: Edit COMM_RANGE variable
4. Add obstacles: Modify GNN input features

**Want to train the GNN model? Let me know and I'll create the training script.**
