# 🧠 Reinforcement Learning System - Complete Guide

## Overview
The Smart Vision Drone GUI now includes **online reinforcement learning** that allows the drone to improve its energy efficiency over multiple runs. The system uses **Proximal Policy Optimization (PPO)** with a **Multi-Head Attention (MHA) Actor network** to learn optimal flight paths.

---

## 🎯 How Reinforcement Learning Works

### 1. **Learning Process**
```
1️⃣ Drone flies to goal (collecting experiences)
2️⃣ Each step: Observes state → Takes action → Receives reward
3️⃣ Goal reached: Trains neural network on collected data
4️⃣ Saves improved model → Next run uses better policy
5️⃣ Repeat: Performance improves over time
```

### 2. **State (What the drone observes)**
- `goal_x, goal_y`: Distance to goal in X and Y directions
- `velocity_x, velocity_y`: Current drone movement speed
- `wind_x, wind_y`: Estimated wind conditions
- `battery_percent`: Remaining power (0-100%)

**Total: 7-dimensional state vector**

### 3. **Actions (What the drone decides)**
- `vx`: Velocity in X direction (-1 to +1, scaled to ±15 m/s)
- `vy`: Velocity in Y direction (-1 to +1, scaled to ±15 m/s)
- `vz`: Velocity in Z direction (altitude control)

**Total: 3-dimensional action vector**

### 4. **Rewards (How performance is measured)**

| Reward Component | Formula | Purpose |
|------------------|---------|---------|
| **Distance** | `-distance × 0.1` | Encourages moving closer to goal |
| **Energy** | `-energy_consumed × 2.0` | Penalizes high power usage |
| **Goal Bonus** | `+100` (if within 5m) | Big reward for success |
| **Speed** | `+1.0` (if 8-15 m/s) | Maintains efficient cruise |
| **Battery** | `-10.0` (if <20%) | Prevents power depletion |

**Total reward** = Sum of all components at each timestep

### 5. **Training Algorithm (PPO)**

After goal is reached:
```python
1. Calculate Returns: 
   R(t) = r(t) + γ×r(t+1) + γ²×r(t+2) + ...
   
2. Update Actor (Decision Maker):
   - Learns which actions lead to higher rewards
   - Uses advantage function: A = Returns - ValueEstimate
   
3. Update Critic (Value Estimator):
   - Learns to predict future rewards
   - Minimizes: MSE(predicted_value, actual_return)
   
4. Save improved model to: trained_models/mha_ppo_runN.pth
```

**Learning Rate:** 0.0001 (when enabled)  
**Training Epochs:** 4 iterations per update  
**Discount Factor (γ):** 0.99 (values future rewards)

---

## 📊 Graph Explanations

### **1. SPEED Graph** 🚀
**Purpose:** Monitor horizontal velocity for efficiency

- **What it shows:** Drone's speed in meters per second
- **Why it matters:** 
  - Too slow = Takes longer, wastes time
  - Too fast = High energy consumption
  - Optimal: **8-15 m/s** for balance
- **Target line:** Green dashed at **12 m/s** (cruise speed)
- **RL Goal:** Learn to maintain steady, efficient speed

**Example:**
```
Good: Smooth line around 12 m/s
Bad: Erratic spikes or very low speed
```

---

### **2. ALTITUDE Graph** 🏔️
**Purpose:** Maintain safe height above terrain

- **What it shows:** Height above ground level (AGL) in meters
- **Why it matters:**
  - Too low = Risk of hitting obstacles
  - Too high = Unnecessary energy to climb
  - Target: **20m AGL** for obstacle clearance
- **Target line:** Green dashed at **20m**
- **RL Goal:** Maintain consistent altitude (saves energy vs constant changes)

**Example:**
```
Good: Stable line around 20m
Bad: Frequent up/down oscillations
```

---

### **3. BATTERY Graph** 🔋
**Purpose:** Monitor remaining power

- **What it shows:** Battery percentage (100% = full, 0% = empty)
- **Why it matters:**
  - Must reach goal before battery depletes
  - Critical level: **20%** (warning threshold)
- **Color coding:**
  - Green > 50%: Healthy
  - Orange 20-50%: Moderate
  - Red < 20%: Critical
- **RL Goal:** Complete mission with battery to spare

**Example:**
```
Good: Gradual decline, ends above 50%
Bad: Rapid drop, reaches critical level
```

---

### **4. ENERGY Graph** ⚡ **(Most Important for RL!)**
**Purpose:** Track total energy consumed

- **What it shows:** Cumulative energy in Watt-hours (Wh)
- **Why it INCREASES over time:**
  ```
  Energy = Power × Time
  - Power ≈ 200W constant (hovering)
  - Power increases with speed²
  - As time passes, total energy accumulates
  - This is CORRECT behavior!
  ```
- **RL Optimization:**
  - **Goal:** Minimize the **FINAL** energy value
  - Better path = Lower end value
  - Shows best run comparison (green dashed line)
- **What RL learns:**
  - Shorter paths consume less total energy
  - Efficient speed reduces power spikes
  - Smooth flight avoids acceleration costs

**Example:**
```
Run #1: Ends at 2.5 Wh (baseline)
Run #2: Ends at 2.3 Wh (8% improvement!)
Run #3: Ends at 2.1 Wh (16% total improvement!)
```

**Comparison Line:**
- Green dashed = Best energy achieved so far
- Current run below line = Improvement! ⬆️
- Current run above line = Worse performance ⬇️

---

### **5. FLIGHT MAP** 🗺️
**Purpose:** Visualize drone trajectory

- **What it shows:**
  - Green circle: Start position (0, 0)
  - Yellow/red star: Goal position (clickable)
  - Cyan path: Drone's flight trajectory
  - Blue circle: Current drone position
  - Terrain background: Satellite-style map
- **Why it matters:**
  - Shows if drone is taking direct path
  - Reveals obstacle avoidance maneuvers
  - Helps debug navigation issues
- **RL Goal:** Learn straighter, more efficient paths

**Example:**
```
Good: Nearly straight line from start to goal
Bad: Zigzag pattern or loops
```

---

## 🎓 Online Learning Panel

Located in left sidebar:

### **Run Statistics:**
- **Run #:** Current flight attempt number
- **Best Energy:** Lowest Wh achieved across all runs
- **Best Time:** Fastest completion time
- **Improvement:** % change vs previous run
  - Green ⬆️: Better than last run
  - Red ⬇️: Worse than last run

### **Enable Online Learning Checkbox:**
- ✅ **Checked (default):** Agent trains after each run
- ⬜ **Unchecked:** Uses pre-trained policy only (no learning)

**Use cases:**
- Keep enabled for automatic improvement
- Disable to test baseline performance
- Compare learned vs pre-trained behavior

---

## 📈 Expected Learning Progression

### **Run #1 (Baseline)**
```
Time: 45.2s
Energy: 2.5 Wh
Path: Somewhat direct, some hesitation
```

### **Run #2 (After 1st training)**
```
Time: 42.8s (-5.3%)
Energy: 2.35 Wh (-6%)
Path: Smoother, fewer speed changes
Improvement: +6% ⬆️
```

### **Run #3 (After 2nd training)**
```
Time: 40.1s (-11.3%)
Energy: 2.2 Wh (-12%)
Path: Nearly straight, efficient speed
Improvement: +6.4% ⬆️
```

### **Run #10+ (Converged)**
```
Time: ~38s (stable)
Energy: ~2.0 Wh (optimal)
Path: Consistently efficient
Improvement: Plateaus around 0-2%
```

---

## 🔧 Technical Details

### **Model Architecture:**
```
MHA_Actor (Actor Network):
  Input: State (7 dims)
  ├─ Linear: 7 → 64
  ├─ Multi-Head Attention (4 heads)
  ├─ Linear: 64 → 128 (ReLU)
  ├─ Linear: 128 → 64 (ReLU)
  └─ Linear: 64 → 3 (Tanh) → Actions

Critic (Value Network):
  Input: State (7 dims)
  ├─ Linear: 7 → 128 (ReLU)
  ├─ Linear: 128 → 64 (ReLU)
  └─ Linear: 64 → 1 → State Value
```

### **Training Hyperparameters:**
```python
Learning Rate: 0.0001 (Adam optimizer)
Discount Factor (γ): 0.99
PPO Clip Range (ε): 0.2
Training Epochs (K): 4
Exploration Noise (σ): 0.3
```

### **Saved Models:**
After each successful run:
```
File: trained_models/mha_ppo_run{N}.pth
Contains:
  - run_number
  - actor_state_dict (decision network)
  - critic_state_dict (value network)
  - final_energy (performance metric)
  - final_time (completion time)
```

---

## 🎯 Why Energy Graph Increases (FAQ)

**Q: Is it a bug that energy keeps going up?**  
**A:** No! This is correct physics:

```
Energy (Wh) = ∫(Power × dt)

Where:
- Power ≈ 200W (hovering) + 0.005×speed²
- dt = time interval (0.1s per step)

As time passes, energy ACCUMULATES:
  t=0s:  Energy = 0 Wh
  t=10s: Energy = 0.56 Wh
  t=20s: Energy = 1.12 Wh
  t=40s: Energy = 2.24 Wh
```

**Q: So what does RL optimize?**  
**A:** RL minimizes the **FINAL** energy value by:
- Finding shorter paths (less time = less accumulation)
- Maintaining efficient speed (reduces power spikes)
- Avoiding unnecessary maneuvers (smooth flight)

**Q: How do I know if RL is working?**  
**A:** Check the green "Best" line on Energy graph:
- Run gets progressively closer to or below this line
- Final energy values decrease across runs
- Improvement % stays positive

---

## 📊 Performance Metrics

### **What to Track:**
1. **Final Energy (Wh)** - Primary RL objective
2. **Completion Time (s)** - Secondary objective
3. **Flight Path Efficiency** - Visual inspection on map
4. **Improvement %** - Run-to-run progress

### **Success Indicators:**
- ✅ Energy decreasing over runs
- ✅ Time decreasing or stable
- ✅ Smoother speed/altitude profiles
- ✅ More direct flight paths
- ✅ Consistent performance (low variance)

---

## 🚀 Quick Start Guide

### **Running with RL:**
1. Launch GUI: `python smart_drone_vision_gui.py`
2. Ensure "Enable Online Learning" is ✅ checked
3. Set goal on map or enter coordinates
4. Click "🚀 START FLIGHT"
5. Wait for goal to be reached
6. Observe training output in console
7. Check "Improvement" in RL panel
8. Repeat for progressive improvement!

### **Console Output:**
```
====================================================
🚁 STARTING RUN #3
====================================================
🧠 Loading MHA-PPO Agent (1M steps training)...
✓ MHA-PPO Brain loaded successfully
🎓 Online Learning ENABLED - Agent will improve
📍 Ground height: 0.0m
📊 Initializing performance graphs...
✓ Graphs ready - starting flight!

Step  0 | Pos: (0.0, 0.0) | Goal: (100.0, 100.0) | Dist: 141.4m
Step 100 | Pos: (45.2, 38.7) | Dist: 78.3m | Battery: 98.2%
...
🏆 GOAL REACHED! Distance: 4.2m

🎓 TRAINING AGENT...
   Collected 342 experiences
   Actor Loss: 0.0234 | Critic Loss: 0.1567
   💾 Model saved: trained_models/mha_ppo_run3.pth
   ✓ Training complete!

📊 RUN #3 SUMMARY:
   Time: 40.1s
   Energy: 2.20 Wh
   Best Energy: 2.20 Wh (NEW RECORD!)
   Best Time: 40.1s (NEW RECORD!)
====================================================
```

---

## 🎉 Benefits

1. **Automatic Improvement:** No manual tuning needed
2. **Energy Efficiency:** Learns to minimize power consumption
3. **Adaptive Navigation:** Handles different goal positions
4. **Obstacle Learning:** Improves avoidance strategies
5. **Performance Tracking:** Clear visualization of progress
6. **Explainability:** All graphs have clear purposes
7. **Flexibility:** Toggle learning on/off anytime

---

## 📚 Further Reading

- **PPO Algorithm:** [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- **Multi-Head Attention:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Reinforcement Learning:** [Sutton & Barto Book](http://incompleteideas.net/book/the-book-2nd.html)

---

**Created:** February 9, 2026  
**Version:** 1.0  
**System:** Smart Vision Drone GUI with MHA-PPO Learning
