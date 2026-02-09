# 🎯 Quick Start Guide - Enhanced Smart Drone

## What's New (Feb 4, 2026)

### ✅ Improvements Implemented:

1. **📊 Performance Graphs** - Automatic comparison of previous vs current sessions
2. **🏔️ Ground Safety** - Maintains 2.5m minimum height with auto-climb
3. **🔮 Enhanced Prediction** - 5-zone analysis with 3-level threat assessment (3m-7m range)

---

## 🚀 How to Use

### First-Time Setup
```bash
# 1. Make sure AirSim is running
# 2. Navigate to project folder
cd C:\Users\abroc\Desktop\UAV_Energy_Sim

# 3. Run the smart drone
python smart_drone_gui.py
```

### During Flight
- **GUI shows live stats**: Altitude, Ground Warnings, Speed
- **Console shows events**: Collision predictions, ground warnings, recoveries
- **Map updates**: Real-time position tracking

### After Flight (Stop Button)
1. Click "Stop Flight"
2. Popup shows session summary
3. **Graph automatically saves** to `performance_graphs/comparison_YYYYMMDD_HHMMSS.png`
4. Open the PNG to see 6 comparison charts

---

## 📊 What the Graphs Show

### 1. Collisions (Lower is Better)
- Previous average vs Current session
- **Goal**: Reduce collisions over time

### 2. Prediction Success (Higher is Better)
- Obstacles avoided through early detection
- **Goal**: Increase predictive avoidance

### 3. Ground Warnings (NEW - Lower is Better)
- Times altitude dropped below 2.5m
- **Goal**: Maintain safe altitude

### 4. Average Speed
- Flight efficiency in m/s
- **Goal**: Maintain high speed while staying safe

### 5. Altitude Maintenance
- Average height above ground
- Green line shows 2.5m minimum safety threshold

### 6. Safety Score
- Formula: `(predictions - collisions) / total * 100`
- **Positive** = More predictions than collisions (GOOD)
- **Negative** = More collisions than predictions (NEEDS IMPROVEMENT)
- **Goal**: Achieve +70% or higher

---

## 🎮 Testing the New Features

### Test 1: Ground Safety
```
1. Start flight
2. Fly close to ground manually (use map to navigate down)
3. Watch console for "⚠️ LOW ALTITUDE WARNING"
4. Drone should auto-climb
5. Check GUI for warning count
```

### Test 2: Enhanced Prediction
```
1. Start flight
2. Navigate toward buildings/obstacles
3. Watch for "🔮 COLLISION PREDICTED!"
4. Drone should avoid smoothly
5. Check graphs after - should show high prediction count
```

### Test 3: Performance Tracking
```
1. Run 3-5 flight sessions
2. After each, check the generated graph
3. Compare safety scores
4. Watch improvement over time!
```

---

## 📁 Files to Check

After running flights, check these folders:

```
performance_graphs/
├── comparison_20260204_143022.png  ← Your first flight
├── comparison_20260204_144510.png  ← Your second flight
├── comparison_20260204_150234.png  ← Your third flight
└── history.json                    ← All session data
```

---

## 🎯 Success Criteria

Your system is working well if you see:

✅ **Safety Score improving** (moving toward +70% or higher)  
✅ **Collision count decreasing** (fewer actual hits)  
✅ **Prediction count increasing** (more early avoidance)  
✅ **Ground warnings low** (0-2 per session)  
✅ **Average altitude stable** (above 2.5m minimum)  

---

## 🐛 Troubleshooting

**Issue: No graphs generated**
```
Solution: Check console for errors when clicking Stop Flight
Verify: performance_graphs/ folder exists
```

**Issue: Ground warnings every second**
```
Cause: Flying in very hilly terrain
Solution: Increase starting altitude to 4-5m
Edit: Change STARTING_ALTITUDE = -4.0 or -5.0 in code
```

**Issue: Previous avg shows 0.0**
```
Cause: This is your first flight
Solution: Run 2-3 more sessions to build history
```

---

## 💡 Pro Tips

1. **Establish Baseline**: First 2-3 flights establish your performance baseline
2. **Progressive Improvement**: Each session should improve your safety score
3. **Study Graphs**: Look at which metric improved most
4. **Test Scenarios**: Intentionally fly toward obstacles to test prediction
5. **Height Advantage**: Flying at 4-5m gives more room for ground safety

---

## 🎉 Expected Results

After 5 flights, you should see:

| Metric | Flight 1 | Flight 5 | Improvement |
|--------|----------|----------|-------------|
| Collisions | 5-7 | 1-2 | 60-70% ↓ |
| Predictions | 8-10 | 15-20 | 80-100% ↑ |
| Ground Warnings | 3-5 | 0-1 | 80-100% ↓ |
| Safety Score | +30% | +75% | 150% ↑ |

---

**Ready to fly? Start AirSim and run `python smart_drone_gui.py`!** 🚀
