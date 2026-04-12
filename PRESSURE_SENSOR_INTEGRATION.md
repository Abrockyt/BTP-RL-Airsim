# 💨 Pressure Sensor Integration for Heavy Wind Navigation

## Overview
Added barometric pressure sensor to enable drone navigation during heavy wind conditions when LiDAR and RGB camera become unreliable.

## Features Implemented

### 1. **Barometer Data Acquisition**
- Reads air pressure from AirSim's barometer sensor
- Calculates wind magnitude from acceleration data
- Updates at every control loop iteration

### 2. **Heavy Wind Detection**
The system detects heavy wind conditions using two criteria:
- **Wind Speed**: > 15.0 m/s (54 km/h)
- **Pressure Change**: > ±5000 Pa from sea level (101325 Pa)

When heavy wind is detected:
- `heavy_wind_detected` flag is set to `True`
- `vision_failed` flag disables camera/LiDAR navigation
- Drone switches to pressure-based navigation mode

### 3. **State Vector Expansion**
**Previous**: 7D state vector
```python
[goal_x, goal_y, vel_x, vel_y, wind_x, wind_y, battery]
```

**Current**: 9D state vector
```python
[goal_x, goal_y, vel_x, vel_y, wind_x, wind_y, battery, 
 pressure_normalized, wind_mag_normalized]
```

**Normalization**:
- `pressure_normalized = (pressure - 96000.0) / 10000.0`  # Range: 96-106 kPa → 0-1
- `wind_mag_normalized = clip(wind_mag / 30.0, 0.0, 1.0)`  # Range: 0-30 m/s → 0-1

### 4. **UI Display Panel**
New "💨 PRESSURE SENSOR" panel shows:
- **Air Pressure**: Current atmospheric pressure in Pascals
- **Wind Magnitude**: Wind speed in m/s
- **Status**: NORMAL ✓ or HEAVY WIND ⚠️
- **Vision**: ACTIVE 📷 or DISABLED ⚠️

### 5. **Navigation Fallback**
When `vision_failed = True`:
- Bypasses obstacle avoidance (vision-based)
- Uses RL agent with pressure data for navigation
- Prints warning: "💨 Heavy wind detected - Using pressure-based navigation"
- Progress logs show wind status and vision status

### 6. **Reward Function Updates**
Added two new penalty components:

**Wind Penalty**:
```python
wind_penalty = -wind_magnitude * 0.5  # Penalize heavy wind navigation
```

**Vision Failure Penalty**:
```python
vision_penalty = -5.0 if vision_failed else 0.0  # Operating blind is riskier
```

**Total Reward**:
```python
total_reward = distance_reward + energy_reward + goal_reward + 
               speed_reward + battery_reward + wind_penalty + vision_penalty
```

## Technical Details

### Sensor Data Function
```python
def get_pressure_sensor_data(self):
    """Get barometer/pressure sensor data from AirSim"""
    try:
        # Get barometer data
        barometer_data = self.client.getBarometerData()
        self.air_pressure = barometer_data.pressure
        
        # Get IMU data for wind estimation
        imu_data = self.client.getImuData()
        accel = imu_data.linear_acceleration
        
        # Estimate wind magnitude from acceleration (simplified model)
        self.wind_magnitude = np.linalg.norm([accel.x_val, accel.y_val]) * 10.0
        
        # Detect heavy wind conditions
        if self.wind_magnitude > 15.0 or abs(self.air_pressure - 101325.0) > 5000:
            self.heavy_wind_detected = True
            self.vision_failed = True  # Vision unreliable in heavy wind
        else:
            self.heavy_wind_detected = False
            self.vision_failed = False
        
        return self.air_pressure, self.wind_magnitude
    except:
        return 101325.0, 0.0
```

### State Vector Function (Updated)
```python
def get_state_vector(self):
    """Get 9D state vector for RL agent with pressure sensor data"""
    try:
        # ... existing code for goal, velocity, wind, battery ...
        
        # Get pressure sensor data
        pressure, wind_mag = self.get_pressure_sensor_data()
        
        # Normalize pressure and wind for RL
        pressure_normalized = (pressure - 96000.0) / 10000.0
        wind_mag_normalized = np.clip(wind_mag / 30.0, 0.0, 1.0)
        
        # 9D state vector
        state = np.array([
            goal_x, goal_y,
            vel_x, vel_y,
            wind_x, wind_y,
            battery_normalized,
            pressure_normalized,
            wind_mag_normalized
        ], dtype=np.float32)
        
        return state, current_pos, velocity
    except:
        return None, None, None
```

## Model Compatibility

⚠️ **IMPORTANT**: The pre-trained model `mha_ppo_1M_steps.pth` was trained on **7D state vectors**. 

**Options**:
1. **Retrain from scratch** with 9D state (recommended for best performance)
2. **Fine-tune** existing model with new pressure inputs
3. **Transfer learning**: Freeze early layers, train new input layer
4. **Load with `strict=False`**: Allow dimension mismatch (may degrade performance)

## Testing Heavy Wind Scenarios

To test the pressure sensor in AirSim:

1. **Simulate Heavy Wind**: Modify AirSim's `settings.json`:
```json
{
  "SimMode": "Multirotor",
  "Wind": {
    "WSpeed": 20.0,
    "WDirection": "0,1,0"
  }
}
```

2. **Monitor Behavior**:
   - Watch pressure sensor panel for "HEAVY WIND ⚠️" status
   - Check console for "💨 Heavy wind detected" messages
   - Verify vision status changes to "DISABLED ⚠️"

3. **Observe Navigation**:
   - Drone should still reach goal using pressure data
   - RL agent learns to navigate with 9D state
   - Reward function penalizes risky heavy wind operation

## Performance Impact

### Advantages
✅ **Sensor Redundancy**: Operates when vision fails
✅ **Safety**: Detects dangerous wind conditions
✅ **RL Learning**: Agent learns to handle adverse weather
✅ **Real-time Monitoring**: Live pressure/wind display

### Challenges
⚠️ **Model Retraining**: Need to retrain for 9D state
⚠️ **Simplified Wind Model**: Acceleration-based estimation is approximate
⚠️ **No Obstacle Avoidance**: Vision-disabled mode bypasses obstacles

## Next Steps

1. **Retrain Model**: Train new model with 9D state space
2. **Improve Wind Estimation**: Use more sophisticated wind models
3. **Pressure-Based Obstacles**: Develop pressure gradient obstacle detection
4. **Adaptive Navigation**: Blend vision and pressure data when both available
5. **Comprehensive Testing**: Validate in various wind scenarios

## Configuration

All pressure sensor thresholds can be adjusted:
```python
# Heavy wind detection thresholds
HEAVY_WIND_THRESHOLD = 15.0  # m/s
PRESSURE_CHANGE_THRESHOLD = 5000  # Pa

# Normalization ranges
PRESSURE_MIN = 96000.0  # Pa (low altitude)
PRESSURE_MAX = 106000.0  # Pa (high pressure system)
WIND_MAX = 30.0  # m/s (hurricane force)
```

## References

- **AirSim Barometer API**: `client.getBarometerData()`
- **AirSim IMU API**: `client.getImuData()`
- **Multi-Head Attention PPO**: `mha_ppo_agent.py`
- **RL System Documentation**: `RL_SYSTEM_EXPLAINED.md`

---

**Status**: ✅ Fully Integrated (UI + Data + Navigation + Reward)
**Version**: 1.0 (9D State Space)
**Date**: February 2026
