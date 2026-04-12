# COMPLETE STARTUP CHECKLIST
## Fix drone not moving issue in under 60 seconds

### STEP 1: Install AirSim settings (30 seconds)
```powershell
# Run this command in PowerShell:
Copy-Item .\airsim_settings_dual_drone.json "$env:USERPROFILE\Documents\AirSim\settings.json" -Force

# Verify it was copied:
Get-Content "$env:USERPROFILE\Documents\AirSim\settings.json" | Select-String "Drone1"
```
**Expected output:** Should show line with `"Drone1": {`

---

### STEP 2: Restart AirSim/Unreal (15 seconds)
1. **Close** Unreal Engine / AirSim completely
2. **Open** AirSim again
3. **Wait** until map fully loads and you see drones
4. **Don't minimize** the simulator window

---

### STEP 3: Run diagnostic test (10 seconds)
```powershell
python test_drone_connection.py
```
**Expected output:** `✓ ALL TESTS PASSED`

If this fails, STOP and report the error message.

---

### STEP 4: Test actual movement (20 seconds)
```powershell
python test_movement.py
```
**Expected output:** You should see:
- Position changing from (0, 0) to (20, 20)
- Distance decreasing
- `✓ GOAL REACHED` message

If position stays at (0, 0):
- Check AirSim window is **focused** (click on it)
- Check physics simulation toggle (Home key)
- Try pressing `1` to switch to Drone1 view

---

### STEP 5: Run the main GUI (5 seconds)
```powershell
python smart_drone_vision_gui.py
```
- Click **START FLIGHT**
- Observe: Position should change within 10 seconds
- Distance to goal should decrease continuously

---

## TROUBLESHOOTING GUIDE

### Problem: "Nothing is working"
**Action:** Run the 4 tests above in order. Tell me which step fails and what error you see.

### Problem: test_movement.py works but GUI doesn't
**Action:** 
1. Close GUI
2. Run: `python smart_drone_vision_gui.py > debug.txt 2>&1`
3. Click START FLIGHT
4. Wait 30 seconds
5. Send me the `debug.txt` file

### Problem: Position updates but drone doesn't move
**Symptoms:** GUI shows changing numbers but drone stays at origin in AirSim
**Fix:** This was the exact bug I fixed. Make sure you're running the LATEST version:
```powershell
python -c "import smart_drone_vision_gui; print(hasattr(smart_drone_vision_gui, 'MAIN_VEHICLE_NAME'))"
```
Should print: `True`

If it prints `False`, the file didn't save correctly. Re-download it.

---

## ONE-MINUTE PERFORMANCE BENCHMARK

Expected timeline for "START FLIGHT" → "Drone moving":
- 0:00 - Click START FLIGHT
- 0:05 - Model loaded
- 0:10 - AirSim connected
- 0:15 - Takeoff complete
- 0:20 - Altitude reached (20m)
- 0:25 - First position update
- 0:30 - Drone clearly moving toward goal
- **If nothing happens by 0:60, timeout error appears**

---

## FILES CREATED FOR DEBUGGING

| File | Purpose | When to use |
|------|---------|-------------|
| `test_drone_connection.py` | Check AirSim connectivity | First diagnostic |
| `test_movement.py` | Verify movement API works | If GUI doesn't move drone |
| `STARTUP_CHECKLIST.md` | This file | Reference guide |

---

## QUICK REFERENCE

**Start fresh session:**
```powershell
# 1. Copy settings
Copy-Item .\airsim_settings_dual_drone.json "$env:USERPROFILE\Documents\AirSim\settings.json" -Force

# 2. Restart AirSim (manually)

# 3. Test
python test_drone_connection.py

# 4. Run GUI
python smart_drone_vision_gui.py
```

**Check if setup is correct:**
```powershell
# Should return "True" (means latest version):
python -c "import smart_drone_vision_gui; print('MAIN_VEHICLE_NAME' in dir(smart_drone_vision_gui))"

# Should return 10 (means all drones configured):
python -c "import json; data=json.load(open(r'$env:USERPROFILE\Documents\AirSim\settings.json')); print(len(data['Vehicles']))"
```
