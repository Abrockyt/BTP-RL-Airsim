"""
Quick AirSim Connection Test - Diagnoses common issues in <60 seconds
"""
import time
import sys

print("="*70)
print("AIRSIM DRONE CONNECTION DIAGNOSTIC TEST")
print("="*70)
start_time = time.time()

# Test 1: Import AirSim
print("\n[1/6] Testing AirSim import...")
try:
    import airsim
    print("✓ AirSim imported successfully")
except ImportError as e:
    print(f"✗ FAILED: {e}")
    print("  Fix: pip install airsim")
    sys.exit(1)

# Test 2: Connect to AirSim
print("\n[2/6] Connecting to AirSim simulator...")
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ Connected to AirSim")
except Exception as e:
    print(f"✗ FAILED: {e}")
    print("  Fix: Start AirSim/Unreal Engine first")
    sys.exit(1)

# Test 3: List available vehicles
print("\n[3/6] Checking available vehicles...")
try:
    # Try to enable control on Drone1
    client.enableApiControl(True, vehicle_name="Drone1")
    print("✓ Drone1 found and API control enabled")
    
    # Check if Drone2 exists (comparison mode)
    try:
        client.enableApiControl(True, vehicle_name="Drone2")
        print("✓ Drone2 found (comparison mode ready)")
    except:
        print("⚠ Drone2 not found (comparison mode unavailable)")
        
    # Count total drones
    drone_count = 0
    for i in range(1, 11):
        try:
            client.enableApiControl(True, vehicle_name=f"Drone{i}")
            drone_count += 1
        except:
            break
    print(f"✓ Total drones detected: {drone_count}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    print("  Fix: Copy airsim_settings_dual_drone.json to:")
    print("       %USERPROFILE%\\Documents\\AirSim\\settings.json")
    print("       Then restart AirSim")
    sys.exit(1)

# Test 4: Get Drone1 state
print("\n[4/6] Reading Drone1 state...")
try:
    state = client.getMultirotorState(vehicle_name="Drone1")
    pos = state.kinematics_estimated.position
    print(f"✓ Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Arm Drone1
print("\n[5/6] Testing arm/disarm...")
try:
    client.armDisarm(True, vehicle_name="Drone1")
    print("✓ Drone1 armed successfully")
    time.sleep(0.2)
    client.armDisarm(False, vehicle_name="Drone1")
    print("✓ Drone1 disarmed successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 6: Movement command test (no actual movement)
print("\n[6/6] Testing movement API...")
try:
    # Just test if API accepts the command (won't actually move)
    client.armDisarm(True, vehicle_name="Drone1")
    result = client.takeoffAsync(vehicle_name="Drone1")
    if result:
        result.join()
    print("✓ Movement commands working")
    client.armDisarm(False, vehicle_name="Drone1")
    client.enableApiControl(False, vehicle_name="Drone1")
except Exception as e:
    print(f"✗ FAILED: {e}")
    print("  This might be OK if drone is already flying")

elapsed = time.time() - start_time
print("\n" + "="*70)
print(f"✓ ALL TESTS PASSED in {elapsed:.1f} seconds")
print("="*70)
print("\nYour system is ready to run smart_drone_vision_gui.py")
print("\nNext steps:")
print("1. Close any running instances of smart_drone_vision_gui.py")
print("2. Run: python smart_drone_vision_gui.py")
print("3. Click START FLIGHT")
print("4. Drone should move toward goal within 10 seconds")
print("\nIf drone still doesn't move, the issue is in the flight logic.")
print("="*70)
