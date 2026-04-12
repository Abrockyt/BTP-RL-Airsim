# Quick Test - Verify Dual Drone Setup
# Run this after restarting AirSim

import airsim
import time

print("=" * 60)
print("🔍 TESTING DUAL-DRONE SETUP")
print("=" * 60)

try:
    # Test Drone 1
    print("\n📡 Connecting to Drone1...")
    client1 = airsim.MultirotorClient()
    client1.confirmConnection()
    client1.enableApiControl(True, vehicle_name="Drone1")
    print("   ✅ Drone1 connected successfully!")
    
    # Test Drone 2
    print("\n📡 Connecting to Drone2...")
    client2 = airsim.MultirotorClient()
    client2.confirmConnection()
    client2.enableApiControl(True, vehicle_name="Drone2")
    print("   ✅ Drone2 connected successfully!")
    
    # Get positions
    print("\n📍 Getting drone positions...")
    state1 = client1.getMultirotorState(vehicle_name="Drone1")
    pos1 = state1.kinematics_estimated.position
    print(f"   Drone1: ({pos1.x_val:.1f}, {pos1.y_val:.1f}, {pos1.z_val:.1f})")
    
    state2 = client2.getMultirotorState(vehicle_name="Drone2")
    pos2 = state2.kinematics_estimated.position
    print(f"   Drone2: ({pos2.x_val:.1f}, {pos2.y_val:.1f}, {pos2.z_val:.1f})")
    
    # Cleanup
    client1.enableApiControl(False, vehicle_name="Drone1")
    client2.enableApiControl(False, vehicle_name="Drone2")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🚀 Ready to run: python smart_drone_vision_gui.py")
    print()
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n⚠️ Make sure:")
    print("  1. AirSim is running")
    print("  2. Two drones spawned (Drone1 and Drone2)")
    print("  3. You restarted AirSim after updating settings")
