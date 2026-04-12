"""
Quick test: Is AirSim ready for GNN swarm?
"""
import airsim
import sys

print("🔍 Checking AirSim setup...")

try:
    # Connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ Connected to AirSim")
    
    # Check drones
    drones_found = []
    for i in range(1, 11):
        try:
            client.enableApiControl(True, vehicle_name=f"Drone{i}")
            drones_found.append(f"Drone{i}")
        except:
            break
    
    print(f"✓ Found {len(drones_found)} drones: {', '.join(drones_found[:3])}...")
    
    if len(drones_found) < 7:
        print(f"\n⚠️  WARNING: Need at least 7 drones, found {len(drones_found)}")
        print("\nFix:")
        print("1. Copy settings: Copy-Item .\\airsim_settings_dual_drone.json \"$env:USERPROFILE\\Documents\\AirSim\\settings.json\" -Force")
        print("2. Restart AirSim/Unreal Engine")
        sys.exit(1)
    
    # Test Drone1 movement
    print("\n🧪 Testing Drone1 movement...")
    client.armDisarm(True, vehicle_name="Drone1")
    state = client.getMultirotorState(vehicle_name="Drone1")
    pos = state.kinematics_estimated.position
    print(f"✓ Drone1 position: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})")
    client.armDisarm(False, vehicle_name="Drone1")
    
    print("\n" + "="*60)
    print("✅ READY! Run: python simple_gnn_swarm.py")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    print("\nMake sure:")
    print("1. AirSim/Unreal Engine is running")
    print("2. Settings file is copied")
    print("3. Drones are visible in the simulator")
    sys.exit(1)
