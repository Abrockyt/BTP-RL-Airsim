import airsim
import numpy as np
import time
import sys

# --- CONFIGURATION & PHYSICS ---
def generate_turbulent_wind(base_velocity_x, base_velocity_y):
    turbulence_std = 1.5
    gust_x = np.random.normal(0, turbulence_std)
    gust_y = np.random.normal(0, turbulence_std)
    return airsim.Vector3r(base_velocity_x + gust_x, base_velocity_y + gust_y, 0)

class DroneBattery:
    def __init__(self, capacity_mah=5000, voltage=11.1):
        self.capacity = capacity_mah
        self.current_charge = capacity_mah
        self.voltage = voltage
        self.p_hover = 200 

    def update(self, velocity_x, velocity_y, dt):
        V = np.sqrt(velocity_x**2 + velocity_y**2)
        power_watts = self.p_hover * (1 + 0.05 * V**2)
        current_amps = power_watts / self.voltage
        drain_mah = (current_amps * 1000) * (dt / 3600)
        self.current_charge -= drain_mah
        return self.current_charge

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- STARTING SIMULATION SCRIPT ---")
    print("1. Looking for AirSim (Blocks.exe)...")
    
    # CONNECT TO AIRSIM
    client = airsim.MultirotorClient()
    
    try:
        client.confirmConnection()
        print("2. SUCCESS: Connected to AirSim!")
    except Exception as e:
        print("\n!!! ERROR: COULD NOT CONNECT !!!")
        print("Make sure 'Blocks.exe' is OPEN and running.")
        print(f"Error details: {e}")
        sys.exit(1)

    print("3. Taking Control of Drone...")
    client.enableApiControl(True)
    client.armDisarm(True)

    print("4. Taking Off...")
    client.takeoffAsync().join()
    
    # VISUALS: Draw the Goal (Red Dot)
    client.simPlotPoints([airsim.Vector3r(50, 50, -10)], color_rgba=[1, 0, 0, 1], size=20, is_persistent=True)

    battery = DroneBattery()
    print("5. Starting Wind Loop (Press Ctrl+C to stop)")

    try:
        while True:
            # Generate Wind
            wind_vector = generate_turbulent_wind(-5, 0)
            client.simSetWind(wind_vector)
            
            # Draw Wind Line (Yellow)
            drone_pos = client.getMultirotorState().kinematics_estimated.position
            wind_end = airsim.Vector3r(drone_pos.x_val + wind_vector.x_val, drone_pos.y_val + wind_vector.y_val, drone_pos.z_val)
            client.simPlotLineList([drone_pos, wind_end], color_rgba=[1, 1, 0, 1], thickness=5, duration=0.1)

            # Update Physics
            state = client.getMultirotorState()
            vel = state.kinematics_estimated.linear_velocity
            remaining = battery.update(vel.x_val, vel.y_val, dt=0.1)
            
            # Print Status
            sys.stdout.write(f"\r[RUNNING] Wind X: {wind_vector.x_val:5.2f} m/s | Battery: {remaining:7.2f} mAh")
            sys.stdout.flush()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        client.reset()