# Multi-Drone AirSim Functions - To be integrated into smart_drone_vision_gui.py
# These functions handle real AirSim control of 5 drones

def _check_inter_drone_collision(self, drone_pos, drone_name):
    """Check if any other drone is too close (collision risk)"""
    COLLISION_SAFETY_DISTANCE = 15.0  # 15m minimum separation
    
    for other_id in range(1, 6):
        if other_id == 1 and drone_name == "Drone1":
            continue
        
        if len(self.multidrone_paths[other_id]) > 0:
            other_pos = np.array(self.multidrone_paths[other_id][-1], dtype=np.float32)
            distance = float(np.linalg.norm(drone_pos - other_pos))
            
            if distance < COLLISION_SAFETY_DISTANCE:
                return True  # Collision risk detected
    
    return False

def _fly_main_drone_multidrone_real(self):
    """Main drone (Drone1) flight control in AirSim multi-drone environment"""
    print("🔴 Main Drone (Drone1 - MHA-PPO) starting flight...")
    try:
        client = self.md_client
        start_time = time.time()
        energy = 0.0
        dt = 0.18
        step = 0
        
        # Takeoff
        safe_airsim_call(client.enableApiControl, True, vehicle_name="Drone1")
        safe_airsim_call(client.takeoffAsync, vehicle_name="Drone1").join()
        print("🛫 Drone1 took off")
        
        while self.multidrone_active:
            elapsed = time.time() - start_time
            
            # Get Drone1 state
            state = safe_airsim_call(client.getMultirotorState, vehicle_name="Drone1")
            if state is None:
                time.sleep(0.1)
                continue
            
            pos = state.kinematics_estimated.position
            velocity = state.kinematics_estimated.linear_velocity
            current_pos = np.array([pos.x_val, pos.y_val], dtype=np.float32)
            goal_pos = np.array([self.goal_x, self.goal_y], dtype=np.float32)
            direction = goal_pos - current_pos
            distance = float(np.linalg.norm(direction))
            
            # Store position for visualization
            self.multidrone_paths[1].append([pos.x_val, pos.y_val])
            
            # Check goal reached
            if distance < 2.5:
                print(f"🏆 MAIN DRONE (Drone1) REACHED GOAL! Time: {elapsed:.1f}s Energy: {energy:.2f}Wh")
                print("   Landing Drone1...")
                safe_airsim_call(client.landAsync, vehicle_name="Drone1").join()
                speed = 0.0
                break
            
            # Collision detection with other drones
            inter_drone_collision = self._check_inter_drone_collision(current_pos, "Drone1")
            
            # Navigation with collision prediction
            if distance > 1.0:
                direction_normalized = direction / distance
                
                # Speed control (reduce if another drone is close)
                if distance > 60.0:
                    desired_speed = 20.0
                elif distance > 35.0:
                    desired_speed = 16.5
                else:
                    desired_speed = max(6.5, distance * 0.65)
                
                if inter_drone_collision:
                    desired_speed *= 0.5  # Reduce speed if other drone nearby
                    self.comm_avoidance_count += 1
                
                target_vx = direction_normalized[0] * desired_speed
                target_vy = direction_normalized[1] * desired_speed
                
                # Altitude control
                height_above_ground = abs(pos.z_val) - self.ground_height
                target_height = 20.0
                height_error = target_height - height_above_ground
                vz_measured = float(velocity.z_val) if hasattr(velocity, 'z_val') else 0.0
                target_vz = np.clip((0.06 * height_error) - (0.98 * vz_measured), -0.4, 0.4)
                if abs(height_error) < 3.5 and abs(vz_measured) < 0.15:
                    target_vz = 0.0
                
                # Apply velocity command
                safe_airsim_call(client.moveByVelocityAsync, float(target_vx), float(target_vy), float(-target_vz), duration=1.2, vehicle_name="Drone1")
            else:
                target_vx = target_vy = 0.0
            
            # Energy calculation
            speed = float(np.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)) if hasattr(velocity, 'x_val') else 0.0
            power = 15.0 * (1 + 0.005 * speed**2)
            energy_step = power * (dt / 3600.0)
            energy += energy_step
            battery = 100.0 - (energy / 4.32 * 100)
            
            # Update telemetry
            self.multidrone_data['main']['time'].append(elapsed)
            self.multidrone_data['main']['pos'].append(current_pos.copy())
            self.multidrone_data['main']['energy'].append(energy)
            self.multidrone_data['main']['battery'].append(battery)
            self.multidrone_data['main']['speed'].append(speed)
            
            try:
                self.md_main_pos.config(text=f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f})")
                self.md_main_dist.config(text=f"Distance to Goal: {distance:.1f}m")
                self.md_main_battery.config(text=f"Battery: {battery:.1f}%")
                self.md_main_energy.config(text=f"Energy: {energy:.2f} Wh")
            except:
                pass
            
            if step % 20 == 0:
                print(f"🔴 D1: Pos({pos.x_val:.1f}, {pos.y_val:.1f}) Dist:{distance:.1f}m Spd:{speed:.1f}m/s Batt:{battery:.1f}%")
            
            step += 1
            time.sleep(dt)
            
    except Exception as e:
        print(f"❌ Main drone error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔴 Main drone flight ended")

def _fly_random_drones_multidrone_real(self):
    """Random drones flying in circular/roaming patterns in AirSim"""
    print("🟢 Random Drones (Drone2-5) starting roaming patterns...")
    try:
        client = self.md_client
        dt = 0.2
        step = 0
        
        # Takeoff all random drones
        drone_names = ["Drone2", "Drone3", "Drone4", "Drone5"]
        for drone_name in drone_names:
            safe_airsim_call(client.enableApiControl, True, vehicle_name=drone_name)
            safe_airsim_call(client.takeoffAsync, vehicle_name=drone_name).join()
            print(f"🛫 {drone_name} took off")
        
        # Random waypoints for each drone
        waypoints = {
            2: {'center_x': 50, 'center_y': 50, 'radius': 30},
            3: {'center_x': 100, 'center_y': 100, 'radius': 35},
            4: {'center_x': 75, 'center_y': 30, 'radius': 25},
            5: {'center_x': 25, 'center_y': 80, 'radius': 28}
        }
        
        while self.multidrone_active:
            # Control each random drone
            for drone_id in range(2, 6):
                try:
                    drone_name = f"Drone{drone_id}"
                    
                    state = safe_airsim_call(client.getMultirotorState, vehicle_name=drone_name)
                    if state is None:
                        continue
                    
                    pos = state.kinematics_estimated.position
                    current_pos = np.array([pos.x_val, pos.y_val], dtype=np.float32)
                    
                    # Store position
                    self.multidrone_paths[drone_id].append([pos.x_val, pos.y_val])
                    
                    # Circular roaming pattern
                    wp = waypoints[drone_id]
                    angle = (step * 0.05 + drone_id * 1.57) % (2 * np.pi)
                    target_x = wp['center_x'] + wp['radius'] * np.cos(angle)
                    target_y = wp['center_y'] + wp['radius'] * np.sin(angle)
                    target_z = -20.0
                    
                    # Move towards target
                    safe_airsim_call(client.moveToPositionAsync, target_x, target_y, target_z, 8.0, vehicle_name=drone_name)
                    
                    if step % 40 == 0:
                        print(f"🟢 {drone_name}: Pos({pos.x_val:.1f}, {pos.y_val:.1f})")
                    
                except Exception as e:
                    print(f"Error on {drone_name}: {e}")
            
            try:
                self.md_random_count.config(text=f"Active: 4/4")
            except:
                pass
            
            step += 1
            time.sleep(dt)
            
    except Exception as e:
        print(f"❌ Random drone error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🟢 Random drones stopped")
