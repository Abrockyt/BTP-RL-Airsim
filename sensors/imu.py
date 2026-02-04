"""
IMU Sensor Module
Handles inertial measurement unit data (acceleration, angular velocity)
"""

import numpy as np
import airsim


class IMUSensor:
    """Inertial Measurement Unit sensor"""
    
    def __init__(self, client):
        """
        Initialize IMU sensor
        
        Args:
            client: AirSim client connection
        """
        self.client = client
        
    def get_imu_data(self):
        """
        Get raw IMU data
        
        Returns:
            dict: IMU data with linear_acceleration and angular_velocity, or None on error
        """
        try:
            imu_data = self.client.getImuData()
            return imu_data
        except Exception as e:
            print(f"⚠️  IMU read error: {type(e).__name__}")
            return None
    
    def get_acceleration(self):
        """
        Get linear acceleration (m/s²) in XYZ axes
        
        Returns:
            tuple: (ax, ay, az) or (0, 0, 0) on error
        """
        try:
            imu_data = self.get_imu_data()
            
            if imu_data is None:
                return (0, 0, 0)
            
            acc = imu_data.linear_acceleration
            return (acc.x_val, acc.y_val, acc.z_val)
            
        except Exception as e:
            print(f"⚠️  Acceleration read error: {type(e).__name__}")
            return (0, 0, 0)
    
    def get_angular_velocity(self):
        """
        Get angular velocity (rad/s) in roll, pitch, yaw rates
        
        Returns:
            tuple: (roll_rate, pitch_rate, yaw_rate) or (0, 0, 0) on error
        """
        try:
            imu_data = self.get_imu_data()
            
            if imu_data is None:
                return (0, 0, 0)
            
            ang_vel = imu_data.angular_velocity
            return (ang_vel.x_val, ang_vel.y_val, ang_vel.z_val)
            
        except Exception as e:
            print(f"⚠️  Angular velocity read error: {type(e).__name__}")
            return (0, 0, 0)
    
    def get_orientation(self):
        """
        Get current orientation as quaternion
        
        Returns:
            tuple: (qw, qx, qy, qz) or None on error
        """
        try:
            imu_data = self.get_imu_data()
            
            if imu_data is None:
                return None
            
            orientation = imu_data.orientation
            return (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
            
        except Exception as e:
            print(f"⚠️  Orientation read error: {type(e).__name__}")
            return None
    
    def get_euler_angles(self):
        """
        Get current orientation as Euler angles (roll, pitch, yaw in radians)
        
        Returns:
            tuple: (roll, pitch, yaw) in radians, or (0, 0, 0) on error
        """
        try:
            imu_data = self.get_imu_data()
            
            if imu_data is None:
                return (0, 0, 0)
            
            euler = airsim.to_eularian_angles(imu_data.orientation)
            return euler
            
        except Exception as e:
            print(f"⚠️  Euler angle conversion error: {type(e).__name__}")
            return (0, 0, 0)
