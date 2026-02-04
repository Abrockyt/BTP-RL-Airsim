"""
Magnetometer Sensor Module
Handles compass/magnetic field measurements
"""

import numpy as np
import airsim


class MagnetometerSensor:
    """Magnetometer sensor for heading/compass data"""
    
    def __init__(self, client):
        """
        Initialize magnetometer sensor
        
        Args:
            client: AirSim client connection
        """
        self.client = client
        
    def get_magnetometer_data(self):
        """
        Get raw magnetometer data
        
        Returns:
            dict: Magnetometer data with magnetic field in XYZ, or None on error
        """
        try:
            mag_data = self.client.getMagnetometerData()
            return mag_data
        except Exception as e:
            print(f"⚠️  Magnetometer read error: {type(e).__name__}")
            return None
    
    def get_magnetic_field(self):
        """
        Get magnetic field vector (Tesla)
        
        Returns:
            tuple: (bx, by, bz) in Tesla, or (0, 0, 0) on error
        """
        try:
            mag_data = self.get_magnetometer_data()
            
            if mag_data is None:
                return (0, 0, 0)
            
            mag_field = mag_data.magnetic_field_body
            return (mag_field.x_val, mag_field.y_val, mag_field.z_val)
            
        except Exception as e:
            print(f"⚠️  Magnetic field read error: {type(e).__name__}")
            return (0, 0, 0)
    
    def get_heading(self):
        """
        Calculate heading from magnetic field (degrees)
        
        Returns:
            float: Heading in degrees (0-360) from North, or -1.0 on error
        """
        try:
            bx, by, bz = self.get_magnetic_field()
            
            # Calculate heading from X and Y components
            # atan2(Y, X) gives angle from East axis
            heading_rad = np.arctan2(by, bx)
            
            # Convert to degrees (0-360)
            heading_deg = np.degrees(heading_rad)
            
            # Normalize to 0-360
            if heading_deg < 0:
                heading_deg += 360
            
            return heading_deg
            
        except Exception as e:
            print(f"⚠️  Heading calculation error: {type(e).__name__}")
            return -1.0
    
    def get_magnitude(self):
        """
        Get magnitude of magnetic field (Tesla)
        
        Returns:
            float: Magnetic field magnitude, or 0.0 on error
        """
        try:
            bx, by, bz = self.get_magnetic_field()
            magnitude = np.sqrt(bx**2 + by**2 + bz**2)
            return magnitude
            
        except Exception as e:
            print(f"⚠️  Magnitude calculation error: {type(e).__name__}")
            return 0.0
