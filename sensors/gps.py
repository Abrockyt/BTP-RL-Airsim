"""
GPS Sensor Module
Handles Global Positioning System data
"""

import airsim


class GPSSensor:
    """GPS sensor for global positioning"""
    
    def __init__(self, client):
        """
        Initialize GPS sensor
        
        Args:
            client: AirSim client connection
        """
        self.client = client
        
    def get_gps_data(self):
        """
        Get raw GPS data
        
        Returns:
            dict: GPS data with latitude, longitude, altitude, or None on error
        """
        try:
            gps_data = self.client.getGpsData()
            return gps_data
        except Exception as e:
            print(f"⚠️  GPS read error: {type(e).__name__}")
            return None
    
    def get_position(self):
        """
        Get current GPS position (latitude, longitude, altitude)
        
        Returns:
            tuple: (latitude, longitude, altitude) or (0, 0, 0) on error
        """
        try:
            gps_data = self.get_gps_data()
            
            if gps_data is None:
                return (0, 0, 0)
            
            return (gps_data.latitude, gps_data.longitude, gps_data.altitude)
            
        except Exception as e:
            print(f"⚠️  Position read error: {type(e).__name__}")
            return (0, 0, 0)
    
    def get_accuracy(self):
        """
        Get GPS accuracy metrics
        
        Returns:
            dict: Accuracy data with eph (horizontal), epv (vertical), etc., or None on error
        """
        try:
            gps_data = self.get_gps_data()
            
            if gps_data is None:
                return None
            
            return {
                'eph': gps_data.eph,  # Horizontal accuracy
                'epv': gps_data.epv,  # Vertical accuracy
                'hdop': gps_data.hdop,  # Horizontal dilution of precision
                'vdop': gps_data.vdop  # Vertical dilution of precision
            }
            
        except Exception as e:
            print(f"⚠️  Accuracy read error: {type(e).__name__}")
            return None
