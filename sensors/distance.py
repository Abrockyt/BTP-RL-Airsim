"""
Distance Sensor Module
Handles ultrasonic/laser distance measurements
"""

import airsim


class DistanceSensor:
    """Distance sensor for altitude/ground clearance measurement"""
    
    def __init__(self, client):
        """
        Initialize distance sensor
        
        Args:
            client: AirSim client connection
        """
        self.client = client
        
    def get_distance_data(self):
        """
        Get raw distance sensor data
        
        Returns:
            dict: Distance data with distance value, or None on error
        """
        try:
            distance_data = self.client.getDistanceSensorData()
            return distance_data
        except Exception as e:
            print(f"⚠️  Distance sensor read error: {type(e).__name__}")
            return None
    
    def get_distance(self):
        """
        Get distance to ground/nearest obstacle below drone (meters)
        
        Returns:
            float: Distance in meters, or -1.0 on error
        """
        try:
            distance_data = self.get_distance_data()
            
            if distance_data is None:
                return -1.0
            
            return distance_data.distance
            
        except Exception as e:
            print(f"⚠️  Distance read error: {type(e).__name__}")
            return -1.0
    
    def is_below_threshold(self, threshold=2.5):
        """
        Check if drone is below safe altitude threshold
        
        Args:
            threshold: Minimum safe altitude in meters (default: 2.5m)
            
        Returns:
            bool: True if below threshold, False otherwise
        """
        distance = self.get_distance()
        return distance >= 0 and distance < threshold
    
    def get_altitude_from_ground(self):
        """
        Get altitude above ground level
        
        Returns:
            float: Height above ground in meters, or -1.0 if not available
        """
        return self.get_distance()
