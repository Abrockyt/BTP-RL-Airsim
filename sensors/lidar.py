"""
LiDAR Sensor Module
Handles 16-channel LiDAR collision detection and obstacle avoidance
"""

import numpy as np
import airsim


class LidarSensor:
    """16-channel LiDAR sensor for obstacle detection"""
    
    def __init__(self, client, lidar_name="Lidar1"):
        """
        Initialize LiDAR sensor
        
        Args:
            client: AirSim client connection
            lidar_name: Name of LiDAR sensor in settings (default: "Lidar1")
        """
        self.client = client
        self.lidar_name = lidar_name
        self.flight_level_z_min = -2.0  # Flight level altitude range
        self.flight_level_z_max = 2.0
        self.max_distance = 40.0  # 40m range from settings
        
    def get_raw_data(self):
        """
        Retrieve raw LiDAR point cloud
        
        Returns:
            dict: LiDAR data with point_cloud and other metadata, or None on error
        """
        try:
            lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
            return lidar_data
        except Exception as e:
            print(f"⚠️  LiDAR read error: {type(e).__name__}")
            return None
    
    def get_obstacles(self):
        """
        Get three-zone obstacle distances (Left, Center, Right)
        
        Returns:
            tuple: (left_dist, center_dist, right_dist, obstacle_level)
                - left_dist: Closest obstacle on left (-180° to -30°)
                - center_dist: Closest obstacle ahead (-30° to 30°)
                - right_dist: Closest obstacle on right (30° to 180°)
                - obstacle_level: 0=clear, 1=warning (6m), 2=danger (<3m)
        """
        try:
            lidar_data = self.get_raw_data()
            
            if lidar_data is None or len(lidar_data.point_cloud) == 0:
                # No LiDAR data - return safe defaults
                return (40.0, 40.0, 40.0, 0)
            
            # Convert point cloud to numpy array
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape((-1, 3))  # Nx3 array [X, Y, Z]
            
            if len(points) == 0:
                return (40.0, 40.0, 40.0, 0)
            
            # Filter to flight level (±2m altitude)
            flight_level_mask = (points[:, 2] > self.flight_level_z_min) & \
                               (points[:, 2] < self.flight_level_z_max)
            points = points[flight_level_mask]
            
            if len(points) == 0:
                return (40.0, 40.0, 40.0, 0)
            
            # Calculate distances and angles (XY plane only)
            distances = np.linalg.norm(points[:, :2], axis=1)
            angles = np.arctan2(points[:, 1], points[:, 0])  # Radians
            
            # Segment into three zones
            # Left: 30° to 180° (left half)
            left_mask = (angles > np.pi/6) & (angles <= np.pi)
            # Center: -30° to 30° (forward facing)
            center_mask = (angles >= -np.pi/6) & (angles <= np.pi/6)
            # Right: -180° to -30° (right half)
            right_mask = (angles < -np.pi/6) & (angles >= -np.pi)
            
            # Get minimum distance in each zone (closest obstacle)
            left_dist = np.min(distances[left_mask]) if np.any(left_mask) else self.max_distance
            center_dist = np.min(distances[center_mask]) if np.any(center_mask) else self.max_distance
            right_dist = np.min(distances[right_mask]) if np.any(right_mask) else self.max_distance
            
            # Determine obstacle level
            DANGER_THRESHOLD = 3.0   # Immediate danger
            WARNING_THRESHOLD = 6.0  # Start adjusting
            
            min_dist = min(left_dist, center_dist, right_dist)
            
            if min_dist < DANGER_THRESHOLD:
                obstacle_level = 2  # DANGER
            elif min_dist < WARNING_THRESHOLD:
                obstacle_level = 1  # WARNING
            else:
                obstacle_level = 0  # CLEAR
            
            return (left_dist, center_dist, right_dist, obstacle_level)
            
        except Exception as e:
            print(f"⚠️  Obstacle detection error: {type(e).__name__}")
            return (40.0, 40.0, 40.0, 0)
    
    def get_distance_matrix(self):
        """
        Get obstacle distances in all 16 sectors (one per LiDAR channel)
        
        Returns:
            list: 16 distances (one per channel), or None on error
        """
        try:
            lidar_data = self.get_raw_data()
            
            if lidar_data is None or len(lidar_data.point_cloud) == 0:
                return [self.max_distance] * 16
            
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape((-1, 3))
            
            if len(points) == 0:
                return [self.max_distance] * 16
            
            # Filter to flight level
            flight_level_mask = (points[:, 2] > self.flight_level_z_min) & \
                               (points[:, 2] < self.flight_level_z_max)
            points = points[flight_level_mask]
            
            if len(points) == 0:
                return [self.max_distance] * 16
            
            distances = np.linalg.norm(points[:, :2], axis=1)
            angles = np.arctan2(points[:, 1], points[:, 0])
            
            # Divide into 16 sectors (360° / 16 = 22.5° per sector)
            sector_distances = [self.max_distance] * 16
            sector_size = 2 * np.pi / 16
            
            for sector in range(16):
                sector_start = -np.pi + (sector * sector_size)
                sector_end = sector_start + sector_size
                
                # Handle wrapping around 180/-180
                if sector_start < -np.pi:
                    sector_start += 2 * np.pi
                if sector_end > np.pi:
                    sector_end -= 2 * np.pi
                
                # Find points in this sector
                if sector_start < sector_end:
                    mask = (angles >= sector_start) & (angles < sector_end)
                else:
                    mask = (angles >= sector_start) | (angles < sector_end)
                
                if np.any(mask):
                    sector_distances[sector] = np.min(distances[mask])
            
            return sector_distances
            
        except Exception as e:
            print(f"⚠️  Distance matrix error: {type(e).__name__}")
            return [self.max_distance] * 16
