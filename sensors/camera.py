"""
Camera Sensor Module
Handles RGB camera capture and image preprocessing
"""

import numpy as np
import airsim


class CameraSensor:
    """RGB camera sensor for visual navigation"""
    
    def __init__(self, client, camera_name="0"):
        """
        Initialize camera sensor
        
        Args:
            client: AirSim client connection
            camera_name: Name of camera in settings (default: "0")
        """
        self.client = client
        self.camera_name = camera_name
        self.image_width = 1920
        self.image_height = 1080
        
    def capture_rgb(self):
        """
        Capture RGB image from camera
        
        Returns:
            np.ndarray: RGB image array (H x W x 3) or None on error
        """
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
            ])
            
            if not responses or len(responses) == 0:
                return None
            
            response = responses[0]
            
            if response.image_data_uint8 is None or len(response.image_data_uint8) == 0:
                return None
            
            # Convert to numpy array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            
            # Reshape to 3D (height x width x 4 channels - RGBA)
            img_rgba = img1d.reshape((response.height, response.width, 4))
            
            # Extract RGB only (drop alpha channel)
            img_rgb = img_rgba[:, :, :3]
            
            return img_rgb
            
        except Exception as e:
            print(f"⚠️  Camera capture error: {type(e).__name__}")
            return None
    
    def capture_depth(self):
        """
        Capture depth image for obstacle detection
        
        Returns:
            np.ndarray: Depth image array (H x W) or None on error
        """
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPlanar, True, False)
            ])
            
            if not responses or len(responses) == 0:
                return None
            
            response = responses[0]
            
            if response.image_data_float is None or len(response.image_data_float) == 0:
                return None
            
            # Convert to numpy array
            depth = np.array(response.image_data_float, dtype=np.float32)
            
            # Reshape to 2D
            depth = depth.reshape((response.height, response.width))
            
            return depth
            
        except Exception as e:
            print(f"⚠️  Depth capture error: {type(e).__name__}")
            return None
    
    def get_zone_depths(self):
        """
        Get depth values for center, left, right zones
        
        Returns:
            tuple: (center_depth, left_depth, right_depth) or (inf, inf, inf) on error
        """
        try:
            depth = self.capture_depth()
            
            if depth is None or depth.size == 0:
                return (float('inf'), float('inf'), float('inf'))
            
            h, w = depth.shape
            
            # Define zones (thirds of image)
            center_depth = np.nanmean(depth[h//3:2*h//3, w//3:2*w//3])
            left_depth = np.nanmean(depth[h//3:2*h//3, :w//3])
            right_depth = np.nanmean(depth[h//3:2*h//3, 2*w//3:])
            
            # Handle NaN values
            if np.isnan(center_depth):
                center_depth = float('inf')
            if np.isnan(left_depth):
                left_depth = float('inf')
            if np.isnan(right_depth):
                right_depth = float('inf')
            
            return (center_depth, left_depth, right_depth)
            
        except Exception as e:
            print(f"⚠️  Zone depth error: {type(e).__name__}")
            return (float('inf'), float('inf'), float('inf'))
