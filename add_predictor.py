import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

collision_predictor_code = """
import numpy as np

class CollisionPredictor:
    \"\"\"Predictive collision checker based on Time-To-Collision (TTC).\"\"\"

    def __init__(self, ttc_threshold=2.5, center_ratio=0.4, min_valid_depth=0.1, max_valid_depth=200.0):
        self.ttc_threshold = float(ttc_threshold)
        self.center_ratio = float(center_ratio)
        self.min_valid_depth = float(min_valid_depth)
        self.max_valid_depth = float(max_valid_depth)
        self.branch_distance = 6.5
        self.branch_ratio_threshold = 0.015
        self.min_side_clearance = 4.5

    def predict_collision(self, depth_image, current_speed):
        if depth_image is None or depth_image.ndim != 2 or depth_image.size == 0:
            return False, 999.0, 999.0, None

        h, w = depth_image.shape
        ch = max(1, int(h * self.center_ratio))
        cw = max(1, int(w * self.center_ratio))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2

        center = depth_image[y0:y0 + ch, x0:x0 + cw]
        valid_center = center[np.isfinite(center) & (center > self.min_valid_depth) & (center < self.max_valid_depth)]
        if valid_center.size == 0:
            return False, 999.0, 999.0, None

        avg_center_distance = float(np.percentile(valid_center, 25))
        near_ratio = float(np.mean(valid_center < self.branch_distance))
        speed = abs(float(current_speed))
        if speed <= 1e-6:
            return False, 999.0, avg_center_distance, None

        ttc_seconds = avg_center_distance / speed

        band = depth_image[h // 3:2 * h // 3, :]
        left = band[:, :w // 2]
        right = band[:, w // 2:]
        valid_left = left[np.isfinite(left) & (left > self.min_valid_depth) & (left < self.max_valid_depth)]
        valid_right = right[np.isfinite(right) & (right > self.min_valid_depth) & (right < self.max_valid_depth)]
        left_clear = float(np.percentile(valid_left, 60)) if valid_left.size > 0 else 0.0
        right_clear = float(np.percentile(valid_right, 60)) if valid_right.size > 0 else 0.0

        if left_clear < self.min_side_clearance and right_clear < self.min_side_clearance:
            avoid_dir = "UP"
        else:
            avoid_dir = "LEFT" if left_clear > right_clear else "RIGHT"

        branch_risk = near_ratio >= self.branch_ratio_threshold and avg_center_distance < (self.branch_distance + 2.0)
        crash_imminent = ttc_seconds < self.ttc_threshold or branch_risk
        return crash_imminent, ttc_seconds, avg_center_distance, avoid_dir
"""

text = text.replace('# =============================================================================\n# GUI CLASS', collision_predictor_code + '\n# =============================================================================\n# GUI CLASS')

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(text)
