# src/plate_detector.py
import cv2
import numpy as np

# Default configuration
DEFAULT_RESIZE_SCALE = 1.0
DEFAULT_PLATE_MIN_WIDTH = 30  # Adjusted for sensitivity
DEFAULT_PLATE_MAX_ASPECT = 7.0

class PlateDetector:
    def __init__(self, resize_scale=DEFAULT_RESIZE_SCALE, 
                 plate_min_width=DEFAULT_PLATE_MIN_WIDTH, 
                 plate_max_aspect=DEFAULT_PLATE_MAX_ASPECT):
        self.resize_scale = resize_scale
        self.plate_min_width = plate_min_width
        self.plate_max_aspect = plate_max_aspect

    def load_image(self, path):
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cv2.imread returned None (cannot read): {path}")
        return img

    def resize_image(self, img, scale=None, target=None):
        if scale is None:
            scale = self.resize_scale
        if target:
            return cv2.resize(img, target)
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)))

    def normalize_lighting(self, img_gray, kernel_size=20):
        # Morphological transformation to remove shadows/glare
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        img_norm = cv2.subtract(img_gray, opened)
        # Binarize
        _, img_norm_bw = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_norm, img_norm_bw

    def find_plate_candidates(self, img_norm_bw, min_width=None, max_aspect=None):
        if min_width is None:
            min_width = self.plate_min_width
        if max_aspect is None:
            max_aspect = self.plate_max_aspect
            
        # Get image dimensions to calculate percentages
        img_h, img_w = img_norm_bw.shape[:2]
            
        contours, _ = cv2.findContours(img_norm_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for i, cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = w / (h + 1e-6)
            
            # --- CRITICAL FIX IS HERE ---
            # 1. Min Width: Is it big enough to be a plate?
            # 2. Max Aspect: Is it too skinny/long?
            # 3. MAX WIDTH: Is it > 80% of the screen? (If so, it's the car body!)
            # 4. MAX HEIGHT: Is it > 60% of the screen? (If so, it's background)
            
            if (w >= min_width) and \
               (aspect_ratio <= max_aspect) and \
               (w < img_w * 0.80) and \
               (h < img_h * 0.60):
                   
                candidates.append((x,y,w,h,i))
                
        # Sort by Area (Largest first), but now the huge car body is gone
        candidates = sorted(candidates, key=lambda c: c[2]*c[3], reverse=True)
        return candidates, contours

    def crop_plate(self, img_gray, candidate):
        x,y,w,h,_ = candidate
        return img_gray[y:y+h, x:x+w], (x,y,w,h)

    def detect_plate(self, image_path):
        """Main method to detect plate in image"""
        img = self.load_image(image_path)
        
        # Resize logic
        img_resized = self.resize_image(img, self.resize_scale)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Normalize lighting and find plate
        img_norm, img_norm_bw = self.normalize_lighting(img_gray)
        candidates, contours = self.find_plate_candidates(img_norm_bw)
        
        if len(candidates) == 0:
            # Try one fallback: Make min_width smaller dynamically
            candidates, _ = self.find_plate_candidates(img_norm_bw, min_width=15)
            
        if len(candidates) == 0:
             # More helpful error message
            raise RuntimeError("No plate candidates found (Try adjusting min_width or checking lighting)")

        # Choose best candidate (First one is largest valid box)
        candidate = candidates[0]
        img_plate_gray, plate_bbox = self.crop_plate(img_gray, candidate)
        
        return {
            "plate_image_gray": img_plate_gray,
            "plate_bbox": plate_bbox,
            "full_image": img_resized,
            "candidates": candidates
        }