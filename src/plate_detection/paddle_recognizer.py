# src/paddle_recognizer.py
from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging

# Suppress PaddleOCR's massive logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

class PaddlePlateRecognizer:
    def __init__(self, use_gpu=False):
        print("   [Init] Loading PaddleOCR (English/Number mode)...")
        
        # Map bool to device string
        target_device = 'gpu' if use_gpu else 'cpu'
        
        # Initialize with v5 compatible settings
        self.ocr = PaddleOCR(
            use_textline_orientation=False, 
            lang='en',
            device=target_device
        )

    def recognize_characters(self, plate_image):
        # Ensure RGB
        if len(plate_image.shape) == 2:
            img_rgb = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = plate_image

        # Run OCR (Try/Catch for API compatibility)
        try:
            # New Paddle requires list conversion for the generator
            raw_result = list(self.ocr.ocr(img_rgb, cls=False))
        except TypeError:
            # Fallback if cls argument is removed
            raw_result = list(self.ocr.ocr(img_rgb))

        full_text = ""
        all_boxes = []
        
        # --- ROBUST PARSING LOGIC ---
        
        # Check if we got any result
        if not raw_result or raw_result[0] is None:
             return {"plate_string": "", "annotated_image": img_rgb}

        # Unwrap the first result (since we processed 1 image)
        first_result = raw_result[0]

        # CASE A: Dictionary Format (PaddleX / v5 Pipeline)
        # The log showed: {'rec_texts': [...], 'dt_polys': [...]}
        if isinstance(first_result, dict):
            texts = first_result.get('rec_texts', [])
            scores = first_result.get('rec_scores', [])
            boxes = first_result.get('dt_polys', [])
            
            for i in range(len(texts)):
                # Check confidence (scores might be missing or different length)
                conf = scores[i] if i < len(scores) else 1.0
                
                if conf > 0.5:
                    full_text += texts[i]
                    # Boxes might be numpy arrays in this format
                    if i < len(boxes):
                        all_boxes.append(boxes[i])

        # CASE B: Standard List Format (Old Paddle / v3 / v4)
        # Format: [ [box, (text, conf)], ... ]
        elif isinstance(first_result, list):
            for line in first_result:
                # Ensure line has the expected structure [box, (text, conf)]
                if len(line) >= 2 and isinstance(line[0], list):
                    box = line[0]
                    # Extract text/conf
                    content = line[1]
                    if isinstance(content, tuple) or isinstance(content, list):
                        text = content[0]
                        conf = content[1]
                    else:
                        text = content
                        conf = 1.0
                    
                    if conf > 0.5:
                        full_text += text
                        all_boxes.append(box)

        # --- CLEANUP & RETURN ---
        
        # Keep Alphanumeric Only (Removes spaces, dots, dashes)
        clean_text = ''.join(e for e in full_text if e.isalnum()).upper()

        # Visualization
        annotated = img_rgb.copy()
        for box in all_boxes:
            # Reshape box to standard polylines format
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [box], True, (0, 255, 0), 2)

        return {
            "plate_string": clean_text,
            "annotated_image": annotated
        }