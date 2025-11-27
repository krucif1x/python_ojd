# tests/test_paddle.py
from paddleocr import PaddleOCR
import logging

# Suppress logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

def test_inference():
    print("🚀 Initializing PaddleOCR...")
    
    # Initialize with orientation disabled (this replaces the old cls=False logic)
    ocr = PaddleOCR(use_textline_orientation=False, lang='en') 
    
    img_path = "/home/pcsistem/Documents/carlo_ojd_ml/test images/B3023KEZ.jpg"
    print(f"📂 Loading image: {img_path}")
    
    # FIX: Removed 'cls=False' argument here
    # The new API creates a generator, so we convert it to a list
    result = list(ocr.ocr(img_path))
    
    if not result:
        print("❌ No text found!")
        return

    print("\n✅ Raw Results:")
    # In v5, the structure might be slightly different, usually a list of dicts or tuples
    # We iterate carefully
    for idx, line in enumerate(result):
        # Standard Paddle format is usually a list of lines inside the first element
        # But v5 pipelines sometimes return just the lines directly.
        # Let's inspect the first item to be safe.
        if isinstance(line, list):
             for sub_line in line:
                 print(f"   Text: {sub_line[1][0]}  (Conf: {sub_line[1][1]:.2f})")
        else:
             # Direct format
             print(f"   Raw Line: {line}")

if __name__ == "__main__":
    test_inference()