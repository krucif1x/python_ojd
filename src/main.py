# src/main.py
import cv2
import os
import traceback
from src.plate_detection.plate_detector import PlateDetector
from src.plate_detection.paddle_recognizer import PaddlePlateRecognizer
from src.nlp.nlp_metadata_extractor import ManualLabelsPlateExtractor


class LicensePlatePipeline:
    """Full pipeline: Detection → OCR → NLP Metadata"""

    def __init__(self, nlp_model_path, labels_csv_path):
        print("→ Initializing Plate Detection...")
        self.detector = PlateDetector(plate_min_width=20, plate_max_aspect=10.0)

        print("→ Initializing PaddleOCR...")
        self.ocr = PaddlePlateRecognizer(use_gpu=False)

        print("→ Loading NLP Metadata Model...")
        self.nlp = ManualLabelsPlateExtractor(manual_labels_path=labels_csv_path)
        self.nlp.load_model(nlp_model_path)

    def process(self, image_path):
        """Runs the full LPR pipeline on an image."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # -------------------- 1. DETECTION --------------------
        print("\n[1] Detecting License Plate...")
        try:
            det = self.detector.detect_plate(image_path)
        except RuntimeError:
            print("⚠️ No plate detected.")
            return None

        gray_crop = det["plate_image_gray"]
        os.makedirs("../outputs", exist_ok=True)
        cv2.imwrite("../outputs/debug_crop.jpg", gray_crop)
        print("   ✔ Plate cropped (saved as debug_crop.jpg)")

        # -------------------- 2. OCR --------------------
        print("[2] Recognizing Characters (OCR)...")
        ocr_result = self.ocr.recognize_characters(gray_crop)
        plate_text = ocr_result["plate_string"]

        # -------------------- 3. NLP METADATA --------------------
        print(f"[3] Extracting Metadata for '{plate_text}'...")
        nlp_result = self.nlp.predict(plate_text)

        # -------------------- FINAL OUTPUT --------------------
        return {
            "plate": plate_text,
            "nlp": nlp_result,
            "annotated": ocr_result["annotated_image"]
        }


def main():
    IMAGE_PATH = "/home/pcsistem/Documents/carlo_ojd_ml/tests/test images/AA5627JT.jpg"
    NLP_MODEL = "/home/pcsistem/Documents/carlo_ojd_ml/src/plate_model_manual_only.pkl"
    LABELS_CSV = "/home/pcsistem/Documents/carlo_ojd_ml/plate_text_dataset/manual_labels.csv"

    try:
        print("=" * 55)
        print("      LICENSE PLATE PIPELINE (OCR + NLP)")
        print("=" * 55)

        pipeline = LicensePlatePipeline(NLP_MODEL, LABELS_CSV)
        result = pipeline.process(IMAGE_PATH)

        if not result:
            print("❌ Pipeline stopped — no plate detected.")
            return

        meta = result["nlp"]

        # Print Results
        print("\n================= ANALYSIS RESULT =================")
        print(f"Detected Plate : {result['plate']}")
        print(f"Vehicle Origin : {meta['province']}, {meta['region']}")
        print(f"Vehicle Type   : {meta['vehicle']}")
        print(f"Confidence     : {meta['confidence']:.2%}")
        print(f"Status         : {meta['status']}")
        print("===================================================\n")

        # Save annotated output
        output_path = "../outputs/annotated_result.jpg"
        os.makedirs("../outputs", exist_ok=True)
        cv2.imwrite(output_path, result["annotated"])
        print(f"✔ Annotated image saved to: {output_path}")

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
