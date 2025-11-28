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

    def _show_and_wait(self, image, title="preview"):
        """Show image in a window and wait until user closes it or presses any key."""
        if image is None:
            return

        # try to create/show window; if fail (no GUI), warn and continue
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, image)
        except cv2.error:
            print(f"⚠️ Unable to create/show window '{title}'. Continuing without GUI.")
            return

        try:
            while True:
                key = cv2.waitKey(100)
                if key != -1:
                    break
                try:
                    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # window property unavailable or window closed
                    break
        finally:
            # destroy window safely; fall back to destroyAllWindows on error
            try:
                cv2.destroyWindow(title)
            except cv2.error:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def _interactive_manual_crop(self, image, window_name="Manual Crop"):
        """
        Let user draw a rectangle with mouse to crop. Controls:
         - Left mouse drag: draw rectangle
         - 'c' key: confirm crop and return cropped BGR image
         - 'r' key: reset selection
         - ESC key: cancel and return None
        """
        if image is None:
            return None

        clone = image.copy()
        start = [(-1, -1)]
        end = [(-1, -1)]
        roi = [None]
        drawing = {"on": False}

        def _mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing["on"] = True
                start[0] = (x, y)
                end[0] = (x, y)
                roi[0] = None
            elif event == cv2.EVENT_MOUSEMOVE and drawing["on"]:
                end[0] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing["on"] = False
                end[0] = (x, y)
                roi[0] = (start[0], end[0])

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, _mouse_cb)

            while True:
                disp = clone.copy()
                if start[0] != (-1, -1) and end[0] != (-1, -1):
                    cv2.rectangle(disp, start[0], end[0], (0, 255, 0), 2)
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass
                    return None
                if key == ord("r"):
                    start[0] = (-1, -1)
                    end[0] = (-1, -1)
                    roi[0] = None
                if key == ord("c") and roi[0] is not None:
                    (x1, y1), (x2, y2) = roi[0]
                    x1, x2 = sorted((max(0, x1), max(0, x2)))
                    y1, y2 = sorted((max(0, y1), max(0, y2)))
                    x2 = min(x2, clone.shape[1])
                    y2 = min(y2, clone.shape[0])
                    if x2 - x1 <= 5 or y2 - y1 <= 5:
                        print("⚠️ Selection too small, try again.")
                        continue
                    crop = clone[y1:y2, x1:x2].copy()
                    try:
                        cv2.destroyWindow(window_name)
                    except Exception:
                        pass
                    return crop
        except cv2.error:
            print("⚠️ GUI unavailable for manual crop.")
            return None

    def _preprocess_plate(self, plate_bgr):
        """
        Minimal preprocessing for PaddleOCR:
         - ensure 3-channel BGR (PaddleOCR expects color images)
         - resize to a reasonable height (keeps aspect ratio)
        Avoid aggressive binarization or heavy morphological ops that remove useful detail.
        Returns a 3-channel BGR image ready for OCR.
        """
        if plate_bgr is None:
            return None

        # ensure 3-channel BGR
        if len(plate_bgr.shape) == 2 or (len(plate_bgr.shape) == 3 and plate_bgr.shape[2] == 1):
            plate_bgr = cv2.cvtColor(plate_bgr, cv2.COLOR_GRAY2BGR)

        # Resize to a moderate height for OCR models (adjust if needed)
        h_target = 96  # try values between 64-128 depending on plate resolution
        h, w = plate_bgr.shape[:2]
        if h > 0 and h != h_target:
            scale = h_target / h
            new_w = max(16, int(w * scale))
            proc_bgr = cv2.resize(plate_bgr, (new_w, h_target), interpolation=cv2.INTER_LINEAR)
        else:
            proc_bgr = plate_bgr.copy()

        return proc_bgr

    def _annotate_with_nlp(self, image, nlp_result, plate_text):
        """Overlay NLP metadata (province, region, vehicle, confidence, status) onto the image."""
        if image is None:
            return None

        # ensure BGR color image
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Prepare text lines
        conf = nlp_result.get("confidence", 0.0)
        lines = [
            f"Plate: {plate_text}",
            f"Province: {nlp_result.get('province', '-')}",
            f"Region: {nlp_result.get('region', '-')}",
            f"Vehicle: {nlp_result.get('vehicle', '-')}",
            f"Confidence: {conf:.2%}",
            f"Status: {nlp_result.get('status', '-')}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        line_spacing = 8
        margin = 8

        # compute background rectangle size
        widths_heights = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        max_w = max(w for (w, h) in widths_heights)
        total_h = sum(h for (w, h) in widths_heights) + (len(lines) - 1) * line_spacing

        x1, y1 = 10, 10
        x2 = x1 + max_w + margin * 2
        y2 = y1 + total_h + margin * 2

        # Draw semi-opaque background
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Put text lines
        y = y1 + margin
        for idx, line in enumerate(lines):
            (w, h) = widths_heights[idx]
            text_org = (x1 + margin, y + h)
            cv2.putText(image, line, text_org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += h + line_spacing

        return image

    def process(self, image_path, interactive=False, manual_crop=False):
        """Runs the full LPR pipeline on an image.
           If interactive=True show windows between stages.
           If manual_crop=True prompt user to manually crop the plate first.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if interactive:
            print("→ Stage 0: Original image (close window or press any key to continue)")
            self._show_and_wait(img, "Original Image")

        # If user wants manual crop, prompt and preprocess that crop
        if manual_crop and interactive:
            print("\n[0.5] Manual cropping: draw rectangle, press 'c' to confirm, 'r' to reset, ESC to cancel.")
            crop_bgr = self._interactive_manual_crop(img, "Manual Crop")
            if crop_bgr is None:
                print("⚠️ Manual cropping cancelled.")
                return None

            proc_bgr = self._preprocess_plate(crop_bgr)
            # save debug color crop
            os.makedirs("../outputs", exist_ok=True)
            cv2.imwrite("../outputs/debug_manual_crop_bgr.jpg", proc_bgr)
            print("   ✔ Manual crop saved to ../outputs/debug_manual_crop_bgr.jpg")

            if interactive:
                print("→ Stage 1: Cropped + preprocessed (close window or press any key to continue)")
                self._show_and_wait(proc_bgr, "Cropped + Preprocessed")

        else:
            # -------------------- 1. DETECTION --------------------
            print("\n[1] Detecting License Plate...")
            try:
                det = self.detector.detect_plate(image_path)
            except RuntimeError:
                print("⚠️ No plate detected.")
                return None

            # prefer a BGR crop if detector provides one, else convert gray->BGR
            if "plate_image_bgr" in det and det["plate_image_bgr"] is not None:
                crop_bgr = det["plate_image_bgr"]
            elif "plate_image" in det and det["plate_image"] is not None:
                crop_bgr = det["plate_image"]
            else:
                # fallback: detector provided grayscale
                gray = det.get("plate_image_gray")
                if gray is None:
                    raise RuntimeError("Detector did not return a valid plate image.")
                crop_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            proc_bgr = self._preprocess_plate(crop_bgr)
            os.makedirs("../outputs", exist_ok=True)
            cv2.imwrite("../outputs/debug_crop_bgr.jpg", proc_bgr)
            print("   ✔ Plate cropped (saved as debug_crop_bgr.jpg)")

            if interactive:
                print("→ Stage 1: Cropped plate (close window or press any key to continue)")
                self._show_and_wait(proc_bgr, "Cropped Plate (BGR)")

        # -------------------- 2. OCR --------------------
        print("[2] Recognizing Characters (OCR)...")
        # feed the color BGR preprocessed image to PaddleOCR (avoid binary images)
        ocr_result = self.ocr.recognize_characters(proc_bgr)
        plate_text = ocr_result["plate_string"]

        if interactive:
            print("→ Stage 2: OCR annotated (close window or press any key to continue)")
            self._show_and_wait(ocr_result.get("annotated_image", None), "OCR Annotated")

        # -------------------- 3. NLP METADATA --------------------
        print(f"[3] Extracting Metadata for '{plate_text}'...")
        nlp_result = self.nlp.predict(plate_text)

        # annotate the OCR image with NLP metadata
        annotated_with_nlp = self._annotate_with_nlp(ocr_result.get("annotated_image"), nlp_result, plate_text)

        # -------------------- FINAL OUTPUT --------------------
        return {
            "plate": plate_text,
            "nlp": nlp_result,
            "annotated": annotated_with_nlp
        }


def main():
    IMAGE_PATH = r"C:\Users\Bernardo Carlo\Documents\python_ojd\test images\D1783TJ.jpeg"
    NLP_MODEL = r"C:\Users\Bernardo Carlo\Documents\python_ojd\model\plate_model_manual_only.pkl"
    LABELS_CSV = r"C:\Users\Bernardo Carlo\Documents\python_ojd\data\manual_labels.csv"

    try:
        print("=" * 55)
        print("      LICENSE PLATE PIPELINE (OCR + NLP)")
        print("=" * 55)

        pipeline = LicensePlatePipeline(NLP_MODEL, LABELS_CSV)
        # set interactive=True to display step windows and require user action to continue
        # enable manual_crop=True to draw the plate region yourself before preprocessing+OCR
        result = pipeline.process(IMAGE_PATH, interactive=True, manual_crop=True)

        if not result:
            print("❌ Pipeline stopped — no plate detected or cancelled.")
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
