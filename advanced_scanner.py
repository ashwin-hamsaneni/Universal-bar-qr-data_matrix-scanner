import cv2
import zxingcpp
import numpy as np
from PIL import Image

class AdvancedDataMatrixDecoder:
    def __init__(self):
        # Configuration for preprocessing
        self.target_width = 600
        self.quiet_zone_pad = 40
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Highly effective for PCBs and shadowing
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def _pil_high_quality_resize(self, gray_matrix):
        """
        Uses Pillow (PIL) for high-fidelity resizing using the LANCZOS algorithm.
        LANCZOS is mathematically superior to standard OpenCV resizing for preserving
        the crisp edges of matrix modules during upscaling/downscaling.
        """
        h, w = gray_matrix.shape
        if w == self.target_width:
            return gray_matrix
            
        scale = self.target_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Convert NumPy array to PIL Image
        pil_img = Image.fromarray(gray_matrix)
        
        # Resize using LANCZOS (High-quality anti-aliasing)
        # Note: In newer Pillow versions, ANTIALIAS is replaced by LANCZOS or Resampling.LANCZOS
        try:
            pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        except AttributeError:
            pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
            
        # Convert back to NumPy array
        return np.array(pil_resized)

    def _advanced_preprocessing(self, gray_img):
        """
        The ultimate pipeline for PCBs, shadows, and laser-engraved (Dot-Peen) codes.
        """
        # 1. High-fidelity Pillow Resize
        resized = self._pil_high_quality_resize(gray_img)
        
        # 2. CLAHE: Equalize shadows (Crucial for PCBs and uneven lighting)
        equalized = self.clahe.apply(resized)
        
        # 3. Gaussian Blur: Remove PCB trace lines and sensor grain
        blur = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        # 4. Adaptive Thresholding: Separate code from background dynamically
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 12)
        
        # 5. Morphological Close: Fill hollow laser rings (Dot-Peen)
        kernel_close = np.ones((3, 3), np.uint8)
        closed_dots = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # 6. Dilation: Bridge gaps between disconnected dots
        kernel_dilate = np.ones((5, 5), np.uint8)
        connected_dots = cv2.dilate(closed_dots, kernel_dilate, iterations=1)
        
        # 7. Enforce Quiet Zone (Wipe edges clean of fake shadows)
        p = self.quiet_zone_pad
        connected_dots[:p, :] = 0
        connected_dots[-p:, :] = 0
        connected_dots[:, :p] = 0
        connected_dots[:, -p:] = 0
        
        # 8. Invert back to standard (Black code on White background)
        processed_final = cv2.bitwise_not(connected_dots)
        
        return processed_final

    def _draw_dynamic_results(self, display_img, results, mode_name, scale_factor=1.0, pad=0):
        """
        Automatically draws bounding boxes and text dynamically, regardless of code size/location.
        """
        for i, obj in enumerate(results):
            raw_format = str(obj.format).split('.')[-1].upper().replace("_", "")
            data_text = obj.text
            
            # Determine color based on format
            color = (255, 0, 0) if "DATAMATRIX" in raw_format else (0, 255, 0)
            label = f"{raw_format} ({mode_name})"
            
            # Extract points, adjust for any padding/scaling done during preprocessing
            pos = obj.position
            pts = np.array([
                [(pos.top_left.x - pad) / scale_factor, (pos.top_left.y - pad) / scale_factor],
                [(pos.top_right.x - pad) / scale_factor, (pos.top_right.y - pad) / scale_factor],
                [(pos.bottom_right.x - pad) / scale_factor, (pos.bottom_right.y - pad) / scale_factor],
                [(pos.bottom_left.x - pad) / scale_factor, (pos.bottom_left.y - pad) / scale_factor]
            ], dtype=np.int32)
            
            # 1. Draw dynamic polygon around the code
            cv2.polylines(display_img, [pts], True, color, 3)
            
            # 2. Dynamic Text Placement (Always stays above the code)
            text_x = max(10, int(pts[0][0]))
            text_y = max(30, int(pts[0][1]) - 10)
            
            # Draw text background for readability
            cv2.rectangle(display_img, (text_x, text_y - 25), (text_x + 200, text_y + 15), (0, 0, 0), -1)
            
            # Draw Mode/Format and Data Text
            cv2.putText(display_img, label, (text_x + 5, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(display_img, data_text[:20], (text_x + 5, text_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            print(f"[SUCCESS] decoded '{data_text}' via {mode_name}")
            
        return display_img

    def scan_image(self, image_path):
        print(f"\n--- Scanning Image: {image_path} ---")
        
        # 1. Load Raw Image (BGR)
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            print("[ERROR] Could not read image path.")
            return

        display_img = raw_img.copy()

        # ==========================================
        # PASS 1: Attempt Raw Image Decoding
        # ==========================================
        results = zxingcpp.read_barcodes(raw_img)
        if results:
            self._draw_dynamic_results(display_img, results, "Pass 1: RAW")
            self._show_result(display_img)
            return

        # ==========================================
        # PASS 2: Attempt Grayscale Decoding
        # ==========================================
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        # We pad the grayscale image to give ZXing the required whitespace
        p = self.quiet_zone_pad
        gray_padded = cv2.copyMakeBorder(gray_img, p, p, p, p, cv2.BORDER_CONSTANT, value=255)
        
        results = zxingcpp.read_barcodes(gray_padded)
        if results:
            self._draw_dynamic_results(display_img, results, "Pass 2: GRAY", pad=p)
            self._show_result(display_img)
            return

        # ==========================================
        # PASS 3: Advanced Preprocessing (PCBs/Shadows/DPM)
        # ==========================================
        print("[INFO] Passes 1 & 2 failed. Engaging Advanced Preprocessing Pipeline...")
        
        # First, pad the image *before* processing to protect edges from shadow bleeding
        gray_padded_for_prep = cv2.copyMakeBorder(gray_img, p, p, p, p, cv2.BORDER_CONSTANT, value=255)
        
        # Calculate scale factor since Pillow will resize it
        h, w = gray_padded_for_prep.shape
        scale_factor = self.target_width / w if w > self.target_width else 1.0
        
        processed_img = self._advanced_preprocessing(gray_padded_for_prep)
        
        results = zxingcpp.read_barcodes(processed_img)
        if results:
            self._draw_dynamic_results(display_img, results, "Pass 3: PREPROCESSED", scale_factor=scale_factor, pad=p)
        else:
            print("[FAILED] All decoding passes failed.")
            cv2.putText(display_img, "DECODE FAILED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self._show_result(display_img)

    def _show_result(self, image):
        cv2.imshow("Advanced PCB & DPM Scanner", image)
        print("Press any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    scanner = AdvancedDataMatrixDecoder()

    test_image = r"C:\Users\HP\Downloads\WhatsApp Image 2026-06-26 at 11.40.38 AM.jpeg"
    scanner.scan_image(test_image)