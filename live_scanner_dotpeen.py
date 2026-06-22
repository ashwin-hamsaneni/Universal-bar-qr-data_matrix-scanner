import cv2
import zxingcpp
import numpy as np
import time

def start_dotpeen_scanner():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("\n=======================================================")
    print(" ADVANCED DOT-PEEN LIVE SCANNER ")
    print("=======================================================")
    print(">> Hold up your printed codes OR the Dot-Peen image.")
    print(">> Watch the 'Mode' in the terminal switch dynamically!")
    print(">> Press the 'q' key on your keyboard to quit.")
    print("-" * 55)

    # PERFORMANCE FIX: Define the window ONCE outside the loop
    window_name = "Advanced Dot-Peen Live Scanner (Press 'q' to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_scanned = ""

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to standard grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # PASS 1: Try reading standard printed codes first (Fastest)
        results = zxingcpp.read_barcodes(gray)
        scan_mode = "STANDARD"

        # PASS 2: If standard fails, activate Heavy Dot-Peen Processing
        if not results:
            # 1. Blur out the bumpy background texture
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 2. Adaptive Thresholding: Turn dots pure white, background pure black
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 15, 6)
            
            # 3. Dilation: Expand the white dots until they crash into each other
            kernel = np.ones((5, 5), np.uint8) 
            connected_dots = cv2.dilate(thresh, kernel, iterations=1)
            
            # 4. Invert: Swap back to black dots on white background for the engine
            dpm_processed = cv2.bitwise_not(connected_dots)
            
            # OPTIMIZATION: Tell zxingcpp to focus heavily on Data Matrix for DPM pass
            # This avoids false positives and speeds up processing time.
            dpm_formats = [zxingcpp.BarcodeFormat.DataMatrix, zxingcpp.BarcodeFormat.QRCode]
            results = zxingcpp.read_barcodes(dpm_processed, formats=dpm_formats)
            scan_mode = "DOT-PEEN"

        latency_ms = (time.time() - start_time) * 1000

        # Process detected targets
        for obj in results:
            # SAFETY CHECK: Skip if zxing returned a partial/broken position structure
            if not obj.position or not hasattr(obj.position, 'top_left'):
                continue

            data_text = obj.text
            code_type = str(obj.format).split('.')[-1].upper().replace("_", "").replace(" ", "")
            pos = obj.position
            
            if data_text != last_scanned:
                print(f"[DETECTED] {code_type:<12} | Mode: {scan_mode:<10} | Latency: {latency_ms:.1f}ms | Data: {data_text}")
                last_scanned = data_text

            try:
                # Bounding box coordinates safely parsed
                pts = np.array([[int(pos.top_left.x), int(pos.top_left.y)],
                                [int(pos.top_right.x), int(pos.top_right.y)],
                                [int(pos.bottom_right.x), int(pos.bottom_right.y)],
                                [int(pos.bottom_left.x), int(pos.bottom_left.y)]], dtype=np.int32)
                
                # Box Colors
                if "DATAMATRIX" in code_type:
                    box_color = (255, 0, 0) # Blue for Data Matrix
                    display_label = f"DATA MATRIX ({scan_mode})"
                else:
                    box_color = (0, 255, 0) # Green for others
                    display_label = f"{code_type} ({scan_mode})"
                    
                # Draw overlays
                cv2.polylines(frame, [pts], True, box_color, 3)
                
                text_y = max(30, int(pos.top_left.y) - 10)
                cv2.putText(frame, display_label, (int(pos.top_left.x), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)
                cv2.putText(frame, data_text, (int(pos.top_left.x), text_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                # Catch-all to prevent a single rendering error from crashing the live camera
                continue

        # REMOVED: Live Latency text overlay function was here
        
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_dotpeen_scanner()