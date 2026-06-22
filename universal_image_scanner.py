import cv2
import zxingcpp
import numpy as np

def scan_universal_image(image_path):
    print(f"\nScanning Image: {image_path}")
    print("-" * 50)
    
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image. Please check the file path.")
        return

    # Normalize image size for predictable processing
    h, w = img.shape[:2]
    target_width = 500
    if w > target_width:
        scale = target_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_img = img.copy()
    
    # --- PASS 1: Standard Scan (For QR Codes & Printed Barcodes) ---
    results = zxingcpp.read_barcodes(gray)
    scan_mode = "Standard"

    # --- PASS 2: Heavy DPM Scan (If standard fails, assume it's laser-engraved) ---
    if len(results) == 0:
        print("[INFO] Standard scan failed. Attempting Heavy DPM Processing...")
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
        
        kernel_close = np.ones((3, 3), np.uint8)
        closed_dots = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_dilate = np.ones((5, 5), np.uint8)
        connected_dots = cv2.dilate(closed_dots, kernel_dilate, iterations=1)
        
        # Border wipe to clean edges
        h_c, w_c = connected_dots.shape
        cv2.rectangle(connected_dots, (0, 0), (w_c, h_c), 0, thickness=15)
        
        # Invert and add quiet zone padding
        processed_gray = cv2.bitwise_not(connected_dots)
        pad = 40
        processed_gray = cv2.copyMakeBorder(processed_gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        
        # Re-scan using the heavily processed matrix
        results = zxingcpp.read_barcodes(processed_gray)
        scan_mode = "DPM Enhanced"
        
        # Pad the display image with white space so the coordinates match the processed image
        display_img = cv2.copyMakeBorder(display_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


    # --- FINAL RESULTS & OUTPUT ---
    if len(results) > 0:
        print(f"\n[SUCCESS] {len(results)} code(s) found using {scan_mode} mode!\n")
        
        for i, obj in enumerate(results):
            raw_format = str(obj.format).split('.')[-1].upper().replace("_", "").replace(" ", "")
            decoded_text = obj.text
            
            # Identify the specific code type
            if "DATAMATRIX" in raw_format:
                code_type = "Data Matrix"
                color = (255, 0, 0) # Blue
            elif "QR" in raw_format:
                code_type = "QR Code"
                color = (0, 255, 0) # Green
            else:
                code_type = f"Barcode ({raw_format})"
                color = (0, 255, 255) # Yellow

            # Print to terminal
            print(f"--- Target #{i+1} ---")
            print(f"Type : {code_type}")
            print(f"Data : {decoded_text}\n")

            # Draw visual bounding box and text on the image
            pos = obj.position
            pts = np.array([[pos.top_left.x, pos.top_left.y],
                            [pos.top_right.x, pos.top_right.y],
                            [pos.bottom_right.x, pos.bottom_right.y],
                            [pos.bottom_left.x, pos.bottom_left.y]], dtype=np.int32)
            
            cv2.polylines(display_img, [pts], True, color, 3)
            
            # Text background box for readability
            text_x, text_y = int(pos.top_left.x), int(pos.top_left.y) - 10
            cv2.rectangle(display_img, (text_x, max(0, text_y - 20)), (text_x + 180, text_y + 5), (0, 0, 0), -1)
            cv2.putText(display_img, f"{code_type}", (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    else:
        print("\n[FAILED] No readable codes found in the image.")
        cv2.putText(display_img, "NO CODE DETECTED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the final image
    cv2.imshow(f"Universal Scanner Result - {image_path.split('/')[-1]}", display_img)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test it by changing this path to ANY image you want to scan!
    image_to_scan = "c:/Users/HP/Desktop/internship_task/WhatsApp Image 2026-06-22 at 1.03.02 PM.jpeg"
    
    scan_universal_image(image_to_scan)