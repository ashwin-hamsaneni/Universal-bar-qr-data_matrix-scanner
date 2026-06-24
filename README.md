Advanced Optical Decoder Suite 🔍

A robust, Python-based optical decoding suite designed to read challenging 1D and 2D barcodes. This suite specializes in decoding Direct Part Marks (DPM), laser-engraved Data Matrices, and codes printed on Printed Circuit Boards (PCBs) where standard commercial scanners fail due to shadows, glare, and disconnected dot modules.

Features ✨

3-Pass Decoding Engine: Progressively attempts to decode images using Raw RGB, Grayscale, and finally an Advanced Preprocessing pipeline to ensure maximum success rates while minimizing latency.

Pillow (LANCZOS) Resampling: Uses high-fidelity mathematical resizing to preserve the crisp, square edges of Data Matrix modules during upscaling, avoiding standard OpenCV pixelation.

CLAHE Shadow Equalization: Utilizes Contrast Limited Adaptive Histogram Equalization to normalize extreme shadows and bright solder glare specifically found on PCBs.

Dot-Peen / Laser Engraving Pipeline: Implements Morphological Close and Dilation techniques to digitally "fill" hollow laser rings and bridge disconnected modules.

Automated Quiet Zone Generation: Artificially injects pure white padding around tightly cropped images to satisfy decoding engine margin requirements without slicing the actual code.

Dynamic Bounding Boxes: Automatically calculates scale factors and padding offsets to draw highly accurate bounding boxes and data overlays directly onto the original image.

Prerequisites 🛠️

Ensure you have Python 3.x installed. You will need the following libraries to run the scripts in this suite:

pip install opencv-python zxing-cpp numpy Pillow


Included Modules 📁

1. advanced_pcb_scanner.py (Core OOP Scanner)

The flagship static image scanner. Built with Object-Oriented Programming (OOP) principles, it automatically runs any provided image through the 3-Pass Decoding Engine. Perfect for evaluating difficult, tightly cropped macro photos of PCBs or metallic surfaces.

Usage: Modify the test_image path at the bottom of the script and run it.

Advanced Optical Decoder Suite 🔍

A robust, Python-based optical decoding suite designed to read challenging 1D and 2D barcodes. This suite specializes in decoding Direct Part Marks (DPM), laser-engraved Data Matrices, and codes printed on Printed Circuit Boards (PCBs) where standard commercial scanners fail due to shadows, glare, and disconnected dot modules.

Features ✨

3-Pass Decoding Engine: Progressively attempts to decode images using Raw RGB, Grayscale, and finally an Advanced Preprocessing pipeline to ensure maximum success rates while minimizing latency.

Pillow (LANCZOS) Resampling: Uses high-fidelity mathematical resizing to preserve the crisp, square edges of Data Matrix modules during upscaling, avoiding standard OpenCV pixelation.

CLAHE Shadow Equalization: Utilizes Contrast Limited Adaptive Histogram Equalization to normalize extreme shadows and bright solder glare specifically found on PCBs.

Dot-Peen / Laser Engraving Pipeline: Implements Morphological Close and Dilation techniques to digitally "fill" hollow laser rings and bridge disconnected modules.

Automated Quiet Zone Generation: Artificially injects pure white padding around tightly cropped images to satisfy decoding engine margin requirements without slicing the actual code.

Dynamic Bounding Boxes: Automatically calculates scale factors and padding offsets to draw highly accurate bounding boxes and data overlays directly onto the original image.

Prerequisites 🛠️

Ensure you have Python 3.x installed. You will need the following libraries to run the scripts in this suite:

pip install opencv-python zxing-cpp numpy Pillow


Included Modules 📁

1. advanced_pcb_scanner.py (Core OOP Scanner)

The flagship static image scanner. Built with Object-Oriented Programming (OOP) principles, it automatically runs any provided image through the 3-Pass Decoding Engine. Perfect for evaluating difficult, tightly cropped macro photos of PCBs or metallic surfaces.

Usage: Modify the test_image path at the bottom of the script and run it.

2. live_scanner_dotpeen.py (Real-Time Webcam Decoder)

A fast, live-feed webcam scanner that dynamically switches between standard and DPM processing based on what it sees in the frame.

Usage: Run the script, hold your paper or part up to the webcam, and press q to quit.

3. evaluation_dashboard.py (Automated Reporting)

An automated benchmarking tool that ingests multiple sample images, processes them, calculates system latency, and spits out professional, side-by-side graphical dashboard panels saved directly to your disk.

4. live_stress_tester.py (Interactive Environmental Tester)

A unique interactive tool that allows you to apply real-world industrial degradation filters (Blur, Low Resolution, Noise, Poor Lighting, Distance) to your live webcam feed at the press of a button. It automatically tracks minimum/maximum readable pixel sizes, latencies, and success rates, outputting a highly detailed markdown table when closed.

Usage: Press SPACE to cycle through filters and D to toggle the heavy DPM math pipeline. Press q to quit and generate the report.

How the Advanced Pipeline Works 🧠

When a standard scan fails, the advanced pipeline engages the following mathematical filters:

LANCZOS Resize: Standardizes the physical pixel footprint of the code.

CLAHE: Equalizes local shadows and highlights.

Gaussian Blur: Melts away PCB fiberglass textures and copper traces.

Adaptive Thresholding: Binarizes the image into pure black and white.

Morphological Close & Dilate: Fills hollow laser rings and connects scattered dot-peen dimples into solid L-patterns.

Edge Wipe & Inversion: Erases false shadows on the image borders and flips the code to black-on-white.

Built using OpenCV and ZXing-C++.

2. live_scanner_dotpeen.py (Real-Time Webcam Decoder)

A fast, live-feed webcam scanner that dynamically switches between standard and DPM processing based on what it sees in the frame.

Usage: Run the script, hold your paper or part up to the webcam, and press q to quit.

3. evaluation_dashboard.py (Automated Reporting)

An automated benchmarking tool that ingests multiple sample images, processes them, calculates system latency, and spits out professional, side-by-side graphical dashboard panels saved directly to your disk.

4. live_stress_tester.py (Interactive Environmental Tester)

A unique interactive tool that allows you to apply real-world industrial degradation filters (Blur, Low Resolution, Noise, Poor Lighting, Distance) to your live webcam feed at the press of a button. It automatically tracks minimum/maximum readable pixel sizes, latencies, and success rates, outputting a highly detailed markdown table when closed.

Usage: Press SPACE to cycle through filters and D to toggle the heavy DPM math pipeline. Press q to quit and generate the report.

How the Advanced Pipeline Works 🧠

When a standard scan fails, the advanced pipeline engages the following mathematical filters:

LANCZOS Resize: Standardizes the physical pixel footprint of the code.

CLAHE: Equalizes local shadows and highlights.

Gaussian Blur: Melts away PCB fiberglass textures and copper traces.

Adaptive Thresholding: Binarizes the image into pure black and white.

Morphological Close & Dilate: Fills hollow laser rings and connects scattered dot-peen dimples into solid L-patterns.

Edge Wipe & Inversion: Erases false shadows on the image borders and flips the code to black-on-white.

Built using OpenCV and ZXing-C++.