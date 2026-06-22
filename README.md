# 🔍 Universal Barcode, QR & Data Matrix Scanner

🤖 **Advanced Real-Time Scanner with Dynamic Dot-Peen (DPM) Processing Capabilities**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Engine: ZXing--C++](https://img.shields.io/badge/Engine-ZXing--C%2B%2B-orange.svg)](https://github.com/zxing-cpp/zxing-cpp)

---

## 💡 Overview

Standard scanning engines often fail when reading codes marked directly onto industrial metals or textured surfaces (known as **Dot-Peen** or **Direct Part Marking**). 

This application implements an intelligent **Dual-Pass Scanning Strategy** using computer vision techniques to dynamically adapt to both pristine printed codes and heavily textured industrial surfaces in real time.

### ⚙️ How the Dual-Pass Engine Works
1. **Pass 1 (Standard Mode):** The frame is processed in native grayscale and fed directly to the `zxingcpp` engine. This ensures sub-millisecond, low-latency performance for traditional barcodes and QR codes.
2. **Pass 2 (Dot-Peen Mode):** If Pass 1 returns zero results, the system instantly engages heavy computer vision pipelines (Gaussian Filtering $\rightarrow$ Adaptive Thresholding $\rightarrow$ Morphological Dilation) to bridge detached dot-peen micro-indents into solid, readable shapes before running the decoder again.

---

## 🛠️ Tech Stack & Requirements

* **Core Language:** Python 3.10+
* **Computer Vision:** OpenCV (`opencv-python`)
* **Mathematical Operations:** NumPy
* **Decoding Engine:** ZXing-C++ (`zxing-cpp`)

---

## 📦 Getting Started & Installation

### 1. Clone the Space
```bash
git clone [https://github.com/ashwin-hamsaneni/Universal-bar-qr-data_matrix-scanner.git](https://github.com/ashwin-hamsaneni/Universal-bar-qr-data_matrix-scanner.git)
cd Universal-bar-qr-data_matrix-scanner