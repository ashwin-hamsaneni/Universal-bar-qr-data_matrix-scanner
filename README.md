# 🔍 Universal Barcode, QR & Data Matrix Scanner

🤖 **Advanced Computer Vision Scanner featuring Live Webcam Streaming & Static Image Decoding with Dynamic Dot-Peen (DPM) Support.**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Engine: ZXing--C++](https://img.shields.io/badge/Engine-ZXing--C%2B%2B-orange.svg)](https://github.com/zxing-cpp/zxing-cpp)

---

## 💡 Overview

Standard scanning engines struggle with barcodes and matrices engraved directly onto industrial metals, plastic textures, or bumpy materials (known as **Dot-Peen** or **Direct Part Marking**). 

This toolkit uses an intelligent **Dual-Pass Strategy** to handle everything from pristine digital barcodes to heavily textured industrial markings across two distinct script interfaces.

### ⚙️ The Dual-Pass Engine
* **Pass 1 (Standard Mode):** Instant, ultra-low-latency scanning for traditional printed barcodes and QR codes.
* **Pass 2 (Dot-Peen Mode):** If Pass 1 finds nothing, the engine applies specialized image transformations (Gaussian Blur $\rightarrow$ Adaptive Thresholding $\rightarrow$ Morphological Dilation/Closing) to bridge isolated dot indents into solid, readable code shapes.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Libraries:** OpenCV (`opencv-python`), NumPy, ZXing-C++ (`zxing-cpp`)

---

## 📦 Quick Start & Installation

### 1. Clone & Navigate
```bash
git clone [https://github.com/ashwin-hamsaneni/Universal-bar-qr-data_matrix-scanner.git](https://github.com/ashwin-hamsaneni/Universal-bar-qr-data_matrix-scanner.git)
cd Universal-bar-qr-data_matrix-scanner