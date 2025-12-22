# Real-Time Fabric Defect Detection on Jetson Orin Nano
![Project Status](https://img.shields.io/badge/status-Completed-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Kaggle-pink.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Classical%20CV-purple.svg) 
![Jetson](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-orange.svg)
![Language](https://img.shields.io/badge/language-Python-blue.svg) 
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## Project Overview

This project implements a **real-time fabric defect detection system** using **purely classical computer vision techniques**, deployed and optimized on the **Jetson Nano**. The system avoids deep learning entirely and relies on **multi-method classical analysis** combined with **IoU-based bounding box fusion** for robust and interpretable defect localization. The solution is designed for **low-power edge devices** and is suitable for **industrial textile inspection** where cost, explainability, and real-time performance are critical.

---

## Problem Statement

Manual fabric inspection is:
- Time-consuming  
- Inconsistent  
- Prone to human error  

While deep learning–based solutions exist, they:
- Require large labeled datasets  
- Demand high computational resources  
- Are expensive to deploy  

**Objective:**  
Develop a **low-cost, real-time, and interpretable fabric defect detection system** using **classical computer vision**, deployable on an **edge device** without any training or labeled data.

---

## Methodology

The system uses **multiple independent classical detectors**, each capturing different defect characteristics. Their outputs are fused using an **IoU-based strategy** to produce stable final detections.

### Classical Techniques Used
- **GLCM Texture Analysis** – detects texture irregularities  
- **FFT Frequency Analysis** – detects disruptions in periodic patterns  
- **Gabor Wavelets** – captures directional and repetitive textures  
- **Statistical Local Variance** – highlights abrupt anomalies  
- **Background Subtraction** – detects stains and fading  
- **Edge Detection + Hough Transform** – detects linear defects  

### Fusion Strategy
- Bounding boxes from all detectors are merged using **Intersection over Union (IoU)**  
- Size-based and border-based filtering removes false positives  

---

## System Pipeline

```

Camera Frame
↓
Preprocessing (Grayscale, Normalization)
↓
Parallel Classical Detectors
↓
Bounding Box Extraction
↓
IoU-Based Box Fusion
↓
Post-Processing Filters
↓
Final Defect Localization

```

---

## Jetson Orin Nano Deployment

- Real-time processing using live USB camera feed  
- Resolution optimized to **640×480** for performance  
- No disk I/O during runtime  
- Lightweight classical algorithms ensure stable FPS  

This makes the system suitable for **on-device industrial inspection**.

---

## Performance Highlights

- Fully real-time execution on Jetson Nano  
- No model training or dataset labeling required  
- Low power consumption  
- High interpretability and explainability  

---

## Industrial Applications

- Textile manufacturing quality control  
- Automated inspection on production lines  
- Low-cost alternatives to GPU-based vision systems  
- Edge-based visual monitoring systems  

## Future Work

* Detector confidence voting
* Further runtime optimization
* Optional deep learning verifier
* Integration with industrial conveyor systems

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Awais Asghar**
Electrical Engineering | Computer Vision | Edge AI

---

⭐ If you find this project useful, consider giving it a star!

```
