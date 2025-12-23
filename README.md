# Real-Time Fabric Defect Detection on Jetson Orin Nano
![Project Status](https://img.shields.io/badge/status-Completed-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Kaggle-pink.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Classical%20CV-purple.svg) 
![Jetson](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-orange.svg)
![Language](https://img.shields.io/badge/language-Python-blue.svg) 
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

<img width="1874" height="1008" alt="Image" src="https://github.com/user-attachments/assets/8fb8f67a-30ea-4a76-b38e-773e8aeb20db" />

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

<img width="1871" height="1009" alt="Image" src="https://github.com/user-attachments/assets/3972a220-71ff-4f08-8d20-efb7db99d994" />

---

## Methodology

The system uses **multiple independent classical detectors**, each capturing different defect characteristics. Their outputs are fused using an **IoU-based strategy** to produce stable final detections.

<img width="1872" height="1013" alt="Image" src="https://github.com/user-attachments/assets/3eff4437-90ba-4775-9dbc-30cf608e98d9" />

<img width="1877" height="1008" alt="Image" src="https://github.com/user-attachments/assets/49a7d389-c4af-490b-a259-247bce1a2dfe" />

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
<img width="1873" height="1010" alt="Image" src="https://github.com/user-attachments/assets/278640ae-0f18-4993-a9b5-cfcdc05c4da7" />

<img width="1873" height="1006" alt="Image" src="https://github.com/user-attachments/assets/1b0af2a4-0858-49b6-9661-6c66c903cc74" />

<img width="1873" height="1014" alt="Image" src="https://github.com/user-attachments/assets/0b5f5be6-7242-4ec0-bdc1-ce76f9d065d1" />

<img width="1871" height="1009" alt="Image" src="https://github.com/user-attachments/assets/ca524093-3cc0-41d1-8272-eb031bd7f92c" />

<img width="1873" height="1011" alt="Image" src="https://github.com/user-attachments/assets/5593ceaf-a71c-49a7-afdc-dd8762805809" />

---

## Jetson Orin Nano Deployment

<img width="1875" height="1013" alt="Image" src="https://github.com/user-attachments/assets/23f6847a-1f8f-401d-850b-6d61c909e37e" />

- Real-time processing using live USB camera feed  
- Resolution optimized to **640×480** for performance  
- No disk I/O during runtime  
- Lightweight classical algorithms ensure stable FPS  

This makes the system suitable for **on-device industrial inspection**.

---

## Performance Highlights

<img width="1875" height="1010" alt="Image" src="https://github.com/user-attachments/assets/8b267f11-01ef-40df-a26d-061d3b9edae0" />

- Fully real-time execution on Jetson Nano  
- No model training or dataset labeling required  
- Low power consumption  
- High interpretability and explainability  

---

## Industrial Applications

<img width="1871" height="1014" alt="Image" src="https://github.com/user-attachments/assets/e5e62e15-5c31-45ff-b044-27e03a15f384" />

- Textile manufacturing quality control  
- Automated inspection on production lines  
- Low-cost alternatives to GPU-based vision systems  
- Edge-based visual monitoring systems  

## Future Work

<img width="1875" height="1007" alt="Image" src="https://github.com/user-attachments/assets/eba93be8-9933-4ceb-9bb7-3cbe3e7b9a58" />

* Detector confidence voting
* Further runtime optimization
* Optional deep learning verifier
* Integration with industrial conveyor systems

<img width="1871" height="1001" alt="Image" src="https://github.com/user-attachments/assets/6f811df9-1452-40f3-aae1-d2101b9ddca9" />

---

## Demo 

![Project Image](https://github.com/user-attachments/assets/4ebbb4b3-4e26-4743-aa35-f8bca5364bca)

https://github.com/user-attachments/assets/fac8b5cb-90fb-4048-a1f4-c30167beda15

https://github.com/user-attachments/assets/c4d4eb7a-723e-45d2-abae-a23ca4ae641c

## License

This project is licensed under the **MIT License**.

---

## Author

**Awais Asghar**
Electrical Engineering | Computer Vision | Edge AI

⭐ If you find this project useful, consider giving it a star! ⭐
