🚀 ARM Edge AI – Real-Time Pothole Detection System

📌 Project Overview

This project presents a real-time pothole detection system optimized for ARM-based edge devices, specifically deployed on a Raspberry Pi 4 (4GB).

The system leverages a lightweight YOLOv5n object detection model that has been quantized to INT8 and converted to TensorFlow Lite for efficient CPU-only inference.

The pipeline performs:
	•	Live video capture from Raspberry Pi Camera Module
	•	Real-time inference using optimized TFLite model
	•	Non-Maximum Suppression (NMS) for accurate detection
	•	Bounding box visualization on potholes
	•	Timestamped anomaly logging
	•	Automatic snapshot saving upon detection

The system achieves real-time performance entirely on CPU without requiring GPU acceleration.

🧠 Problem Statement

Poor road conditions and undetected potholes lead to:
	•	Vehicle damage
	•	Increased accident risk
	•	High maintenance costs
	•	Delayed infrastructure repair

Most detection systems rely on heavy cloud computation or GPU hardware.

This project demonstrates that edge-optimized AI can detect road anomalies in real-time using only ARM CPU hardware, enabling scalable, low-cost deployment.

🏗️ System Architecture
Raspberry Pi Camera
        ↓
Frame Preprocessing
        ↓
INT8 YOLOv5n TFLite Model
        ↓
Confidence Filtering
        ↓
Non-Maximum Suppression
        ↓
Bounding Box Rendering
        ↓
Snapshot + CSV Logging

🧠 Model Details

Base Architecture
	•	Model: YOLOv5n (Nano Variant)
	•	Input Resolution: 320 × 320
	•	Framework: PyTorch → ONNX → TensorFlow Lite
	•	Deployment Backend: TensorFlow Lite with XNNPACK delegate

Why YOLOv5n?

YOLOv5n was selected because:
	•	Extremely lightweight (~1.9M parameters)
	•	Designed for edge deployment
	•	Fast CPU inference
	•	High performance-to-compute efficiency
	•	Supports quantization without major accuracy drop

⸻

📊 Dataset & Training
	•	Dataset: IIT Madras Road Damage Dataset
	•	Classes: Single-class (Pothole)
	•	Total Images: ~6000
	•	Training Epochs: 120
	•	Optimizer: Adam
	•	Image Size: 320 × 320
	•	Augmentations:
	•	Mosaic
	•	Horizontal Flip
	•	HSV color augmentation

⚡ Model Optimization Strategy

To ensure real-time performance on ARM CPU:

1️⃣ Quantization
	•	Post-training INT8 quantization
	•	Reduced model size
	•	Reduced memory footprint
	•	Improved inference speed
	•	Minimal accuracy degradation

2️⃣ Multi-threaded Inference
	•	num_threads=4
	•	Utilizes all Raspberry Pi cores

3️⃣ Reduced Input Resolution
	•	320 × 320 input
	•	Balanced accuracy and speed

4️⃣ Efficient Post-Processing
	•	Vectorized confidence calculation
	•	Non-Maximum Suppression (NMS)
	•	Cooldown-based logging to prevent IO bottlenecks

📁 Repository Structure

arm-edge-pothole-detection/
│
├── models/
│   └── best-int8.tflite
│
├── src/
│   ├── pothole_detection.py
│   ├── config.py
│   └── utils.py
│
├── output/
│   ├── detections.csv
│   └── snapshots/
│
├── demo/
│   └── demo_video.mp4
│
├── docs/
│   └── report.pdf
│
├── requirements.txt
└── README.md    

▶️ How To Run

Install Dependencies
pip install -r requirements.txt
Run Detection
python src/pothole_detection.py

📝 Logging System

When a pothole is detected:
	•	Snapshot is saved in /output/snapshots
	•	Entry is appended to /output/detections.csv
	•	Cooldown prevents duplicate logs

Example CSV entry:
Timestamp, Type, Confidence, Snapshot
2026-02-20_18-32-10, pothole, 0.87, pothole_2026-02-20_18-32-10.jpg

🧪 Deployment Environment
	•	Device: Raspberry Pi 4 (4GB)
	•	OS: Raspberry Pi OS
	•	Inference Engine: TensorFlow Lite
	•	CPU Optimization: XNNPACK Delegate
	•	Camera: Raspberry Pi Camera Module

⸻

🔮 Scalability Potential

This architecture can be extended for:
	•	Crack detection
	•	Road quality monitoring
	•	Smart city integration
	•	IoT-based anomaly mapping
	•	Cloud-connected reporting systems
	•	Autonomous vehicle perception modules

⸻

🏆 Key Achievements
	•	Real-time edge inference on ARM CPU
	•	Fully quantized lightweight detection model
	•	Low memory footprint
	•	Production-style modular architecture
	•	Automated anomaly logging
	•	Practical deployment on embedded hardware

⸻

🎯 Conclusion

This project demonstrates that lightweight object detection networks, when properly optimized through quantization and system-level tuning, can achieve real-time anomaly detection entirely on ARM-based edge hardware.

The system successfully balances:
	•	Accuracy
	•	Speed
	•	Computational efficiency
	•	Deployment practicality

Making it suitable for scalable smart infrastructure applications.

⸻

👨‍💻 Author

Shrey Patel
Electronics & Communication Engineering
ARM Edge AI Hackathon Project
