# 🚀 Model Information & Optimization Strategy

## 📌 Model Overview

**Model Name:** Custom YOLOv5n (Nano Variant)  
**Task:** Real-Time Pothole Detection  
**Framework:** PyTorch → ONNX → TensorFlow Lite  
**Deployment Target:** Raspberry Pi 4 (ARM Cortex-A72 CPU)  
**Inference Backend:** TensorFlow Lite (XNNPACK Delegate)  

This model is specifically optimized for low-power ARM edge deployment while maintaining real-time inference performance.

---

## 🧠 Why YOLOv5n?

YOLOv5n (Nano variant) was selected due to:

- Extremely lightweight architecture
- Low parameter count (~1.9M parameters)
- High speed on CPU-only systems
- Strong performance-to-compute ratio
- Compatibility with INT8 quantization

Unlike heavier models (YOLOv5s, YOLOv8m, etc.), YOLOv5n ensures:

- Faster inference
- Lower memory footprint
- Better suitability for embedded devices

---

## 📊 Dataset & Training

- Dataset: IIT Madras Road Damage Dataset
- Classes: Pothole (Single-class configuration)
- Training Resolution: 320×320
- Optimizer: Adam
- Epochs: 120
- Augmentation: Mosaic, Flip, HSV adjustment
- Validation mAP: mAP@0.5: 0.90

The dataset was refined to focus only on pothole class to:

- Reduce output dimensionality
- Improve detection confidence
- Improve real-time performance

---

## ⚡ Optimization Pipeline

### 1️⃣ Model Conversion
PyTorch → ONNX → TensorFlow Lite

### 2️⃣ Post-Training Quantization
- INT8 Quantization applied
- Reduced model size significantly
- Improved inference latency
- Maintained high detection accuracy

Final Model Size:
~1.9 MB (INT8 optimized)

---

## 🧮 Edge Optimization Techniques

| Technique | Purpose |
|------------|---------|
| INT8 Quantization | Reduce latency & memory |
| Multi-threaded inference (4 threads) | Use all ARM cores |
| Reduced Input Resolution (320×320) | Lower computation |
| Efficient NMS | Remove duplicate detections |
| Cooldown-based logging | Prevent IO bottlenecks |
| Video-mode camera configuration | Faster frame capture |

---

## 🚀 Performance Metrics (Raspberry Pi 4 - CPU Only)

| Mode | FPS |
|------|------|
| Headless (No Display) | 12–14 FPS |
| Live Display Mode | 5–8 FPS |

Average Inference Time:
~0.05 seconds per frame

All performance achieved without GPU or external accelerators.

---

## 💡 Why TensorFlow Lite?

TensorFlow Lite was chosen due to:

- Native ARM optimization
- XNNPACK delegate support
- Small runtime footprint
- High CPU efficiency
- Production-ready edge deployment

---

## 🏆 Key Achievements

- Fully CPU-only real-time inference
- Sub-100ms latency
- Accurate anomaly detection
- Automated timestamp logging
- Evidence snapshot generation
- Edge-ready architecture

---

## 🔮 Scalability Potential

This architecture can be extended to:

- Crack detection
- Road surface monitoring
- Smart city integration
- IoT-based road damage mapping
- Cloud-connected anomaly reporting

---

## 🎯 Summary

This model demonstrates that lightweight object detection networks, when properly optimized through quantization and system-level tuning, can achieve real-time anomaly detection entirely on ARM-based edge hardware without GPU acceleration.

It balances:

✔ Accuracy  
✔ Speed  
✔ Memory efficiency  
✔ Deployment practicality  

Making it a strong candidate for real-world smart infrastructure systems.