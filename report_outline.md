# Thesis Report Outline: Energy-based OOD Detection for Cassava Disease Diagnosis

## 1. Introduction
-   **Problem Statement:** Cassava is a crucial crop; diseases threaten yield. Manual diagnosis is slow.
-   **Challenge:** Deep Learning models are "overconfident" on unknown data (OOD). Deploying on edge devices (Jetson Nano) requires efficiency.
-   **Objective:** Develop a lightweight OOD detection system using Energy-based Models (EBM) on MobileNetV3.

## 2. Related Work
-   **Cassava Disease Classification:** Existing CNN approaches (ResNet, EfficientNet).
-   **OOD Detection:** Softmax Confidence vs. Energy Scores (Liu et al., NeurIPS 2020).
-   **Edge AI:** Model optimization (MobileNet, TensorRT) for embedded systems.

## 3. Methodology
### 3.1 System Architecture
-   **Backbone:** MobileNetV3-Large (Pretrained on ImageNet).
-   **Input:** 224x224 RGB Images.
-   **Output:** 5 Classes (CBB, CBSD, CGM, CMD, Healthy).

### 3.2 Energy-based OOD Detection
-   **Concept:** Energy function $E(x) = -T \cdot \log \sum_{k} e^{f_k(x)/T}$.
-   **Hypothesis:** ID data has lower energy (more negative); OOD data has higher energy.
-   **Calibration:** Temperature Scaling ($T$) to smooth softmax distributions.

### 3.3 Edge Implementation
-   **Hardware:** NVIDIA Jetson Nano.
-   **Optimization:** ONNX / TensorRT (FP16).
-   **Pipeline:** Camera Capture -> Preprocessing -> Inference -> Energy Thresholding -> Visualization.

## 4. Experiments & Results
### 4.1 Setup
-   **Dataset:** Cassava Leaf Disease (21k images).
-   **OOD Dataset:** CIFAR-10 (Proxy for unknown objects).
-   **Training Config:** AdamW, LR=0.001, 5 Epochs.

### 4.2 Quantitative Results
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Accuracy (ID)** | 84.46% | Good baseline for MobileNetV3. |
| **AUROC (OOD)** | 0.6816 | Indicates separability (room for improvement). |
| **FPR95** | 0.9595 | High false positive rate (needs better OOD data). |
| **Inference Speed** | ~11 FPS | CPU baseline. |

### 4.3 Qualitative Results
-   Show confusion matrix.
-   Show examples of "PASS" (Correct ID) vs. "BLOCK" (OOD).
-   Discuss failure cases (e.g., OOD images with low energy).

## 5. Discussion
-   **Effectiveness of Energy Score:** Comparison with Softmax Confidence.
-   **Real-world Challenges:** Lighting, camera blur, similar-looking plants.
-   **Future Work:** Use "exposure" OOD training (Outlier Exposure), Quantization (INT8).

## 6. Conclusion
-   Summary of contributions.
-   Feasibility of deploying OOD detection on low-power edge devices.

## 7. References
-   Liu, W., et al. "Energy-based Out-of-Distribution Detection." NeurIPS 2020.
-   Howard, A., et al. "Searching for MobileNetV3." ICCV 2019.
