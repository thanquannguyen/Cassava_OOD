# Energy-based OOD Detection on Jetson Nano 🌿🤖

A real-time **Cassava Leaf Disease Detection** system optimized for edge devices (Jetson Nano), featuring **Energy-based Out-of-Distribution (OOD) Detection** to identify unknown objects or anomalies.

![Demo ID](demo_id.jpg) | ![Demo OOD](demo_ood.jpg)
--- | ---
*In-Distribution (Correctly Classified)* | *Out-of-Distribution (Detected as OOD)*

## 🚀 Key Features
-   **Model:** MobileNetV3-Large (Pretrained).
-   **OOD Detection:** Energy-based Models (EBM) with Temperature Scaling.
-   **Performance:** ~11 FPS (CPU) / >15 FPS (TensorRT expected).
-   **Accuracy:** 84.5% (Validation).
-   **OOD Metrics:** AUROC 0.68 (CIFAR-10 proxy).

## 📂 Project Structure
```
├── data/               # Datasets (Cassava & OOD)
├── src/
│   ├── models/         # MobileNetV3 definition
│   ├── training/       # Train & Evaluate scripts
│   └── utils/          # Metrics, Dataset, Export
├── checkpoints/        # Saved models & calibration params
├── inference.py        # Real-time inference script
└── requirements.txt    # Dependencies
```

## 🛠️ Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thanquannguyen/cassava_OOD.git
    cd cassava_OOD
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Preparation:**
    -   **Real Data:** Download [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) and extract to `data/cassava/`.
    -   **OOD Data:** Run `python src/utils/prepare_ood.py` to download CIFAR-10 as OOD data.
    -   **Dummy Data:** Run `python src/utils/create_dummy_data.py` for quick testing.

## 🏃 Usage

### 1. Training
Train the MobileNetV3 model:
```bash
python src/training/train.py --epochs 10 --batch_size 32
```

### 2. OOD Analysis & Calibration
Calculate Energy Scores and find optimal Temperature ($T$) and Threshold:
```bash
python src/training/evaluate_ood.py
```
*Results are saved to `checkpoints/calibration_params.txt`.*

### 3. Inference (Edge Deployment)
Run real-time detection on webcam:
```bash
python inference.py --camera_id 0
```
Or test on a single image:
```bash
python inference.py --image_path data/ood/ood_0.jpg
```

### 4. Benchmark
Measure FPS and Latency:
```bash
python src/utils/benchmark.py
```

## 📊 Results
| Metric | Value |
| :--- | :--- |
| **Val Accuracy** | 84.46% |
| **OOD AUROC** | 0.6816 |
| **FPR95** | 0.9595 |
| **Optimal T** | 1.41 |
| **Latency (PC)** | ~88 ms |

## 📝 License
MIT License.

