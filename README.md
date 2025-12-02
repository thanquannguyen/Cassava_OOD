# Energy-based OOD Detection on Jetson Nano

## Project Structure
- `data/`: Contains datasets.
    - `cassava/`: In-Distribution (ID) dataset.
    - `ood/`: Out-of-Distribution (OOD) dataset (e.g., ImageNet, Nature).
- `src/`: Source code.
    - `models/`: Model definitions (MobileNetV3).
    - `training/`: Training scripts.
    - `utils/`: Utility functions (metrics, loading).
- `notebooks/`: Jupyter notebooks for analysis.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   - Download the **Cassava Leaf Disease Classification** dataset from Kaggle.
   - Extract images to `data/cassava/images/`.
   - Place `train.csv` in `data/cassava/`.
   - Prepare OOD data in `data/ood/`.
