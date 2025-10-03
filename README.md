# YOLOv8 QR Code Detection Project

This project implements **multi-QR code detection** using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) on medicine packs.  
The pipeline covers dataset setup, training, evaluation, and inference with visualization and JSON submission file generation.

---

##  Project Structure

```
yolov8-qr-detection-project/
│── data/
│   ├── test_images/                # Test images for inference
│   ├── medicine_qr_codes-1/        # Roboflow dataset (train/val split)
│   │   ├── data.yaml
│   │   ├── README.dataset.txt
│   │   └── README.roboflow.txt
│   └── outputs/
│       ├── submission_detection_1.json
│       └── visualizations/
│
│── runs/detect/                    # YOLOv8 run logs and weights
│
│── src/
│   ├── datasets/                   # Train/val/test splits
│   ├── models/                     # Saved model weights
│   │   ├── best.pt
│   │   └── last.pt
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│
│── requirements.txt
│── README.md
```

---

##  Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/yolov8-qr-detection-project.git
   cd yolov8-qr-detection-project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv qr
   ```

   - Activate (Linux/Mac):
     ```bash
     source qr/bin/activate
     ```
   - Activate (Windows):
     ```bash
     qr\Scripts\activate.bat
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Dependencies (`requirements.txt`)

```
ultralytics
roboflow
Pillow
numpy
opencv-python-headless
tqdm
```

---

##  Training the Model

Update **`train.py`** configuration before running:

```python
MODEL_SAVE_ROOT = "src/models"
ROBOFLOW_API_KEY = "YOUR_API_KEY"
ROBOFLOW_PROJECT_NAME = "medicine_qr_codes"
ROBOFLOW_VERSION_NUMBER = 1
```

Then train:
```bash
python src/train.py
```

 Output: `best.pt` and `last.pt` saved under `src/models/`.

---

## 📊 Evaluation

Update **`evaluate.py`** paths:

```python
MODEL_PATH = "src/models/best.pt"
DATA_YAML_PATH = "data/medicine_qr_codes-1/data.yaml"
```

Run evaluation:
```bash
python src/evaluate.py
```

---

##  Inference & Visualization

Update **`visualize.py`** configuration:

```python
MODEL_PATH = "src/models/best.pt"
INPUT_DIR = "data/test_images"
```

Run inference:
```bash
python src/visualize.py
```

 Output:
- Bounding box predictions
- `submission_detection_1.json` file under `outputs/`
- Visualization images under `outputs/visualizations/`

---

##  Notes

- Ensure correct **absolute paths** are set inside `train.py`, `evaluate.py`, and `visualize.py`.
- Dataset is managed via [Roboflow](https://roboflow.com/).
- Supports multiple QR codes per image.

---

##  Hackathon Problem Statement

This project solves the challenge of **multi-QR code detection on medicine packs**.  
Most medicine packs have more than one QR code (manufacturer, batch number, distributor, regulator). The model detects and outputs all present QR codes per image.
