# evaluate.py

import sys
from ultralytics import YOLO
from pathlib import Path

# --- Configuration (UPDATED MODEL PATH) ---
MODEL_PATH = '/Users/neerajprao/Downloads/multiqr-hackathon-down/src/models/qr_train_custom_colab/weights/best.pt'
DATA_YAML_PATH = 'medicine_qr_codes-1/data.yaml'

def run_evaluation():
    """Loads the trained model and runs a formal validation on the test split."""
    
    # 1. Check for required files
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model weights not found at {MODEL_PATH}.", file=sys.stderr)
        sys.exit(1)
    
    if not Path(DATA_YAML_PATH).exists():
        print(f"Error: Data YAML file not found at {DATA_YAML_PATH}.", file=sys.stderr)
        sys.exit(1)

    # 2. Load Model
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Run Validation
    print(f"Running validation on the test split using {DATA_YAML_PATH}...")
    try:
        metrics = model.val(
            data=DATA_YAML_PATH,
            split="test",
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.65       
        )

        # 4. Print Results (Using corrected .mp and .mr access)
        print("\n--- Evaluation Results (Self-Check) ---")
        print(f"Model: {MODEL_PATH}")
        print("-" * 35)
        print(f"Mean Average Precision (mAP50-95): {metrics.box.map:.4f}")
        print(f"mAP50 (Common Metric for Object Detection): {metrics.box.map50:.4f}")
        print(f"mAP75 (Higher Precision Requirement): {metrics.box.map75:.4f}")
        print(f"Overall Precision (Mean Precision): {metrics.box.mp:.4f}")
        print(f"Overall Recall (Mean Recall): {metrics.box.mr:.4f}")
        print("-" * 35)

    except Exception as e:
        print(f"An error occurred during validation: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_evaluation()