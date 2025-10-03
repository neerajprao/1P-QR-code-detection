# train.py

import sys
import os
from roboflow import Roboflow
from ultralytics import YOLO

# --- Configuration (REPLACE API KEY) ---
ROBOFLOW_API_KEY = "1DqThHodBicS3iUCkPIC" 
WORKSPACE_NAME = "test1-umekt"
PROJECT_NAME = "medicine_qr_codes-mmb4l"
VERSION_NUMBER = 1
MODEL_ARCH = "yolov8n.yaml"
DATASET_FORMAT = "yolov8"
DATASET_DIR = "medicine_qr_codes-1"
DATA_YAML_PATH = f"{DATASET_DIR}/data.yaml"

# --- MODEL SAVING CONFIGURATION (UPDATED) ---
# The weights will be saved to: 
# /Users/neerajprao/Downloads/multiqr-hackathon-down/src/models/qr_train_custom_colab/weights/best.pt
MODEL_SAVE_ROOT = "/Users/neerajprao/Downloads/multiqr-hackathon-down/src/models"
RUN_NAME = "qr_train_custom_colab"

def setup_and_train():
    """Downloads dataset from Roboflow and trains the YOLOv8 model."""
    
    # 1. Roboflow Setup and Dataset Download
    print("--- 1. Setting up Roboflow and downloading dataset ---")
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
        version = project.version(VERSION_NUMBER)
        
        dataset = version.download(DATASET_FORMAT)
        print(f"Dataset downloaded to: {dataset.location}")

    except Exception as e:
        print(f"Error during Roboflow operations: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. YOLO Model Training
    print("\n--- 2. Starting YOLOv8 model training ---")
    try:
        model = YOLO(MODEL_ARCH)

        results = model.train(
            data=DATA_YAML_PATH,  
            epochs=100,
            imgsz=640,
            batch=16,
            # Direct saving to the new location
            project=MODEL_SAVE_ROOT,  
            name=RUN_NAME             
        )
        print(f"\nTraining completed successfully. Weights saved under: {MODEL_SAVE_ROOT}/{RUN_NAME}")
        
    except Exception as e:
        print(f"Error during YOLOv8 training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    setup_and_train()