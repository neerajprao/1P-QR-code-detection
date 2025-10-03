# infer.py (Modified for Zero Arguments)

import json
from pathlib import Path
from ultralytics import YOLO
import yaml
import sys

# --- HARDCODED CONFIGURATION ---
MODEL_PATH = 'runs/detect/qr_train_custom_colab/weights/best.pt'
DATA_YAML_PATH = 'medicine_qr_codes-1/data.yaml'

# --- INPUT/OUTPUT PATHS (CHANGE THESE TO YOUR ABSOLUTE PATHS) ---
# 🚨 IMPORTANT: Replace the path below with your correct absolute path.
INPUT_DIR = Path('/Users/neerajprao/Downloads/multiqr-hackathon-down/data/test_images') 
JSON_OUTPUT_FILE = Path('outputs/submission_detection_1.json')

# --- INFERENCE SETTINGS ---
CONFIDENCE_THRESHOLD = 0.4 # Adjust this value if needed

# --- Utility Functions ---
def load_class_names(yaml_path):
    """Loads class names from the data.yaml file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('names', [])
    except FileNotFoundError:
        print(f"Warning: {yaml_path} not found. Using default class name: 'qr_code'.", file=sys.stderr)
        return ['qr_code']

def run_batch_inference(input_dir: Path, json_output_file: Path, confidence_threshold: float):
    """
    Runs inference on all images in a directory and saves a single JSON output file.
    """
    
    # 1. Load Model and Configuration
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        class_names = load_class_names(DATA_YAML_PATH)
    except FileNotFoundError as e:
        print(f"Error: Model file not found at {MODEL_PATH}. Aborting: {e}", file=sys.stderr)
        return
    
    # 2. Identify Image Files
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_files:
        print(f"No images found in {input_dir}. Aborting.", file=sys.stderr)
        return

    final_submission_data = {}
    print(f"Found {len(image_files)} images. Starting inference...")
    
    # 3. Run Batch Inference
    image_paths_str = [str(f) for f in image_files]
    
    try:
        results = model.predict(
            source=image_paths_str,
            conf=confidence_threshold,
            imgsz=640,
            save=False,       # IMPORTANT: Do not save visual results
            save_txt=False, 
            verbose=False    
        )
    except Exception as e:
        print(f"Error during YOLOv8 prediction: {e}", file=sys.stderr)
        return

    # 4. Process Results for JSON Output
    for img_path_str, result in zip(image_paths_str, results):
        img_path = Path(img_path_str)
        image_id = img_path.name
        detections = []
        
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

            detections.append({
                "box_2d": [int(round(c)) for c in xyxy], 
                "confidence": round(conf, 4),
                "label": cls_name
            })
            
        final_submission_data[image_id] = detections

    # 5. Save the JSON Submission Output
    try:
        json_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_output_file, 'w') as f:
            json.dump(final_submission_data, f, indent=4)
        
        print(f"\n✅ Inference complete. JSON Submission saved to: {json_output_file.resolve()}")

    except Exception as e:
        print(f"Error saving JSON output to {json_output_file}: {e}", file=sys.stderr)


if __name__ == "__main__":
    # --- Execute without arguments ---
    if not INPUT_DIR.is_dir():
        print(f"Error: Hardcoded input directory not found at {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)
    
    # Run the core function using the hardcoded variables
    run_batch_inference(INPUT_DIR, JSON_OUTPUT_FILE, CONFIDENCE_THRESHOLD)