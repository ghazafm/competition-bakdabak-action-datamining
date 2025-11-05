import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()
start_time = time.time()
data_version = os.getenv("ROBOFLOW_DATA_VERSION", "1")
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE_NAME")).project(
    os.getenv("ROBOFLOW_PROJECT_NAME")
)
version = project.version(int(data_version))
dataset = version.download("folder", location=f"data/{data_version}")

yolo_model_version = os.getenv("YOLO_MODEL_VERSION", "8")
yolo_model_size = os.getenv("YOLO_MODEL_SIZE", "l")
yolo_model_name = f"yolov{yolo_model_version}{yolo_model_size}-cls.pt"
model = YOLO("model/" + os.getenv("MODE") + "/yolo/" + yolo_model_name)
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using device: mps (Apple Silicon)")
else:
    device = "cpu"
    print("Using device: cpu (No GPU detected)")

# Allow environment override
if os.getenv("DEVICE"):
    device = os.getenv("DEVICE")
    print(f"Device overridden by env: {device}")

model.to(device)

# Resolve dataset root robustly so YOLO doesn't double-append '/train'
data_root = Path(f"data/{data_version}").resolve()

# If user accidentally points to .../train, step up to parent
if data_root.name.lower() == "train":
    data_root = data_root.parent

# Prefer passing explicit splits. If no validation folder, reuse 'train' for validation to avoid NoneType errors.

results = model.train(
    data=data_root,
    epochs=int(os.getenv("EPOCHS", 10)),
    imgsz=int(os.getenv("IMG_SIZE", 640)),
    batch=int(os.getenv("BATCH_SIZE", 10)),
    name=os.getenv("YOLO_TRAINING_NAME"),
    exist_ok=True,
    workers=int(os.getenv("WORKERS", 0)),
    patience=50,  # Early stopping patience
    verbose=True,
)

trained_model_dir = f"model/trained/{data_version}/{yolo_model_version}/{yolo_model_size}/{int(start_time)}"
os.makedirs(trained_model_dir, exist_ok=True)
model.save(f"{trained_model_dir}/{os.getenv('YOLO_TRAINING_NAME')}.pt")
