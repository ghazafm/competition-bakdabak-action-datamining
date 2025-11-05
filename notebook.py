import os
import time

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

data_dir = f"data/{data_version}"
train_dir = os.path.join(data_dir, "train")
val_exists = os.path.isdir(os.path.join(data_dir, "val")) or os.path.isdir(
    os.path.join(data_dir, "valid")
)

# Determine split parameter based on validation folder existence
if not val_exists:
    print(
        f"No validation folder found in '{data_dir}'. Using automatic split from training data."
    )
    # Use only the train directory and let YOLO automatically split it
    training_data = train_dir
    split_val = "test"  # Use a different split mode that won't fail
else:
    print(
        f"Validation folder found in '{data_dir}'; running training with validation enabled."
    )
    training_data = data_dir
    split_val = "val"

results = model.train(
    data=training_data,
    epochs=int(os.getenv("EPOCHS", 10)),
    imgsz=int(os.getenv("IMG_SIZE", 640)),
    batch=int(os.getenv("BATCH_SIZE", 10)),
    name=os.getenv("YOLO_TRAINING_NAME"),
    exist_ok=True,
    workers=int(os.getenv("WORKERS", 0)),  # Default to 0 to avoid shared memory issues
    patience=50,  # Early stopping patience
    verbose=True,
    split=split_val,
)

trained_model_dir = f"model/trained/{data_version}/{yolo_model_version}/{yolo_model_size}/{int(start_time)}"
os.makedirs(trained_model_dir, exist_ok=True)
model.save(f"{trained_model_dir}/{os.getenv('YOLO_TRAINING_NAME')}.pt")
