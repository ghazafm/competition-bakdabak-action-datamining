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


if os.getenv("DEVICE"):
    device = os.getenv("DEVICE")
    print(f"Device overridden by env: {device}")

model.to(device)


data_root = Path(f"data/{data_version}").resolve()


if data_root.name.lower() == "train":
    data_root = data_root.parent

train_dir = (data_root / "train") if (data_root / "train").exists() else data_root


val_candidates = [
    data_root / "val",
    data_root / "valid",
    data_root / "validation",
]
val_dir = next((p for p in val_candidates if p.exists() and p.is_dir()), None)

if val_dir is not None:
    yolo_data_arg = {"train": str(train_dir), "val": str(val_dir)}
    print(f"Resolved dataset - train: {train_dir}, val: {val_dir}")
else:
    yolo_data_arg = {"train": str(train_dir), "val": str(train_dir)}
    print(
        f"Resolved dataset - train: {train_dir}. No validation folder found; using train for validation as well."
    )

results = model.train(
    data=yolo_data_arg,
    epochs=int(os.getenv("EPOCHS", 10)),
    imgsz=int(os.getenv("IMG_SIZE", 640)),
    batch=int(os.getenv("BATCH_SIZE", 10)),
    name=os.getenv("YOLO_TRAINING_NAME"),
    exist_ok=True,
    workers=int(os.getenv("WORKERS", 0)),
    patience=50,
    verbose=True,
)

trained_model_dir = f"model/trained/{data_version}/{yolo_model_version}/{yolo_model_size}/{int(start_time)}"
os.makedirs(trained_model_dir, exist_ok=True)
model.save(f"{trained_model_dir}/{os.getenv('YOLO_TRAINING_NAME')}.pt")
