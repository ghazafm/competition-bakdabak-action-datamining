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

# Ensure YOLO finds validation: if 'val' missing but 'valid' or 'validation' exists, create a 'val' symlink
val_path = data_root / "val"
if not val_path.exists():
    for alt in ("valid", "validation"):
        alt_path = data_root / alt
        if alt_path.exists() and alt_path.is_dir():
            try:
                os.symlink(alt_path, val_path)
                print(f"Created symlink: {val_path} -> {alt_path}")
            except FileExistsError:
                pass
            except OSError as e:
                print(f"Warning: could not create symlink {val_path} -> {alt_path} ({e}). YOLO may auto-split.")
            break

train_dir = (data_root / "train") if (data_root / "train").exists() else data_root
if val_path.exists() and val_path.is_dir():
    print(f"Resolved dataset - train: {train_dir}, val: {val_path}")
else:
    print("No validation folder detected; Ultralytics will auto-split from training data.")

# For Ultralytics classification, pass the dataset root path (string), not a dict
yolo_data_arg = str(data_root)

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
