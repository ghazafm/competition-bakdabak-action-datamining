#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import shutil


# In[ ]:


load_dotenv()
start_time = time.time()
data_version = os.getenv("ROBOFLOW_DATA_VERSION", "1")
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE_NAME")).project(
    os.getenv("ROBOFLOW_PROJECT_NAME")
)
version = project.version(int(data_version))
dataset = version.download("folder", location=f"data/{data_version}")


# In[ ]:


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


# In[ ]:


data_dir = Path(f"data/{data_version}")
old_dir = data_dir / "valid"
new_dir = data_dir / "val"

if old_dir.exists():
    if new_dir.exists():
        print(f"'{new_dir}' already exists. Skipping rename.")
    else:
        os.rename(old_dir, new_dir)
        print(f"Renamed '{old_dir}' → '{new_dir}'")
else:
    print(f"'{old_dir}' not found. Nothing to rename.")

train_name = os.getenv("YOLO_TRAINING_NAME", f"run-{int(start_time)}")


# In[ ]:


imgsz = int(os.getenv("IMG_SIZE", 640))
results = model.train(
    data=data_dir,
    epochs=int(os.getenv("EPOCHS", 10)),
    imgsz=imgsz,
    batch=int(os.getenv("BATCH_SIZE", 10)),
    name=os.getenv("YOLO_TRAINING_NAME"),
    exist_ok=True,
    workers=int(os.getenv("WORKERS", 0)),
    patience=50,
    verbose=True,
)


# In[ ]:


model.val(data=str(data_dir), device=device)


# In[ ]:


trained_model_dir = Path(
    f"model/trained/{data_version}/{yolo_model_version}/{yolo_model_size}/{int(start_time)}"
)
trained_model_dir.mkdir(parents=True, exist_ok=True)

# results.save_dir is the run folder; best.pt is always in weights/
best_path = Path(results.save_dir) / "weights" / "best.pt"
dst_path = trained_model_dir / f"{train_name}.pt"
if best_path.exists():
    shutil.copy2(best_path, dst_path)
    print("Saved best weights to:", dst_path)
else:
    print("[warn] best.pt not found at", best_path)

if best_path.exists():
    exp_model = YOLO(str(best_path))
    exp_model.export(
        format="onnx",
        imgsz=imgsz,
        device=device,
        opset=12,
        dynamic=True,
        simplify=True,
    )

    onnx_candidates = [
        best_path.with_suffix(".onnx"),
        Path(results.save_dir) / "weights" / "best.onnx",
        Path(results.save_dir) / "best.onnx",
    ]
    onnx_src = next((p for p in onnx_candidates if p.exists()), None)
    if onnx_src:
        dst_onnx = trained_model_dir / f"{train_name}.onnx"
        shutil.copy2(onnx_src, dst_onnx)
        print("Saved ONNX export to:", dst_onnx)
    else:
        print("[warn] Could not locate exported ONNX file after export()")

