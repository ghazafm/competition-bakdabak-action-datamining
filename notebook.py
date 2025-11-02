#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import time


# In[ ]:


load_dotenv()
start_time = time.time()
data_version = os.getenv("ROBOFLOW_DATA_VERSION", "1")
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE_NAME")).project(os.getenv("ROBOFLOW_PROJECT_NAME"))
version = project.version(2)
dataset = version.download("folder",location=f"data/{data_version}")


# In[ ]:


model = YOLO("model/yolo/"+os.getenv("YOLO_MODEL_NAME"))
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


# In[ ]:


results = model.train(
    data="data/train",
    epochs=int(os.getenv("EPOCHS", 10)),
    imgsz=int(os.getenv("IMG_SIZE", 640)),
    batch=int(os.getenv("BATCH_SIZE", 10)),
    name=os.getenv("YOLO_TRAINING_NAME"),
    exist_ok=True,
    workers=int(os.getenv("WORKERS", 0)),  # Default to 0 to avoid shared memory issues
    patience=50,  # Early stopping patience
    verbose=True,
)


# In[ ]:


trained_model_dir = f"model/trained/{data_version}/{int(start_time)}"
os.makedirs(trained_model_dir, exist_ok=True)
model.save(f"{trained_model_dir}/{os.getenv('YOLO_TRAINING_NAME')}.pt")

