#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import os


# In[ ]:


load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE_NAME")).project(os.getenv("ROBOFLOW_PROJECT_NAME"))
version = project.version(1)
dataset = version.download("folder",location="data")


# In[ ]:


model = YOLO("model/yolo/yolo11l-cls.pt")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
model.to(device)


# In[ ]:


results = model.train(
    data="data/train",
    epochs=10,
    imgsz=640,
    batch=10,
    name="yolo11l-bakdabak-action-datamining",
    exist_ok=True,
)


# In[ ]:


model.save("model/trained/yolo11l-bakdabak-action-datamining.pt")

