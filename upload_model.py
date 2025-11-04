#!/usr/bin/env python
"""
Upload YOLOv8 classification model to Roboflow.

This script uploads a trained classification model to Roboflow using the Python API,
which properly handles classification models unlike the CLI.
"""

import os

from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    # Load environment variables
    load_dotenv()

    # Configuration
    WORKSPACE_ID = os.getenv("ROBOFLOW_WORKSPACE_NAME", "")
    PROJECT_ID = os.getenv("ROBOFLOW_PROJECT_NAME", "")
    VERSION = os.getenv("ROBOFLOW_DATA_VERSION", "2")
    MODEL_PATH = "."
    MODEL_FILE = "model.pt"
    MODEL_NAME = f"yolo{os.getenv('YOLO_MODEL_VERSION', '8')}{os.getenv('YOLO_MODEL_SIZE', 'l')}-cls.pt"

    print("Configuration:")
    print(f"  Workspace: {WORKSPACE_ID}")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Version: {VERSION}")
    print(f"  Model File: {MODEL_FILE}")
    print()

    # Initialize Roboflow
    print("Initializing Roboflow...")
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

    # Get workspace and project
    print(f"Loading workspace: {WORKSPACE_ID}")
    workspace = rf.workspace(WORKSPACE_ID)

    print(f"Loading project: {PROJECT_ID}")
    project = workspace.project(PROJECT_ID)

    # Get version
    print(f"Getting version: {VERSION}")
    version = project.version(VERSION)

    # Upload model
    # For classification models, we need to use a different approach
    print(f"Uploading model: {MODEL_FILE}")
    print(
        "Note: For classification models, you may need to upload via Roboflow web interface"
    )
    print("or convert to detection format.")

    # Alternative: Try deploying with classification type
    try:
        # This may work for some model types
        version.deploy("yolov8-cls", MODEL_PATH, MODEL_FILE)
        print("✓ Model uploaded successfully!")
    except AttributeError as e:
        print(f"\n✗ Error: {e}")
        print("\n" + "=" * 70)
        print("WORKAROUND: Upload manually via Roboflow web interface")
        print("=" * 70)
        print(f"""
1. Go to: https://app.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}/{VERSION}
2. Click on "Deploy" or "Upload Model"
3. Select "YOLOv8 Classification" as the model type
4. Upload the file: {MODEL_FILE}
5. Set model name as: {MODEL_NAME}

Alternatively, if you need to use this model for inference:
- Export your model to a format supported by Roboflow
- Or use the model locally with Ultralytics for inference
        """)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("Please check your Roboflow project settings and model file.")


if __name__ == "__main__":
    main()
