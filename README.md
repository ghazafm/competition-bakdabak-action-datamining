# competition-bakdabak-action-datamining

## Requirements

- **Python**: 3.10.x only
- **GPU**: NVIDIA GPU with CUDA 11.6+ (recommended for training)
- **RAM**: 8GB+ minimum

## Setup

### 1. Install Python 3.10

This project requires Python 3.10 specifically. Install via:

```bash
# Using pyenv (recommended)
pyenv install 3.10.15
pyenv local 3.10.15

# Or using Homebrew (macOS)
brew install python@3.10
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `ROBOFLOW_API_KEY`: Your Roboflow API key
- `ROBOFLOW_WORKSPACE_NAME`: Your workspace name
- `ROBOFLOW_PROJECT_NAME`: Your project name
- `YOLO_MODEL_NAME`: Model filename (e.g., `yolo11l-cls.pt`)
- `YOLO_TRAINING_NAME`: Name for your training run
- `EPOCHS`: Number of training epochs (default: 10)
- `IMG_SIZE`: Image size (default: 640)
- `BATCH_SIZE`: Batch size (default: 10)
- `DEVICE`: (Optional) Force device selection: `cuda`, `mps`, or `cpu`

## CUDA/GPU Issues

### Issue: Training stops or hangs

**Symptoms:**
- Training stops after showing "Epoch GPU_mem loss Instances Size"
- Warning: "The NVIDIA driver on your system is too old"
- Falls back to CPU even with GPU available

**Root Cause:**
Your PyTorch installation expects a newer CUDA version than what's available on your system.

**Solution:**

#### Option 1: Install PyTorch for CUDA 11.6 (Recommended for your system)

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install PyTorch compiled for CUDA 11.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

#### Option 2: Force CPU Training (slower but reliable)

Set in your `.env`:
```bash
DEVICE=cpu
BATCH_SIZE=4  # Reduce batch size for CPU
EPOCHS=5      # Reduce epochs for faster testing
```

#### Option 3: Update NVIDIA Driver (requires system admin access)

Update your NVIDIA driver to version 470.141.03 or newer.

### Verify GPU Detection

After fixing CUDA, verify GPU is detected:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output for working GPU:
```
CUDA available: True
GPU: NVIDIA A100-SXM4-80GB
```

## Running Training

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # Or: uv venv && source .venv/bin/activate

# Run training
python notebook.py
# Or with uv:
uv run notebook.py
```

## Troubleshooting

### Training hangs at first epoch

**Fix Applied:** The code now automatically:
- Sets `workers=0` when using CPU (prevents multiprocessing hang)
- Sets `workers=4` when using GPU (better performance)
- Properly detects CUDA > MPS > CPU in that priority order

### Out of memory errors

Reduce batch size in `.env`:
```bash
BATCH_SIZE=4  # Or even 2 for very limited memory
```

### Still having issues?

Check the training logs in: `runs/classify/yolo11l-bakdabak-action-datamining/`
