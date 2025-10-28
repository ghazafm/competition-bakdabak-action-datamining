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
- `WORKERS`: Number of dataloader workers (default: 0, recommended to avoid shared memory issues)

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

## Performance Tuning

### Training Speed Optimization

Training speed depends on several factors:

#### Check Your Shared Memory

```bash
# Run the shared memory check script
bash check_shm.sh
```

Based on the output:
- **If you have > 2GB shared memory**: You can use `WORKERS=2` or `WORKERS=4`
- **If limited shared memory**: Keep `WORKERS=0`

#### Recommended Settings for A100 GPU

**Fast Training (if shared memory is available):**
```bash
DEVICE=cuda
BATCH_SIZE=128
WORKERS=2  # or 4 if you have plenty of shm
EPOCHS=10
```

**Stable Training (always works):**
```bash
DEVICE=cuda
BATCH_SIZE=128
WORKERS=0  # Slower but no memory issues
EPOCHS=10
```

**Memory-Constrained:**
```bash
DEVICE=cuda
BATCH_SIZE=64
WORKERS=0
EPOCHS=10
```

### Understanding Training Progress

When training starts, you'll see:
1. **Training phase**: `Epoch X/Y` with progress bar showing batches
2. **Validation phase**: May appear to pause (especially with `workers=0`) while validating
3. **Next epoch**: Continues automatically

**Example output:**
```
Epoch 1/10    GPU_mem  loss  Instances  Size
1/10          58.4G    2.767    36      640: 100% 10/10 44.6s
(validation running...)
Epoch 2/10    ...
```

If training appears stuck after an epoch completes, **wait 30-60 seconds** - it's likely running validation.

## Troubleshooting

### DataLoader bus error / shared memory issues

**Symptoms:**
- Error: "DataLoader worker is killed by signal: Bus error"
- Error: "insufficient shared memory (shm)"
- Training crashes when loading data

**Root Cause:**
DataLoader workers require shared memory (`/dev/shm`) which may be limited in Docker containers or some systems.

**Solutions:**

#### Option 1: Use 0 workers (Recommended - Always Works)

In your `.env`:
```bash
WORKERS=0
```

This disables multiprocessing but is the most reliable solution. Training will still be fast on GPU.

#### Option 2: Increase shared memory (Docker/Container environments)

If running in Docker, increase shared memory:
```bash
docker run --shm-size=2g ...
```

For Kubernetes/cloud environments, check your pod's `/dev/shm` mount size.

#### Option 3: Use fewer workers

Try reducing workers gradually:
```bash
WORKERS=1  # or 2
```

### Training hangs at first epoch

**Fix Applied:** The code now uses `workers=0` by default to prevent both:
- Multiprocessing hangs on CPU
- Shared memory issues on GPU

### Out of memory errors

Reduce batch size in `.env`:
```bash
BATCH_SIZE=4  # Or even 2 for very limited memory
```

### NumPy compatibility errors

**Error:** "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"

**Fix:** Already handled in `pyproject.toml` with `numpy<2.0.0` constraint.

### Still having issues?

Check the training logs in: `runs/classify/yolo11l-bakdabak-action-datamining/`
