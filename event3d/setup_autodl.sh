#!/bin/bash
# E2V AutoDL One-Click Setup Script
# ===================================
# Usage: bash setup_autodl.sh
# Tested on: AutoDL Ubuntu 22.04 + CUDA 11.8 / 12.1

set -e

echo "========================================="
echo " E2V Event-to-Voxel 3D Reconstruction"
echo " AutoDL Deployment Setup"
echo "========================================="

# ---------- 1. Environment ----------
echo "[1/5] Setting up conda environment..."
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true

# Check CUDA
echo "  CUDA version:"
nvcc --version 2>/dev/null | grep release || echo "  (checking nvidia-smi)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  GPU info unavailable"

# ---------- 2. Create env & install PyTorch ----------
echo "[2/5] Installing PyTorch with CUDA..."

# Detect CUDA version
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K\d+\.\d+' || echo "11.8")
echo "  Detected CUDA: $CUDA_VER"

if [ -n "$CONDA_DEFAULT_ENV" ]; then
    # Already in conda, install directly
    if echo "$CUDA_VER" | grep -q "^12"; then
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    fi
else
    # Create new conda env
    conda create -n e2v python=3.10 -y
    conda activate e2v
    if echo "$CUDA_VER" | grep -q "^12"; then
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    fi
fi

python -c "import torch; print('  PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# ---------- 3. Install dependencies ----------
echo "[3/5] Installing Python dependencies..."
pip install numpy scipy matplotlib pyyaml tqdm tensorboard

# ---------- 4. Clone repo (if not already cloned) ----------
echo "[4/5] Cloning repository..."
REPO_DIR="/root/e2v"
if [ -d "$REPO_DIR" ]; then
    echo "  Repository already exists, pulling latest..."
    cd "$REPO_DIR" && git pull
else
    git clone git@github.com:TermInaL1111/lw.git "$REPO_DIR"
    # Or use HTTPS if SSH not configured:
    # git clone https://github.com/TermInaL1111/lw.git "$REPO_DIR"
fi

# ---------- 5. Configure data paths ----------
echo "[5/5] Configuring data paths..."
cd "$REPO_DIR"

# AutoDL network storage (persistent across instances)
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/synthevox3d}"
echo "  Data root: $DATA_ROOT"
echo "  (Change DATA_ROOT env var if you stored dataset elsewhere)"

echo ""
echo "========================================="
echo " SETUP COMPLETE"
echo "========================================="
echo ""
echo "Dataset preparation (do this once):"
echo "  1. Upload SynthEVox3D-Tiny to ${DATA_ROOT}/"
echo "  2. Ensure structure: ${DATA_ROOT}/SynthEVox3D-Tiny/event_3d_scan_tiny/"
echo "  3. Ensure split CSV: ${DATA_ROOT}/SynthEVox3D-Tiny/data_split/"
echo ""
echo "Start training:"
echo "  cd ${REPO_DIR}"
echo "  python event3d/train.py \\"
echo "    --data_root ${DATA_ROOT}/SynthEVox3D-Tiny \\"
echo "    --split_csv ${DATA_ROOT}/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv \\"
echo "    --config event3d/configs/default.yaml \\"
echo "    --device cuda \\"
echo "    --batch_size 5"
echo ""
echo "Evaluation:"
echo "  python event3d/evaluate.py \\"
echo "    --checkpoint event3d/checkpoints/best_model.pth \\"
echo "    --data_root ${DATA_ROOT}/SynthEVox3D-Tiny \\"
echo "    --split_csv ${DATA_ROOT}/SynthEVox3D-Tiny/data_split/SynthEVox3D-Tiny_data_split.csv \\"
echo "    --split test"
