#!/bin/bash
# E2V Training Script for AutoDL
# ================================
# Usage: bash run_train.sh

set -e

# ---------- Configuration ----------
# Change these paths to match your AutoDL setup
DATA_ROOT="${DATA_ROOT:-/root/autodl-fs/synthevox3d/SynthEVox3D-Tiny}"
SPLIT_CSV="${DATA_ROOT}/data_split/SynthEVox3D-Tiny_data_split.csv"
CONFIG_PATH="event3d/configs/default.yaml"
BATCH_SIZE="${BATCH_SIZE:-5}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda}"

echo "========================================="
echo " E2V Training"
echo "========================================="
echo "Data:    $DATA_ROOT"
echo "Config:  $CONFIG_PATH"
echo "Batch:   $BATCH_SIZE"
echo "Epochs:  $EPOCHS"
echo "Device:  $DEVICE"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

# Verify data exists
if [ ! -d "$DATA_ROOT/event_3d_scan_tiny" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT/event_3d_scan_tiny"
    echo "Upload SynthEVox3D-Tiny first:"
    echo "  scp -rP <port> SynthEVox3D-Tiny root@<host>:/root/autodl-fs/synthevox3d/"
    exit 1
fi

# Run training
python event3d/train.py \
    --data_root "$DATA_ROOT" \
    --split_csv "$SPLIT_CSV" \
    --config "$CONFIG_PATH" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS"

echo "========================================="
echo " Training complete"
echo " Checkpoints in: event3d/checkpoints/"
echo " TensorBoard:    event3d/logs/"
echo "========================================="
echo ""
echo "Run evaluation:"
echo "  python event3d/evaluate.py \\"
echo "    --checkpoint event3d/checkpoints/best_model.pth \\"
echo "    --data_root $DATA_ROOT \\"
echo "    --split_csv $SPLIT_CSV \\"
echo "    --split test"
