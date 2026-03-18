#!/bin/bash

# ==========================================
# DFL Experiment Runner
# ==========================================

# Data parameters
DATASET="Knapsack_Capacity_Dataset"
DATA_PATH="datasets/KP/knapsack_capacity_data.pt"

# Model parameters
MODEL="PFL_Baseline"
HIDDEN_DIM=128

# Training parameters
TRAIN_SPLIT=0.7
VAL_SPLIT=0.15
EPOCHS=1000
BATCH_SIZE=32
LR=0.01
SEED=0
NORMALIZE_INPUT=1
NORMALIZE_OUTPUT=0

# Solver parameters
SOLVER="Knapsack_Solver"

# ==========================================
# Execution
# ==========================================

echo "Starting experiment with MODEL: $MODEL, HIDDEN_DIM: $HIDDEN_DIM, LR: $LR"

python -m src.main \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --hidden_dim $HIDDEN_DIM \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --normalize_input $NORMALIZE_INPUT \
    --normalize_output $NORMALIZE_OUTPUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --solver "$SOLVER"

echo "Experiment finished!"