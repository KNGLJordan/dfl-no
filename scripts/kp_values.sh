#!/bin/bash

# ==========================================
# DFL Experiment Runner
# ==========================================

# Data parameters
DATASET="Knapsack_Values_Dataset"
DATA_PATH="datasets/KP/knapsack_values_data.pt"

# Model parameters
MODEL="PFL_Baseline"
HIDDEN_DIM=256

# Training parameters
TRAIN_SPLIT=0.7
VAL_SPLIT=0.15
EPOCHS=2000
BATCH_SIZE=32
LR=0.01
SEED=0

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
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED \
    --solver "$SOLVER"

echo "Experiment finished!"