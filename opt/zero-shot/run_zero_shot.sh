#!/bin/bash

# Set the checkpoint path
CKPT_PATH="/path/to/your/checkpoint"

# List of datasets to run
datasets=(
    # "squad"
    "arc-easy"
    "arc-challenge"
    "openbookqa"
    "hellaswag"
    "piqa"
    "winogrande"
    "copa"
    "boolq"
    "cb"
    "wic"
    "rte"
    "wsc"
)

# Loop through datasets and run the script for each one
for dataset in "${datasets[@]}"; do
    # Set max_input_length: 400 for piqa; otherwise 450
    if [[ "$dataset" == "piqa" ]]; then
        MAX_INPUT_LENGTH=350
    else
        MAX_INPUT_LENGTH=450
    fi

    echo "Running model on $dataset with max_input_length=$MAX_INPUT_LENGTH"
    python zero-shot-eval.py --ckpt_path "$CKPT_PATH" --dataset "$dataset" --max_input_length "$MAX_INPUT_LENGTH" | tee "zero-shot-outputs/opt-1.3b/${dataset}.log"
done