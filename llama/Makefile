# Makefile for accelerate_llama_control.py

# Environment variables
CONDA_ENV_NAME = llama-ndlinear
PYTHON_VERSION = 3.10
CUDA_VERSION = 11.8

# Default values
MODEL_NAME ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATASET ?= tatsu-lab/alpaca
BATCH_SIZE ?= 4
GRAD_ACCUM ?= 4
LEARNING_RATE ?= 2e-4
NUM_EPOCHS ?= 3
MAX_LENGTH ?= 512
RANK ?= 8
ALPHA ?= 16
DROPOUT ?= 0.1
OUTPUT_DIR ?= ./output
NUM_GPUS ?= 8

# Environment name
ENV_NAME := llama-ndlinear-env

# Accelerate config
ACCELERATE_CONFIG := accelerate_config.yaml

# Default target
.PHONY: all
all: install

.PHONY: setup train clean test-math10k install run run_factorized_ndlinear_lora

install:
	@echo "Setting up Miniconda environment..."
	@if [ ! -d "$$HOME/miniconda3" ]; then \
		echo "Installing Miniconda..."; \
		wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
		bash miniconda.sh -b -p $$HOME/miniconda3; \
		rm miniconda.sh; \
	fi
	@echo "Creating conda environment..."
	$$HOME/miniconda3/bin/conda create -y -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)
	@echo "Installing PyTorch with CUDA $(CUDA_VERSION)..."
	$$HOME/miniconda3/bin/conda run -n $(CONDA_ENV_NAME) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@echo "Installing other requirements..."
	$$HOME/miniconda3/bin/conda run -n $(CONDA_ENV_NAME) pip install -r requirements.txt
	@echo "Installation complete! Activate the environment with: conda activate $(CONDA_ENV_NAME)"

setup:
	@echo "Setting up accelerate configuration..."
	. $(ENV_NAME)/bin/activate && \
	accelerate config default \
		--config_file accelerate_config.yaml \
		--mixed_precision no \
		--gradient_accumulation_steps 4 \
		--gradient_clipping 1.0 \
		--machine_type local \
		--compute_environment local \
		--distributed_type no \
		--downcast_bf16 no \
		--use_mps_device yes
	@echo "Setup complete. You can now run 'make train' to start training."

train:
	@echo "Starting training on $(NUM_GPUS) H100 GPUs..."
	accelerate launch --config_file $(ACCELERATE_CONFIG) \
		accelerate_llama_control.py \
		--model_name $(MODEL_NAME) \
		--dataset $(DATASET) \
		--batch_size $(BATCH_SIZE) \
		--gradient_accumulation_steps $(GRAD_ACCUM) \
		--learning_rate $(LEARNING_RATE) \
		--num_epochs $(NUM_EPOCHS) \
		--max_length $(MAX_LENGTH) \
		--rank $(RANK) \
		--alpha $(ALPHA) \
		--dropout $(DROPOUT) \
		--output_dir $(OUTPUT_DIR)

test-math10k:
	@echo "Starting Math10K test with Llama-3.2-1B..."
	accelerate launch --config_file $(ACCELERATE_CONFIG) \
		accelerate_llama_control.py \
		--model_name meta-llama/Llama-3.2-1B \
		--dataset lmms-lab/Math10K \
		--batch_size 8 \
		--gradient_accumulation_steps 8 \
		--learning_rate 1e-4 \
		--num_epochs 5 \
		--max_length 1024 \
		--rank 16 \
		--alpha 32 \
		--dropout 0.1 \
		--output_dir ./output_math10k \
		--lora_type ndlinear \
		--use_hf_dataset \
		--sample_size 1000

run:
	@echo "Running experiment..."
	$$HOME/miniconda3/bin/conda run -n $(CONDA_ENV_NAME) accelerate launch accelerate_llama_control.py \
		--model_name "meta-llama/Llama-3.2-1B" \
		--dataset "lmms-lab/Math10K" \
		--lora_type "ndlinear" \
		--output_dir "./output" \
		--batch_size 2 \
		--learning_rate 1e-3 \
		--epochs 2 \
		--mixed_precision "bf16" \
		--gradient_accumulation_steps 8 \
		--sample_size 10 \
		--device "cuda"

run_factorized_ndlinear_lora:
	accelerate launch accelerate_llama_control.py \
		--model_name "meta-llama/Llama-3.2-1B" \
		--lora_type "ndlinear_factorized" \
		--lora_alpha 16 \
		--target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
		--dataset "lmms-lab/Math10K" \
		--sample_size 10 \
		--batch_size 1 \
		--epochs 2 \
		--output_dir "./test_output" \
		--device "cuda"

clean:
	@echo "Cleaning up..."
	rm -f $(ACCELERATE_CONFIG)
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(ENV_NAME)
	rm -rf ./test_output
	$$HOME/miniconda3/bin/conda env remove -n $(CONDA_ENV_NAME)

# Help target
help:
	@echo "Available targets:"
	@echo "  install       - Set up Python environment and install dependencies"
	@echo "  setup         - Configure accelerate for multi-GPU training"
	@echo "  train         - Start training on multiple H100 GPUs"
	@echo "  test-math10k  - Run test with Llama-3.2-1B on Math10K dataset"
	@echo "  run           - Run a minimal test with small sample size"
	@echo "  run_factorized_ndlinear_lora - Run a test with ndlinear_factorized LoRA type"
	@echo "  clean         - Clean up configuration and output files"
	@echo ""
	@echo "Variables (with defaults):"
	@echo "  MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	@echo "  DATASET=tatsu-lab/alpaca"
	@echo "  BATCH_SIZE=4"
	@echo "  GRAD_ACCUM=4"
	@echo "  LEARNING_RATE=2e-4"
	@echo "  NUM_EPOCHS=3"
	@echo "  MAX_LENGTH=512"
	@echo "  RANK=8"
	@echo "  ALPHA=16"
	@echo "  DROPOUT=0.1"
	@echo "  OUTPUT_DIR=./output"
	@echo "  NUM_GPUS=8" 