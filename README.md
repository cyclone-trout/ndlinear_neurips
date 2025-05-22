# NdLinear: Don't Flatten! Building Superior Neural Architectures by Preserving N-D Structure

This repository contains code for ***NdLinear: Don't Flatten! Building Superior Neural Architectures by Preserving N-D Structure*** at NeurIPS 2025.

---

## Table of Contents

- [Requirements](#requirements)
- [Vision Models](#vision-models)
  - [CNN Image Classification](#cnn-image-classification)
  - [Vision Transformer (ViT)](#vision-transformer-vit)
  - [Diffusion Transformer (DiT)](#diffusion-transformer-dit)
- [NLP Models](#nlp-models)
  - [Text Classification with BERT](#text-classification-with-bert)
- [Tabular Models](#tabular-models)
  - [Tabular Data Classification](#tabular-data-classification)
  - [Tabular Data Regression](#tabular-data-regression)
- [Open Pre-trained Transformer (OPT) Models](#open-pre-trained-transformer-opt-models)
- [LLAMA-based Models](#llama-based-models)

---

## Requirements

Install the dependencies with:
```bash
pip install -r requirements.txt
```
---

## Vision Models

### CNN Image Classification

- **File:** `cnn_img_classification.py`
- **Description:** Standard CNN pipeline for image classification tasks.
- **Usage:**
```bash
python cnn_img_classification.py
```

### Vision Transformer (ViT)

- **Files:** `vit/vit.py`, `vit/vit_distill.py`, `vit/ndlinear.py`
- **Description:** Implements Vision Transformer model and a distillation variant.
- **Usage:**
```bash
    torchrun --nnodes 1 --nproc_per_node=4 \
    src/vit_distill.py \
    --num_epochs 30 \
    --num_transformers 6 \
    --dataset CIFAR100 \
    --batch_size 256 \
    --lr 2.75e-4
```

### Diffusion Transformer (DiT)

- **Files:** `dit/train_dit.py`, `dit/model.py`, `dit/mlp_ndlinear.py`, `dit/ndlinear.py`
- **Description:** Implements a DiT model for vision tasks, including custom nonlinear/MLP layers.
- **Usage:**
```bash
python dit/train_dit.py \
  --feature-path /path/to/features \
  --results-dir ./results \
  --model DiT-XS/8 \
  --image-size 256 \
  --num-classes 100 \
  --epochs 1400 \
  --global-batch-size 128 \
  --num-workers 8 \
  --log-every 100 \
  --ckpt-every 50000 \
  --lr 2e-4 \
  --accumulation-steps 1 \
  --use-ndmlp \
  --use-ndtse \
  --mlp-variant 4 \
  --tse-scale-factor 1 \
  --use-num-transforms 2
  ```

---

## NLP Models

### Text Classification with BERT

- **File:** `txt_classify_bert.py`
- **Description:** Uses HuggingFace BERT for text classification.
- **Usage:**
```bash
python txt_classify_bert.py \
  --learning_rate 3e-5 \
  --epochs 10 \
  --batch_size 32 \
  --file_path data/CoLA/train.tsv
```

---

## Tabular Models

### Tabular Data Classification

- **File:** `tabluar/mlp_cls.py`
- **Description:** Compares dense Linear and NdLinear neural architectures for tabular data classification on the Cardio Disease dataset, reporting accuracy and plotting learning curves.
- **Usage:**
```bash
python tabluar/mlp_cls.py
```
This will train both models on `datasets/cardio_disease.csv`, and produce:
- `training_loss_classification.png`
- `test_accuracy_classification.png`

### Tabular Data Regression

- **File:** `tabluar/mlp_reg.py`
- **Description:** Benchmarks Linear vs. NdLinear models for regression on the Delivery Time dataset, comparing mean squared error and saving loss curves.
- **Usage:**
```bash
python tabluar/mlp_reg.py
```
This will train both models on `datasets/delivery_time.csv`, and produce:
- `training_loss_regression.png`
- `test_loss_regression.png`

---

## Open Pre-trained Transformer (OPT) Models

See [`opt/README.md`](opt/README.md) for instructions on using OPT models.

---

## LLAMA-based Models

- **Files:**  
  - `llama/accelerate_llama_control.py`  
  - `llama/evaluate_ndlinear_model.py`  
  - `llama/ndlinear.py`  
  - `llama/Makefile`
- **Description:** Inference, evaluation, and custom (NdLinear) adaptation layers for LLaMA models.

### Usage

**With Makefile:**

```bash
make install
make run_factorized_ndlinear_lora
```

**Manual Execution:**

```bash
python llama/accelerate_llama_control.py \
  --model_name "Qwen/Qwen3-1.7B-Base" \
  --dataset "lmms-lab/Math10K" \
  --lora_type "ndlinear" \
  --output_dir "./output_qwen3_1.7B_math10k_ndlinear_factorized" \
  --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --lora_r 1 \
  --lora_alpha 1 \
  --epochs 2 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --max_length 512 \
  --seed 42
```

**Evaluation:**

```bash
python llama/evaluate_ndlinear_model.py \
  --model_path "./output_qwen3_1.7B_math10k_ndlinear_factorized/ndlinear_lora/" \
  --base_model_name "Qwen/Qwen3-1.7B-Base" \
  --dataset_name "openai/gsm8k" \
  --dataset_config "main" \
  --dataset_split "test" \
  --max_examples 2000 \
  --output_dir "./evaluation_results_test_ndlinear_model"
```

See [`llama/Makefile`](llama/Makefile) and docstrings within the scripts for more advanced options and details.

---
