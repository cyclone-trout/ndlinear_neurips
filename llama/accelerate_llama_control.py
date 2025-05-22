#!/usr/bin/env python3

import argparse
import os
import math
import logging
import time
import random
import json
from pathlib import Path
import traceback
import glob
import re  # Move import to top of file

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, PeftModel
import datasets
from datasets import load_dataset
from ndlinear import NdLinear
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneralTextDataset(Dataset):
    """Dataset for handling text data from various sources."""
    def __init__(self, dataset, tokenizer, max_length=512, dataset_name="unknown"):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name # Store dataset name for specific logic
        logger.info(f"[DEBUG] GeneralTextDataset initialized with dataset_name: '{self.dataset_name}'") # DEBUG PRINT

        # Ensure PAD token is set for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer pad_token set to eos_token: {self.tokenizer.eos_token}")

        logger.info(f"Processing dataset: {self.dataset_name}")
        for item_idx, item in enumerate(tqdm(dataset, desc="Processing dataset items")):
            prompt_text = ""
            target_text_for_loss = "" # This will be the part model learns to predict
            full_text_for_tokenizer = "" # This is the complete input to the tokenizer
            # logger.info(f"[DEBUG] Item {item_idx} keys: {item.keys()}") # Optional: very verbose

            # Case -1: CommonsenseQA (SHOULD BE THE FIRST CHECK if dataset_name is specific)
            if self.dataset_name == "commonsense_qa" or self.dataset_name == "tau/commonsense_qa": # DEBUG: Check both forms
                # logger.info(f"[DEBUG] Item {item_idx}: Entering CSQA block. self.dataset_name = '{self.dataset_name}'") # DEBUG PRINT
                if "question" in item and "choices" in item and "answerKey" in item:
                    question_content = str(item["question"]).strip()
                    answer_key = str(item["answerKey"]).strip()

                    choices_lines = []
                    # Ensure choices labels and texts are lists of the same length
                    if isinstance(item["choices"].get("label"), list) and \
                       isinstance(item["choices"].get("text"), list) and \
                       len(item["choices"]["label"]) == len(item["choices"]["text"]):
                        for i, label in enumerate(item["choices"]["label"]):
                            text = item["choices"]["text"][i]
                            choices_lines.append(f"{label}. {text}")
                        choices_str = "\n".join(choices_lines)
                        
                        prompt_text = f"Question: {question_content}\nChoices:\n{choices_str}\nAnswer:"
                        target_text_for_loss = answer_key 
                        full_text_for_tokenizer = prompt_text + " " + target_text_for_loss # Add a space before the answer key
                    else:
                        logger.warning(f"[DEBUG] Item {item_idx}: CSQA block - malformed choices. Item: {item}") # DEBUG PRINT
                        logger.warning(f"Skipping commonsense_qa item due to malformed choices: {item['choices']}")
                        continue # Skip this item
                else:
                    logger.warning(f"[DEBUG] Item {item_idx}: CSQA block - missing fields. Keys: {item.keys()}") # DEBUG PRINT
                    logger.warning(f"Skipping item for commonsense_qa due to missing fields (question, choices, or answerKey): {item.keys()}")
                    continue # Skip this item

            # Case 0: New 3-part template (Instruction, Solution, Final Answer)
            # This is the most specific and preferred format if all fields are present (and not CSQA).
            elif "instruction" in item and "output" in item and "answer" in item:
                # logger.info(f"[DEBUG] Item {item_idx}: Entering 3-part template block.") # DEBUG PRINT
                instruction_content = str(item["instruction"]).strip()
                output_content = str(item["output"]).strip()
                answer_content = str(item["answer"]).strip()

                # Prompt part (will be masked in labels)
                prompt_text = f"### Instruction:\n{instruction_content}\n\n### Solution:\n"
                
                # Target part (model learns to predict this)
                target_text_for_loss = f"{output_content}\n\n### Final Answer:\n\boxed{{{answer_content}}}"

                full_text_for_tokenizer = prompt_text + target_text_for_loss

            # Case 1: Fallback to 'instruction' + 'answer' if 'output' is missing from the 3-part template.
            elif "instruction" in item and "answer" in item:
                instruction_content = str(item["instruction"]).strip()
                input_text_content = item.get("input", "").strip()
                
                if input_text_content:
                    prompt_text = f"{instruction_content}\n{input_text_content}"
                else:
                    prompt_text = instruction_content
                target_text_for_loss = str(item["answer"]).strip()
                full_text_for_tokenizer = prompt_text + "\n" + target_text_for_loss


            # Case 2: Fallback to 'instruction' + 'output' (for reasoning if 'answer' is missing from 3-part).
            elif "instruction" in item and "output" in item:
                instruction_content = str(item["instruction"]).strip()
                input_text_content = item.get("input", "").strip()
                output_content = str(item["output"]).strip()

                if input_text_content:
                    prompt_text = f"{instruction_content}\n{input_text_content}"
                else:
                    prompt_text = instruction_content
                target_text_for_loss = output_content
                full_text_for_tokenizer = prompt_text + "\n" + target_text_for_loss

            # Case 3: Generic 'text' field (e.g., for WikiText).
            elif "text" in item:
                full_text_content = str(item["text"]).strip()
                # For 'text' based datasets, the whole thing is input and target for Causal LM.
                # We effectively set prompt_text to be empty for label masking,
                # so the model learns to predict the entire sequence.
                prompt_text = "" # Or some BOS token if preferred and handled consistently
                target_text_for_loss = full_text_content
                full_text_for_tokenizer = full_text_content
            
            # Case 4: 'question' and 'answer' (e.g., for other QA datasets).
            elif "question" in item and "answer" in item:
                # logger.info(f"[DEBUG] Item {item_idx}: Entering question/answer block.") # DEBUG PRINT
                prompt_text = str(item["question"]).strip()
                target_text_for_loss = str(item["answer"]).strip()
                full_text_for_tokenizer = prompt_text + "\n" + target_text_for_loss
            
            else:
                logger.warning(f"[DEBUG] Item {item_idx}: Falling to final else. self.dataset_name = '{self.dataset_name}', Item keys: {item.keys()}") # DEBUG PRINT
                logger.warning(f"Skipping item due to unrecognized or incomplete format for templating: {item.keys()}")
                continue

            # Add EOS token to the full text that will be tokenized
            if not full_text_for_tokenizer.endswith(self.tokenizer.eos_token):
                full_text_for_tokenizer += self.tokenizer.eos_token

            # Tokenize the full text
            full_tokenized = self.tokenizer(
                full_text_for_tokenizer,
                truncation=True,
                max_length=self.max_length,
                padding=False, 
                return_attention_mask=True
            )

            # Tokenize prompt_text separately to determine its length for label masking
            # No EOS for prompt_text here as it's part of the larger full_text_for_tokenizer
            prompt_tokenized = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length, 
                padding=False,
                return_attention_mask=False
            )
            
            input_ids = full_tokenized['input_ids']
            attention_mask = full_tokenized['attention_mask']
            
            labels = [-100] * len(input_ids) # Initialize all labels to -100 (masked)
            
            prompt_tokens_len = len(prompt_tokenized['input_ids'])

            # Only create labels for the target_text_for_loss part
            # Ensure we don't try to label beyond the length of input_ids (due to truncation of full_text_for_tokenizer)
            
            # The tokens for target_text_for_loss start after prompt_tokens_len
            # and go up to the length of input_ids (excluding any EOS added by tokenizer if not part of target_text_for_loss explicitly)
            
            # If prompt_text is empty (e.g. for pure Causal LM on 'text' field), prompt_tokens_len might be 0 or 1 (for BOS).
            # In this case, target_start_index will be close to 0.
            target_start_index = min(prompt_tokens_len, len(input_ids))

            for i in range(target_start_index, len(input_ids)):
                # Check if the current token is a pad token introduced by full_tokenized due to truncation of target_text_for_loss
                # This check is implicitly handled because labels are initialized to -100
                # and we only overwrite labels for actual tokens from input_ids.
                # The EOS token that we appended to full_text_for_tokenizer should get a valid label.
                labels[i] = input_ids[i]
            
            # Pad input_ids, attention_mask, and labels to max_length
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length
            elif padding_length < 0: # Should not happen if max_length in tokenizer is same as self.max_length
                logger.error(f"Input_ids length ({len(input_ids)}) exceeded max_length ({self.max_length}) after tokenization. This is unexpected.")
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                labels = labels[:self.max_length]

            
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def find_linear_modules(model, target_modules):
    """Find all linear modules matching the specified target names."""
    linear_modules = {}
    
    # Convert target_modules to a set for faster lookup
    target_set = set(target_modules)
    
    # Find all modules matching the target names
    for name, module in model.named_modules():
        if any(target in name for target in target_set) and isinstance(module, nn.Linear):
            linear_modules[name] = module
    
    return linear_modules

def find_factor(n):
    for i in range(int(n ** 0.5), 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)

class NdLinearFactorizedLoRA(nn.Module):
    def __init__(self, d_in, d_out, alpha=1.0, dropout=0.05):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.in_factors = find_factor(d_in)
        self.out_factors = find_factor(d_out)
        self.adapter = NdLinear(
            input_dims=self.in_factors,
            hidden_size=self.out_factors, # Note: NdLinear uses hidden_size for output_dims
            transform_outer=False
        )
        self.scaling = alpha
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        orig = x.shape  # (B, L, d_in)
        x = self.drop(x).view(-1, *self.in_factors)
        y = self.adapter(x).view(*orig[:-1], self.d_out)
        return y * self.scaling

class LinearWithNdLinearFactorizedLoRA(nn.Module):
    def __init__(self, base_layer, alpha=1.0, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.adapter = NdLinearFactorizedLoRA(
            d_in=self.base_layer.in_features,
            d_out=self.base_layer.out_features,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x):
        return self.base_layer(x) + self.adapter(x)

def apply_ndlinear_lora(model, target_modules, alpha=16, dropout=0.05):
    device = next(model.parameters()).device
    wrapped_modules = {}
    linear_modules = find_linear_modules(model, target_modules)
    logger.info(f"Applying NdLinear Factorized LoRA to {len(linear_modules)} modules")
    for name, module in linear_modules.items():
        logger.info(f"  Targeting module: {name} (in={module.in_features}, out={module.out_features})")
    total_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_adapter_params = 0
    for name, module in linear_modules.items():
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        
        wrapped = LinearWithNdLinearFactorizedLoRA(
            module,
            alpha=alpha,
            dropout=dropout
        )
        
        adapter_trainable_params = sum(p.numel() for p in wrapped.adapter.parameters() if p.requires_grad)
        total_adapter_params += adapter_trainable_params
        setattr(parent, child_name, wrapped)
        wrapped_modules[name] = wrapped

    total_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"NdLinear Factorized LoRA trainable parameters: {total_adapter_params:,}")
    logger.info(f"Total trainable parameters before: {total_params_before:,}")
    logger.info(f"Total trainable parameters after: {total_params_after:,}")
    logger.info(f"Parameters added by LoRA: {total_params_after - total_params_before:,}")
    return model, wrapped_modules

def train(model, train_dataloader, optimizer, device, scheduler=None, gradient_accumulation_steps=1, accelerator=None, current_epoch=0, args=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_examples = 0
    
    # Training loop
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {current_epoch+1}")
    for step, batch in enumerate(progress_bar):
        # Move batch to the same device as the model
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Scale loss by gradient accumulation steps
        actual_loss = loss.item()
        loss = loss / gradient_accumulation_steps
        
        # Backward pass with accelerator
        if accelerator is not None:
            accelerator.backward(loss)
            
        else:
            loss.backward()
        
        # Update weights based on gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        # Log loss
        total_loss += actual_loss
        total_examples += batch["input_ids"].size(0)
        
        # Per-step logging to WandB
        if args and args.use_wandb and accelerator and accelerator.is_main_process:
            if (step + 1) % args.log_interval == 0:
                accelerator.log({
                    f"{args.lora_type}_lora/step_loss": actual_loss, 
                    "step_in_epoch": step +1
                })

        progress_bar.set_postfix({"loss": total_loss / (step + 1)})
    
    return total_loss / (step + 1)

def evaluate(model, eval_dataloader, device, accelerator=None):
    """Evaluate the model on the provided dataloader."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Log loss
            total_loss += loss.item()
    
    return total_loss / len(eval_dataloader)

def count_parameters(model):
    """Count trainable and total parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params

def load_dataset_from_huggingface(dataset_id, split="train", sample_size=None):
    """Load a dataset directly from HuggingFace."""
    try:
        dataset = load_dataset(dataset_id, split=split)
        logger.info(f"Loaded {len(dataset)} examples from HuggingFace dataset {dataset_id} (split: {split})")
        
        # Sample if needed
        if sample_size and sample_size < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
            logger.info(f"Sampled {sample_size} examples from dataset")
        
        return dataset
    except Exception as e:
        logger.error(f"Error loading HuggingFace dataset {dataset_id}: {str(e)}")
        raise

def run_experiment(args):
    """Run the experiment comparing Classic LoRA and NdLinear-LoRA."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Initialize accelerator
    # Pass wandb project name to accelerator for automatic initialization
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if not args.force_cpu else None,
        log_with="wandb" if args.use_wandb else None, # Use wandb if enabled
    )

    # Initialize wandb tracker if enabled
    if args.use_wandb and accelerator.is_main_process:
        wandb_kwargs = {}  # Initialize as an empty dict
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        # Accelerator's init_trackers will call wandb.init() and handle config
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": wandb_kwargs})

    # Log distributed training setup
    logger.info(f"Distributed training setup:")
    logger.info(f"  - Process index: {accelerator.process_index}")
    logger.info(f"  - Local process index: {accelerator.local_process_index}")
    logger.info(f"  - World size: {accelerator.num_processes}")
    logger.info(f"  - Device: {accelerator.device}")
    logger.info(f"  - Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Set device based on accelerator
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    
    if args.use_hf_dataset or "/" in args.dataset:  # Detect HuggingFace dataset format
        # Use HuggingFace datasets directly
        args.use_hf_dataset = True
        train_dataset = load_dataset_from_huggingface(args.dataset, split="train", sample_size=args.sample_size)
        
        # For validation/test, use appropriate sample sizes based on train dataset size
        val_sample_size = args.sample_size // 8 if args.sample_size else min(len(train_dataset) // 8, 100)
        
        # For validation/test, either use the validation split if it exists, or sample from train
        try:
            val_dataset = load_dataset_from_huggingface(args.dataset, split="validation", sample_size=val_sample_size)
        except Exception:
            # If no validation split, use a small portion of train
            val_size = min(len(train_dataset) // 10, 100)
            train_size = len(train_dataset) - val_size
            train_val_split = train_dataset.train_test_split(test_size=val_size, seed=42)
            train_dataset = train_val_split["train"]
            val_dataset = train_val_split["test"]
            logger.info(f"Created validation set with {len(val_dataset)} examples from train")
    else:
        # Use local dataset files - default to 1000 if not specified for local files
        sample_size = args.sample_size if args.sample_size is not None else 1000
        train_file = os.path.join(args.data_dir, f"{args.dataset}_train_{sample_size}.jsonl")
        val_file = os.path.join(args.data_dir, f"{args.dataset}_validation_{sample_size}.jsonl")
        
        train_dataset = load_dataset_from_jsonl(train_file)
        val_dataset = load_dataset_from_jsonl(val_file)
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Loaded {len(val_dataset)} validation examples")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token to eos_token")
    
    # Create datasets
    logger.info("Creating datasets")
    train_data = GeneralTextDataset(train_dataset, tokenizer, max_length=args.max_length, dataset_name=args.dataset)
    val_data = GeneralTextDataset(val_dataset, tokenizer, max_length=args.max_length, dataset_name=args.dataset)
    
    if not train_data.examples:
        logger.warning(f"No valid examples found in train_dataset for dataset: {args.dataset}")
        return
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Dictionary to store results
    results = {
        "args": vars(args)
    }
    
    # Early stopping parameters
    patience = args.patience if hasattr(args, 'patience') else 3  # Default patience: 3 epochs
    
    # ------------------------------
    # Train with Classic LoRA
    # ------------------------------
    if args.lora_type == "classic":
        logger.info(f"=== Running Classic LoRA experiment ===")
        
        # Load model for Classic LoRA
        logger.info(f"Loading model: {args.model_name}")
        classic_model = AutoModelForCausalLM.from_pretrained(args.model_name, token=hf_token)
        
        # Apply Classic LoRA
        logger.info(f"Applying Classic LoRA with rank={args.lora_r}")
        target_modules = args.target_modules.split(",")
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        classic_model = get_peft_model(classic_model, lora_config)
        
        # If continuing from a checkpoint, load the existing weights
        if args.continue_from_checkpoint and os.path.exists(args.continue_from_checkpoint):
            logger.info(f"Loading existing Classic LoRA checkpoint from {args.continue_from_checkpoint}")
            try:
                classic_model = PeftModel.from_pretrained(
                    classic_model,
                    args.continue_from_checkpoint,
                    is_trainable=True
                )
                logger.info(f"Successfully loaded Classic LoRA weights")
            except Exception as e:
                logger.error(f"Error loading Classic LoRA checkpoint: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Continuing with freshly initialized weights")
        
        # Create optimizer
        classic_optimizer = torch.optim.AdamW(
            classic_model.parameters(),
            lr=args.learning_rate
        )
        
        # Prepare model, optimizer, and dataloaders with accelerator
        classic_model, classic_optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            classic_model, classic_optimizer, train_dataloader, val_dataloader
        )
        
        # Count parameters
        trainable_params, total_params = count_parameters(classic_model)
        logger.info(f"Classic LoRA - Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        # Train Classic LoRA model with early stopping
        logger.info(f"Training Classic LoRA model for {args.epochs} epochs with early stopping (patience={patience})")
        classic_train_losses = []
        classic_val_losses = []
        
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_without_improvement = 0
        best_model_state = None
        best_model_path = None
        
        for epoch in range(args.epochs):
            # Train
            train_loss = train(
                classic_model, 
                train_dataloader, 
                classic_optimizer, 
                device, # This device is from accelerator, should be correct
                None,  # No scheduler
                args.gradient_accumulation_steps,
                accelerator,
                epoch, # Pass current epoch
                args   # Pass args
            )
            classic_train_losses.append(train_loss)
            if args.use_wandb:
                accelerator.log({"classic_lora/train_loss": train_loss, "epoch": epoch+1})
            
            # Evaluate
            val_loss = evaluate(classic_model, val_dataloader, device, accelerator)
            classic_val_losses.append(val_loss)
            if args.use_wandb:
                accelerator.log({"classic_lora/val_loss": val_loss, "epoch": epoch+1})
            
            logger.info(f"Classic LoRA - Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
            
            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save best model state immediately
                classic_model_dir = os.path.join(args.output_dir, "classic_lora")
                os.makedirs(classic_model_dir, exist_ok=True)
                
                # Remove previous best model if it exists
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                        logger.info(f"Removed previous best model: {best_model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove previous best model: {e}")
                
                # Save new best model
                best_model_path = os.path.join(classic_model_dir, f"pytorch_model_best_epoch_{epoch+1}.bin")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(classic_model)
                accelerator.save(unwrapped_model.state_dict(), best_model_path)
                logger.info(f"New best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")
                logger.info(f"Saved best model to {best_model_path}")
                
                # Also keep the best model state in memory
                best_model_state = {k: v.cpu().clone() for k, v in unwrapped_model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                    break
        
        # Restore best model if we have one
        if best_model_state is not None:
            unwrapped_model = accelerator.unwrap_model(classic_model)
            unwrapped_model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch+1}")
        
        # Save Classic LoRA model
        classic_model_dir = os.path.join(args.output_dir, "classic_lora")
        os.makedirs(classic_model_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(classic_model)
        unwrapped_model.save_pretrained(classic_model_dir)
        
        # Store results
        results["classic_lora"] = {
            "train_losses": classic_train_losses,
            "val_losses": classic_val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch + 1,
            "early_stopped": epochs_without_improvement >= patience
        }
    
    # ------------------------------
    # Train with Factorized NdLinear-LoRA (now just "ndlinear")
    # ------------------------------
    elif args.lora_type == "ndlinear":
        logger.info(f"=== Running NdLinear Factorized LoRA experiment ===")
        logger.info(f"Loading model: {args.model_name}")
        ndlinear_model = AutoModelForCausalLM.from_pretrained(args.model_name, token=hf_token)
        ndlinear_model = ndlinear_model.to(device)
        logger.info(f"Freezing all model parameters before applying NdLinear Factorized LoRA")
        for param in ndlinear_model.parameters():
            param.requires_grad = False
        logger.info(f"Applying NdLinear Factorized LoRA")
        target_modules = args.target_modules.split(",")
        ndlinear_model, wrapped_modules = apply_ndlinear_lora(
            ndlinear_model,
            target_modules,
            alpha=args.lora_alpha,
            dropout=args.dropout
        )
        if args.continue_from_checkpoint and os.path.exists(args.continue_from_checkpoint):
            logger.info(f"Loading existing checkpoint from {args.continue_from_checkpoint}")
            checkpoint_path = os.path.join(args.continue_from_checkpoint, "pytorch_model.bin")
            # If not found, look for best model file
            if not os.path.exists(checkpoint_path):
                best_model_files = glob.glob(os.path.join(args.continue_from_checkpoint, "pytorch_model_best_epoch_*.bin"))
                if best_model_files:
                    # Sort by epoch number and get the latest
                    best_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    checkpoint_path = best_model_files[-1]
                    logger.info(f"Found best model file: {checkpoint_path}")
            
            if os.path.exists(checkpoint_path):
                try:
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    incompatible_keys = ndlinear_model.load_state_dict(state_dict, strict=False)
                    if incompatible_keys.missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint: {len(incompatible_keys.missing_keys)}")
                    if incompatible_keys.unexpected_keys:
                        logger.warning(f"Unexpected keys when loading checkpoint: {len(incompatible_keys.unexpected_keys)}")
                    logger.info(f"Successfully loaded weights from {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    logger.error(traceback.format_exc())
                    logger.warning("Continuing with freshly initialized weights")
            else:
                logger.warning(f"No pytorch_model.bin found in {args.continue_from_checkpoint}")
        
        ndlinear_model.to(device) # Ensure model is on the correct device after potential checkpoint loading
        
        # Prepare model, optimizer, and dataloaders with accelerator for NdLinear LoRA
        ndlinear_optimizer = torch.optim.AdamW(
            [p for p in ndlinear_model.parameters() if p.requires_grad],
            lr=args.learning_rate
        )
        ndlinear_model, ndlinear_optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            ndlinear_model, ndlinear_optimizer, train_dataloader, val_dataloader
        )

        trainable_params, total_params = count_parameters(ndlinear_model)
        logger.info(f"NdLinear Factorized LoRA - Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        logger.info(f"Training NdLinear Factorized LoRA model for {args.epochs} epochs with early stopping (patience={patience})")
        ndlinear_train_losses = []
        ndlinear_val_losses = []
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_without_improvement = 0
        best_model_state = None
        best_model_path = None
        for epoch in range(args.epochs):
            train_loss = train(
                ndlinear_model, 
                train_dataloader, 
                ndlinear_optimizer, 
                device, 
                None,  # No scheduler
                args.gradient_accumulation_steps,
                accelerator,
                epoch, # Pass current epoch
                args   # Pass args
            )
            ndlinear_train_losses.append(train_loss)
            if args.use_wandb:
                accelerator.log({"ndlinear_lora/train_loss": train_loss, "epoch": epoch+1})
            
            val_loss = evaluate(ndlinear_model, val_dataloader, device, accelerator)
            ndlinear_val_losses.append(val_loss)
            if args.use_wandb:
                accelerator.log({"ndlinear_lora/val_loss": val_loss, "epoch": epoch+1})
            
            logger.info(f"NdLinear Factorized LoRA - Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                ndlinear_model_dir = os.path.join(args.output_dir, "ndlinear_lora") # Changed from ndlinear_factorized_lora
                os.makedirs(ndlinear_model_dir, exist_ok=True)
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                        logger.info(f"Removed previous best model: {best_model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove previous best model: {e}")
                best_model_path = os.path.join(ndlinear_model_dir, f"pytorch_model_best_epoch_{epoch+1}.bin")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(ndlinear_model)
                accelerator.save(unwrapped_model.state_dict(), best_model_path)
                logger.info(f"New best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")
                logger.info(f"Saved best model to {best_model_path}")
                best_model_state = {k: v.cpu().clone() for k, v in unwrapped_model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                    break
        if best_model_state is not None:
            unwrapped_model = accelerator.unwrap_model(ndlinear_model)
            unwrapped_model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch+1}")
        ndlinear_model_dir = os.path.join(args.output_dir, "ndlinear_lora") # Changed from ndlinear_factorized_lora
        os.makedirs(ndlinear_model_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(ndlinear_model)
        torch.save(unwrapped_model.state_dict(), os.path.join(ndlinear_model_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(ndlinear_model_dir)
        with open(os.path.join(ndlinear_model_dir, "ndlinear_lora_config.json"), "w") as f: # Changed from ndlinear_factorized_lora_config.json
            wrapped_info = {}
            for name, module in wrapped_modules.items():
                if hasattr(module.adapter, "in_factors"):
                    wrapped_info[name] = {
                        "in_features": module.adapter.d_in,
                        "out_features": module.adapter.d_out,
                        "in_factors": list(module.adapter.in_factors),
                        "out_factors": list(module.adapter.out_factors)
                    }
            json.dump({
                "model_name": args.model_name,
                "lora_alpha": args.lora_alpha,
                "target_modules": target_modules,
                "wrapped_modules": wrapped_info,
                "best_epoch": best_epoch + 1,
                "early_stopped": epochs_without_improvement >= patience,
                "best_val_loss": best_val_loss
            }, f, indent=2)
        results["ndlinear_lora"] = { # Changed from ndlinear_factorized_lora
            "train_losses": ndlinear_train_losses,
            "val_losses": ndlinear_val_losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch + 1,
            "early_stopped": epochs_without_improvement >= patience
        }
    
    # ------------------------------
    # Save combined results
    # ------------------------------
    logger.info(f"Saving combined results")
    # Ensure accelerator.end_training() is called to close wandb run
    if args.use_wandb:
        accelerator.end_training()

    with open(os.path.join(args.output_dir, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"=== Experiment completed successfully ===")
    if "classic_lora" in results:
        logger.info(f"Classic LoRA - Best validation loss: {results['classic_lora']['best_val_loss']:.4f}")
        logger.info(f"Classic LoRA - Best epoch: {results['classic_lora']['best_epoch']}")
    if "ndlinear_lora" in results:
        logger.info(f"NdLinear Factorized LoRA - Best validation loss: {results['ndlinear_lora']['best_val_loss']:.4f}")
        logger.info(f"NdLinear Factorized LoRA - Best epoch: {results['ndlinear_lora']['best_epoch']}")

def main():
    parser = argparse.ArgumentParser(description="TinyLlama Control Experiment: Classic LoRA vs NdLinear Factorized LoRA")
    
    # Model args
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Model name or path")
    parser.add_argument("--target_modules", type=str, 
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of module names to apply LoRA")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, or cpu). If not specified, will use CUDA if available, then MPS, then CPU.")
    parser.add_argument("--continue_from_checkpoint", type=str, default=None,
                        help="Path to existing model checkpoint to continue training from")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training type")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to use. Can be 'math10k', 'commonsense170k', 'alpaca' or a HuggingFace dataset ID (e.g., 'lmms-lab/Math10K')")
    parser.add_argument("--data_dir", type=str, default="./dataset_subsets",
                        help="Directory containing the dataset files (for local datasets)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size for dataset. If not provided, uses the full dataset (for HuggingFace) or 1000 (for local datasets)")
    parser.add_argument("--use_hf_dataset", action="store_true",
                        help="Whether to load the dataset directly from HuggingFace")
    
    # LoRA args
    parser.add_argument("--lora_alpha", type=int, default=1,
                        help="LoRA alpha scaling factor (used by both classic and ndlinear LoRA)")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA layers")
    # lora_r is still needed for classic LoRA
    parser.add_argument("--lora_r", type=int, default=1, help="LoRA rank (used by classic LoRA)")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--patience", type=int, default=1,
                        help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log training step loss to WandB every N steps")
    
    # LoRA type selection
    parser.add_argument("--lora_type", type=str, choices=["classic", "ndlinear"], required=True,
                        help="Type of LoRA to run: classic (PEFT LoRA) or ndlinear (NdLinear Factorized LoRA)")
    
    # Output args
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Wandb args
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="ndlinear_experiments",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name (defaults to auto-generated)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (team name)")

    # New dataset args
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Dataset name from Hugging Face datasets library (e.g., wikitext, lmms-lab/Math10K, commonsense_qa)")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration name (e.g., wikitext-2-raw-v1, or None for datasets like lmms-lab/Math10K or commonsense_qa)")
    parser.add_argument("--validation_split_percentage", type=float, default=0.1,
                        help="Percentage of training data to use for validation if no standard validation split is available (e.g., for Math10K). Only used if not streaming.")
    parser.add_argument("--validation_split_size", type=int, default=1000,
                        help="Number of samples to take for validation from the start of a streaming dataset if no standard validation split is available.")
    parser.add_argument("--streaming_dataset", action="store_true", help="Enable dataset streaming")
    
    args = parser.parse_args()

    # Check if dataset_config_name should be None for specific datasets
    if args.dataset_name == "lmms-lab/Math10K" or args.dataset_name == "commonsense_qa":
        args.dataset_config_name = None # These datasets don't use a config name or use their default
        logger.info(f"Setting dataset_config_name to None for {args.dataset_name}")

    results = run_experiment(args)

if __name__ == "__main__":
    main() 

