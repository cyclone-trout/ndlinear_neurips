#!/usr/bin/env python3
'''
Script to evaluate NdLinear models on benchmark datasets.
This is specifically designed for models trained with NdLinear architecture.
'''

import argparse
import json
import logging
import torch
import os
import re
import sys
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn as nn
import math
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ndlinear import NdLinear
except ImportError:
    logger.warning("Could not import NdLinear directly, will try to use it from the model")
    NdLinear = None

# Added class definitions from tiny_llama_control.py
# NdLinear implementation of a LoRA adapter.
class NdLinearLoraLayer(nn.Module):
    """
    NdLinear implementation of a LoRA adapter.
    This implements ΔW(x) = B_ND(A_ND(x.unsqueeze(-1))).squeeze(-1)
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        super().__init__()
        
        if NdLinear is None:
            raise ImportError("NdLinear class is required but could not be imported.")

        # Configure dimensions for NdLinear
        # Input: reshape to (in_features, 1)
        # Rank: reshape to (r, 1)
        # Output: reshape to (out_features, 1)
        self.input_dims = (in_features, 1)
        self.rank_dims = (r, 1)
        self.output_dims = (out_features, 1)
        
        # NdLinear alternatives to Linear LoRA layers
        self.lora_A_nd = NdLinear(
            self.input_dims, 
            self.rank_dims, 
            bias=True, 
            transform_outer=False
        )
        
        self.lora_B_nd = NdLinear(
            self.rank_dims, 
            self.output_dims, 
            bias=True, 
            transform_outer=False
        )
        
        # Initialize weights according to LoRA paper
        # A with small values, B with zeros
        torch.nn.init.kaiming_uniform_(self.lora_A_nd.align_layers[0].weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B_nd.align_layers[0].weight)
        
        # Make sure the auxiliary 1x1 transforms are identity
        if len(self.lora_A_nd.align_layers) > 1 and self.lora_A_nd.align_layers[1].weight.numel() == 1:
             self.lora_A_nd.align_layers[1].weight.data.fill_(1.0)
        if len(self.lora_B_nd.align_layers) > 1 and self.lora_B_nd.align_layers[1].weight.numel() == 1:
             self.lora_B_nd.align_layers[1].weight.data.fill_(1.0)

        # LoRA scaling factor
        self.scaling = alpha / r
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # For tracking statistics
        self.in_features = in_features
        self.out_features = out_features
        self.rank = r
    
    def forward(self, x):
        # Original shape: (batch_size, seq_len, in_features)
        orig_shape = x.shape
        
        # For NdLinear, reshape to include the extra dimension
        # From: (batch_size * seq_len, in_features)
        # To: (batch_size * seq_len, in_features, 1)
        x_reshaped = x.reshape(-1, self.in_features, 1)
        
        # Apply dropout
        x_reshaped = self.dropout(x_reshaped)
        
        # NdLinear LoRA forward
        # A: (batch_size * seq_len, in_features, 1) -> (batch_size * seq_len, rank, 1)
        # B: (batch_size * seq_len, rank, 1) -> (batch_size * seq_len, out_features, 1)
        result = self.lora_B_nd(self.lora_A_nd(x_reshaped)) * self.scaling
        
        # Reshape back to original shape but with out_features
        # From: (batch_size * seq_len, out_features, 1)
        # To: (batch_size, seq_len, out_features) 
        result = result.squeeze(-1).view(*orig_shape[:-1], self.out_features)
        
        return result

# Added class definition from tiny_llama_control.py
# Wraps a linear layer with a NdLinear-based LoRA adapter.
class LinearLayerWithNdLinearLoRA(nn.Module):
    """
    Wraps a linear layer with a NdLinear-based LoRA adapter.
    Forward pass: h = W·x + ΔW(x) where ΔW(x) = B_ND(A_ND(x))
    """
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer # Keep the original layer
        
        # Freeze the base layer parameters by default during wrapper init
        # They might be loaded from the state_dict later
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # Extract dimensions from the base layer
        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features
        
        # Create NdLinear LoRA adapter
        self.adapter = NdLinearLoraLayer(
            in_features=in_features,
            out_features=out_features,
            r=r,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x):
        # Base layer output
        base_output = self.base_layer(x)
        
        # Add LoRA adapter contribution
        return base_output + self.adapter(x)

# ----- Added Factorized NdLinear LoRA Components from tiny_llama_control.py -----

def find_factor(n):
    for i in range(int(n ** 0.5), 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)

class NdLinearFactorizedLoRA(nn.Module):
    def __init__(self, d_in, d_out, alpha=1.0, dropout=0.): # Added alpha, default 1.0
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.in_factors = find_factor(d_in)
        self.out_factors = find_factor(d_out)
        self.adapter = NdLinear(
            input_dims=self.in_factors,
            hidden_size=self.out_factors,
            transform_outer=False
        )
        self.scaling = alpha # Use alpha for scaling
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        orig = x.shape  # (B, L, d_in)
        # Apply dropout and reshape for NdLinear
        x = self.drop(x).view(-1, *self.in_factors)
        # Apply NdLinear adapter and reshape back
        y = self.adapter(x).view(*orig[:-1], self.d_out)
        # Apply scaling
        return y * self.scaling

class LinearWithNdLinearFactorizedLoRA(nn.Module):
    def __init__(self, base_layer, alpha=1.0, dropout=0.0): # Added alpha, default 1.0
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.adapter = NdLinearFactorizedLoRA(
            d_in=self.base_layer.in_features,
            d_out=self.base_layer.out_features,
            alpha=alpha, # Pass alpha
            dropout=dropout
        )

    def forward(self, x):
        return self.base_layer(x) + self.adapter(x)

# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA models (PEFT classic or NdLinear Factorized)")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the LoRA model directory to evaluate (containing adapter_config.json or ndlinear_lora_config.json)"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model name (e.g., 'meta-llama/Llama-3.2-1B', 'Qwen/Qwen3-0.6B-Base'). If None, attempts to infer from LoRA config."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Friendly name for the model (used in output filenames)"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="openai/gsm8k", 
        help="HuggingFace dataset name for evaluation (e.g., 'openai/gsm8k', 'chilled/multiarith', 'allenai/ai2_arc')"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="main",
        help="Dataset configuration name (e.g., 'main' for gsm8k, 'default' for multiarith, 'ARC-Challenge' for ai2_arc)"
    )
    parser.add_argument(
        "--dataset_split", 
        type=str, 
        default="test", 
        help="Dataset split to use for evaluation"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save the evaluation results"
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bfloat16 precision"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use for evaluation (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Set the model_name if not provided
    if args.model_name is None:
        if '/' in args.model_path:
            args.model_name = args.model_path.split('/')[-1]
        else:
            args.model_name = os.path.basename(args.model_path)
    
    return args

def load_evaluation_model(model_path, cli_base_model_name=None, use_bf16=False, device=None):
    """Load a model with LoRA weights (PEFT classic or NdLinear Factorized)."""
    logger.info(f"Loading LoRA model from: {model_path}")

    model_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if use_bf16:
        logger.info("Using bfloat16 precision")

    peft_config_path = os.path.join(model_path, "adapter_config.json")
    ndlinear_config_path = os.path.join(model_path, "ndlinear_lora_config.json")

    base_model_name_from_config = None
    tokenizer = None
    model = None

    if os.path.exists(peft_config_path):
        logger.info(f"Found PEFT adapter_config.json. Loading as Classic LoRA model.")
        try:
            # For PEFT models, base model name is usually in adapter_config.json
            with open(peft_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name_from_config = adapter_config.get("base_model_name_or_path")
            
            if not base_model_name_from_config:
                if cli_base_model_name:
                    logger.warning("Could not infer base model from adapter_config.json, using provided --base_model_name.")
                    base_model_name_from_config = cli_base_model_name
                else:
                    raise ValueError("Base model name not found in adapter_config.json and --base_model_name not provided.")
            elif cli_base_model_name and cli_base_model_name != base_model_name_from_config:
                 logger.warning(f"Provided --base_model_name '{cli_base_model_name}' differs from adapter_config.json '{base_model_name_from_config}'. Using value from adapter_config.json: {base_model_name_from_config}")


            logger.info(f"Loading tokenizer for base model: {base_model_name_from_config}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name_from_config, trust_remote_code=True)

            logger.info(f"Loading base model: {base_model_name_from_config}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_from_config,
                torch_dtype=model_dtype, # Apply dtype here
                trust_remote_code=True
            )
            logger.info(f"Loading PEFT LoRA weights from {model_path}")
            # PeftModel will load onto the device of the base_model if not specified
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload() # Optional: merge for faster inference if not training further
            logger.info("Successfully loaded Classic PEFT LoRA model.")

        except Exception as e:
            logger.error(f"Error loading Classic PEFT LoRA model: {e}")
            raise
    
    elif os.path.exists(ndlinear_config_path):
        logger.info(f"Found ndlinear_lora_config.json. Assuming Factorized NdLinear LoRA model.")
        if NdLinear is None:
            raise ImportError("NdLinear class is required for Factorized NdLinear LoRA but could not be imported.")

        with open(ndlinear_config_path, 'r') as f:
            lora_config = json.load(f)

        base_model_name_from_config = lora_config.get("model_name")
        lora_alpha = lora_config.get("lora_alpha", 1.0) # Scaling factor for factorized
        lora_dropout = lora_config.get("dropout", 0.0)
        target_modules_list = lora_config.get("target_modules", [])
        
        if not base_model_name_from_config:
            if cli_base_model_name:
                logger.warning("Could not infer base model from ndlinear_lora_config.json, using provided --base_model_name.")
                base_model_name_from_config = cli_base_model_name
            else:
                raise ValueError("Base model name not found in ndlinear_lora_config.json and --base_model_name not provided.")
        elif cli_base_model_name and cli_base_model_name != base_model_name_from_config:
            logger.warning(f"Provided --base_model_name '{cli_base_model_name}' differs from ndlinear_lora_config.json '{base_model_name_from_config}'. Using value from config: {base_model_name_from_config}")

        logger.info(f"Factorized NdLinear LoRA config: alpha_scaling={lora_alpha}, dropout={lora_dropout}, targets={target_modules_list}")

        if not target_modules_list:
            raise ValueError("ndlinear_lora_config.json must specify target_modules.")

        logger.info(f"Loading tokenizer from base model: {base_model_name_from_config}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_from_config, trust_remote_code=True)

        logger.info(f"Loading base model config: {base_model_name_from_config}")
        config = AutoConfig.from_pretrained(base_model_name_from_config, trust_remote_code=True)
        
        logger.info("Initializing base model from config (on CPU initially)")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.config.use_cache = False

        logger.info("Wrapping target modules with LinearWithNdLinearFactorizedLoRA...")
        wrapped_count = 0
        target_modules_set = set(target_modules_list)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module_type = name.split('.')[-1]
                if module_type in target_modules_set:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                    
                    wrapped_layer = LinearWithNdLinearFactorizedLoRA(
                        module,
                        alpha=lora_alpha,
                        dropout=lora_dropout
                    )
                    setattr(parent, child_name, wrapped_layer)
                    wrapped_count += 1
        
        logger.info(f"Wrapped {wrapped_count} Factorized NdLinearLoRA modules.")
        if wrapped_count == 0:
             logger.warning("No target modules were found or wrapped for Factorized NdLinearLoRA!")

        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(pytorch_model_path):
            # Try to find best_epoch model if pytorch_model.bin is not present
            best_model_files = sorted(glob.glob(os.path.join(model_path, "pytorch_model_best_epoch_*.bin")))
            if best_model_files:
                pytorch_model_path = best_model_files[-1]
                logger.info(f"Found best epoch model: {pytorch_model_path}")
            else:
                raise FileNotFoundError(f"Weights file 'pytorch_model.bin' or 'pytorch_model_best_epoch_*.bin' not found in {model_path}")

        logger.info(f"Loading state dict from {pytorch_model_path}")
        state_dict = torch.load(pytorch_model_path, map_location="cpu")
        
        logger.info("Loading state dict into Factorized NdLinear LoRA model structure...")
        try:
            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            if incompatible_keys.missing_keys:
                logger.warning(f"Missing keys when loading state_dict: {len(incompatible_keys.missing_keys)}")
                logger.debug(f"First few missing keys: {incompatible_keys.missing_keys[:5]}")
            if incompatible_keys.unexpected_keys:
                logger.warning(f"Unexpected keys when loading state_dict: {len(incompatible_keys.unexpected_keys)}")
                logger.debug(f"First few unexpected keys: {incompatible_keys.unexpected_keys[:5]}")
            logger.info("Successfully loaded state dict with non-strict loading for Factorized NdLinear LoRA model.")
        except Exception as e:
            logger.error(f"Failed to load state dict for Factorized NdLinear LoRA model: {e}")
            raise
    else:
        raise FileNotFoundError(f"Could not find 'adapter_config.json' (for PEFT LoRA) or 'ndlinear_lora_config.json' (for Factorized NdLinear LoRA) in {model_path}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token for the loaded tokenizer.")

    model.eval()
    model.config.use_cache = True

    if device:
        model = model.to(device)
        logger.info(f"Model moved to {device}")
    
    # Apply dtype after moving to device and after all modifications
    model = model.to(dtype=model_dtype)
    logger.info(f"Model dtype set to {model_dtype}")

    return model, tokenizer

def extract_answer(text, dataset_type="gsm8k"):
    """Extract the numerical answer from the generated text for GSM8K, or letter for ARC."""
    if dataset_type == "arc":
        # For ARC, expect a single letter, possibly followed by a period or other text.
        # Try to find common patterns like "A.", "A)", "A " or just "A" at the start.
        clean_text = text.strip()
        if not clean_text:
            return None
        
        # Option 1: Check if the first char is a letter (and optionally next is punctuation/space)
        first_char = clean_text[0].upper()
        if 'A' <= first_char <= 'E': # Assuming up to E for typical choices
            if len(clean_text) == 1:
                return first_char
            # Check if it's like "A." or "A)"
            if clean_text[1] in ['.', ')', ' ']:
                return first_char
            # If it's just "Apple", we don't want "A" unless it's the only thing.
            # This simple check might need refinement if model outputs full words.
            # For now, prioritize if it seems to be just the letter.
            # Let's assume model is prompted to give just the letter.

        # Fallback: look for "Answer: X" pattern where X is a single letter
        match = re.search(r"Answer:\s*([A-E])", clean_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # If still nothing, return the first character if it's a letter (last resort for ARC)
        if 'A' <= first_char <= 'E':
             return first_char # Might be too aggressive, depends on model output format

        return None # Fallback if no clear letter found for ARC

    # GSM8K and other numerical extraction (existing logic)
    answer_pattern = r"####\s*(-?\d+(?:\.\d+)?)"
    match = re.search(answer_pattern, text)
    if match:
        return match.group(1)
    
    # Look for "the answer is X" pattern
    answer_pattern = r"(?:answer|result)(?:\s+is)?(?:\s*=\s*|\s+)(-?\d+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Look for the last occurrence of "= X" pattern
    equals_pattern = r"=\s*(-?\d+(?:\.\d+)?)"
    matches = list(re.finditer(equals_pattern, text))
    if matches:
        return matches[-1].group(1)
    
    # Find the last integer in the text as a fallback
    number_pattern = r"(-?\d+(?:\.\d+)?)"
    matches = list(re.finditer(number_pattern, text))
    if matches:
        return matches[-1].group(1)
    
    # If nothing found, return None
    return None

def extract_ref_answer(answer_text):
    """Extract the reference answer from the GSM8K answer."""
    # GSM8K format has "#### X" at the end
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1)
    # Fallback: if no "####", try to extract the last number, assuming it might be a simpler format
    # or the answer_text itself is the number.
    number_pattern = r"(-?\d+(?:\.\d+)?)"
    matches = list(re.finditer(number_pattern, str(answer_text)))
    if matches:
        return matches[-1].group(1)
    return str(answer_text).strip() # Default to returning the stripped text if no pattern matches

def main():
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
        logger.info(f"Using specified device: {device}")
    else:
        if torch.cuda.is_available():
            # Default to cuda:0 or the current primary CUDA device
            current_cuda_idx = torch.cuda.current_device()
            device = torch.device(f"cuda:{current_cuda_idx}")
            logger.info(f"Auto-detected CUDA. Using device: {device}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"Auto-detected MPS. Using device: mps")
        else:
            device = torch.device("cpu")
            logger.info("No GPU detected. Using CPU.")
    
    # logger.info(f"Using device: {device}") # Already logged in the branches above
    
    # Explicitly set the default CUDA device if a CUDA device is selected
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        logger.info(f"Set default CUDA device to: cuda:{torch.cuda.current_device()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}, config: {args.dataset_config}")
    
    dataset_type_for_extraction = "gsm8k" # Default
    if args.dataset_name.lower() == "chilled/multiarith":
        logger.info("Forcing name='default' for ChilleD/MultiArith dataset.")
        dataset = load_dataset(args.dataset_name, name="default", split=args.dataset_split)
        dataset_type_for_extraction = "gsm8k" # Numerical
    elif args.dataset_name.lower() == "allenai/ai2_arc":
        logger.info(f"Loading ARC dataset with config: {args.dataset_config}") # ARC-Challenge
        dataset = load_dataset(args.dataset_name, name=args.dataset_config, split=args.dataset_split)
        dataset_type_for_extraction = "arc" # Multiple choice letter
    else: # Default to openai/gsm8k or other specified
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
        dataset_type_for_extraction = "gsm8k" # Numerical

    # Load model and tokenizer
    model, tokenizer = load_evaluation_model(
        args.model_path, 
        args.base_model_name,
        args.use_bf16,
        device
    )
    
    # Limit number of examples if specified
    if args.max_examples > 0 and len(dataset) > args.max_examples:
        dataset = dataset.select(range(args.max_examples))
    
    logger.info(f"Evaluating on {len(dataset)} examples")
    
    # Initialize results tracking
    results = {
        "correct": 0,
        "total": len(dataset),
        "samples": []
    }
    
    # Generate answers
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        example = dataset[i]
        
        formatted_prompt = None
        ref_answer = None
        prompt_for_logging = "N/A" # Initialize for logging

        current_dataset_name_lower = args.dataset_name.lower()

        if "multiarith" in current_dataset_name_lower:
            prompt = example["question"]
            prompt_for_logging = prompt
            ref_answer_raw = example["final_ans"]
            ref_answer = str(ref_answer_raw).strip()
            formatted_prompt = f"Solve the following math problem:\\n\\n{prompt}\\n\\nAnswer: "
        elif "allenai/ai2_arc" in current_dataset_name_lower:
            question_text = example["question"]
            prompt_for_logging = question_text
            choices_data = example["choices"] # This is a dict with 'text' and 'label' lists
            
            choice_lines = []
            for label, text in zip(choices_data["label"], choices_data["text"]):
                choice_lines.append(f"{label}. {text}")
            choices_str = "\\n".join(choice_lines)
            
            formatted_prompt = f"Question: {question_text}\\nChoices:\\n{choices_str}\\nAnswer:"
            ref_answer = example["answerKey"].strip().upper()
        
        else: # Default to GSM8K or other datasets assumed to have GSM8K-like structure
            prompt = example["question"]
            prompt_for_logging = prompt
            if "answer" in example:
                 ref_answer_raw = example["answer"]
                 ref_answer = extract_ref_answer(ref_answer_raw)
            elif "final_ans" in example: 
                 ref_answer_raw = example["final_ans"]
                 ref_answer = str(ref_answer_raw).strip()
            else:
                logger.error(f"Could not find 'answer' or 'final_ans' field in example {i} for dataset {args.dataset_name}. Skipping.")
                results["samples"].append({
                    "id": i,
                    "question": prompt_for_logging,
                    "model_output": "Error: No answer field found in dataset",
                    "extracted_answer": None,
                    "reference_answer": None,
                    "is_correct": False
                })
                results["total"] -=1 # Adjust total count as this example is skipped for accuracy calculation
                continue
            formatted_prompt = f"Solve the following math problem step by step:\\n\\n{prompt}\\n\\nAnswer: "
        
        if formatted_prompt is None or ref_answer is None:
            logger.error(f"Could not prepare prompt or reference answer for example {i}. Dataset: {args.dataset_name}. Skipping.")
            results["samples"].append({
                "id": i,
                "question": prompt_for_logging,
                "model_output": "Error: Could not prepare prompt or reference answer",
                "extracted_answer": None,
                "reference_answer": None,
                "is_correct": False
            })
            continue
            
        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
        
        # Generate the output
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id
                )
            except Exception as e:
                logger.error(f"Error generating output for example {i}: {e}")
                continue
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # --- Added logging for raw output ---
        if i < 5: # Log raw output for first 5 examples
            logger.info(f"-- Raw Generated Text (Example {i}) --\n{generated_text}\n---------------------------------------")
        # ------------------------------------
        
        answer_portion = generated_text[len(formatted_prompt):]
        
        # Extract answer
        extracted_answer = extract_answer(answer_portion, dataset_type=dataset_type_for_extraction)
        
        # Check if correct
        is_correct = False
        if extracted_answer is not None and ref_answer is not None:
            if dataset_type_for_extraction == "arc":
                is_correct = (str(extracted_answer).strip().upper() == str(ref_answer).strip().upper())
            else: # Numerical comparison for gsm8k/multiarith
                try:
                    extracted_float = float(extracted_answer)
                    ref_float = float(ref_answer)
                    is_correct = abs(extracted_float - ref_float) < 1e-5
                except ValueError: # Handle cases where conversion to float fails
                    logger.debug(f"Could not convert extracted_answer ('{extracted_answer}') or ref_answer ('{ref_answer}') to float for example {i}.")
                    is_correct = False # If conversion fails, it's not correct numerically
                except Exception as e: # Catch any other unexpected errors during comparison
                    logger.warning(f"Error during numerical comparison for example {i}: {e}. Answers: Extracted='{extracted_answer}', Ref='{ref_answer}'.")
                    is_correct = False
        
        if is_correct:
            results["correct"] += 1
        
        # Log first few examples and any correct answers
        if i < 3 or is_correct:
            logger.info(f"""Example {i}:
Question: {prompt_for_logging} 
Extracted answer: {extracted_answer}
Reference answer: {ref_answer}
Is correct: {is_correct}
""")
        
        # Save result
        results["samples"].append({
            "id": i,
            "question": prompt_for_logging,
            "model_output": answer_portion,
            "extracted_answer": extracted_answer,
            "reference_answer": ref_answer,
            "is_correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["accuracy"] = accuracy * 100  # Convert to percentage
    results["model_name"] = args.model_name
    results["model_path"] = args.model_path
    
    # Generate output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name.replace('/', '_')}_{timestamp}.json")
    summary_file = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name.replace('/', '_')}_{timestamp}_summary.json")
    
    # Save detailed results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    with open(summary_file, "w") as f:
        summary = {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "dataset": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.dataset_split,
            "num_examples": results["total"],
            "num_correct": results["correct"],
            "accuracy": accuracy * 100,
            "timestamp": timestamp
        }
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info(f"Evaluation complete:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Accuracy: {accuracy:.2%} ({results['correct']}/{results['total']})")
    logger.info(f"  Results saved to: {output_file}")
    logger.info(f"  Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 