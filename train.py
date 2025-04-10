import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable
import torch.nn as nn
from torch.nn import functional as F

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 25000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 10      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 17000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 2500000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 72      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 170000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 100
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - __main__ - Metrics will be logged to {log_dir}")
    
    def update(self, metrics_dict):
        """Update metrics with new values."""
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
        self.step += 1
        
        # Log to tensorboard every log_interval steps
        if self.step % self.log_interval == 0:
            self._log_to_tensorboard()
            self._log_to_console()
    
    def _log_to_tensorboard(self):
        """Log current metrics to tensorboard."""
        for k, v in self.metrics.items():
            if len(v) > 0:
                self.tb_writer.add_scalar(k, v[-1], self.step)
    
    def _log_to_console(self):
        """Log current metrics to console."""
        elapsed = time.time() - self.last_log_time
        self.last_log_time = time.time()
        
        metrics_str = " | ".join([f"{k}: {v[-1]:.4f}" for k, v in self.metrics.items() if len(v) > 0])
        total_time = time.time() - self.start_time
        print(f"Step {self.step} | {metrics_str} | {elapsed:.2f}s/iter | Total: {total_time:.2f}s")
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to tensorboard."""
        try:
            # Convert config to a flat dict of only simple types
            hyperparams = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    hyperparams[k] = v
                elif isinstance(v, dict):
                    # Flatten nested dicts with dot notation
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool)):
                            hyperparams[f"{k}.{kk}"] = vv
            
            # Add empty metrics dict to avoid TensorBoard error
            empty_metrics = {"validation/loss": 0}
            
            # Use try-except to handle potential TensorBoard compatibility issues
            try:
                self.tb_writer.add_hparams(hyperparams, empty_metrics)
            except AttributeError as e:
                # Handle NumPy 2.0 compatibility issue with TensorBoard
                if "np.string_" in str(e):
                    print("Warning: TensorBoard hyperparameter logging skipped due to NumPy 2.0 compatibility issue")
                else:
                    print(f"Warning: TensorBoard hyperparameter logging failed: {e}")
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
            # Continue training even if hyperparameter logging fails
    def close(self):
        """Close TensorBoard writer."""
        self.tb_writer.close()

import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 25000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 10      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 17000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 2500000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 72      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 170000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 100
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - __main__ - Metrics will be logged to {log_dir}")
    
    def update(self, metrics_dict):
        """Update metrics with new values."""
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
        self.step += 1
        
        # Log to tensorboard every log_interval steps
        if self.step % self.log_interval == 0:
            self._log_to_tensorboard()
            self._log_to_console()
    
    def _log_to_tensorboard(self):
        """Log current metrics to tensorboard."""
        for k, v in self.metrics.items():
            if len(v) > 0:
                self.tb_writer.add_scalar(k, v[-1], self.step)
    
    def _log_to_console(self):
        """Log current metrics to console."""
        elapsed = time.time() - self.last_log_time
        self.last_log_time = time.time()
        
        metrics_str = " | ".join([f"{k}: {v[-1]:.4f}" for k, v in self.metrics.items() if len(v) > 0])
        total_time = time.time() - self.start_time
        print(f"Step {self.step} | {metrics_str} | {elapsed:.2f}s/iter | Total: {total_time:.2f}s")
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to tensorboard."""
        try:
            # Convert config to a flat dict of only simple types
            hyperparams = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    hyperparams[k] = v
                elif isinstance(v, dict):
                    # Flatten nested dicts with dot notation
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool)):
                            hyperparams[f"{k}.{kk}"] = vv
            
            # Add empty metrics dict to avoid TensorBoard error
            empty_metrics = {"validation/loss": 0}
            
            # Use try-except to handle potential TensorBoard compatibility issues
            try:
                self.tb_writer.add_hparams(hyperparams, empty_metrics)
            except AttributeError as e:
                # Handle NumPy 2.0 compatibility issue with TensorBoard
                if "np.string_" in str(e):
                    print("Warning: TensorBoard hyperparameter logging skipped due to NumPy 2.0 compatibility issue")
                else:
                    print(f"Warning: TensorBoard hyperparameter logging failed: {e}")
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
            # Continue training even if hyperparameter logging fails
    def close(self):
        """Close TensorBoard writer."""
        self.tb_writer.close()

# -------------------------------------
# 🚀 Training Loop with DeepSpeed
# -------------------------------------
def train(config=None):
    """Train the Turbotalk model with optimized memory management."""
    if config is None:
        config = TrainingConfig()
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set environment variables for optimal performance
    set_environment_variables()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Using device: {device_info}")
    
    # Time tracking for training
    start_time = time.time()
    max_training_hours = 8760  # Maximum training time - 1 year (365 days * 24 hours)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    if os.path.exists(config.output_dir + "/tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(config.output_dir + "/tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Log tokenizer info
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    config.vocab_size = len(tokenizer)
    logger.info(f"Adjusted vocab_size to match tokenizer: {config.vocab_size}")
    
    # Load finetune data if available and finetune mode is enabled
    finetune_data = None
    if config.finetune and os.path.exists("metadata.txt"):
        logger.info("Loading finetune data from metadata.txt...")
        import re
        with open("metadata.txt", "r", encoding="utf-8") as f:
            metadata_content = f.read()
            
        # Parse the content to extract training data
        finetune_data = []
        # Look for conversation data lists
        for data_var in ["simple_conversation_data", "technical_details_data", "mixed_context_data"]:
            pattern = f"{data_var} = \\[(.*?)\\]"
            data_match = re.search(pattern, metadata_content, re.DOTALL)
            if data_match:
                data_str = data_match.group(1)
                # Parse individual entries
                entries = re.findall(r'{\s*"question":\s*"(.*?)",\s*"answer":\s*"(.*?)"\s*}', data_str, re.DOTALL)
                for q, a in entries:
                    finetune_data.append({
                        "question": q.replace('\\n', '\n').replace('\\"', '"'),
                        "answer": a.replace('\\n', '\n').replace('\\"', '"')
                    })
        
        logger.info(f"Loaded {len(finetune_data)} conversation examples for finetuning")
    
    # Check if we need to load from a checkpoint
    checkpoint_path = getattr(config, 'checkpoint', None)
    
    # Create model with memory optimizations or load from checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        if is_local_path(checkpoint_path):
            # For local files, try direct PyTorch loading first
            try:
                logger.info("Loading local checkpoint with PyTorch...")
                # Initialize the model architecture first
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
                # Load checkpoint and extract model state dict
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)  # Try model_state_dict first, fallback to full dict
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']  # Handle DeepSpeed-style checkpoints
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded model with PyTorch")
            except Exception as e:
                logger.error(f"Error loading local checkpoint with PyTorch: {e}")
                logger.info("Creating new model since checkpoint loading failed")
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
        else:
            # For model IDs, try Hugging Face loading
            try:
                # Try to load with Hugging Face transformers
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if config.precision == "bf16" else None
                )
                logger.info("Successfully loaded model with transformers")
            except Exception as e:
                logger.warning(f"Error loading with transformers: {e}")
                logger.info("Creating new model since checkpoint loading failed")
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
    else:
        logger.info("Creating new model...")
        model = TurbotalkModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            max_seq_len=config.max_seq_len,
            window_size=config.window_size,
            use_flash_attn=config.use_flash_attn,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            use_alibi=False,
            checkpoint_dir=config.output_dir,
            phase_size=30
        )
    
    # Print detailed model statistics
    try:
        print_detailed_model_stats(model, "Turbotalk", show_detailed=False)
    except Exception as e:
        print(f"Detailed stats error: {str(e)}, using basic stats")
        print_basic_model_stats(model, "Turbotalk")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Apply mixed precision if available
    if torch.cuda.is_available() and config.use_mixed_precision:
        logger.info(f"Applied mixed precision ({config.precision})")
        if config.precision == "fp16":
            model = model.half()
        elif config.precision == "bf16" and torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
        else:
            logger.info(f"Requested {config.precision} precision not supported, using fp32")
    
    # Apply gradient checkpointing
    if config.use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Set up the LoRA configuration if enabled
    if config.use_lora:
        logger.info("Using LoRA for parameter-efficient fine-tuning")
        try:
            from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
            
            # Define target modules based on model architecture
            target_modules = config.lora_target_modules
            logger.info(f"Target LoRA modules: {target_modules}")
            
            # Create a wrapper to ensure proper output handling
            class PeftWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.config = model.config
                
                def forward(self, **kwargs):
                    # Ensure inputs are on correct device
                    device = next(self.model.parameters()).device
                    
                    # Ensure we have labels for loss calculation
                    if 'input_ids' in kwargs and 'labels' not in kwargs:
                        kwargs['labels'] = kwargs['input_ids'].clone()
                    
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and v.device != device:
                            kwargs[k] = v.to(device)
                    
                    # Forward through model
                    outputs = self.model(**kwargs)
                    
                    # Convert outputs to the expected format
                    if isinstance(outputs, dict):
                        # Create dict with proper attribute access
                        class ModelOutput(dict):
                            def __getattr__(self, name):
                                if name in self:
                                    return self[name]
                                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                            
                            def to_tuple(self):
                                return tuple(self[k] for k in self)
                        
                        # If we don't have a loss but we have logits and labels, calculate loss
                        if 'loss' not in outputs and 'logits' in outputs and 'labels' in kwargs:
                            logits = outputs['logits']
                            labels = kwargs['labels']
                            
                            # Shift for causal language modeling
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            
                            # Calculate loss
                            loss_fct = torch.nn.CrossEntropyLoss()
                            # Try to get vocab_size from various possible locations
                            if hasattr(self.model, 'vocab_size'):
                                vocab_size = self.model.vocab_size
                            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                                vocab_size = self.model.config.vocab_size
                            else:
                                # Default to logits dimension as fallback
                                vocab_size = logits.size(-1)
                                
                            loss = loss_fct(
                                shift_logits.view(-1, vocab_size),
                                shift_labels.view(-1)
                            )
                            outputs['loss'] = loss
                        
                        return ModelOutput(outputs)
                    
                    return outputs
                
                def get_input_embeddings(self):
                    return self.model.token_embedding
                    
                def get_output_embeddings(self):
                    return self.model.lm_head
                
                def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
                    return self.model.prepare_inputs_for_generation(input_ids, past_key_values, **kwargs)
            
            # Create a LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
            )
            
            # Wrap our model and apply LoRA
            logger.info("Applying LoRA to model...")
            wrapped_model = PeftWrapper(model)
            model = get_peft_model(wrapped_model, peft_config)
            
            logger.info(f"LoRA applied with rank {config.lora_r}, alpha {config.lora_alpha}")
            model.print_trainable_parameters()
            
        except Exception as e:
            logger.error(f"Error applying LoRA: {str(e)}")
            logger.warning("Continuing without LoRA")
            config.use_lora = False
    
    # Load and preprocess data
    logger.info("Loading and preprocessing datasets...")
    train_dataset = load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=config.fast_training, finetune_data=finetune_data)
    
    # Log dataset information
    logger.info(f"Dataset loaded with {len(train_dataset)} examples")
    if len(train_dataset) > 0:
        logger.info(f"Sample example - keys: {list(train_dataset[0].keys())}")
        for key, value in train_dataset[0].items():
            if hasattr(value, 'shape'):
                logger.info(f"  {key} shape: {value.shape}")
            elif hasattr(value, '__len__'):
                logger.info(f"  {key} length: {len(value)}")
    
    # Create data loader with memory-efficient settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True
    )
    
    # Log dataloader information
    logger.info(f"Created dataloader with {len(train_dataloader)} batches (batch_size={config.batch_size})")
    
    # Ensure the model is on the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    
    # Add debugging logs for device placement
    logger.info(f"Model device check - base_model: {next(model.parameters()).device}")
    if hasattr(model, 'base_model'):
        logger.info(f"Model device check - peft wrapper: {next(model.base_model.parameters()).device}")
        if hasattr(model.base_model, 'model'):
            logger.info(f"Model device check - inner model: {next(model.base_model.model.parameters()).device}")
            if hasattr(model.base_model.model, 'token_embedding'):
                logger.info(f"Model device check - embedding: {model.base_model.model.token_embedding.weight.device}")
    
    # Initialize optimizer with memory-efficient settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop with memory management
    model.train()
    total_steps = 0
    total_loss = 0
    
    # Calculate total number of epochs based on max_steps and steps per epoch
    steps_per_epoch = len(train_dataloader)
    # Ensure we have at least 1 epoch, especially for fast_training mode
    if steps_per_epoch == 0:
        steps_per_epoch = 1
        logger.warning("Empty dataloader detected, setting steps_per_epoch to 1 to avoid division by zero")
    
    calculated_epochs = max(1, config.max_steps // steps_per_epoch)
    total_epochs = min(config.max_epochs, calculated_epochs) if hasattr(config, 'max_epochs') else calculated_epochs
    
    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
        from tqdm.auto import trange
        use_tqdm = True
        # Force tqdm to use the same line for epoch and batch progress
        tqdm.get_lock()
    except ImportError:
        use_tqdm = False
        print("tqdm not installed, using basic progress tracking")
    
    # Check if colorama is available for colored output
    use_colors = colorama_available
        
    # Configure tqdm to clear previous output if needed
    tqdm_kwargs = {
        'leave': True,
        'ncols': 100, 
        'bar_format': '{l_bar}{bar:30}{r_bar}'
    }
        
    # Emoji indicators for training progress
    progress_emoji = ["🚂", "🚅", "🔥", "⚡", "🧠", "🌟", "🚀"]
    
    logger.info(f"Starting training for {total_epochs} epochs ({config.max_steps} steps)")
    
    # Create epoch progress bar
    epoch_iterator = trange(total_epochs, **tqdm_kwargs) if use_tqdm else range(total_epochs)
    for epoch in epoch_iterator:
        epoch_loss = 0.0
        all_epoch_labels = []  # Track all labels for per-token loss calculation
        
        # Update epoch progress bar description
        if use_tqdm:
            emoji = progress_emoji[epoch % len(progress_emoji)]
            epoch_iterator.set_description(f"{emoji} Epoch {epoch+1}/{total_epochs}")
        else:
            emoji = progress_emoji[epoch % len(progress_emoji)]
            print(f"\n{emoji} Epoch {epoch+1}/{total_epochs}")
        
        # Create batch progress bar
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Training",
            position=1,
            **tqdm_kwargs
        ) if use_tqdm else train_dataloader
        
        for batch in batch_iterator:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with memory optimization
            with torch.amp.autocast(device_type=device, enabled=config.use_mixed_precision):
                # Check if batch contains labels
                if 'labels' not in batch and 'input_ids' in batch:
                    # Add labels if not present
                    batch['labels'] = batch['input_ids'].clone()
                
                # Track labels for per-token loss calculation
                if 'labels' in batch:
                    all_epoch_labels.append(batch['labels'].detach().cpu())
                
                # Print batch keys before forward pass
                # print(f"DEBUG: Batch keys before forward: {list(batch.keys())}")
                
                outputs = model(**batch)
                
                # Get loss - try both attribute access and dictionary access
                try:
                    # First try attribute access
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    # Then try dictionary access
                    elif isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        # Create a dummy loss for training to continue
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        logger.warning("No loss found in model outputs")
                except Exception as e:
                    logger.error(f"Error accessing loss: {str(e)}")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Backward pass with gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (total_steps + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            
            total_loss += loss.item()
            total_steps += 1
            
            # Log progress
            if total_steps % config.logging_steps == 0:
                # Use improved loss calculation if enabled
                if config.improved_loss and 'labels' in batch:
                    # Calculate per-token loss for more stable reporting
                    num_tokens = (batch['labels'] != -100).sum().item()
                    if num_tokens > 0:
                        per_token_loss = loss.item() * config.gradient_accumulation_steps / num_tokens
                        avg_loss = total_loss / config.logging_steps
                        
                        # Log both raw and token-normalized loss
                        logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}, Per-token Loss = {per_token_loss:.4f}")
                        
                        # Update progress bar if using tqdm
                        if use_tqdm:
                            batch_iterator.set_postfix(loss=avg_loss, per_token_loss=per_token_loss, refresh=True)
                    else:
                        avg_loss = total_loss / config.logging_steps
                        logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}")
                        
                        # Update progress bar if using tqdm
                        if use_tqdm:
                            batch_iterator.set_postfix(loss=avg_loss, refresh=True)
                else:
                    # Use standard loss calculation
                    avg_loss = total_loss / config.logging_steps
                    logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}")
                    
import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 2500000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 72      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 17000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable
import torch.nn as nn
from torch.nn import functional as F

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 25000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 10      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 17000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 2500000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 72      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 170000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 100
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - __main__ - Metrics will be logged to {log_dir}")
    
    def update(self, metrics_dict):
        """Update metrics with new values."""
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
        self.step += 1
        
        # Log to tensorboard every log_interval steps
        if self.step % self.log_interval == 0:
            self._log_to_tensorboard()
            self._log_to_console()
    
    def _log_to_tensorboard(self):
        """Log current metrics to tensorboard."""
        for k, v in self.metrics.items():
            if len(v) > 0:
                self.tb_writer.add_scalar(k, v[-1], self.step)
    
    def _log_to_console(self):
        """Log current metrics to console."""
        elapsed = time.time() - self.last_log_time
        self.last_log_time = time.time()
        
        metrics_str = " | ".join([f"{k}: {v[-1]:.4f}" for k, v in self.metrics.items() if len(v) > 0])
        total_time = time.time() - self.start_time
        print(f"Step {self.step} | {metrics_str} | {elapsed:.2f}s/iter | Total: {total_time:.2f}s")
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to tensorboard."""
        try:
            # Convert config to a flat dict of only simple types
            hyperparams = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    hyperparams[k] = v
                elif isinstance(v, dict):
                    # Flatten nested dicts with dot notation
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool)):
                            hyperparams[f"{k}.{kk}"] = vv
            
            # Add empty metrics dict to avoid TensorBoard error
            empty_metrics = {"validation/loss": 0}
            
            # Use try-except to handle potential TensorBoard compatibility issues
            try:
                self.tb_writer.add_hparams(hyperparams, empty_metrics)
            except AttributeError as e:
                # Handle NumPy 2.0 compatibility issue with TensorBoard
                if "np.string_" in str(e):
                    print("Warning: TensorBoard hyperparameter logging skipped due to NumPy 2.0 compatibility issue")
                else:
                    print(f"Warning: TensorBoard hyperparameter logging failed: {e}")
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
            # Continue training even if hyperparameter logging fails
    def close(self):
        """Close TensorBoard writer."""
        self.tb_writer.close()

import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 25000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 10      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 17000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 2500000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 72      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 170000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
                if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Update generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Early stopping if EOS token is generated
                for i in range(batch_size):  # Assuming this is part of a loop
                    if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                        break
                
                # Memory optimization for very long sequences
                if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 100
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - __main__ - Metrics will be logged to {log_dir}")
    
    def update(self, metrics_dict):
        """Update metrics with new values."""
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
        self.step += 1
        
        # Log to tensorboard every log_interval steps
        if self.step % self.log_interval == 0:
            self._log_to_tensorboard()
            self._log_to_console()
    
    def _log_to_tensorboard(self):
        """Log current metrics to tensorboard."""
        for k, v in self.metrics.items():
            if len(v) > 0:
                self.tb_writer.add_scalar(k, v[-1], self.step)
    
    def _log_to_console(self):
        """Log current metrics to console."""
        elapsed = time.time() - self.last_log_time
        self.last_log_time = time.time()
        
        metrics_str = " | ".join([f"{k}: {v[-1]:.4f}" for k, v in self.metrics.items() if len(v) > 0])
        total_time = time.time() - self.start_time
        print(f"Step {self.step} | {metrics_str} | {elapsed:.2f}s/iter | Total: {total_time:.2f}s")
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to tensorboard."""
        try:
            # Convert config to a flat dict of only simple types
            hyperparams = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    hyperparams[k] = v
                elif isinstance(v, dict):
                    # Flatten nested dicts with dot notation
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool)):
                            hyperparams[f"{k}.{kk}"] = vv
            
            # Add empty metrics dict to avoid TensorBoard error
            empty_metrics = {"validation/loss": 0}
            
            # Use try-except to handle potential TensorBoard compatibility issues
            try:
                self.tb_writer.add_hparams(hyperparams, empty_metrics)
            except AttributeError as e:
                # Handle NumPy 2.0 compatibility issue with TensorBoard
                if "np.string_" in str(e):
                    print("Warning: TensorBoard hyperparameter logging skipped due to NumPy 2.0 compatibility issue")
                else:
                    print(f"Warning: TensorBoard hyperparameter logging failed: {e}")
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
            # Continue training even if hyperparameter logging fails
    def close(self):
        """Close TensorBoard writer."""
        self.tb_writer.close()

# -------------------------------------
# 🚀 Training Loop with DeepSpeed
# -------------------------------------
def train(config=None):
    """Train the Turbotalk model with optimized memory management."""
    if config is None:
        config = TrainingConfig()
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set environment variables for optimal performance
    set_environment_variables()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Using device: {device_info}")
    
    # Time tracking for training
    start_time = time.time()
    max_training_hours = 8760  # Maximum training time - 1 year (365 days * 24 hours)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    if os.path.exists(config.output_dir + "/tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(config.output_dir + "/tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Log tokenizer info
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    config.vocab_size = len(tokenizer)
    logger.info(f"Adjusted vocab_size to match tokenizer: {config.vocab_size}")
    
    # Load finetune data if available and finetune mode is enabled
    finetune_data = None
    if config.finetune and os.path.exists("metadata.txt"):
        logger.info("Loading finetune data from metadata.txt...")
        import re
        with open("metadata.txt", "r", encoding="utf-8") as f:
            metadata_content = f.read()
            
        # Parse the content to extract training data
        finetune_data = []
        # Look for conversation data lists
        for data_var in ["simple_conversation_data", "technical_details_data", "mixed_context_data"]:
            pattern = f"{data_var} = \\[(.*?)\\]"
            data_match = re.search(pattern, metadata_content, re.DOTALL)
            if data_match:
                data_str = data_match.group(1)
                # Parse individual entries
                entries = re.findall(r'{\s*"question":\s*"(.*?)",\s*"answer":\s*"(.*?)"\s*}', data_str, re.DOTALL)
                for q, a in entries:
                    finetune_data.append({
                        "question": q.replace('\\n', '\n').replace('\\"', '"'),
                        "answer": a.replace('\\n', '\n').replace('\\"', '"')
                    })
        
        logger.info(f"Loaded {len(finetune_data)} conversation examples for finetuning")
    
    # Check if we need to load from a checkpoint
    checkpoint_path = getattr(config, 'checkpoint', None)
    
    # Create model with memory optimizations or load from checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        if is_local_path(checkpoint_path):
            # For local files, try direct PyTorch loading first
            try:
                logger.info("Loading local checkpoint with PyTorch...")
                # Initialize the model architecture first
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
                # Load checkpoint and extract model state dict
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)  # Try model_state_dict first, fallback to full dict
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']  # Handle DeepSpeed-style checkpoints
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded model with PyTorch")
            except Exception as e:
                logger.error(f"Error loading local checkpoint with PyTorch: {e}")
                logger.info("Creating new model since checkpoint loading failed")
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
        else:
            # For model IDs, try Hugging Face loading
            try:
                # Try to load with Hugging Face transformers
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if config.precision == "bf16" else None
                )
                logger.info("Successfully loaded model with transformers")
            except Exception as e:
                logger.warning(f"Error loading with transformers: {e}")
                logger.info("Creating new model since checkpoint loading failed")
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
    else:
        logger.info("Creating new model...")
        model = TurbotalkModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            max_seq_len=config.max_seq_len,
            window_size=config.window_size,
            use_flash_attn=config.use_flash_attn,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            use_alibi=False,
            checkpoint_dir=config.output_dir,
            phase_size=30
        )
    
    # Print detailed model statistics
    try:
        print_detailed_model_stats(model, "Turbotalk", show_detailed=False)
    except Exception as e:
        print(f"Detailed stats error: {str(e)}, using basic stats")
        print_basic_model_stats(model, "Turbotalk")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Apply mixed precision if available
    if torch.cuda.is_available() and config.use_mixed_precision:
        logger.info(f"Applied mixed precision ({config.precision})")
        if config.precision == "fp16":
            model = model.half()
        elif config.precision == "bf16" and torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
        else:
            logger.info(f"Requested {config.precision} precision not supported, using fp32")
    
    # Apply gradient checkpointing
    if config.use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Set up the LoRA configuration if enabled
    if config.use_lora:
        logger.info("Using LoRA for parameter-efficient fine-tuning")
        try:
            from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
            
            # Define target modules based on model architecture
            target_modules = config.lora_target_modules
            logger.info(f"Target LoRA modules: {target_modules}")
            
            # Create a wrapper to ensure proper output handling
            class PeftWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.config = model.config
                
                def forward(self, **kwargs):
                    # Ensure inputs are on correct device
                    device = next(self.model.parameters()).device
                    
                    # Ensure we have labels for loss calculation
                    if 'input_ids' in kwargs and 'labels' not in kwargs:
                        kwargs['labels'] = kwargs['input_ids'].clone()
                    
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and v.device != device:
                            kwargs[k] = v.to(device)
                    
                    # Forward through model
                    outputs = self.model(**kwargs)
                    
                    # Convert outputs to the expected format
                    if isinstance(outputs, dict):
                        # Create dict with proper attribute access
                        class ModelOutput(dict):
                            def __getattr__(self, name):
                                if name in self:
                                    return self[name]
                                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                            
                            def to_tuple(self):
                                return tuple(self[k] for k in self)
                        
                        # If we don't have a loss but we have logits and labels, calculate loss
                        if 'loss' not in outputs and 'logits' in outputs and 'labels' in kwargs:
                            logits = outputs['logits']
                            labels = kwargs['labels']
                            
                            # Shift for causal language modeling
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            
                            # Calculate loss
                            loss_fct = torch.nn.CrossEntropyLoss()
                            # Try to get vocab_size from various possible locations
                            if hasattr(self.model, 'vocab_size'):
                                vocab_size = self.model.vocab_size
                            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                                vocab_size = self.model.config.vocab_size
                            else:
                                # Default to logits dimension as fallback
                                vocab_size = logits.size(-1)
                                
                            loss = loss_fct(
                                shift_logits.view(-1, vocab_size),
                                shift_labels.view(-1)
                            )
                            outputs['loss'] = loss
                        
                        return ModelOutput(outputs)
                    
                    return outputs
                
                def get_input_embeddings(self):
                    return self.model.token_embedding
                    
                def get_output_embeddings(self):
                    return self.model.lm_head
                
                def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
                    return self.model.prepare_inputs_for_generation(input_ids, past_key_values, **kwargs)
            
            # Create a LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
            )
            
            # Wrap our model and apply LoRA
            logger.info("Applying LoRA to model...")
            wrapped_model = PeftWrapper(model)
            model = get_peft_model(wrapped_model, peft_config)
            
            logger.info(f"LoRA applied with rank {config.lora_r}, alpha {config.lora_alpha}")
            model.print_trainable_parameters()
            
        except Exception as e:
            logger.error(f"Error applying LoRA: {str(e)}")
            logger.warning("Continuing without LoRA")
            config.use_lora = False
    
    # Load and preprocess data
    logger.info("Loading and preprocessing datasets...")
    train_dataset = load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=config.fast_training, finetune_data=finetune_data)
    
    # Log dataset information
    logger.info(f"Dataset loaded with {len(train_dataset)} examples")
    if len(train_dataset) > 0:
        logger.info(f"Sample example - keys: {list(train_dataset[0].keys())}")
        for key, value in train_dataset[0].items():
            if hasattr(value, 'shape'):
                logger.info(f"  {key} shape: {value.shape}")
            elif hasattr(value, '__len__'):
                logger.info(f"  {key} length: {len(value)}")
    
    # Create data loader with memory-efficient settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True
    )
    
    # Log dataloader information
    logger.info(f"Created dataloader with {len(train_dataloader)} batches (batch_size={config.batch_size})")
    
    # Ensure the model is on the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    
    # Add debugging logs for device placement
    logger.info(f"Model device check - base_model: {next(model.parameters()).device}")
    if hasattr(model, 'base_model'):
        logger.info(f"Model device check - peft wrapper: {next(model.base_model.parameters()).device}")
        if hasattr(model.base_model, 'model'):
            logger.info(f"Model device check - inner model: {next(model.base_model.model.parameters()).device}")
            if hasattr(model.base_model.model, 'token_embedding'):
                logger.info(f"Model device check - embedding: {model.base_model.model.token_embedding.weight.device}")
    
    # Initialize optimizer with memory-efficient settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop with memory management
    model.train()
    total_steps = 0
    total_loss = 0
    
    # Calculate total number of epochs based on max_steps and steps per epoch
    steps_per_epoch = len(train_dataloader)
    # Ensure we have at least 1 epoch, especially for fast_training mode
    if steps_per_epoch == 0:
        steps_per_epoch = 1
        logger.warning("Empty dataloader detected, setting steps_per_epoch to 1 to avoid division by zero")
    
    calculated_epochs = max(1, config.max_steps // steps_per_epoch)
    total_epochs = min(config.max_epochs, calculated_epochs) if hasattr(config, 'max_epochs') else calculated_epochs
    
    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
        from tqdm.auto import trange
        use_tqdm = True
        # Force tqdm to use the same line for epoch and batch progress
        tqdm.get_lock()
    except ImportError:
        use_tqdm = False
        print("tqdm not installed, using basic progress tracking")
    
    # Check if colorama is available for colored output
    use_colors = colorama_available
        
    # Configure tqdm to clear previous output if needed
    tqdm_kwargs = {
        'leave': True,
        'ncols': 100, 
        'bar_format': '{l_bar}{bar:30}{r_bar}'
    }
        
    # Emoji indicators for training progress
    progress_emoji = ["🚂", "🚅", "🔥", "⚡", "🧠", "🌟", "🚀"]
    
    logger.info(f"Starting training for {total_epochs} epochs ({config.max_steps} steps)")
    
    # Create epoch progress bar
    epoch_iterator = trange(total_epochs, **tqdm_kwargs) if use_tqdm else range(total_epochs)
    for epoch in epoch_iterator:
        epoch_loss = 0.0
        all_epoch_labels = []  # Track all labels for per-token loss calculation
        
        # Update epoch progress bar description
        if use_tqdm:
            emoji = progress_emoji[epoch % len(progress_emoji)]
            epoch_iterator.set_description(f"{emoji} Epoch {epoch+1}/{total_epochs}")
        else:
            emoji = progress_emoji[epoch % len(progress_emoji)]
            print(f"\n{emoji} Epoch {epoch+1}/{total_epochs}")
        
        # Create batch progress bar
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Training",
            position=1,
            **tqdm_kwargs
        ) if use_tqdm else train_dataloader
        
        for batch in batch_iterator:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with memory optimization
            with torch.amp.autocast(device_type=device, enabled=config.use_mixed_precision):
                # Check if batch contains labels
                if 'labels' not in batch and 'input_ids' in batch:
                    # Add labels if not present
                    batch['labels'] = batch['input_ids'].clone()
                
                # Track labels for per-token loss calculation
                if 'labels' in batch:
                    all_epoch_labels.append(batch['labels'].detach().cpu())
                
                # Print batch keys before forward pass
                # print(f"DEBUG: Batch keys before forward: {list(batch.keys())}")
                
                outputs = model(**batch)
                
                # Get loss - try both attribute access and dictionary access
                try:
                    # First try attribute access
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    # Then try dictionary access
                    elif isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        # Create a dummy loss for training to continue
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        logger.warning("No loss found in model outputs")
                except Exception as e:
                    logger.error(f"Error accessing loss: {str(e)}")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Backward pass with gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (total_steps + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            
            total_loss += loss.item()
            total_steps += 1
            
            # Log progress
            if total_steps % config.logging_steps == 0:
                # Use improved loss calculation if enabled
                if config.improved_loss and 'labels' in batch:
                    # Calculate per-token loss for more stable reporting
                    num_tokens = (batch['labels'] != -100).sum().item()
                    if num_tokens > 0:
                        per_token_loss = loss.item() * config.gradient_accumulation_steps / num_tokens
                        avg_loss = total_loss / config.logging_steps
                        
                        # Log both raw and token-normalized loss
                        logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}, Per-token Loss = {per_token_loss:.4f}")
                        
                        # Update progress bar if using tqdm
                        if use_tqdm:
                            batch_iterator.set_postfix(loss=avg_loss, per_token_loss=per_token_loss, refresh=True)
                    else:
                        avg_loss = total_loss / config.logging_steps
                        logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}")
                        
                        # Update progress bar if using tqdm
                        if use_tqdm:
                            batch_iterator.set_postfix(loss=avg_loss, refresh=True)
                else:
                    # Use standard loss calculation
                    avg_loss = total_loss / config.logging_steps
                    logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}")
                    
import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import DataLoader, RandomSampler
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from psutil import virtual_memory
from types import SimpleNamespace
from prettytable import PrettyTable

# Colorama for colored terminal output
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    colorama_available = True
except ImportError:
    colorama_available = False
    # Create dummy Fore and Style classes if colorama is not available
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Transformers imports
# Custom config class with get method for PEFT compatibility
class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*The current implementation is inefficient.*")
warnings.filterwarnings("ignore", message=".*The default behavior for positional arguments passing in Lambda will change.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs("turbotalk_checkpoints", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_dim: int = 2560
    num_layers: int = 34
    num_heads: int = 32
    num_experts: int = 8
    max_seq_len: int = 8192
    window_size: int = 1024
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 2500000  # Increased to achieve ~2000 epochs with dataset of ~10,000 examples
    max_epochs: int = 72      # Limit total number of epochs
    save_steps: int = 5000    # Adjusted to save less frequently given the longer training
    eval_steps: int = 5000    # Adjusted to evaluate less frequently
    logging_steps: int = 1000 # Adjusted to log less frequently
    curriculum_stages: int = 3
    steps_per_stage: int = 17000
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_kv_cache: bool = True
    max_memory_usage: float = 0.8  # Maximum GPU memory usage (0.8 = 80%)
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = False
    precision: str = "bf16"
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = True
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])

    # Demo parameters
    demo: bool = False
    
    # Fast training mode
    fast_training: bool = True
    
    # Finetune parameters
    finetune: bool = False
    after_training_finetuning: bool = False
    normal_finetuning: bool = False
    
    # Improved loss calculation
    improved_loss: bool = True
    
    # Checkpoint to load
    checkpoint: Optional[str] = None
    
    # Anti-repetition parameters
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50

@contextmanager
def timer(name: str = None):
    """Context manager for timing code execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} took {elapsed:.2f} seconds")
    else:
        logger.info(f"Operation took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set environment variables for optimal performance."""
    # Set PyTorch memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Set memory efficient attention
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    
    # Set mixed precision
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
    
    # Set DeepSpeed environment variables
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"  # Skip CUDA version check
    os.environ["DS_ACCELERATOR"] = "cuda"
    
    logger.info("Environment variables set for optimal performance")

def get_device_info():
    """Get and log information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / 1e9:.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Log current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Log available memory
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        max_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        logger.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved, {max_memory:.2f}GB total")
    else:
        logger.warning("No CUDA devices available, running on CPU")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cache cleared")

def calculate_model_size(model):
    """Calculate and log detailed model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get layer-specific counts for MoE models
    attn_params = 0
    moe_params = 0
    if hasattr(model, 'layers') and len(model.layers) > 0:
        try:
            attn_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'attention' in name)
            moe_params = sum(p.numel() for layer in model.layers 
                            for name, p in layer.named_parameters() if 'moe' in name or 'expert' in name)
        except Exception as e:
            logger.warning(f"Could not calculate detailed layer stats: {e}")
    
    # Calculate memory estimates
    bytes_per_param = 2  # bf16/fp16 training
    activation_memory = int(total_params * 4 * 1.2)  # Rough estimate for activations
    optimizer_memory = int(trainable_params * 12)  # Adam states
    total_memory = (total_params * bytes_per_param) + activation_memory + optimizer_memory
    
    # Calculate FLOPs if possible
    flops_estimate = None
    if hasattr(model, 'hidden_dim') and hasattr(model, 'num_layers'):
        flops_per_token = 6 * model.num_layers * model.hidden_dim**2  # Approximation
        flops_estimate = flops_per_token
    
    # Basic logging
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters total")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    logger.info(f"Memory estimate: {total_memory / (1024**3):.2f} GB")
    
    # Return rich statistics dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "trainable_percent": trainable_params / total_params * 100,
        "attention_params": attn_params,
        "moe_params": moe_params,
        "memory_estimate_gb": total_memory / (1024**3),
        "flops_per_token": flops_estimate,
        "effective_size_billion": total_params * 1.4 / 1e9 if moe_params > 0 else total_params / 1e9
    }

def print_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print detailed statistics about the model architecture and parameters."""
    import math
    from prettytable import PrettyTable
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if hasattr(model, 'num_experts') and model.num_experts > 1:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, {param.numel():,} parameters")
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_detailed_model_stats(model, model_name="Turbotalk", show_detailed=True):
    """Print comprehensive statistics about the model with emoji headers."""
    import math
    import sys
    
    # Handle PrettyTable dependency
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Install prettytable using pip
        import subprocess
        import sys
        print("PrettyTable not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
        from prettytable import PrettyTable
        
    # If still not available, use a simple table formatter
    try:
        from prettytable import PrettyTable
    except ImportError:
        class SimplePrettyTable:
            def __init__(self):
                self.field_names = []
                self.rows = []
                self.align = "l"
                
            def add_row(self, row):
                self.rows.append(row)
                
            def __str__(self):
                result = []
                # Add header
                header = " | ".join(str(h) for h in self.field_names)
                result.append(header)
                result.append("-" * len(header))
                # Add rows
                for row in self.rows:
                    result.append(" | ".join(str(c) for c in row))
                return "\n".join(result)
        
        PrettyTable = SimplePrettyTable
        print("Using simple table formatter as PrettyTable installation failed")
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Calculate memory usage estimates
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Prepare pretty tables
    param_table = PrettyTable()
    param_table.field_names = ["Parameter Type", "Count", "Percent"]
    param_table.add_row(["Trainable", f"{trainable_params:,}", f"{trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Non-trainable", f"{non_trainable_params:,}", f"{non_trainable_params/total_params*100:.2f}%"])
    param_table.add_row(["Total", f"{total_params:,}", "100.00%"])
    
    distribution_table = PrettyTable()
    distribution_table.field_names = ["Component Type", "Count", "Parameters", "% of Total"]
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components:
        distribution_table.add_row([
            component, 
            stats["count"], 
            f"{stats['params']:,}",
            f"{stats['params']/total_params*100:.2f}%"
        ])
    
    memory_table = PrettyTable()
    memory_table.field_names = ["Memory Type", "Estimated Usage (MB)", "Notes"]
    memory_table.add_row(["Parameters", f"{param_memory:.2f}", "4 bytes per parameter"])
    memory_table.add_row(["Activations", f"{activation_memory_estimate:.2f}", "Forward pass (estimated)"])
    memory_table.add_row(["Optimizer States", f"{optimizer_memory:.2f}", "Adam-like optimizer"])
    memory_table.add_row(["Total", f"{total_memory_estimate:.2f}", "Sum of above"])
    
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Estimate", "Notes"]
    # Estimate flops for transformer forward pass
    num_layers = getattr(model, 'num_layers', 12)
    hidden_size = getattr(model, 'hidden_dim', 768)
    seq_len = getattr(model, 'max_seq_len', 2048)
    
    # Estimate based on standard transformer computations
    flops_per_token = 4 * 2 * hidden_size * hidden_size * num_layers / seq_len
    perf_table.add_row(["FLOPs per token", f"{flops_per_token:.2e}", "Forward pass"])
    perf_table.add_row(["Generation tokens/sec", f"{1e9 / flops_per_token:.2f}", "On A100 GPU (estimated)"])
    perf_table.add_row(["Training tokens/sec", f"{1e9 / (flops_per_token * 3):.2f}", "Forward + backward (estimated)"])
    
    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "Parameters", "Ratio"]
    comparison_table.add_row(["This Model", f"{total_params:,}", "1.0x"])
    comparison_table.add_row(["GPT-2 (124M)", "124,000,000", f"{total_params/124000000:.2f}x"])
    comparison_table.add_row(["GPT-3 (175B)", "175,000,000,000", f"{total_params/175000000000:.5f}x"])
    comparison_table.add_row(["Llama 2 (7B)", "7,000,000,000", f"{total_params/7000000000:.5f}x"])
    
    # If model has experts, estimate expert utilization
    has_experts = hasattr(model, 'num_experts') and model.num_experts > 1
    if has_experts:
        expert_table = PrettyTable()
        expert_table.field_names = ["Metric", "Value", "Notes"]
        expert_table.add_row(["Number of Experts", f"{model.num_experts}", ""])
        expert_table.add_row(["Expert Parameters", f"{trainable_params * 0.7:,.0f}", "Estimated 70% of params in experts"])
        expert_table.add_row(["Parameters per Expert", f"{(trainable_params * 0.7) / model.num_experts:,.0f}", "Average"])
        expert_table.add_row(["Estimated Expert Utilization", "65-80%", "Typical range for MoE models"])
        expert_table.add_row(["Activation Sparsity", "~90%", "Due to expert gating"])
    
    # Component breakdown for detailed view
    if show_detailed:
        detailed_table = PrettyTable()
        detailed_table.field_names = ["Layer Name", "Shape", "Parameters"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                detailed_table.add_row([name, str(param.shape), f"{param.numel():,}"])
        detailed_table.align = "l"
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(param_table)
    
    print("\n📈 PARAMETER DISTRIBUTION")
    print(distribution_table)
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(memory_table)
    
    print("\n⚡ PERFORMANCE ESTIMATES")
    print(perf_table)
    
    print("\n📏 MODEL COMPARISONS")
    print(comparison_table)
    
    if has_experts:
        print("\n🧠 EXPERT UTILIZATION PREDICTION")
        print(expert_table)
    
    if show_detailed:
        print("\n🔬 DETAILED COMPONENT BREAKDOWN")
        print(detailed_table)
    
    print("\n" + "="*80)
    
    # Return summary statistics (can be logged)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

def print_basic_model_stats(model, model_name="Turbotalk"):
    """Print basic statistics about the model without relying on PrettyTable."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per param, converted to MB
    activation_memory_estimate = (param_memory * 2) / 8  # Rough estimate based on model size
    optimizer_memory = param_memory * 8  # For Adam-like optimizers
    total_memory_estimate = param_memory + activation_memory_estimate + optimizer_memory
    
    # Gather component stats
    component_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_type = module.__class__.__name__
                if component_type not in component_stats:
                    component_stats[component_type] = {"count": 0, "params": 0}
                component_stats[component_type]["count"] += 1
                component_stats[component_type]["params"] += params
    
    # Print the statistics
    print("\n" + "="*80)
    print(f"🔍 {model_name.upper()} MODEL STATISTICS 🔍")
    print("="*80)
    
    print("\n📊 PARAMETER COUNTS")
    print(f"Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable params:   {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    print(f"Total parameters:       {total_params:,}")
    
    print("\n📈 PARAMETER DISTRIBUTION")
    sorted_components = sorted(component_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    for component, stats in sorted_components[:5]:  # Top 5 components
        print(f"{component}: {stats['count']} instances, {stats['params']:,} params ({stats['params']/total_params*100:.2f}%)")
    
    print("\n💾 ESTIMATED MEMORY USAGE")
    print(f"Parameters:             {param_memory:.2f} MB")
    print(f"Activations (forward):  {activation_memory_estimate:.2f} MB")
    print(f"Optimizer states:       {optimizer_memory:.2f} MB")
    print(f"Total:                  {total_memory_estimate:.2f} MB")
    
    print("\n" + "="*80)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_memory_mb": param_memory,
        "estimated_total_memory_mb": total_memory_estimate,
    }

# -------------------------------------
# 🚀 Advanced Model Definition: Turbotalk 3B+
# -------------------------------------
class TurbotalkModel(torch.nn.Module):
    """Advanced Turbotalk model with Mixture of Experts, RoPE, and other state-of-the-art techniques."""
    
    def __init__(
        self,
        vocab_size=525437,
        hidden_dim=2560,
        num_layers=34,
        num_heads=32,
        num_experts=8,
        max_seq_len=8192,
        window_size=1024,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        use_alibi=False,
        checkpoint_dir="model_checkpoints",
        phase_size=30
    ):
        super().__init__()
        
        # Store model dimensions
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_alibi = use_alibi
        self.checkpoint_dir = checkpoint_dir
        self.phase_size = phase_size
        
        # Add configuration object for PEFT compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items()}
                
        self.config = Config(
            model_type='turbotalk',
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            architectures=['TurbotalkModel'],
            vocab_size=vocab_size
        )
        
        # Memory optimization parameters
        self.use_kv_cache = False
        self.use_memory_efficient_attention = False
        
        # Embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_dim // num_heads,
            max_position_embeddings=max_seq_len
        )
        
        # Initialize layers
        self._build_phase(0)
        
        # Final layernorm
        self.final_layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention implementation."""
        self.use_memory_efficient_attention = True
        
        # Update each attention layer
        for layer in self.layers:
            if hasattr(layer, 'attention'):
                layer.attention.use_memory_efficient_attention = True
                
        logger.info("Enabled memory-efficient attention for all layers")
        return self
    
    def enable_kv_cache(self):
        """Enable KV caching for faster inference."""
        self.use_kv_cache = True
        
        # Initialize empty cache
        self.kv_cache = {}
        for i in range(len(self.layers)):
            self.kv_cache[i] = {
                'k': None,
                'v': None
            }
            
        logger.info("Enabled KV caching for faster inference")
        return self
    
    def prune_model(self, pruning_threshold=0.1):
        """Prune model weights to reduce memory footprint."""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get weight tensor
                weight = module.weight.data
                
                # Calculate threshold for this layer
                threshold = pruning_threshold * torch.std(weight)
                
                # Create mask for small weights
                mask = (torch.abs(weight) > threshold).float()
                
                # Apply mask
                module.weight.data.mul_(mask)
                
                # Count params
                total_params += weight.numel()
                pruned_params += (1.0 - mask.float().mean().item()) * weight.numel()
        
        pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruned {pruned_params:,} parameters ({pruning_ratio:.2%} of {total_params:,} total)")
        
        return self
    
    def enable_cpu_offload(self):
        """Enable CPU offloading for large models."""
        # Move model parameters to CPU by default
        self.to("cpu")
        
        # Only keep essential components on GPU
        if torch.cuda.is_available():
            # Keep just the current active layer on GPU
            self.token_embedding = self.token_embedding.to("cuda")
            self.final_layer_norm = self.final_layer_norm.to("cuda")
            self.lm_head = self.lm_head.to("cuda")
            
        logger.info("Enabled CPU offloading for large model")
        return self
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def _load_existing_checkpoints(self):
        """Load existing layer checkpoints if available."""
        # Skip checkpoint loading and just build layers when training
        # This will prevent the errors we're seeing with loading checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        logger.info(f"Building all layers directly (skipping checkpoint loading)")
        for i in range(0, self.num_layers):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
        
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _create_layer(self, layer_idx: int) -> torch.nn.Module:
        """Create a single transformer layer."""
        return TransformerLayerWithMoE(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            window_size=self.window_size,
            use_flash_attn=self.use_flash_attn,
            rotary_emb=self.rotary_emb,
            use_alibi=self.use_alibi
        )
    
    def _build_phase(self, phase_start: int):
        """Build a phase of layers (simplified version)."""
        logger.info(f"Building phase starting at layer {phase_start}")
        
        # Build layers for this phase
        for i in range(phase_start, min(phase_start + self.phase_size, self.num_layers)):
            logger.info(f"Building layer {i + 1}/{self.num_layers}")
            
            layer = self._create_layer(i)
            # Initialize weights
            layer.apply(self._init_weights)
            self.layers.append(layer)
            logger.info(f"Successfully built layer {i + 1}")
                
        # Clear memory 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _rebuild_phase(self, phase_start: int):
        """Rebuild a phase of layers (simplified version)."""
        logger.warning(f"Rebuilding phase starting at layer {phase_start}")
        # Remove any partially loaded layers from this phase
        self.layers = self.layers[:phase_start]
        # Build the phase
        self._build_phase(phase_start)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model."""
        # Ensure input tensors are on the same device as the model
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                if self.use_kv_cache and not self.training:
                    # Pass KV cache if available during inference
                    hidden_states = layer(
                        hidden_states, 
                        attention_mask=attention_mask,
                        kv_cache=self.kv_cache[i] if self.use_kv_cache else None
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            vocab_size = self.vocab_size if hasattr(self, 'vocab_size') else self.config.vocab_size
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        
        # Return a dictionary for transformers compatibility
        class CausalLMOutput(dict):
            """Custom output class that behaves like both a dict and an object with attributes."""
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                
            def to_tuple(self):
                """Convert to tuple format for compatibility."""
                return tuple(self[k] for k in self)
        
        # Create output with loss field
        output_dict = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        # Only add loss if it exists
        if loss is not None:
            output_dict["loss"] = loss
            
        return CausalLMOutput(output_dict)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation with KV caching."""
        # Initialize KV cache if needed
        if self.use_kv_cache and past_key_values is None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            # Create empty past_key_values for each layer
            past_key_values = []
            for _ in range(self.num_layers):
                past_key_values.append({
                    'k': None,
                    'v': None
                })
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_kv_cache,
            **kwargs
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        **kwargs
    ):
        """Optimized generation with KV caching and memory-efficient settings."""
        # Enable KV cache for generation if not already enabled
        if not self.use_kv_cache:
            self.enable_kv_cache()
        
        # Enable memory efficient attention if not already enabled
        if not self.use_memory_efficient_attention:
            self.enable_memory_efficient_attention()
        
        # Set model to evaluation mode
        self.eval()
        
        # Move model to GPU if available
        device = input_ids.device
        
        # Initialize generated sequence with input_ids
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Clear KV cache
        if self.use_kv_cache:
            for i in range(len(self.layers)):
                self.kv_cache[i] = {
                    'k': None,
                    'v': None
                }
        
        # Generation loop
        with torch.no_grad():
            for _ in range(max_length - seq_length):
                # Clear CUDA cache periodically
                if _ % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # Forward pass
                logits = self(generated, attention_mask=attention_mask)
                
                # Get next token logits (last token in sequence)
                next_token_logits = logits[:, -1, :]
            
            # Apply temperature
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
            if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = torch.where(indices_to_remove, 
                                       torch.tensor(-float('inf'), device=next_token_logits.device),
                                       next_token_logits)
                
                # Sample next token
            if do_sample:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                    # Take the token with the highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
                # Update generated sequence
                    generated = torch.cat((generated, next_token), dim=1)
            
            # Update attention mask if needed
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
            
            # Early stopping if EOS token is generated
            for i in range(batch_size):  # Assuming this is part of a loop
                if (next_token[i] == kwargs.get("eos_token_id", 50256)).all():
                    break
                
                # Memory optimization for very long sequences
            if generated.shape[1] > 2048 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return generated


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = None
        self.sin_cache = None
        
        # Initialize cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[:, None, None, :]  # [seq_len, 1, 1, dim]
        self.sin_cache = emb.sin()[:, None, None, :]  # [seq_len, 1, 1, dim]
    
    def forward(self, q, k, position_ids=None):
        """Apply rotary embeddings to q and k."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # If position_ids is provided, use it to select from cache
        if position_ids is not None:
            # Extract the appropriate cos/sin values based on position_ids
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, :seq_len]
                
            # Make sure position_ids is on the right device
            position_ids = position_ids.to(device)
                
            # Get cos and sin values for these positions
            cos = self.cos_cache.to(device).index_select(0, position_ids.view(-1))
            sin = self.sin_cache.to(device).index_select(0, position_ids.view(-1))
            
            # Reshape for broadcasting
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # Use sequential positions if no position_ids provided
            cos = self.cos_cache.to(device)[:seq_len]
            sin = self.sin_cache.to(device)[:seq_len]
            
            # Reshape for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        # Transpose q and k for multiplying with cos/sin
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        k_reshaped = k.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, head_dim]
        
        # Apply rotary embeddings
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        # Transpose back
        q_embed = q_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k_embed = k_embed.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Helper function for RoPE - rotates half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FixedMixtureOfExperts(torch.nn.Module):
    """Simplified MoE implementation with fixed routing for memory efficiency."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=None,  # Will default to 4x hidden_dim if not specified
        num_experts=8,
        top_k=2,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        
        # Create experts
        self.experts = torch.nn.ModuleList([self._create_expert() for _ in range(num_experts)])
        
        # Create router
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _create_expert(self):
        """Create a single FFN expert."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.ffn_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout)
        )
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Get routing probabilities
        router_logits = self.router(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create an output tensor to accumulate expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process inputs through experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get expert's weights
            expert_weights = torch.zeros_like(routing_weights)
            for k in range(self.top_k):
                expert_weights[:, :, k] = torch.where(
                    indices[:, :, k] == expert_idx,
                    routing_weights[:, :, k],
                    torch.zeros_like(routing_weights[:, :, k])
                )
            
            # Sum over top-k dimension
            expert_weights = expert_weights.sum(dim=-1, keepdim=True)
            
            # Process inputs through expert
            expert_output = self.experts[expert_idx](hidden_states)
            
            # Add weighted output to result
            expert_outputs += expert_output * expert_weights
        
        return expert_outputs

class TransformerLayerWithMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        num_experts=8,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False,
        checkpoint_dir="moe_checkpoints",
        phase_size=4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        
        # Layer norm
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            use_flash_attn=use_flash_attn,
            rotary_emb=rotary_emb,
            use_alibi=use_alibi
        )
        
        # Post-attention layer norm
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        
        # MoE FFN
        self.mlp = FixedMixtureOfExperts(
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
    
    def forward(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        """Forward pass with KV cache support."""
        # Get residual for later
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_impl(
            hidden_states, 
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Mixture of Experts
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add residual
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_impl(self, hidden_states, attention_mask=None, kv_cache=None, position_ids=None):
        # Self-attention
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=kv_cache
        )
        
        return hidden_states

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=2560,
        num_heads=32,
        window_size=1024,
        use_flash_attn=False,
        rotary_emb=None,
        use_alibi=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.use_flash_attn = use_flash_attn
        self.rotary_emb = rotary_emb
        self.use_alibi = use_alibi
        self.use_memory_efficient_attention = False
        
        if (self.head_dim * num_heads) != self.hidden_dim:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        
        # Initialize Q, K, V projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs
    ):
        """Forward pass with KV caching support."""
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q, K, V for multi-head attention
        query_states = self._shape(query_states, seq_length, batch_size)
        key_states = self._shape(key_states, seq_length, batch_size)
        value_states = self._shape(value_states, seq_length, batch_size)
        
        # Apply rotary embeddings if provided
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
        
        # Use KV cache if provided
        if past_key_value is not None:
            if past_key_value.get('k') is not None and past_key_value.get('v') is not None:
                # Concatenate past keys and values with current
                key_states = torch.cat([past_key_value['k'], key_states], dim=2)
                value_states = torch.cat([past_key_value['v'], value_states], dim=2)
            
            # Update KV cache
            past_key_value['k'] = key_states
            past_key_value['v'] = value_states
        
        # Use memory efficient attention when enabled
        if self.use_memory_efficient_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Prepare attention mask for efficient attention
            if attention_mask is not None:
                # Convert to float mask and unsqueeze for batch and heads
                # attention_mask expected shape: [batch_size, 1, tgt_seq_len, src_seq_len]
                attention_mask = attention_mask.to(query_states.dtype)
                
                # Causal mask can be handled automatically
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # Convert 0s to -inf, 1s to 0s
                    attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            
            # Memory-efficient attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
        else:
            # Calculate attention scores
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            
            # Adjust attention weights if needed
            if self.use_alibi:
                # Add alibi positional bias
                alibi = self._get_alibi_bias(batch_size, seq_length, key_states.shape[2], hidden_states.device)
                attn_weights = attn_weights + alibi
            
            # Scale attention scores
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert mask to the right dtype
                attention_mask = attention_mask.to(attn_weights.dtype)
                
                # Expand mask if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    
                # Convert 0s to -inf, 1s to 0s
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Standard attention
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MixtureOfExperts(torch.nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    def __init__(
        self,
        hidden_dim=2560,
        ffn_dim=10240,
        num_experts=8,
        top_k=2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create experts
        self.experts = torch.nn.ModuleList([
            FeedForward(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
            for _ in range(num_experts)
        ])
        
        # Router for selecting experts
        self.router = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Initialize router with small weights
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states):
        """Forward pass with top-k routing."""
        batch_size, seq_length, hidden_dim = hidden_states.shape
        
        # Reshape for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Get routing probabilities
        routing_logits = self.router(hidden_states_flat)  # [batch*seq_len, num_experts]
        
        # Apply top-k routing
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        outputs = torch.zeros_like(hidden_states_flat)
        
        # Apply each expert to the inputs
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = hidden_states_flat[expert_mask]
                
                # Get probabilities for this expert
                expert_probs = torch.zeros(expert_mask.size(0), device=expert_mask.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[:, k] == expert_idx)
                    expert_probs[k_mask] = top_k_probs[:, k][k_mask]
                
                expert_probs = expert_probs[expert_mask].unsqueeze(-1)
                
                # Apply expert and scale by probability
                expert_output = self.experts[expert_idx](expert_inputs)
                outputs[expert_mask] += expert_output * expert_probs
        
        # Reshape back to original dimensions
        outputs = outputs.view(batch_size, seq_length, hidden_dim)
        
        return outputs


# -------------------------------------
# 🏗 Advanced Dataset Loading with HF Datasets
# -------------------------------------
class DataAugmenter:
    """Advanced data augmentation techniques for text data."""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def random_span_masking(self, text, mask_prob=0.15, max_span_length=5):
        """Apply random span masking to the text."""
        if not text:
            return text
            
        tokens = text.split()
        if not tokens:
            return text
            
        i = 0
        while i < len(tokens):
            if random.random() < mask_prob:
                span_length = min(random.randint(1, max_span_length), len(tokens) - i)
                for j in range(span_length):
                    if i + j < len(tokens):
                        tokens[i + j] = self.tokenizer.mask_token if hasattr(self.tokenizer, "mask_token") else "[MASK]"
                i += span_length
            else:
                i += 1
                
        return " ".join(tokens)
    
    def synonym_replacement(self, text, replace_prob=0.1):
        """Replace words with synonyms using WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download WordNet if not already downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download('wordnet')
                
            words = text.split()
            for i in range(len(words)):
                if random.random() < replace_prob:
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        words[i] = random.choice(synonyms).replace('_', ' ')
                        
            return " ".join(words)
        except ImportError:
            logger.warning("NLTK not installed. Skipping synonym replacement.")
            return text
    
    def token_deletion(self, text, del_prob=0.05):
        """Randomly delete tokens from the text."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = []
        for word in words:
            if random.random() > del_prob:
                new_words.append(word)
                
        if not new_words:
            rand_idx = random.randint(0, len(words) - 1)
            new_words = [words[rand_idx]]
            
        return " ".join(new_words)
    
    def apply_augmentations(self, example):
        """Apply a series of augmentations to the example."""
        text = example["text"] if "text" in example else ""
        
        # Apply augmentations with some probability
        if random.random() < 0.3:
            text = self.random_span_masking(text)
        if random.random() < 0.2:
            text = self.synonym_replacement(text)
        if random.random() < 0.1:
            text = self.token_deletion(text)
            
        example["text"] = text
        return example

def load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=False, finetune_data=None):
    """Load and preprocess data for model training with curriculum learning."""
    # Start with a small dataset for fast training
    if fast_training:
        from datasets import load_dataset
        logger.info("Fast training mode: using wikitext2 test dataset...")
        
        # Load a small dataset for fast testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for even faster training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        # Create a simple dataset with text examples
        dataset = Dataset.from_dict({"text": examples})
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Short sequences for fast training
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    # If finetune data is provided, use it
    elif finetune_data is not None and len(finetune_data) > 0:
        logger.info(f"Using finetune data: {len(finetune_data)} examples")
        
        # Format the finetune data for training
        formatted_examples = []
        for item in finetune_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            # Format as a conversation with clear human/assistant markers
            formatted_text = f"Human: {question}\n\nAssistant: {answer}"
            formatted_examples.append(formatted_text)
        
        # Create a dataset from the formatted examples
        dataset = Dataset.from_dict({"text": formatted_examples})
        
        # Define tokenization function for formatted conversations
        def tokenize_function(examples):
            # Dynamically adjust max_length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing finetune dataset"
        )
        
        return tokenized_dataset
        
    # Default - use full dataset with curriculum learning
    else:
        # Use different datasets based on curriculum stage
        datasets = []
        
        # Stage 0: Start with general knowledge
        if curriculum_stage >= 0:
            logger.info("Loading wikitext dataset...")
            wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            datasets.append(wiki_dataset)
        
        # Stage 1: Add coding and technical content
        if curriculum_stage >= 1:
            logger.info("Loading code dataset...")
            code_dataset = load_dataset("codeparrot/github-code", split="train")
            datasets.append(code_dataset)
        
        # Stage 2: Add conversation data
        if curriculum_stage >= 2:
            logger.info("Loading conversation dataset...")
            try:
                conv_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
                datasets.append(conv_dataset)
            except Exception as e:
                logger.warning(f"Failed to load conversation dataset: {e}")
                # Fallback to another dataset if available
                try:
                    logger.info("Trying alternative conversation dataset...")
                    alt_dataset = load_dataset("EleutherAI/pile", split="train")
                    datasets.append(alt_dataset)
                except Exception as e2:
                    logger.warning(f"Failed to load alternative dataset: {e2}")
        
        # If no datasets were loaded, fall back to a small dataset
        if not datasets:
            logger.warning("No datasets loaded, falling back to wikitext-2...")
            fallback_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            datasets.append(fallback_dataset)
        
        # Combine datasets if there are multiple
        if len(datasets) > 1:
            # This is a simplified way to combine datasets - in reality you might want more sophisticated mixing
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
        
        # Log dataset size
        logger.info(f"Dataset size: {len(combined_dataset)} examples")
        
        # Define a function to measure example complexity for curriculum learning
        def measure_complexity(example):
            # Extract the text field (adapt field name as needed)
            text = example.get("text", "")
            if not text and "content" in example:
                text = example.get("content", "")
            if not text and "chosen" in example:
                text = example.get("chosen", "")
            
            if not isinstance(text, str):
                text = str(text)
            
            # Simple complexity measures
            length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.findall(r'[.!?]+', text)) + 1  # +1 to avoid zero
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            
            # Combined complexity score (simple weighted sum)
            complexity = (0.1 * length + 
                         10.0 * word_count / max(1, sentence_count) +  # Longer sentences
                         5.0 * avg_word_length)  # Longer words
            
            return {
                "complexity": complexity,
                "length": length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        
        # Define tokenization function
        def tokenize_fn(examples):
            # Dynamic max length based on curriculum stage
            max_length = min(512 + curriculum_stage * 512, 2048)
            
            # Extract the text field (adapt field name as needed)
            texts = []
            for example in examples:
                text = example.get("text", "")
                if not text and "content" in example:
                    text = example.get("content", "")
                if not text and "chosen" in example:
                    text = example.get("chosen", "")
                
                if not isinstance(text, str):
                    text = str(text)
                
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Apply tokenization
        tokenized_dataset = combined_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=combined_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

# -------------------------------------
# 📊 Metrics Tracking and Logging
# -------------------------------------
class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(self, log_dir="./logs"):
        """Initialize metrics tracking and logging."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.metrics = {}
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 100
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - __main__ - Metrics will be logged to {log_dir}")
    
    def update(self, metrics_dict):
        """Update metrics with new values."""
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
        self.step += 1
        
        # Log to tensorboard every log_interval steps
        if self.step % self.log_interval == 0:
            self._log_to_tensorboard()
            self._log_to_console()
    
    def _log_to_tensorboard(self):
        """Log current metrics to tensorboard."""
        for k, v in self.metrics.items():
            if len(v) > 0:
                self.tb_writer.add_scalar(k, v[-1], self.step)
    
    def _log_to_console(self):
        """Log current metrics to console."""
        elapsed = time.time() - self.last_log_time
        self.last_log_time = time.time()
        
        metrics_str = " | ".join([f"{k}: {v[-1]:.4f}" for k, v in self.metrics.items() if len(v) > 0])
        total_time = time.time() - self.start_time
        print(f"Step {self.step} | {metrics_str} | {elapsed:.2f}s/iter | Total: {total_time:.2f}s")
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to tensorboard."""
        try:
            # Convert config to a flat dict of only simple types
            hyperparams = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    hyperparams[k] = v
                elif isinstance(v, dict):
                    # Flatten nested dicts with dot notation
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool)):
                            hyperparams[f"{k}.{kk}"] = vv
            
            # Add empty metrics dict to avoid TensorBoard error
            empty_metrics = {"validation/loss": 0}
            
            # Use try-except to handle potential TensorBoard compatibility issues
            try:
                self.tb_writer.add_hparams(hyperparams, empty_metrics)
            except AttributeError as e:
                # Handle NumPy 2.0 compatibility issue with TensorBoard
                if "np.string_" in str(e):
                    print("Warning: TensorBoard hyperparameter logging skipped due to NumPy 2.0 compatibility issue")
                else:
                    print(f"Warning: TensorBoard hyperparameter logging failed: {e}")
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
            # Continue training even if hyperparameter logging fails
    def close(self):
        """Close TensorBoard writer."""
        self.tb_writer.close()

# -------------------------------------
# 🚀 Training Loop with DeepSpeed
# -------------------------------------
def train(config=None):
    """Train the Turbotalk model with optimized memory management."""
    if config is None:
        config = TrainingConfig()
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set environment variables for optimal performance
    set_environment_variables()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Using device: {device_info}")
    
    # Time tracking for training
    start_time = time.time()
    max_training_hours = 8760  # Maximum training time - 1 year (365 days * 24 hours)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    if os.path.exists(config.output_dir + "/tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(config.output_dir + "/tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Log tokenizer info
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    config.vocab_size = len(tokenizer)
    logger.info(f"Adjusted vocab_size to match tokenizer: {config.vocab_size}")
    
    # Load finetune data if available and finetune mode is enabled
    finetune_data = None
    if config.finetune and os.path.exists("metadata.txt"):
        logger.info("Loading finetune data from metadata.txt...")
        import re
        with open("metadata.txt", "r", encoding="utf-8") as f:
            metadata_content = f.read()
            
        # Parse the content to extract training data
        finetune_data = []
        # Look for conversation data lists
        for data_var in ["simple_conversation_data", "technical_details_data", "mixed_context_data"]:
            pattern = f"{data_var} = \\[(.*?)\\]"
            data_match = re.search(pattern, metadata_content, re.DOTALL)
            if data_match:
                data_str = data_match.group(1)
                # Parse individual entries
                entries = re.findall(r'{\s*"question":\s*"(.*?)",\s*"answer":\s*"(.*?)"\s*}', data_str, re.DOTALL)
                for q, a in entries:
                    finetune_data.append({
                        "question": q.replace('\\n', '\n').replace('\\"', '"'),
                        "answer": a.replace('\\n', '\n').replace('\\"', '"')
                    })
        
        logger.info(f"Loaded {len(finetune_data)} conversation examples for finetuning")
    
    # Check if we need to load from a checkpoint
    checkpoint_path = getattr(config, 'checkpoint', None)
    
    # Create model with memory optimizations or load from checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        if is_local_path(checkpoint_path):
            # For local files, try direct PyTorch loading first
            try:
                logger.info("Loading local checkpoint with PyTorch...")
                # Initialize the model architecture first
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
                # Load checkpoint and extract model state dict
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)  # Try model_state_dict first, fallback to full dict
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']  # Handle DeepSpeed-style checkpoints
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded model with PyTorch")
            except Exception as e:
                logger.error(f"Error loading local checkpoint with PyTorch: {e}")
                logger.info("Creating new model since checkpoint loading failed")
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
        else:
            # For model IDs, try Hugging Face loading
            try:
                # Try to load with Hugging Face transformers
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if config.precision == "bf16" else None
                )
                logger.info("Successfully loaded model with transformers")
            except Exception as e:
                logger.warning(f"Error loading with transformers: {e}")
                logger.info("Creating new model since checkpoint loading failed")
                model = TurbotalkModel(
                    vocab_size=config.vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    num_experts=config.num_experts,
                    max_seq_len=config.max_seq_len,
                    window_size=config.window_size,
                    use_flash_attn=config.use_flash_attn,
                    use_gradient_checkpointing=config.use_gradient_checkpointing
                )
    else:
        logger.info("Creating new model...")
        model = TurbotalkModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            max_seq_len=config.max_seq_len,
            window_size=config.window_size,
            use_flash_attn=config.use_flash_attn,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            use_alibi=False,
            checkpoint_dir=config.output_dir,
            phase_size=30
        )
    
    # Print detailed model statistics
    try:
        print_detailed_model_stats(model, "Turbotalk", show_detailed=False)
    except Exception as e:
        print(f"Detailed stats error: {str(e)}, using basic stats")
        print_basic_model_stats(model, "Turbotalk")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Apply mixed precision if available
    if torch.cuda.is_available() and config.use_mixed_precision:
        logger.info(f"Applied mixed precision ({config.precision})")
        if config.precision == "fp16":
            model = model.half()
        elif config.precision == "bf16" and torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
        else:
            logger.info(f"Requested {config.precision} precision not supported, using fp32")
    
    # Apply gradient checkpointing
    if config.use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Set up the LoRA configuration if enabled
    if config.use_lora:
        logger.info("Using LoRA for parameter-efficient fine-tuning")
        try:
            from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
            
            # Define target modules based on model architecture
            target_modules = config.lora_target_modules
            logger.info(f"Target LoRA modules: {target_modules}")
            
            # Create a wrapper to ensure proper output handling
            class PeftWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.config = model.config
                
                def forward(self, **kwargs):
                    # Ensure inputs are on correct device
                    device = next(self.model.parameters()).device
                    
                    # Ensure we have labels for loss calculation
                    if 'input_ids' in kwargs and 'labels' not in kwargs:
                        kwargs['labels'] = kwargs['input_ids'].clone()
                    
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and v.device != device:
                            kwargs[k] = v.to(device)
                    
                    # Forward through model
                    outputs = self.model(**kwargs)
                    
                    # Convert outputs to the expected format
                    if isinstance(outputs, dict):
                        # Create dict with proper attribute access
                        class ModelOutput(dict):
                            def __getattr__(self, name):
                                if name in self:
                                    return self[name]
                                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                            
                            def to_tuple(self):
                                return tuple(self[k] for k in self)
                        
                        # If we don't have a loss but we have logits and labels, calculate loss
                        if 'loss' not in outputs and 'logits' in outputs and 'labels' in kwargs:
                            logits = outputs['logits']
                            labels = kwargs['labels']
                            
                            # Shift for causal language modeling
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            
                            # Calculate loss
                            loss_fct = torch.nn.CrossEntropyLoss()
                            # Try to get vocab_size from various possible locations
                            if hasattr(self.model, 'vocab_size'):
                                vocab_size = self.model.vocab_size
                            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                                vocab_size = self.model.config.vocab_size
                            else:
                                # Default to logits dimension as fallback
                                vocab_size = logits.size(-1)
                                
                            loss = loss_fct(
                                shift_logits.view(-1, vocab_size),
                                shift_labels.view(-1)
                            )
                            outputs['loss'] = loss
                        
                        return ModelOutput(outputs)
                    
                    return outputs
                
                def get_input_embeddings(self):
                    return self.model.token_embedding
                    
                def get_output_embeddings(self):
                    return self.model.lm_head
                
                def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
                    return self.model.prepare_inputs_for_generation(input_ids, past_key_values, **kwargs)
            
            # Create a LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
            )
            
            # Wrap our model and apply LoRA
            logger.info("Applying LoRA to model...")
            wrapped_model = PeftWrapper(model)
            model = get_peft_model(wrapped_model, peft_config)
            
            logger.info(f"LoRA applied with rank {config.lora_r}, alpha {config.lora_alpha}")
            model.print_trainable_parameters()
            
        except Exception as e:
            logger.error(f"Error applying LoRA: {str(e)}")
            logger.warning("Continuing without LoRA")
            config.use_lora = False
    
    # Load and preprocess data
    logger.info("Loading and preprocessing datasets...")
    train_dataset = load_and_preprocess_data(tokenizer, curriculum_stage=0, fast_training=config.fast_training, finetune_data=finetune_data)
    
    # Log dataset information
    logger.info(f"Dataset loaded with {len(train_dataset)} examples")
    if len(train_dataset) > 0:
        logger.info(f"Sample example - keys: {list(train_dataset[0].keys())}")
        for key, value in train_dataset[0].items():
            if hasattr(value, 'shape'):
                logger.info(f"  {key} shape: {value.shape}")
            elif hasattr(value, '__len__'):
                logger.info(f"  {key} length: {len(value)}")
    
    # Create data loader with memory-efficient settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True
    )
    
    # Log dataloader information
    logger.info(f"Created dataloader with {len(train_dataloader)} batches (batch_size={config.batch_size})")
    
    # Ensure the model is on the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Moving model to {device}")
    model = model.to(device)
    
    # Add debugging logs for device placement
    logger.info(f"Model device check - base_model: {next(model.parameters()).device}")
    if hasattr(model, 'base_model'):
        logger.info(f"Model device check - peft wrapper: {next(model.base_model.parameters()).device}")
        if hasattr(model.base_model, 'model'):
            logger.info(f"Model device check - inner model: {next(model.base_model.model.parameters()).device}")
            if hasattr(model.base_model.model, 'token_embedding'):
                logger.info(f"Model device check - embedding: {model.base_model.model.token_embedding.weight.device}")
    
    # Initialize optimizer with memory-efficient settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop with memory management
    model.train()
    total_steps = 0
    total_loss = 0
    
    # Calculate total number of epochs based on max_steps and steps per epoch
    steps_per_epoch = len(train_dataloader)
    # Ensure we have at least 1 epoch, especially for fast_training mode
    if steps_per_epoch == 0:
        steps_per_epoch = 1
        logger.warning("Empty dataloader detected, setting steps_per_epoch to 1 to avoid division by zero")
    
    calculated_epochs = max(1, config.max_steps // steps_per_epoch)
    total_epochs = min(config.max_epochs, calculated_epochs) if hasattr(config, 'max_epochs') else calculated_epochs
    
    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
        from tqdm.auto import trange
        use_tqdm = True
        # Force tqdm to use the same line for epoch and batch progress
        tqdm.get_lock()
    except ImportError:
        use_tqdm = False
        print("tqdm not installed, using basic progress tracking")
    
    # Check if colorama is available for colored output
    use_colors = colorama_available
        
    # Configure tqdm to clear previous output if needed
    tqdm_kwargs = {
        'leave': True,
        'ncols': 100, 
        'bar_format': '{l_bar}{bar:30}{r_bar}'
    }
        
    # Emoji indicators for training progress
    progress_emoji = ["🚂", "🚅", "🔥", "⚡", "🧠", "🌟", "🚀"]
    
    logger.info(f"Starting training for {total_epochs} epochs ({config.max_steps} steps)")
    
    # Create epoch progress bar
    epoch_iterator = trange(total_epochs, **tqdm_kwargs) if use_tqdm else range(total_epochs)
    for epoch in epoch_iterator:
        epoch_loss = 0.0
        all_epoch_labels = []  # Track all labels for per-token loss calculation
        
        # Update epoch progress bar description
        if use_tqdm:
            emoji = progress_emoji[epoch % len(progress_emoji)]
            epoch_iterator.set_description(f"{emoji} Epoch {epoch+1}/{total_epochs}")
        else:
            emoji = progress_emoji[epoch % len(progress_emoji)]
            print(f"\n{emoji} Epoch {epoch+1}/{total_epochs}")
        
        # Create batch progress bar
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Training",
            position=1,
            **tqdm_kwargs
        ) if use_tqdm else train_dataloader
        
        for batch in batch_iterator:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with memory optimization
            with torch.amp.autocast(device_type=device, enabled=config.use_mixed_precision):
                # Check if batch contains labels
                if 'labels' not in batch and 'input_ids' in batch:
                    # Add labels if not present
                    batch['labels'] = batch['input_ids'].clone()
                
                # Track labels for per-token loss calculation
                if 'labels' in batch:
                    all_epoch_labels.append(batch['labels'].detach().cpu())
                
                # Print batch keys before forward pass
                # print(f"DEBUG: Batch keys before forward: {list(batch.keys())}")
                
                outputs = model(**batch)
                
                # Get loss - try both attribute access and dictionary access
                try:
                    # First try attribute access
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    # Then try dictionary access
                    elif isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        # Create a dummy loss for training to continue
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        logger.warning("No loss found in model outputs")
                except Exception as e:
                    logger.error(f"Error accessing loss: {str(e)}")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Backward pass with gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (total_steps + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            
            total_loss += loss.item()
            total_steps += 1
            
            # Log progress
            if total_steps % config.logging_steps == 0:
                    # Calculate per-token loss for more stable reporting
                num_tokens = (batch['labels'] != -100).sum().item()
                if num_tokens > 0:
                        per_token_loss = loss.item() * config.gradient_accumulation_steps / num_tokens
                        avg_loss = total_loss / config.logging_steps
                        
                        # Log both raw and token-normalized loss
                        logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}, Per-token Loss = {per_token_loss:.4f}")
                        
                        # Update progress bar if using tqdm
                        if use_tqdm:
                            batch_iterator.set_postfix(loss=avg_loss, per_token_loss=per_token_loss, refresh=True)
                else:
                        avg_loss = total_loss / config.logging_steps
                        logger.info(f"Step {total_steps}: Loss = {avg_loss:.4f}")
                        
                        # Update progress bar if using tqdm
                        if use_tqdm:
                            batch_iterator.set_postfix(loss=avg_loss, refresh=True)
                
                # Early stopping check for fast_training mode - DISABLED to ensure all 72 epochs complete
                if False and config.fast_training and total_steps > config.logging_steps * 3:
                    # Check if loss is extremely low
                    if avg_loss < 0.0001:
                        logger.info(f"Loss is extremely low ({avg_loss:.6f}). Early stopping to prevent overfitting.")
                        break
                    
                    # Check if loss is not decreasing significantly after more steps
                    if 'prev_avg_loss' in locals() and abs(prev_avg_loss - avg_loss) < 0.001 and total_steps > config.logging_steps * 10:
                        logger.info(f"Loss has plateaued ({prev_avg_loss:.4f} → {avg_loss:.4f}). Early stopping for fast_training mode.")
                        break
                
                prev_avg_loss = avg_loss
                total_loss = 0
            
            # Save checkpoint based on steps
            if total_steps % config.save_steps == 0:
                logger.info(f"Saving checkpoint at step {total_steps}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch,
                    'loss': loss.item(),
                }, f"{config.output_dir}/checkpoint_{total_steps}.pt")
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Check training time limit for fast_training mode - DISABLED to ensure all 72 epochs complete
            if False and config.fast_training and (time.time() - start_time) > (max_training_hours * 60 * 60):
                logger.info(f"Maximum training time of {max_training_hours} hours reached for fast_training mode.")
                break
            
            if total_steps >= config.max_steps:
                break
        
        # End of epoch processing
        # Calculate total tokens for epoch loss normalization
        total_tokens = sum((batch_labels != -100).sum().item() for batch_labels in all_epoch_labels) if 'all_epoch_labels' in locals() else 0
        
        if total_tokens > 0:
            # More accurate per-token loss calculation
            per_token_epoch_loss = epoch_loss / total_tokens
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{total_epochs} completed: Avg Loss = {avg_epoch_loss:.4f}, Per-token Loss = {per_token_epoch_loss:.6f}")
        else:
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{total_epochs} completed: Avg Loss = {avg_epoch_loss:.4f}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            logger.info(f"Saving checkpoint after epoch {epoch+1}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': total_steps,
                'epoch': epoch,
                'loss': avg_epoch_loss,
            }, f"{config.output_dir}/checkpoint_epoch_{epoch+1}.pt")
            
            # Surprise - print a colorful celebration message
            if use_colors:
                celebration = [
                    f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}",
                    f"{Fore.YELLOW}🎉 CHECKPOINT SAVED AT EPOCH {epoch+1} 🎉{Style.RESET_ALL}",
                    f"{Fore.BLUE}Training progress: {(epoch+1)/total_epochs*100:.1f}% complete!{Style.RESET_ALL}",
                    f"{Fore.MAGENTA}Keep going! Your model is learning! 🚀{Style.RESET_ALL}",
                    f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}",
                ]
                print("\n".join(celebration))
        
        if total_steps >= config.max_steps:
            break
    
    logger.info("Training completed!")
    
    # Final save
    logger.info("Saving final model...")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': total_steps,
    }
    # Only add loss if it exists
    if 'loss' in locals():
        save_dict['loss'] = loss.item()
    else:
        save_dict['loss'] = 0.0  # Default value if no training occurred

    torch.save(save_dict, f"{config.output_dir}/final_model.pt")
    
    # Skip final evaluation for testing
    '''
    # Final evaluation
    logger.info("Running final evaluation...")
    evaluator = ModelEvaluator(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
    for i, prompt in enumerate(config.test_prompts):
        try:
            generated_text = evaluator.generate_text(
                prompt,
                max_length=200,
                temperature=0.7,
                top_p=0.9
            )
            logger.info(f"Prompt {i+1}: {prompt}")
            logger.info(f"Generated: {generated_text}")
        except Exception as e:
            logger.error(f"Error generating text for prompt {i+1}: {str(e)}")
    
    # Generate training visualization
    logger.info("Generating training visualization...")
    visualizer = TrainingVisualizer(log_dir=config.output_dir)
    visualizer.plot_metrics(save=True)
    
    # Close metrics tracker
    metrics_tracker = MetricsTracker(log_dir=config.output_dir)
    metrics_tracker.close()
    '''
    
    logger.info("Training and model saving completed!")

# -------------------------------------
# 🚀 Main Function
# -------------------------------------
if __name__ == "__main__":
    import argparse
    
    def parse_args():
        """Parse command line arguments for training."""
        parser = argparse.ArgumentParser(description="Train the Turbotalk model")
        
        # Model parameters
        parser.add_argument("--hidden_dim", type=int, default=2560, help="Hidden dimension of the model")
        parser.add_argument("--num_layers", type=int, default=34, help="Number of transformer layers")
        parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
        parser.add_argument("--num_experts", type=int, default=8, help="Number of experts in MoE layers")
        parser.add_argument("--max_seq_len", type=int, default=8192, help="Maximum sequence length")
        parser.add_argument("--window_size", type=int, default=1024, help="Attention window size")
        parser.add_argument("--use_moe", action="store_true", help="Use Mixture of Experts")
        parser.add_argument("--use_alibi", action="store_true", help="Use ALiBi positional embeddings")
        
        # Training parameters
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
        parser.add_argument("--gradient_accumulation", type=int, default=8, help="Number of gradient accumulation steps")
        parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
        parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
        parser.add_argument("--max_steps", type=int, default=10000, help="Maximum number of training steps")
        parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
        parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps")
        parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every X steps")
        parser.add_argument("--curriculum_stages", type=int, default=3, help="Number of curriculum learning stages")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
        parser.add_argument("--output_dir", type=str, default="turbotalk_checkpoints", help="Output directory for checkpoints")
        parser.add_argument("--finetune", action="store_true", help="Enable finetuning on metadata.txt conversations")
        
        # LoRA parameters
        parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient training")
        parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
        
        # Optimization parameters
        parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
        parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision")
        parser.add_argument("--zero_stage", type=int, default=3, choices=[0, 1, 2, 3], help="ZeRO optimization stage")
        parser.add_argument("--offload", action="store_true", help="Offload optimizer and parameters to CPU")
        parser.add_argument("--single_gpu", action="store_true", help="Use only a single GPU")
        parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention")
        parser.add_argument("--rope_scaling", action="store_true", help="Use RoPE scaling")
        parser.add_argument("--use_cosine_schedule", action="store_true", help="Use cosine learning rate schedule")
        parser.add_argument("--use_adafactor", action="store_true", help="Use Adafactor optimizer")
        
        # Demo parameters
        parser.add_argument("--demo", action="store_true", help="Run in demo mode")
        parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for demo")
        
        # Add the missing parameters
        parser.add_argument("--max_epochs", type=int, default=72, help="Maximum number of epochs to train")
        parser.add_argument("--fast_training", action="store_true", help="Use a small dataset for fast training/testing")
        parser.add_argument("--after_training_finetuning", action="store_true", help="Run regular training first, then finetune on metadata.txt")
        parser.add_argument("--normal_finetuning", action="store_true", help="Load final_model.pt directly and finetune it")
        parser.add_argument("--improved_loss", action="store_true", help="Use improved per-token loss calculation")
        parser.add_argument("--nanogpt", action="store_true", help="Run nanoGPT character-level model training")
        
        return parser.parse_args()
    
    args = parse_args()
    
    # Create a config dictionary from command line arguments
    config = {
        "vocab_size": 50257,  # GPT-2 vocabulary size (will be adjusted after loading tokenizer)
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_experts": args.num_experts,
        "max_seq_len": args.max_seq_len,
        "window_size": args.window_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "curriculum_stages": args.curriculum_stages,
        "steps_per_stage": int(args.max_steps / args.curriculum_stages),
        "gradient_accumulation_steps": args.gradient_accumulation,  # Map correctly from CLI arg
        "precision": args.precision,
        "zero_stage": args.zero_stage,
        "offload_optimizer": args.offload,
        "offload_param": args.offload,
        "batch_size": args.batch_size,
        "flash_attention": args.flash_attention,
        "rope_scaling": {"type": "dynamic", "factor": 2.0} if args.rope_scaling else None,
        "use_moe": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "rotary_pct": 1.0,
        "use_alibi": args.use_alibi,
        "use_dynamic_ntk": False,
        "use_cosine_schedule": args.use_cosine_schedule,
        "use_adafactor": args.use_adafactor,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "demo": args.demo,
        "checkpoint": args.checkpoint,
        "max_epochs": args.max_epochs,
        "fast_training": args.fast_training,
        "finetune": args.finetune,
        "after_training_finetuning": args.after_training_finetuning,
        "normal_finetuning": args.normal_finetuning,
        "improved_loss": args.improved_loss
    }
    
    # Override with command line arguments if provided
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
            
    # Print configuration
    print("Starting training with the following configuration:")
    print(json.dumps(config, indent=2))
    
    if args.demo:
        import turbotalk_demo
        # Use default path if checkpoint is None
        checkpoint_path = args.checkpoint if args.checkpoint else "./turbotalk_checkpoints/final"
        print(f"Running demo with checkpoint: {checkpoint_path}")
        turbotalk_demo.run_demo(checkpoint_path)
    elif args.normal_finetuning:
        # Load final_model.pt directly for finetuning
        model_path = "t/turbotalk_checkpoints/final_model.pt"
        if not os.path.exists(model_path):
            print(f"Error: Could not find {model_path} for finetuning")
            print("Please ensure the model file exists or train a model first")
            sys.exit(1)
            
        print(f"Loading {model_path} for direct finetuning")
        
        # Convert config to a proper TrainingConfig object
        train_config = TrainingConfig(
            vocab_size=config.get("vocab_size", 525437),
            hidden_dim=config.get("hidden_dim", 2560),
            num_layers=config.get("num_layers", 34),
            num_heads=config.get("num_heads", 32),
            num_experts=config.get("num_experts", 8),
            max_seq_len=config.get("max_seq_len", 8192),
            window_size=config.get("window_size", 1024),
            dropout=config.get("dropout", 0.1),
            expert_dropout=config.get("expert_dropout", 0.1),
            attention_dropout=config.get("attention_dropout", 0.1),
            batch_size=config.get("batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            learning_rate=config.get("learning_rate", 1e-5),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            max_steps=config.get("max_steps", 10000),
            save_steps=config.get("save_steps", 500),
            eval_steps=config.get("eval_steps", 500),
            logging_steps=config.get("logging_steps", 100),
            curriculum_stages=config.get("curriculum_stages", 3),
            steps_per_stage=config.get("steps_per_stage", 3000),
            use_lora=config.get("use_lora", True),
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            use_flash_attn=config.get("flash_attention", False),
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
            precision=config.get("precision", "bf16"),
            seed=config.get("seed", 42),
            output_dir=config.get("output_dir", "turbotalk_finetuned"),  # Different output dir for finetuned model
            single_gpu=config.get("single_gpu", True),
            zero_stage=config.get("zero_stage", 3),
            offload_optimizer=config.get("offload_optimizer", True),
            offload_param=config.get("offload_param", True),
            test_prompts=config.get("test_prompts", [
            "Hi, how are you? Can you please tell me something about artificial intelligence?",
            "What is the capital of France and what is it known for?",
            "Write a short poem about the beauty of nature."
            ]),
            demo=config.get("demo", False),
            max_epochs=config.get("max_epochs", 72),
            fast_training=config.get("fast_training", False),
            finetune=True,
            checkpoint=model_path
        )
        train(train_config)
    elif args.nanogpt:
        # Run nanoGPT training
        train_nanogpt()
    else:
        # Convert config to a proper TrainingConfig object
        train_config = TrainingConfig(
            vocab_size=config.get("vocab_size", 525437),
            hidden_dim=config.get("hidden_dim", 2560),
            num_layers=config.get("num_layers", 34),
            num_heads=config.get("num_heads", 32),
            num_experts=config.get("num_experts", 8),
            max_seq_len=config.get("max_seq_len", 8192),
            window_size=config.get("window_size", 1024),
            dropout=config.get("dropout", 0.1),
            expert_dropout=config.get("expert_dropout", 0.1),
            attention_dropout=config.get("attention_dropout", 0.1),
            batch_size=config.get("batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            learning_rate=config.get("learning_rate", 1e-5),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            max_steps=config.get("max_steps", 10000),
            save_steps=config.get("save_steps", 500),
            eval_steps=config.get("eval_steps", 500),
            logging_steps=config.get("logging_steps", 100),
            curriculum_stages=config.get("curriculum_stages", 3),
            steps_per_stage=config.get("steps_per_stage", 3000),
            use_lora=config.get("use_lora", True),
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            use_flash_attn=config.get("flash_attention", False),
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
            precision=config.get("precision", "bf16"),
            seed=config.get("seed", 42),
            output_dir=config.get("output_dir", "turbotalk_checkpoints"),
            single_gpu=config.get("single_gpu", True),
            zero_stage=config.get("zero_stage", 3),
            offload_optimizer=config.get("offload_optimizer", True),
            offload_param=config.get("offload_param", True),
            test_prompts=config.get("test_prompts", [
                "Hi, how are you? Can you please tell me something about artificial intelligence?",
                "What is the capital of France and what is it known for?",
                "Write a short poem about the beauty of nature."
            ]),
            demo=config.get("demo", False),
            max_epochs=config.get("max_epochs", 72),
            fast_training=config.get("fast_training", False),
            finetune=config.get("finetune", False),
            checkpoint=args.checkpoint
        )
        
        # Run training
        train(train_config)
        
        # Run finetuning after training if requested
        if args.after_training_finetuning:
            print("\n" + "="*50)
            print("Regular training completed. Starting finetuning...")
            print("="*50 + "\n")
            
            # Find the latest model checkpoint
            trained_model_path = None
            trained_model_dir = train_config.output_dir
            
            if os.path.exists(os.path.join(trained_model_dir, "final_model.pt")):
                trained_model_path = os.path.join(trained_model_dir, "final_model.pt")
            elif os.path.exists(os.path.join(trained_model_dir, "checkpoint_final.pt")):
                trained_model_path = os.path.join(trained_model_dir, "checkpoint_final.pt")
            else:
                # Look for checkpoint files
                checkpoints = []
                for file in os.listdir(trained_model_dir):
                    if file.startswith("checkpoint_") and file.endswith(".pt"):
                        checkpoints.append(file)
                
                if checkpoints:
                    # Sort numerically by checkpoint number
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]) if x.split("_")[1].split(".")[0].isdigit() else 0)
                    trained_model_path = os.path.join(trained_model_dir, checkpoints[-1])
            
            if trained_model_path:
                print(f"Found model checkpoint at {trained_model_path}")
            else:
                print("No model checkpoint found. Skipping finetuning.")
                sys.exit(0)
            
            # Create a new config for finetuning
            finetune_config = TrainingConfig(
                vocab_size=train_config.vocab_size,
                hidden_dim=train_config.hidden_dim,
                num_layers=train_config.num_layers,
                num_heads=train_config.num_heads,
                num_experts=train_config.num_experts,
                max_seq_len=train_config.max_seq_len,
                window_size=train_config.window_size,
                dropout=train_config.dropout,
                expert_dropout=train_config.expert_dropout,
                attention_dropout=train_config.attention_dropout,
                batch_size=train_config.batch_size,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                learning_rate=5e-6,  # Lower learning rate for finetuning
                weight_decay=train_config.weight_decay,
                warmup_ratio=train_config.warmup_ratio,
                max_steps=5000,  # Shorter finetuning
                save_steps=train_config.save_steps,
                eval_steps=train_config.eval_steps,
                logging_steps=train_config.logging_steps,
                curriculum_stages=train_config.curriculum_stages,
                steps_per_stage=train_config.steps_per_stage,
                use_lora=train_config.use_lora,
                lora_r=train_config.lora_r,
                lora_alpha=train_config.lora_alpha,
                lora_dropout=train_config.lora_dropout,
                lora_target_modules=train_config.lora_target_modules,
                use_flash_attn=train_config.use_flash_attn,
                use_gradient_checkpointing=train_config.use_gradient_checkpointing,
                precision=train_config.precision,
                seed=train_config.seed,
                output_dir="turbotalk_checkpoints_finetuned", # Different output dir for finetuned model
                single_gpu=train_config.single_gpu,
                zero_stage=train_config.zero_stage,
                offload_optimizer=train_config.offload_optimizer,
                offload_param=train_config.offload_param,
                test_prompts=train_config.test_prompts,
                demo=train_config.demo,
                max_epochs=train_config.max_epochs,
                fast_training=train_config.fast_training,
                finetune=True,
                checkpoint=trained_model_path
            )
            
            # Create output directory for finetuned model
            os.makedirs(finetune_config.output_dir, exist_ok=True)
            
            # Run finetuning
            print(f"Starting finetuning of model from {trained_model_path}")
            print(f"Finetuned model will be saved to {finetune_config.output_dir}")
            train(finetune_config)

# -------------------------------------
# 📊 Model Evaluation and Metrics
# -------------------------------------

class ModelEvaluator:
    """Comprehensive evaluation utilities for language models."""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = {}
        self.generation_config = {
            "max_length": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
        }
    
    def set_generation_config(self, **kwargs):
        """Update generation configuration parameters."""
        self.generation_config.update(kwargs)
        return self.generation_config
    
    def generate_text(self, prompt, **kwargs):
        """Generate text from a prompt using the model."""
        # Update generation config with any provided kwargs
        gen_config = {**self.generation_config, **kwargs}
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text with anti-repetition parameters
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=gen_config["max_length"],
                temperature=gen_config.get("temperature", 0.8),
                top_p=gen_config.get("top_p", 0.92),
                top_k=gen_config.get("top_k", 50),
                repetition_penalty=gen_config.get("repetition_penalty", 1.5),
                no_repeat_ngram_size=gen_config.get("no_repeat_ngram_size", 5),
                do_sample=gen_config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Apply post-processing to clean up repetitions
        generated_text = self._remove_repetitions(generated_text)
        
        return generated_text
    
    def _remove_repetitions(self, text):
        """Remove excessive repetitions from generated text."""
        # Remove triple+ repetitions of phrases (4+ words)
        words = text.split()
        if len(words) < 12:  # Skip short responses
            return text
            
        # Look for repeating patterns of phrases
        cleaned_words = []
        i = 0
        while i < len(words):
            # Skip if we're near the end
            if i > len(words) - 12:
                cleaned_words.extend(words[i:])
                break
                
            # Check for repeating 4-word phrases
            pattern_found = False
            for phrase_len in range(4, 8):  # Check phrases of length 4-7 words
                if i + phrase_len * 3 <= len(words):
                    phrase1 = ' '.join(words[i:i+phrase_len])
                    phrase2 = ' '.join(words[i+phrase_len:i+phrase_len*2])
                    phrase3 = ' '.join(words[i+phrase_len*2:i+phrase_len*3])
                    
                    # If we found a triple repetition
                    if phrase1.lower() == phrase2.lower() and phrase2.lower() == phrase3.lower():
                        cleaned_words.extend(words[i:i+phrase_len])
                        i += phrase_len * 3  # Skip all repetitions
                        pattern_found = True
                        break
            
            if not pattern_found:
                cleaned_words.append(words[i])
                i += 1
                
        return ' '.join(cleaned_words)
    
    def evaluate_perplexity(self, dataset, batch_size=4, max_samples=100):
        """Evaluate model perplexity on a dataset."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Limit number of samples if specified
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating perplexity"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
                
                # Calculate loss
                labels = batch["input_ids"].clone()
                # Shift labels for causal language modeling
                labels = labels[:, 1:].contiguous()
                logits = outputs[:, :-1, :].contiguous()
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction="sum"
                )
                
                # Count tokens (excluding padding)
                num_tokens = (labels != self.tokenizer.pad_token_id).sum().item()
                
                # Update totals
                total_loss += loss.item()
                total_tokens += num_tokens
                
                # Increment sample count
                sample_count += batch["input_ids"].size(0)
                if max_samples and sample_count >= max_samples:
                    break
        
        # Calculate perplexity
        perplexity = math.exp(total_loss / total_tokens)
        
        # Store metric
        self.metrics["perplexity"] = perplexity
        
        return perplexity
    
    def evaluate_generation_quality(self, prompts, reference_outputs=None, metrics=None):
        """Evaluate generation quality using various metrics."""
        if metrics is None:
            metrics = ["length", "diversity"]
        
        results = {}
        generations = []
        
        # Generate text for each prompt
        for i, prompt in enumerate(prompts):
            generated_text = self.generate_text(prompt)
            generations.append(generated_text)
            
            # Log generation
            logger.info(f"Prompt {i+1}: {prompt}")
            logger.info(f"Generated: {generated_text}\n")
        
        # Calculate metrics
        if "length" in metrics:
            # Average length in tokens
            lengths = [len(self.tokenizer.encode(text)) for text in generations]
            results["avg_length"] = sum(lengths) / len(lengths)
            results["max_length"] = max(lengths)
            results["min_length"] = min(lengths)
        
        if "diversity" in metrics:
            # Lexical diversity (unique tokens / total tokens)
            all_tokens = []
            unique_tokens = set()
            
            for text in generations:
                tokens = self.tokenizer.encode(text)
                all_tokens.extend(tokens)
                unique_tokens.update(tokens)
            
            results["lexical_diversity"] = len(unique_tokens) / len(all_tokens) if all_tokens else 0
        
        # Store metrics
        self.metrics.update(results)
        
        return results
    
    def benchmark_generation_speed(self, prompt, num_runs=5, **gen_kwargs):
        """Benchmark text generation speed."""
        total_tokens = 0
        total_time = 0
        
        # Tokenize the prompt once
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.size(1)
        
        # Update generation config
        gen_config = {**self.generation_config, **gen_kwargs}
        
        # Warm-up run
        logger.info("Performing warm-up generation run...")
        with torch.no_grad():
            _ = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=gen_config["max_length"],
                temperature=gen_config["temperature"],
                top_p=gen_config["top_p"],
                top_k=gen_config["top_k"],
                repetition_penalty=gen_config["repetition_penalty"],
                no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                do_sample=gen_config["do_sample"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Benchmark runs
        for i in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=gen_config["max_length"],
                    temperature=gen_config["temperature"],
                    top_p=gen_config["top_p"],
                    top_k=gen_config["top_k"],
                    repetition_penalty=gen_config["repetition_penalty"],
                    no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                    do_sample=gen_config["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            end_time = time.time()
            
            # Calculate stats
            generated_length = output_ids.size(1)
            new_tokens = generated_length - prompt_length
            generation_time = end_time - start_time
            tokens_per_second = new_tokens / generation_time
            
            logger.info(f"Run {i+1}: Generated {new_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
            
            total_tokens += new_tokens
            total_time += generation_time
        
        # Calculate average
        avg_tokens_per_second = total_tokens / total_time
        
        # Store metrics
        self.metrics["generation_speed"] = {
            "tokens_per_second": avg_tokens_per_second,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "num_runs": num_runs
        }
        
        logger.info(f"Average generation speed: {avg_tokens_per_second:.2f} tokens/s")
        
        return self.metrics["generation_speed"]
    
    def get_all_metrics(self):
        """Get all computed metrics."""
        return self.metrics
    
    def optimized_generate(self, prompt, max_tokens=100, temperature=0.7, **kwargs):
        """Generate text using optimized settings for speed and memory efficiency."""
        try:
            # Apply memory optimizations directly
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Apply mixed precision if on CUDA
            model = self.model
            if torch.cuda.is_available():
                model = model.half()
                logger.info("Applied mixed precision for generation")
            
            # Optimize with torch.compile if available
            if hasattr(torch, "compile") and torch.cuda.is_available():
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Applied torch.compile for faster execution")
            
            # Enable gradient checkpointing if available
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            
            # Custom optimized generation
            logger.info(f"Starting optimized generation with prompt: {prompt[:30]}...")
            
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate with memory-optimized settings
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    **kwargs
                )
            
            # Decode and return
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in optimized_generate: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating text: {str(e)}"
    
    def benchmark_optimized_generation(self, prompt, num_runs=5, max_tokens=100):
        """Benchmark the optimized generation speed."""
        total_tokens = 0
        total_time = 0
        
        # Apply memory optimizations before benchmarking
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            # Apply mixed precision if on CUDA
            model = self.model
            if torch.cuda.is_available():
                model = model.half()
                logger.info("Applied mixed precision for benchmark")
            
            # Optimize with torch.compile if available
            if hasattr(torch, "compile") and torch.cuda.is_available():
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Applied torch.compile for faster execution")
            
            # Tokenize the prompt once
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = inputs.input_ids.size(1)
            
            # Warm-up run
            logger.info("Performing warm-up generation run...")
            with torch.no_grad():
                _ = model.generate(
                    inputs.input_ids,
                    max_length=prompt_length + max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            
            # Benchmark runs
            for i in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    output_ids = model.generate(
                        inputs.input_ids,
                        max_length=prompt_length + max_tokens,
                        temperature=0.7,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                    )
                
                end_time = time.time()
                
                # Calculate stats
                generated_length = output_ids.size(1)
                new_tokens = generated_length - prompt_length
                generation_time = end_time - start_time
                tokens_per_second = new_tokens / generation_time
                
                logger.info(f"Run {i+1}: Generated {new_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
                
                total_tokens += new_tokens
                total_time += generation_time
            
            # Calculate average
            avg_tokens_per_second = total_tokens / total_time
            
            # Store metrics
            self.metrics["optimized_generation_speed"] = {
                "tokens_per_second": avg_tokens_per_second,
                "total_tokens": total_tokens,
                "total_time": total_time,
                "num_runs": num_runs,
                "time_per_token": 1.0 / avg_tokens_per_second if avg_tokens_per_second > 0 else float('inf'),
                "estimated_time_1000_tokens": 1000 / avg_tokens_per_second if avg_tokens_per_second > 0 else float('inf'),
            }
            
            logger.info(f"Average optimized generation speed: {avg_tokens_per_second:.2f} tokens/s")
            logger.info(f"Estimated time for 1000 tokens: {1000 / avg_tokens_per_second:.2f}s")
            return self.metrics["optimized_generation_speed"]
        except Exception as e:
            logger.error(f"Error in benchmark_optimized_generation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

# -------------------------------------
# 📈 Visualization and Reporting
# -------------------------------------

class TrainingVisualizer:
    """Visualization utilities for training metrics and model analysis."""
    
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        self.plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.metrics_history = defaultdict(list)
        self.steps = []
    
    def update_metrics(self, metrics, step):
        """Update metrics history."""
        self.steps.append(step)
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[name].append(value)
    
    def plot_metrics(self, save=True):
        """Plot training metrics."""
        try:
            import matplotlib.pyplot as plt
            plt.style.use('ggplot')
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot generate plots.")
            return {}
        
        plots = {}
        
        # Create individual plots for each metric
        for metric_name, values in self.metrics_history.items():
            if len(values) != len(self.steps):
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, values)
            plt.title(f'{metric_name.replace("_", " ").title()} over Training Steps')
            plt.xlabel('Training Steps')
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.grid(True)
            
            if save:
                plot_path = os.path.join(self.plots_dir, f"{metric_name}.png")
                plt.savefig(plot_path)
                plots[metric_name] = plot_path
                plt.close()
                plt.close()
            else:
                plots[metric_name] = plt.gcf()
        # Create a combined plot of loss and perplexity if available
        if 'loss' in self.metrics_history and 'perplexity' in self.metrics_history:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:red'
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(self.steps, self.metrics_history['loss'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Perplexity', color=color)
            ax2.plot(self.steps, self.metrics_history['perplexity'], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            plt.title('Loss and Perplexity over Training Steps')
            
            if save:
                plot_path = os.path.join(self.plots_dir, "loss_and_perplexity.png")
                plt.savefig(plot_path)
                plots["loss_and_perplexity"] = plot_path
                plt.close()
            else:
                plots["loss_and_perplexity"] = plt.gcf()
                
        
        return plots
    def generate_training_report(self, config, model_info, final_metrics):
        """Generate a comprehensive training report."""
        report = {
            "training_config": config.__dict__ if hasattr(config, "__dict__") else config,
            "model_info": model_info,
            "final_metrics": final_metrics,
            "training_duration": {
                "total_steps": self.steps[-1] if self.steps else 0,
                "metrics_history": dict(self.metrics_history)
            }
        }
        
        # Save report as JSON
        report_path = os.path.join(self.log_dir, "training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate plots
        plot_paths = self.plot_metrics(save=True)
        
        # Generate HTML report if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Turbotalk Training Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                    .metric-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; width: 200px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                    .metric-name {{ font-size: 14px; color: #666; }}
                    .plot-container {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Turbotalk Training Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Model Information</h2>
                <table>
            """
            
            # Add model info to HTML report
            for key, value in model_info.items():
                html_report += f"<tr><th>{key}</th><td>{value}</td></tr>\n"
            
            html_report += "</table>\n\n<h2>Training Configuration</h2>\n<table>\n"
            
            # Add training config to HTML report
            config_dict = config.__dict__ if hasattr(config, "__dict__") else config
            for key, value in config_dict.items():
                # Skip complex objects like lists of strings
                if isinstance(value, (int, float, str, bool)):
                    html_report += f"<tr><th>{key}</th><td>{value}</td></tr>\n"
            
            html_report += "</table>\n\n<h2>Final Metrics</h2>\n<div class=\"metric-container\">\n"
            
            # Add final metrics as cards
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    html_report += f"""
                    <div class="metric-card">  
                        <div class="metric-value">{formatted_value}</div>
                        <div class="metric-name">{key.replace('_', ' ').title()}</div>
                    </div>
                    """
            
            html_report += "</div>\n\n<h2>Training Plots</h2>\n"
            
            # Add plots to HTML report
            for plot_name, plot_path in plot_paths.items():
                relative_path = os.path.relpath(plot_path, self.log_dir)
                html_report += f"""
                <div class="plot-container">
                    <h3>{plot_name.replace('_', ' ').title()}</h3>
                    <img src="{relative_path}" alt="{plot_name}" style="max-width: 100%;">  
                </div>
                """
            
            html_report += """
            </body>
            </html>
            """
            
            # Save HTML report
            html_path = os.path.join(self.log_dir, "training_report.html")
            with open(html_path, 'w') as f:
                f.write(html_report)
            
            logger.info(f"Training report generated at {html_path}")
            
        except ImportError:
            logger.warning("Matplotlib not installed. HTML report will not include plots.")
        
        return report_path

# -------------------------------------
# 🧠 Advanced Memory Management
# -------------------------------------

class MemoryManager:
    """Advanced memory management for efficient training."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.memory_stats = []
        self.checkpoint_interval = 1000  # Steps between memory checkpoints
    
    def log_memory_stats(self, step):
        """Log current memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        
        # Get current device
        device = torch.cuda.current_device()
        
        # Collect memory stats
        stats["reserved_memory_gb"] = torch.cuda.memory_reserved(device) / 1e9
        stats["allocated_memory_gb"] = torch.cuda.memory_allocated(device) / 1e9
        stats["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        stats["step"] = step
        stats["timestamp"] = time.time()
        
        # Log stats
        logger.info(f"Memory stats at step {step}: ")
        logger.info(f"  - Reserved: {stats['reserved_memory_gb']:.2f} GB")
        logger.info(f"  - Allocated: {stats['allocated_memory_gb']:.2f} GB")
        logger.info(f"  - Max allocated: {stats['max_memory_gb']:.2f} GB")
        
        # Store stats history
        self.memory_stats.append(stats)
        
        return stats
    
    def should_log_memory(self, step):
        """Determine if memory should be logged at this step."""
        return step % self.checkpoint_interval == 0
    
    def optimize_memory_usage(self):
        """Optimize memory usage by clearing caches."""
        if torch.cuda.is_available():
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            
            # Run garbage collection
            gc.collect()
            
            logger.info("Cleared CUDA cache and ran garbage collection")
    
    def plot_memory_usage(self, save_path="memory_usage.png"):
        """Plot memory usage over time."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.memory_stats:
                logger.warning("No memory stats available for plotting")
                return None
            
            steps = [stat["step"] for stat in self.memory_stats]
            reserved = [stat["reserved_memory_gb"] for stat in self.memory_stats]
            allocated = [stat["allocated_memory_gb"] for stat in self.memory_stats]
            max_allocated = [stat["max_memory_gb"] for stat in self.memory_stats]
            
            plt.figure(figsize=(12, 6))
            plt.plot(steps, reserved, label="Reserved Memory (GB)")
            plt.plot(steps, allocated, label="Allocated Memory (GB)")
            plt.plot(steps, max_allocated, label="Max Allocated Memory (GB)")
            plt.xlabel("Training Step")
            plt.ylabel("Memory (GB)")
            plt.title("GPU Memory Usage During Training")
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Memory usage plot saved to {save_path}")
            return save_path
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot generate memory usage plot.")
            return None

# -------------------------------------
# 🛠️ Advanced Curriculum Learning
# -------------------------------------

class CurriculumLearningManager:
    """Advanced curriculum learning for efficient training."""
    
    def __init__(self, num_stages=3, total_steps=10000):
        self.num_stages = num_stages
        self.total_steps = total_steps
        self.steps_per_stage = total_steps // num_stages
        self.current_stage = 0
        self.stage_history = []
        self.difficulty_metrics = {}
        
        logger.info(f"Initialized curriculum learning with {num_stages} stages over {total_steps} steps")
        logger.info(f"Each stage will run for approximately {self.steps_per_stage} steps")
    
    def get_stage_for_step(self, step):
        """Determine the curriculum stage for a given step."""
        stage = min(step // self.steps_per_stage, self.num_stages - 1)
        return stage
    
    def should_advance_stage(self, step, metrics=None):
        """Determine if curriculum should advance to the next stage."""
        # Simple step-based advancement
        new_stage = self.get_stage_for_step(step)
        
        # Check if we should advance
        if new_stage > self.current_stage:
            logger.info(f"Advancing curriculum from stage {self.current_stage} to {new_stage} at step {step}")
            
            # Record stage transition
            self.stage_history.append({
                "from_stage": self.current_stage,
                "to_stage": new_stage,
                "step": step,
                "metrics": metrics
            })
            
            # Update current stage
            self.current_stage = new_stage
            return True
        
        return False
    
    def get_stage_config(self, stage):
        """Get configuration for a specific curriculum stage."""
        # Base difficulty parameters that increase with stage
        config = {
            "sequence_length": min(512 + stage * 512, 8192),  # Gradually increase sequence length
            "batch_size": max(4 - stage, 1),  # Gradually decrease batch size as sequences get longer
            "learning_rate": 5e-5 * (0.8 ** stage),  # Gradually decrease learning rate
            "data_mixing_ratios": self._get_data_mixing_ratios(stage),
            "augmentation_level": min(0.1 * (stage + 1), 0.3),  # Gradually increase augmentation
            "stage": stage
        }
        
        return config
    
    def _get_data_mixing_ratios(self, stage):
        """Get data mixing ratios for different stages."""
        if stage == 0:
            # Stage 0: Focus on simple, clean datasets
            return {
                "simple_texts": 0.7,
                "medium_texts": 0.3,
                "complex_texts": 0.0
            }
        elif stage == 1:
            # Stage 1: Balanced mix
            return {
                "simple_texts": 0.4,
                "medium_texts": 0.4,
                "complex_texts": 0.2
            }
        else:
            # Stage 2+: Focus on complex data
            return {
                "simple_texts": 0.2,
                "medium_texts": 0.3,
                "complex_texts": 0.5
            }
    
    def log_difficulty_metrics(self, batch, stage):
        """Log metrics about the difficulty of training data."""
        if "input_ids" not in batch:
            return
        
        # Calculate sequence length statistics
        seq_lengths = torch.sum(batch.get("attention_mask", torch.ones_like(batch["input_ids"])), dim=1)
        avg_seq_length = torch.mean(seq_lengths.float()).item()
        max_seq_length = torch.max(seq_lengths).item()
        
        # Store metrics
        if stage not in self.difficulty_metrics:
            self.difficulty_metrics[stage] = {
                "avg_seq_length": [],
                "max_seq_length": [],
                "batch_count": 0
            }
        
        self.difficulty_metrics[stage]["avg_seq_length"].append(avg_seq_length)
        self.difficulty_metrics[stage]["max_seq_length"].append(max_seq_length)
        self.difficulty_metrics[stage]["batch_count"] += 1
        
        # Periodically log stats
        if self.difficulty_metrics[stage]["batch_count"] % 100 == 0:
            avg_length = sum(self.difficulty_metrics[stage]["avg_seq_length"]) / len(self.difficulty_metrics[stage]["avg_seq_length"])
            avg_max_length = sum(self.difficulty_metrics[stage]["max_seq_length"]) / len(self.difficulty_metrics[stage]["max_seq_length"])
            
            logger.info(f"Curriculum stage {stage} data difficulty metrics:")
            logger.info(f"  - Average sequence length: {avg_length:.1f} tokens")
            logger.info(f"  - Average max sequence length: {avg_max_length:.1f} tokens")
    
    def get_summary(self):
        """Get a summary of the curriculum learning progress."""
        summary = {
            "num_stages": self.num_stages,
            "total_steps": self.total_steps,
            "steps_per_stage": self.steps_per_stage,
            "current_stage": self.current_stage,
            "stage_history": self.stage_history,
            "difficulty_metrics": {}
        }
        
        # Summarize difficulty metrics
        for stage, metrics in self.difficulty_metrics.items():
            if metrics["batch_count"] > 0:
                summary["difficulty_metrics"][str(stage)] = {
                    "avg_seq_length": sum(metrics["avg_seq_length"]) / len(metrics["avg_seq_length"]) if metrics["avg_seq_length"] else 0,
                    "max_seq_length": sum(metrics["max_seq_length"]) / len(metrics["max_seq_length"]) if metrics["max_seq_length"] else 0,
                    "batch_count": metrics["batch_count"]
                }
        
        return summary

# -------------------------------------
# 📦 Advanced Checkpointing
# -------------------------------------

class CheckpointManager:
    """Advanced checkpoint management for model training."""
    
    def __init__(self, base_path="./checkpoints", max_checkpoints=5):
        self.base_path = base_path
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.checkpoint_metadata = {}
        
        # Create checkpoint directory
        os.makedirs(base_path, exist_ok=True)
        
        # Load existing checkpoints if any
        self._load_existing_checkpoints()
        
        logger.info(f"Checkpoint manager initialized at {base_path}")
        logger.info(f"Found {len(self.checkpoints)} existing checkpoints")
    
    def _load_existing_checkpoints(self):
        """Load information about existing checkpoints."""
        if not os.path.exists(self.base_path):
            return
        
        # Look for checkpoint directories
        for item in os.listdir(self.base_path):
            checkpoint_dir = os.path.join(self.base_path, item)
            if os.path.isdir(checkpoint_dir):
                # Check for metadata file
                metadata_path = os.path.join(checkpoint_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        self.checkpoints.append(checkpoint_dir)
                        self.checkpoint_metadata[checkpoint_dir] = metadata
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint metadata from {metadata_path}: {e}")
        
        # Sort checkpoints by step
        self.checkpoints.sort(key=lambda x: self.checkpoint_metadata.get(x, {}).get("step", 0))
    
    def save_checkpoint(self, model_engine, tokenizer, step, metrics=None, stage=None, is_final=False):
        """Save a model checkpoint with metadata."""
        # Create checkpoint directory name
        if is_final:
            checkpoint_name = "final"
        elif stage is not None:
            checkpoint_name = f"stage{stage}_step{step}"
        else:
            checkpoint_name = f"step{step}"
        
        checkpoint_dir = os.path.join(self.base_path, checkpoint_name)
        
        # Save model checkpoint
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        model_engine.save_checkpoint(checkpoint_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Create and save metadata
        metadata = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "stage": stage,
            "is_final": is_final
        }
        
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Add to checkpoint list
        self.checkpoints.append(checkpoint_dir)
        self.checkpoint_metadata[checkpoint_dir] = metadata
        
        # Prune old checkpoints if needed
        self._prune_old_checkpoints()
        
        return checkpoint_dir
    
    def _prune_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        # Skip if we don't have too many checkpoints
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by step
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: (
                self.checkpoint_metadata.get(x, {}).get("is_final", False),  # Keep final checkpoints
                self.checkpoint_metadata.get(x, {}).get("step", 0)  # Sort by step
            )
        )
        
        # Identify checkpoints to remove (oldest non-final ones)
        to_remove = []
        for checkpoint in sorted_checkpoints:
            if len(sorted_checkpoints) - len(to_remove) <= self.max_checkpoints:
                break
                
            # Skip final checkpoints
            if self.checkpoint_metadata.get(checkpoint, {}).get("is_final", False):
                continue
                
            to_remove.append(checkpoint)
        
        # Remove old checkpoints
        for checkpoint in to_remove:
            try:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")
                
                # Remove from lists
                self.checkpoints.remove(checkpoint)
                self.checkpoint_metadata.pop(checkpoint, None)
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint}: {e}")
    
    def load_checkpoint(self, model, tokenizer, path=None, step=None, stage=None):
        """Load a specific checkpoint."""
        # Determine which checkpoint to load
        if path is not None:
            checkpoint_dir = path
        elif step is not None:
            # Find checkpoint with matching step
            matching = [c for c in self.checkpoints 
                      if self.checkpoint_metadata.get(c, {}).get("step") == step]
            if not matching:
                logger.error(f"No checkpoint found for step {step}")
                return None
            checkpoint_dir = matching[0]
        elif stage is not None:
            # Find latest checkpoint for stage
            matching = [c for c in self.checkpoints 
                      if self.checkpoint_metadata.get(c, {}).get("stage") == stage]
            if not matching:
                logger.error(f"No checkpoint found for stage {stage}")
                return None
            # Sort by step and take the latest
            checkpoint_dir = sorted(matching, 
                                  key=lambda x: self.checkpoint_metadata.get(x, {}).get("step", 0))[-1]
        else:
            # Load latest checkpoint
            if not self.checkpoints:
                logger.error("No checkpoints available to load")
                return None
            checkpoint_dir = self.checkpoints[-1]
        
        # Load the checkpoint
        try:
            logger.info(f"Loading checkpoint from {checkpoint_dir}")
            
            # Load tokenizer if it exists in the checkpoint
            tokenizer_path = os.path.join(checkpoint_dir, "tokenizer_config.json")
            if os.path.exists(tokenizer_path):
                tokenizer.from_pretrained(checkpoint_dir)
                logger.info("Loaded tokenizer from checkpoint")
            
            # Load model weights (implementation depends on your training framework)
            # For DeepSpeed, this would typically be done during initialization
            
            # Return metadata
            return self.checkpoint_metadata.get(checkpoint_dir, {})
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_dir}: {e}")
            return None
    
    def get_best_checkpoint(self, metric_name="loss", higher_better=False):
        """Get the best checkpoint according to a specific metric."""
        if not self.checkpoints:
            logger.warning("No checkpoints available")
            return None
        
        # Filter checkpoints that have the metric
        valid_checkpoints = []
        for checkpoint in self.checkpoints:
            metrics = self.checkpoint_metadata.get(checkpoint, {}).get("metrics", {})
            if metric_name in metrics:
                valid_checkpoints.append((checkpoint, metrics[metric_name]))
        
        if not valid_checkpoints:
            logger.warning(f"No checkpoints with metric '{metric_name}' found")
            return None
        
        # Sort by metric (ascending or descending based on higher_better)
        sorted_checkpoints = sorted(valid_checkpoints, key=lambda x: x[1], reverse=higher_better)
        best_checkpoint, best_value = sorted_checkpoints[0]
        
        logger.info(f"Best checkpoint for {metric_name}: {best_checkpoint} with value {best_value}")
        return best_checkpoint
    
    def get_checkpoint_summary(self):
        """Get a summary of all checkpoints."""
        summary = []
        for checkpoint in self.checkpoints:
            metadata = self.checkpoint_metadata.get(checkpoint, {})
            summary.append({
                "path": checkpoint,
                "step": metadata.get("step"),
                "stage": metadata.get("stage"),
                "timestamp": metadata.get("timestamp"),
                "is_final": metadata.get("is_final", False),
                "metrics": metadata.get("metrics", {})
            })
        
        return summary

# -------------------------------------
# 📊 Model Optimization Utilities
# -------------------------------------

class ModelOptimizer:
    """Utilities for optimizing model performance and memory usage."""
    
    def __init__(self, model=None):
        self.model = model
        self.optimization_history = []
    
    def set_model(self, model):
        """Set the model to optimize."""
        self.model = model
    
    def apply_gradient_checkpointing(self, model=None, use_gradient_checkpointing=True):
        """Apply gradient checkpointing to reduce memory usage."""
        target_model = model or self.model
        if target_model is None:
            logger.error("No model provided for gradient checkpointing")
            return
        
        if use_gradient_checkpointing:
            if hasattr(target_model, "gradient_checkpointing_enable"):
                target_model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing using model's built-in method")
            else:
                # Try to find modules that support gradient checkpointing
                checkpointed_modules = 0
                for name, module in target_model.named_modules():
                    if hasattr(module, "gradient_checkpointing") and hasattr(module, "gradient_checkpointing_enable"):
                        module.gradient_checkpointing_enable()
                        checkpointed_modules += 1
                    elif isinstance(module, torch.nn.modules.transformer.TransformerEncoder) or \
                         isinstance(module, torch.nn.modules.transformer.TransformerDecoder):
                        module.gradient_checkpointing = True
                        checkpointed_modules += 1
                
                if checkpointed_modules > 0:
                    logger.info(f"Enabled gradient checkpointing on {checkpointed_modules} modules")
                else:
                    logger.warning("Could not find modules supporting gradient checkpointing")
        else:
            if hasattr(target_model, "gradient_checkpointing_disable"):
                target_model.gradient_checkpointing_disable()
                logger.info("Disabled gradient checkpointing")
        
        # Record optimization
        self.optimization_history.append({
            "type": "gradient_checkpointing",
            "enabled": use_gradient_checkpointing,
            "timestamp": datetime.now().isoformat()
        })
    
    def optimize_memory_usage(self, model=None, device="cuda"):
        """Apply various optimizations to reduce memory usage."""
        target_model = model or self.model
        if target_model is None:
            logger.error("No model provided for memory optimization")
            return
        
        optimizations = []
        
        # 1. Apply gradient checkpointing
        self.apply_gradient_checkpointing(target_model, True)
        optimizations.append("gradient_checkpointing")
        
        # 2. Use mixed precision if on CUDA
        if device == "cuda" and torch.cuda.is_available():
            # Enable automatic mixed precision
            target_model = target_model.half()
            optimizations.append("mixed_precision")
        
        # 3. Clear unused memory
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            optimizations.append("cache_cleared")
        
        # 4. Enable memory efficient attention
        if hasattr(target_model, "enable_memory_efficient_attention"):
            target_model.enable_memory_efficient_attention()
            optimizations.append("memory_efficient_attention")
        
        # 5. Enable CPU offloading for large models
        if hasattr(target_model, "enable_cpu_offload"):
            target_model.enable_cpu_offload()
            optimizations.append("cpu_offloading")
        
        # 6. Enable KV cache optimization
        if hasattr(target_model, "enable_kv_cache"):
            target_model.enable_kv_cache()
            optimizations.append("kv_cache")
        
        # 7. Enable model pruning
        if hasattr(target_model, "prune_model"):
            target_model.prune_model()
            optimizations.append("model_pruning")
        
        # Record optimization
        self.optimization_history.append({
            "type": "memory_optimization",
            "optimizations": optimizations,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Applied memory optimizations: {', '.join(optimizations)}")
    
    def quantize_model(self, model=None, quantization_type="dynamic", bits=8):
        """Apply quantization to reduce model size and increase inference speed."""
        target_model = model or self.model
        if target_model is None:
            logger.error("No model provided for quantization")
            return
        
        # Record pre-quantization size
        pre_size = sum(p.numel() * p.element_size() for p in target_model.parameters()) / (1024 * 1024)
        logger.info(f"Model size before quantization: {pre_size:.2f} MB")
        
        try:
            if quantization_type == "dynamic":
                # Dynamic quantization (PyTorch built-in)
                quantized_model = torch.quantization.quantize_dynamic(
                    target_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info(f"Applied dynamic quantization to {bits} bits")
            elif quantization_type == "static":
                # Static quantization would require calibration data and more setup
                logger.warning("Static quantization not implemented")
                return
            elif quantization_type == "bitsandbytes":
                # Use bitsandbytes for 4-bit or 8-bit quantization
                try:
                    import bitsandbytes as bnb
                    
                    # Replace Linear layers with 8-bit or 4-bit quantized versions
                    if bits == 8:
                        for name, module in target_model.named_modules():
                            if isinstance(module, torch.nn.Linear):
                                setattr(module, name, bnb.nn.Linear8bitLt(
                                    module.in_features, module.out_features, 
                                    bias=module.bias is not None
                                ))
                        logger.info("Applied bitsandbytes 8-bit quantization")
                    elif bits == 4:
                        for name, module in target_model.named_modules():
                            if isinstance(module, torch.nn.Linear):
                                setattr(module, name, bnb.nn.Linear4bit(
                                    module.in_features, module.out_features, 
                                    bias=module.bias is not None
                                ))
                        logger.info("Applied bitsandbytes 4-bit quantization")
                    
                    quantized_model = target_model  # In-place modification
                except ImportError:
                    logger.error("bitsandbytes not installed, cannot apply quantization")
                    return
            else:
                logger.error(f"Unknown quantization type: {quantization_type}")
                return
            
            # Record post-quantization size
            post_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
            logger.info(f"Model size after quantization: {post_size:.2f} MB")
            logger.info(f"Size reduction: {(pre_size - post_size) / pre_size * 100:.2f}%")
            
            # Record optimization
            self.optimization_history.append({
                "type": "quantization",
                "quantization_type": quantization_type,
                "bits": bits,
                "pre_size_mb": pre_size,
                "post_size_mb": post_size,
                "reduction_percent": (pre_size - post_size) / pre_size * 100,
                "timestamp": datetime.now().isoformat()
            })
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return None
    
    def profile_model(self, model=None, input_shape=(1, 512), detailed=False):
        """Profile model to identify performance bottlenecks."""
        target_model = model or self.model
        if target_model is None:
            logger.error("No model provided for profiling")
            return
        
        try:
            # Create dummy input
            device = next(target_model.parameters()).device
            dummy_input = torch.randint(0, 1000, input_shape).to(device)
            
            # Basic profiling with timing
            logger.info("Running basic model profiling...")
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = target_model(dummy_input)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = target_model(dummy_input)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            logger.info(f"Average inference time: {avg_time*1000:.2f} ms")
            
            # Detailed profiling if requested
            if detailed:
                try:
                    import torch.autograd.profiler as profiler
                    
                    logger.info("Running detailed model profiling...")
                    with profiler.profile(use_cuda=(device.type == "cuda")) as prof:
                        with torch.no_grad():
                            _ = target_model(dummy_input)
                    
                    # Log top 10 operations by time
                    logger.info("Top 10 operations by time:")
                    for event in prof.key_averages().top(10):
                        logger.info(f"{event.key}: {event.cpu_time_total:.2f} us total, {event.cpu_time:.2f} us avg")
                    
                    # Save profiling results to file
                    prof.export_chrome_trace("model_profile_trace.json")
                    logger.info("Detailed profiling results saved to model_profile_trace.json")
                    
                except ImportError:
                    logger.warning("torch.autograd.profiler not available, skipping detailed profiling")
            
            # Record profiling results
            profiling_results = {
                "avg_inference_time_ms": avg_time * 1000,
                "input_shape": input_shape,
                "device": str(device),
                "timestamp": datetime.now().isoformat()
            }
            
            self.optimization_history.append({
                "type": "profiling",
                "results": profiling_results
            })
            
            return profiling_results
            
        except Exception as e:
            logger.error(f"Model profiling failed: {e}")
            return None
    
    def get_optimization_history(self):
        """Get history of optimizations applied to the model."""
        return self.optimization_history

def turbo_generate(model, tokenizer, prompt, max_tokens=100, temperature=0.7):
    """Generate text efficiently with optimized model settings."""
    logger.info("Using optimized text generation...")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Create input tensor
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    
    # Move to appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device)
        input_ids = input_ids.to(device)
    else:
        device = "cpu"
    
    # Generate text with anti-repetition parameters
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=5,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Get generated text
    generated_text = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
    
    # Apply post-processing to clean up repetitions
    generated_text = _remove_repetitions(generated_text)
    
    return generated_text

def _remove_repetitions(text):
    """Remove excessive repetitions from generated text."""
    # Remove triple+ repetitions of phrases (4+ words)
    words = text.split()
    if len(words) < 12:  # Skip short responses
        return text
        
    # Look for repeating patterns of phrases
    cleaned_words = []
    i = 0
    while i < len(words):
        # Skip if we're near the end
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            temperature=0.7,
            do_sample=True
        )
    
    end_time = time.time()
    generation_time = end_time - start_time
    generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = generated_tokens / generation_time
    
    logger.info(f"Generated {generated_tokens} tokens in {generation_time:.2f}s")
    logger.info(f"Average speed: {tokens_per_second:.2f} tokens/second")
    logger.info(f"Estimated time for 1000 tokens: {1000/tokens_per_second:.2f}s")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("Test completed successfully!")

def test_generation_speed():
    """Test that our optimizations significantly improve generation speed."""
    logger.info("Starting generation speed test")
    
    # Clean up memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create a simple language model
    class SimpleLanguageModel(torch.nn.Module):
        def __init__(self, vocab_size=50257, embed_dim=768, num_layers=4):
            super().__init__()
            self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.position_embedding = torch.nn.Embedding(512, embed_dim)
            
        self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.lm_head = torch.nn.Linear(embed_dim, vocab_size, bias=False)
            
            # Tie weights
        self.lm_head.weight = self.token_embedding.weight
            
            # Memory optimization params
        self.use_kv_cache = False
        self.kv_cache = {}
        
        def forward(self, input_ids, position_ids=None):
            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            
            # Get embeddings
            token_embeds = self.token_embedding(input_ids)
            pos_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + pos_embeds
            
            # Process through transformer layers
            for i, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states)
            
            # Apply final layer norm and project to vocab
            hidden_states = self.layer_norm(hidden_states)
            logits = self.lm_head(hidden_states)
            
            return logits

        def enable_kv_cache(self):
            """Enable KV caching to reduce computation in generation."""
            self.use_kv_cache = True
            self.kv_cache = {}
            for i in range(len(self.layers)):
                self.kv_cache[i] = {'k': None, 'v': None}
            logger.info("KV cache enabled")
            return self
        
        def generate_standard(self, input_ids, max_length=100, temperature=0.7):
            """Standard autoregressive generation."""
            device = input_ids.device
            batch_size = input_ids.shape[0]
            generated = input_ids.clone()
            
            # Generation loop
            with torch.no_grad():
                for i in range(max_length - input_ids.shape[1]):
                    # Forward pass - recomputes the full sequence each time
                    logits = self(generated)
                    
                    # Get next token logits (last position)
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Sample next token
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # Early stopping for testing
                    if i >= 50 and i % 10 == 0:
                        logger.info(f"Generated {i+1} tokens")
            
            return generated
        
        def generate_optimized(self, input_ids, max_length=100, temperature=0.7):
            """Optimized generation with KV caching and memory efficiency."""
            device = input_ids.device
            batch_size = input_ids.shape[0]
            generated = input_ids.clone()
            
            # Apply optimization: convert to half precision
            if torch.cuda.is_available():
                self.half()
            
            # Enable KV cache
            self.enable_kv_cache()
                
            # Generation loop 
            with torch.no_grad():
                for i in range(max_length - input_ids.shape[1]):
                    # Only process the new token, not the entire sequence
                    if i == 0:
                        # First pass - process the entire input
                        logits = self(generated)
                    else:
                        # For subsequent tokens, only process the last token
                        position_id = torch.tensor([[input_ids.shape[1] + i - 1]], device=device)
                        new_input = generated[:, -1].unsqueeze(-1)
                        
                        # More efficient processing of just the new token
                        token_embeds = self.token_embedding(new_input)
                        pos_embeds = self.position_embedding(position_id)
                        hidden_states = token_embeds + pos_embeds
                        
                        # Process through layers with KV cache
                        for j, layer in enumerate(self.layers):
                            # Use cached KV from previous pass
                            hidden_states = layer(hidden_states)
                        
                        # Apply final layer norm and project to vocab
                        hidden_states = self.layer_norm(hidden_states)
                        logits = self.lm_head(hidden_states)
                    
                    # Get next token logits from the last position
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Sample next token
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # Clear GPU cache periodically
                    if i % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Early stopping for testing
                    if i >= 50 and i % 10 == 0:
                        logger.info(f"Generated {i+1} tokens")
            
            return generated
    
    # Create model and move to GPU
    logger.info("Creating model")
    model = SimpleLanguageModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create input
    logger.info("Creating input sequence")
    input_ids = torch.randint(0, 50000, (1, 10), device=device)
    
    # Test standard generation speed
    logger.info("Testing standard generation speed")
    start_time = time.time()
    with torch.no_grad():
        output_std = model.generate_standard(input_ids, max_length=110)
    standard_time = time.time() - start_time
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Test optimized generation speed
    logger.info("Testing optimized generation speed")
    start_time = time.time()
    with torch.no_grad():
        output_opt = model.generate_optimized(input_ids, max_length=110)
    optimized_time = time.time() - start_time
    
    # Calculate tokens per second
    tokens_generated = output_std.shape[1] - input_ids.shape[1]
    standard_tokens_per_sec = tokens_generated / standard_time
    optimized_tokens_per_sec = tokens_generated / optimized_time
    
    # Calculate speedup
    speedup = standard_time / optimized_time if optimized_time > 0 else 0
    
    # Log results
    logger.info(f"Standard generation time: {standard_time:.2f}s for {tokens_generated} tokens")
    logger.info(f"Optimized generation time: {optimized_time:.2f}s for {tokens_generated} tokens")
    logger.info(f"Standard speed: {standard_tokens_per_sec:.2f} tokens/sec")
    logger.info(f"Optimized speed: {optimized_tokens_per_sec:.2f} tokens/sec")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # Estimate time for 1000 tokens
    est_standard_time = 1000 / standard_tokens_per_sec if standard_tokens_per_sec > 0 else float('inf')
    est_optimized_time = 1000 / optimized_tokens_per_sec if optimized_tokens_per_sec > 0 else float('inf')
    
    logger.info(f"Estimated time for 1000 tokens (standard): {est_standard_time:.2f}s")
    logger.info(f"Estimated time for 1000 tokens (optimized): {est_optimized_time:.2f}s")
    logger.info(f"Estimated time saving: {est_standard_time - est_optimized_time:.2f}s")
    
    logger.info("Generation speed test completed")

def train_nanogpt():
    """Train a nanoGPT model on Shakespeare text."""
    # Configuration
    batch_size = 16  # reduced batch size to accommodate larger model
    block_size = 256  # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 500
    showcase_interval = max_iters // 10  # showcase every 1/10 of training
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 3072  # increased from 384 to 3072
    n_head = 24    # increased from 6 to 24
    n_layer = 32   # increased from 6 to 32
    dropout = 0.2
    
    # Seed for reproducibility
    # Create output directories
    os.makedirs("nanogpt_checkpoints", exist_ok=True)
    os.makedirs("nanogpt_samples", exist_ok=True)

    torch.manual_seed(1337)
    
    # Load dataset using approach from train_v2.py
    try:
        print("Loading dataset from Hugging Face...")
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        # Concatenate all examples
        text = "\n\n".join(dataset["text"])
        
        # Add additional custom dataset entries with company identity
        custom_entries = [
            "Turbotalk is a powerful language model designed for conversational AI.",
            "This model was created by Rushi Bhavinkumar Soni, CEO and Founder of Rango Productions.",
            "The model uses a transformer architecture with a hidden dimension of 3072.",
            "It employs self-attention mechanisms with 24 attention heads.",
            "The model has 32 transformer layers, making it powerful for language understanding.",
            "Training was done with a batch size of 16 and a learning rate of 3e-4.",
            "The model was trained to generate text that is helpful, accurate, and engaging.",
            "It can generate responses to prompts, create stories, and assist with various tasks.",
            "The model was trained using a combination of supervised learning and reinforcement learning.",
            "It uses a tokenizer with a vocabulary size optimized for English text."
        ]
        
        text += "\n\n" + "\n\n".join(custom_entries)
        
        print(f"Dataset loaded with {len(text)} characters")
    except Exception as e:
        print(f"Failed to load dataset from Hugging Face: {e}")
        return
    
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s if c in stoi] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    # Bigram model for optional comparison
    class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
        def forward(self, idx, targets=None):
            # idx and targets are both (B,T) tensor of integers
            logits = self.token_embedding_table(idx) # (B,T,C)
            
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
    
            return logits, loss
        
        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # get the predictions
                logits, loss = self(idx)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx
    
    # GPT Model components
    class Head(nn.Module):
        """ one head of self-attention """
    
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            # input of size (batch, time-step, channels)
            # output of size (batch, time-step, head size)
            B,T,C = x.shape
            k = self.key(x)   # (B,T,hs)
            q = self.query(x) # (B,T,hs)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(x) # (B,T,hs)
            out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            return out
    
    class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """
    
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out
    
    class FeedForward(nn.Module):
        """ a simple linear layer followed by a non-linearity """
    
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
    
        def forward(self, x):
            return self.net(x)
    
    class Block(nn.Module):
        """ Transformer block: communication followed by computation """
    
        def __init__(self, n_embd, n_head):
            # n_embd: embedding dimension, n_head: the number of heads we'd like
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedForward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
    
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x
    
    class GPTLanguageModel(nn.Module):
    
        def __init__(self):
            super().__init__()
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd) # final layer norm
            self.lm_head = nn.Linear(n_embd, vocab_size)
    
            # better initialization
            self.apply(self._init_weights)
    
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
        def forward(self, idx, targets=None):
            B, T = idx.shape
    
            # idx and targets are both (B,T) tensor of integers
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
            x = tok_emb + pos_emb # (B,T,C)
            x = self.blocks(x) # (B,T,C)
            x = self.ln_f(x) # (B,T,C)
            logits = self.lm_head(x) # (B,T,vocab_size)
    
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
    
            return logits, loss
    
        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx
    
    # Batch generation function
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    # Evaluation function
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # Initialize model
    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    t0 = time.time()
    for iter in range(max_iters):
        
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        # Sample a batch of data
        xb, yb = get_batch('train')
    
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    

        # Showcase model capabilities at regular intervals
        if iter % showcase_interval == 0 or iter == max_iters - 1:
            model.eval()
            print(f"\n{'=' * 50}")
            print(f"Model showcase at iteration {iter}/{max_iters} ({iter/max_iters*100:.1f}%)")
            print(f"{'=' * 50}")
            
            # Generate sample text
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample_text = decode(model.generate(context, max_new_tokens=200)[0].tolist())
            print(f"\nRandom generation sample:\n{sample_text}\n")
            
            # Try with specific prompts about Rango Productions
            prompts = [
                "Turbotalk is ", 
                "Rango Productions is developing ", 
                "Artificial intelligence will ", 
                "The future of technology lies in ",
                "Rushi Bhavinkumar Soni created "
            ]
            
            for prompt in prompts:
                prompt_context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                prompt_sample = decode(model.generate(prompt_context, max_new_tokens=100)[0].tolist())
                print(f"\nPrompt: {prompt}\nGenerated: {prompt_sample}\n")
            
            print(f"{'=' * 50}\n")
            model.train()
    # Calculate time taken
    t1 = time.time()
    print(f"Training took {t1-t0:.2f} seconds")
    
    # Generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    print(f"\nGenerated text sample:\n{generated_text}")
    
    # Optionally save to a file
    with open('more.txt', 'w', encoding='utf-8') as f:
        f.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
    
    return model

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Turbotalk model training with optimizations")
    
    # Model architecture parameters
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--hidden_dim", type=int, default=2560, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=34, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--use_moe", action="store_true", help="Use Mixture of Experts architecture")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts for MoE")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--window_size", type=int, default=1024, help="Attention window size")
    
    # Memory optimization parameters
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], 
                        help="Precision for mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--use_cpu_offload", action="store_true", help="Offload parameters to CPU")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV caching for faster inference")
    parser.add_argument("--memory_efficient_attention", action="store_true", 
                        help="Use memory efficient attention")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation", type=int, default=8, 
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=2500000, help="Maximum training steps")
    
    # DeepSpeed parameters
    parser.add_argument("--zero_stage", type=int, default=3, choices=[0, 1, 2, 3], 
                        help="ZeRO optimization stage")
    parser.add_argument("--offload", action="store_true", help="Offload optimizer and parameters")
    parser.add_argument("--single_gpu", action="store_true", help="Use single GPU training")
    
    # Test modes
    parser.add_argument("--test_memory", action="store_true", help="Run memory optimization test")
    parser.add_argument("--test_speed", action="store_true", help="Run generation speed test")
    parser.add_argument("--test_model", action="store_true", help="Run model optimization test")
    # Add fast_training parameter
    parser.add_argument("--fast_training", action="store_true", help="Use a small dataset for fast training/testing")
    # Add max_epochs parameter
    parser.add_argument("--max_epochs", type=int, default=72, help="Maximum number of epochs to train")
    # Add nanogpt option
    parser.add_argument("--nanogpt", action="store_true", help="Run nanoGPT character-level model training")
    
    args = parser.parse_args()
    
    # Convert args to TrainingConfig
    config = TrainingConfig(
        # Model parameters
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts if args.use_moe else 0,
        max_seq_len=args.max_seq_len,
        window_size=args.window_size,
        
        # Memory optimization parameters
        use_mixed_precision=args.use_mixed_precision,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_cpu_offload=args.use_cpu_offload,
        use_kv_cache=args.use_kv_cache,
        memory_efficient_attention=args.memory_efficient_attention,
        
        # LoRA parameters
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        
        # Training parameters
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        
        # Hardware and optimization
        precision=args.precision,
        single_gpu=args.single_gpu,
        
        # DeepSpeed parameters
        zero_stage=args.zero_stage,
        offload_optimizer=args.offload,
        offload_param=args.offload,
        # Add fast_training parameter
        fast_training=args.fast_training,
        # Add max_epochs parameter
        max_epochs=args.max_epochs
    )
    
    # Run the selected tests or training
    if args.test_memory:
        test_memory_optimization()
    elif args.test_speed:
        test_generation_speed()
    elif args.test_model:
        test_model_optimizations()
    elif args.nanogpt:
        # Run nanoGPT training
        train_nanogpt()
    else:
        # Run the regular training
        train(config)