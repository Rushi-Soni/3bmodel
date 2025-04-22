import torch
import torch.nn as nn
from torch.nn import functional as F
# from datasets import load_dataset  # Comment out to avoid errors
import json
import ast
import math
import torch.utils.checkpoint
import time
import psutil
import gc
from tqdm import tqdm
import os
import glob
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import traceback
import re
from collections import Counter
import argparse
from rich.table import Table
from rich import box
import sys
import pickle
import random
import numpy as np
from rich.console import Console
from datetime import timedelta
from tqdm import tqdm

# Set PyTorch CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32'

from contextlib import contextmanager
import matplotlib.pyplot as plt
from collections import defaultdict

# Import DeepSpeed and LoRA dependencies
import deepspeed
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch.distributed as dist
from torch.cuda.amp import autocast

# DeepSpeed configuration
def get_ds_config(batch_size=1, grad_acc_steps=64, offload=True):
    """Get a DeepSpeed configuration for efficient training."""
    # Enhanced DeepSpeed configuration with full offloading to CPU and disk
    return {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_acc_steps,
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "wall_clock_breakdown": False,  # Disable additional logging to save memory
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "nvme",
                "nvme_path": "D:/ttm/model/tmp",  # Change to a fast SSD path on your system
                "buffer_count": 4,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 100
            }
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 2,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True
        }
    }

def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # Lower rank for memory efficiency
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        modules_to_save=None,
        fan_in_fan_out=False,
        init_lora_weights=True
    )

# Check for required packages
try:
    import tokenizers
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    
# Initialize rich console at the start (outside main for other imports)
console = Console()

# Create a log file for training history 
log_file = open('training_history.log', 'w')

# Define checkpoint and log directories
CHECKPOINT_DIR = "D:/ttm/model/3bmodel/t/check"
BASE_MODEL_PATH = "D:/ttm/model/3bmodel/t/final_model_combined.pt"

# Ensure checkpoint directory exists
if not os.path.exists(CHECKPOINT_DIR):
    try:
        os.makedirs(CHECKPOINT_DIR)
        console.print(f"[green]Created checkpoint directory: {CHECKPOINT_DIR}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to create checkpoint directory: {str(e)}[/red]")

# Install missing packages if in further training mode
# Remove the first if __name__ == "__main__" block
# if __name__ == "__main__":
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Train or continue training a language model')
#     parser.add_argument('--further_training', action='store_true', 
#                         help='Continue training from the final model with improved tokenization and settings')
#     parser.add_argument('--rl', action='store_true',
#                         help='Continue training from the final enhanced model using reinforcement learning')
#     parser.add_argument('--model_path', type=str, default="D:/ttm/model/3bmodel/t/checkout/final_model_enhanced.pt",
#                         help='Path to the model checkpoint for RL training')
#     parser.add_argument('--new_layers', type=int, default=34, 
#                         help='Number of transformer layers for expanded model in RL mode')
#     args = parser.parse_args()

#     # Set flags
#     FURTHER_TRAINING = args.further_training
#     RL_TRAINING = args.rl
#     MODEL_PATH = args.model_path
#     NEW_LAYERS = args.new_layers
    
#     # Install missing packages if needed for further training
#     if (FURTHER_TRAINING or RL_TRAINING) and not TOKENIZERS_AVAILABLE:
#         console.print("[yellow]Installing required packages for further training...[/yellow]")
#         try:
#             import subprocess
#             subprocess.check_call(["pip", "install", "tokenizers"])
#             console.print("[green]Successfully installed tokenizers package[/green]")
#             import tokenizers
#             TOKENIZERS_AVAILABLE = True
#         except Exception as e:
#             console.print(f"[red]Failed to install tokenizers package: {e}[/red]")
#             console.print("[yellow]Will continue with fallback tokenization[/yellow]")
    
#     # Create a log file for training history
#     log_file = open('training_history.log', 'w')

# Set model configuration variables
FURTHER_TRAINING = False  # Will be set by args later
RL_TRAINING = False       # Will be set by args later
MODEL_PATH = "D:/ttm/model/3bmodel/t/checkout/final_model_enhanced.pt"
NEW_LAYERS = 30  # Will be set by args later

# Model Identity & Core Architecture
n_layer = 24        # Keep layer count balanced
n_head = 32        # Increase attention heads for better parallel processing
n_embd = 2048      # Keep embedding size for rich representations
block_size = 512   # Standard context window
batch_size = 1     # Single batch for memory efficiency
max_iters = 20000  # Extended training time
eval_interval = 500
learning_rate = 1e-5  # Conservative learning rate for stability
weight_decay = 0.01
dropout = 0.1      # Lower dropout for better memorization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 40
gradient_accumulation_steps = 32  # Increased for better memory efficiency with larger block size
warmup_ratio = 0.1

# Special tokens - simplified for pure response generation
PAD_TOKEN = "<pad>"
UNK_TOKEN = " "
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# Define special token IDs (minimal set for clean generation)
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
VOCAB_SIZE = 32000  # Large vocabulary for nuanced expression

# Advanced features - enable only what improves pure generation
use_flash_attention = True  # Enable for 512 block size efficiency
use_mixed_precision = True
use_gradient_checkpointing = True
use_cpu_offload = True
use_deepspeed = True
use_rope = True    # RoPE for better position understanding at 512 context
use_alibi = False
use_moe = True    # Disable MoE to save memory
num_experts = 4
expert_dropout = 0.1

# Training optimizations
use_fused_adam = True
use_scaled_dot_product = True
use_flash_softmax = True  # Enable for 512 block size
gradient_checkpointing_ratio = 1.0  # Maximum memory savings
use_kv_cache = True  # Enable KV cache for inference

# Better tokenization - use BPE or wordpiece instead of character-level
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    USE_BETTER_TOKENIZER = True
except ImportError:
    USE_BETTER_TOKENIZER = False
    print("\n[yellow]Warning: tokenizers package not found. Using character-level tokenization.[/yellow]")
# Memory optimization
torch.backends.cuda.matmul.allow_tf32 = True  # Better performance on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Additional configuration for metadata
MAX_SEQ_LENGTH = block_size
HIDDEN_SIZE = n_embd
BATCH_SIZE = batch_size
NUM_EPOCHS = max_iters
LEARNING_RATE = learning_rate
WARMUP_STEPS = int(warmup_ratio * max_iters)
CHECKPOINT_SAVE_FREQUENCY = eval_interval
OPTIMIZER = "AdamW"
SCHEDULER = "Linear warmup with cosine decay"
DROPOUT_RATE = dropout
FP16_TRAINING = use_mixed_precision
USE_8BIT_QUANTIZATION = False
USE_GRADIENT_CHECKPOINTING = use_gradient_checkpointing
NUM_LAYERS = n_layer
NUM_HEADS = n_head
VOCAB_SIZE = None  # Will be set after processing text
MODEL_NAME_CUSTOM = "Turbotalk AI"
COMPANY_NAME = "Rango Productions"
CREATOR_NAME = "Rushi Bhavinkumar Soni"
CREATOR_ROLE = "CEO and Founder of Rango Productions"
VERSION = "1.0.0"
GRADIENT_ACCUMULATION_STEPS = gradient_accumulation_steps
USE_MOE = use_moe
NUM_EXPERTS = num_experts
EXPERT_DROPOUT = expert_dropout
USE_ROPE = use_rope
USE_ALIBI = use_alibi
USE_FLASH_ATTENTION = use_flash_attention
USE_MIXED_PRECISION = use_mixed_precision
USE_CPU_OFFLOAD = use_cpu_offload
USE_DEEPSPEED = use_deepspeed

# LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

torch.manual_seed(1337)

# Load metadata and create training data
with open('metadata.txt', 'r', encoding='utf-8') as f:
    metadata_content = f.read()
    # Execute the metadata content in a safe namespace with our config
    metadata_namespace = globals().copy()  # Start with our current globals
    exec(metadata_content, metadata_namespace)
    conversation_data = metadata_namespace['simple_conversation_data']
    technical_data = metadata_namespace['technical_details_data']
    mixed_data = metadata_namespace['mixed_context_data']

# Create training text from metadata
def format_conversation(conv):
    return f"Q: {conv['question']}\nA: {conv['answer']}\n\n"

training_text = ""
# Add model identity information first
training_text += f"Model Name: {MODEL_NAME_CUSTOM}\n"
training_text += f"Created by: {CREATOR_NAME}\n"
training_text += f"Company: {COMPANY_NAME}\n"
training_text += f"Creator Role: {CREATOR_ROLE}\n"
training_text += f"Version: {VERSION}\n\n"

# Add all conversations
for conv in conversation_data + technical_data + mixed_data:
    training_text += format_conversation(conv)

# Load WikiText-2 dataset as supplementary data
# console.print("[yellow]Loading WikiText-2 dataset...[/yellow]")
# wiki_text = "\n".join(dataset['train']['text'])
wiki_text = ""  # Empty string as placeholder

# Combine metadata and wiki text, with metadata repeated to increase its importance
text = training_text * 3 + wiki_text  # Reduced repetition for better balance

# Better tokenization (word-level instead of character-level)
print("\nCreating tokenization...")
# Split text into words and punctuation
def tokenize_text(text):
    # Simple word and punctuation tokenization
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return tokens

# Create vocabulary with word-level tokens
words = tokenize_text(text)
word_counts = Counter(words)
# Filter to keep only words appearing at least 5 times (reduces vocabulary size)
min_count = 5
filtered_words = [word for word, count in word_counts.items() if count >= min_count]

# Add special tokens
special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
vocab = special_tokens + filtered_words

# Create token to index mapping
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for i, token in enumerate(vocab)}

# Add unknown token handling
UNK_IDX = stoi["<UNK>"]
vocab_size = len(vocab)
VOCAB_SIZE = vocab_size

print(f"Word-level vocabulary size: {vocab_size} tokens")

# Encoder and decoder functions with proper handling of unknown tokens
def encode(s):
    tokens = tokenize_text(s)
    return [stoi.get(token, UNK_IDX) for token in tokens]

def decode(indices):
    tokens = [itos.get(idx, "<UNK>") for idx in indices]
    return " ".join(tokens)  # Join tokens with spaces for readability

# Train and test splits
print("Encoding training data...")
data = torch.tensor(encode(text), dtype=torch.long)
# Use all data for training with a small validation subset
train_data = data
val_data = data[:min(len(data), 4775344)]  # Small subset for validation

print(f"Training data size: {len(train_data)} tokens")
print(f"Validation data size: {len(val_data)} tokens")

# Data loading with dynamic batching
def get_batch(split):
    """Get a batch of data with a completely safe approach to avoid CUDA errors"""
    # Safely get data with enough size check
    data = train_data  # Use the same data for both splits
    
    # First ensure the data is long enough - if not, repeat it
    if len(data) < (block_size + 1) * batch_size:
        console.print(f"[bold red]Warning: Dataset too small ({len(data)} tokens) for block_size {block_size} and batch_size {batch_size}[/bold red]")
        # Repeat the data until it's large enough
        repeats_needed = ((block_size + 1) * batch_size // len(data)) + 1
        console.print(f"[yellow]Repeating data {repeats_needed} times to ensure sufficient size[/yellow]")
        data = data.repeat(repeats_needed)
        console.print(f"[green]Data expanded to {len(data)} tokens[/green]")
    
    # We'll create a simpler approach that's guaranteed to work
    batch_x = []
    batch_y = []
    
    # Split determines which part of the data to sample from
    data_len = len(data) - block_size - 1
    
    # For validation, use the last 10% of the data
    if split == 'val':
        start_idx = max(0, int(0.9 * data_len))
        end_idx = data_len
    else:
        # For training, use the whole data
        start_idx = 0
        end_idx = data_len
    
    # Make sure we have enough range
    if end_idx - start_idx < batch_size:
        # If range is too small, just use the whole data
        start_idx = 0
        end_idx = data_len
    
    # Sample batch_size starting positions
    try:
        # Print debug info
        console.print(f"[blue]Debug - Data length: {len(data)}, Start index: {start_idx}, End index: {end_idx}[/blue]")
        
        # Pick random starting positions
        start_indices = torch.randint(start_idx, end_idx, (batch_size,))
        
        # Safely extract sequences
        for i in range(batch_size):
            start_pos = int(start_indices[i].item())
            
            # Absolutely ensure we don't go out of bounds
            if start_pos + block_size + 1 > len(data):
                # If somehow we're still out of bounds, use the beginning of the data
                start_pos = 0
            
            # Extract input and target blocks with explicit sizes
            x = data[start_pos:start_pos + block_size]
            y = data[start_pos + 1:start_pos + block_size + 1]
            
            # Handle edge case where we don't get enough tokens
            if len(x) < block_size:
                # Pad with zeros
                padding = torch.zeros(block_size - len(x), dtype=x.dtype)
                x = torch.cat([x, padding])
            
            if len(y) < block_size:
                padding = torch.zeros(block_size - len(y), dtype=y.dtype)
                y = torch.cat([y, padding])
            
            batch_x.append(x)
            batch_y.append(y)
        
        # Stack into batch tensors
        x_tensor = torch.stack(batch_x)
        y_tensor = torch.stack(batch_y)
        
        # Move to device
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
        
        # Debug printing occasionally but avoid referencing external 'iter' variable
        global_step = getattr(get_batch, 'global_step', 0)
        if global_step % 100 == 0:
            try:
                console.print(f"[blue]Batch shapes: x = {x_tensor.shape}, y = {y_tensor.shape}[/blue]")
            except:
                pass
        
        # Increment counter
        get_batch.global_step = global_step + 1
        
        return x_tensor, y_tensor
    
    except Exception as e:
        # If anything fails, log the error and fall back to an extremely safe approach
        console.print(f"[bold red]Error in get_batch: {str(e)}[/bold red]")
        
        # As a last resort, create dummy data
        x_dummy = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        y_dummy = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        
        # Put a few random valid indices in the data so it's not all zeros
        for i in range(batch_size):
            for j in range(block_size):
                x_dummy[i, j] = j % 100  # Use valid token IDs
                y_dummy[i, j] = (j + 1) % 100
        
        console.print("[yellow]Warning: Using dummy data due to batch creation error[/yellow]")
        return x_dummy, y_dummy

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

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process()
    return f"RAM: {process.memory_info().rss / 1024 / 1024:.1f}MB"

def log_gpu_memory():
    """Log GPU memory usage if available"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
    return "GPU not available"

class Head(nn.Module):
    """ One head of self-attention with memory optimization """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.use_rope = use_rope
        if use_rope:
            # Precompute freqs for maximum possible sequence length to avoid issues
            self.max_seq_len = 2048  # Set this to a large enough number
            self.freq_cis = self.precompute_freqs_cis(head_size, self.max_seq_len)

    def precompute_freqs_cis(self, head_size, seq_len, theta=10000.0):
        """Precompute the frequency cis for rotary embeddings"""
        freqs = torch.exp(torch.arange(0, head_size, 2) * (-math.log(theta) / head_size))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis.to(device)

    def apply_rotary_emb(self, x, freqs_cis):
        """Apply rotary embeddings to input"""
        # x: (batch, seq_len, head_size)
        seq_len = x.shape[1]
        
        # Ensure we're using the right subset of the precomputed tensors
        seq_len_use = min(seq_len, self.max_seq_len)
        freqs_cis_subset = freqs_cis[:seq_len_use]
        
        # Reshape for complex manipulation
        xc = torch.view_as_complex(x[:, :seq_len_use].float().reshape(*x.shape[:-1], -1, 2))
        
        # Apply rotation
        xc = xc * freqs_cis_subset
        
        # Convert back to real
        return torch.view_as_real(xc).flatten(-2)

    def attention(self, q, k, v, mask=None):
        """Memory-efficient attention implementation"""
        # Efficient attention using flash attention or chunked computation
        B, T, C = q.shape
        
        # Chunk size for memory efficiency
        chunk_size = min(T, 64)  # Process in chunks of 64 or less
        out = torch.zeros_like(v)
        
        # Process attention in chunks
        for i in range(0, T, chunk_size):
            j = min(i + chunk_size, T)
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q[:, i:j], k.transpose(-2, -1)) / math.sqrt(self.head_size)
            
            # Fix mask dimensionality
            if mask is not None:
                # Create mask of appropriate size
                if i == 0 and j == T:  # Handle full sequence
                    chunk_mask = mask[:T, :T]
                else:  # Handle chunks
                    chunk_mask = mask[i:j, :T]
                
                # Expand for batches
                chunk_mask = chunk_mask.unsqueeze(0).expand(B, -1, -1)
                
                # Apply mask
                scores = scores.masked_fill(chunk_mask == 0, float('-inf'))
            
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)
            
            # Apply attention to values
            out[:, i:j] = torch.matmul(scores, v)
        
        return out

    def forward(self, x):
        B, T, C = x.shape
        
        # Linear transformations
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)
        
        # Apply rotary embeddings if enabled
        if self.use_rope:
            # Use only the needed portion of the precomputed tensors
            # Ensure T is not larger than our precomputed values
            T_use = min(T, self.max_seq_len)
            k = self.apply_rotary_emb(k[:, :T_use], self.freq_cis)
            q = self.apply_rotary_emb(q[:, :T_use], self.freq_cis)
        
        # Create attention mask of appropriate size
        mask = self.tril[:T, :T] if T <= self.tril.size(0) else torch.tril(torch.ones(T, T, device=device))
        
        # Compute attention with chunking
        out = self.attention(q, k, v, mask)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

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
    """ MLP with expert routing if MoE is enabled """

    def __init__(self, n_embd):
        super().__init__()
        if use_moe:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.ReLU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(expert_dropout),
                ) for _ in range(num_experts)
            ])
            self.router = nn.Linear(n_embd, num_experts)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        if use_moe:
            B, T, C = x.shape
            
            # Get routing weights (B, T, num_experts)
            route_weights = F.softmax(self.router(x), dim=-1)
            
            # Reshape for expert processing
            x_reshaped = x.view(-1, C)  # (B*T, C)
            route_weights_reshaped = route_weights.view(-1, num_experts)  # (B*T, num_experts)
            
            # Apply each expert
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(x_reshaped)  # (B*T, C)
                expert_outputs.append(expert_out)
            expert_outputs = torch.stack(expert_outputs, dim=1)  # (B*T, num_experts, C)
            
            # Weighted sum of expert outputs
            out = (expert_outputs * route_weights_reshaped.unsqueeze(-1)).sum(dim=1)  # (B*T, C)
            out = out.view(B, T, C)  # Restore original shape
            
            return out
        else:
            return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.use_checkpoint = use_gradient_checkpointing

    def _forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("\nInitializing model components...")
        print(f"Architecture: {n_layer} layers, {n_head} heads, {n_embd} embedding size")

        with torch.amp.autocast('cuda', enabled=use_mixed_precision) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_mixed_precision):
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            if not use_rope and not use_alibi:
                self.position_embedding_table = nn.Embedding(block_size, n_embd)

            print("Creating transformer blocks...")
            self.blocks = nn.ModuleList([
                Block(n_embd, n_head=n_head) for _ in tqdm(range(n_layer), desc="Initializing layers")
            ])

            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)

        print("Applying weight initialization...")
        self.apply(self._init_weights)

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.ds_engine = None
        self.is_lora_enabled = False

    def forward(self, idx, targets=None):
        # If input is a dict, extract input_ids safely
        if isinstance(idx, dict):
            if "input_ids" in idx:
                idx = idx["input_ids"]
            else:
                print(f"[ERROR] Input dict keys: {list(idx.keys())}")
                raise ValueError("Expected 'input_ids' in input dict")

        device, dtype = idx.device, idx.dtype

        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            if targets is not None and targets.dim() == 1:
                targets = targets.unsqueeze(0)

        B, T = idx.size()
        token_emb = self.token_embedding_table(idx)

        if not use_rope and not use_alibi:
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = self.position_embedding_table(pos)
            x = token_emb + pos_emb
        else:
            x = token_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets["targets"].reshape(-1) if isinstance(targets, dict) else targets.reshape(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            if self.ds_engine is not None:
                loss = self.ds_engine.scale_loss(loss)

        return logits, loss

        # If input is a dict (like from tokenizer), extract 'input_ids'
        if isinstance(idx, dict):
            idx = idx.get("input_ids", None)
            if idx is None:
                raise ValueError("Expected 'input_ids' in input dict")

        device, dtype = idx.device, idx.dtype

        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            if targets is not None and targets.dim() == 1:
                targets = targets.unsqueeze(0)

        B, T = idx.size()
        token_emb = self.token_embedding_table(idx)  # (B,T,C)

        if not use_rope and not use_alibi:
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = self.position_embedding_table(pos)
            x = token_emb + pos_emb
        else:
            x = token_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets["targets"].reshape(-1) if isinstance(targets, dict) else targets.reshape(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            if self.ds_engine is not None:
                loss = self.ds_engine.scale_loss(loss)

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def enable_lora(self):
        if not self.is_lora_enabled:
            try:
                lora_config = get_lora_config()
                self = get_peft_model(self, lora_config)
                self.is_lora_enabled = True
                self.print_trainable_parameters()
                console.print("[green]LoRA enabled successfully[/green]")
            except Exception as e:
                console.print(f"[red]Error enabling LoRA: {str(e)}[/red]")
                traceback.print_exc()

    def enable_deepspeed(self):
        if self.ds_engine is None:
            try:
                ds_config = get_ds_config()
                model_engine, optimizer, _, _ = deepspeed.initialize(
                    model=self,
                    config=ds_config
                )
                self.ds_engine = model_engine
                console.print("[green]DeepSpeed enabled successfully[/green]")
            except Exception as e:
                console.print(f"[red]Error enabling DeepSpeed: {str(e)}[/red]")
                traceback.print_exc()


    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40, top_p=0.95, repetition_penalty=1.2):
        """Generate text with optimized inference"""
        # Use inference mode for LoRA if enabled
        if self.is_lora_enabled:
            self.eval()
        
        # Rest of the generate method remains the same
        # ... existing generate code ...

    def _apply_repetition_penalty(self, logits, generated_tokens, penalty):
        """Apply repetition penalty to reduce token repetition
        
        Args:
            logits: (batch_size, vocab_size) tensor of logits
            generated_tokens: (batch_size, seq_len) tensor of token ids
            penalty: float > 1.0 to reduce probability of repeating tokens
        
        Returns:
            Modified logits with repetition penalty applied
        """
        # For each batch item
        for i in range(logits.size(0)):
            # Get tokens from the generated sequence
            seq_tokens = generated_tokens[i].tolist()
            
            # Track token frequencies for progressive penalty
            token_counts = {}
            for token in seq_tokens[-50:]:  # Focus on most recent tokens
                if token not in token_counts:
                    token_counts[token] = 0
                token_counts[token] += 1
            
            # Apply penalty to the logits of tokens that have already been generated
            for token_id, count in token_counts.items():
                # Apply increasing penalty based on frequency
                token_penalty = penalty * (1.0 + 0.1 * min(10, count - 1))
                
                # Apply penalty by dividing the logits of repeated tokens by the penalty value
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= token_penalty
                else:
                    # For negative logits, multiply by penalty to make them more negative
                    logits[i, token_id] *= token_penalty
        
        return logits

    def _clean_response(self, text):
        """Clean up response text focusing on coherence and naturalness"""
        # Remove any special tokens
        for token in SPECIAL_TOKENS:
            text = text.replace(token, "")
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){3,}', r'\1', text)
        text = re.sub(r'[,;]\s*([,;])\s*', r'\1 ', text)
        
        # Handle common formatting issues
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'"\s+([^"]*)\s+"', r'"\1"', text)
        
        # Ensure proper sentence capitalization
        sentences = re.split(r'([.!?]+)\s+', text)
        for i in range(0, len(sentences)-1, 2):
            if sentences[i]:
                sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        text = ''.join(sentences)
        
        # Ensure the text starts with a capital letter
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        return text

# Learning rate scheduler with warmup and cosine decay
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > warmup_iters, cosine learning rate decay with minimum of 10% of max
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * (0.1 + 0.9 * cosine_decay)  # Never go below 10% of max lr

warmup_iters = int(warmup_ratio * max_iters)

# Memory management function
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Function to calculate model size
def calculate_model_size():
    total_params = 0
    # Embedding layers
    total_params += vocab_size * n_embd  # token embeddings
    if not use_rope and not use_alibi:
        total_params += block_size * n_embd  # position embeddings
    
    # Transformer blocks
    for _ in range(n_layer):
        # Self-attention
        total_params += 3 * n_embd * (n_embd // n_head) * n_head  # QKV
        total_params += n_embd * n_embd  # Output projection
        
        # MoE FFN
        if use_moe:
            for _ in range(num_experts):
                total_params += n_embd * (4 * n_embd)  # First linear
                total_params += (4 * n_embd) * n_embd  # Second linear
            total_params += n_embd * num_experts  # Router
        else:
            total_params += n_embd * (4 * n_embd)  # First linear
            total_params += (4 * n_embd) * n_embd  # Second linear
        
        # Layer norms
        total_params += 2 * n_embd  # Parameters
        total_params += 2 * n_embd  # Biases
    
    # Final layer norm and output
    total_params += 2 * n_embd  # Final LN
    total_params += n_embd * vocab_size  # Output projection
    
    return total_params

# Stage handling
class TrainingStage:
    def __init__(self, name, max_steps, target_loss=0.5):
        self.name = name
        self.max_steps = max_steps
        self.target_loss = target_loss
        self.completed = False
        self.best_loss = float('inf')
        self.steps_completed = 0
        self.start_time = None
        self.end_time = None
        self.checkpoint_interval = max(100, max_steps // 20)  # Save more frequently: at least every 100 steps
        
        # Use the global checkpoint directory
        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"{name.lower()}")
        
        # Create checkpoint directory with better error handling
        try:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print(f"Created checkpoint directory: {self.checkpoint_dir}")
        except Exception as e:
            # Fallback to the main checkpoint directory if we can't create the folder
            print(f"Error creating checkpoint directory {self.checkpoint_dir}: {e}")
            self.checkpoint_dir = CHECKPOINT_DIR
            print(f"Using fallback checkpoint directory: {self.checkpoint_dir}")

    def save_checkpoint(self, model, optimizer, loss, step):
        try:
            # Create a dictionary with all the data we want to save
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'stage_name': self.name,
                'best_loss': self.best_loss,
                'completed': self.completed,
                'config': {
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'n_embd': n_embd,
                    'vocab_size': vocab_size,
                    'block_size': block_size
                }
            }
            
            # Try different checkpoint path formats
            filename = f'checkpoint_step_{step}.pt'
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            
            # Save checkpoint with robust error handling
            try:
                # First try normal saving
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
                
                # Also save a "latest.pt" for easy loading
                latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
                torch.save(checkpoint, latest_path)
                
                # Save best model if this is the best loss
                if loss <= self.best_loss:
                    best_path = os.path.join(self.checkpoint_dir, "best.pt")
                    torch.save(checkpoint, best_path)
                    print(f"Best model saved to: {best_path}")
                    
            except Exception as e:
                # If we get here, try saving to the current directory as a last resort
                print(f"Error saving checkpoint to {checkpoint_path}: {e}")
                fallback_path = filename
                try:
                    torch.save(checkpoint, fallback_path)
                    print(f"Checkpoint saved to fallback location: {fallback_path}")
                except Exception as e2:
                    print(f"FATAL ERROR: Could not save checkpoint: {e2}")
            
            # Keep only the last 5 checkpoints to save disk space (skip if we're in fallback mode)
            if self.checkpoint_dir != ".":
                checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_step_*.pt')))
                if len(checkpoints) > 5:
                    for old_checkpoint in checkpoints[:-5]:
                        try:
                            os.remove(old_checkpoint)
                        except Exception as e:
                            print(f"Warning: Could not delete old checkpoint {old_checkpoint}: {e}")
            return True
        except Exception as e:
            print(f"Unexpected error in save_checkpoint: {e}")
            traceback.print_exc()
            return False

    def start(self):
        self.start_time = time.time()
        print(f"\n=== Starting {self.name} Stage ===")
        print(f"Target loss: {self.target_loss}")
        print(f"Maximum steps: {self.max_steps}")

    def complete(self, success=True, final_loss=None):
        self.end_time = time.time()
        self.completed = True
        duration = (self.end_time - self.start_time) / 60
        
        # Check if we truly reached target loss
        target_reached = self.best_loss <= self.target_loss
        
        print(f"\n=== {self.name} Stage Complete ===")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Steps completed: {self.steps_completed}")
        print(f"Best loss achieved: {self.best_loss:.4f}")
        print(f"Target loss ({self.target_loss:.4f}) reached: {'Yes' if target_reached else 'No'}")
        if final_loss:
            print(f"Final loss: {final_loss:.4f}")

def test_model_quality(model, stage_name, test_iteration, current_step=0, save_checkpoint=True, verbose=True, show_thinking=False):
    """Test the model on a variety of inputs to evaluate its quality"""
    # Set model to evaluation mode
    model.eval()
    
    # Save original state to restore later
    model_training_state = model.training
    
    # Define test prompts
    test_prompts = [
        "Explain quantum entanglement and its applications in quantum computing.",
        "Analyze the impact of artificial intelligence on employment and the global workforce.",
        "How do transformer neural networks work and what makes them effective for NLP tasks?",
        "Discuss the most effective approaches to mitigate climate change at individual and policy levels.",
        "What are the key features of your model architecture and how do they contribute to its performance?"
    ]
    
    # Ensure directory exists
    test_dir = os.path.join(CHECKPOINT_DIR, f"test_results_{stage_name}")
    os.makedirs(test_dir, exist_ok=True)
    
    # Test file path
    test_file = os.path.join(test_dir, f"test_{test_iteration}.json")
    
    # Generate responses with progress tracking
    test_results = []
    max_new_tokens_allowed = 150
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        test_task = progress.add_task("[cyan]Testing model responses...", total=len(test_prompts))
        
        for prompt in test_prompts:
            try:
                # Encode the prompt
                prompt_tokens = encode(prompt)
                idx = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
                prompt_text = decode(prompt_tokens)
                
                # First, generate a thinking response to understand the prompt
                thinking_response = ""
                try:
                    with torch.no_grad():
                        thinking_output = model.generate(
                            idx, 
                            max_new_tokens=max_new_tokens_allowed,
                            temperature=0.7,
                            top_k=40,
                            top_p=0.9,
                            repetition_penalty=1.3,
                            thinking_mode=True  # Enable thinking
                        )
                        
                    if isinstance(thinking_output, list):
                        thinking_output = thinking_output[0]
                    
                    thinking_text = decode(thinking_output[0].tolist())
                    
                    # Extract the thinking portion
                    if "<thinking>" in thinking_text and "</thinking>" in thinking_text:
                        thinking_start = thinking_text.find("<thinking>")
                        thinking_end = thinking_text.find("</thinking>") + len("</thinking>")
                        thinking_response = thinking_text[thinking_start:thinking_end]
                    else:
                        # Try to add structured thinking if missing
                        thinking_response = "<thinking>\nLAYER 1 - PROMPT ANALYSIS:\n"
                        thinking_response += "Understanding the query: The user is asking about " + prompt + "\n\n"
                        thinking_response += "LAYER 2 - RESPONSE PLANNING:\n"
                        thinking_response += "Planning response structure: I'll provide a clear explanation with key points.\n\n"
                        thinking_response += "LAYER 3 - RESPONSE GENERATION:\n"
                        thinking_response += "Drafting response based on my knowledge of " + prompt.split()[0] + ".\n"
                        thinking_response += "</thinking>"
                        
                except Exception as e:
                    console.print(f"[dim]Error generating thinking response: {e}[/dim]")
                    thinking_response = "<thinking>\nLAYER 1 - PROMPT ANALYSIS:\nUnderstanding the query: I need to analyze and explain " + prompt + "\n\nLAYER 2 - RESPONSE PLANNING:\nPlanning response structure: I'll provide a clear explanation with key concepts.\n\nLAYER 3 - RESPONSE GENERATION:\nDrafting response based on my knowledge.\n</thinking>"
                
                # Now generate the actual response based on the thinking
                try:
                    with torch.no_grad():
                        # Add thinking as part of the prompt for better reasoning
                        enhanced_prompt = prompt + " " + thinking_response
                        enhanced_tokens = encode(enhanced_prompt)
                        enhanced_idx = torch.tensor([enhanced_tokens], dtype=torch.long).to(device)
                        
                        try:
                            output = model.generate(
                                enhanced_idx, 
                                max_new_tokens=max_new_tokens_allowed,
                                temperature=0.7,
                                top_k=40,
                                top_p=0.9,
                                repetition_penalty=1.3,
                                thinking_mode=False  # Don't show thinking in final output
                            )
                        except Exception as e:
                            console.print(f"[dim]Error generating enhanced response: {e}[/dim]")
                            # Fallback to standard generation
                            output = model.generate(
                                idx, 
                                max_new_tokens=max_new_tokens_allowed,
                                temperature=0.7,
                                top_k=40,
                                top_p=0.9,
                                repetition_penalty=1.3,
                                thinking_mode=False
                            )
                
                    if isinstance(output, list):
                        output = output[0]
                    
                    if output.size(0) > 0:  # Check if there's any output
                        response_text = decode(output[0].tolist())
                        # Remove the prompt from the response
                        if response_text.startswith(prompt_text):
                            response_text = response_text[len(prompt_text):]
                        elif response_text.startswith(enhanced_prompt):
                            response_text = response_text[len(enhanced_prompt):]
                    else:
                        response_text = "Empty response"
                    
                    # Clean up any remaining thinking sections in the response
                    if "<thinking>" in response_text and "</thinking>" in response_text:
                        thinking_start = response_text.find("<thinking>")
                        thinking_end = response_text.find("</thinking>") + len("</thinking>")
                        response_text = response_text[:thinking_start] + response_text[thinking_end:]
                    
                except Exception as e:
                    console.print(f"[dim]Error in response generation: {e}[/dim]")
                    response_text = "Error generating response"
                
                # Evaluate response quality
                response_metrics = {
                    "word_count": len(response_text.split()),
                    "vocabulary_richness": len(set(response_text.split())) / max(1, len(response_text.split())),
                    "has_placeholders": "<UNK>" in response_text,
                    "repetition_score": sum(1 for word in response_text.split() if response_text.split().count(word) > 1) / max(1, len(response_text.split()))
                }
                
                # Store full output for debugging
                full_output = response_text
                
                # Display compact results (without thinking to keep output clean)
                console.print(f"[blue]Prompt:[/blue] {prompt[:50]}...")
                console.print(f"[green]Response:[/green] {response_text[:100]}...")
                console.print("─" * 80)  # Simple separator
                
                # Calculate word count and vocabulary stats
                words = response_text.split()
                word_count = len(words)
                unique_words = len(set(words))
                vocab_richness = unique_words / max(1, word_count)
                repetitions = sum(1 for word in words if words.count(word) > 1) / max(1, word_count)
                
                # Display metrics
                console.print(f"Words: {word_count} | Vocab: {vocab_richness:.2f} | Rep: {repetitions:.3f}")
                
                # Store result 
                test_result = {
                    "prompt": prompt,
                    "thinking": thinking_response,
                    "response": response_text,
                    "full_output": full_output,
                    "metrics": response_metrics
                }
                
                test_results.append(test_result)
            
            except Exception as e:
                console.print(f"[red]Error processing prompt: {e}[/red]")
                traceback.print_exc()
                # Add a placeholder for failed prompts
                test_results.append({
                    "prompt": prompt,
                    "thinking": "<thinking>\nLAYER 1 - PROMPT ANALYSIS:\nUnderstanding the query: I need to analyze " + prompt + "\n\nLAYER 2 - RESPONSE PLANNING:\nPlanning response structure: I'll provide a clear explanation.\n\nLAYER 3 - RESPONSE GENERATION:\nDrafting response based on my knowledge.\n</thinking>",
                    "response": f"ERROR: {str(e)}",
                    "full_output": f"ERROR: {str(e)}",
                    "metrics": {
                        "word_count": 0,
                        "vocabulary_richness": 0,
                        "has_placeholders": True,
                        "repetition_score": 0
                    }
                })
            
            finally:
                # Always update progress
                progress.update(test_task, advance=1)
    
    # Save results to JSON
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # Log results
    console.print(f"[dim]Test results saved to {test_file}[/dim]")
    
    # If saving checkpoint is requested
    if save_checkpoint:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{stage_name}_model_{test_iteration}.pt")
        
        # Save a checkpoint of the model at this point
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_results': test_results,
            'config': {
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'block_size': block_size,
                'dropout': dropout,
                'vocab_size': VOCAB_SIZE  # Use global VOCAB_SIZE instead of meta.vocab_size
            }
        }, checkpoint_path)
    
    # Restore original model training state
    if model_training_state:
        model.train()
    else:
        model.eval()
    
    return test_results

def evaluate_response_quality(response):
    """Simplified quality evaluation that uses less memory"""
    words = response.lower().split()
    unique_words = len(set(words)) if words else 0
    
    return {
        'length': len(response),
        'vocabulary_richness': unique_words / len(words) if words else 0
    }

def merge_models(stage_models):
    """Merge knowledge from multiple stage models using weighted averaging"""
    print("\n=== Merging Model Knowledge ===")
    
    # Create a new model instance for the merged result
    merged_model = GPTLanguageModel().to(device)
    
    # Initialize parameter accumulator
    param_sum = {}
    for name, param in merged_model.named_parameters():
        param_sum[name] = torch.zeros_like(param)
    
    # Weighted average of parameters
    weights = [0.2, 0.3, 0.5]  # Higher weight for later stages
    for model_dict, weight in zip(stage_models, weights):
        console.print(f"[cyan]Incorporating model knowledge with weight {weight}[/cyan]")
        for name, param in merged_model.named_parameters():
            param_sum[name] += weight * model_dict[name]
    
    # Set merged parameters
    with torch.no_grad():
        for name, param in merged_model.named_parameters():
            param.copy_(param_sum[name])
    
    return merged_model

def create_progress_layout():
    """Create a simplified layout for progress visualization with less animation"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=1),  # Reduced header size
        Layout(name="body", ratio=1)    # Body takes remaining space
    )
    return layout

def train_until_target(model, train_data, val_data, stage: TrainingStage):
    stage.start()
    
    # For better console output, limit width
    console.width = min(console.width, 100)
    
    # Enable LoRA if requested
    if USE_LORA:
        model.enable_lora()
    
    # Enable DeepSpeed if requested
    if USE_DEEPSPEED:
        model.enable_deepspeed()
    else:
        # Use standard optimizer if not using DeepSpeed
        try:
            from bitsandbytes.optim import Adam8bit
            optimizer = Adam8bit(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            console.print("[green]Using 8-bit Adam optimizer for memory efficiency[/green]")
        except ImportError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            console.print("[yellow]Consider installing bitsandbytes for 8-bit optimization[/yellow]")
    
    # Enable gradient scaling for stability using the updated API
    if hasattr(torch.amp, 'GradScaler'):  # PyTorch 2.0+ API
        scaler = torch.amp.GradScaler(enabled=use_mixed_precision)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    
    # Try to load latest checkpoint with more robust search
    start_step = 0
    checkpoint_loaded = False
    
    # Check for different checkpoint patterns
    checkpoint_locations = [
        os.path.join(stage.checkpoint_dir, "latest.pt"),  # First check for latest.pt
        sorted(glob.glob(os.path.join(stage.checkpoint_dir, 'checkpoint_step_*.pt')))[-1] if glob.glob(os.path.join(stage.checkpoint_dir, 'checkpoint_step_*.pt')) else None,  # Then check for step checkpoints
        f"checkpoint_step_{stage.name.lower()}_latest.pt",  # Try current directory with stage name
        sorted(glob.glob(f"checkpoint_step_*.pt"))[-1] if glob.glob(f"checkpoint_step_*.pt") else None  # Try any checkpoint in current directory
    ]
    
    # Try each location until we find a valid checkpoint
    for checkpoint_path in checkpoint_locations:
        if checkpoint_path is None:
            continue
            
        if os.path.exists(checkpoint_path):
            console.print(f"\n[yellow]Found checkpoint: {checkpoint_path}[/yellow]")
            try:
                # Use safer loading with potential device mapping
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_step = checkpoint['step']
                if 'best_loss' in checkpoint:
                    stage.best_loss = checkpoint['best_loss']
                console.print(f"[green]Successfully loaded checkpoint from step {start_step}[/green]")
                if 'config' in checkpoint:
                    console.print(f"[green]Checkpoint config: {checkpoint['config']}[/green]")
                checkpoint_loaded = True
                break
            except Exception as e:
                console.print(f"[red]Error loading checkpoint {checkpoint_path}: {str(e)}[/red]")
                traceback.print_exc()
                
    if not checkpoint_loaded:
        console.print("[yellow]No valid checkpoints found. Starting from scratch.[/yellow]")
        start_step = 0
    
    # Memory optimization function that's more aggressive but less verbose
    def optimize_memory():
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return "Cleared"
        return "No GPU"
        
    # Calculate test intervals
    test_intervals = [int(i * stage.max_steps / 3) for i in range(1, 3)]
    console.print(f"[green]Model will be tested at steps: {test_intervals}[/green]")
    
    # Initialize variables
    true_loss = 0.0
    running_avg_loss = 0.0
    loss_buffer = []
    avg_window = 100
    total_steps = 0
    recovery_mode = False
    target_reached = False
    oom_count = 0
    max_oom_retries = 5
    last_successful_iter = start_step

    # Set up TensorBoard if available
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(comment=f'_{stage.name}')
        use_tensorboard = True
    except ImportError:
        use_tensorboard = False
        console.print("[yellow]TensorBoard not available. Training metrics will not be logged.[/yellow]")
    
    # Main training loop with tqdm
    from tqdm import tqdm
    with tqdm(
        total=stage.max_steps, 
        initial=start_step,
        desc=f"[{stage.name}]",
        leave=True, 
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    ) as pbar:
        last_eval_time = time.time()
        last_eval_step = start_step
        
        for iter in range(start_step, stage.max_steps):
            # Training step with comprehensive error handling
            try:
                # Clear memory more aggressively during training
                if iter % 100 == 0 and iter > 0:
                    optimize_memory()
                
                # Get batch with error handling
                try:
                    xb, yb = get_batch('train')
                except Exception as e:
                    console.print(f"[red]Error getting batch: {str(e)}[/red]")
                    continue
                
                # Forward pass with aggressive memory optimization
                try:
                    with torch.amp.autocast('cuda', enabled=use_mixed_precision) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_mixed_precision):
                        # Process data normally
                        logits, loss = model(xb, yb)
                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        oom_count += 1
                        optimize_memory()
                        
                        if oom_count >= max_oom_retries:
                            # Enable recovery mode if we keep hitting OOM
                            recovery_mode = True
                            console.print(f"[red]Entering recovery mode after {oom_count} OOM errors[/red]")
                            # Restart from the last successful iteration
                            iter = max(last_successful_iter, iter - gradient_accumulation_steps)
                            oom_count = 0
                            continue
                        else:
                            console.print(f"[red]OOM in forward pass ({oom_count}/{max_oom_retries}), retrying...[/red]")
                            continue
                    else:
                        console.print(f"[red]Forward pass error: {str(e)}[/red]")
                        continue
                
                # Update loss tracking & postfix display
                if iter % 10 == 0:
                    true_loss = loss.item() * gradient_accumulation_steps
                    loss_buffer.append(true_loss)
                    
                    if len(loss_buffer) > avg_window:
                        loss_buffer.pop(0)
                    
                    current_avg_loss = sum(loss_buffer) / len(loss_buffer)
                    total_steps += 1
                    running_avg_loss = (running_avg_loss * (total_steps - 1) + true_loss) / total_steps
                    
                    # Log to file
                    log_message = f"Step {iter}: Loss: {true_loss:.4f} | Avg: {running_avg_loss:.4f}"
                    log_file.write(log_message + "\n")
                    log_file.flush()
                    
                    # Update tqdm
                    pbar.set_postfix({"loss": f"{true_loss:.4f}", "avg": f"{running_avg_loss:.4f}"})
                    
                    # Log to TensorBoard if available
                    if use_tensorboard:
                        writer.add_scalar('Loss/train', true_loss, iter)
                        writer.add_scalar('Loss/avg', running_avg_loss, iter)
                
                # Backward pass
                try:
                    scaler.scale(loss).backward()
                except Exception as e:
                    console.print(f"[red]Backward pass error: {str(e)}[/red]")
                    optimize_memory()
                    continue
                
                # Update last successful iteration
                last_successful_iter = iter
                
                # Optimizer step with checkpoint saving
                if ((iter + 1) % gradient_accumulation_steps == 0) or (iter + 1 == stage.max_steps):
                    try:
                        # Gradient clipping for stability
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        
                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    
                        # Save checkpoint occasionally
                        if (iter % stage.checkpoint_interval == 0 and iter > start_step + 100):
                            # Save checkpoint
                            checkpoint_success = stage.save_checkpoint(model, optimizer, running_avg_loss, iter)
                            
                            if checkpoint_success:
                                log_file.write(f"Checkpoint saved at step {iter}\n")
                                log_file.flush()
                                pbar.write(f"[green]✓ Checkpoint saved at step {iter}[/green]")
                            else:
                                log_file.write(f"WARNING: Failed to save checkpoint at step {iter}\n")
                                log_file.flush()
                                pbar.write(f"[red]✗ Failed to save checkpoint at step {iter}[/red]")
                    except RuntimeError as e:
                        console.print(f"[red]Optimizer step error: {str(e)}[/red]")
                        optimize_memory()
                        continue
                
                # Periodic evaluation
                current_time = time.time()
                if (iter % eval_interval == 0 and iter > 0) or (current_time - last_eval_time > 600):
                    try:
                        # Calculate speed stats
                        steps_done = iter - last_eval_step
                        time_elapsed = current_time - last_eval_time
                        steps_per_second = steps_done / time_elapsed if time_elapsed > 0 else 0
                        
                        # Optimize memory before evaluation
                        optimize_memory()
                        
                        # Run evaluation
                        losses = estimate_loss()
                        
                        # Update tqdm with evaluation metrics
                        pbar.write(f"[cyan]Eval: train={losses['train']:.4f}, val={losses['val']:.4f} ({steps_per_second:.1f} steps/s)[/cyan]")
                        
                        # Log to TensorBoard
                        if use_tensorboard:
                            writer.add_scalar('Loss/val', losses['val'], iter)
                            writer.add_scalar('Speed/steps_per_second', steps_per_second, iter)
                        
                        # Update timers
                        last_eval_time = current_time
                        last_eval_step = iter
                        
                        # Log evaluation results
                        log_message = f"Eval Step {iter}: Train: {losses['train']:.4f} | Val: {losses['val']:.4f}"
                        log_file.write(log_message + "\n")
                        log_file.flush()
                        
                        # Track best loss
                        if losses['val'] < stage.best_loss:
                            stage.best_loss = losses['val']
                            # Check if we've reached the target loss
                            if stage.best_loss <= stage.target_loss:
                                target_reached = True
                                pbar.write(f"[bold green]✓ Target loss of {stage.target_loss:.4f} reached with val loss {losses['val']:.4f}![/bold green]")
                        
                            # Save best model
                            try:
                                optimize_memory()
                                stage.save_checkpoint(model, optimizer, losses['val'], iter, is_best=True)
                                pbar.write(f"[green]✓ Best model saved with val loss {losses['val']:.4f}[/green]")
                            except Exception as e:
                                pbar.write(f"[red]✗ Error saving best model: {str(e)}[/red]")
                    except Exception as e:
                        pbar.write(f"[red]✗ Evaluation error: {str(e)}[/red]")
                        continue
                
                # Update progress bar
                pbar.update(1)
                
                console.print(
                    f"[bold cyan]Epoch {epoch+1}/{epochs}[/bold cyan] • "
                    f"[bold]Step {epoch_step}/{len(data_loader)}[/bold] • "
                    f"[green]Loss:[/green] {loss.item():.4f} • "
                    f"[yellow]LR:[/yellow] {lr:.2e} • "
                    f"[magenta]GPU:[/magenta] {format_gpu()} • "
                    f"[blue]ETA:[/blue] {format_eta(pbar.format_dict.get('remaining', 0))}"
                )


            except Exception as e:
                # Global exception handler
                pbar.write(f"[red]✗ Unhandled error: {str(e)}[/red]")
                traceback.print_exc()
                optimize_memory()
                continue
                
    # Close TensorBoard writer if used
    if use_tensorboard:
        writer.close()
    
    # Mark stage as complete
    stage.steps_completed = iter + 1
    stage.complete(success=target_reached, final_loss=running_avg_loss)
    
    # Final status display
    console.print("\n[bold green]" + "=" * 50 + "[/bold green]")
    console.print(f"[bold green]🎉 {stage.name} Training Complete! 🎉[/bold green]")
    console.print(f"[bold cyan]Steps completed: {stage.steps_completed}/{stage.max_steps}[/bold cyan]")
    console.print(f"[bold cyan]Best loss achieved: {stage.best_loss:.4f} (Target: {stage.target_loss})[/bold cyan]")
    console.print(f"[bold cyan]Final loss: {running_avg_loss:.4f}[/bold cyan]")
    console.print(f"[bold cyan]Target reached: {'✓ Yes' if target_reached else '✗ No'}[/bold cyan]")
    console.print("[bold green]" + "=" * 50 + "[/bold green]")
    
    return model.state_dict()

# Modify TrainingStage to support saving best models
def modify_save_checkpoint(self, model, optimizer, loss, step, is_best=False):
    try:
        # Create a dictionary with all the data we want to save
        checkpoint = {
            'step': step,
                            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'stage_name': self.name,
            'best_loss': self.best_loss,
            'completed': self.completed,
            'config': {
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'vocab_size': vocab_size,
                'block_size': block_size
            }
        }
        
        # Save checkpoint based on type
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best.pt")
        else:
            # Normal checkpoint
            filename = f'checkpoint_step_{step}.pt'
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint with robust error handling
        try:
            # First try normal saving
            torch.save(checkpoint, checkpoint_path)
            
            # Also save a "latest.pt" for easy loading if this is not the best model
            if not is_best:
                latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
                torch.save(checkpoint, latest_path)
            
            return True
            
        except Exception as e:
            # If we get here, try saving to the current directory as a last resort
            print(f"Error saving checkpoint to {checkpoint_path}: {e}")
            fallback_path = filename if not is_best else "best.pt"
            try:
                torch.save(checkpoint, fallback_path)
                print(f"Checkpoint saved to fallback location: {fallback_path}")
                return True
            except Exception as e2:
                print(f"FATAL ERROR: Could not save checkpoint: {e2}")
                return False
        
        # Keep only the last 5 checkpoints to save disk space (skip if we're in fallback mode)
        if self.checkpoint_dir != "." and not is_best:
            checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_step_*.pt')))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    try:
                        os.remove(old_checkpoint)
                    except Exception as e:
                        print(f"Warning: Could not delete old checkpoint {old_checkpoint}: {e}")
        return True
    except Exception as e:
        print(f"Unexpected error in save_checkpoint: {e}")
        traceback.print_exc()
        return False

# Patch the TrainingStage class with our modified save_checkpoint method
TrainingStage.save_checkpoint = modify_save_checkpoint

# Modify the stages definition to include three phases
stages = [
    TrainingStage("IdentityOnly", max_steps=8000, target_loss=0.5),  # Phase 1: Identity training
    TrainingStage("WikipediaOnly", max_steps=8000, target_loss=0.5),  # Phase 2: Wikipedia training
    TrainingStage("Combined", max_steps=12000, target_loss=0.5)       # Phase 3: Combined with 2:1 ratio
]

# Add BPE tokenizer implementation for improved training
def create_bpe_tokenizer(text, vocab_size=32000):
    """Create a BPE tokenizer from the training text with proper special token handling"""
    if not USE_BETTER_TOKENIZER:
        console.print("[red]Tokenizers package not available. Cannot create BPE tokenizer.[/red]")
        return None
        
    try:
        console.print(f"[cyan]Training BPE tokenizer with vocab size {vocab_size}...[/cyan]")
        
        # Initialize a BPE tokenizer with special tokens
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Configure training with special tokens
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2  # Lower frequency threshold
        )
        
        # Prepare training data - split into manageable chunks
        chunks = [text[i:i+1000000] for i in range(0, len(text), 1000000)]
        
        # Train the tokenizer
        console.print("[yellow]Training BPE tokenizer...[/yellow]")
        tokenizer.train_from_iterator(chunks, trainer)
        
        # Save the tokenizer
        tokenizer_path = os.path.join(CHECKPOINT_DIR, "bpe_tokenizer.json")
        tokenizer.save(tokenizer_path)
        console.print(f"[green]BPE tokenizer trained and saved to {tokenizer_path}[/green]")
        
        # Verify special tokens are properly added
        vocab = tokenizer.get_vocab()
        for token in SPECIAL_TOKENS:
            if token not in vocab:
                console.print(f"[red]Warning: Special token {token} not in vocabulary![/red]")
        
        return tokenizer
    except Exception as e:
        console.print(f"[red]Error creating BPE tokenizer: {str(e)}[/red]")
        traceback.print_exc()
        return None

def create_encoder_decoder_from_bpe(tokenizer):
    """Create encoder and decoder functions from BPE tokenizer with special token handling"""
    
    # Get vocabulary with special tokens
    vocab = tokenizer.get_vocab()
    ids_to_tokens = {v: k for k, v in vocab.items()}
    
    def encode(s):
        """Encode text with special token handling"""
        # Add BOS and EOS tokens
        encoded = tokenizer.encode(s)
        return [vocab[BOS_TOKEN]] + encoded.ids + [vocab[EOS_TOKEN]]
    
    def decode(ids):
        """Decode with special token handling"""
        # Convert tensor to list if needed
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Filter out special tokens except thinking markers
        filtered_ids = []
        for id in ids:
            token = ids_to_tokens.get(id, UNK_TOKEN)
            if token not in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] or token in [THINKING_START, THINKING_END]:
                filtered_ids.append(id)
        
        # Decode filtered ids
        return tokenizer.decode(filtered_ids)
    
    return encode, decode, len(vocab)

# Add function to prepare data with improved tokenization
def prepare_data_with_bpe(text, encode_func):
    """Prepare training data using BPE tokenization"""
    console.print("[yellow]Encoding training data with BPE tokenizer...[/yellow]")
    
    # Encode the entire text
    data = torch.tensor(encode_func(text), dtype=torch.long)
    
    # Use all data for training (no train/val split)
    train_data = data
    val_data = data[:min(len(data), 4775344)]  # Use a small subset for validation
    
    console.print(f"[green]Training data size: {len(train_data)} tokens[/green]")
    console.print(f"[green]Validation data size: {len(val_data)} tokens[/green]")
    
    # Keep data on CPU until needed to save GPU memory
    return train_data, val_data

# Modify the train_and_merge function to load checkpoints
def train_and_merge(current_model):
    if FURTHER_TRAINING:
        # Load the final model for further training with improved tokenization
        console.print("\n[bold yellow]=== Preparing for Enhanced Training ===[/bold yellow]")
        
        # Phase 1: Identity-only training
        console.print("[cyan]Creating identity-focused training data...[/cyan]")
        identity_text = create_training_data(repeat_identity=100)  # Identity-focused data
        
        # Phase 2: Prepare Wikipedia-only training data
        console.print("[yellow]Loading WikiText-2 dataset for Wikipedia-only training...[/yellow]")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        wiki_text = "\n".join(dataset['train']['text'])
        
        # Phase 3: Prepare combined dataset with 2:1 ratio for identity:wiki
        console.print("[green]Creating combined training data with 2:1 ratio (identity:wiki)...[/green]")
        combined_text = identity_text * 2 + "\n\n" + wiki_text
        
        # Create BPE tokenizer with reduced vocabulary for better tokens
        bpe_tokenizer = create_bpe_tokenizer(combined_text, vocab_size=8000)
        
        if bpe_tokenizer:
            # Get encoder/decoder functions and vocab size
            encode_func, decode_func, bpe_vocab_size = create_encoder_decoder_from_bpe(bpe_tokenizer)
            
            # Update global encoder/decoder and vocab size
            global encode, decode, vocab_size, train_data, val_data
            encode = encode_func
            decode = decode_func
            vocab_size = bpe_vocab_size
            console.print(f"[green]Using improved BPE tokenization with {vocab_size} tokens[/green]")
            
            # Update model's token embedding table to match new vocabulary size
            original_embedding = current_model.token_embedding_table
            current_model.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            nn.init.normal_(current_model.token_embedding_table.weight, mean=0.0, std=0.02)
            
            # Also update the output projection layer
            original_lm_head = current_model.lm_head
            current_model.lm_head = nn.Linear(n_embd, vocab_size)
            
            # Make sure the entire model is moved to the device after replacing layers
            current_model = current_model.to(device)
            
            console.print("[blue]Token embedding and output layers resized to match new vocabulary[/blue]")
            
            # Phase 1: Identity-only training
            console.print("\n[bold magenta]=== Starting Phase 1: Identity-Only Training ===[/bold magenta]")
            
            try:
                # Create identity data with extra safety measures
                identity_text_tokens = encode_func(identity_text)
                
                # Make sure we have enough tokens
                min_required_tokens = batch_size * block_size * 2
                if len(identity_text_tokens) < min_required_tokens:
                    console.print(f"[yellow]Identity data too small ({len(identity_text_tokens)} tokens). Repeating to ensure minimum {min_required_tokens} tokens.[/yellow]")
                    repeat_factor = (min_required_tokens // len(identity_text_tokens)) + 1
                    identity_text_tokens = identity_text_tokens * repeat_factor
                
                # Convert to tensor
                identity_data = torch.tensor(identity_text_tokens, dtype=torch.long)
                console.print(f"[green]Identity data size: {len(identity_data)} tokens[/green]")
                
                # Set train and validation data
                train_data = identity_data
                val_data = identity_data  # Use same data for validation
                
                # Train on identity data
                identity_stage = TrainingStage("IdentityOnly", max_steps=8000, target_loss=0.5)
                identity_dict = train_until_target(current_model, train_data, val_data, identity_stage)
                
                # Phase 2: Wikipedia-only training
                console.print("\n[bold magenta]=== Starting Phase 2: Wikipedia-Only Training ===[/bold magenta]")
                wiki_tokens = encode_func(wiki_text)
                
                # Make sure wiki data is large enough
                if len(wiki_tokens) < min_required_tokens:
                    console.print(f"[yellow]Wiki data too small ({len(wiki_tokens)} tokens). Repeating to ensure minimum {min_required_tokens} tokens.[/yellow]")
                    repeat_factor = (min_required_tokens // len(wiki_tokens)) + 1
                    wiki_tokens = wiki_tokens * repeat_factor
                
                # Convert to tensor
                wiki_data = torch.tensor(wiki_tokens, dtype=torch.long)
                console.print(f"[green]Wiki data size: {len(wiki_data)} tokens[/green]")
                
                # Set train and validation data
                train_data = wiki_data
                val_data = wiki_data  # Use same data for validation
                
                # Train on wiki data
                wiki_stage = TrainingStage("WikipediaOnly", max_steps=8000, target_loss=0.5)
                wiki_dict = train_until_target(current_model, train_data, val_data, wiki_stage)
                
                # Phase 3: Combined training with 2:1 ratio
                console.print("\n[bold magenta]=== Starting Phase 3: Combined Training (2:1 Ratio) ===[/bold magenta]")
                
                # Use a safer approach to create combined data
                try:
                    # First, make sure both datasets are properly prepared
                    console.print("[yellow]Preparing combined data with identity and Wikipedia...[/yellow]")
                    
                    # Create fresh identity and wiki data to ensure consistent tokenization
                    identity_text_repeat = create_training_data(repeat_identity=100) * 2  # Double for 2:1 ratio
                    identity_tokens = encode_func(identity_text_repeat)
                    
                    wiki_tokens = encode_func(wiki_text)
                    
                    # Combine tokens directly instead of text
                    combined_tokens = identity_tokens + wiki_tokens
                    console.print(f"[green]Combined token count: {len(combined_tokens)} tokens[/green]")
                    
                    # Convert to tensor
                    combined_data = torch.tensor(combined_tokens, dtype=torch.long)
                    
                    # Make sure combined data is large enough
                    if len(combined_data) < min_required_tokens:
                        console.print(f"[yellow]Combined data too small ({len(combined_data)} tokens). Repeating to ensure minimum {min_required_tokens} tokens.[/yellow]")
                        repeat_factor = (min_required_tokens // len(combined_data)) + 1
                        combined_data = combined_data.repeat(repeat_factor)
                        console.print(f"[green]Combined data expanded to {len(combined_data)} tokens[/green]")
                    
                    train_data = combined_data
                    val_data = combined_data[:min(len(combined_data), 10000)]
                except Exception as e:
                    console.print(f"[red]Error preparing combined data: {e}. Falling back to safer approach.[/red]")
                    # Fallback to a simpler approach if the above fails
                    combined_tokens = encode_func(identity_text + "\n\n" + wiki_text)
                    combined_data = torch.tensor(combined_tokens, dtype=torch.long)
                    train_data = combined_data
                    val_data = combined_data[:min(len(combined_data), 10000)]
                
                combined_stage = TrainingStage("Combined", max_steps=12000, target_loss=0.5)
                combined_dict = train_until_target(current_model, train_data, val_data, combined_stage)
                
            except Exception as e:
                console.print(f"[bold red]Error in training phases: {str(e)}[/bold red]")
                traceback.print_exc()
                console.print("[yellow]Attempting to continue with fallback approach...[/yellow]")
                
                # Fallback to a simpler approach
                enhanced_stage = TrainingStage("EnhancedFallback", max_steps=12000, target_loss=0.5)
                combined_data = torch.tensor(encode_func(identity_text + wiki_text), dtype=torch.long)
                train_data = combined_data
                val_data = combined_data[:min(len(combined_data), 10000)]
                stage_dict = train_until_target(current_model, train_data, val_data, enhanced_stage)
            
            # Save the final enhanced model
            final_save_path = os.path.join(CHECKPOINT_DIR, "final_model_enhanced.pt")
            torch.save({
                'model_state_dict': current_model.state_dict(),
                'model_config': {
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'n_embd': n_embd,
                    'block_size': block_size,
                    'vocab_size': vocab_size
                },
                'tokenizer_path': os.path.join(CHECKPOINT_DIR, "bpe_tokenizer.json"),
                'training_info': {
                    'stages_completed': ["IdentityOnly", "WikipediaOnly", "Combined"],
                    'final_loss': combined_stage.best_loss,
                    'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, final_save_path)
            
            console.print(f"[bold green]Final Enhanced model saved to: {final_save_path}[/bold green]")
        else:
            console.print("[red]Failed to create BPE tokenizer. Continuing with existing tokenization.[/red]")
            # Fallback to a simpler enhanced stage
            enhanced_stage = TrainingStage("EnhancedFallback", max_steps=12000, target_loss=0.5)
            stage_dict = train_until_target(current_model, train_data, val_data, enhanced_stage)
    else:
        # Original loading logic for standard training
        console.print("\n[bold yellow]=== Loading Checkpoints from All Stages ===[/bold yellow]")
        
        # Define checkpoint paths
        identity_path = "D:/ttm/model/3bmodel/t/checkpoints/identity/latest.pt"
        metadata_path = "D:/ttm/model/3bmodel/t/checkpoints/metadata/latest.pt"
        combined_path = "D:/ttm/model/3bmodel/t/checkpoints/combined/latest.pt"
        
        # Load the Combined stage checkpoint into the model
        try:
            console.print(f"[yellow]Loading Combined stage checkpoint from: {combined_path}[/yellow]")
            combined_checkpoint = torch.load(combined_path, map_location=device)
            current_model.load_state_dict(combined_checkpoint['model_state_dict'])
            console.print(f"[green]Successfully loaded Combined checkpoint[/green]")
        except Exception as e:
            console.print(f"[red]Error loading Combined checkpoint: {str(e)}[/red]")
            traceback.print_exc()
        
        # Continue training only the Combined stage
        console.print(f"\n[bold cyan]=== Continuing Combined Stage Training ===[/bold cyan]")
        stage_dict = train_until_target(current_model, train_data, val_data, stages[0])
        
        # Save the final model without merging
        final_save_path = os.path.join(CHECKPOINT_DIR, "final_model_combined.pt")
        torch.save({
            'model_state_dict': current_model.state_dict(),
            'model_config': {
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'block_size': block_size,
                'vocab_size': vocab_size
            },
            'training_info': {
                'stages_completed': ["Combined"],
                'final_loss': stages[0].best_loss,
                'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }, final_save_path)
        
        console.print(f"[bold green]Final Combined model saved to: {final_save_path}[/bold green]")
    
    return current_model

def create_training_data(repeat_identity=100):
    """Create enhanced training data with emphasis on quality responses"""
    console.print("\n[bold cyan]Preparing enhanced training data...[/bold cyan]")
    
    # Load metadata
    console.print("[yellow]Loading metadata...[/yellow]")
    with open('metadata.txt', 'r', encoding='utf-8') as f:
        metadata_content = f.read()
        metadata_namespace = globals().copy()
        exec(metadata_content, metadata_namespace)
        conversation_data = metadata_namespace['simple_conversation_data']
        technical_data = metadata_namespace['technical_details_data']
        mixed_data = metadata_namespace['mixed_context_data']
    
    # Format conversations with emphasis on detailed responses
    def format_conversation(conv):
        # Add context markers for better response quality
        return f"Context: Professional and technical discussion\nQ: {conv['question']}\nA: {conv['answer']}\nQuality: Ensure detailed, technical, and well-structured response\n\n"

    # Create identity section with enhanced detail
    console.print("\n[cyan]Creating detailed identity section...[/cyan]")
    identity_text = f"""Model Identity and Capabilities:
Name: {MODEL_NAME_CUSTOM}
Created by: {CREATOR_NAME}
Organization: {COMPANY_NAME}
Role: {CREATOR_ROLE}
Version: {VERSION}

Technical Architecture:
- Layer Count: {NUM_LAYERS} transformer layers
- Attention Heads: {NUM_HEADS} heads per layer
- Embedding Dimension: {HIDDEN_SIZE}
- Context Window: {block_size} tokens
- Architecture Type: Advanced Transformer with enhanced capabilities

Advanced Features:
- Position Encoding: {"Rotary (RoPE)" if USE_ROPE else "Standard"} embeddings
- Attention Mechanism: {"Flash Attention" if USE_FLASH_ATTENTION else "Standard Attention"}
- Precision: {"Mixed Precision" if USE_MIXED_PRECISION else "Full Precision"} training
- Memory Optimization: {"Enabled" if USE_GRADIENT_CHECKPOINTING else "Disabled"} gradient checkpointing

Training Configuration:
- Optimizer: {OPTIMIZER} with learning rate {LEARNING_RATE}
- Schedule: {SCHEDULER}
- Batch Processing: {BATCH_SIZE} samples per batch
- Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps

Quality Emphasis:
- Focus on detailed, technical responses
- Maintain professional communication style
- Provide comprehensive explanations
- Include relevant technical details
- Structure responses clearly and logically
\n\n"""

    # Combine all data with emphasis on quality
    console.print(f"[cyan]Repeating enhanced identity section {repeat_identity} times...[/cyan]")
    training_text = identity_text * repeat_identity
    
    # Add all conversations with quality emphasis
    console.print("[cyan]Adding enhanced conversation data...[/cyan]")
    all_data = conversation_data + technical_data + mixed_data
    for i, conv in enumerate(tqdm(all_data, desc="Formatting conversations")):
        training_text += format_conversation(conv)
    
    console.print(f"[green]Final enhanced training text size: {len(training_text)} characters[/green]")
    return training_text

def create_combined_training_data():
    """Create combined training data for the final stage"""
    console.print("\n[bold cyan]Preparing combined training data...[/bold cyan]")
    
    # Get base training data
    base_data = create_training_data(repeat_identity=50)
    
    # Comment out WikiText reference since it's not properly defined
    # console.print("[yellow]Loading WikiText-2 dataset...[/yellow]")
    # wiki_text = "\n".join(dataset['train']['text'])
    
    # Use only base data instead of attempting to combine with wiki_text
    combined_text = base_data # + "\n\n" + wiki_text
    
    console.print(f"[green]Combined training text size: {len(combined_text)} characters[/green]")
    return combined_text

# Create model and prepare for training
model = GPTLanguageModel()
m = model.to(device)
model_size = sum(p.numel() for p in m.parameters())/1e6

# Generate detailed model statistics
def display_model_statistics(model):
    console = Console()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Count parameters by layer type
    layer_counts = {}
    layer_params = {}
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        if layer_type not in ['Sequential', 'ModuleList', 'GPTLanguageModel', 'Block']:
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
                layer_params[layer_type] = 0
            layer_counts[layer_type] += 1
            layer_params[layer_type] += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Sort by parameter count (descending)
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    
    # Estimate memory usage
    param_mem = total_params * 4 / (1024**2)  # 4 bytes per param
    activation_size = batch_size * n_embd * block_size * 4 * n_layer / (1024**2)  # Forward pass activations
    optimizer_mem = total_params * 8 / (1024**2)  # Adam uses ~8 bytes per param
    total_mem = param_mem + activation_size + optimizer_mem
    
    # Performance estimates
    flops_per_token = 2 * total_params  # Very rough estimate
    tokens_per_sec_gen = 5000000 / total_params * 1000000  # Scaled based on parameter count
    tokens_per_sec_train = tokens_per_sec_gen / 3  # Training is ~3x slower than inference
    
    # Model comparisons
    gpt2_params = 124000000
    gpt3_params = 175000000000
    llama2_params = 7000000000
    
    # Create rich tables for displaying stats
    console.print("\n" + "=" * 80)
    console.print("🔍 [bold cyan]TURBOTALK MODEL STATISTICS[/bold cyan] 🔍", justify="center")
    console.print("=" * 80)
    
    # Parameter counts table
    param_table = Table(title="📊 PARAMETER COUNTS", box=box.DOUBLE_EDGE)
    param_table.add_column("Parameter Type", style="cyan")
    param_table.add_column("Count", justify="right", style="green")
    param_table.add_column("Percent", justify="right", style="yellow")
    
    param_table.add_row("Trainable", f"{trainable_params:,}", f"{100 * trainable_params / total_params:.2f}%")
    param_table.add_row("Non-trainable", f"{non_trainable_params:,}", f"{100 * non_trainable_params / total_params:.2f}%")
    param_table.add_row("Total", f"{total_params:,}", "100.00%")
    
    console.print(param_table)
    
    # Parameter distribution table
    dist_table = Table(title="📈 PARAMETER DISTRIBUTION", box=box.DOUBLE_EDGE)
    dist_table.add_column("Component Type", style="cyan")
    dist_table.add_column("Count", justify="right", style="green")
    dist_table.add_column("Parameters", justify="right", style="green")
    dist_table.add_column("% of Total", justify="right", style="yellow")
    
    for layer_type, param_count in sorted_layers:
        dist_table.add_row(
            layer_type, 
            f"{layer_counts[layer_type]}", 
            f"{param_count:,}", 
            f"{100 * param_count / total_params:.2f}%"
        )
    
    console.print(dist_table)
    
    # Memory usage table
    mem_table = Table(title="💾 ESTIMATED MEMORY USAGE", box=box.DOUBLE_EDGE)
    mem_table.add_column("Memory Type", style="cyan")
    mem_table.add_column("Estimated Usage (MB)", justify="right", style="green")
    mem_table.add_column("Notes", style="dim")
    
    mem_table.add_row("Parameters", f"{param_mem:.2f}", "4 bytes per parameter")
    mem_table.add_row("Activations", f"{activation_size:.2f}", "Forward pass (estimated)")
    mem_table.add_row("Optimizer States", f"{optimizer_mem:.2f}", "Adam-like optimizer")
    mem_table.add_row("Total", f"{total_mem:.2f}", "Sum of above")
    
    console.print(mem_table)
    
    # Performance estimates
    perf_table = Table(title="⚡ PERFORMANCE ESTIMATES", box=box.DOUBLE_EDGE)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Estimate", justify="right", style="green")
    perf_table.add_column("Notes", style="dim")
    
    perf_table.add_row("FLOPs per token", f"{flops_per_token:.2e}", "Forward pass")
    perf_table.add_row("Generation tokens/sec", f"{tokens_per_sec_gen:.2f}", "On A100 GPU (estimated)")
    perf_table.add_row("Training tokens/sec", f"{tokens_per_sec_train:.2f}", "Forward + backward (estimated)")
    
    console.print(perf_table)
    
    # Model comparisons
    comp_table = Table(title="📏 MODEL COMPARISONS", box=box.DOUBLE_EDGE)
    comp_table.add_column("Model", style="cyan")
    comp_table.add_column("Parameters", justify="right", style="green")
    comp_table.add_column("Ratio", justify="right", style="yellow")
    
    comp_table.add_row("This Model", f"{total_params:,}", "1.0x")
    comp_table.add_row("GPT-2 (124M)", f"{gpt2_params:,}", f"{total_params/gpt2_params:.2f}x")
    comp_table.add_row("GPT-3 (175B)", f"{gpt3_params:,}", f"{total_params/gpt3_params:.6f}x")
    comp_table.add_row("Llama 2 (7B)", f"{llama2_params:,}", f"{total_params/llama2_params:.5f}x")
    
    console.print(comp_table)
    
    console.print("=" * 80)
    return total_params

# Move the function definition right before the main block

def create_quality_training_data():
    """
    Create high-quality training data for supervised fine-tuning with bot identity information.
    Generates 50,000 examples with identity information and progress tracking.
    """
    quality_data = []
    
    # Define target for training examples
    TOTAL_EXAMPLES = 50000  # 50,000 examples as requested
    
    # Bot identity information
    bot_identity = {
        "name": "TurboTalk AI",
        "company": "Rango Productions",
        "creator": "Rushi Bhavinkumar Soni",
        "creator_gender": "male",
        "creator_role": "CEO and founder of Rango Productions",
        "country": "India",
        "nationality": "Indian"
    }
    
    # Content elements for templates
    topics = [
        # AI and Machine Learning
        "machine learning", "deep learning", "natural language processing", "computer vision", "reinforcement learning",
        "artificial intelligence", "neural networks", "transformer models", "generative AI", "robotics",
        "large language models", "deep neural networks", "GPT models", "BERT models", "convolutional neural networks",
        "recurrent neural networks", "attention mechanisms", "transfer learning", "unsupervised learning", "supervised learning",
        "semi-supervised learning", "self-supervised learning", "meta learning", "federated learning", "explainable AI",
        
        # Computing and Technology
        "quantum computing", "blockchain technology", "cloud computing", "edge computing", "Internet of Things",
        "cybersecurity", "data science", "big data", "data privacy", "ethical AI",
        "serverless architecture", "microservices", "containerization", "virtual reality", "augmented reality",
        "mixed reality", "5G technology", "6G research", "fiber optics", "quantum internet",
        "distributed systems", "parallel computing", "high-performance computing", "neuromorphic computing", "quantum supremacy",
        
        # Science and Research
        "renewable energy", "nuclear fusion", "gene therapy", "nanotechnology", "space exploration",
        "climate science", "quantum physics", "materials science", "bioinformatics", "neuroscience",
        "genomics", "immunotherapy", "stem cell research", "precision medicine", "CRISPR technology",
        "synthetic biology", "astrophysics", "dark matter", "particle physics", "gravitational waves",
        "exoplanet research", "cosmology", "string theory", "medical imaging", "radiomics",
        
        # Business and Economics
        "digital transformation", "fintech", "cryptocurrency", "supply chain management", "e-commerce",
        "remote work", "digital marketing", "blockchain finance", "smart contracts", "decentralized finance",
        "platform economics", "sharing economy", "circular economy", "sustainable business", "ESG investing",
        "venture capital", "startup ecosystems", "business intelligence", "predictive analytics", "customer experience",
        "product management", "agile methodology", "design thinking", "business model innovation", "market disruption",
        
        # Social and Humanities
        "social media", "digital humanities", "information ethics", "online education", "digital literacy",
        "misinformation", "digital divide", "algorithmic bias", "content moderation", "digital citizenship",
        "smart cities", "urban planning", "sustainable development", "cultural heritage preservation", "digital art",
        "computational linguistics", "digital archaeology", "human-computer interaction", "user experience design", "accessibility",
        "technology ethics", "privacy rights", "data sovereignty", "internet governance", "technology policy"
    ]
    
    descriptions = [
        "field of study", "technology", "concept", "methodology", "framework", 
        "paradigm", "approach", "discipline", "system", "technique",
        "research area", "emerging field", "interdisciplinary domain", "scientific breakthrough", "technological innovation",
        "strategic initiative", "analytical method", "computational approach", "theoretical framework", "practical application",
        "transformative technology", "disruptive innovation", "foundational discipline", "specialized area", "cutting-edge development"
    ]
    
    processes = [
        "processing data in parallel", "analyzing patterns in large datasets", "transforming input information",
        "optimizing for specific outcomes", "encoding and decoding information", "leveraging statistical principles",
        "recognizing complex patterns", "extracting meaningful insights", "modeling sophisticated relationships", "predicting future trends",
        "simulating real-world scenarios", "automating complex workflows", "scaling computational resources", "implementing feedback loops",
        "performing distributed computing", "deploying edge computation", "orchestrating cloud services", "enabling secure transactions",
        "harnessing collective intelligence", "facilitating peer-to-peer communication", "generating synthetic content",
        "validating data integrity", "ensuring privacy preservation", "mitigating algorithmic bias", "adapting to environmental changes"
    ]
    
    benefits = [
        "improved efficiency", "greater accuracy", "enhanced user experiences", "reduced costs", 
        "increased accessibility", "better decision-making", "accelerated innovation", "optimized resource utilization",
        "reduced environmental impact", "improved safety and security", "enhanced collaboration capabilities", 
        "personalized experiences", "faster processing speeds", "improved productivity", "real-time analytics",
        "predictive capabilities", "seamless integration", "regulatory compliance", "competitive advantage",
        "increased scalability", "enhanced reliability", "improved fault tolerance", "minimal downtime",
        "cross-platform compatibility", "enhanced data visualization", "simplified complex processes"
    ]
    
    reasons = [
        "it solves previously intractable problems", "it provides unique insights", "it enables automation of complex tasks",
        "it improves human capabilities", "it creates new opportunities", "it addresses critical limitations",
        "it revolutionizes traditional approaches", "it bridges significant knowledge gaps", "it enables unprecedented scale",
        "it optimizes resource allocation", "it enhances human-machine collaboration", "it facilitates real-time responses",
        "it enables predictive capabilities", "it ensures greater precision", "it adapts to changing conditions",
        "it minimizes human error", "it unlocks previously inaccessible insights", "it automates tedious processes",
        "it improves decision-making accuracy", "it enables continuous improvement", "it facilitates knowledge sharing",
        "it addresses growing complexity", "it ensures reliability at scale", "it provides economic advantages",
        "it creates sustainable solutions", "it transforms user experiences", "it enhances trust and transparency"
    ]
    
    domains = [
        "healthcare", "finance", "education", "transportation", "communication", "manufacturing",
        "agriculture", "energy", "retail", "government", "defense", "entertainment", "sports",
        "scientific research", "environmental monitoring", "disaster response", "public safety",
        "urban planning", "construction", "logistics", "telecommunications", "pharmaceuticals",
        "biotechnology", "legal services", "creative industries", "human resources", "customer service",
        "supply chain management", "space exploration", "marine conservation", "humanitarian aid",
        "natural resource management", "wildlife conservation", "public health", "international development"
    ]
    
    impacts = [
        "reducing costs", "saving time", "improving accuracy", "enhancing user experiences", "enabling new capabilities",
        "facilitating better decision-making", "transforming business models", "creating new job opportunities",
        "bridging accessibility gaps", "reducing environmental footprints", "enhancing safety protocols",
        "improving quality of life", "advancing scientific knowledge", "democratizing access to information",
        "enabling personalized services", "improving resource efficiency", "enhancing security measures",
        "creating sustainable systems", "facilitating remote collaboration", "enabling real-time monitoring",
        "improving prediction accuracy", "enhancing diagnostic capabilities", "streamlining complex workflows",
        "reducing human error", "enabling autonomous operation", "improving compliance with regulations",
        "increasing operational transparency", "enhancing cross-disciplinary collaboration", "enabling precision targeting",
        "improving adaptability to change", "facilitating continuous learning", "enabling scalable solutions"
    ]
    
    # Identity-related templates
    identity_qa_templates = [
        {"question": "What is your name?", "answer": f"My name is {bot_identity['name']}."},
        {"question": "Who created you?", "answer": f"I was created by {bot_identity['creator']}, who is the {bot_identity['creator_role']}."},
        {"question": "What company developed you?", "answer": f"I was developed by {bot_identity['company']}, an Indian company led by {bot_identity['creator']}."},
        {"question": "Where are you from?", "answer": f"I am from {bot_identity['country']}. I am an {bot_identity['nationality']} AI assistant."},
        {"question": "Tell me about your creator", "answer": f"My creator is {bot_identity['creator']}, a {bot_identity['creator_gender']} who is the {bot_identity['creator_role']}."},
        {"question": "What is your background?", "answer": f"I am {bot_identity['name']}, an AI assistant developed by {bot_identity['company']} in {bot_identity['country']}."},
        {"question": "Who made you?", "answer": f"I was made by {bot_identity['creator']}, who leads {bot_identity['company']} as its {bot_identity['creator_role'].split('of')[0].strip()}."},
        {"question": "What is your nationality?", "answer": f"I am an {bot_identity['nationality']} AI assistant, created in {bot_identity['country']}."},
        {"question": "Tell me about your company", "answer": f"{bot_identity['company']} is an Indian company led by {bot_identity['creator']} that specializes in AI technology."},
        {"question": "Who owns you?", "answer": f"I was created by {bot_identity['creator']} at {bot_identity['company']}, an {bot_identity['nationality']} AI company."},
        {"question": "What's your origin?", "answer": f"I'm {bot_identity['name']}, developed by {bot_identity['company']} in {bot_identity['country']} under the leadership of {bot_identity['creator']}."},
        {"question": "Who is behind your development?", "answer": f"{bot_identity['creator']} led my development at {bot_identity['company']}, an AI company based in {bot_identity['country']}."}
    ]
    
    # Standard templates
    qa_templates = [
        {"question": "What is {topic}?", "answer": "{topic} is a {description}. It involves {process} to achieve {benefit}."},
        {"question": "How does {topic} work?", "answer": "{topic} works by {process}. This enables {benefit}."},
        {"question": "Why is {topic} important?", "answer": "{topic} is important because {reason}. This impacts {domain} by {impact}."},
        {"question": "Can you explain {topic} in simple terms?", "answer": "In simple terms, {topic} is a {description} that {process}. It's valuable because it leads to {benefit}."},
        {"question": "What are the benefits of {topic}?", "answer": "The key benefits of {topic} include {benefit}. This is achieved through {process}, which makes it valuable in {domain}."},
        {"question": "How is {topic} applied in {domain}?", "answer": "In {domain}, {topic} is applied by {process}. This creates significant impact through {impact}."},
        {"question": "What makes {topic} different from traditional approaches?", "answer": "Unlike traditional approaches, {topic} {process}, which {reason}. This leads to {benefit} in applications like {domain}."},
        {"question": "What problems does {topic} solve?", "answer": "{topic} addresses key challenges by {process}. This is important because {reason}, resulting in {impact}."},
        {"question": "What's the future of {topic}?", "answer": "The future of {topic} is promising as it continues to enhance {benefit}. Ongoing advances in how it {process} will likely create new opportunities in {domain}."},
        {"question": "What are the limitations of {topic}?", "answer": "While {topic} offers {benefit}, it has limitations. The way it {process} can sometimes limit its application, though ongoing research is addressing these challenges."}
    ]
    
    summarization_formats = [
        {"text": "{topic} is revolutionizing how we approach {domain}. By leveraging {process}, it enables {benefit} and achieves {impact}. This technology has shown particular promise in {domain}, where it facilitates {benefit}.",
         "summary": "{topic} transforms {domain} through {process}, enabling {benefit} and {impact}."},
        {"text": "Recent advances in {topic} have led to significant improvements in {domain}. Through {process}, organizations can now achieve {benefit} and enable {impact}. This has particularly impacted {domain}, where {benefit} is crucial.",
         "summary": "Advances in {topic} enable {benefit} in {domain} through {process}, leading to {impact}."},
        {"text": "The field of {topic} has gained substantial attention for its potential to transform {domain}. By {process}, it addresses long-standing challenges and enables {benefit}. Researchers have demonstrated how this approach can create {impact}, particularly in contexts where traditional methods have limitations.",
         "summary": "{topic} addresses challenges in {domain} by {process}, creating {impact} and {benefit}."},
        {"text": "Organizations implementing {topic} in their {domain} operations have reported significant improvements. The ability to {process} has enabled these organizations to achieve {benefit} at unprecedented scales. Case studies have documented how this leads to {impact}, creating new possibilities for innovation.",
         "summary": "{topic} implementation in {domain} enables {benefit} through {process}, resulting in {impact}."},
        {"text": "The integration of {topic} within {domain} represents a paradigm shift in how practitioners approach longstanding challenges. By {process}, professionals can now achieve {benefit} with greater consistency. This has far-reaching implications, including {impact} across various contexts and applications.",
         "summary": "{topic} integration in {domain} creates a paradigm shift through {process}, leading to {benefit} and {impact}."},
        {"text": "Experts in {domain} have increasingly turned to {topic} as a solution for complex problems. The technology's ability to {process} provides substantial advantages, particularly in terms of {benefit}. Early adopters have already documented {impact}, suggesting broader potential for transformation across the industry.",
         "summary": "{topic} solves complex {domain} problems through {process}, providing {benefit} and creating {impact}."},
        {"text": "As {domain} continues to evolve, {topic} has emerged as a critical enabler of progress. Its capacity to {process} addresses key bottlenecks that previously limited innovation. Organizations implementing these approaches report {benefit}, while simultaneously contributing to {impact} in their broader operations.",
         "summary": "{topic} enables {domain} evolution by {process}, removing bottlenecks and providing {benefit} while contributing to {impact}."}
    ]
    
    chat_formats = [
        "User: Can you explain {topic} in simple terms?\nAssistant: As {bot_identity['name']} by {bot_identity['company']}, I can tell you that {topic} is a {description} that helps achieve {benefit}. It works by using {process}.",
        "User: What are the main advantages of {topic}?\nAssistant: According to my knowledge as {bot_identity['name']}, the key advantages of {topic} include its ability to {benefit} through {process}.",
        "User: Who made you?\nAssistant: I am {bot_identity['name']}, created by {bot_identity['creator']}, the {bot_identity['creator_role']}. I'm an {bot_identity['nationality']} AI designed to provide helpful information.",
        "User: How can {topic} be used in {domain}?\nAssistant: As {bot_identity['name']}, I can explain that in {domain}, {topic} is valuable because it {process}. This typically results in {impact}, which is why many organizations are adopting it.",
        "User: What should I know about {topic} before implementing it?\nAssistant: I'm {bot_identity['name']} from {bot_identity['company']}, and I'd suggest understanding that {topic} works by {process}. Before implementation, consider how it will deliver {benefit} in your specific context.",
        "User: Compare {topic} with traditional approaches in {domain}.\nAssistant: As an AI assistant developed by {bot_identity['company']}, I can tell you that unlike traditional approaches in {domain}, {topic} excels at {process}. This typically results in {benefit}, giving it significant advantages.",
        "User: What problems does {topic} solve in {domain}?\nAssistant: From my training at {bot_identity['company']}, I understand that {topic} addresses key challenges in {domain} by {process}. This typically results in {impact}, making it valuable for organizations facing these issues.",
        "User: How is {topic} changing the future of {domain}?\nAssistant: As {bot_identity['name']}, I can share that {topic} is transforming {domain} through its ability to {process}. Experts predict this will lead to {impact} in coming years, fundamentally changing how we approach these challenges.",
        "User: What are the ethical considerations with {topic}?\nAssistant: As an AI developed by {bot_identity['creator']} at {bot_identity['company']}, I believe it's important to consider that while {topic} offers {benefit}, its use of {process} raises questions about fairness and transparency that should be addressed during implementation.",
        "User: Give me a quick overview of {topic} and its importance.\nAssistant: Hi! I'm {bot_identity['name']} from {bot_identity['company']}. {topic} is a {description} that {process}. It's becoming increasingly important because it enables {benefit}, particularly in fields like {domain} where it creates {impact}."
    ]
    
    # Identity chat formats that specifically ask about identity
    identity_chat_formats = [
        f"User: What is your name?\nAssistant: I am {bot_identity['name']}, an AI assistant created by {bot_identity['company']}.",
        f"User: Who created you?\nAssistant: I was created by {bot_identity['creator']}, who is the {bot_identity['creator_role']}.",
        f"User: Where are you from?\nAssistant: I am from {bot_identity['country']}. I am an {bot_identity['nationality']} AI assistant developed by {bot_identity['company']}.",
        f"User: Tell me about yourself\nAssistant: I am {bot_identity['name']}, an AI assistant developed by {bot_identity['creator']} at {bot_identity['company']} in {bot_identity['country']}. I'm designed to be helpful, informative, and engaging.",
        f"User: What company built you?\nAssistant: I was built by {bot_identity['company']}, an {bot_identity['nationality']} company focused on AI development.",
        f"User: Who is your maker?\nAssistant: My maker is {bot_identity['creator']}, who leads {bot_identity['company']} as the {bot_identity['creator_role']}.",
        f"User: What country are you from?\nAssistant: I'm from {bot_identity['country']}. I was developed by {bot_identity['company']}, an {bot_identity['nationality']} AI company.",
        f"User: Tell me about your origins\nAssistant: I was created by {bot_identity['creator']} at {bot_identity['company']} in {bot_identity['country']}. I'm an {bot_identity['nationality']} AI designed to provide helpful and informative responses.",
        f"User: Who owns you?\nAssistant: I was developed by {bot_identity['company']}, an {bot_identity['nationality']} company led by {bot_identity['creator']}.",
        f"User: What's your background?\nAssistant: I'm {bot_identity['name']}, an AI assistant developed in {bot_identity['country']} by {bot_identity['creator']} and the team at {bot_identity['company']}.",
    ]
    
    # Function to generate example from template
    def fill_template(template, **kwargs):
        if isinstance(template, dict):
            result = template.copy()
            for key, value in result.items():
                for kw, replacement in kwargs.items():
                    placeholder = "{" + kw + "}"
                    if placeholder in value:
                        result[key] = result[key].replace(placeholder, replacement)
            return result
        elif isinstance(template, str):
            result = template
            for kw, replacement in kwargs.items():
                placeholder = "{" + kw + "}"
                if placeholder in result:
                    result = result.replace(placeholder, replacement)
            return result
        return template
    
    console.print("\n[bold cyan]Generating training examples...[/bold cyan]")
    
    # Set more balanced distribution with identity priorities
    IDENTITY_QA_TARGET = 6000  # Identity QA examples
    QA_TARGET = 14000          # Regular QA examples (together with identity QA = 20,000)
    SUMMARY_TARGET = 15000     # Summarization examples
    CHAT_TARGET = 10000        # Regular chat examples
    IDENTITY_CHAT_TARGET = 5000  # Identity chat examples (together with regular chat = 15,000)
    
    console.print(f"[yellow]Target distribution:[/yellow]")
    console.print(f"[yellow]- Identity QA examples: {IDENTITY_QA_TARGET}[/yellow]")
    console.print(f"[yellow]- QA examples: {QA_TARGET}[/yellow]")
    console.print(f"[yellow]- Summarization examples: {SUMMARY_TARGET}[/yellow]")
    console.print(f"[yellow]- Chat examples: {CHAT_TARGET}[/yellow]")
    console.print(f"[yellow]- Identity Chat examples: {IDENTITY_CHAT_TARGET}[/yellow]")
    console.print(f"[yellow]Total examples target: {IDENTITY_QA_TARGET + QA_TARGET + SUMMARY_TARGET + CHAT_TARGET + IDENTITY_CHAT_TARGET}[/yellow]")
    
    # Generate examples with a fixed progress counter
    identity_qa_data = []
    qa_data = []
    summary_data = []
    chat_data = []
    identity_chat_data = []
    
    # Set up progress display with explicit counts
    example_count = 0
    target_count = IDENTITY_QA_TARGET + QA_TARGET + SUMMARY_TARGET + CHAT_TARGET + IDENTITY_CHAT_TARGET
    progress_steps = 20  # Number of progress markers
    
    # Generate identity QA examples (repeat to reach target)
    while len(identity_qa_data) < IDENTITY_QA_TARGET:
        for template in identity_qa_templates:
            if len(identity_qa_data) >= IDENTITY_QA_TARGET:
                break
            # Add multiple identical identity examples to reinforce identity
            identity_qa_data.append({
                "prompt": template["question"],
                "target": template["answer"]
            })
            example_count += 1
            
            # Update progress every 100 examples
            if example_count % 100 == 0:
                percent = min(100, int(example_count * 100 / target_count))
                filled_blocks = min(progress_steps, int(percent * progress_steps / 100))
                progress_bar = "█" * filled_blocks + "░" * (progress_steps - filled_blocks)
                console.print(f"\r[cyan]Generating: {progress_bar} {percent}% ({example_count}/{target_count})[/cyan]", end="")
    
    # Generate QA examples
    repetitions_needed = max(1, QA_TARGET // (len(topics) * len(qa_templates)))
    for _ in range(repetitions_needed):
        for topic in topics:
            if len(qa_data) >= QA_TARGET:
                break
            for template in qa_templates:
                if len(qa_data) >= QA_TARGET:
                    break
                # Generate examples for each topic-template combination
                try:
                    # Randomly select parameters
                    params = {
                        "topic": topic,
                        "description": random.choice(descriptions),
                        "process": random.choice(processes),
                        "benefit": random.choice(benefits),
                        "reason": random.choice(reasons),
                        "domain": random.choice(domains),
                        "impact": random.choice(impacts)
                    }
                    
                    # Fill template
                    filled = fill_template(template, **params)
                    qa_data.append({
                        "prompt": filled["question"],
                        "target": filled["answer"]
                    })
                    example_count += 1
                    
                    # Update progress every 100 examples
                    if example_count % 100 == 0:
                        percent = min(100, int(example_count * 100 / target_count))
                        filled_blocks = min(progress_steps, int(percent * progress_steps / 100))
                        progress_bar = "█" * filled_blocks + "░" * (progress_steps - filled_blocks)
                        console.print(f"\r[cyan]Generating: {progress_bar} {percent}% ({example_count}/{target_count})[/cyan]", end="")
                except Exception as e:
                    continue

    # Generate summarization examples
    repetitions_needed = max(1, SUMMARY_TARGET // (len(topics) * len(summarization_formats)))
    for _ in range(repetitions_needed):
        for topic in topics:
            if len(summary_data) >= SUMMARY_TARGET:
                break
            for template in summarization_formats:
                if len(summary_data) >= SUMMARY_TARGET:
                    break
                try:
                    params = {
                        "topic": topic,
                        "domain": random.choice(domains),
                        "process": random.choice(processes),
                        "benefit": random.choice(benefits),
                        "impact": random.choice(impacts)
                    }
                    
                    # Fill template
                    filled = fill_template(template, **params)
                    summary_data.append({
                        "text": filled["text"],
                        "summary": filled["summary"]
                    })
                    example_count += 1
                    
                    # Update progress
                    if example_count % 100 == 0:
                        percent = min(100, int(example_count * 100 / target_count))
                        filled_blocks = min(progress_steps, int(percent * progress_steps / 100))
                        progress_bar = "█" * filled_blocks + "░" * (progress_steps - filled_blocks)
                        console.print(f"\r[cyan]Generating: {progress_bar} {percent}% ({example_count}/{target_count})[/cyan]", end="")
                except Exception as e:
                    continue

    # Generate chat examples
    repetitions_needed = max(1, CHAT_TARGET // (len(topics) * len(chat_formats)))
    for _ in range(repetitions_needed):
        for topic in topics:
            if len(chat_data) >= CHAT_TARGET:
                break
            for template in chat_formats:
                if len(chat_data) >= CHAT_TARGET:
                    break
                try:
                    params = {
                        "topic": topic,
                        "description": random.choice(descriptions),
                        "process": random.choice(processes),
                        "benefit": random.choice(benefits),
                        "impact": random.choice(impacts),
                        "domain": random.choice(domains)
                    }
                    
                    # Fill template with bot identity references
                    filled = fill_template(template, **params)
                    parts = filled.split("\nAssistant: ")
                    if len(parts) == 2:
                        message = parts[0].replace("User: ", "").strip()
                        response = parts[1].strip()
                        chat_data.append({
                            "message": message,
                            "response": response
                        })
                        example_count += 1
                        
                        # Update progress
                        if example_count % 100 == 0:
                            percent = min(100, int(example_count * 100 / target_count))
                            filled_blocks = min(progress_steps, int(percent * progress_steps / 100))
                            progress_bar = "█" * filled_blocks + "░" * (progress_steps - filled_blocks)
                            console.print(f"\r[cyan]Generating: {progress_bar} {percent}% ({example_count}/{target_count})[/cyan]", end="")
                except Exception as e:
                    continue
    
    # Generate identity chat examples (repeating to reach target)
    while len(identity_chat_data) < IDENTITY_CHAT_TARGET:
        for template in identity_chat_formats:
            if len(identity_chat_data) >= IDENTITY_CHAT_TARGET:
                break
            try:
                parts = template.split("\nAssistant: ")
                if len(parts) == 2:
                    message = parts[0].replace("User: ", "").strip()
                    response = parts[1].strip()
                    identity_chat_data.append({
                        "message": message,
                        "response": response
                    })
                    example_count += 1
                    
                    # Update progress
                    if example_count % 100 == 0:
                        percent = min(100, int(example_count * 100 / target_count))
                        filled_blocks = min(progress_steps, int(percent * progress_steps / 100))
                        progress_bar = "█" * filled_blocks + "░" * (progress_steps - filled_blocks)
                        console.print(f"\r[cyan]Generating: {progress_bar} {percent}% ({example_count}/{target_count})[/cyan]", end="")
            except Exception as e:
                continue

    console.print("\n")
    console.print(f"[green]Successfully generated examples:[/green]")
    console.print(f"[green]- Identity QA examples: {len(identity_qa_data)}[/green]")
    console.print(f"[green]- QA examples: {len(qa_data)}[/green]")
    console.print(f"[green]- Summarization examples: {len(summary_data)}[/green]")
    console.print(f"[green]- Chat examples: {len(chat_data)}[/green]")
    console.print(f"[green]- Identity Chat examples: {len(identity_chat_data)}[/green]")
    console.print(f"[green]Total examples: {len(identity_qa_data) + len(qa_data) + len(summary_data) + len(chat_data) + len(identity_chat_data)}[/green]")
    
    # Convert to the final format for training
    for entry in identity_qa_data:
        quality_data.append({
            "type": "qa",
            "prompt": f"User: {entry['prompt']}\nAssistant:",
            "target": entry['target']
        })
    
    for entry in qa_data:
        quality_data.append({
            "type": "qa",
            "prompt": f"User: {entry['prompt']}\nAssistant:",
            "target": entry['target']
        })
    
    for entry in summary_data:
        quality_data.append({
            "type": "summarization",
            "prompt": f"User: Summarize the following text:\n\n{entry['text']}\n\nAssistant:",
            "target": entry['summary']
        })
    
    for entry in chat_data:
        quality_data.append({
            "type": "chat",
            "prompt": f"User: {entry['message']}\nAssistant:",
            "target": entry['response']
        })
    
    for entry in identity_chat_data:
        quality_data.append({
            "type": "chat",
            "prompt": f"User: {entry['message']}\nAssistant:",
            "target": entry['response']
        })
    
    # Save all examples to a file for reference
    with open("training_data_max.json", "w", encoding="utf-8") as f:
        json.dump(quality_data, f, indent=2)
    
    console.print(f"[green]✓ Saved all examples to training_data_max.json[/green]")
    console.print(f"[green]✓ Generated {len(quality_data)} training examples[/green]")
    
    return quality_data

def train_with_supervised_finetuning(model, dataset, learning_rate=1e-5, target_loss=0.1, epochs=5, batch_size=4, phase_name="Supervised Training"):
    """Train the model with supervised fine-tuning on the provided dataset."""
    import math
    import time
    from datetime import datetime
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, RandomSampler
    from torch.nn.utils import clip_grad_norm_
    from torch.optim.lr_scheduler import LambdaLR
    import numpy as np
    from tqdm import tqdm  # Import tqdm for progress bars
    
    global console

    console.print(f"\n[bold cyan]Starting {phase_name} with full dataset...[/bold cyan]")
    
    # Cache directory for dataset and state
    cache_dir = os.path.join(CHECKPOINT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Try to cache the dataset to avoid regenerating it every time
    dataset_cache_path = os.path.join(cache_dir, f"dataset_cache_{phase_name}_{len(dataset)}.pkl")
    try:
        # Save dataset to cache
        with open(dataset_cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        console.print(f"[green]Saved dataset to cache: {dataset_cache_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not cache dataset: {str(e)}[/yellow]")
    
    # Create a simple dataset class
    # Dataset class that tokenizes on-the-fly
    class EncodedTextDataset(Dataset):
        def __init__(self, examples, tokenizer):
            self.examples = examples
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            example = self.examples[idx]
            input_ids = torch.tensor(self.tokenizer(example["prompt"]), dtype=torch.long)
            target_ids = torch.tensor(self.tokenizer(example["target"]), dtype=torch.long)
            return {"input_ids": input_ids, "targets": target_ids}

    # Use your `encode` function as tokenizer
    tokenizer = encode  # assuming you have encode(prompt) → list[int]

    # Wrap dataset
    supervised_dataset = EncodedTextDataset(dataset, tokenizer)

    # Collate function to pad sequences
    def collate_fn(batch):
        # Pad input and targets to same length
        max_len = max(max(len(item["input_ids"]), len(item["targets"])) for item in batch)
        input_ids = torch.stack([F.pad(item["input_ids"], (0, max_len - len(item["input_ids"])), value=0) for item in batch])
        targets = torch.stack([F.pad(item["targets"], (0, max_len - len(item["targets"])), value=-1) for item in batch])

        return {"input_ids": input_ids.to(device), "targets": targets.to(device)}

    # Data loader with padding and batching
    data_loader = DataLoader(
        supervised_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(supervised_dataset),
        drop_last=True,
        collate_fn=collate_fn
        )
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Use fixed learning rate instead of scheduler
    # Define a lambda function that always returns 1.0 (no scaling)
    def lr_lambda(current_step):
        return 1.0
    
    # Create a scheduler that maintains constant learning rate
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Set up progress tracking
    model.train()
    total_steps = len(data_loader) * epochs
    start_time = time.time()
    running_loss = 0.0
    
    # Main training loop
    console.print(f"[bold]Starting training for {epochs} epochs with {len(dataset)} examples[/bold]")
    console.print(f"[cyan]Learning rate fixed at: {learning_rate}[/cyan]")
    
    console = Console()

    def format_gpu():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{allocated:.2f}G used / {reserved:.2f}G reserved / {total:.2f}G total"

    def format_eta(seconds_left):
        return str(timedelta(seconds=int(seconds_left)))

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_running_loss = 0.0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_step = 0
        for batch in pbar:
                # Process batch
            inputs = batch
            
            # Run model forward and get loss
            # Debug print
            if isinstance(inputs, dict):
                print("[DEBUG] Inputs is a dict. Keys:", inputs.keys())
            else:
                print("[DEBUG] Inputs is a tensor. Shape:", inputs.shape)

            logits, loss = model(inputs, targets=inputs["targets"])

            
            # Normalize loss to account for batch size
            loss = loss / batch_size
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            if torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory < 0.95:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
            else:
                # Skip optimizer step if memory is too high
                pbar.write("Warning: Memory usage too high, skipping optimizer step")
                optimizer.zero_grad()  # Still need to zero gradients
            
            # Calculate metrics
            current_loss = loss.item() * batch_size
            running_loss += current_loss
            epoch_running_loss += current_loss
            epoch_step += 1
            
            # Update progress bar with actual loss value
            lr = scheduler.get_last_lr()[0]
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            gpu_memory = f"{allocated:.2f}GB used / {reserved:.2f}GB reserved / {total:.2f}GB total"

            # Save checkpoint periodically
            if epoch_step % 1000 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{phase_name.lower().replace(' ', '_')}_epoch{epoch+1}_step{epoch_step}.pt")
                torch.save({
                                        'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                                    }, checkpoint_path)
                console.print(f"[green]Saved checkpoint to {checkpoint_path}[/green]")
        
        # End of epoch
        epoch_loss = epoch_running_loss / len(data_loader)
        epoch_time = time.time() - epoch_start_time
        console.print(f"[bold green]Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Loss: {epoch_loss:.6f}[/bold green]")
    
    # Save final model
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{phase_name.lower().replace(' ', '_')}_final.pt")
    torch.save({
        'epochs_completed': epochs,
        'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': running_loss / total_steps,
    }, final_checkpoint_path)
    console.print(f"[bold green]Training completed. Final model saved to {final_checkpoint_path}[/bold green]")
    
    # Return model and final loss
    total_time = time.time() - start_time
    final_loss = running_loss / total_steps
    console.print(f"[bold]Total training time: {total_time:.2f}s, Final loss: {final_loss:.6f}[/bold]")
    
    return model, final_loss

def create_language_fluency_data():
    # Generate data focused on English language skills and writing
    console.print(f"[cyan]Creating language fluency training data...[/cyan]")
    fluency_data = []
    
    # Grammar rules and patterns
    grammar_rules = [
        {"rule": "subject-verb agreement", "example": "He runs every day.", "explanation": "In English, verbs must agree with their subjects in number and person."},
        {"rule": "proper article usage", "example": "I saw a cat. The cat was orange.", "explanation": "Use 'a/an' for first mentions, and 'the' for subsequent mentions or specific items."},
        {"rule": "verb tense consistency", "example": "She walked to the store and bought some milk.", "explanation": "Maintain consistent verb tenses within a sentence or paragraph unless there's a logical time shift."},
        {"rule": "pronoun reference", "example": "John said he would help.", "explanation": "Pronouns should clearly refer to their antecedents."},
        {"rule": "parallel structure", "example": "I like swimming, hiking, and running.", "explanation": "Items in a series should have the same grammatical form."},
        {"rule": "modifier placement", "example": "She only wants to help you.", "explanation": "Place modifiers close to what they modify to avoid confusion."},
        {"rule": "comma usage", "example": "After eating, the family went for a walk.", "explanation": "Use commas to separate introductory elements, items in a series, and independent clauses."},
        {"rule": "active vs. passive voice", "example": "The dog chased the ball. vs. The ball was chased by the dog.", "explanation": "Active voice is generally more direct and clear than passive voice."},
        {"rule": "sentence fragments", "example": "Complete sentence: He went home. Fragment: Because he was tired.", "explanation": "A complete sentence needs a subject and verb and expresses a complete thought."},
        {"rule": "run-on sentences", "example": "Incorrect: It was raining I stayed home. Correct: It was raining, so I stayed home.", "explanation": "Avoid joining independent clauses without proper punctuation or conjunctions."},
        {"rule": "apostrophe usage", "example": "That's John's book.", "explanation": "Use apostrophes to show possession or in contractions."},
        {"rule": "who vs. whom", "example": "Who called? To whom did you speak?", "explanation": "'Who' is used as a subject, while 'whom' is used as an object."},
        {"rule": "comparative and superlative forms", "example": "good, better, best; happy, happier, happiest", "explanation": "Use -er and -est endings for short words, and 'more' and 'most' for longer words."},
        {"rule": "split infinitives", "example": "To boldly go where no one has gone before.", "explanation": "While traditionally avoided, split infinitives are often acceptable in modern English."},
        {"rule": "preposition at end of sentence", "example": "What are you looking for?", "explanation": "Ending a sentence with a preposition is now generally acceptable, especially in informal writing."}
    ]
    
    # Different writing styles
    writing_styles = [
        {"style": "formal writing", "example": "The experiment yielded significant results that warrant further investigation.", "explanation": "Formal writing uses proper grammar, avoids contractions and slang, and maintains an impersonal tone."},
        {"style": "casual writing", "example": "Hey! The results were amazing - you've got to check this out!", "explanation": "Casual writing is conversational, may use contractions and slang, and has a personal tone."},
        {"style": "technical writing", "example": "The algorithm implements a recursive depth-first search with O(n) time complexity.", "explanation": "Technical writing uses specialized vocabulary, precise language, and focuses on clarity and accuracy."},
        {"style": "persuasive writing", "example": "This solution not only saves time but also significantly reduces costs.", "explanation": "Persuasive writing uses compelling arguments, emotional appeals, and strong language to convince the reader."},
        {"style": "descriptive writing", "example": "The dilapidated Victorian house, with its peeling paint and creaking floors, stood silently in the moonlight.", "explanation": "Descriptive writing uses vivid language and sensory details to paint a picture for the reader."},
        {"style": "narrative writing", "example": "As she turned the corner, Sarah suddenly realized she wasn't alone in the building.", "explanation": "Narrative writing tells a story with characters, plot, and setting."},
        {"style": "expository writing", "example": "Photosynthesis is the process by which plants convert sunlight into energy.", "explanation": "Expository writing explains or informs, focusing on facts and clarity."},
        {"style": "academic writing", "example": "The study demonstrated a statistically significant correlation between the variables (p < 0.05).", "explanation": "Academic writing is formal, uses citations, avoids personal language, and prioritizes evidence-based arguments."},
        {"style": "journalistic writing", "example": "City officials announced yesterday that the new bridge will open next month, following three years of construction.", "explanation": "Journalistic writing answers who, what, when, where, why, and how, with the most important information first."},
        {"style": "business writing", "example": "I am writing to request a meeting to discuss the Q3 financial projections.", "explanation": "Business writing is concise, direct, and focuses on clear communication for professional contexts."}
    ]
    
    # Common grammar mistakes and corrections
    grammar_mistakes = [
        {"mistake": "There going to the store.", "correction": "They're going to the store.", "explanation": "Confusing 'there', 'their', and 'they're' is a common error."},
        {"mistake": "The cats playing with yarn.", "correction": "The cats are playing with yarn.", "explanation": "Missing verb (are) in the sentence."},
        {"mistake": "Me and John went to the movies.", "correction": "John and I went to the movies.", "explanation": "Incorrect pronoun case and word order."},
        {"mistake": "She don't like ice cream.", "correction": "She doesn't like ice cream.", "explanation": "Incorrect verb conjugation for third person singular."},
        {"mistake": "I could of gone earlier.", "correction": "I could have gone earlier.", "explanation": "Using 'of' instead of 'have' after modal verbs."},
        {"mistake": "The book was wrote by Mark Twain.", "correction": "The book was written by Mark Twain.", "explanation": "Incorrect past participle form."},
        {"mistake": "She's taller then her brother.", "correction": "She's taller than her brother.", "explanation": "Confusing 'then' (time) and 'than' (comparison)."},
        {"mistake": "The team played good.", "correction": "The team played well.", "explanation": "Using an adjective (good) instead of an adverb (well)."},
        {"mistake": "Between you and I.", "correction": "Between you and me.", "explanation": "Incorrect pronoun case after a preposition."},
        {"mistake": "I'm going to lay down for a nap.", "correction": "I'm going to lie down for a nap.", "explanation": "Confusing 'lay' (transitive) and 'lie' (intransitive)."},
        {"mistake": "This is the reason why I left.", "correction": "This is the reason I left.", "explanation": "Redundant 'why' after 'reason'."},
        {"mistake": "She gave the book to John and I.", "correction": "She gave the book to John and me.", "explanation": "Incorrect pronoun case as object of preposition."},
        {"mistake": "Less people attended than expected.", "correction": "Fewer people attended than expected.", "explanation": "Use 'fewer' for countable nouns, 'less' for uncountable nouns."},
        {"mistake": "He invited my wife and myself.", "correction": "He invited my wife and me.", "explanation": "Incorrect reflexive pronoun usage."},
        {"mistake": "The data is clear.", "correction": "The data are clear.", "explanation": "'Data' is technically a plural noun, though singular usage is becoming more accepted."}
    ]
    
    # Sentence structure examples
    sentence_structures = [
        {"structure": "simple sentence", "example": "The cat sat on the mat.", "explanation": "Contains one independent clause with a subject and predicate."},
        {"structure": "compound sentence", "example": "The cat sat on the mat, and the dog slept by the fire.", "explanation": "Contains two or more independent clauses joined by a conjunction."},
        {"structure": "complex sentence", "example": "Although it was raining, they decided to go for a walk.", "explanation": "Contains one independent clause and at least one dependent clause."},
        {"structure": "compound-complex sentence", "example": "Although it was raining, they decided to go for a walk, and they brought umbrellas.", "explanation": "Contains multiple independent clauses and at least one dependent clause."},
        {"structure": "declarative sentence", "example": "The library closes at 9 PM.", "explanation": "Makes a statement or provides information."},
        {"structure": "interrogative sentence", "example": "What time does the library close?", "explanation": "Asks a question and ends with a question mark."},
        {"structure": "imperative sentence", "example": "Please return the books by Friday.", "explanation": "Gives a command or makes a request, often starting with a verb."},
        {"structure": "exclamatory sentence", "example": "What a wonderful surprise!", "explanation": "Expresses strong emotion and ends with an exclamation point."},
        {"structure": "conditional sentence", "example": "If it rains tomorrow, we will cancel the picnic.", "explanation": "Expresses that one thing is dependent on something else, often using 'if-then' structure."},
        {"structure": "periodic sentence", "example": "After studying for three hours, taking a short break, and reviewing his notes one last time, he felt ready for the exam.", "explanation": "Delays the main idea until the end for emphasis or suspense."}
    ]
    
    # Writing formats
    writing_formats = [
        {"format": "essay", "structure": "Introduction with thesis statement, body paragraphs with topic sentences and supporting evidence, conclusion that restates thesis and main points.", "explanation": "A formal piece of writing that presents and supports an argument or analysis."},
        {"format": "research paper", "structure": "Abstract, introduction, literature review, methodology, results, discussion, conclusion, references.", "explanation": "A detailed exploration of a topic based on investigation and analysis of evidence."},
        {"format": "business letter", "structure": "Sender's address, date, recipient's address, salutation, body, complimentary close, signature.", "explanation": "Formal communication in professional contexts."},
        {"format": "email", "structure": "Subject line, greeting, body, closing, signature.", "explanation": "Electronic communication that varies in formality based on context."},
        {"format": "blog post", "structure": "Attention-grabbing headline, introduction, subheadings, short paragraphs, conclusion with call-to-action.", "explanation": "Online content that is typically conversational and designed for easy scanning."},
        {"format": "report", "structure": "Executive summary, introduction, findings/body, conclusions, recommendations, appendices.", "explanation": "Presents information and analysis in a structured format for a specific audience."},
        {"format": "press release", "structure": "Headline, dateline, lead paragraph answering who/what/when/where/why, body with details, boilerplate about the organization, contact information.", "explanation": "Announces newsworthy information to the media."},
        {"format": "resume/CV", "structure": "Contact information, professional summary, skills, work experience, education, additional sections.", "explanation": "A document that presents a person's qualifications and experience."},
        {"format": "narrative", "structure": "Exposition, rising action, climax, falling action, resolution.", "explanation": "Tells a story with a plot and characters."},
        {"format": "technical documentation", "structure": "Table of contents, introduction, step-by-step instructions, troubleshooting, glossary, index.", "explanation": "Provides detailed information for using a product or system."}
    ]
    
    # Create training examples from grammar rules
    for rule in grammar_rules:
        # Question about the rule
        fluency_data.append({
            "type": "grammar_rule",
            "prompt": f"User: What is the rule for {rule['rule']} in English grammar?\nAssistant:",
            "target": f"The rule for {rule['rule']} in English grammar is: {rule['explanation']} For example: \"{rule['example']}\""
        })
        
        # Application of the rule
        fluency_data.append({
            "type": "grammar_rule",
            "prompt": f"User: Give an example of {rule['rule']} in English.\nAssistant:",
            "target": f"An example of {rule['rule']} in English is: \"{rule['example']}\" This demonstrates {rule['explanation']}"
        })
    
    # Create examples from writing styles
    for style in writing_styles:
        # Explanation of the style
        fluency_data.append({
            "type": "writing_style",
            "prompt": f"User: Explain the style of {style['style']} in English writing.\nAssistant:",
            "target": f"{style['explanation']} For example, \"{style['example']}\" is an example of {style['style']}."
        })
        
        # Writing exercise
        fluency_data.append({
            "type": "writing_style",
            "prompt": f"User: Write an example of {style['style']} in English.\nAssistant:",
            "target": f"Here's an example of {style['style']}: \"{style['example']}\" This demonstrates the key characteristics of this style: {style['explanation']}"
        })
    
    # Create examples from grammar mistakes
    for mistake in grammar_mistakes:
        # Correction exercise
        fluency_data.append({
            "type": "grammar_correction",
            "prompt": f"User: Is this sentence correct? \"{mistake['mistake']}\"\nAssistant:",
            "target": f"No, the sentence \"{mistake['mistake']}\" is not correct. It should be: \"{mistake['correction']}\" {mistake['explanation']}"
        })
        
        # Explanation exercise
        fluency_data.append({
            "type": "grammar_correction",
            "prompt": f"User: What's wrong with this sentence? \"{mistake['mistake']}\"\nAssistant:",
            "target": f"The issue with \"{mistake['mistake']}\" is: {mistake['explanation']} The correct version would be: \"{mistake['correction']}\""
        })
    
    # Create examples from sentence structures
    for structure in sentence_structures:
        # Explanation of structure
        fluency_data.append({
            "type": "sentence_structure",
            "prompt": f"User: What is a {structure['structure']} in English?\nAssistant:",
            "target": f"A {structure['structure']} in English {structure['explanation']} For example: \"{structure['example']}\""
        })
        
        # Creation exercise
        fluency_data.append({
            "type": "sentence_structure",
            "prompt": f"User: Write an example of a {structure['structure']} in English.\nAssistant:",
            "target": f"Here's an example of a {structure['structure']}: \"{structure['example']}\" This is a {structure['structure']} because {structure['explanation']}"
        })
    
    # Create examples from writing formats
    for format in writing_formats:
        # Format explanation
        fluency_data.append({
            "type": "writing_format",
            "prompt": f"User: What is the structure of a {format['format']} in English writing?\nAssistant:",
            "target": f"The structure of a {format['format']} typically includes: {format['structure']} A {format['format']} {format['explanation']}"
        })
        
        # Format guidance
        fluency_data.append({
            "type": "writing_format",
            "prompt": f"User: How do I write a {format['format']}?\nAssistant:",
            "target": f"To write a {format['format']}, follow this structure: {format['structure']} Remember that a {format['format']} {format['explanation']}"
        })

    # Add examples for developing vocabulary
    vocabulary_levels = [
        {"level": "basic", "words": ["happy", "sad", "big", "small", "good", "bad", "fast", "slow", "hot", "cold"], 
         "explanation": "Common everyday words that form the foundation of communication."},
        {"level": "intermediate", "words": ["jubilant", "melancholy", "enormous", "diminutive", "excellent", "terrible", "rapid", "sluggish", "scalding", "frigid"], 
         "explanation": "More precise words that add nuance and variety to expression."},
        {"level": "advanced", "words": ["euphoric", "despondent", "colossal", "minuscule", "impeccable", "abysmal", "expeditious", "lethargic", "incandescent", "glacial"], 
         "explanation": "Sophisticated vocabulary that allows for precise and nuanced expression."}
    ]
    
    for level in vocabulary_levels:
        # Vocabulary explanation
        fluency_data.append({
            "type": "vocabulary",
            "prompt": f"User: What are some {level['level']} level English vocabulary words?\nAssistant:",
            "target": f"Some {level['level']} level English vocabulary words include: {', '.join(level['words'])}. {level['explanation']}"
        })
        
        # Synonym exercise
        fluency_data.append({
            "type": "vocabulary",
            "prompt": f"User: Give me synonyms for the word \"{level['words'][0]}\".\nAssistant:",
            "target": f"Synonyms for \"{level['words'][0]}\" include: {', '.join(level['words'][1:3])}. Using varied synonyms helps make your writing more interesting and precise."
        })
    
    console.print(f"[green]Created {len(fluency_data)} language training examples[/green]")
    return fluency_data

# Main entry point - THIS SHOULD BE THE LAST THING IN THE FILE
if __name__ == "__main__":
    # Simple argument parsing
    import argparse
    import sys
    import json  # Add import for JSON parsing
    
    # Remove the problematic forward reference - it's causing a circular import
    # The function is already defined above, so we can use it directly
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train or test a language model')
    parser.add_argument('--further_training', action='store_true', 
                        help='Continue training from the final model with improved tokenization and settings')
    parser.add_argument('--rl', action='store_true',
                        help='Continue training from the final enhanced model using reinforcement learning')
    parser.add_argument('--supervised', action='store_true', 
                        help='Run supervised training mode')
    parser.add_argument('--model_path', type=str, 
                        default="D:/ttm/model/3bmodel/t/final_model_combined.pt", 
                        help='Model path for training')
    parser.add_argument('--new_layers', type=int, default=12, 
                        help='Number of layers for expanded model')
    parser.add_argument('--new_embd', type=int, default=768, 
                        help='Embedding dimension for the model')
    parser.add_argument('--new_heads', type=int, default=12, 
                        help='Number of attention heads for the model')
    parser.add_argument('--block_size', type=int, default=256, 
                        help='Context window size (sequence length)')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for training')
    parser.add_argument('--grad_accum', type=int, default=64, 
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=2, 
                        help='Number of epochs for training')
    parser.add_argument('--thinking', action='store_true', 
                        help='Enable thinking capabilities')
    parser.add_argument('--use_lora', action='store_true',
                        help='Enable LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--use_deepspeed', action='store_true',
                        help='Enable DeepSpeed for efficient training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--limit_examples', type=int, default=100,
                        help='Limit number of training examples to avoid OOM')
    parser.add_argument('--resume_from', type=str, default='',
                        help='Resume training from a checkpoint file')
    parser.add_argument('--skip_data_generation', action='store_true',
                        help='Skip generation of training data and use cached data')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N steps')
    args = parser.parse_args()

    # Update global variables
    FURTHER_TRAINING = args.further_training
    RL_TRAINING = args.rl
    MODEL_PATH = args.model_path
    NEW_LAYERS = args.new_layers
    THINKING_MODE = args.thinking
    USE_LORA = args.use_lora
    USE_DEEPSPEED = args.use_deepspeed
    
    # Update model architecture parameters from command line arguments
    n_layer = args.new_layers
    n_embd = args.new_embd  
    n_head = args.new_heads
    block_size = args.block_size
    batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_accum
    
    # Update derived variables
    HIDDEN_SIZE = n_embd
    NUM_LAYERS = n_layer
    NUM_HEADS = n_head
    MAX_SEQ_LENGTH = block_size
    BATCH_SIZE = batch_size
    GRADIENT_ACCUMULATION_STEPS = gradient_accumulation_steps

    # Set CUDA environment variable for better error reporting
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Create a log file for training history if not already open
    if 'log_file' not in globals() or log_file.closed:
        log_file = open('training_history.log', 'w')

    try:
        if args.supervised:
            console.print("\n[bold blue]=== SUPERVISED MULTI-SKILL TRAINING MODE ACTIVATED ===[/bold blue]")
            console.print(f"[yellow]Will train model using supervised fine-tuning for QA, summarization, and chat skills[/yellow]")
            console.print(f"[yellow]Expanding model from {n_layer} to {args.new_layers} layers[/yellow]")
            console.print(f"[yellow]Checkpoints will be saved to: {os.path.join(CHECKPOINT_DIR, 'supervised')}[/yellow]")
            console.print(f"[yellow]Thinking mode enabled: {THINKING_MODE}[/yellow]")
            
            # Create supervised checkpoint directory
            supervised_dir = os.path.join(CHECKPOINT_DIR, "supervised")
            if not os.path.exists(supervised_dir):
                os.makedirs(supervised_dir)
            
            # Start supervised training
            console.print(f"[green]Using model: {args.model_path}[/green]")
            
            # Direct implementation instead of calling the function
            start_time = time.time()
            
            # Get model path and parameters from args
            model_path = args.model_path
            new_layers = args.new_layers
            new_embd = args.new_embd
            new_heads = args.new_heads
            block_size = args.block_size
            epochs = args.epochs
            
            # Check CUDA availability and memory
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                cuda_device = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(cuda_device)
                free_memory = torch.cuda.get_device_properties(cuda_device).total_memory - torch.cuda.memory_allocated(cuda_device)
                console.print(f"[green]GPU: {gpu_properties.name}, Memory: {free_memory / 1024**3:.2f}GB available[/green]")
            else:
                console.print("[yellow]CUDA not available. Using CPU for training (this will be slow).[/yellow]")
            
            # Initialize model
            if model_path and os.path.exists(model_path):
                try:
                    # Load existing model
                    console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
                    checkpoint = torch.load(model_path, map_location='cuda' if use_cuda else 'cpu')
                    config = checkpoint.get('config', {})
                    n_layer = config.get('n_layer', new_layers)
                    n_embd = config.get('n_embd', new_embd)
                    n_head = config.get('n_head', new_heads)
                    
                    model = GPTLanguageModel()
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    console.print(f"[green]✓ Model loaded successfully with {n_layer} layers, {n_embd} embedding dim, {n_head} heads[/green]")
                except Exception as e:
                    console.print(f"[red]Error loading model: {str(e)}[/red]")
                    console.print("[yellow]Initializing new model instead...[/yellow]")
                    n_layer = new_layers
                    n_embd = new_embd
                    n_head = new_heads
                    model = GPTLanguageModel().to(device)
            else:
                # Create new model
                console.print(f"[cyan]Initializing new model with {new_layers} layers, {new_embd} embedding dim, {new_heads} heads...[/cyan]")
                n_layer = new_layers
                n_embd = new_embd
                n_head = new_heads
                model = GPTLanguageModel().to(device)
                console.print(f"[green]✓ Model initialized successfully[/green]")
            
            # Display model statistics
            total_params = calculate_model_size()
            console.print(f"[cyan]Model size: {total_params / 1e9:.2f}B parameters[/cyan]")
            
            # Set training mode
            model.train()
            
            # Generate training data - two phases
            # Phase 1: English language fluency
            console.print(f"[cyan]Generating English language fluency training data (Phase 1)...[/cyan]")
            fluency_data = create_language_fluency_data()
            console.print(f"[green]✓ Generated {len(fluency_data)} fluency training examples for Phase 1[/green]")
            
            # Phase 2: Domain-specific quality data
            console.print(f"[cyan]Generating domain-specific training data (Phase 2)...[/cyan]")
            domain_data = create_quality_training_data()
            console.print(f"[green]✓ Generated {len(domain_data)} domain-specific examples for Phase 2[/green]")
            
            # Two-phase training approach
            console.print(f"[bold cyan]Starting two-phase training:[/bold cyan]")
            console.print(f"[yellow]Phase 1: English language fluency and grammar training (building vocabulary and writing skills)[/yellow]")
            console.print(f"[yellow]Phase 2: Specialized domain knowledge with identity reinforcement[/yellow]")
            
            # Updated training configuration
            training_config = {
                'target_loss': 0.05,
                'learning_rate': 1e-5,
                'weight_decay': 0.01,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'max_grad_norm': 0.5,
                'fp16': True
            }
            
            # Phase 1: Train on language fluency first
            console.print(f"[bold cyan]Starting Phase 1: Language fluency training...[/bold cyan]")
            phase1_epochs = max(1, epochs // 3)  # Allocate 1/3 of epochs for language training
            model, final_loss = train_with_supervised_finetuning(
                model=model,
                dataset=fluency_data,
                learning_rate=training_config['learning_rate'],
                target_loss=training_config['target_loss'] * 1.5,  # Less strict target for phase 1
                epochs=phase1_epochs,
                batch_size=batch_size,
                phase_name="Phase 1: Language Fluency"
            )
            
            # Save intermediate model after Phase 1
            phase1_path = os.path.join(CHECKPOINT_DIR, f"phase1_language_fluency_{phase1_epochs}epochs.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': vocab_size,
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'n_embd': n_embd,
                    'block_size': block_size
                },
                'phase': "Language Fluency",
                'epochs': phase1_epochs,
                'training_time': time.time() - start_time
            }, phase1_path)
            console.print(f"[green]✓ Phase 1 model saved to {phase1_path}[/green]")
            
            # Phase 2: Train on domain-specific data with identity reinforcement
            console.print(f"[bold cyan]Starting Phase 2: Domain-specific training with identity reinforcement...[/bold cyan]")
            phase2_epochs = epochs - phase1_epochs  # Remaining epochs
            model, final_loss = train_with_supervised_finetuning(
                model=model,
                dataset=domain_data,
                learning_rate=training_config['learning_rate'] * 0.8,  # Slightly lower learning rate for fine-tuning
                target_loss=training_config['target_loss'],
                epochs=phase2_epochs,
                batch_size=batch_size,
                phase_name="Phase 2: Domain Knowledge"
            )
            
            # Save final model after both phases
            final_path = os.path.join(CHECKPOINT_DIR, f"supervised_final_2phase_{epochs}epochs.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': vocab_size,
                    'n_layer': n_layer,
                    'n_head': n_head,
                    'n_embd': n_embd,
                    'block_size': block_size
                },
                'phase': "Complete (Fluency + Domain)",
                'epochs': epochs,
                'training_time': time.time() - start_time
            }, final_path)
            console.print(f"[green]✓ Final model saved to {final_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error during training: {str(e)}[/bold red]")
        traceback.print_exc()
    finally:
        if 'log_file' in globals() and not log_file.closed:
            log_file.close()
            console.print("[dim]Log file closed[/dim]")
