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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
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
NEW_LAYERS = 34

# Model Identity & Core Architecture
n_layer = 24        # Reduced size for better learning
n_head = 24        # Balanced architecture
n_embd = 1024       # Standard embedding size
block_size = 128   # Keep reduced for memory
batch_size = 4     # Increase to help with learning
max_iters = 12000  # Increase training duration
eval_interval = 250  # More frequent evaluation
learning_rate = 6e-5  # Increased from 3e-4
weight_decay = 0.01  # Increased for better regularization
dropout = 0.2      # Increased for better regularization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 40    # Balanced for valid evaluation
gradient_accumulation_steps = 16  # Reduced to update more frequently
warmup_ratio = 0.1  # Longer warmup for stability

# Advanced features
use_flash_attention = False
use_mixed_precision = True
use_gradient_checkpointing = True
use_cpu_offload = False  # Disabled to improve learning consistency
use_deepspeed = False
use_rope = True
use_alibi = False
use_moe = False
num_experts = 4
expert_dropout = 0.1
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05

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
LORA_RANK = lora_rank
LORA_ALPHA = lora_alpha
LORA_DROPOUT = lora_dropout
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
        
        # Ensure we're using the right subset of frequency tensor
        freqs_cis_subset = freqs_cis[:seq_len]
        
        # Reshape for complex manipulation
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
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
        
        # Use mixed precision for initialization to save memory
        with torch.amp.autocast('cuda', enabled=use_mixed_precision) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_mixed_precision):
            # Token and position embeddings
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            if not use_rope and not use_alibi:
                self.position_embedding_table = nn.Embedding(block_size, n_embd)
            
            # Transformer blocks
            print("Creating transformer blocks...")
            blocks = []
            for i in tqdm(range(n_layer), desc="Initializing layers"):
                blocks.append(Block(n_embd, n_head=n_head))
            self.blocks = nn.ModuleList(blocks)
            
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Apply better initialization
        print("Applying weight initialization...")
        self.apply(self._init_weights)
        
        # Move model to device after initialization to avoid OOM during loading
        self.to(device)
        
        # Additional attributes for clarity
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
    
    def _init_weights(self, module):
        """Better weight initialization for stability and faster convergence"""
        if isinstance(module, nn.Linear):
            # Scaled initialization for better gradient flow
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        # Get device and dtype for consistency
        device, dtype = idx.device, idx.dtype

        # Get sequence dimensions, handling both 1D and 2D inputs
        if idx.dim() == 1:
            # If input is a single sequence, add batch dimension
            idx = idx.unsqueeze(0)
            if targets is not None:
                targets = targets.unsqueeze(0)
        
        B, T = idx.size()
        
        # Token embeddings
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        
        # Add positional embeddings if not using rotary embeddings
        if not use_rope and not use_alibi:
            # Position is limited to block_size
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
            pos_emb = self.position_embedding_table(pos)  # (1,T,C)
            x = token_emb + pos_emb  # (B,T,C)
        else:
            x = token_emb  # Already has rotary embeddings included via attention

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Calculate logits
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        loss = None
        if targets is not None:
            # Reshape logits to match expected input for cross_entropy
            # Use reshape instead of view for safety with non-contiguous tensors
            logits_reshaped = logits.reshape(-1, logits.size(-1))  # (B*T,vocab_size)
            targets_reshaped = targets.reshape(-1)  # (B*T)
            
            # Calculate cross entropy loss
            loss = F.cross_entropy(logits_reshaped, targets_reshaped, ignore_index=-1)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        """Generate text with various sampling methods"""
        # idx is (B, T) tensor of indices in current context
        self.eval()  # Set to evaluation mode
        
        # Hold generated tokens for all sequences in batch
        generated_sequences = [idx[i].tolist() for i in range(idx.size(0))]
        
        for _ in range(max_new_tokens):
            # If context is too long, crop it to the last block_size tokens
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            
            # Forward pass to get logits
            with torch.no_grad():
                logits, _ = self(idx_cond)
            
            # Take just the last time step
            logits = logits[:, -1, :] # (B, vocab_size)
            
            # Apply temperature
            logits = logits / max(temperature, 1e-5)
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Store tokens for each batch item
            for i, next_token in enumerate(idx_next):
                generated_sequences[i].append(next_token.item())
        
        self.train()  # Set back to training mode
        return generated_sequences

    def _apply_repetition_penalty(self, logits, generated_tokens, penalty):
        """Apply repetition penalty in a safe way with bounds checking"""
        try:
            vocab_size = logits.size(-1)
            
            for b in range(logits.size(0)):
                for token_idx in generated_tokens[b]:
                    # Safety check to avoid index errors
                    if 0 <= token_idx < vocab_size:
                        logits[b, token_idx] /= penalty
        except Exception as e:
            # If anything goes wrong, just skip applying the penalty
            print(f"Warning: Could not apply repetition penalty: {e}")
            pass

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

def test_model_quality(model, stage_name, test_iteration, current_step=0, save_checkpoint=True):
    print(f"\n=== Testing Model Quality (Stage: {stage_name}, Test #{test_iteration}) ===")
    
    # Make sure model is in eval mode and on the correct device
    model.eval()
    model = model.to(device)
    
    # Optimize memory before testing
    torch.cuda.empty_cache()
    gc.collect()
    
    # Test prompts - use more relevant prompts for further training
    if FURTHER_TRAINING:
        test_prompts = [
            "Explain quantum entanglement and its applications in quantum computing.",
            "Analyze the impact of artificial intelligence on employment and the global workforce.",
            "How do transformer neural networks work and what makes them effective for NLP tasks?",
            "Discuss the most effective approaches to mitigate climate change at individual and policy levels.",
            "What are the key features of your model architecture and how do they contribute to its performance?"
        ]
    else:
        test_prompts = [
            "Explain quantum entanglement and its applications.",
            "Analyze the impact of AI on the global workforce.",
            "Explain how transformer neural networks work.",
            "Discuss climate change mitigation approaches."
        ]
    
    # Set reasonable limits for generation
    max_tokens = 150 if FURTHER_TRAINING else 100  # Generate more tokens in further training mode
    
    results = []
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}/{len(test_prompts)}: Generating response...")
            try:
                # Encode the prompt
                encoded_prompt = encode(prompt)
                encoded = torch.tensor(encoded_prompt, dtype=torch.long).to(device).unsqueeze(0)
                
                # Record prompt length for proper trimming later
                prompt_len = len(encoded_prompt)
                
                # Use mixed precision during generation - use modern autocast API
                with torch.amp.autocast('cuda', enabled=use_mixed_precision) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_mixed_precision):
                    # Generate with improved settings
                    temperature = 0.8  # Better temperature for more focused output
                    top_k = 40
                    top_p = 0.9
                    
                    # Use a safer repetition penalty value
                    repetition_penalty = 1.1  # Reduced from 1.2 for more stable generation
                    
                    try:
                        response = model.generate(
                            encoded,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty
                        )[0]
                    except Exception as gen_error:
                        # If generation fails with repetition penalty, try again without it
                        console.print(f"[yellow]Generation failed with repetition penalty: {gen_error}. Retrying without repetition penalty.[/yellow]")
                        response = model.generate(
                            encoded,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=1.0  # Disable repetition penalty
                        )[0]
                
                # Decode the response and only keep generated part
                full_text = decode(response.tolist())
                
                # Try to separate prompt from response
                # This is more complex with BPE tokenization
                if FURTHER_TRAINING:
                    # For BPE we might need a different approach as tokens don't map 1:1 with words
                    try:
                        # Try to find where the prompt ends and response begins
                        prompt_decoded = decode(encoded_prompt)
                        if full_text.startswith(prompt_decoded):
                            response_text = full_text[len(prompt_decoded):]
                        else:
                            # Fallback - just trim out approximately the right number of tokens
                            words = full_text.split()
                            response_text = " ".join(words[min(prompt_len, len(words)-1):])
                    except:
                        # Last resort fallback
                        response_text = full_text
                else:
                    # Word tokenization approach
                    words = full_text.split()
                    response_text = " ".join(words[prompt_len:])
                
                # Memory cleanup
                del encoded, response
                torch.cuda.empty_cache()
                
                # Evaluate response quality
                words = response_text.split()
                length = len(words)
                unique_words = len(set(words)) if words else 0
                vocab_richness = unique_words / length if length > 0 else 0
                
                # Add additional metrics for further training
                if FURTHER_TRAINING:
                    # Check for common placeholder patterns that indicate poor quality
                    placeholder_pattern = re.compile(r'\{[a-z_]+\}')
                    has_placeholders = bool(placeholder_pattern.search(response_text))
                    
                    # Check for repetition patterns
                    repetition_score = 0
                    if length > 10:
                        word_bigrams = [' '.join(words[i:i+2]) for i in range(length-1)]
                        bigram_counts = Counter(word_bigrams)
                        # Count bigrams that appear more than once
                        repetition_score = sum(1 for count in bigram_counts.values() if count > 1) / len(bigram_counts) if bigram_counts else 0
                    
                    quality_metrics = {
                        'word_count': length,
                        'vocabulary_richness': round(vocab_richness, 3),
                        'has_placeholders': has_placeholders,
                        'repetition_score': round(repetition_score, 3)
                    }
                else:
                    quality_metrics = {
                        'word_count': length,
                        'vocabulary_richness': round(vocab_richness, 3)
                    }
                
                results.append({
                    'prompt': prompt,
                    'response': response_text,
                    'metrics': quality_metrics
                })
                
                print(f"Response word count: {length} words")
                print(f"Generated text: {response_text[:200]}...")
                
                # Additional quality metrics for further training
                if FURTHER_TRAINING:
                    print(f"Has placeholders: {'Yes' if has_placeholders else 'No'}")
                    print(f"Repetition score: {repetition_score:.3f} (lower is better)")
                
            except Exception as e:
                print(f"Error generating response: {e}")
                traceback.print_exc()  # Print full traceback for debugging
                results.append({
                    'prompt': prompt,
                    'error': str(e)
                })
                
            # Clean up memory after each prompt
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save test results
    test_dir = os.path.join(CHECKPOINT_DIR, f"test_results_{stage_name.lower()}")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    try:
        with open(os.path.join(test_dir, f'test_{test_iteration}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Test results saved to {test_dir}/test_{test_iteration}.json")
        
        # Also save a checkpoint after testing if requested
        if save_checkpoint and current_step > 0:
            test_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"test_checkpoint_{stage_name.lower()}_{test_iteration}.pt")
            print(f"Saving test checkpoint to {test_checkpoint_path}")
            try:
                torch.save({
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'stage_name': stage_name,
                    'test_iteration': test_iteration,
                    'config': {
                        'n_layer': n_layer,
                        'n_head': n_head,
                        'n_embd': n_embd,
                        'vocab_size': vocab_size
                    }
                }, test_checkpoint_path)
                print(f"Test checkpoint saved successfully")
            except Exception as e:
                print(f"Error saving test checkpoint: {e}")
    except Exception as e:
        print(f"Error saving test results: {e}")
    
    return results

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
    
    # Use a hybrid approach with both tqdm and rich text
    with console.status("[bold green]Preparing training environment...", spinner="dots") as status:
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
def create_bpe_tokenizer(text, vocab_size=8000):
    """Create a BPE tokenizer from the training text"""
    if not USE_BETTER_TOKENIZER:
        console.print("[red]Tokenizers package not available. Cannot create BPE tokenizer.[/red]")
        return None
        
    try:
        console.print(f"[cyan]Training BPE tokenizer with vocab size {vocab_size}...[/cyan]")
        
        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        
        # Configure pre-tokenization and training
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            min_frequency=5  # Increased min frequency for better tokens
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
        
        return tokenizer
    except Exception as e:
        console.print(f"[red]Error creating BPE tokenizer: {str(e)}[/red]")
        traceback.print_exc()
        return None

# Add function to convert tokenizer to encoder/decoder functions
def create_encoder_decoder_from_bpe(tokenizer):
    """Create encoder and decoder functions from BPE tokenizer"""
    
    # Get vocabulary
    vocab = tokenizer.get_vocab()
    ids_to_tokens = {v: k for k, v in vocab.items()}
    
    # Create encode and decode functions
    def encode(s):
        return tokenizer.encode(s).ids
    
    def decode(ids):
        # Convert to list if it's a tensor
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Decode tokens
        return tokenizer.decode(ids)
    
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
    
    # Expert stats if MoE is enabled
    moe_experts = num_experts if use_moe else 0
    expert_params = int(total_params * 0.7) if use_moe else 0
    per_expert_params = int(expert_params / num_experts) if use_moe else 0
    
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
    
    # MoE stats if applicable
    if use_moe:
        moe_table = Table(title="🧠 EXPERT UTILIZATION PREDICTION", box=box.DOUBLE_EDGE)
        moe_table.add_column("Metric", style="cyan")
        moe_table.add_column("Value", justify="right", style="green")
        moe_table.add_column("Notes", style="dim")
        
        moe_table.add_row("Number of Experts", f"{moe_experts}", "")
        moe_table.add_row("Expert Parameters", f"{expert_params:,}", "Estimated 70% of params in experts")
        moe_table.add_row("Parameters per Expert", f"{per_expert_params:,}", "Average")
        moe_table.add_row("Estimated Expert Utilization", "65-80%", "Typical range for MoE models")
        moe_table.add_row("Activation Sparsity", "~90%", "Due to expert gating")
        
        console.print(moe_table)
    
    console.print("=" * 80)
    return total_params

# Move the function definition right before the main block

def create_quality_training_data():
    """
    Create high-quality training data for supervised fine-tuning
    with specific examples for question-answering, summarization, and basic chatting.
    Generate at least 35,000 examples to provide sufficient training data.
    """
    quality_data = []
    
    # Base examples (these are the high-quality manually crafted ones)
    # ===== 1. QUESTION-ANSWERING EXAMPLES =====
    qa_examples = [
        {
            "prompt": "Explain quantum entanglement and its applications in quantum computing.",
            "target": "Quantum entanglement is a physical phenomenon where pairs or groups of particles interact in ways such that the quantum state of each particle cannot be described independently. In quantum computing, entanglement is a crucial resource that enables quantum advantage. It allows quantum bits (qubits) to be correlated in ways that have no classical analog, enabling applications like quantum teleportation, superdense coding, and quantum cryptography. In quantum algorithms such as Shor's algorithm for factoring large numbers, entanglement helps create exponential speedups over classical algorithms."
        },
        {
            "prompt": "How do transformer neural networks work and what makes them effective for NLP tasks?",
            "target": "Transformer neural networks work through a self-attention mechanism that directly models relationships between all words in a sequence, regardless of their distance. The architecture consists of stacked encoder and decoder layers, each containing multi-head self-attention and feed-forward neural networks. Key components include positional encodings (to capture word order) and attention masks. Transformers are effective for NLP tasks because they: 1) Process all tokens in parallel, allowing efficient training; 2) Capture long-range dependencies better than RNNs; 3) Enable transfer learning through pre-training on large text corpora; and 4) Scale well with model and data size, as demonstrated by models like GPT and BERT."
        },
        {
            "prompt": "What is the difference between a CPU and a GPU?",
            "target": "CPUs (Central Processing Units) and GPUs (Graphics Processing Units) differ in architecture and purpose. CPUs have fewer cores (typically 4-16) but each core is more powerful with higher clock speeds, optimized for sequential processing and general computing tasks. They excel at handling complex instructions and are the primary processor for most computer operations. GPUs have many more cores (hundreds or thousands) designed for parallel processing of simpler calculations simultaneously. Originally designed for rendering graphics, GPUs are now also used for tasks like machine learning and cryptocurrency mining due to their parallel processing capabilities. While CPUs are versatile for diverse tasks, GPUs are specialized for computationally intensive tasks that can be broken into parallel workloads."
        },
        {
            "prompt": "How does photosynthesis work in plants?",
            "target": "Photosynthesis is the process by which plants convert light energy into chemical energy stored as glucose. It occurs in two main stages: the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, which take place in the thylakoid membranes of chloroplasts, chlorophyll pigments capture sunlight energy. This energy is used to split water molecules, releasing oxygen as a byproduct and generating ATP and NADPH as energy carriers. In the Calvin cycle, which occurs in the stroma, carbon dioxide from the atmosphere is incorporated into existing organic molecules using the ATP and NADPH from the light reactions, ultimately producing glucose. This glucose serves as both an energy source and a building block for plant growth and development."
        },
        {
            "prompt": "What is the theory of relativity?",
            "target": "The theory of relativity, developed by Albert Einstein, consists of two major frameworks: Special Relativity (1905) and General Relativity (1915). Special Relativity states that the laws of physics are the same for all non-accelerating observers, and that the speed of light in a vacuum is constant regardless of the observer's motion. It introduced revolutionary concepts like the space-time continuum, mass-energy equivalence (E=mc²), and time dilation. General Relativity extends these principles to accelerating reference frames and gravity, describing gravity not as a force but as a curvature of space-time caused by mass and energy. It predicted phenomena like gravitational waves, black holes, and the bending of light around massive objects, all of which have been experimentally verified."
        }
    ]
    quality_data.extend(qa_examples)
    
    # ===== 2. SUMMARIZATION EXAMPLES =====
    summarization_examples = [
        {
            "prompt": "Summarize the following text about climate change: 'Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures. Examples of greenhouse gas emissions that are causing climate change include carbon dioxide and methane. These come from using gasoline for driving a car or coal for heating a building, for example. Clearing land and forests can also release carbon dioxide. Landfills for garbage are a major source of methane emissions. Energy, industry, transport, buildings, agriculture and land use are among the main emitters.'",
            "target": "Climate change involves long-term shifts in temperatures and weather patterns. While natural causes exist, human activities, particularly burning fossil fuels since the 1800s, have become the primary driver. Fossil fuel combustion creates greenhouse gases like carbon dioxide and methane that trap heat around Earth, raising temperatures. These emissions come from various sources including transportation, building heating, deforestation, and landfills. The main contributing sectors include energy, industry, transport, buildings, agriculture, and land use."
        },
        {
            "prompt": "Summarize the following text about artificial intelligence: 'Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by humans or by other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs. AI applications include advanced web search engines, recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars, generative or creative tools, automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require \"intelligence\" are often removed from the definition of AI, a phenomenon known as the AI effect.'",
            "target": "Artificial intelligence (AI) refers to machine-demonstrated intelligence involving perception, synthesis, and inference of information. It handles tasks like speech recognition, computer vision, and language translation. Practical AI applications include advanced search engines, recommendation systems (YouTube, Amazon, Netflix), voice assistants (Siri, Alexa), self-driving vehicles, creative tools, automated decision-making, and strategic gaming systems (chess, Go). Interestingly, as machines advance, capabilities once considered signs of intelligence are often redefined and no longer classified as AI—a phenomenon called the AI effect."
        },
        {
            "prompt": "Summarize the following article about quantum computing: 'Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. Today's computers encode information in bits — binary 0s and 1s. Quantum computers, however, use quantum bits, or qubits, which can exist in multiple states simultaneously due to a property called superposition. Additionally, qubits can be linked together through quantum entanglement, allowing groups of qubits to express higher degrees of correlation than is possible with classical bits. These properties enable quantum computers to perform certain calculations exponentially faster than classical computers. Potential applications include cryptography, optimization problems, drug discovery, materials science, and artificial intelligence. Companies like IBM, Google, Microsoft, and startups like Rigetti and IonQ are currently developing quantum computing hardware and software. While still in early stages with limitations like qubit coherence time and error rates, quantum computing has the potential to revolutionize many fields by solving previously intractable problems.'",
            "target": "Quantum computing uses quantum mechanics to solve complex problems beyond classical computers' capabilities. Unlike traditional bits (0s and 1s), quantum computers use qubits that can exist in multiple states simultaneously through superposition and can be linked via quantum entanglement. These properties allow quantum computers to perform certain calculations exponentially faster. Potential applications span cryptography, optimization, drug discovery, materials science, and AI. Major tech companies and startups are actively developing quantum hardware and software. Despite current limitations in qubit coherence time and error rates, quantum computing shows promise for revolutionizing multiple fields by tackling previously unsolvable problems."
        },
        {
            "prompt": "Summarize this text about blockchain technology: 'Blockchain is a distributed ledger technology (DLT) that enables the recording of transactions and the tracking of assets in a peer-to-peer network. An asset can be tangible (a house, car, cash, land) or intangible (intellectual property, patents, copyrights, branding). On a blockchain network, anything of value can be tracked and traded, reducing risk and cutting costs for all involved. Blockchain is ideal for delivering information because it provides immediate, shared, and completely transparent information stored on an immutable ledger that can be accessed only by permissioned network members. A blockchain network can track orders, payments, accounts, production and much more. And because members share a single view of the truth, you can see all details of a transaction end to end, giving you greater confidence, as well as new efficiencies and opportunities.'",
            "target": "Blockchain is a distributed ledger technology enabling peer-to-peer transaction recording and asset tracking. It can manage both tangible assets (like property and cash) and intangible ones (such as intellectual property). The technology provides immediate, transparent information on an immutable ledger accessible only to authorized network members. Blockchain systems can track various elements including orders, payments, accounts, and production processes. By providing all members with a single, shared view of transactions from end to end, blockchain reduces risk, cuts costs, builds confidence, and creates new efficiencies and opportunities."
        }
    ]
    quality_data.extend(summarization_examples)
    
    # ===== 3. BASIC CHATTING EXAMPLES =====
    chat_examples = [
        {
            "prompt": "Hi there, how are you doing today?",
            "target": "Hello! I'm doing well, thank you for asking. I'm ready to assist you with any questions or tasks you might have. How can I help you today?"
        },
        {
            "prompt": "Can you tell me a bit about yourself?",
            "target": "I'm an AI language model designed to assist with various tasks like answering questions, providing information, helping with creative writing, and engaging in conversations. I've been trained on a diverse range of texts to understand and generate human-like responses. While I don't have personal experiences or consciousness, I aim to be helpful, accurate, and ethical in all my interactions. Is there something specific you'd like to know about my capabilities or how I can assist you?"
        },
        {
            "prompt": "I'm feeling a bit stressed about my upcoming exam.",
            "target": "I understand exam stress can be challenging. Consider these strategies: 1) Create a study schedule to manage your time efficiently, 2) Take regular breaks to maintain focus, 3) Get enough sleep, as it's crucial for memory consolidation, 4) Try relaxation techniques like deep breathing when feeling overwhelmed, and 5) Remember that some nervousness is normal and can actually improve performance. Breaking your study material into smaller, manageable sections might also help. You've got this! Is there a particular subject you're concerned about?"
        },
        {
            "prompt": "What are some good movies to watch this weekend?",
            "target": "For your weekend viewing, consider these diverse options: 'Everything Everywhere All at Once' (award-winning sci-fi adventure), 'Top Gun: Maverick' (action-packed sequel), 'The Shawshank Redemption' (timeless classic), 'Parasite' (thought-provoking thriller), or 'Coco' (heartwarming animation). For something recent, 'Dune' offers spectacular sci-fi, while 'Knives Out' delivers an entertaining mystery. Comedy fans might enjoy 'The Grand Budapest Hotel' or 'Superbad.' What genres do you typically prefer? That would help me suggest movies more tailored to your taste."
        },
        {
            "prompt": "I just got a new puppy! Any advice for a first-time dog owner?",
            "target": "Congratulations on your new puppy! Here's some essential advice: 1) Schedule a veterinary check-up and follow vaccination guidelines, 2) Create a consistent routine for feeding, potty breaks, and sleep, 3) Start basic training early using positive reinforcement, 4) Puppy-proof your home by removing hazards and providing appropriate chew toys, 5) Socialize your puppy with different people, animals, and environments, 6) Be patient with house training—consistency is key, 7) Invest in quality food appropriate for puppies, and 8) Give plenty of love and attention to build a strong bond. Remember, the first few months are crucial for developing good habits. What breed is your puppy?"
        },
        {
            "prompt": "Could you help me plan a one-week trip to Japan?",
            "target": "For a one-week Japan trip, I recommend this itinerary: Days 1-3: Tokyo (explore Shibuya, Shinjuku, visit Senso-ji Temple, Tokyo Skytree, and experience Akihabara). Day 4: Day trip to Hakone or Kamakura from Tokyo. Days 5-7: Kyoto (visit Fushimi Inari Shrine, Kinkaku-ji, Arashiyama Bamboo Grove, and Gion district). Consider purchasing a 7-day Japan Rail Pass for transportation. April (cherry blossoms) and November (fall colors) offer beautiful scenery. Pack comfortable walking shoes, a portable Wi-Fi device, and download translation apps. For budget planning, prepare around $120-150 per day excluding flights. Would you like more specific recommendations for any of these locations?"
        }
    ]
    quality_data.extend(chat_examples)
    
    # Add metadata examples if available
    try:
        console.print("[yellow]Loading metadata examples...[/yellow]")
        with open("D:/ttm/model/3bmodel/metadata.txt", "r", encoding="utf-8") as f:
            metadata_lines = f.readlines()
            
        # Generate QA pairs from metadata
        metadata_examples = []
        for i in range(0, len(metadata_lines) - 10, 10):  # Process chunks of 10 lines
            chunk = " ".join(metadata_lines[i:i+10]).strip()
            if len(chunk) > 50:  # Only meaningful chunks
                # Create different question types for each metadata chunk
                questions = [
                    f"What does the following mean? {chunk[:100]}",
                    f"Explain this metadata: {chunk[:100]}",
                    f"Extract key information from this: {chunk[:100]}",
                    f"Summarize this data: {chunk[:100]}"
                ]
                for question in questions:
                    metadata_examples.append({
                        "prompt": question,
                        "target": f"This metadata describes {chunk[:200]}... The key points are related to model parameters, training data, or configuration settings. This information can be used for tracking model provenance and ensuring proper documentation of AI systems."
                    })
        
        # Add metadata examples
        console.print(f"[green]Created {len(metadata_examples)} examples from metadata[/green]")
        quality_data.extend(metadata_examples)
    except Exception as e:
        console.print(f"[red]Failed to load metadata examples: {str(e)}[/red]")
    
    # Generate synthetic examples based on templates to reach 35000+ examples
    console.print("[yellow]Generating synthetic examples to reach target count...[/yellow]")
    
    # Template-based generation
    qa_templates = [
        {"question": "What is {topic}?", "answer": "{topic} is a {description}. It {additional_info}."},
        {"question": "How does {topic} work?", "answer": "{topic} works by {process}. This enables {benefit}."},
        {"question": "Why is {topic} important?", "answer": "{topic} is important because {reason}. This impacts {domain} by {impact}."},
        {"question": "Explain the concept of {topic}.", "answer": "The concept of {topic} refers to {explanation}. It's characterized by {characteristics}."},
        {"question": "What are the applications of {topic}?", "answer": "{topic} has several applications including {app1}, {app2}, and {app3}. It's particularly valuable in {field}."},
        {"question": "Compare {topic1} and {topic2}.", "answer": "While {topic1} is {desc1}, {topic2} is {desc2}. The key differences include {diff1} and {diff2}. However, they both {similarity}."},
    ]
    
    summary_templates = [
        {"text": "The article discusses {topic}. It covers {point1}, {point2}, and {point3}. The conclusion is that {conclusion}.", 
         "summary": "{topic} encompasses {point1} and {point2}. The article concludes {conclusion}."},
        {"text": "Research on {topic} has shown that {finding1} and {finding2}. This contradicts previous assumptions that {assumption}. Future work should focus on {future}.",
         "summary": "New research reveals {finding1} about {topic}, challenging the assumption of {assumption}. Future research will explore {future}."},
    ]
    
    chat_templates = [
        {"message": "I'm interested in learning about {topic}. Can you recommend some resources?", 
         "response": "I'd be happy to recommend resources for learning about {topic}! Here are some great starting points: {resource1}, {resource2}, and {resource3}. {additional_tip}"},
        {"message": "What do you think about {topic}?", 
         "response": "When it comes to {topic}, there are several perspectives to consider. {perspective1} suggests {view1}, while {perspective2} indicates {view2}. Personally, I think it's important to consider {consideration}."},
        {"message": "I'm having trouble with {problem}. Do you have any advice?", 
         "response": "I'm sorry to hear you're struggling with {problem}. Here are some strategies that might help: First, try {strategy1}. If that doesn't work, consider {strategy2}. Many people also find that {strategy3} can be effective. Remember that {encouragement}."},
    ]
    
    # Content elements for templates
    topics = ["machine learning", "deep learning", "natural language processing", "computer vision", "reinforcement learning",
             "artificial intelligence", "neural networks", "transformer models", "generative AI", "robotics",
             "quantum computing", "blockchain technology", "cloud computing", "edge computing", "Internet of Things",
             "cybersecurity", "data science", "big data", "data privacy", "ethical AI",
             "virtual reality", "augmented reality", "metaverse", "cryptocurrency", "web development",
             "mobile development", "game development", "3D printing", "biotechnology", "nanotechnology",
             "renewable energy", "climate science", "space exploration", "astrophysics", "genomics",
             "psychology", "philosophy", "economics", "political science", "sociology",
             "history", "literature", "art", "music", "film",
             "nutrition", "exercise science", "medicine", "public health", "mental health"]
    
    descriptions = ["field of study", "technology", "concept", "methodology", "framework", "paradigm", "approach", "discipline", "system", "technique"]
    
    processes = ["processing data in parallel", "analyzing patterns in large datasets", "transforming input information",
                "optimizing for specific outcomes", "encoding and decoding information", "leveraging statistical principles",
                "applying mathematical transformations", "iteratively improving performance", "abstracting complex systems",
                "combining multiple specialized components"]
    
    benefits = ["improved efficiency", "greater accuracy", "enhanced user experiences", "reduced costs", 
               "increased accessibility", "better decision-making", "faster processing times", "more reliable outcomes",
               "greater flexibility", "improved scalability"]
    
    reasons = ["it solves previously intractable problems", "it provides unique insights", "it enables automation of complex tasks",
              "it improves human capabilities", "it creates new opportunities", "it addresses critical limitations",
              "it optimizes resource utilization", "it enables personalization at scale", "it improves safety and security",
              "it facilitates collaboration across domains"]
    
    domains = ["healthcare", "finance", "education", "transportation", "communication", "manufacturing", "entertainment",
              "agriculture", "energy", "retail", "public services", "scientific research", "social media", "gaming"]
    
    impacts = ["reducing costs", "saving time", "improving accuracy", "enhancing user experiences", "enabling new capabilities",
              "facilitating better decision-making", "creating new business models", "advancing scientific understanding",
              "democratizing access", "fostering innovation"]
    
    # Function to generate example from template
    def fill_template(template, **kwargs):
        result = template.copy()
        for key, value in result.items():
            for kw, replacement in kwargs.items():
                placeholder = "{" + kw + "}"
                if placeholder in value:
                    result[key] = result[key].replace(placeholder, replacement)
        return result
    
    # Generate QA examples - INCREASED COUNT
    synthetic_qa = []
    for i in range(12000):  # Increased from 5000
        template = random.choice(qa_templates)
        topic = random.choice(topics)
        
        params = {
            "topic": topic,
            "topic1": random.choice(topics),
            "topic2": random.choice(topics),
            "description": random.choice(descriptions),
            "process": random.choice(processes),
            "benefit": random.choice(benefits),
            "reason": random.choice(reasons),
            "domain": random.choice(domains),
            "impact": random.choice(impacts),
            "explanation": f"a {random.choice(descriptions)} that {random.choice(processes)}",
            "characteristics": f"its {random.choice(benefits)} and ability to {random.choice(impacts)}",
            "app1": f"{random.choice(domains)} optimization",
            "app2": f"enhanced {random.choice(domains)} analysis",
            "app3": f"{random.choice(domains)} automation",
            "field": random.choice(domains),
            "desc1": f"focused on {random.choice(processes)}",
            "desc2": f"centered around {random.choice(processes)}",
            "diff1": f"their approach to {random.choice(domains)}",
            "diff2": f"how they handle {random.choice(benefits)}",
            "similarity": f"aim to {random.choice(impacts)}"
        }
        
        filled = fill_template(template, **params)
        synthetic_qa.append({"prompt": filled["question"], "target": filled["answer"]})
    
    # Generate summarization examples - INCREASED COUNT
    synthetic_summaries = []
    for i in range(10000):  # Increased from 5000
        template = random.choice(summary_templates)
        topic = random.choice(topics)
        
        params = {
            "topic": topic,
            "point1": f"the importance of {random.choice(benefits)}",
            "point2": f"how it relates to {random.choice(domains)}",
            "point3": f"recent advances in {random.choice(processes)}",
            "conclusion": f"further research in {topic} will lead to {random.choice(impacts)}",
            "finding1": f"{topic} can improve {random.choice(benefits)}",
            "finding2": f"it works best when combined with {random.choice(topics)}",
            "assumption": f"{topic} was too complex for practical applications",
            "future": f"integrating {topic} with {random.choice(topics)}"
        }
        
        filled = fill_template(template, **params)
        synthetic_summaries.append({"prompt": filled["text"], "target": filled["summary"]})
    
    # Generate chat examples - INCREASED COUNT
    synthetic_chats = []
    for i in range(10000):  # Increased from 5000
        template = random.choice(chat_templates)
        topic = random.choice(topics)
        problem = random.choice([
            "learning a new programming language", 
            "understanding complex concepts", 
            "staying motivated", 
            "managing time effectively",
            "finding reliable information online",
            "dealing with technical errors",
            "choosing between different technologies",
            "keeping up with rapid technological changes",
            "balancing theory and practice",
            "developing good study habits"
        ])
        
        params = {
            "topic": topic,
            "resource1": f"the official documentation for {topic}",
            "resource2": f"online courses on platforms like Coursera or edX about {topic}",
            "resource3": f"community forums where experts discuss {topic}",
            "additional_tip": f"Remember that practical application of {topic} concepts is key to mastery.",
            "perspective1": f"proponents of traditional approaches",
            "perspective2": f"advocates for innovative methods",
            "view1": f"that {topic} should be approached systematically",
            "view2": f"that experimentation is essential when learning {topic}",
            "consideration": f"balance theoretical understanding with practical applications of {topic}",
            "problem": problem,
            "strategy1": f"breaking down {problem} into smaller, manageable steps",
            "strategy2": f"finding a community or mentor who can provide guidance",
            "strategy3": f"connecting it to familiar concepts like {random.choice(topics)}",
            "encouragement": f"learning {topic} takes time, but it's worth the effort for the {random.choice(benefits)} it provides"
        }
        
        filled = fill_template(template, **params)
        synthetic_chats.append({"prompt": filled["message"], "target": filled["response"]})
    
    # Add technical Q&A pairs for additional diversity
    technical_qa = []
    for i in range(3000):
        prog_languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript", "Ruby", "PHP", "Swift"]
        frameworks = ["React", "TensorFlow", "PyTorch", "Django", "Flask", "Angular", "Vue.js", "Spring Boot", "Express.js", "FastAPI"]
        concepts = ["Recursion", "Object-Oriented Programming", "Functional Programming", "Memory Management", "Concurrency", "API Design", "Database Indexing", "Authentication", "Encryption", "Microservices"]
        
        qa_type = random.choice(["language", "framework", "concept"])
        
        if qa_type == "language":
            lang = random.choice(prog_languages)
            q_templates = [
                f"How do you implement a {random.choice(concepts).lower()} in {lang}?",
                f"What are the best practices for error handling in {lang}?",
                f"Explain how to work with asynchronous code in {lang}.",
                f"What's the difference between {lang} and {random.choice([l for l in prog_languages if l != lang])}?",
                f"How do you optimize performance in {lang} applications?"
            ]
            question = random.choice(q_templates)
            answer = f"When working with {lang}, it's important to understand its unique approach to {question.split('in {lang}')[0].strip().lower() if 'in {lang}' in question else random.choice(concepts).lower()}. First, you should consider the language's paradigm and design philosophy. {lang} provides several built-in tools and libraries that can help you accomplish this task efficiently. For best results, follow the community's established patterns and conventions while maintaining code readability and testability. Examples of good {lang} code typically emphasize {random.choice(['simplicity', 'expressiveness', 'type safety', 'performance', 'readability'])}."
        
        elif qa_type == "framework":
            framework = random.choice(frameworks)
            q_templates = [
                f"How do you set up a new project with {framework}?",
                f"What are the key components of a {framework} application?",
                f"Explain the routing system in {framework}.",
                f"How does state management work in {framework}?",
                f"What are some common performance optimizations for {framework}?"
            ]
            question = random.choice(q_templates)
            answer = f"{framework} is a powerful tool for {random.choice(['web development', 'data science', 'building APIs', 'creating user interfaces', 'machine learning'])}. When working with {question.split('in {framework}')[0].strip().lower() if 'in {framework}' in question else 'this framework'}, it's important to understand its core principles and architecture. The framework provides a structured approach that balances flexibility with convention. Best practices include organizing your code according to the framework's patterns, leveraging built-in features rather than reinventing solutions, and staying updated with the ecosystem as it evolves."
        
        else:  # concept
            concept = random.choice(concepts)
            q_templates = [
                f"Can you explain {concept} in simple terms?",
                f"How is {concept} implemented in modern systems?",
                f"What are the advantages and disadvantages of {concept}?",
                f"How has {concept} evolved over time?",
                f"When should developers use {concept} in their architecture?"
            ]
            question = random.choice(q_templates)
            answer = f"{concept} is a fundamental principle in computer science and software engineering. At its core, it involves {random.choice(['organizing code', 'managing resources', 'structuring data', 'processing information', 'designing systems'])} in a way that {random.choice(['maximizes efficiency', 'improves maintainability', 'enhances security', 'optimizes performance', 'simplifies complexity'])}. Understanding {concept} helps developers create more robust and scalable applications. The key aspects to consider include the appropriate context for application, implementation trade-offs, and how it integrates with other architectural patterns. Modern approaches to {concept} often emphasize {random.choice(['modularity', 'testability', 'separation of concerns', 'consistency', 'abstraction'])}."
        
        technical_qa.append({"prompt": question, "target": answer})
    
    # Add the synthetic examples to main data
    quality_data.extend(synthetic_qa)
    quality_data.extend(synthetic_summaries)
    quality_data.extend(synthetic_chats)
    quality_data.extend(technical_qa)
    
    # Final count
    print(f"Created {len(quality_data)} high-quality training examples")
    console.print(f"[bold green]Successfully generated {len(quality_data)} training examples![/bold green]")
    console.print(f"[yellow]Breakdown: {len(qa_examples) + len(synthetic_qa) + len(technical_qa)} QA, {len(summarization_examples) + len(synthetic_summaries)} summarization, {len(chat_examples) + len(synthetic_chats)} chat examples[/yellow]")
    
    return quality_data

# Move use_supervised_finetuning function before main entry point
def use_supervised_finetuning(model_path=None, new_layers=48):
    """
    Perform supervised fine-tuning focusing on three skills:
    1. Question-answering
    2. Summarization
    3. Chatting
    """
    global n_layer
    
    # Define the training function locally to avoid import issues
    def train_model(model, dataset, epochs=3, batch_size=8, learning_rate=5e-5, stage_name="supervised"):
        """
        Train the model using supervised fine-tuning
        """
        console.print(f"[bold yellow]Starting supervised training with {len(dataset)} examples for {epochs} epochs[/bold yellow]")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, "supervised")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Mixed precision training
        use_fp16 = torch.cuda.is_available()
        if hasattr(torch.amp, 'GradScaler'):  # PyTorch 2.0+ API
            scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
        
        # Dataset preprocessing - convert to tokenized format
        tokenized_dataset = []
        console.print("[cyan]Tokenizing dataset...[/cyan]")
        
        max_context_length = 512  # Maximum context length for training
        
        for item in dataset:
            if isinstance(item, dict) and "prompt" in item and ("target" in item or "response" in item):
                prompt = item["prompt"]
                response = item.get("target", item.get("response", ""))
                
                # Format as instruction-response
                full_text = f"Q: {prompt}\nA: {response}"
                
                # Tokenize
                tokens = encode(full_text)
                
                # Skip if too long
                if len(tokens) > max_context_length:
                    # Truncate to fit
                    tokens = tokens[:max_context_length]
                
                tokenized_dataset.append(tokens)
        
        console.print(f"[green]Processed {len(tokenized_dataset)} examples[/green]")
        
        # Training loop
        from tqdm import tqdm
        
        for epoch in range(epochs):
            # Shuffle dataset
            random.shuffle(tokenized_dataset)
            
            # Training loop with progress bar
            total_loss = 0
            num_batches = 0
            
            with tqdm(total=len(tokenized_dataset)//batch_size, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for i in range(0, len(tokenized_dataset), batch_size):
                    batch = tokenized_dataset[i:i+batch_size]
                    
                    # Skip incomplete batches
                    if len(batch) < batch_size:
                        continue
                    
                    # Create tensors
                    batch_tensors = []
                    for sequence in batch:
                        # Convert to tensor
                        tensor = torch.tensor(sequence, dtype=torch.long)
                        batch_tensors.append(tensor)
                    
                    # Pad sequences to the same length
                    max_len = max(len(t) for t in batch_tensors)
                    padded_batch = []
                    
                    for tensor in batch_tensors:
                        if len(tensor) < max_len:
                            padding = torch.zeros(max_len - len(tensor), dtype=torch.long)
                            padded_tensor = torch.cat([tensor, padding])
                        else:
                            padded_tensor = tensor
                        padded_batch.append(padded_tensor)
                    
                    # Stack into a single tensor
                    x = torch.stack(padded_batch).to(device)
                    
                    # Prepare inputs and targets for causal language modeling
                    # If sequence is too short, skip
                    if x.size(1) < 2:
                        continue
                    
                    # Inputs are all tokens except the last one
                    inputs = x[:, :-1]
                    # Targets are all tokens except the first one
                    targets = x[:, 1:]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    try:
                        # Use proper autocast API with PyTorch version check
                        if hasattr(torch.amp, 'autocast'):  # PyTorch 2.0+ API
                            with torch.amp.autocast('cuda', enabled=use_fp16):
                                logits, loss = model(inputs, targets)
                        else:
                            with torch.cuda.amp.autocast(enabled=use_fp16):
                                logits, loss = model(inputs, targets)
                        
                        # Backward pass
                        scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        
                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # Update metrics
                        total_loss += loss.item()
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                        pbar.update(1)
                        
                    except Exception as e:
                        console.print(f"[red]Error in training batch: {str(e)}[/red]")
                        console.print(f"[red]Batch shape: {x.shape}[/red]")
                        # Print traceback for better debugging
                        import traceback
                        console.print(traceback.format_exc())
                        # Clear GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
            
            # Epoch summary
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            console.print(f"[bold green]Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}[/bold green]")
            
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"{stage_name}_epoch_{epoch+1}.pt")
            console.print(f"[yellow]Saving checkpoint to {checkpoint_path}[/yellow]")
            
            # Save model, optimizer state, and training configuration
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'vocab_size': model.vocab_size,
                    'n_layer': model.n_layer,
                    'n_head': model.n_head,
                    'n_embd': model.n_embd,
                },
                'training_stage': stage_name
            }, checkpoint_path)
            
            console.print(f"[green]Checkpoint saved successfully[/green]")
        
        console.print("[bold green]Supervised training complete![/bold green]")
        return model
    
    print("Starting supervised multi-skill fine-tuning...")
    print(f"Loading model from {model_path if model_path else 'scratch'}")
    
    # Create a new expanded model if new_layers is different from default
    if new_layers != n_layer:
        console.print(f"[bold yellow]Expanding model from {n_layer} to {new_layers} layers[/bold yellow]")
        
        # If we have a base model, load it first
        base_model_dict = None
        if model_path:
            console.print(f"Loading base model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            base_model_dict = checkpoint['model_state_dict']
        
        # Temporarily override n_layer to create expanded model
        original_n_layer = n_layer
        n_layer = new_layers
        
        # Create the expanded model
        model = GPTLanguageModel()
        
        # Restore original n_layer
        n_layer = original_n_layer
        
        # If we have a base model, copy matching weights
        if base_model_dict:
            expanded_dict = model.state_dict()
            copied_params = 0
            
            for name, param in base_model_dict.items():
                if name in expanded_dict and expanded_dict[name].shape == param.shape:
                    expanded_dict[name].copy_(param)
                    copied_params += 1
                # Special handling for transformer blocks
                elif "transformer.h." in name:
                    # For transformer blocks, copy up to the available layers
                    layer_num = int(name.split(".")[2])
                    if layer_num < original_n_layer:
                        # Copy to corresponding layer in new model
                        new_name = name
                        if new_name in expanded_dict and expanded_dict[new_name].shape == param.shape:
                            expanded_dict[new_name].copy_(param)
                            copied_params += 1
                        
                        # Also copy to later layers (copy 0->24, 1->25, etc.)
                        if layer_num + original_n_layer < new_layers:
                            # For the second set of layers
                            new_name = name.replace(f".h.{layer_num}.", f".h.{layer_num + original_n_layer}.")
                            if new_name in expanded_dict and expanded_dict[new_name].shape == param.shape:
                                expanded_dict[new_name].copy_(param)
                                copied_params += 1
            
            console.print(f"[green]Copied {copied_params} parameters from base model to expanded model[/green]")
            
            # Load the expanded state dict
            model.load_state_dict(expanded_dict)
        
        model = model.to(device)
        console.print(f"[bold green]Successfully created expanded model with {new_layers} layers[/bold green]")
    
    else:
        # Load model normally
        if model_path:
            console.print(f"Loading model from {model_path}")
            model = GPTLanguageModel()
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            console.print("Creating new model")
            model = GPTLanguageModel()
        
        # Move model to device
        model = model.to(device)
    
    # Create high-quality training data
    train_data = create_quality_training_data()
    
    # Augment with existing data if available
    try:
        existing_data = json.load(open("./data/training_data.json", "r"))
        # Filter only good examples
        good_examples = []
        for item in existing_data:
            if "response" in item and len(item["response"].split()) > 50:
                # Simple heuristic for coherent responses
                words = item["response"].lower().split()
                unique_words = len(set(words))
                if unique_words / len(words) > 0.4:
                    good_examples.append(item)
        
        console.print(f"Added {len(good_examples)} good examples from existing data")
        train_data.extend(good_examples)
    except:
        console.print("No existing training data found, using only quality examples")
    
    # Shuffle data
    random.shuffle(train_data)
    
    # Updated training configuration for larger dataset and larger model
    training_config = {
        'epochs': 3,                        # Reduced from 5 to 3 for larger dataset
        'batch_size': 2,                    # Reduced from 4 to 2 to save memory
        'learning_rate': 2e-5,              # Slightly reduced from 3e-5
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'gradient_accumulation_steps': 8,   # Increased from 4 to 8
        'max_grad_norm': 0.5,               # Added gradient clipping
        'lr_scheduler_type': 'cosine',      # Added cosine schedule
        'fp16': True,                       # Use mixed precision by default
        'max_seq_length': 384               # Cap sequence length
    }
    
    console.print("[bold]Training Configuration:[/bold]")
    for key, value in training_config.items():
        console.print(f"{key}: {value}")
    
    # Train model with a multi-stage approach
    console.print("[bold green]Starting Stage 1: Question-Answering Training[/bold green]")
    # Filter data to focus on QA examples
    qa_data = [item for item in train_data if 
              any(q in item["prompt"].lower() for q in ["what", "how", "why", "explain", "describe", "define"])]
    
    # Keep proper dataset size for memory constraints
    if len(qa_data) > 15000:
        console.print(f"[yellow]Limiting QA dataset to 15,000 examples (from {len(qa_data)}) to manage memory[/yellow]")
        qa_data = random.sample(qa_data, 15000)
    
    model = train_model(
        model=model,
        dataset=qa_data,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        stage_name="qa"
    )
    
    # Save QA checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'n_layer': new_layers,
            'n_head': model.n_head,
            'n_embd': model.n_embd,
        },
        'training_stage': 'qa'
    }, "./models/supervised_qa_model.pt")
    
    console.print("[bold green]Starting Stage 2: Summarization Training[/bold green]")
    # Filter data to focus on summarization examples
    summary_data = [item for item in train_data if 
                  "summarize" in item["prompt"].lower() or 
                  "summary" in item["prompt"].lower()]
    
    # Keep proper dataset size for memory constraints
    if len(summary_data) > 12000:
        console.print(f"[yellow]Limiting summarization dataset to 12,000 examples (from {len(summary_data)}) to manage memory[/yellow]")
        summary_data = random.sample(summary_data, 12000)
    
    model = train_model(
        model=model,
        dataset=summary_data,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'] / 2,  # Lower learning rate for refinement
        stage_name="summarization"
    )
    
    # Save summarization checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'n_layer': new_layers,
            'n_head': model.n_head,
            'n_embd': model.n_embd,
        },
        'training_stage': 'summarization'
    }, "./models/supervised_summary_model.pt")
    
    console.print("[bold green]Starting Stage 3: Chat Training[/bold green]")
    # Filter data to focus on chat examples
    chat_data = [item for item in train_data if 
               not ("summarize" in item["prompt"].lower() or 
                   "summary" in item["prompt"].lower() or
                   any(q in item["prompt"].lower() for q in ["what is", "how does", "why is", "explain", "describe", "define"]))]
    
    # Keep proper dataset size for memory constraints
    if len(chat_data) > 12000:
        console.print(f"[yellow]Limiting chat dataset to 12,000 examples (from {len(chat_data)}) to manage memory[/yellow]")
        chat_data = random.sample(chat_data, 12000)
    
    model = train_model(
        model=model,
        dataset=chat_data,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'] / 3,  # Even lower learning rate for fine adjustments
        stage_name="chat"
    )
    
    # Final combined training to integrate all skills - use a small subset of all data
    console.print("[bold blue]Starting Final Integration Training[/bold blue]")
    
    # Create a balanced subset with examples from each category
    integrated_data = []
    integrated_data.extend(random.sample(qa_data, min(len(qa_data), 2000)))
    integrated_data.extend(random.sample(summary_data, min(len(summary_data), 2000)))
    integrated_data.extend(random.sample(chat_data, min(len(chat_data), 2000)))
    random.shuffle(integrated_data)
    
    model = train_model(
        model=model,
        dataset=integrated_data,  # Use balanced subset
        epochs=1,  # Single integration epoch
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'] / 5,  # Very low learning rate for integration
        stage_name="integration"
    )
    
    # Test the model with some examples
    console.print("\n[bold green]=== Testing Final Model ===[/bold green]")
    test_prompts = [
        "What is machine learning?",
        "Summarize the following text: The transformer architecture has revolutionized natural language processing by enabling parallel processing and capturing long-range dependencies through self-attention mechanisms.",
        "How's the weather today?"
    ]
    
    for prompt in test_prompts:
        console.print(f"[bold]Prompt:[/bold] {prompt}")
        
        # Generate response
        input_ids = encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate with temperature sampling
        with torch.no_grad():
            response_ids = model.generate(
                input_tensor, 
                max_new_tokens=100,
                temperature=0.7,
                top_k=50
            )[0]  # Model returns list of sequences
        
        # Extract only the newly generated tokens
        response_text = decode(response_ids[len(input_ids):])
        
        console.print(f"[bold green]Response:[/bold green] {response_text}")
        console.print("---")
    
    # Save final model
    model_save_path = "./models/supervised_multi_skill_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'n_layer': new_layers,
            'n_head': model.n_head,
            'n_embd': model.n_embd,
        },
        'skills': ['question-answering', 'summarization', 'chatting'],
        'training_info': {
            'dataset_size': len(train_data),
            'qa_examples': len(qa_data),
            'summarization_examples': len(summary_data),
            'chat_examples': len(chat_data),
            'training_config': training_config,
            'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }, model_save_path)
    
    console.print(f"[bold green]Multi-skill model saved to {model_save_path}[/bold green]")
    
    return model

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
    parser.add_argument('--rl', action='store_true', help='Run in RL mode')
    parser.add_argument('--new_layers', type=int, default=48, help='Number of layers for expanded model (works with both RL and supervised training)')
    parser.add_argument('--model_path', type=str, default="D:/ttm/model/3bmodel/t/checkout/final_model_enhanced.pt", help='Model path for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--dataset_size', type=int, default=-1, help='Dataset size for training (-1 for full)')
    parser.add_argument('--test_only', action='store_true', help='Only test the model')
    parser.add_argument('--further_training', action='store_true', help='Run further training on base model')
    parser.add_argument('--supervised', action='store_true', help='Train with supervised learning for QA, summarization, and chat')
    args = parser.parse_args()
    
    # Update global variables
    FURTHER_TRAINING = args.further_training
    RL_TRAINING = args.rl
    MODEL_PATH = args.model_path
    NEW_LAYERS = args.new_layers
    
    # Set CUDA environment variable for better error reporting
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Create a log file for training history if not already open
    if 'log_file' not in globals() or log_file.closed:
        log_file = open('training_history.log', 'w')
    
    # Check for supervised training mode
    if args.supervised:
        console.print("\n[bold blue]=== SUPERVISED MULTI-SKILL TRAINING MODE ACTIVATED ===[/bold blue]")
        console.print(f"[yellow]Will train model using supervised fine-tuning for QA, summarization, and chat skills[/yellow]")
        console.print(f"[yellow]Expanding model from {n_layer} to {args.new_layers} layers[/yellow]")
        console.print(f"[yellow]Checkpoints will be saved to: {os.path.join(CHECKPOINT_DIR, 'supervised')}[/yellow]")
        
        # Create supervised checkpoint directory
        supervised_dir = os.path.join(CHECKPOINT_DIR, "supervised")
        if not os.path.exists(supervised_dir):
            os.makedirs(supervised_dir)
        
        try:
            # Start supervised training
            console.print(f"[green]Using model: {args.model_path}[/green]")
            use_supervised_finetuning(model_path=args.model_path, new_layers=args.new_layers)
            
        except Exception as e:
            console.print(f"[red]Error during supervised training: {str(e)}[/red]")
            console.print("[red]Traceback:[/red]")
            import traceback
            console.print(traceback.format_exc())
    
    # Check for RL mode
    elif args.rl:
        console.print("\n[bold magenta]=== REINFORCEMENT LEARNING MODE ACTIVATED ===[/bold magenta]")
        console.print(f"[yellow]Will expand model to {args.new_layers} layers and train with RL[/yellow]")
        console.print(f"[yellow]Checkpoints will be saved to: {CHECKPOINT_DIR}[/yellow]")
        
        try:
            # Rest of the RL code remains the same
            # Display original model statistics
            model = GPTLanguageModel()
            display_model_statistics(model)
            
            if args.test_only:
                # Test existing RL model
                console.print("[bold cyan]=== Testing Existing RL Model ===[/bold cyan]")
                rl_model_path = os.path.join(CHECKPOINT_DIR, "final_model_rl.pt")
                
                if os.path.exists(rl_model_path):
                    test_rl_model(model_path=rl_model_path)
                else:
                    console.print(f"[red]RL model not found at {rl_model_path}[/red]")
            else:
                # Start RL training process
                console.print(f"[green]Using model: {args.model_path}[/green]")
                console.print(f"[green]Expanding to {args.new_layers} layers and training for {args.epochs} epochs with {args.dataset_size} examples[/green]")
                
                try:
                    # Implement a simpler version to just expand the model
                    console.print(f"\n[bold magenta]=== Starting Direct RL Training ===[/bold magenta]")
                    
                    # Create a new model with the increased layer count
                    original_n_layer = n_layer  # Save original value
                    temp_n_layer = args.new_layers  # New layer count
                    
                    # Load the trained model checkpoint
                    console.print(f"[yellow]Loading model from: {args.model_path}[/yellow]")
                    checkpoint = torch.load(args.model_path, map_location=device)
                    trained_dict = checkpoint['model_state_dict']
                    
                    # Create an enlarged model
                    class TempConfig:
                        def __init__(self, n_layer):
                            self.n_layer = n_layer
                    
                    # Create expanded model
                    expanded_config = TempConfig(temp_n_layer)
                    
                    # Create a new model with expanded layers
                    console.print(f"[cyan]Creating model with {temp_n_layer} layers...[/cyan]")
                    
                    # Temporarily override n_layer
                    temp_original = n_layer
                    n_layer = temp_n_layer
                    
                    # Create the new model
                    rl_model = GPTLanguageModel()
                    
                    # Restore original n_layer
                    n_layer = temp_original
                    
                    # Copy matching weights
                    expanded_dict = rl_model.state_dict()
                    copied_params = 0
                    
                    for name, param in trained_dict.items():
                        if name in expanded_dict and expanded_dict[name].shape == param.shape:
                            expanded_dict[name].copy_(param)
                            copied_params += 1
                    
                    console.print(f"[green]Copied {copied_params} matching parameters[/green]")
                    
                    # Load the state dict into the expanded model
                    rl_model.load_state_dict(expanded_dict)
                    rl_model = rl_model.to(device)
                    
                    # Now start the actual RL training
                    console.print(f"\n[bold green]=== Model Successfully Expanded to {temp_n_layer} Layers ===[/bold green]")
                    console.print("[bold yellow]Starting Reinforcement Learning Training...[/bold yellow]")
                    
                    # Load the OpenAssistant dataset
                    console.print("[yellow]Loading OpenAssistant dataset...[/yellow]")
                    try:
                        from datasets import load_dataset
                        # Load full dataset or specified number of examples
                        split = "train" if args.dataset_size == -1 else f"train[:{args.dataset_size}]"
                        dataset = load_dataset("OpenAssistant/oasst1", split=split)
                        dataset_size_str = "full dataset" if args.dataset_size == -1 else f"{len(dataset)} samples"
                        console.print(f"[green]Successfully loaded {dataset_size_str} from OpenAssistant dataset[/green]")
                        
                        # Convert to simpler format for training
                        formatted_dataset = []
                        for item in dataset:
                            # Extract data based on structure
                            if 'text' in item and isinstance(item['text'], str):
                                # Extract instruction and response from text field
                                parts = item['text'].split('\n\n')
                                if len(parts) >= 2:
                                    prompt = parts[0].strip()
                                    response = '\n\n'.join(parts[1:]).strip()
                                    formatted_dataset.append({"prompt": prompt, "response": response})
                            elif 'instruction' in item and 'output' in item:
                                formatted_dataset.append({"prompt": item['instruction'], "response": item['output']})
                            elif 'prompt' in item and 'response' in item:
                                formatted_dataset.append({"prompt": item['prompt'], "response": item['response']})
                        
                        if formatted_dataset:
                            dataset = formatted_dataset
                            console.print(f"[green]Formatted {len(dataset)} examples for training[/green]")
                        
                    except Exception as e:
                        console.print(f"[red]Error loading OpenAssistant dataset: {e}[/red]")
                        console.print("[yellow]Falling back to dummy dataset for demonstration[/yellow]")
                        # Create dummy dataset
                        dataset = [
                            {"prompt": "What is machine learning?", "response": "Machine learning is a field of AI focused on algorithms that learn from data."},
                            {"prompt": "Explain quantum computing.", "response": "Quantum computing uses quantum bits to perform computations that would be difficult for classical computers."},
                            {"prompt": "How does a transformer model work?", "response": "Transformer models use self-attention mechanisms to process input data in parallel, capturing relationships between elements regardless of their position in the sequence."},
                            {"prompt": "What is reinforcement learning?", "response": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward."}
                        ] * 50  # Repeat to create more samples
                    
                    # Create optimizer for RL training
                    optimizer = torch.optim.AdamW(rl_model.parameters(), lr=1e-5, weight_decay=0.01)
                    
                    # RL training parameters
                    batch_size = 2
                    max_seq_len = min(block_size, 256)  # Use smaller context for efficiency
                    
                    # Create RL checkpoint directory
                    rl_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "rl_training")
                    if not os.path.exists(rl_checkpoint_dir):
                        os.makedirs(rl_checkpoint_dir)
                    
                    # Total steps counter
                    total_steps = 0
                    cumulative_reward = 0
                    
                    # Define reward computation function
                    def compute_reward(response, target):
                        # Simple reward mechanism based on token overlap
                        response_tokens = set(response.lower().split())
                        target_tokens = set(target.lower().split())
                        
                        # Calculate overlap ratio
                        if len(target_tokens) == 0:
                            return 0.0
                            
                        overlap = len(response_tokens.intersection(target_tokens))
                        overlap_ratio = overlap / len(target_tokens)
                        
                        # Apply reward rules - less punitive for poor matches
                        if overlap_ratio > 0.7:
                            return 20.0  # Excellent match
                        elif overlap_ratio > 0.5:
                            return 10.0  # Good match
                        elif overlap_ratio > 0.2:
                            return 0.0   # Neutral
                        else:
                            return -5.0  # Poor match - less punitive
                    
                    # Configure training settings
                    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
                    use_fp16 = True  # Use mixed precision for better memory efficiency
                    
                    # Increase batch size for faster training
                    batch_size = 16  # Increase from default
                    
                    # Set up mixed precision training
                    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
                    
                    # Configure optimizer with increased learning rate and weight decay
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=5e-5,  # Increased learning rate
                        betas=(0.9, 0.95),
                        weight_decay=0.1,
                    )
                    
                    # Learning rate scheduler for better convergence
                    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=5e-5,
                        total_steps=args.epochs * (min(len(dataset), 5000) // batch_size),
                        pct_start=0.1,
                        div_factor=10.0,
                        final_div_factor=50.0,
                    )
                    
                    # Train for specified epochs
                    for epoch in range(args.epochs):
                        epoch_rewards = []
                        
                        # Clean memory
                        gc.collect()
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, 'memory_summary'):
                            console.print(f"[dim]GPU memory: {torch.cuda.memory_reserved() / 1024**2:.1f}MB[/dim]")
                        
                        # Early stopping checks
                        if epoch > 3 and np.mean(epoch_rewards[-3:]) < -3.0:
                            console.print("[bold red]Training not improving. Early stopping...[/bold red]")
                            break
                        
                        # Use a subset of data for faster iterations in early epochs
                        training_size = min(len(dataset), 2000 if epoch < 5 else 5000)
                        subset_dataset = random.sample(dataset, training_size)
                        
                        # Training loop with progress tracking
                        with tqdm(total=len(subset_dataset) // batch_size, desc=f"RL Epoch {epoch+1}/{args.epochs}") as pbar:
                            for i in range(0, len(subset_dataset), batch_size):
                                batch = subset_dataset[i:i+batch_size]
                                batch_rewards = []
                                
                                # Process each example
                                for example in batch:
                                    try:
                                        # Get prompt and target
                                        if isinstance(example, dict):
                                            prompt = example.get("prompt", example.get("instruction", ""))
                                            target = example.get("response", example.get("output", ""))
                                        else:
                                            continue
                                        
                                        # Skip empty examples
                                        if not prompt or not target:
                                            continue
                                        
                                        # Encode prompt
                                        prompt_tokens = encode(prompt)[:max_seq_len-10]
                                        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
                                        
                                        # Generate response
                                        with torch.cuda.amp.autocast(enabled=use_fp16):
                                            with torch.no_grad():
                                                response_tensor = rl_model.generate(
                                                    prompt_tensor, 
                                                    max_new_tokens=50, 
                                                    temperature=0.8, 
                                                    top_k=40
                                                )[0]
                                        
                                        # Decode response
                                        full_response = decode(response_tensor.tolist())
                                        prompt_text = decode(prompt_tokens)
                                        model_response = full_response[len(prompt_text):] if full_response.startswith(prompt_text) else full_response
                                        
                                        # Compute reward
                                        reward = compute_reward(model_response, target)
                                        batch_rewards.append(reward)
                                        cumulative_reward += reward
                                        
                                        # Log occasionally
                                        if total_steps % 20 == 0:
                                            console.print(f"\n[cyan]Example {total_steps}[/cyan]")
                                            console.print(f"[green]Prompt: {prompt[:100]}...[/green]")
                                            console.print(f"[yellow]Response: {model_response[:100]}...[/yellow]")
                                            console.print(f"[{'green' if reward > 0 else 'red'}]Reward: {reward}[/{'green' if reward > 0 else 'red'}]")
                                        
                                        # Training update
                                        optimizer.zero_grad()
                                        
                                        with torch.cuda.amp.autocast(enabled=use_fp16):
                                            logits, _ = rl_model(prompt_tensor, None)
                                            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                                            loss = -log_probs.mean() * reward
                                        
                                        # Backward and optimize
                                        if use_fp16:
                                            scaler.scale(loss).backward()
                                            scaler.unscale_(optimizer)
                                            torch.nn.utils.clip_grad_norm_(rl_model.parameters(), 1.0)
                                            scaler.step(optimizer)
                                            scaler.update()
                                        else:
                                            loss.backward()
                                            torch.nn.utils.clip_grad_norm_(rl_model.parameters(), 1.0)
                                            optimizer.step()
                                        
                                        total_steps += 1
                                        
                                    except Exception as e:
                                        console.print(f"[red]Error processing example: {e}[/red]")
                                        continue
                                
                                # Update progress bar
                                if batch_rewards:
                                    avg_reward = sum(batch_rewards) / len(batch_rewards)
                                    epoch_rewards.extend(batch_rewards)
                                    pbar.set_postfix({"avg_reward": f"{avg_reward:.2f}"})
                                
                                pbar.update(1)
                                
                                # Save checkpoint periodically
                                if total_steps % 100 == 0 and total_steps > 0:
                                    checkpoint_path = os.path.join(rl_checkpoint_dir, f"rl_checkpoint_{total_steps}.pt")
                                    console.print(f"[cyan]Saving checkpoint at step {total_steps}...[/cyan]")
                                    
                                    torch.save({
                                        'model_state_dict': rl_model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'steps': total_steps,
                                        'new_n_layer': temp_n_layer,
                                        'cumulative_reward': cumulative_reward,
                                        'epoch': epoch,
                                    }, checkpoint_path)
                        
                        # End of epoch summary
                        if epoch_rewards:
                            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
                            console.print(f"\n[bold cyan]Epoch {epoch+1} complete. Average reward: {avg_epoch_reward:.4f}[/bold cyan]")
                    
                    # Save final model after training
                    final_model_path = os.path.join(CHECKPOINT_DIR, "final_model_rl.pt")
                    torch.save({
                        'model_state_dict': rl_model.state_dict(),
                        'config': {
                            'n_layer': temp_n_layer,
                            'n_head': n_head,
                            'n_embd': n_embd,
                            'vocab_size': vocab_size,
                            'block_size': block_size
                        },
                        'training_info': {
                            'type': 'reinforcement_learning',
                            'epochs': args.epochs,
                            'steps': total_steps,
                            'final_reward': cumulative_reward,
                            'model_path': args.model_path,
                            'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }, final_model_path)
                    
                    console.print(f"\n[bold green]=== RL Training Complete! Model saved to {final_model_path} ===[/bold green]")
                    
                    # Store the model for later use
                    global expanded_rl_model
                    expanded_rl_model = rl_model
                    
                except Exception as e:
                    console.print(f"[bold red]Error in RL training: {str(e)}[/bold red]")
                    traceback.print_exc()
                
                # Test the model if it was created successfully
                if 'expanded_rl_model' in globals():
                    try:
                        console.print("[cyan]Testing expanded model...[/cyan]")
                        test_rl_model(model=expanded_rl_model)
                    except Exception as e:
                        console.print(f"[bold red]Error testing model: {str(e)}[/bold red]")
                        traceback.print_exc()
        
        except Exception as e:
            console.print(f"[bold red]Error in RL training: {str(e)}[/bold red]")
            traceback.print_exc()
        
        # Close log file
        if 'log_file' in globals() and not log_file.closed:
            log_file.close()
            
        # Exit after RL training is complete (IMPORTANT: prevents falling through to standard training)
        sys.exit(0)
    
    # Otherwise, run standard or further training code
    elif args.further_training:
        # Display model statistics
        model = GPTLanguageModel()
        display_model_statistics(model)
        
        console.print("\n[bold magenta]=== Further Training Mode Activated ===[/bold magenta]")
        console.print("[yellow]Loading existing model and improving tokenization...[/yellow]")
        
        # Use improved tokenizer if further training
        USE_BETTER_TOKENIZER = True
        
        # Set enhanced hyperparameters for further training
        learning_rate = 5e-5  # Lower learning rate for fine-tuning
        max_iters = 5000      # Shorter training cycle
        eval_interval = 100   # More frequent evaluation
        batch_size = 4        # Increase batch size if possible
        gradient_accumulation_steps = 16  # Reduced for more updates
        
        # Continue training from the specified base model
        try:
            console.print(f"[yellow]Loading model from: {BASE_MODEL_PATH}[/yellow]")
            checkpoint = torch.load(BASE_MODEL_PATH, map_location=device)
            
            # Initialize and load model
            further_training_model = GPTLanguageModel().to(device)
            further_training_model.load_state_dict(checkpoint['model_state_dict'])
            console.print("[green]Successfully loaded model checkpoint[/green]")
            
            # Enhanced BPE tokenization
            console.print("[cyan]Creating improved BPE tokenization...[/cyan]")
            
            # Continue enhanced training
            final_model = train_and_merge(further_training_model)
            
            # Test final model
            console.print("\n[bold yellow]=== Testing Enhanced Model ===[/bold yellow]")
            test_results = test_model_quality(final_model, "enhanced", 1, max_iters, True)
            
            console.print("\n[bold green]=== Enhanced Training Complete ===[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Error in further training: {str(e)}[/bold red]")
            traceback.print_exc()
    else:
        # Display model statistics
        model = GPTLanguageModel()
        display_model_statistics(model)
        
        # Original training code
        console.print("\n[bold cyan]=== Standard Training Mode ===[/bold cyan]")
        console.print(f"[green]Model size: {model_size:.2f}M parameters[/green]")
        console.print(f"[green]Initial memory state: {get_memory_usage()}[/green]")
        console.print(f"[green]Initial GPU state: {log_gpu_memory()}[/green]")
        
        # Make sure the model is on the correct device
        model = model.to(device)
        
        # Run training pipeline for Combined stage only
        final_model = train_and_merge(model)
        
        # Test final model
        console.print("\n[bold yellow]=== Testing Final Combined Model ===[/bold yellow]")
        test_results = test_model_quality(final_model, "combined_final", 1, max_iters, True)
        
        console.print("\n[bold green]=== Combined Stage Training Complete ===[/bold green]")

# Add a function to test the RL model before trying to call it
def test_rl_model(model_path=None, model=None, num_examples=5):
    """Test the RL-trained model on some examples"""
    console.print(f"\n[bold yellow]=== Testing RL-Trained Model ===[/bold yellow]")
    
    if model is None and model_path is not None:
        # Load the model from path
        console.print(f"[yellow]Loading model from: {model_path}[/yellow]")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model with the right number of layers
        if 'config' in checkpoint and 'n_layer' in checkpoint['config']:
            # Use a temporary variable for n_layer
            temp_n_layer = n_layer
            n_layer = checkpoint['config']['n_layer']
            model = GPTLanguageModel()
            model.load_state_dict(checkpoint['model_state_dict'])
            n_layer = temp_n_layer
        else:
            model = GPTLanguageModel()
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
    
    if model is None:
        console.print("[red]No model provided for testing![/red]")
        return
    
    # Set model to eval mode
    model.eval()
    
    # Test prompts
    test_prompts = [
        "Explain the concept of reinforcement learning in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the key benefits of transformer-based language models?",
        "How would you explain quantum computing to a 10-year-old?",
        "Summarize the impact of deep learning on modern technology."
    ]
    
    # Generate responses
    results = []
    for i, prompt in enumerate(test_prompts[:num_examples]):
        console.print(f"\n[cyan]Test Example {i+1}[/cyan]")
        console.print(f"[green]Prompt: {prompt}[/green]")
        
        try:
            # Encode prompt
            prompt_tokens = encode(prompt)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
            
            # Generate with temperature sampling
            with torch.no_grad():
                response_tensor = model.generate(
                    prompt_tensor,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9
                )[0]
            
            # Decode the full response
            full_response = decode(response_tensor.tolist())
            
            # Remove the prompt from the response
            prompt_text = decode(prompt_tokens)
            if full_response.startswith(prompt_text):
                response_text = full_response[len(prompt_text):]
            else:
                response_text = full_response
            
            console.print(f"[yellow]Response: {response_text}[/yellow]")
            
            results.append({
                "prompt": prompt,
                "response": response_text
            })
            
        except Exception as e:
            console.print(f"[red]Error generating response: {e}[/red]")
            traceback.print_exc()
    
    return results

def train_with_supervised_finetuning(model, dataset, epochs=3, batch_size=8, learning_rate=5e-5):
    """
    Train the model using supervised fine-tuning
    """
    console.print(f"[bold yellow]Starting supervised training with {len(dataset)} examples for {epochs} epochs[/bold yellow]")
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Mixed precision training
    use_fp16 = torch.cuda.is_available()
    if hasattr(torch.amp, 'GradScaler'):  # PyTorch 2.0+ API
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    # Dataset preprocessing - convert to tokenized format
    tokenized_dataset = []
    console.print("[cyan]Tokenizing dataset...[/cyan]")
    
    max_context_length = 512  # Maximum context length for training
    
    for item in dataset:
        if isinstance(item, dict) and "prompt" in item and ("target" in item or "response" in item):
            prompt = item["prompt"]
            response = item.get("target", item.get("response", ""))
            
            # Format as instruction-response
            full_text = f"Q: {prompt}\nA: {response}"
            
            # Tokenize
            tokens = encode(full_text)
            
            # Skip if too long
            if len(tokens) > max_context_length:
                # Truncate to fit
                tokens = tokens[:max_context_length]
            
            tokenized_dataset.append(tokens)
    
    console.print(f"[green]Processed {len(tokenized_dataset)} examples[/green]")
    
    # Training loop
    from tqdm import tqdm
    
    for epoch in range(epochs):
        # Shuffle dataset
        random.shuffle(tokenized_dataset)
        
        # Training loop with progress bar
        total_loss = 0
        num_batches = 0
        
        with tqdm(total=len(tokenized_dataset)//batch_size, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i in range(0, len(tokenized_dataset), batch_size):
                batch = tokenized_dataset[i:i+batch_size]
                
                # Skip incomplete batches
                if len(batch) < batch_size:
                    continue
                
                # Create tensors
                batch_tensors = []
                for sequence in batch:
                    # Convert to tensor
                    tensor = torch.tensor(sequence, dtype=torch.long)
                    batch_tensors.append(tensor)
                
                # Pad sequences to the same length
                max_len = max(len(t) for t in batch_tensors)
                padded_batch = []
                
                for tensor in batch_tensors:
                    if len(tensor) < max_len:
                        padding = torch.zeros(max_len - len(tensor), dtype=torch.long)
                        padded_tensor = torch.cat([tensor, padding])
                    else:
                        padded_tensor = tensor
                    padded_batch.append(padded_tensor)
                
                # Stack into a single tensor
                x = torch.stack(padded_batch).to(device)
                
                # Prepare inputs and targets for causal language modeling
                # If sequence is too short, skip
                if x.size(1) < 2:
                    continue
                    
                # Inputs are all tokens except the last one
                inputs = x[:, :-1]
                # Targets are all tokens except the first one
                targets = x[:, 1:]
                
                # Forward pass
                optimizer.zero_grad()
                
                try:
                    # Use proper autocast API with PyTorch version check
                    if hasattr(torch.amp, 'autocast'):  # PyTorch 2.0+ API
                        with torch.amp.autocast('cuda', enabled=use_fp16):
                            logits, loss = model(inputs, targets)
                    else:
                        with torch.cuda.amp.autocast(enabled=use_fp16):
                            logits, loss = model(inputs, targets)
                        
                    # Backward pass
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    pbar.update(1)
                    
                except Exception as e:
                    console.print(f"[red]Error in training batch: {str(e)}[/red]")
                    console.print(f"[red]Batch shape: {x.shape}[/red]")
                    # Print traceback for better debugging
                    import traceback
                    console.print(traceback.format_exc())
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
        
        # Epoch summary
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        console.print(f"[bold green]Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}[/bold green]")
    
    console.print("[bold green]Supervised training complete![/bold green]")
    return model
