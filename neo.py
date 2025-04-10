import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
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

# Initialize rich console at the start
console = Console()

# Create a log file for training history
log_file = open('training_history.log', 'w')

# Model Identity & Core Architecture
n_layer = 24        # Reduced size for better learning
n_head = 24        # Balanced architecture
n_embd = 1024       # Standard embedding size
block_size = 128   # Keep reduced for memory
batch_size = 2     # Increase to help with learning
max_iters = 30000  # Reduced but sufficient
eval_interval = 250  # More frequent evaluation
learning_rate = 1e-4  # Increased from 3e-4
weight_decay = 0.001  # Reduced further
dropout = 0.1      # Decreased for better learning
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 40    # Balanced for valid evaluation
gradient_accumulation_steps = 32  # Reduced to update more frequently
warmup_ratio = 0.05  # Shorter warmup

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
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
wiki_text = "\n".join(dataset['train']['text'])

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
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training data size: {len(train_data)} tokens")
print(f"Validation data size: {len(val_data)} tokens")

# Data loading with dynamic batching
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
            self.freq_cis = self.precompute_freqs_cis(head_size, block_size)

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
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        xc = xc * freqs_cis
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
                # Ensure mask has proper dimensions (B, T_i, T)
                chunk_mask = mask[i:j]  # Get the relevant part of the mask (T_chunk, T)
                chunk_mask = chunk_mask.unsqueeze(0).expand(B, -1, -1)  # Add batch dim
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
            k = self.apply_rotary_emb(k, self.freq_cis[:T])
            q = self.apply_rotary_emb(q, self.freq_cis[:T])
        
        # Create attention mask (corrected shape)
        mask = self.tril[:T, :T]  # Shape: (T, T)
        
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
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
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
        
        print("Model initialization complete!")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Better initialization for stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Special initialization for embedding table
            if module.weight.shape[0] == vocab_size:  # Only for token embeddings
                # Initialize with smaller values for better gradient flow
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
                # Add more initialization for special tokens
                for i in range(len(special_tokens)):
                    with torch.no_grad():
                        # Initialize special tokens with zeros for better convergence
                        module.weight[i].fill_(0.0)
            else:  # Position embeddings
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm for stability
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Use aggressive memory optimization with mixed precision
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            # Get token embeddings - free intermediate memory immediately
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            
            # Add positional embeddings if not using RoPE or ALiBi
            if not use_rope and not use_alibi:
                pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
                x = tok_emb + pos_emb # (B,T,C)
            else:
                x = tok_emb
            
            # Clear unneeded tensors
            del tok_emb
            torch.cuda.empty_cache()
            
            # Apply transformer blocks with aggressive memory cleanup
            for i, block in enumerate(self.blocks):
                x = block(x)
                # Clear block caches periodically 
                if (i+1) % 4 == 0 and i < len(self.blocks)-1:
                    # This helps prevent memory fragmentation
                    torch.cuda.empty_cache()
                
            # Apply final layer norm and project to vocabulary
            x = self.ln_f(x) # (B,T,C)
            logits = self.lm_head(x) # (B,T,vocab_size)
            
            # Clear more memory
            del x
            torch.cuda.empty_cache()

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C)
            targets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targets_view)
            # Free memory immediately
            del logits_view, targets_view
            torch.cuda.empty_cache()

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Optional top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Learning rate scheduler with warmup and cosine decay
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > warmup_iters, cosine learning rate decay
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

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
        
        # Create more robust checkpoint directory naming
        base_dir = "checkpoints"
        if not os.path.exists(base_dir):
            try:
                os.makedirs(base_dir)
                print(f"Created base checkpoint directory: {base_dir}")
            except Exception as e:
                print(f"Warning: Could not create base directory {base_dir}: {e}")
                
        self.checkpoint_dir = os.path.join(base_dir, f"{name.lower()}")
        
        # Create checkpoint directory with better error handling
        try:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print(f"Created checkpoint directory: {self.checkpoint_dir}")
        except Exception as e:
            # Fallback to current directory if we can't create the folder
            print(f"Error creating checkpoint directory {self.checkpoint_dir}: {e}")
            self.checkpoint_dir = f"checkpoint_{name.lower()}"
            if not os.path.exists(self.checkpoint_dir):
                try:
                    os.makedirs(self.checkpoint_dir)
                    print(f"Created fallback checkpoint directory: {self.checkpoint_dir}")
                except Exception as e2:
                    print(f"Fatal error creating checkpoint directories: {e2}")
                    self.checkpoint_dir = "."  # Last resort: use current directory

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
        
        print(f"\n=== {self.name} Stage Complete ===")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Steps completed: {self.steps_completed}")
        print(f"Best loss achieved: {self.best_loss:.4f}")
        print(f"Target loss reached: {'Yes' if success else 'No'}")
        if final_loss:
            print(f"Final loss: {final_loss:.4f}")

def test_model_quality(model, stage_name, test_iteration, current_step=0, save_checkpoint=True):
    print(f"\n=== Testing Model Quality (Stage: {stage_name}, Test #{test_iteration}) ===")
    
    # Optimize memory before testing
    torch.cuda.empty_cache()
    gc.collect()
    
    # Test prompts
    test_prompts = [
        "Explain quantum entanglement and its applications.",
        "Analyze the impact of AI on the global workforce.",
        "Explain how transformer neural networks work.",
        "Discuss climate change mitigation approaches."
    ]
    
    # Set reasonable limits for word-based models
    max_tokens = 100  # This will generate enough words for evaluation
    
    results = []
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}/{len(test_prompts)}: Generating response...")
            try:
                # Encode the prompt
                encoded_prompt = encode(prompt)
                encoded = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
                
                # Record prompt length for proper trimming later
                prompt_len = len(encoded_prompt)
                
                # Use mixed precision during generation
                with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                    # Generate with balanced settings
                    response = model.generate(
                        encoded,
                        max_new_tokens=max_tokens,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.95
                    )[0]
                
                # Decode the response and only keep generated part
                full_text = decode(response.tolist())
                # Try to separate prompt from response using word tokens
                response_text = " ".join(full_text.split()[prompt_len:])
                
                # Memory cleanup
                del encoded, response
                torch.cuda.empty_cache()
                
                # Evaluate response quality with relevant metrics for word tokens
                words = response_text.split()
                length = len(words)
                unique_words = len(set(words)) if words else 0
                vocab_richness = unique_words / length if length > 0 else 0
                
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
    test_dir = f"test_results_{stage_name.lower()}"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    try:
        with open(os.path.join(test_dir, f'test_{test_iteration}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Test results saved to {test_dir}/test_{test_iteration}.json")
        
        # Also save a checkpoint after testing if requested
        if save_checkpoint and current_step > 0:
            test_checkpoint_path = f"test_checkpoint_{stage_name.lower()}_{test_iteration}.pt"
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
    """Create a rich layout for progress visualization with smooth animations"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),  # Fixed size for header
        Layout(name="body", ratio=6),   # Main content
        Layout(name="footer", size=3)   # Fixed size for footer
    )
    return layout

def train_until_target(model, train_data, val_data, stage: TrainingStage):
    stage.start()
    
    # For smooth console output, adjust refresh rate
    console.width = min(console.width, 120)  # Prevent too wide outputs that could clip
    console.height = min(console.height, 60)  # Prevent too tall outputs
    
    # Use 8-bit Adam for memory efficiency if available
    try:
        from bitsandbytes.optim import Adam8bit
        optimizer = Adam8bit(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        console.print("[green]Using 8-bit Adam optimizer for memory efficiency[/green]")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        console.print("[yellow]Consider installing bitsandbytes for 8-bit optimization[/yellow]")
    
    # Enable gradient scaling for stability
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    
    # Calculate test intervals - reduced frequency to save memory
    test_intervals = [int(i * stage.max_steps / 3) for i in range(1, 3)]
    console.print(f"[green]Model will be tested at steps: {test_intervals}[/green]")
    
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
    
    # Create progress layout with lower refresh rate
    layout = create_progress_layout()
    refresh_rate = 3  # Even lower refresh rate to reduce overhead
    
    # Memory optimization function that's more aggressive
    def optimize_memory():
        if torch.cuda.is_available():
            # Force garbage collection first
            gc.collect()
            # Empty cache and fragment memory
            torch.cuda.empty_cache()
            # Try to defragment memory
            torch.cuda.synchronize()
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats()
                if 'active_bytes.all.current' in stats:
                    active_mb = stats['active_bytes.all.current'] / 1024 / 1024
                    return f"Cleared (Active: {active_mb:.1f}MB)"
            return "Cleared"
        return "GPU not available"
    
    # Initialize variables
    true_loss = 0.0
    running_avg_loss = 0.0
    steps_per_second = 0.0
    eta_str = "00:00:00"
    loss_buffer = []
    avg_window = 100
    current_avg_loss = 0.0
    total_steps = 0
    recovery_mode = False
    
    # Add OOM recovery counter
    oom_count = 0
    max_oom_retries = 5
    last_successful_iter = start_step

    with Live(layout, refresh_per_second=refresh_rate, screen=False, auto_refresh=True) as live:
        # Initialize header with stage info
        layout["header"].update(
            Panel(f"[bold cyan]Stage: {stage.name} | Max Steps: {stage.max_steps} | Target Loss: {stage.target_loss}[/bold cyan]")
        )
        
        # Add model info to footer with memory settings
        layout["footer"].update(
            Panel(f"[cyan]Model: {n_layer}L/{n_head}H/{n_embd}E | "
                f"Batch: {batch_size} | Grad Acc: {gradient_accumulation_steps} | "
                f"Memory Opts: {'Enabled' if use_mixed_precision else 'Disabled'}[/cyan]")
        )
        
        # Setup progress bars
        progress_width = min(console.width - 20, 80)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=progress_width, complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
            auto_refresh=False
        ) as progress:
            
            main_task = progress.add_task(f"[cyan]{stage.name} Training", total=stage.max_steps)
            loss_task = progress.add_task("[yellow]Current Loss", total=None)
            memory_task = progress.add_task("[green]Memory Usage", total=None)
            
            # Initial layout update
            layout["body"].update(progress)
            
            last_eval_time = time.time()
            last_eval_step = start_step
            
            # Initial memory optimization
            optimize_memory()
            
            for iter in range(start_step, stage.max_steps):
                # Update progress
                progress.update(main_task, completed=iter, refresh=False)
                
                # Update header less frequently
                if iter % 20 == 0:
                    layout["header"].update(
                        Panel(f"[bold cyan]Stage: {stage.name} | Step: {iter}/{stage.max_steps} | "
                            f"Loss: {running_avg_loss:.4f} | Speed: {steps_per_second:.1f} steps/s | ETA: {eta_str}[/bold cyan]")
                    )
                
                # Training step with comprehensive error handling
                try:
                    # Clear memory more aggressively during training
                    if iter % 100 == 0 and iter > 0:
                        status = optimize_memory()
                        layout["footer"].update(Panel(f"[blue]Memory optimization: {status}[/blue]"))
                    
                    # Get batch with error handling
                    try:
                        xb, yb = get_batch('train')
                    except Exception as e:
                        layout["footer"].update(Panel(f"[red]Error getting batch: {str(e)}[/red]"))
                        continue
                    
                    # Forward pass with aggressive memory optimization
                    try:
                        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                            # Split computation if in recovery mode
                            if recovery_mode and block_size > 64:
                                # Process first half
                                half_size = block_size // 2
                                logits1, loss1 = model(xb[:,:half_size], yb[:,:half_size])
                                # Free memory
                                del logits1
                                torch.cuda.empty_cache()
                                # Process second half
                                logits2, loss2 = model(xb[:,half_size:], yb[:,half_size:]) 
                                # Combine losses
                                loss = (loss1 + loss2) / 2
                                del logits2, loss1, loss2
                            else:
                                logits, loss = model(xb, yb)
                                del logits
                            
                            # Scale loss for gradient accumulation
                            loss = loss / gradient_accumulation_steps
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            oom_count += 1
                            optimize_memory()
                            
                            if oom_count >= max_oom_retries:
                                # Enable recovery mode if we keep hitting OOM
                                recovery_mode = True
                                layout["footer"].update(Panel(f"[red]Entering recovery mode after {oom_count} OOM errors[/red]"))
                                # Restart from the last successful iteration
                                iter = max(last_successful_iter, iter - gradient_accumulation_steps)
                                oom_count = 0
                            else:
                                layout["footer"].update(Panel(f"[red]OOM in forward pass ({oom_count}/{max_oom_retries}), retrying...[/red]"))
                            continue
                        else:
                            layout["footer"].update(Panel(f"[red]Forward pass error: {str(e)}[/red]"))
                            continue
                    
                    # Update loss tracking
                    if iter % 10 == 0:
                        true_loss = loss.item() * gradient_accumulation_steps
                        loss_buffer.append(true_loss)
                        
                        if len(loss_buffer) > avg_window:
                            loss_buffer.pop(0)
                        
                        current_avg_loss = sum(loss_buffer) / len(loss_buffer)
                        total_steps += 1
                        running_avg_loss = (running_avg_loss * (total_steps - 1) + true_loss) / total_steps
                        
                        log_message = f"Step {iter}: Loss: {true_loss:.4f} | Avg: {running_avg_loss:.4f}"
                        log_file.write(log_message + "\n")
                        log_file.flush()
                        
                        progress.update(loss_task, description=f"[yellow]Loss: {true_loss:.4f} | Avg: {running_avg_loss:.4f}[/yellow]", refresh=False)
                    
                    # Backward pass with error handling
                    try:
                        scaler.scale(loss).backward()
                    except Exception as e:
                        layout["footer"].update(Panel(f"[red]Backward pass error: {str(e)}[/red]"))
                        optimize_memory()
                        continue
                    
                    # Update last successful iteration
                    last_successful_iter = iter
                    
                    # Optimizer step
                    if ((iter + 1) % gradient_accumulation_steps == 0) or (iter + 1 == stage.max_steps):
                        try:
                            # Gradient clipping for stability
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            
                            # Optimizer step
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)  # More memory efficient
                            
                            # Update memory display
                            if iter % 50 == 0:
                                gpu_mem = log_gpu_memory()
                                ram_mem = get_memory_usage()
                                recovery_status = "[red]RECOVERY MODE[/red] | " if recovery_mode else ""
                                progress.update(memory_task, description=f"[green]{recovery_status}GPU: {gpu_mem} | RAM: {ram_mem}", refresh=False)
                            
                            # Save checkpoint with more frequent attempts and better feedback
                            if (iter % stage.checkpoint_interval == 0 and iter > 0) or (iter == start_step + 10):  # Save early checkpoint to verify working
                                # Try to save checkpoint with clear feedback
                                layout["footer"].update(Panel(f"[blue]Saving checkpoint at step {iter}...[/blue]"))
                                
                                checkpoint_success = stage.save_checkpoint(model, optimizer, running_avg_loss, iter)
                                
                                if checkpoint_success:
                                    # Log successful checkpoint
                                    log_file.write(f"Checkpoint saved at step {iter}\n")
                                    log_file.flush()
                                    layout["footer"].update(Panel(f"[green]Checkpoint successfully saved at step {iter}[/green]"))
                                else:
                                    # Log checkpoint failure
                                    layout["footer"].update(Panel(f"[red]Failed to save checkpoint at step {iter}. Will try again later.[/red]"))
                                    log_file.write(f"WARNING: Failed to save checkpoint at step {iter}\n")
                                    log_file.flush()
                        
                        except RuntimeError as e:
                            layout["footer"].update(Panel(f"[red]Optimizer step error: {str(e)}[/red]"))
                            optimize_memory()
                            continue
                    
                    # Periodic evaluation with reduced frequency
                    current_time = time.time()
                    if (iter % eval_interval == 0 and iter > 0) or (current_time - last_eval_time > 600):  # 10 minutes max
                        try:
                            # Calculate speed and ETA
                            steps_done = iter - last_eval_step
                            time_elapsed = current_time - last_eval_time
                            steps_per_second = steps_done / time_elapsed if time_elapsed > 0 else 0
                            remaining_steps = stage.max_steps - iter
                            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                            
                            # Optimize memory before evaluation
                            optimize_memory()
                            
                            # Run evaluation with reduced iterations
                            losses = estimate_loss()
                            
                            # Update display
                            layout["header"].update(
                                Panel(f"[bold cyan]Stage: {stage.name} | Step: {iter}/{stage.max_steps} | "
                                    f"Loss: {running_avg_loss:.4f} | Val Loss: {losses['val']:.4f} | "
                                    f"Speed: {steps_per_second:.1f} steps/s | ETA: {eta_str}[/bold cyan]")
                            )
                            
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
                                # Save best model
                                try:
                                    optimize_memory()
                                    stage.save_checkpoint(model, optimizer, losses['val'], iter)
                                    layout["footer"].update(Panel(f"[green]Best model saved with val loss {losses['val']:.4f}[/green]"))
                                except Exception as e:
                                    layout["footer"].update(Panel(f"[red]Error saving best model: {str(e)}[/red]"))
                            
                            # Check for target loss
                            if running_avg_loss <= stage.target_loss:
                                layout["footer"].update(Panel(f"[bold green]Target loss {stage.target_loss} achieved![/bold green]"))
                                break
                        
                        except Exception as e:
                            layout["footer"].update(Panel(f"[red]Evaluation error: {str(e)}[/red]"))
                            continue
                    
                    # Reduce testing frequency to save memory
                    if iter in test_intervals:
                        try:
                            test_iteration = test_intervals.index(iter) + 1
                            console.print(f"\n[cyan]Running model test {test_iteration}/{len(test_intervals)} at step {iter}...[/cyan]")
                            
                            # Optimize memory before testing
                            optimize_memory()
                            
                            # Run test with reduced number of prompts
                            test_results = test_model_quality(model, stage.name, test_iteration, iter, True)
                            model.train()  # Return to training mode
                        except Exception as e:
                            layout["footer"].update(Panel(f"[red]Testing error: {str(e)}[/red]"))
                
                except Exception as e:
                    # Global exception handler
                    layout["footer"].update(Panel(f"[red]Unhandled error: {str(e)}[/red]"))
                    traceback.print_exc()
                    optimize_memory()
                    continue
                
                # Added safety checkpoint every 1000 steps regardless of interval
                if iter > 0 and iter % 1000 == 0:
                    safety_path = f"safety_checkpoint_step_{iter}.pt"
                    try:
                        # Basic checkpoint with minimal data to ensure we have something
                        safety_checkpoint = {
                            'step': iter,
                            'model_state_dict': model.state_dict(),
                            'loss': running_avg_loss,
                            'stage_name': stage.name
                        }
                        torch.save(safety_checkpoint, safety_path)
                        log_file.write(f"Safety checkpoint saved to {safety_path}\n")
                        log_file.flush()
                        layout["footer"].update(Panel(f"[blue]Safety checkpoint saved to {safety_path}[/blue]"))
                    except Exception as e:
                        layout["footer"].update(Panel(f"[red]Even safety checkpoint failed: {str(e)}[/red]"))
                        log_file.write(f"ERROR: Safety checkpoint failed: {str(e)}\n")
                        log_file.flush()
    
    # Mark stage as complete
    stage.complete(success=True, final_loss=running_avg_loss)
    return model.state_dict()

# After all stages complete, merge the models
def train_and_merge():
    stage_models = []
    
    for stage in stages:
        console.print(f"\n[bold cyan]=== Starting {stage.name} Stage ===[/bold cyan]")
        stage_dict = train_until_target(model, train_data, val_data, stage)
        stage_models.append(stage_dict)
        console.print(f"[bold green]{stage.name} Stage Complete![/bold green]")
    
    # Merge models and save final version
    console.print("\n[bold yellow]=== Creating Final Enhanced Model ===[/bold yellow]")
    final_model = merge_models(stage_models)
    
    # Save the final merged model
    final_save_path = "final_enhanced_model.pt"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': {
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'block_size': block_size,
            'vocab_size': vocab_size
        },
        'training_info': {
            'stages_completed': [s.name for s in stages],
            'final_losses': [s.best_loss for s in stages],
            'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }, final_save_path)
    
    console.print(f"[bold green]Final enhanced model saved to: {final_save_path}[/bold green]")
    return final_model

# Move create_training_data and related functions before the training pipeline
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
    
    # Load WikiText-2 dataset as supplementary data
    console.print("[yellow]Loading WikiText-2 dataset...[/yellow]")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    wiki_text = "\n".join(dataset['train']['text'])
    
    # Combine with emphasis on model's identity
    combined_text = base_data + "\n\n" + wiki_text
    
    console.print(f"[green]Combined training text size: {len(combined_text)} characters[/green]")
    return combined_text

# Create model and prepare for training
model = GPTLanguageModel()
m = model.to(device)
model_size = sum(p.numel() for p in m.parameters())/1e6

if model_size < 1000:  # Less than 1B parameters
    console.print("\n[yellow]⚠️ Warning: Model size is less than 1B parameters![/yellow]")
    console.print("[yellow]Consider increasing n_layer, n_head, or n_embd[/yellow]")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        console.print("[red]Exiting...[/red]")
        exit()

# Define training stages with more appropriate targets for word-level tokenization
stages = [
    TrainingStage("Identity", max_steps=5000, target_loss=7.0),  # Much higher for early training
    TrainingStage("Metadata", max_steps=8000, target_loss=6.0), 
    TrainingStage("Combined", max_steps=15000, target_loss=5.0)
]

# Training pipeline
if __name__ == "__main__":
    console.print("\n[bold cyan]=== Starting Training Pipeline ===[/bold cyan]")
    console.print(f"[green]Model size: {model_size:.2f}M parameters[/green]")
    console.print(f"[green]Initial memory state: {get_memory_usage()}[/green]")
    console.print(f"[green]Initial GPU state: {log_gpu_memory()}[/green]")
    
    # Run training and merging pipeline
    final_model = train_and_merge()
    
    # Test final model
    console.print("\n[bold yellow]=== Testing Final Enhanced Model ===[/bold yellow]")
    test_results = test_model_quality(final_model, "final", 1, max_iters, True)
    
    console.print("\n[bold green]=== Training Pipeline Complete ===[/bold green]")

# At the end of training, close the log file
log_file.close()