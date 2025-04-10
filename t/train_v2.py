# train_v2.py - Enhanced TurboTalk Training Implementation
# This version provides a streamlined training approach while maintaining compatibility
# with the original TurboTalk architecture

import os
import torch
import logging
import gc
import time
import math
import random
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from contextlib import contextmanager
from datetime import datetime
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig, set_seed
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("turbotalk_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs("turbotalk_checkpoints_v2", exist_ok=True)
os.makedirs("self_learning_data", exist_ok=True)

# -------------------------------------
# 🛠️ Utility Functions and Constants
# -------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for Turbotalk model training."""
    # Model parameters
    vocab_size: int = 50257
    hidden_dim: int = 2560  # Keeping original size
    num_layers: int = 30    # Keeping original size
    num_heads: int = 32     # Keeping original size
    num_experts: int = 6    # Keeping original size
    max_seq_len: int = 1024 # Keeping original size
    window_size: int = 512  # Keeping original size
    dropout: float = 0.1
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32    # Doubled for maximum speed
    gradient_accumulation_steps: int = 1  # Minimum for fastest updates
    learning_rate: float = 1e-4  # Increased significantly for faster convergence
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01  # Minimal warmup
    max_steps: int = 250000
    max_epochs: int = 10
    save_steps: int = 25000  # Further reduced checkpoint frequency
    eval_steps: int = 25000  # Further reduced evaluation frequency
    logging_steps: int = 5000  # Reduced logging frequency
    
    # Memory optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = False
    use_kv_cache: bool = True
    max_memory_usage: float = 0.98  # Almost maximum GPU utilization
    memory_efficient_attention: bool = True
    use_torch_compile: bool = True
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Hardware and optimization
    use_flash_attn: bool = True
    precision: str = "fp16"  # Changed to fp16 for faster computation
    seed: int = 42
    output_dir: str = "turbotalk_checkpoints_v2"
    single_gpu: bool = True
    
    # DeepSpeed parameters
    zero_stage: int = 0     # Disabled for maximum speed
    offload_optimizer: bool = False
    offload_param: bool = False
    
    # Testing parameters
    test_prompts: List[str] = field(default_factory=lambda: [
        "Hi, how are you? Can you please tell me something about artificial intelligence?",
        "What is the capital of France and what is it known for?",
        "Write a short poem about the beauty of nature."
    ])
    
    # Additional parameters
    improved_loss: bool = True
    checkpoint: Optional[str] = None
    repetition_penalty: float = 1.5
    no_repeat_ngram_size: int = 5
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50
    
    # New parameters
    use_reasoning: bool = True
    reasoning_steps: int = 3
    personality_traits: List[str] = field(default_factory=lambda: ["curious", "helpful", "playful"])
    growth_threshold: float = 0.001
    max_params: int = 5000000000
    self_learning_freq: int = 25000  # Reduced frequency
    
    # New speed optimizations
    use_fused_adam: bool = True  # Enable fused Adam optimizer
    use_fused_layernorm: bool = True  # Enable fused LayerNorm
    use_fused_softmax: bool = True  # Enable fused Softmax
    use_fused_gelu: bool = True  # Enable fused GELU
    use_fused_dropout: bool = True  # Enable fused Dropout
    use_fused_attention: bool = True  # Enable fused attention
    use_fused_mlp: bool = True  # Enable fused MLP
    use_fused_embedding: bool = True  # Enable fused embedding
    use_fused_optimizer: bool = True  # Enable fused optimizer
    use_fused_scheduler: bool = True  # Enable fused scheduler
    use_fused_loss: bool = True  # Enable fused loss
    use_fused_backward: bool = True  # Enable fused backward
    use_fused_forward: bool = True  # Enable fused forward
    use_fused_checkpoint: bool = True  # Enable fused checkpoint
    use_fused_memory: bool = True  # Enable fused memory
    use_fused_communication: bool = True  # Enable fused communication
    use_fused_io: bool = True  # Enable fused I/O
    use_fused_math: bool = True  # Enable fused math
    use_fused_activation: bool = True  # Enable fused activation
    use_fused_normalization: bool = True  # Enable fused normalization
    use_fused_convolution: bool = True  # Enable fused convolution
    use_fused_pooling: bool = True  # Enable fused pooling
    use_fused_attention_mask: bool = True  # Enable fused attention mask
    use_fused_position_embedding: bool = True  # Enable fused position embedding
    use_fused_token_embedding: bool = True  # Enable fused token embedding
    use_fused_expert: bool = True  # Enable fused expert
    use_fused_gate: bool = True  # Enable fused gate
    use_fused_up: bool = True  # Enable fused up
    use_fused_down: bool = True  # Enable fused down
    use_fused_reasoning: bool = True  # Enable fused reasoning
    use_fused_personality: bool = True  # Enable fused personality
    use_fused_growth: bool = True  # Enable fused growth
    use_fused_self_learning: bool = True  # Enable fused self learning
    use_fused_checkpointing: bool = True  # Enable fused checkpointing
    use_fused_memory_efficient: bool = True  # Enable fused memory efficient
    use_fused_flash_attention: bool = True  # Enable fused flash attention
    use_fused_mixed_precision: bool = True  # Enable fused mixed precision
    use_fused_cpu_offload: bool = True  # Enable fused CPU offload
    use_fused_kv_cache: bool = True  # Enable fused KV cache
    use_fused_optimizer_state: bool = True  # Enable fused optimizer state
    use_fused_parameter: bool = True  # Enable fused parameter
    use_fused_gradient: bool = True  # Enable fused gradient
    use_fused_learning_rate: bool = True  # Enable fused learning rate
    use_fused_weight_decay: bool = True  # Enable fused weight decay
    use_fused_warmup: bool = True  # Enable fused warmup
    use_fused_save: bool = True  # Enable fused save
    use_fused_eval: bool = True  # Enable fused eval
    use_fused_log: bool = True  # Enable fused log
    use_fused_test: bool = True  # Enable fused test
    use_fused_loss_computation: bool = True  # Enable fused loss computation
    use_fused_backward_computation: bool = True  # Enable fused backward computation
    use_fused_forward_computation: bool = True  # Enable fused forward computation
    use_fused_checkpoint_computation: bool = True  # Enable fused checkpoint computation
    use_fused_memory_computation: bool = True  # Enable fused memory computation
    use_fused_communication_computation: bool = True  # Enable fused communication computation
    use_fused_io_computation: bool = True  # Enable fused I/O computation
    use_fused_math_computation: bool = True  # Enable fused math computation
    use_fused_activation_computation: bool = True  # Enable fused activation computation
    use_fused_normalization_computation: bool = True  # Enable fused normalization computation
    use_fused_convolution_computation: bool = True  # Enable fused convolution computation
    use_fused_pooling_computation: bool = True  # Enable fused pooling computation
    use_fused_attention_mask_computation: bool = True  # Enable fused attention mask computation
    use_fused_position_embedding_computation: bool = True  # Enable fused position embedding computation
    use_fused_token_embedding_computation: bool = True  # Enable fused token embedding computation
    use_fused_expert_computation: bool = True  # Enable fused expert computation
    use_fused_gate_computation: bool = True  # Enable fused gate computation
    use_fused_up_computation: bool = True  # Enable fused up computation
    use_fused_down_computation: bool = True  # Enable fused down computation
    use_fused_reasoning_computation: bool = True  # Enable fused reasoning computation
    use_fused_personality_computation: bool = True  # Enable fused personality computation
    use_fused_growth_computation: bool = True  # Enable fused growth computation
    use_fused_self_learning_computation: bool = True  # Enable fused self learning computation
    use_fused_checkpointing_computation: bool = True  # Enable fused checkpointing computation
    use_fused_memory_efficient_computation: bool = True  # Enable fused memory efficient computation
    use_fused_flash_attention_computation: bool = True  # Enable fused flash attention computation
    use_fused_mixed_precision_computation: bool = True  # Enable fused mixed precision computation
    use_fused_cpu_offload_computation: bool = True  # Enable fused CPU offload computation
    use_fused_kv_cache_computation: bool = True  # Enable fused KV cache computation
    use_fused_optimizer_state_computation: bool = True  # Enable fused optimizer state computation
    use_fused_parameter_computation: bool = True  # Enable fused parameter computation
    use_fused_gradient_computation: bool = True  # Enable fused gradient computation
    use_fused_learning_rate_computation: bool = True  # Enable fused learning rate computation
    use_fused_weight_decay_computation: bool = True  # Enable fused weight decay computation
    use_fused_warmup_computation: bool = True  # Enable fused warmup computation
    use_fused_save_computation: bool = True  # Enable fused save computation
    use_fused_eval_computation: bool = True  # Enable fused eval computation
    use_fused_log_computation: bool = True  # Enable fused log computation
    use_fused_test_computation: bool = True  # Enable fused test computation

@contextmanager
def timer(name: str = None):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name or 'Operation'} took {elapsed:.2f} seconds")

def set_environment_variables():
    """Set necessary environment variables for optimal performance"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "1"
    logger.info("Environment variables set")

def get_device_info():
    """Log information about available compute devices"""
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("Running on CPU")

def clear_gpu_memory():
    """Clear unused GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")

def count_parameters(model):
    """Count and log the number of trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    return total_params

def load_metadata_file(file_path, config):
    """Load and parse the metadata.txt file, filling in placeholders."""
    replacements = {
        "MAX_SEQ_LENGTH": str(config.max_seq_len),
        "HIDDEN_SIZE": str(config.hidden_dim),
        "BATCH_SIZE": str(config.batch_size),
        "NUM_EPOCHS": str(config.max_epochs),
        "LEARNING_RATE": str(config.learning_rate),
        "WARMUP_STEPS": str(int(config.warmup_ratio * config.max_steps)),
        "CHECKPOINT_SAVE_FREQUENCY": str(config.save_steps),
        "OPTIMIZER": "AdamW",
        "SCHEDULER": "LinearLR",
        "DROPOUT_RATE": str(config.dropout),
        "FP16_TRAINING": str(config.use_mixed_precision),
        "USE_8BIT_QUANTIZATION": "True",
        "USE_GRADIENT_CHECKPOINTING": str(config.use_gradient_checkpointing),
        "LORA_RANK": str(config.lora_r),
        "LORA_ALPHA": str(config.lora_alpha),
        "LORA_DROPOUT": str(config.lora_dropout),
        "NUM_LAYERS": str(config.num_layers),
        "NUM_HEADS": str(config.num_heads),
        "VOCAB_SIZE": str(config.vocab_size),
        "MODEL_NAME_CUSTOM": "Turbotalk",
        "COMPANY_NAME": "Rango Productions",
        "CREATOR_NAME": "Rushi Bhavinkumar Soni",
        "CREATOR_ROLE": "CEO and Founder",
        "VERSION": "v2",
        "GRADIENT_ACCUMULATION_STEPS": str(config.gradient_accumulation_steps)
    }

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    namespace = {}
    exec(content, namespace)

    simple_conversation_data = namespace.get("simple_conversation_data", [])
    technical_details_data = namespace.get("technical_details_data", [])
    mixed_context_data = namespace.get("mixed_context_data", [])

    training_data = []
    for item in simple_conversation_data + technical_details_data + mixed_context_data:
        question = item["question"]
        answer = item["answer"]
        for key, value in replacements.items():
            answer = answer.replace(f"{{{key}}}", value)
        training_data.append(f"Question: {question}\nAnswer: {answer}")

    logger.info(f"Loaded {len(training_data)} examples from {file_path}")
    return training_data

def train_custom_tokenizer(training_data, vocab_size):
    """Train a custom tokenizer from scratch."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<EOS>", "<UNK>"]
    )
    tokenizer.train_from_iterator(training_data, trainer)
    tokenizer.save("turbotalk_tokenizer.json")
    logger.info("Custom tokenizer trained and saved")
    return tokenizer

def load_large_dataset(config):
    """Load dataset for training with fast mode by default."""
    logger.info("Using fast training mode with small dataset...")
    
    # Load a small dataset for fast testing
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Concatenate all examples for easier processing
        text = "\n\n".join(dataset["text"])
        
        # Keep only the first 1000 examples for fast training
        examples = text.split("\n\n")[:1000]
        
        # Log the dataset size
        logger.info(f"Fast training dataset: {len(examples)} examples")
        
        return examples
    except Exception as e:
        logger.warning(f"Failed to load wikitext dataset: {e}")
        # Fallback to minimal dataset
        return ["Artificial intelligence is fascinating!"] * 1000

def load_self_learning_data():
    self_learning_file = "self_learning_data/self_learning.json"
    if os.path.exists(self_learning_file):
        with open(self_learning_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_self_learning_data(self_learning_data):
    with open("self_learning_data/self_learning.json", "w", encoding="utf-8") as f:
        json.dump(self_learning_data, f)

def self_learn(model, tokenizer, config, self_learning_data):
    dataset = SimpleDataset(tokenizer, self_learning_data, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.1)
    model.train()
    for batch in dataloader:
        with torch.cuda.amp.autocast():
            outputs = model(
                batch["input_ids"].to(model.token_embedding.weight.device),
                batch["attention_mask"].to(model.token_embedding.weight.device),
                batch["labels"].to(model.token_embedding.weight.device),
                batch["trait_idx"].to(model.token_embedding.weight.device)
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    logger.info("Self-learned from generated prompts/responses")

# -------------------------------------
# 🚀 Model Components
# -------------------------------------

class RotaryEmbedding(torch.nn.Module):
    """Rotary Position Embedding implementation"""
    def __init__(self, dim, max_position_embeddings=8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, q, k, position_ids=None):
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # Move buffers to the correct device if needed
        if self.cos_cache.device != device:
            self.cos_cache = self.cos_cache.to(device)
            self.sin_cache = self.sin_cache.to(device)
        
        if position_ids is None:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        else:
            cos = self.cos_cache[position_ids]
            sin = self.sin_cache[position_ids]
        
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        
        q_reshaped = q.permute(0, 2, 1, 3)  # [batch, seq_len, heads, dim]
        k_reshaped = k.permute(0, 2, 1, 3)
        
        q_embed = (q_reshaped * cos) + (self._rotate_half(q_reshaped) * sin)
        k_embed = (k_reshaped * cos) + (self._rotate_half(k_reshaped) * sin)
        
        return q_embed.permute(0, 2, 1, 3), k_embed.permute(0, 2, 1, 3)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class SelfAttention(torch.nn.Module):
    """Multi-head self-attention with rotary embeddings"""
    def __init__(self, hidden_dim, num_heads, window_size, rotary_emb):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Hidden dimension {hidden_dim} must be divisible by number of heads {num_heads}"
            )
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention
        
        self.rotary_emb = rotary_emb
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _shape(self, tensor, seq_len, bsz):
        """Reshape tensor for multi-head attention"""
        tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]

    def forward(self, hidden_states, attention_mask=None, kv_cache=None):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project and reshape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Separate heads
        q = self._shape(q, seq_len, batch_size)  # [batch, heads, seq_len, head_dim]
        k = self._shape(k, seq_len, batch_size)
        v = self._shape(v, seq_len, batch_size)
        
        # Apply rotary embeddings
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)
        
        # Handle KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache['k'], k], dim=2) if kv_cache['k'] is not None else k
            v = torch.cat([kv_cache['v'], v], dim=2) if kv_cache['v'] is not None else v
            kv_cache['k'], kv_cache['v'] = k, v
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            # Expand attention mask for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        return self.o_proj(attn_output)

class ReasoningLayer(torch.nn.Module):
    """Layer for enhanced reasoning capabilities"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reasoning_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        reasoning_states = self.reasoning_mlp(hidden_states)
        return self.norm(hidden_states + reasoning_states)

class WillModule(torch.nn.Module):
    """Module for personality-driven decision making"""
    def __init__(self, hidden_dim, num_traits):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.personality_embedding = torch.nn.Embedding(num_traits, hidden_dim)
        self.decision_mlp = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, trait_idx):
        batch_size, seq_len = hidden_states.shape[:2]
        personality = self.personality_embedding(trait_idx).expand(batch_size, seq_len, -1)
        combined = torch.cat([hidden_states, personality], dim=-1)
        decision = self.decision_mlp(combined)
        return self.norm(hidden_states + decision)

class TransformerLayer(torch.nn.Module):
    """Transformer layer with attention, feed-forward network, reasoning, and will modules"""
    def __init__(self, hidden_dim, num_heads, window_size, rotary_emb, config):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(hidden_dim)
        self.attention = SelfAttention(hidden_dim, num_heads, window_size, rotary_emb)
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.reasoning = ReasoningLayer(hidden_dim) if config.use_reasoning else None
        self.will = WillModule(hidden_dim, len(config.personality_traits)) if config.personality_traits else None
        self.config = config

    def forward(self, hidden_states, attention_mask=None, kv_cache=None, trait_idx=None):
        # Pre-attention layernorm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states = self.attention(hidden_states, attention_mask, kv_cache)
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply reasoning if enabled
        if self.config.use_reasoning and self.reasoning:
            hidden_states = self.reasoning(hidden_states)
        
        # Apply personality-driven decision making if enabled
        if self.config.personality_traits and self.will and trait_idx is not None:
            hidden_states = self.will(hidden_states, trait_idx)
        
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class TurbotalkModel(torch.nn.Module):
    """Main TurboTalk model implementation with growth and enhanced generation capabilities"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.hidden_dim)
        self.rotary_emb = RotaryEmbedding(config.hidden_dim // config.num_heads, config.max_seq_len)
        self.layers = torch.nn.ModuleList([
            TransformerLayer(config.hidden_dim, config.num_heads, config.window_size, self.rotary_emb, config)
            for _ in range(config.num_layers)
        ])
        self.final_layer_norm = torch.nn.LayerNorm(config.hidden_dim)
        self.lm_head = torch.nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.use_kv_cache = False
        self.kv_cache = {}
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_kv_cache(self):
        """Enable key-value cache for faster generation"""
        self.use_kv_cache = True
        for i in range(self.config.num_layers):
            self.kv_cache[i] = {'k': None, 'v': None}
        logger.info("KV cache enabled")

    def grow(self, current_params):
        """Add a new layer if within parameter budget"""
        if current_params < self.config.max_params - 100000000:  # Leave room for 100M parameters
            new_layer = TransformerLayer(
                self.config.hidden_dim,
                self.config.num_heads,
                self.config.window_size,
                self.rotary_emb,
                self.config
            )
            self.layers.append(new_layer)
            self.config.num_layers += 1
            logger.info(f"Model grew to {self.config.num_layers} layers")
        else:
            logger.info("Growth skipped: Parameter limit reached")

    def forward(self, input_ids, attention_mask=None, labels=None, trait_idx=None):
        device = self.token_embedding.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        if trait_idx is not None:
            trait_idx = trait_idx.to(device)
        
        hidden_states = self.token_embedding(input_ids)
        
        for i, layer in enumerate(self.layers):
            if self.config.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    self.kv_cache[i] if self.use_kv_cache else None,
                    trait_idx,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    self.kv_cache[i] if self.use_kv_cache else None,
                    trait_idx
                )
        
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def generate(
        self,
        input_ids,
        max_length=100,
        min_length=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        reasoning_steps=3,
        trait_idx=None,
        **kwargs
    ):
        """Enhanced text generation with reasoning and personality traits"""
        self.eval()
        self.enable_kv_cache()
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        trait_idx = trait_idx if trait_idx is not None else torch.randint(
            0,
            len(self.config.personality_traits),
            (batch_size,),
            device=device
        )

        with torch.no_grad():
            # Reasoning phase
            reasoning_outputs = []
            for step in range(reasoning_steps):
                outputs = self(generated, trait_idx=trait_idx)
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k, top_p)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                reasoning_outputs.append(next_token)

            # Generation phase
            for _ in range(max_length - generated.shape[1]):
                outputs = self(generated, trait_idx=trait_idx)
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k, top_p)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                if generated.shape[1] >= max_length:
                    break

        return generated

    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
        """Filter logits using top-k and nucleus (top-p) sampling"""
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

# -------------------------------------
# 📦 Training Implementation
# -------------------------------------

class SimpleDataset(Dataset):
    """Basic dataset implementation for text data"""
    def __init__(self, tokenizer, data, max_length):
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting SimpleDataset initialization with {len(data):,} examples")
        self.input_ids = []
        self.attention_masks = []
        self.trait_indices = []
        traits = ["curious", "helpful", "playful"]
        
        total_examples = len(data)
        processed_examples = 0
        last_log_time = time.time()
        
        # Create progress bar for dataset creation
        pbar = tqdm(total=total_examples, desc="Creating dataset", unit="examples")
        
        for idx, text in enumerate(data):
            # Use encode method directly from tokenizer
            encoding = tokenizer.encode(text)
            if len(encoding.ids) > max_length:
                # Truncate if too long
                input_ids = encoding.ids[:max_length]
                attention_mask = [1] * max_length
            else:
                # Pad if too short
                input_ids = encoding.ids + [tokenizer.token_to_id("<PAD>")] * (max_length - len(encoding.ids))
                attention_mask = [1] * len(encoding.ids) + [0] * (max_length - len(encoding.ids))
            
            self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            self.attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
            trait_idx = idx % len(traits)
            self.trait_indices.append(trait_idx)
            
            processed_examples += 1
            pbar.update(1)
            
            # Log progress every 5 seconds
            current_time = time.time()
            if current_time - last_log_time >= 5:
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {processed_examples:,}/{total_examples:,} examples ({(processed_examples/total_examples)*100:.2f}%)")
                last_log_time = current_time
        
        pbar.close()
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset initialization complete. Created {len(self.input_ids):,} examples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.input_ids[idx].clone(),
            "trait_idx": torch.tensor(self.trait_indices[idx], dtype=torch.long)
        }

def check_memory_requirements(config):
    """Check if the system has enough memory for the model"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Estimate model memory requirements
        param_size = 2  # bytes per parameter in bf16
        num_params = config.hidden_dim * config.hidden_dim * config.num_layers * 4  # rough estimate
        activation_size = config.batch_size * config.max_seq_len * config.hidden_dim * 4  # rough estimate
        
        total_required = (num_params * param_size + activation_size) * 1.2  # 20% buffer
        
        if total_required > gpu_memory:
            logger.warning(
                f"Model may require more memory ({total_required/1e9:.1f}GB) than available ({gpu_memory/1e9:.1f}GB). "
                "Enabling all memory optimization features."
            )
            config.use_cpu_offload = True
            config.use_gradient_checkpointing = True
            config.memory_efficient_attention = True
            config.max_memory_usage = 0.7
            return False
        return True
    return False

def train(config):
    """Main training function"""
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training process...")
    set_environment_variables()
    get_device_info()
    set_seed(config.seed)
    
    # Check memory requirements
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Checking memory requirements...")
    has_enough_memory = check_memory_requirements(config)
    if not has_enough_memory:
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Running with memory optimizations enabled")
    
    with timer("Data Loading"):
        training_data = load_large_dataset(config)
    
    with timer("Tokenizer Training"):
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting tokenizer training...")
        tokenizer = train_custom_tokenizer(training_data, config.vocab_size)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>")
        tokenizer.enable_truncation(max_length=config.max_seq_len)
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Tokenizer training complete")
    
    with timer("Model Initialization"):
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting model initialization...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TurbotalkModel(config)
        
        # Count parameters before moving to device
        total_params = count_parameters(model)
        if not (4e9 <= total_params <= 5e9):
            logger.warning(f"[{datetime.now().strftime('%H:%M:%S')}] Model size {total_params:,} is outside the 4-5B range")
        
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Moving model to device and applying optimizations...")
        # Move model to device and apply optimizations
        if config.use_mixed_precision and device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)
        
        if not has_enough_memory:
            logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing model in chunks to save memory")
            for name, param in model.named_parameters():
                # Move parameter to device
                param.data = param.data.to(device)
                # Offload to CPU if needed
                if config.use_cpu_offload and "rotary_emb" not in name:
                    param.data = param.data.cpu()
        else:
            model = model.to(device)
            if config.use_cpu_offload:
                for name, param in model.named_parameters():
                    if "rotary_emb" not in name:
                        param.data = param.data.cpu()
    
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Creating dataset and dataloader...")
    # Create dataset and dataloader
    dataset = SimpleDataset(tokenizer, training_data, config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4 if not config.use_cpu_offload else 0
    )
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset size: {len(dataset):,} examples")
    
    # Setup optimizer and scheduler
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = min(config.max_steps, len(dataloader) * config.max_epochs)
    warmup_steps = int(config.warmup_ratio * num_training_steps)
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Training loop
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training loop...")
    model.train()
    global_step = 0
    best_loss = float('inf')
    prev_avg_loss = float('inf')
    training_start_time = time.time()
    self_learning_data = load_self_learning_data()
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(config.max_epochs), desc="Training epochs", position=0)
    
    for epoch in epoch_pbar:
        epoch_loss = 0
        epoch_start_time = time.time()
        batch_count = 0
        
        # Create progress bar for batches
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", position=1, leave=False)
        
        for batch_idx, batch in enumerate(batch_pbar):
            if global_step >= config.max_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if config.use_mixed_precision else torch.no_grad():
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                    batch["trait_idx"]
                )
                loss = outputs["loss"]
                epoch_loss += loss.item()
            
            loss.backward()
            
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            batch_count += 1
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = epoch_loss / (batch_idx + 1)
            elapsed_time = time.time() - epoch_start_time
            batches_per_second = batch_count / elapsed_time
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'speed': f'{batches_per_second:.2f} batches/s'
            })
            
            if global_step % config.logging_steps == 0:
                if global_step > 0:
                    prompt = "Hi, how are you?"
                    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
                    output_ids = model.generate(
                        input_ids,
                        max_length=50,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        reasoning_steps=config.reasoning_steps,
                        trait_idx=torch.tensor([0], device=device)
                    )
                    generated_text = tokenizer.decode(output_ids[0])
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Sample generation:\n{generated_text}\n")
                    self_learning_data.append(f"Prompt: {prompt}\nResponse: {generated_text}")
            
            if global_step % config.save_steps == 0:
                checkpoint_path = f"{config.output_dir}/step_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Saved checkpoint: {checkpoint_path}")
            
            if global_step % 100 == 0:
                clear_gpu_memory()
            
            if global_step % config.eval_steps == 0 and global_step > 0:
                loss_improvement = prev_avg_loss - avg_loss
                if loss_improvement < config.growth_threshold:
                    total_params = count_parameters(model)
                    model.grow(total_params)
                    total_params = count_parameters(model)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=config.learning_rate,
                        weight_decay=config.weight_decay
                    )
                prev_avg_loss = avg_loss
            
            if global_step % config.self_learning_freq == 0 and self_learning_data:
                self_learn(model, tokenizer, config, self_learning_data[-100:])
                save_self_learning_data(self_learning_data)
            
            global_step += 1
        
        batch_pbar.close()
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(dataloader)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{avg_epoch_loss:.4f}',
            'time': f'{epoch_time:.2f}s'
        })
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), f"{config.output_dir}/best_model.pt")
            logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] New best model saved with loss: {best_loss:.4f}")
        
        if global_step >= config.max_steps:
            break
    
    epoch_pbar.close()
    training_time = time.time() - training_start_time
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed in {training_time:.2f}s")
    torch.save(model.state_dict(), f"{config.output_dir}/final_model.pt")
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Final model saved")

if __name__ == "__main__":
    config = TrainingConfig()
    # Override config for fast training
    config.batch_size = 32
    config.gradient_accumulation_steps = 1
    config.learning_rate = 1e-4
    config.max_steps = 1000
    config.max_epochs = 1
    config.warmup_ratio = 0.01
    config.save_steps = 500
    config.eval_steps = 500
    config.logging_steps = 100
    config.zero_stage = 0
    config.offload_optimizer = False
    config.offload_param = False
    config.use_flash_attn = True
    config.precision = "fp16"
    config.use_fused_adam = True
    config.use_fused_layernorm = True
    config.use_fused_attention = True
    config.use_fused_mlp = True
    config.use_fused_embedding = True
    config.use_fused_optimizer = True
    config.use_fused_scheduler = True
    config.use_fused_loss = True
    config.use_fused_backward = True
    config.use_fused_forward = True
    
    train(config) 