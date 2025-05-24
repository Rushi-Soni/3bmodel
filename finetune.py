import json
import os
import random
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import torch
# from torch.distributed.fsdp.fully_sharded_data_parallel import DTensor # Add this import
try:
    # Try the original import path
    from torch.distributed.fsdp.fully_sharded_data_parallel import DTensor
except ImportError:
    try:
        # Try another common import path for DTensor
        from torch.distributed._tensor import DTensor
    except ImportError:
        print("Warning: DTensor could not be imported from common paths. Defining a dummy DTensor class.")
        # Define a dummy DTensor class if it cannot be imported.
        # This is a workaround for potential isinstance checks in save_pretrained.
        class DTensor:
            pass

# Disable problematic features that cause DTensor issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define paths
model_path = r"D:\ttm\model\3bmodel\t\M\TTM\model\snapshots\7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
dataset_path = r"D:\ttm\model\3bmodel\t\M\TTM\autotrain_ready_data.jsonl"
output_dir = r"D:\ttm\model\3bmodel\t\M\TTM\finetuned_model"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model and tokenizer loaded successfully!")

# Load and prepare the dataset
print("Loading dataset...")
def load_jsonl_dataset(file_path, max_samples=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    if max_samples is not None and len(data) > max_samples:
        print(f"Sampling {max_samples} examples from {len(data)} total examples")
        step = len(data) // (max_samples // 2)
        systematic_samples = data[::step][:max_samples//2]
        remaining_indices = set(range(len(data))) - set(range(0, len(data), step)[:max_samples//2])
        random_samples = [data[i] for i in random.sample(list(remaining_indices), max_samples - len(systematic_samples))]
        data = systematic_samples + random_samples
        random.shuffle(data)
    
    return Dataset.from_list(data)

dataset = load_jsonl_dataset(dataset_path)
print(f"Dataset loaded with {len(dataset)} examples")

dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"Training set: {len(dataset['train'])} examples")
print(f"Validation set: {len(dataset['test'])} examples")

# Fixed preprocessing function with proper label masking
print("Tokenizing dataset...")
def preprocess_function(examples):
    inputs = examples["instruction"]
    targets = examples["response"]
    
    # Print a sample to debug
    if random.random() < 0.001:  # Print ~0.1% of examples
        print(f"Sample input: {inputs[0] if isinstance(inputs, list) else inputs}")
        print(f"Sample target: {targets[0] if isinstance(targets, list) else targets}")
    
    # Ensure inputs and targets are strings, not nested lists
    if isinstance(inputs, list) and inputs and isinstance(inputs[0], list):
        inputs = [" ".join(item) if isinstance(item, list) else item for item in inputs]
    
    if isinstance(targets, list) and targets and isinstance(targets[0], list):
        targets = [" ".join(item) if isinstance(item, list) else item for item in targets]
    
    # Tokenize inputs and targets separately
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    
    # Replace padding tokens in labels with -100 (ignored in loss calculation)
    labels_input_ids = labels["input_ids"]
    labels_input_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_input_ids]
    
    model_inputs["labels"] = labels_input_ids
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
print("Dataset tokenized successfully!")

# Define training arguments
# Define training arguments
training_args_dict = {
    "output_dir": output_dir,
    "learning_rate": 1e-5,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "weight_decay": 0.01,
    "save_total_limit": 3,
    "num_train_epochs": 2,
    "predict_with_generate": True,
    "gradient_accumulation_steps": 16,
    "max_grad_norm": 0.5,
    "logging_steps": 10,
    "save_steps": 250,  # Changed from 100 to 10
    "report_to": "none",
    "push_to_hub": False,
    "dataloader_pin_memory": False,
    "remove_unused_columns": True,
    "label_smoothing_factor": 0.1,
}

# Check evaluation and save strategy support
eval_save_supported = (hasattr(Seq2SeqTrainingArguments, "evaluation_strategy") and 
                       hasattr(Seq2SeqTrainingArguments, "save_strategy"))

if eval_save_supported:
    training_args_dict["evaluation_strategy"] = "steps"
    training_args_dict["eval_steps"] = 200
    training_args_dict["save_strategy"] = "steps"
    training_args_dict["save_steps"] = 200
    training_args_dict["metric_for_best_model"] = "eval_loss"
    training_args_dict["greater_is_better"] = False
    
    if hasattr(Seq2SeqTrainingArguments, "load_best_model_at_end"):
        training_args_dict["load_best_model_at_end"] = True
else:
    print("Warning: evaluation_strategy or save_strategy not supported. Disabling load_best_model_at_end.")

if hasattr(Seq2SeqTrainingArguments, "fp16"):
    training_args_dict["fp16"] = True

# Initialize training arguments BEFORE using them
training_args = Seq2SeqTrainingArguments(**training_args_dict)

# Force disable problematic features
training_args.dataloader_num_workers = 0
training_args.past_index = -1

# Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize callbacks
callbacks = []
if hasattr(transformers, "EarlyStoppingCallback"):
    if "metric_for_best_model" in training_args_dict:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    else:
        print("Warning: Cannot use EarlyStoppingCallback without metric_for_best_model. Disabling early stopping.")

# Custom trainer class to handle DTensor issues
class SafeSeq2SeqTrainer(Seq2SeqTrainer):
    def _save(self, output_dir: str, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        # Import DTensor in the method to ensure it's in scope
        try:
            from torch.distributed.fsdp.fully_sharded_data_parallel import DTensor
        except ImportError:
            try:
                from torch.distributed._tensor import DTensor
            except ImportError:
                # Define a dummy DTensor class if it cannot be imported
                class DTensor:
                    pass
        
        if self.args.should_save:
            # Monkey patch the modeling_utils module to recognize our DTensor
            import transformers.modeling_utils
            if not hasattr(transformers.modeling_utils, 'DTensor'):
                transformers.modeling_utils.DTensor = DTensor
            
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

# Initialize trainer
trainer = SafeSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    callbacks=callbacks if callbacks else None,
)

# Validation check before training - replace the current validation check with this
print("Running validation check on small batch...")
try:
    # Create a properly formatted small batch that matches the expected format
    small_batch = tokenized_datasets["train"].select(range(2))
    
    # Run a manual forward pass instead of using predict
    inputs = data_collator([{k: v for k, v in small_batch[i].items()} for i in range(len(small_batch))])
    
    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    
    # Run a test forward pass with no_grad to check for errors
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Test forward pass successful!")
    print(f"Test loss: {outputs.loss.item()}")
    
    # If we get here, the model can process the data correctly
    print("Model and data format validation successful")
    
except Exception as e:
    print(f"Validation check failed with error: {e}")
    print("Attempting to continue with training anyway...")
# test_outputs = trainer.predict(small_batch)
# print(f"Test prediction shape: {test_outputs.predictions.shape}")
# print(f"Test labels shape: {test_outputs.label_ids.shape}")

# if hasattr(test_outputs, 'metrics') and 'test_loss' in test_outputs.metrics:
#     print(f"Test loss: {test_outputs.metrics['test_loss']}")
# else:
#     print("Warning: No loss calculated in test run")

# Train the model
print("Starting fine-tuning...")

# Debug: Check a batch to see what's going on with the inputs and labels
print("Debugging a sample batch...")
try:
    sample_batch = next(iter(trainer.get_train_dataloader()))
    print(f"Batch keys: {sample_batch.keys()}")
    
    # Check if labels exist and are not all -100
    if "labels" in sample_batch:
        labels = sample_batch["labels"]
        non_ignored = (labels != -100).sum().item()
        total = labels.numel()
        print(f"Labels shape: {labels.shape}, Non-ignored tokens: {non_ignored}/{total} ({non_ignored/total:.2%})")
        
        # If all or most tokens are ignored (-100), that would explain zero loss
        if non_ignored / total < 0.1:  # Less than 10% of tokens contribute to loss
            print("WARNING: Very few tokens contribute to loss calculation!")
    else:
        print("WARNING: No 'labels' key in batch!")
        
    # Check input_ids to ensure they're not all padding
    if "input_ids" in sample_batch:
        input_ids = sample_batch["input_ids"]
        non_padding = (input_ids != tokenizer.pad_token_id).sum().item()
        total_inputs = input_ids.numel()
        print(f"Input_ids shape: {input_ids.shape}, Non-padding tokens: {non_padding}/{total_inputs} ({non_padding/total_inputs:.2%})")
    
    # Run a forward pass with this batch to check loss
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in sample_batch.items() if hasattr(v, "to")})
        print(f"Forward pass loss: {outputs.loss.item() if outputs.loss is not None else 'None'}")
        
except Exception as e:
    print(f"Debugging batch failed with error: {e}")

# Custom training loop to debug the zero loss issue
print("\nAttempting manual training for a few steps to debug loss issue...")
try:
    # Set up optimizer manually
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Get a few batches
    train_dataloader = trainer.get_train_dataloader()
    model.train()
    
    # Train for a few steps manually
    for step, batch in enumerate(train_dataloader):
        if step >= 5:  # Just do a few steps to debug
            break
            
        # Move batch to device
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        print(f"Manual step {step}, raw loss: {loss.item()}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm}")
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
    print("Manual training completed successfully!")
    print("Now continuing with the regular trainer...")
    
except Exception as e:
    print(f"Manual training failed with error: {e}")
    print("Continuing with regular training...")

# Continue with regular training
trainer.train()
print("Fine-tuning completed!")

# Save the model
print("Saving the fine-tuned model...")
model.save_pretrained(output_dir, safe_serialization=False)
tokenizer.save_pretrained(output_dir)

# Modify model metadata to remove references to the original model
print("Modifying model metadata to remove all traces of the original model...")

# Update config.json
config_path = os.path.join(output_dir, "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update model metadata
    config["model_type"] = "turbotalk_ai"
    config["architectures"] = ["TurboTalkAIForConditionalGeneration"]
    config["_name_or_path"] = "turbotalk_ai"
    config["model_name"] = "TurboTalk AI"
    config["company"] = "Rango Productions"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("Updated config.json")

# Update tokenizer_config.json
tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
if os.path.exists(tokenizer_config_path):
    with open(tokenizer_config_path, 'r') as f:
        tokenizer_config = json.load(f)
    
    # Update tokenizer metadata
    if "model_name" in tokenizer_config:
        tokenizer_config["model_name"] = "turbotalk_ai"
    tokenizer_config["name_or_path"] = "turbotalk_ai"
    tokenizer_config["auto_map"] = {"AutoTokenizer": ["turbotalk_ai", None]}
    
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print("Updated tokenizer_config.json")

# Update generation_config.json
gen_config_path = os.path.join(output_dir, "generation_config.json")
if os.path.exists(gen_config_path):
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    
    # Update generation config metadata
    if "_name_or_path" in gen_config:
        gen_config["_name_or_path"] = "turbotalk_ai"
    
    with open(gen_config_path, 'w') as f:
        json.dump(gen_config, f, indent=2)
    print("Updated generation_config.json")

# Update special_tokens_map.json
special_tokens_path = os.path.join(output_dir, "special_tokens_map.json")
if os.path.exists(special_tokens_path):
    with open(special_tokens_path, 'r') as f:
        special_tokens = json.load(f)
    
    # No specific fields to update, but ensure no references remain
    
    with open(special_tokens_path, 'w') as f:
        json.dump(special_tokens, f, indent=2)
    print("Updated special_tokens_map.json")

# Create a model card README.md
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, 'w') as f:
    f.write("""# TurboTalk AI

A powerful language model developed by Rango Productions.

## Model Description

TurboTalk AI is designed to provide helpful, accurate, and engaging responses to a wide variety of queries.

## Usage

```python
from transformers import pipeline

pipe = pipeline("text2text-generation", model="turbotalk_ai")
response = pipe("What is your name?")
print(response[0]['generated_text'])
# Output: My name is TurboTalk AI.
```""")
print("Created README.md")

print("\nFine-tuning and metadata modification complete!")
print(f"Fine-tuned model saved to: {output_dir}")
print("All traces of the original model have been removed from the metadata.")
print("The model is now branded as 'TurboTalk AI' by 'Rango Productions'.")