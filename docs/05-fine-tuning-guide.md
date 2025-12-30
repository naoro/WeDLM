# Comprehensive Fine-Tuning Guide for WeDLM

## Table of Contents
1. [Overview](#overview)
2. [HuggingFace Integration](#huggingface-integration)
3. [Full Fine-Tuning](#full-fine-tuning)
4. [LoRA Fine-Tuning](#lora-fine-tuning)
5. [QLoRA Fine-Tuning](#qlora-fine-tuning)
6. [Instruction Tuning](#instruction-tuning)
7. [Memory Requirements](#memory-requirements)
8. [Training to Inference Pipeline](#training-to-inference-pipeline)
9. [Best Practices](#best-practices)

---

## Overview

**Good News:** WeDLM supports **full fine-tuning** through standard HuggingFace APIs!

**Key Points:**
- ✅ Use HF `AutoModelForCausalLM` for training
- ✅ Standard PyTorch/HF Trainer workflow
- ✅ Supports LoRA, QLoRA, full fine-tuning
- ✅ After training, load into `wedlm` engine for fast inference
- ⚠️ HF interface is for **training only** (slow inference)
- ⚠️ For fast inference, must use `wedlm.LLM` engine

**Architecture:**
```
Training Phase:              Inference Phase:
┌─────────────────┐         ┌─────────────────┐
│ HF Interface    │         │  wedlm Engine   │
│ AutoModelFor    │  →→→→   │  LLM()          │
│ CausalLM        │  save   │                 │
│                 │  load   │  3-6× faster    │
│ (slow, full     │         │  (optimized)    │
│  features)      │         │                 │
└─────────────────┘         └─────────────────┘
```

---

## HuggingFace Integration

### Model Architecture

**File:** `hf_compat/modeling_wedlm.py`

WeDLM provides a `PreTrainedModel` implementation with:
- Standard forward pass (returns loss if labels provided)
- Gradient checkpointing support
- Cache management (KV cache for generation)
- Full compatibility with HF Trainer

### Loading for Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model for training
model = AutoModelForCausalLM.from_pretrained(
    "tencent/WeDLM-8B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Recommended for A100/H100
    device_map="auto",  # Auto multi-GPU distribution
)

tokenizer = AutoTokenizer.from_pretrained(
    "tencent/WeDLM-8B-Instruct",
    trust_remote_code=True
)

# Check model size
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# Output:
# Total params: 8,030,261,248
# Trainable params: 8,030,261,248
```

### Configuration Access

```python
from hf_compat.configuration_wedlm import WeDLMConfig

config = WeDLMConfig.from_pretrained("tencent/WeDLM-8B-Instruct")

print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num heads: {config.num_attention_heads}")
print(f"Vocab size: {config.vocab_size}")
print(f"Mask token ID: {config.mask_token_id}")

# Output:
# Hidden size: 4096
# Num layers: 32
# Num heads: 32
# Vocab size: 151936
# Mask token ID: 151665
```

---

## Full Fine-Tuning

### Memory Requirements

**WeDLM-7B Full Fine-Tuning:**

```
Model Parameters: 7.6B
Precision: BF16 (2 bytes per param)

┌─────────────────────────┬────────────────┬──────────────┐
│ Component               │ Memory (GB)    │ Notes        │
├─────────────────────────┼────────────────┼──────────────┤
│ Model Weights           │ 15.2           │ 7.6B × 2     │
│ Gradients               │ 15.2           │ 7.6B × 2     │
│ Optimizer States (Adam) │ 60.8           │ 7.6B × 8     │
│ Activations (batch=4)   │ ~8             │ Depends on   │
│                         │                │ seq length   │
│ Buffer/Workspace        │ ~4             │ Misc         │
├─────────────────────────┼────────────────┼──────────────┤
│ TOTAL                   │ ~103 GB        │              │
└─────────────────────────┴────────────────┴──────────────┘

Required GPU: 2× A100 80GB or 2× H100 80GB (with DeepSpeed ZeRO-2)
              1× H100 94GB (without DeepSpeed)
```

**WeDLM-8B Full Fine-Tuning:**

```
Model Parameters: 8.0B

┌─────────────────────────┬────────────────┬──────────────┐
│ Component               │ Memory (GB)    │ Notes        │
├─────────────────────────┼────────────────┼──────────────┤
│ Model Weights           │ 16.0           │ 8.0B × 2     │
│ Gradients               │ 16.0           │ 8.0B × 2     │
│ Optimizer States (Adam) │ 64.0           │ 8.0B × 8     │
│ Activations (batch=4)   │ ~8             │              │
│ Buffer/Workspace        │ ~4             │              │
├─────────────────────────┼────────────────┼──────────────┤
│ TOTAL                   │ ~108 GB        │              │
└─────────────────────────┴────────────────┴──────────────┘

Required GPU: 2× A100 80GB or 2× H100 80GB (with DeepSpeed ZeRO-2)
```

### Full Fine-Tuning Code

```python
#!/usr/bin/env python
"""Full fine-tuning of WeDLM-8B."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ========== Configuration ==========
MODEL_NAME = "tencent/WeDLM-8B-Instruct"
OUTPUT_DIR = "./wedlm-8b-finetuned"
DATASET_NAME = "your_dataset"  # Replace with your dataset

# ========== Load Model and Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Distribute across GPUs automatically
    use_cache=False,    # Disable KV cache during training
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# ========== Prepare Dataset ==========
dataset = load_dataset(DATASET_NAME)

def tokenize_function(examples):
    """Tokenize examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# ========== Training Configuration ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=1,      # Small batch per GPU
    gradient_accumulation_steps=16,     # Effective batch = 16
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,

    # Optimization
    bf16=True,                          # Use BF16 precision
    gradient_checkpointing=True,        # Save memory
    optim="adamw_torch",                # Optimizer

    # Logging and saving
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,

    # Evaluation (if you have validation set)
    evaluation_strategy="steps" if "validation" in dataset else "no",
    eval_steps=500 if "validation" in dataset else None,

    # DeepSpeed (optional, for multi-GPU)
    # deepspeed="ds_config.json",  # Uncomment if using DeepSpeed
)

# ========== Data Collator ==========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM (not masked LM)
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation"),
    data_collator=data_collator,
)

# ========== Train ==========
print("Starting training...")
trainer.train()

# ========== Save Model ==========
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")
```

### DeepSpeed Configuration (Optional)

**File:** `ds_config.json`

```json
{
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

**Usage:**
```bash
# Multi-GPU training with DeepSpeed
deepspeed --num_gpus=2 train_full.py
```

---

## LoRA Fine-Tuning

### Memory Requirements

**WeDLM-7B LoRA Fine-Tuning:**

```
Base Model: 7.6B params (frozen)
LoRA Adapters: r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj]

Per layer: 4 adapters × (hidden_size × r + r × hidden_size)
         = 4 × (4096 × 16 + 16 × 4096)
         = 4 × 131,072 = 524,288 params per layer

Total LoRA: 524,288 × 32 layers = 16,777,216 params (~17M)

┌─────────────────────────┬────────────────┬──────────────┐
│ Component               │ Memory (GB)    │ Notes        │
├─────────────────────────┼────────────────┼──────────────┤
│ Base Model (frozen)     │ 15.2           │ BF16         │
│ LoRA Adapters           │ 0.034          │ Trainable    │
│ LoRA Gradients          │ 0.034          │              │
│ Optimizer (Adam)        │ 0.27           │ 8× params    │
│ Activations (batch=8)   │ ~6             │              │
│ Buffer/Workspace        │ ~2             │              │
├─────────────────────────┼────────────────┼──────────────┤
│ TOTAL                   │ ~24 GB         │              │
└─────────────────────────┴────────────────┴──────────────┘

Required GPU: 1× RTX 4090 24GB or 1× A100 40GB
```

**WeDLM-8B LoRA Fine-Tuning:**

```
Total LoRA: ~18M params

┌─────────────────────────┬────────────────┬──────────────┐
│ Component               │ Memory (GB)    │ Notes        │
├─────────────────────────┼────────────────┼──────────────┤
│ Base Model (frozen)     │ 16.0           │ BF16         │
│ LoRA Adapters           │ 0.036          │ Trainable    │
│ LoRA Gradients          │ 0.036          │              │
│ Optimizer (Adam)        │ 0.29           │              │
│ Activations (batch=8)   │ ~6             │              │
│ Buffer/Workspace        │ ~2             │              │
├─────────────────────────┼────────────────┼──────────────┤
│ TOTAL                   │ ~25 GB         │              │
└─────────────────────────┴────────────────┴──────────────┘

Required GPU: 1× RTX 4090 24GB or 1× A100 40GB
```

### LoRA Fine-Tuning Code

```python
#!/usr/bin/env python
"""LoRA fine-tuning of WeDLM-8B."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ========== Configuration ==========
MODEL_NAME = "tencent/WeDLM-8B-Instruct"
OUTPUT_DIR = "./wedlm-8b-lora"
DATASET_NAME = "your_dataset"

# ========== Load Base Model ==========
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
)

# ========== LoRA Configuration ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                    # LoRA rank (higher = more capacity, more memory)
    lora_alpha=32,           # LoRA scaling factor (typically 2× rank)
    lora_dropout=0.05,       # Dropout for regularization
    target_modules=[         # Which modules to apply LoRA to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # Optionally add MLP layers:
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ],
    bias="none",             # Don't train biases
    modules_to_save=None,    # Additional modules to train (e.g., ["lm_head"])
)

# Apply LoRA to base model
model = get_peft_model(base_model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output:
# trainable params: 16,777,216 || all params: 8,047,038,464 || trainable%: 0.21%

# ========== Prepare Dataset ==========
dataset = load_dataset(DATASET_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# ========== Training Configuration ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=8,      # Larger batch (less memory needed)
    gradient_accumulation_steps=4,
    learning_rate=3e-4,                 # Higher LR for LoRA
    weight_decay=0.01,
    warmup_steps=100,

    # Optimization
    bf16=True,
    optim="adamw_torch",

    # Logging and saving
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,

    # Evaluation
    evaluation_strategy="steps" if "validation" in dataset else "no",
    eval_steps=500 if "validation" in dataset else None,
)

# ========== Data Collator ==========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation"),
    data_collator=data_collator,
)

# ========== Train ==========
print("Starting LoRA training...")
trainer.train()

# ========== Save LoRA Adapters ==========
print(f"Saving LoRA adapters to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training complete!")
print(f"Adapter size: ~{16.8 * 2:.1f} MB")
```

### Merging LoRA Back to Base Model

```python
#!/usr/bin/env python
"""Merge LoRA adapters back into base model."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "tencent/WeDLM-8B-Instruct"
LORA_DIR = "./wedlm-8b-lora"
OUTPUT_DIR = "./wedlm-8b-merged"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Merge on CPU to save GPU memory
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, LORA_DIR)

# Merge adapters into base model
print("Merging LoRA adapters...")
merged_model = model.merge_and_unload()

# Save merged model
print(f"Saving merged model to {OUTPUT_DIR}")
merged_model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Merge complete!")
```

---

## QLoRA Fine-Tuning

### Memory Requirements

**WeDLM-7B QLoRA (4-bit quantization):**

```
Base Model: 7.6B params (4-bit quantized)
LoRA: r=16

┌─────────────────────────┬────────────────┬──────────────┐
│ Component               │ Memory (GB)    │ Notes        │
├─────────────────────────┼────────────────┼──────────────┤
│ Base Model (4-bit)      │ 4.75           │ 7.6B × 0.625 │
│ LoRA Adapters (BF16)    │ 0.034          │              │
│ LoRA Gradients          │ 0.034          │              │
│ Optimizer (Adam)        │ 0.27           │              │
│ Activations (batch=16)  │ ~4             │              │
│ Dequant workspace       │ ~2             │              │
├─────────────────────────┼────────────────┼──────────────┤
│ TOTAL                   │ ~11 GB         │              │
└─────────────────────────┴────────────────┴──────────────┘

Required GPU: 1× RTX 3090 24GB, RTX 4080 16GB, or better
```

**WeDLM-8B QLoRA:**

```
┌─────────────────────────┬────────────────┬──────────────┐
│ Component               │ Memory (GB)    │ Notes        │
├─────────────────────────┼────────────────┼──────────────┤
│ Base Model (4-bit)      │ 5.0            │              │
│ LoRA Adapters (BF16)    │ 0.036          │              │
│ LoRA Gradients          │ 0.036          │              │
│ Optimizer (Adam)        │ 0.29           │              │
│ Activations (batch=16)  │ ~4             │              │
│ Dequant workspace       │ ~2             │              │
├─────────────────────────┼────────────────┼──────────────┤
│ TOTAL                   │ ~12 GB         │              │
└─────────────────────────┴────────────────┴──────────────┘

Required GPU: 1× RTX 3090 24GB, RTX 4080 16GB, or better
```

### QLoRA Fine-Tuning Code

```python
#!/usr/bin/env python
"""QLoRA (4-bit) fine-tuning of WeDLM-8B."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ========== Configuration ==========
MODEL_NAME = "tencent/WeDLM-8B-Instruct"
OUTPUT_DIR = "./wedlm-8b-qlora"
DATASET_NAME = "your_dataset"

# ========== Quantization Configuration ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 4-bit quantization
    bnb_4bit_quant_type="nf4",          # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=True,     # Double quantization for extra savings
)

# ========== Load Quantized Model ==========
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# ========== LoRA Configuration ==========
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== Prepare Dataset ==========
dataset = load_dataset(DATASET_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# ========== Training Configuration ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=16,     # Can use larger batch with QLoRA
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=100,

    # Optimization
    bf16=True,
    optim="paged_adamw_8bit",           # 8-bit optimizer for extra savings

    # Logging
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
)

# ========== Data Collator ==========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# ========== Train ==========
print("Starting QLoRA training...")
trainer.train()

# ========== Save ==========
print(f"Saving QLoRA adapters to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("QLoRA training complete!")
```

---

## Instruction Tuning

### Memory Requirements

Same as full/LoRA/QLoRA (depends on method chosen).

### Instruction Tuning Dataset Format

```python
# Example: Alpaca-style format
{
    "instruction": "Solve the equation: 2x + 5 = 13",
    "input": "",
    "output": "To solve 2x + 5 = 13:\n1. Subtract 5: 2x = 8\n2. Divide by 2: x = 4\n\nAnswer: x = 4"
}

# Example: Chat format
{
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
}
```

### Instruction Tuning Code

```python
#!/usr/bin/env python
"""Instruction tuning with LoRA."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, Sequence

# ========== Configuration ==========
MODEL_NAME = "tencent/WeDLM-8B"  # Use base model, not instruct
OUTPUT_DIR = "./wedlm-8b-instruct-custom"

# ========== Load Model and Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    "tencent/WeDLM-8B-Instruct",  # Use instruct tokenizer for chat template
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ========== Load Instruction Dataset ==========
# Example: Using Alpaca dataset
dataset = load_dataset("tatsu-lab/alpaca")

def format_instruction(example):
    """Format instruction as chat."""
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    # Combine instruction and input
    if input_text:
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction

    # Format as chat
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output_text}
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": formatted}

# Format dataset
formatted_dataset = dataset.map(
    format_instruction,
    remove_columns=dataset["train"].column_names
)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# ========== Custom Data Collator ==========
@dataclass
class DataCollatorForSupervisedDataset:
    """Collator for supervised fine-tuning."""
    tokenizer: AutoTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["input_ids"].copy() for instance in instances]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# ========== Training ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

---

## Memory Requirements

### Complete Summary Table

```
┌──────────────────┬────────┬─────────────┬───────────────┬─────────────────────────┐
│ Method           │ Model  │ Trainable   │ GPU Memory    │ Recommended GPU         │
│                  │        │ Params      │ Required      │                         │
├──────────────────┼────────┼─────────────┼───────────────┼─────────────────────────┤
│ Full FT          │ 7B     │ 7.6B (100%) │ ~103 GB       │ 2× A100 80GB + ZeRO-2   │
│                  │ 8B     │ 8.0B (100%) │ ~108 GB       │ 2× A100 80GB + ZeRO-2   │
├──────────────────┼────────┼─────────────┼───────────────┼─────────────────────────┤
│ LoRA (r=16)      │ 7B     │ 17M (0.22%) │ ~24 GB        │ 1× RTX 4090 24GB        │
│                  │ 8B     │ 18M (0.22%) │ ~25 GB        │ 1× A100 40GB            │
├──────────────────┼────────┼─────────────┼───────────────┼─────────────────────────┤
│ LoRA (r=64)      │ 7B     │ 67M (0.88%) │ ~26 GB        │ 1× A100 40GB            │
│                  │ 8B     │ 71M (0.89%) │ ~28 GB        │ 1× A100 40GB            │
├──────────────────┼────────┼─────────────┼───────────────┼─────────────────────────┤
│ QLoRA (r=16)     │ 7B     │ 17M (0.22%) │ ~11 GB        │ 1× RTX 3090 24GB        │
│                  │ 8B     │ 18M (0.22%) │ ~12 GB        │ 1× RTX 4080 16GB        │
├──────────────────┼────────┼─────────────┼───────────────┼─────────────────────────┤
│ QLoRA (r=64)     │ 7B     │ 67M (0.88%) │ ~13 GB        │ 1× RTX 4090 24GB        │
│                  │ 8B     │ 71M (0.89%) │ ~14 GB        │ 1× RTX 4090 24GB        │
└──────────────────┴────────┴─────────────┴───────────────┴─────────────────────────┘
```

### Batch Size Impact on Memory

```python
# For LoRA r=16 on WeDLM-8B
base_memory = 16.0 + 0.07 + 0.29 = 16.36 GB  # Model + LoRA + optimizer

per_sample_activation = 0.75 GB  # Approximate (seq_len=2048)

batch_1:  16.36 + 1 * 0.75 = 17.11 GB  ✓ Fits in RTX 4090 24GB
batch_4:  16.36 + 4 * 0.75 = 19.36 GB  ✓ Fits
batch_8:  16.36 + 8 * 0.75 = 22.36 GB  ✓ Fits
batch_12: 16.36 + 12 * 0.75 = 25.36 GB ✗ OOM on RTX 4090

Recommendation: Use gradient accumulation if you need larger effective batch size
```

---

## Training to Inference Pipeline

### Complete Workflow

```python
# ========== Step 1: Train with HuggingFace ==========
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained(
    "tencent/WeDLM-8B-Instruct",
    trust_remote_code=True
)

# ... training code ...

trainer.train()
model.save_pretrained("./my-finetuned-wedlm")

# ========== Step 2: Test with HuggingFace (SLOW) ==========
model = AutoModelForCausalLM.from_pretrained(
    "./my-finetuned-wedlm",
    trust_remote_code=True
)

# This works but is SLOW (no optimizations)
# Use only for validation

# ========== Step 3: Production Inference with wedlm Engine ==========
from wedlm import LLM, SamplingParams

# Load your fine-tuned model into wedlm engine
llm = LLM(
    model="./my-finetuned-wedlm",
    wedlm_window_size=16,
    gpu_memory_utilization=0.9,
)

# Fast inference (3-6× faster than vLLM)
outputs = llm.generate(
    prompts,
    SamplingParams(
        temperature=0.0,
        max_tokens=512,
        wedlm_entropy_threshold=0.4,
    )
)
```

### Compatibility Notes

**✅ Compatible:**
- Model weights are identical (same format)
- wedlm engine can load HF-trained checkpoints
- No conversion needed

**⚠️ Important:**
- Mask token ID must match (default: 151665)
- Config must have `wedlm_window_size` (auto-added if missing)
- Tokenizer must be compatible with base model

---

## Best Practices

### 1. Choose the Right Method

```
┌─────────────────────────┬─────────────────────────────────────┐
│ Scenario                │ Recommended Method                  │
├─────────────────────────┼─────────────────────────────────────┤
│ Large dataset (>100K)   │ Full fine-tuning or LoRA r=64       │
│ Medium dataset (10-100K)│ LoRA r=16-32                        │
│ Small dataset (<10K)    │ LoRA r=8-16 or QLoRA                │
│ Limited GPU (< 24GB)    │ QLoRA                               │
│ Domain adaptation       │ Full fine-tuning (if possible)      │
│ Task-specific tuning    │ LoRA r=16                           │
│ Instruction following   │ LoRA r=16-32 on instruct base       │
└─────────────────────────┴─────────────────────────────────────┘
```

### 2. Hyperparameter Recommendations

```python
# Full Fine-Tuning
learning_rate = 2e-5
num_epochs = 3
batch_size = 1-2 per GPU
gradient_accumulation = 16

# LoRA
learning_rate = 3e-4  # Higher than full FT
num_epochs = 3-5
batch_size = 4-8 per GPU
lora_r = 16 (standard), 32-64 (complex tasks)
lora_alpha = 2 * lora_r  # Common heuristic

# QLoRA
learning_rate = 3e-4
num_epochs = 3-5
batch_size = 8-16 per GPU
```

### 3. Data Preparation

```python
# Good practices for dataset preparation

# 1. Remove duplicate examples
dataset = dataset.unique(column="text")

# 2. Filter by length (remove too short/long)
def filter_length(example):
    return 50 < len(example["text"]) < 2048

dataset = dataset.filter(filter_length)

# 3. Balance dataset (if classification/instruction)
# Use stratified sampling if needed

# 4. Shuffle thoroughly
dataset = dataset.shuffle(seed=42)

# 5. Add special tokens if needed
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))
```

### 4. Validation During Training

```python
# Always use validation set for monitoring

training_args = TrainingArguments(
    # ...
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Custom metrics
from transformers import EvalPrediction
import numpy as np

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    # Compute perplexity
    loss = np.mean(predictions)
    perplexity = np.exp(loss)
    return {"perplexity": perplexity}

trainer = Trainer(
    # ...
    compute_metrics=compute_metrics,
)
```

### 5. Checkpoint Management

```python
# Save best checkpoints only
training_args = TrainingArguments(
    # ...
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # Keep only 2 best checkpoints
    load_best_model_at_end=True,
)

# Resume from checkpoint
trainer.train(resume_from_checkpoint="./checkpoint-1000")
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
Solutions:
- Reduce batch size (per_device_train_batch_size)
- Increase gradient accumulation
- Enable gradient checkpointing
- Use DeepSpeed ZeRO
- Switch to LoRA or QLoRA
- Reduce sequence length (max_length)
```

**2. Training Loss Not Decreasing**
```python
Solutions:
- Check learning rate (try 1e-5 to 5e-4 range)
- Verify data quality and format
- Check for label masking (-100 for padding)
- Increase training epochs
- Reduce regularization (dropout, weight_decay)
```

**3. Model Output Degraded After Fine-Tuning**
```python
Solutions:
- Reduce learning rate (catastrophic forgetting)
- Use smaller LoRA rank
- Add more diverse training data
- Use fewer training epochs
- Check if mask token ID changed
```

**4. Slow Training Speed**
```python
Solutions:
- Enable BF16/FP16
- Use gradient checkpointing
- Increase batch size with gradient accumulation
- Use DeepSpeed
- Check data loading (use num_workers > 0)
```

---

## Summary

**Fine-Tuning WeDLM is Fully Supported:**

✅ Use HuggingFace `AutoModelForCausalLM` for training
✅ Supports full FT, LoRA, QLoRA
✅ After training, load into `wedlm.LLM` for fast inference
✅ Memory requirements: 11 GB (QLoRA) to 108 GB (full FT)
✅ No special modifications needed (standard HF workflow)

**Recommended Path:**
1. Start with LoRA r=16 (good balance)
2. Train with HF Trainer
3. Validate with HF interface
4. Deploy with wedlm engine for 3-6× speedup

**Key Insight:** WeDLM combines the flexibility of standard HF training with production-grade inference performance!
