"""
Shared configuration for Qwen3 0.6B fine-tuning.
Adjust these before running training.
"""

# Model
BASE_MODEL = "Qwen/Qwen3-0.6B"  # or "Qwen/Qwen3-0.6B-Instruct" for chat-tuned base

# Hub output
HF_USERNAME = ""  # Set your HuggingFace username
OUTPUT_MODEL_NAME = "qwen3-0.6b-chat-finetuned"
HUB_MODEL_ID = f"{HF_USERNAME}/{OUTPUT_MODEL_NAME}" if HF_USERNAME else ""

# Dataset
DATASET_PATH = "./data/raw_text.txt"  # Your raw text file
DATASET_HUB_ID = ""  # If uploading to HF Hub

# Training hyperparameters (optimized for 0.6B model)
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # effective batch = 16
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "bf16": True,
    "gradient_checkpointing": True,
    "max_length": 1024,
    "push_to_hub": True,
}

# LoRA config (optional, for memory efficiency)
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Data preparation
CHUNK_SIZE = 512          # tokens per training example
OVERLAP = 64              # token overlap between chunks
SYSTEM_PROMPT = "You are a helpful assistant."  # System message for chat format

# Hardware
HF_JOBS_FLAVOR = "t4-small"  # ~$0.75/hr, sufficient for 0.6B model
HF_JOBS_TIMEOUT = "2h"
MODAL_GPU = "T4"  # Modal GPU type (T4, A10G, A100)
