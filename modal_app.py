"""
Qwen3 0.6B SFT Training on Modal Cloud GPU.

Usage:
    modal run modal_app.py          # Train locally, run on Modal
    modal deploy modal_app.py       # Deploy as persistent app

Prerequisites:
    pip install modal
    modal token new                 # Authenticate with Modal
"""

import modal

# ============================================================
# MODAL SETUP
# ============================================================

app = modal.App("qwen3-finetune")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.45.0",
        "trl>=0.12.0",
        "peft>=0.15.0",
        "datasets>=3.0.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
    )
)

# Persistent volume for model cache and outputs
volume = modal.Volume.from_name("qwen3-finetune-data", create_if_missing=True)

# Secrets (set via: modal secret create huggingface HF_TOKEN=<your-token>)
hf_secret = modal.Secret.from_name("huggingface")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"
HUB_MODEL_ID = ""  # Set: "username/qwen3-0.6b-chat-finetuned"
DATASET_PATH = "/data/train.jsonl"  # Path inside the volume
EVAL_PATH = "/data/eval.jsonl"

TRAINING_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="every_save",
)


# ============================================================
# UPLOAD DATA (run locally to push dataset to Modal volume)
# ============================================================

@app.function(
    image=image,
    volumes={"/data": volume},
)
def upload_data(train_jsonl: str, eval_jsonl: str):
    """Upload local dataset files to Modal volume."""
    import shutil

    shutil.copy(train_jsonl, "/data/train.jsonl")
    shutil.copy(eval_jsonl, "/data/eval.jsonl")
    volume.commit()
    print("Dataset uploaded to Modal volume.")


# ============================================================
# TRAINING FUNCTION
# ============================================================

@app.function(
    image=image,
    gpu="T4",             # T4 is sufficient for 0.6B model
    volumes={"/data": volume},
    secrets=[hf_secret],
    timeout=7200,          # 2 hours
    memory=16384,          # 16 GB RAM
)
def train():
    """Fine-tune Qwen3 0.6B with SFT on Modal GPU."""
    import os
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    train_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_PATH, split="train")

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Configure training
    training_args = SFTConfig(
        output_dir="/data/output",
        **TRAINING_CONFIG,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    train_result = trainer.train()

    print(f"Training complete!")
    print(f"  Loss: {train_result.training_loss:.4f}")

    # Save to volume
    trainer.save_model("/data/output/final")
    tokenizer.save_pretrained("/data/output/final")
    volume.commit()

    # Push to Hub
    if HUB_MODEL_ID and hf_token:
        print(f"Pushing to Hub: {HUB_MODEL_ID}")
        trainer.push_to_hub()
        print("Pushed to Hub successfully!")

    return {
        "status": "success",
        "loss": train_result.training_loss,
        "hub_model_id": HUB_MODEL_ID,
    }


# ============================================================
# ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main(train_path: str = "./data/train.jsonl", eval_path: str = "./data/eval.jsonl"):
    """Upload data and start training."""
    print("Uploading dataset to Modal volume...")
    upload_data.remote(train_path, eval_path)

    print("Starting training on Modal GPU...")
    result = train.remote()
    print(f"Result: {result}")
