# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.15.0", "datasets>=3.0.0", "trackio"]
# ///
"""
Qwen3 0.6B SFT Training on HuggingFace Jobs.

This script is designed to run on HF Jobs infrastructure.
Upload your dataset to HF Hub first, then submit this job.

Usage (submit via hf_jobs MCP tool or CLI):
    hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN train_hf_jobs.py

Or submit inline via MCP:
    hf_jobs("uv", {"script": "<this file content>", "flavor": "t4-small", ...})
"""

import os
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import trackio

# ============================================================
# CONFIGURE THESE - adjust for your dataset and model
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = ""  # e.g., "username/my-dataset" on HF Hub, or path to local file
HUB_MODEL_ID = ""  # e.g., "username/qwen3-0.6b-chat-finetuned"

# Set via environment or hardcode
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Training config
TRAINING_ARGS = SFTConfig(
    output_dir="qwen3-0.6b-chat-finetuned",
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
    # Hub push (CRITICAL - environment is ephemeral)
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="every_save",
    # Trackio monitoring
    report_to="trackio",
    project="qwen3-finetune",
    run_name="qwen3-0.6b-sft-chat",
)


# ============================================================
# TRAINING
# ============================================================

def main():
    print(f"Loading model: {MODEL_NAME}")

    # Load dataset
    if DATASET_NAME.endswith(".json") or DATASET_NAME.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=DATASET_NAME, split="train")
    else:
        dataset = load_dataset(DATASET_NAME, split="train")

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    # Split train/eval if not already split
    if "eval" not in DATASET_NAME:
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 'None'}")

    # Initialize trainer
    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model()
    trainer.push_to_hub()

    print(f"Training complete! Model pushed to: {HUB_MODEL_ID}")


if __name__ == "__main__":
    main()
