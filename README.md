# Qwen3 0.6B SFT Fine-Tuning Pipeline

A clean, minimalist pipeline template for fine-tuning the **Qwen3 0.6B** model using Supervised Fine-Tuning (SFT). This repository contains code to prepare raw text datasets and execute training either on **Modal Labs** serverless GPUs or **Hugging Face Jobs**.

## Project Structure

* `config.py` - Shared training configurations, hyperparameters, LoRA target modules, and cloud setup parameters.
* `prepare_data.py` - Utility to convert unstructured raw text into SFT-compatible chat/message format JSONL files.
* `modal_app.py` - Cloud training application designed for serverless orchestration on Modal GPU nodes (e.g., NVIDIA T4).
* `train_hf_jobs.py` - Script configured to run training workflows on Hugging Face Jobs infrastructure.
* `requirements.txt` - Dependency file for core training, cloud drivers, and monitoring integrations.

---

## Getting Started

### 1. Installation
Install the required dependencies locally:
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
1. Place your raw text knowledge base, handbook, or QA text file in `./data/raw_text.txt`.
2. Convert it into the chat messages format:
   ```bash
   python prepare_data.py --input ./data/raw_text.txt --output ./data/train.jsonl
   ```
   *This automatically generates `data/train.jsonl` and `data/eval.jsonl` files using a sliding-window chunking or structure detection strategy.*

---

## Running Training

### Path A: Modal Labs (Recommended)
Modal provides cost-efficient, on-demand GPUs (e.g. NVIDIA T4 or A10G) without needing to configure complex server environments.

1. Install Modal and log in:
   ```bash
   pip install modal
   modal token new
   ```
2. Set your Hugging Face API key as a Modal secret (optional, for model uploading):
   ```bash
   modal secret create huggingface HF_TOKEN=your_token_here
   ```
3. Launch the training app:
   ```bash
   modal run modal_app.py
   ```

### Path B: Hugging Face Jobs
To train on Hugging Face's serverless jobs infrastructure:

1. Upload your prepared JSONL datasets to the Hugging Face Hub.
2. Update the configuration variables in `train_hf_jobs.py`.
3. Submit the job using the `hf` CLI or API:
   ```bash
   hf jobs uv run --flavor t4-small --timeout 2h --secrets HF_TOKEN train_hf_jobs.py
   ```
