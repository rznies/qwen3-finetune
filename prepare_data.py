"""
Convert raw text into SFT-compatible chat format for Qwen3 fine-tuning.

Input:  A plain text file (raw_text.txt)
Output: A JSON dataset in chat/messages format compatible with TRL SFTTrainer.

Usage:
    python prepare_data.py --input ./data/raw_text.txt --output ./data/train.jsonl
    python prepare_data.py --input ./data/raw_text.txt --output ./data/train.jsonl --chunk-size 512
"""

import argparse
import json
import re
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split raw text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:  # Skip tiny fragments
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def detect_sections(text: str) -> list[dict]:
    """
    Try to detect natural sections (headings, Q&A pairs, numbered items).
    Falls back to chunking if no structure found.
    """
    # Pattern: lines that look like headings or questions
    heading_pattern = re.compile(
        r'^(#{1,4}\s+.+|[A-Z][A-Z\s]{5,}|'  # Markdown headings or ALL CAPS
        r'\d+[\.\)]\s+.+|'                     # Numbered items
        r'Q[\.\):]\s+.+|'                       # Q: or Q. questions
        r'(?:What|How|Why|When|Where|Who|Is|Are|Can|Do|Does)\s+.+\??$)',  # Questions
        re.MULTILINE
    )

    matches = list(heading_pattern.finditer(text))

    if len(matches) < 3:
        return []  # No clear structure, use chunking

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = match.group().strip()
        content = text[start:end].strip()

        # Strip the header from content if it's at the start
        if content.startswith(header):
            content = content[len(header):].strip()

        if len(content) > 20:
            sections.append({"header": header, "content": content})

    return sections


def sections_to_messages(sections: list[dict], system_prompt: str) -> list[dict]:
    """Convert detected sections into chat message format."""
    dataset = []
    for section in sections:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": section["header"]},
            {"role": "assistant", "content": section["content"]},
        ]
        dataset.append({"messages": messages})
    return dataset


def chunks_to_messages(chunks: list[str], system_prompt: str) -> list[dict]:
    """Convert raw text chunks into chat message format (completion style)."""
    dataset = []
    for chunk in chunks:
        # Use the first sentence as the prompt, rest as completion
        sentences = re.split(r'(?<=[.!?])\s+', chunk, maxsplit=1)
        if len(sentences) == 2:
            prompt = sentences[0]
            completion = sentences[1]
        else:
            # Use a generic prompt for continuation
            prompt = "Continue the following text:"
            completion = chunk

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        dataset.append({"messages": messages})
    return dataset


def prepare_dataset(
    input_path: str,
    output_path: str,
    chunk_size: int = 512,
    overlap: int = 64,
    system_prompt: str = "You are a helpful assistant.",
    split_ratio: float = 0.9,
):
    """
    Main data preparation pipeline.

    Args:
        input_path: Path to raw text file
        output_path: Path to output JSONL file
        chunk_size: Words per chunk
        overlap: Word overlap between chunks
        system_prompt: System message for chat format
        split_ratio: Train/eval split ratio
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_file.read_text(encoding="utf-8")
    print(f"Loaded {len(text):,} characters from {input_path}")

    # Try structured extraction first
    sections = detect_sections(text)

    if sections:
        print(f"Detected {len(sections)} sections (headings/Q&A)")
        dataset = sections_to_messages(sections, system_prompt)
    else:
        print("No clear structure found, using chunking strategy")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"Created {len(chunks)} chunks")
        dataset = chunks_to_messages(chunks, system_prompt)

    # Split into train/eval
    split_idx = int(len(dataset) * split_ratio)
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]

    # Write train set
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    train_path = output.parent / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_data)} training examples to {train_path}")

    # Write eval set
    eval_path = output.parent / "eval.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(eval_data)} eval examples to {eval_path}")

    # Print sample
    if dataset:
        print("\n--- Sample ---")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    return train_path, eval_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare raw text for SFT training")
    parser.add_argument("--input", required=True, help="Path to raw text file")
    parser.add_argument("--output", default="./data/train.jsonl", help="Output path")
    parser.add_argument("--chunk-size", type=int, default=512, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Word overlap")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.",
                        help="System prompt for chat format")
    parser.add_argument("--split-ratio", type=float, default=0.9,
                        help="Train/eval split ratio")
    args = parser.parse_args()

    prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        system_prompt=args.system_prompt,
        split_ratio=args.split_ratio,
    )
