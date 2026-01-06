"""Calculate token statistics for a dataset by iterating the full dataset.

This script helps determine the LR divisor for sum loss training.
Uses .shuffle().batch().map() pipeline for fast processing.

Usage:
    uv run python research/check_batch_stats.py
    uv run python research/check_batch_stats.py --batch_size 32 --max_seq_length 4096
"""

import argparse
from functools import partial
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm
from termcolor import colored


def apply_chat_template_map(
    batch: dict,
    tokenizer: PreTrainedTokenizerBase,
    prompt_column: str,
    answer_column: str,
    max_seq_length: int,
) -> dict:
    """Apply chat template to a batch of examples.

    Args:
        batch: Dict with prompt and answer columns (lists).
        tokenizer: Tokenizer with chat template.
        prompt_column: Column name for prompts.
        answer_column: Column name for answers.
        max_seq_length: Maximum sequence length.

    Returns:
        Dict with input_tokens and input_mask arrays.
    """
    batch_size = len(batch[prompt_column])
    all_input_tokens = []
    all_input_masks = []

    for i in range(batch_size):
        prompt = batch[prompt_column][i]
        answer = batch[answer_column][i]
        messages = [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(answer)},
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_tensors="np",
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_dict=True,
        )
        all_input_tokens.append(encoded["input_ids"][0])
        all_input_masks.append(np.array(encoded["assistant_masks"][0], dtype=np.int32))

    return {
        "input_tokens": np.stack(all_input_tokens),
        "input_mask": np.stack(all_input_masks),
    }


def count_batch_tokens(batch: dict) -> dict:
    """Count tokens in a batch based on assistant_mask.

    Args:
        batch: Dict with input_tokens and input_mask.

    Returns:
        Dict with batch_tokens count added.
    """
    batch["batch_tokens"] = int(np.sum(batch["input_mask"]))
    return batch


def format_colorized(
    tokens: list[int],
    weights: list[float],
    tokenizer: PreTrainedTokenizerBase,
    draw_newline_arrow: bool = True,
) -> str:
    """Colour-code text according to per-token weights.

    * Green text → weight > 0 (completion/assistant tokens, loss computed)
    * Yellow text → weight = 0 (prompt/padding tokens, no loss)
    """
    if len(tokens) != len(weights):
        raise ValueError("`tokens` and `weights` must be the same length.")

    chunks, current_ids, current_color = [], [], None

    def flush_current_run():
        if not current_ids:
            return
        decoded = tokenizer.decode(current_ids)
        if draw_newline_arrow:
            decoded = decoded.replace("\n", "↵\n")
        chunks.append(colored(decoded, current_color))

    for tok_id, w in zip(tokens, weights, strict=True):
        if w > 0:
            color = "green"
        else:
            color = "yellow"

        if color != current_color and current_ids:
            flush_current_run()
            current_ids = []

        current_ids.append(tok_id)
        current_color = color

    flush_current_run()
    return "".join(chunks)


def visualize_batch(
    batch: dict,
    tokenizer: PreTrainedTokenizerBase,
    num_examples: int = 2,
) -> None:
    """Visualize tokenization with colorized loss mask.

    Green = assistant tokens (loss computed)
    Yellow = prompt/padding tokens (no loss)
    """
    print("\n" + "=" * 80)
    print("TOKENIZATION VISUALIZATION (Green=loss, Yellow=no loss/padding)")
    print("=" * 80)

    batch_tokens = batch["input_tokens"]
    batch_masks = batch["input_mask"]

    for i in range(min(num_examples, len(batch_tokens))):
        tokens = batch_tokens[i].tolist()
        weights = batch_masks[i].tolist()

        # Count non-padding tokens and assistant tokens
        non_pad = sum(1 for t in tokens if t != 0)
        assistant_tokens = sum(weights)

        print(
            f"\n--- Example {i + 1} (total_len={len(tokens)}, non_pad={non_pad}, assistant_tokens={assistant_tokens}) ---"
        )
        print(format_colorized(tokens, weights, tokenizer))

    print("\n" + "=" * 80 + "\n")


def check_batch_stats(
    tokenizer_path: str = "google/gemma-2-2b-it",
    chat_template_path: str = "gemma_think_new.jinja",
    dataset_path: str = "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools",
    prompt_column: str = "prompt",
    answer_column: str = "generated",
    batch_size: int = 32,
    max_seq_length: int = 2048,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    shuffle_buffer_size: int = 10000,
) -> dict[str, float]:
    """Calculate token statistics by iterating the full dataset.

    Uses .shuffle().batch().map() pipeline for fast processing.

    Args:
        tokenizer_path: HuggingFace tokenizer path.
        chat_template_path: Path to chat template file.
        dataset_path: HuggingFace dataset path.
        prompt_column: Column name for prompts.
        answer_column: Column name for answers.
        batch_size: Batch size for stats calculation.
        max_seq_length: Maximum sequence length.
        shuffle: Whether to shuffle the dataset.
        shuffle_seed: Seed for shuffling dataset.
        shuffle_buffer_size: Buffer size for shuffle.

    Returns:
        Dict with comprehensive token statistics.
    """
    print(f"\n{'=' * 70}")
    print("Token Statistics Calculator for Sum Loss Training")
    print(f"{'=' * 70}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"{'=' * 70}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if chat_template_path:
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()

    # Load dataset as iterable for streaming
    print("Loading dataset...")
    ds = load_dataset(dataset_path, split="train", streaming=True)

    # Create map functions with partial
    tokenize_fn = partial(
        apply_chat_template_map,
        tokenizer=tokenizer,
        prompt_column=prompt_column,
        answer_column=answer_column,
        max_seq_length=max_seq_length,
    )

    # Pipeline: shuffle -> batch -> map(tokenize) -> map(count_tokens)
    if shuffle:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)

    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(tokenize_fn)

    # Visualize first batch using next(iter(ds))
    print("\nVisualizing first batch...")
    first_batch = next(iter(ds))
    visualize_batch(first_batch, tokenizer, num_examples=2)

    # Re-create pipeline for full iteration (since we consumed first batch)
    ds = load_dataset(dataset_path, split="train", streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(tokenize_fn)
    ds = ds.map(count_batch_tokens)

    # Process all batches and collect token counts
    print("\nProcessing dataset...")
    token_counts_per_batch = []

    for batch in tqdm(ds, desc="Processing batches"):
        token_counts_per_batch.append(batch["batch_tokens"])

    # Calculate statistics
    token_counts_per_batch = np.array(token_counts_per_batch)
    total_tokens = int(np.sum(token_counts_per_batch))

    stats = {
        # Per-batch stats (for LR adjustment)
        "num_batches": len(token_counts_per_batch),
        "mean_tokens_per_batch": float(np.mean(token_counts_per_batch)),
        "median_tokens_per_batch": float(np.median(token_counts_per_batch)),
        "std_tokens_per_batch": float(np.std(token_counts_per_batch)),
        "min_tokens_per_batch": float(np.min(token_counts_per_batch)),
        "max_tokens_per_batch": float(np.max(token_counts_per_batch)),
        "p10_tokens_per_batch": float(np.percentile(token_counts_per_batch, 10)),
        "p90_tokens_per_batch": float(np.percentile(token_counts_per_batch, 90)),
        # Total
        "total_tokens": total_tokens,
        "total_examples": len(token_counts_per_batch) * batch_size,
        "mean_tokens_per_example": float(np.mean(token_counts_per_batch)) / batch_size,
    }

    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"\nDataset Summary:")
    print(f"  Total batches: {stats['num_batches']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Approx examples: {stats['total_examples']:,}")

    print(f"\nPer-Batch Token Stats (batch_size={batch_size}):")
    print(f"  Mean:   {stats['mean_tokens_per_batch']:.1f}")
    print(f"  Median: {stats['median_tokens_per_batch']:.1f}")
    print(f"  Std:    {stats['std_tokens_per_batch']:.1f}")
    print(f"  Min:    {stats['min_tokens_per_batch']:.0f}")
    print(f"  Max:    {stats['max_tokens_per_batch']:.0f}")
    print(f"  P10:    {stats['p10_tokens_per_batch']:.0f}")
    print(f"  P90:    {stats['p90_tokens_per_batch']:.0f}")

    print(f"\nPer-Example Token Stats:")
    print(f"  Mean:   {stats['mean_tokens_per_example']:.1f}")

    print(f"\n{'=' * 70}")
    print("RECOMMENDED LR DIVISOR")
    print(f"{'=' * 70}")
    print(f"  Use mean tokens per batch: {stats['mean_tokens_per_batch']:.1f}")
    print(
        f"  Example: if base LR is 1e-4, use LR = 1e-4 / {stats['mean_tokens_per_batch']:.0f}"
    )
    print(f"           = {1e-4 / stats['mean_tokens_per_batch']:.2e}")
    print(f"{'=' * 70}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Calculate token statistics for a dataset"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="google/gemma-2-2b-it",
        help="HuggingFace tokenizer path",
    )
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default="gemma_think_new.jinja",
        help="Path to chat template file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Column name for prompts",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="generated",
        help="Column name for answers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for stats calculation",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable dataset shuffling",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Seed for shuffling dataset",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10000,
        help="Buffer size for shuffle",
    )

    args = parser.parse_args()

    stats = check_batch_stats(
        tokenizer_path=args.tokenizer_path,
        chat_template_path=args.chat_template_path,
        dataset_path=args.dataset_path,
        prompt_column=args.prompt_column,
        answer_column=args.answer_column,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        shuffle=not args.no_shuffle,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )


if __name__ == "__main__":
    main()
