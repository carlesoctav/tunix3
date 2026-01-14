import os
import sentencepiece as spm
from transformers import AutoTokenizer


def compare_tokenizers():
    hf_model_id = "google/gemma-3-1b-it"
    sp_model_path = "./gemma3.model"

    print(f"Loading HuggingFace AutoTokenizer for {hf_model_id}...")
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    except Exception as e:
        print(f"Failed to load AutoTokenizer: {e}")
        return

    print(f"Loading SentencePiece Tokenizer from {sp_model_path}...")
    try:
        sp_tokenizer = spm.SentencePieceProcessor()
        sp_tokenizer.Load(sp_model_path)
    except Exception as e:
        print(f"Failed to load SentencePiece tokenizer: {e}")
        return

    # 1. Answer the len() vs vocab_size question
    print("\n" + "=" * 40)
    print("SIZE COMPARISON")
    print("=" * 40)
    print(f"HF tokenizer.vocab_size: {hf_tokenizer.vocab_size}")
    print(f"HF len(tokenizer):       {len(hf_tokenizer)}")
    print(f"SP GetPieceSize():       {sp_tokenizer.GetPieceSize()}")

    if len(hf_tokenizer) != hf_tokenizer.vocab_size:
        print("\nExplanation: len(tokenizer) != vocab_size because of ADDED tokens.")
        print(f"Number of added tokens: {len(hf_tokenizer) - hf_tokenizer.vocab_size}")
        print("Added tokens (ID -> Token):")
        added_tokens = hf_tokenizer.get_added_vocab()
        # Sort by ID
        sorted_added = sorted(added_tokens.items(), key=lambda x: x[1])
        for token, id in sorted_added:
            print(f"  ID {id}: {repr(token)}")
    else:
        print("\nlen(tokenizer) == vocab_size. No added tokens found.")

    # 2. Run the entire vocab comparison
    print("\n" + "=" * 40)
    print("FULL VOCABULARY COMPARISON")
    print("=" * 40)

    # We compare only the shared range (the base vocab)
    limit = sp_tokenizer.GetPieceSize()
    print(f"Comparing base vocabulary (0 to {limit - 1})...")

    mismatches = []

    # Check in chunks to report progress
    chunk_size = 50000
    for i in range(limit):
        if i % chunk_size == 0:
            print(f"Checking {i} / {limit}...")

        # HF: convert_ids_to_tokens
        hf_token = hf_tokenizer.convert_ids_to_tokens(i)

        # SP: IdToPiece
        sp_token = sp_tokenizer.IdToPiece(i)

        if hf_token != sp_token:
            # Special case: SentencePiece might return <unk> for unused tokens,
            # while HF might have a different placeholder or byte fallback representation.
            mismatches.append((i, hf_token, sp_token))
            if len(mismatches) < 10:
                print(f"Mismatch at ID {i}: HF={repr(hf_token)} SP={repr(sp_token)}")

    print(f"Checking {limit} / {limit}... Done.")

    if not mismatches:
        print("\nSUCCESS: All base tokens match exactly!")
    else:
        print(f"\nFAILURE: Found {len(mismatches)} mismatches in base vocab.")


if __name__ == "__main__":
    compare_tokenizers()
