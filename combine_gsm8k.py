import os
from datasets import load_dataset, concatenate_datasets, Value

# Set HF_HOME
os.environ["HF_HOME"] = "/mnt/ssd/.cache"

target_repo = "carlesoctav/IT-WC-GSM8k-combine"
main_dataset_name = "carlesoctav/IT-WDIT-CWIT-WC"
gsm8k_dataset_name = "carlesoctav/gemma-3-27b-gsm8k-dataset"

print(f"Loading main dataset: {main_dataset_name}")
ds_main = load_dataset(main_dataset_name, split="train")
print(f"Main dataset rows: {len(ds_main)}")

print(f"Loading GSM8K dataset: {gsm8k_dataset_name}")
ds_gsm8k = load_dataset(gsm8k_dataset_name, split="train")
print(f"GSM8K dataset rows: {len(ds_gsm8k)}")


def transform_gsm8k(example):
    # Add <reasoning> tag to generated
    gen = example.get("generated", "")
    if gen is None:
        gen = ""
    # Just in case, convert to string
    gen = str(gen)

    # Prepend tag. User said "at every start".
    # I'll check if it already starts with it to avoid double adding, though usually safe to just add if instructed.
    # User instruction: "but before that you need to add <reasoning> at every start."
    example["generated"] = "<reasoning>\n" + gen

    # Add dataset_source
    example["dataset_source"] = "gsm8k"

    # Map idx to doc_id/original_id
    idx = example.get("idx")
    example["doc_id"] = str(idx) if idx is not None else None
    example["original_id"] = str(idx) if idx is not None else None

    # Add missing columns
    example["stop_reason"] = None

    return example


print("Transforming GSM8K dataset...")
ds_gsm8k_transformed = ds_gsm8k.map(transform_gsm8k)

# Select and Cast columns to match main dataset
# Main features: doc_id, original_id, prompt, generated, stop_reason, dataset_source
columns_to_keep = [
    "doc_id",
    "original_id",
    "prompt",
    "generated",
    "stop_reason",
    "dataset_source",
]

# Ensure gsm8k has all these and only these
ds_gsm8k_aligned = ds_gsm8k_transformed.select_columns(columns_to_keep)

# Cast features to match exactly (especially doc_id/original_id which were int in gsm8k originally maybe)
features = ds_main.features.copy()
# We need to ensure the schema matches.
# ds_main features: {'doc_id': Value('string'), ...}
# We can force cast.
ds_gsm8k_aligned = ds_gsm8k_aligned.cast(features)

print("Concatenating...")
combined_ds = concatenate_datasets([ds_main, ds_gsm8k_aligned])
print(f"Total rows: {len(combined_ds)}")

print(f"Pushing to hub: {target_repo}")
combined_ds.push_to_hub(target_repo)
print("Upload successful!")
