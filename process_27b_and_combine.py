import os
import sys
from datasets import load_dataset, concatenate_datasets

# Set HF_HOME
os.environ["HF_HOME"] = "/mnt/ssd/.cache"


def filter_criteria(example):
    source = example.get("dataset_source")
    if not source:
        return False
    source_lower = source.lower()
    return "instruct" in source_lower or "wildchat" in source_lower


# --- Step 1: Process 27b version ---
source_27b = "carlesoctav/filtered-2048-27b-generated-Dolci-Instruct-SFT-No-Tools"
target_27b = "carlesoctav/IT-WC-2048-27b-generated-Dolci-Instruct-SFT-No-Tools"

print(f"Loading 27b dataset: {source_27b}")
ds_27b = load_dataset(source_27b, split="train")
print(f"Original 27b row count: {len(ds_27b)}")

print("Filtering 27b dataset...")
filtered_27b = ds_27b.filter(filter_criteria)
print(f"Filtered 27b row count: {len(filtered_27b)}")

print(f"Pushing filtered 27b to hub: {target_27b}")
filtered_27b.push_to_hub(target_27b)
print("27b Upload successful!")

# --- Step 2: Combine with 4b version ---
source_4b_filtered = "carlesoctav/IT-WC-2048-4b-generated-Dolci-Instruct-SFT-No-Tools"
target_combined = "carlesoctav/IT-WDIT-CWIT-WC"

print(f"Loading filtered 4b dataset: {source_4b_filtered}")
# We load the one we just created (or download it if not in memory, but it's cache should be there)
ds_4b = load_dataset(source_4b_filtered, split="train")
print(f"Filtered 4b row count: {len(ds_4b)}")

print("Concatenating datasets...")
combined_ds = concatenate_datasets([ds_4b, filtered_27b])
print(f"Combined row count: {len(combined_ds)}")

print(f"Pushing combined dataset to hub: {target_combined}")
combined_ds.push_to_hub(target_combined)
print("Combined Upload successful!")
