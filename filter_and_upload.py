import os
import sys

# Set HF_HOME before importing datasets
os.environ["HF_HOME"] = "/mnt/ssd/.cache"

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"]
    )
    from datasets import load_dataset

dataset_name = "carlesoctav/filtered-2048-4b-generated-Dolci-Instruct-SFT-No-Tools"
new_dataset_name = "carlesoctav/IT-WC-2048-4b-generated-Dolci-Instruct-SFT-No-Tools"

print(f"Loading dataset: {dataset_name}")
# Load full dataset (not streaming)
ds = load_dataset(dataset_name, split="train")

print(f"Original row count: {len(ds)}")


def filter_criteria(example):
    source = example.get("dataset_source")
    if not source:
        return False
    source_lower = source.lower()
    # User asked for 'instrcut' (assuming instruct) and 'wildchat'
    # "only includ instrcut and wildchat"
    return "instruct" in source_lower or "wildchat" in source_lower


print(
    "Filtering dataset (keeping rows with 'instruct' or 'wildchat' in dataset_source)..."
)
filtered_ds = ds.filter(filter_criteria)

print(f"Filtered row count: {len(filtered_ds)}")

if len(filtered_ds) > 0:
    print("First 5 sources in filtered dataset:")
    for src in filtered_ds[:5]["dataset_source"]:
        print(f" - {src}")
else:
    print("WARNING: Filter resulted in 0 rows. Please check the filter criteria.")
    sys.exit(1)

print(f"Attempting to push to hub: {new_dataset_name}")
try:
    filtered_ds.push_to_hub(new_dataset_name)
    print("Upload successful!")
except Exception as e:
    print(f"\nUpload failed. You might need to login.")
    print(f"Error: {e}")
    # Save locally as backup
    local_path = "filtered_dataset_backup"
    print(f"Saving locally to {local_path} instead...")
    filtered_ds.save_to_disk(local_path)
    print("Saved locally.")
