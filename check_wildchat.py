from datasets import load_dataset
import os

# Set HF_HOME
os.environ["HF_HOME"] = "/mnt/ssd/.cache"

dataset_name = "carlesoctav/IT-WDIT-CWIT-WC"
print(f"Checking dataset: {dataset_name}")

try:
    ds = load_dataset(dataset_name, split="train", streaming=True)

    wildchat_count = 0
    total_checked = 0
    unique_sources = set()

    print("Scanning first 50,000 rows for 'wildchat'...")

    for item in ds:
        source = item.get("dataset_source", "")
        if source:
            unique_sources.add(source)
            if "wildchat" in source.lower():
                wildchat_count += 1

        total_checked += 1
        if total_checked >= 50000:
            break

    print(f"Checked {total_checked} rows.")
    print(f"Found {wildchat_count} rows containing 'wildchat' in dataset_source.")
    print("\nUnique sources found in sample:")
    for s in unique_sources:
        print(f" - {s}")

except Exception as e:
    print(f"Error: {e}")
