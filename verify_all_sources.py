from datasets import load_dataset
import os
import collections

# Set HF_HOME
os.environ["HF_HOME"] = "/mnt/ssd/.cache"

# Check the final combined dataset directly
dataset_name = "carlesoctav/IT-WC-GSM8k-combine"
print(f"Checking dataset: {dataset_name}")

try:
    # Use streaming to scan efficiently
    ds = load_dataset(dataset_name, split="train", streaming=True)

    source_counts = collections.Counter()
    total_checked = 0

    # We'll check more rows this time, or until we find all 3
    print("Scanning dataset sources (checking up to 200,000 rows)...")

    found_wildchat = False
    found_instruct = False
    found_gsm8k = False

    for item in ds:
        source = item.get("dataset_source", "")
        # Normalize for easier checking
        if source:
            lower_source = source.lower()
            if "wildchat" in lower_source:
                source_counts["wildchat"] += 1
                found_wildchat = True
            elif "instruct" in lower_source:
                source_counts["instruct"] += 1
                found_instruct = True
            elif "gsm8k" in lower_source:
                source_counts["gsm8k"] += 1
                found_gsm8k = True
            else:
                source_counts[source] += 1

        total_checked += 1
        if total_checked % 50000 == 0:
            print(f"Checked {total_checked} rows. Found: {dict(source_counts)}")

        if total_checked >= 200000:
            break

    print(f"\nFinal check after {total_checked} rows:")
    print(f"Counts: {dict(source_counts)}")

    # Verify expected
    print("\nVerification:")
    print(f"- Wildchat found: {found_wildchat}")
    print(f"- Instruct found: {found_instruct}")
    print(f"- GSM8K found: {found_gsm8k}")

except Exception as e:
    print(f"Error: {e}")
