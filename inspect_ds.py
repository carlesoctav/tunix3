from datasets import load_dataset
import sys

dataset_name = "carlesoctav/filtered-2048-4b-generated-Dolci-Instruct-SFT-No-Tools"
print(f"Loading {dataset_name}...")

try:
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # Get first item to see columns
    first_item = next(iter(ds))
    print("Columns:", first_item.keys())
    print("First item sample:", first_item)

    # Let's try to find the distribution of a 'source' column if it exists
    # If not, we'll check other columns.

    potential_source_cols = [
        c
        for c in first_item.keys()
        if c in ["source", "subset", "dataset", "origin", "category", "task"]
    ]

    if potential_source_cols:
        col = potential_source_cols[0]
        print(f"Checking unique values for column: {col}")

        # Collect some unique values (streaming so we can't get all easily without iterating a lot, but let's check first 1000)
        unique_vals = set()
        count = 0
        for item in ds:
            unique_vals.add(item[col])
            count += 1
            if count > 2000:
                break
        print(f"Found values in first 2000 rows: {unique_vals}")
    else:
        print("No obvious source column found. Please check the sample above.")

except Exception as e:
    print(e)
