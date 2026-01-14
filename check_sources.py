from datasets import load_dataset

dataset_name = "carlesoctav/filtered-2048-4b-generated-Dolci-Instruct-SFT-No-Tools"
ds = load_dataset(dataset_name, split="train", streaming=True)

unique_vals = set()
count = 0
for item in ds:
    val = item["dataset_source"]
    unique_vals.add(val)
    count += 1
    if count > 10000:  # Check more rows to ensure we find them
        break

print("Unique dataset_source values found:")
for v in unique_vals:
    print(f" - {v}")
