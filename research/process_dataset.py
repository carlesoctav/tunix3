"""
Script to shuffle, downsample, and process the Hugging Face dataset.
Uses streaming to avoid disk space issues during loading.
"""
import os

# Set cache directory
os.environ["HF_HOME"] = "/mnt/ssd/.cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/ssd/.cache/datasets"

from datasets import load_dataset
import pandas as pd

# Load the dataset with streaming to avoid filling up disk
print("Loading dataset with streaming...")
dataset = load_dataset(
    "carlesoctav/filtered-2048-4b-generated-Dolci-Instruct-SFT-No-Tools",
    streaming=True
)

# Get the train split
ds_stream = dataset["train"]

# Shuffle with a buffer and take only 500k samples
print("Shuffling and sampling 500k examples...")
target_size = 500_000
# Corrected column name: dataset_source instead of data_source
columns_to_keep = ["original_id", "prompt", "dataset_source"]

# Collect samples with only the columns we need
samples = []
for i, example in enumerate(ds_stream.shuffle(seed=42, buffer_size=10000)):
    # Keep only required columns
    sample = {col: example[col] for col in columns_to_keep}
    samples.append(sample)
    
    if (i + 1) % 50000 == 0:
        print(f"Collected {i + 1} samples...")
    
    if len(samples) >= target_size:
        break

print(f"Collected {len(samples)} samples")

# Convert to pandas DataFrame
print("\nConverting to DataFrame...")
df = pd.DataFrame(samples)

# Check dataset_source statistics
print("\n" + "="*60)
print("DATASET SOURCE STATISTICS")
print("="*60)

data_source_counts = df["dataset_source"].value_counts()
print(f"\nDataset source distribution:\n{data_source_counts}")
print(f"\nTotal unique dataset sources: {len(data_source_counts)}")
print(f"\nPercentage breakdown:")
print((data_source_counts / len(df) * 100).round(2))

# Convert back to HF dataset and save
print("\n" + "="*60)
print("SAVING DATASET")
print("="*60)

from datasets import Dataset
ds_final = Dataset.from_pandas(df)

# Save locally
output_path = "/mnt/ssd/tunix3/research/processed_dataset_500k"
ds_final.save_to_disk(output_path)
print(f"Dataset saved to: {output_path}")

# Also show a sample
print("\n" + "="*60)
print("SAMPLE DATA (first 3 rows)")
print("="*60)
print(df.head(3).to_string())
