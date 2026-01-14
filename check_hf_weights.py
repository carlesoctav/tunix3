from huggingface_hub import hf_hub_download
from safetensors import safe_open
import json
import os


def check_embedding_weight():
    model_id = "google/gemma-3-1b-it"
    print(f"Inspecting weights for {model_id}...")

    # Check for index file first (sharded model)
    try:
        index_path = hf_hub_download(
            repo_id=model_id, filename="model.safetensors.index.json"
        )
        print("Found sharded model index.")
        with open(index_path, "r") as f:
            index = json.load(f)

        # Find which file contains the embedding weights
        weight_name = "model.embed_tokens.weight"
        if weight_name in index["weight_map"]:
            shard_file = index["weight_map"][weight_name]
            print(f"Embedding weights found in shard: {shard_file}")

            # Download that specific shard (header only if possible, but safetensors needs file)
            # We'll download the file, it's 1B model, so shards might be small enough or just one file.
            # Actually gemma-1b might not be sharded.
            file_path = hf_hub_download(repo_id=model_id, filename=shard_file)
        else:
            print(f"Could not find {weight_name} in index.")
            return

    except Exception:
        # Not sharded or error, try downloading single file
        print(
            "Model likely not sharded (or index fetch failed). Checking for single model.safetensors..."
        )
        try:
            file_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        except Exception as e:
            print(f"Failed to find model.safetensors: {e}")
            return

    # Inspect the file
    print(f"Inspecting file: {file_path}")
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor("model.embed_tokens.weight")
            print(f"\nSUCCESS: Found 'model.embed_tokens.weight'")
            print(f"Shape: {tensor.shape}")
            print(f"Vocab Size (Row count): {tensor.shape[0]}")
            print(f"Embed Dim (Col count):  {tensor.shape[1]}")
    except Exception as e:
        print(f"Error reading safetensors: {e}")
        # Try finding key if name is different
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            print("Available keys prefix check:")
            for k in keys:
                if "embed" in k or "token" in k:
                    print(f"  {k}: {f.get_slice(k).get_shape()}")


if __name__ == "__main__":
    check_embedding_weight()
