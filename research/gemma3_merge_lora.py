#!/usr/bin/env python3
"""Merge Gemma3 LoRA weights and push to HuggingFace Hub.

Usage:
    python research/gemma3_merge_lora.py \
        gs://carles-git-good/tunix-sft/stm-dolci-250K-3/6000 \
        carlesoctav/gemma-3-1b-it-dolci-250k-4b \
        --base-model-path /path/to/google/gemma-3-1b-it
"""

import argparse
import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import HfApi, snapshot_download
import numpy as np
from orbax import checkpoint as ocp
import qwix

from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params as gemma3_params


def load_gemma3_with_lora_from_orbax(
    ckpt_path: str,
    base_model_path: str,
    lora_rank: int = 64,
    lora_alpha: float = 64.0,
    lora_module_path: str = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
) -> nnx.Module:
    """Load a Gemma3 model with LoRA weights from Orbax checkpoint.

    Args:
        ckpt_path: Path to the Orbax checkpoint directory (can be GCS path).
        base_model_path: Path to the base model (HuggingFace format with safetensors).
        lora_rank: LoRA rank used during training.
        lora_alpha: LoRA alpha used during training.
        lora_module_path: Regex pattern for LoRA modules.

    Returns:
        The model with LoRA weights loaded.
    """
    from tunix.cli.utils import model as cli_model_lib

    # Create mesh
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices >= 4:
        mesh_shape = (4, 1)
    elif num_devices >= 2:
        mesh_shape = (2, 1)
    else:
        mesh_shape = (1, 1)

    mesh = jax.sharding.Mesh(
        np.array(devices[: mesh_shape[0] * mesh_shape[1]]).reshape(mesh_shape),
        axis_names=("fsdp", "tp"),
    )

    # Load base model from safetensors
    print(f"Loading base model from {base_model_path}...")
    model, _ = cli_model_lib.create_model_from_safe_tensors(
        model_name="gemma3-1b-it",
        local_path=base_model_path,
        mesh=mesh,
    )

    # Apply LoRA to get the structure
    print("Applying LoRA structure...")
    lora_config = {
        "module_path": lora_module_path,
        "rank": lora_rank,
        "alpha": lora_alpha,
    }
    lora_model = cli_model_lib.apply_lora_to_model(model, mesh, lora_config)

    # Prepare abstract state for restore
    abs_state = nnx.state(lora_model)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )

    # Restore LoRA weights from checkpoint
    print(f"Restoring LoRA weights from {ckpt_path}...")
    checkpointer = ocp.StandardCheckpointer()
    state_path = os.path.join(ckpt_path, "state")
    restored_params = checkpointer.restore(state_path, target=abs_state)

    # Merge graph def with restored params
    graph_def, _ = nnx.split(lora_model)
    model = nnx.merge(graph_def, restored_params)

    return model, mesh


def push_to_hub(local_dir: str, hub_name: str, private: bool = False):
    """Push merged model to Hugging Face Hub."""
    api = HfApi()
    api.create_repo(repo_id=hub_name, exist_ok=True, private=private)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=hub_name,
        repo_type="model",
    )
    print(f"Model successfully pushed to: https://huggingface.co/{hub_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Gemma3 LoRA weights and push to Hub"
    )
    parser.add_argument(
        "ckpt",
        type=str,
        help="Path to the checkpoint directory (Orbax format, can be GCS path)",
    )
    parser.add_argument(
        "hub_name",
        type=str,
        help="Hugging Face Hub repository name (e.g., 'my-org/my-model')",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model ID for base model (default: google/gemma-3-1b-it)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Local path to the base model (if already downloaded)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="LoRA rank used during training (default: 64)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=64.0,
        help="LoRA alpha used during training (default: 64.0)",
    )
    parser.add_argument(
        "--lora-module-path",
        type=str,
        default=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
        help="Regex pattern for LoRA modules",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged model (default: temp directory)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository on the Hub",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing to Hub (only merge and save locally)",
    )

    args = parser.parse_args()

    # Download base model if not provided
    if args.base_model_path:
        base_model_path = args.base_model_path
    else:
        print(f"Downloading base model {args.base_model_id}...")
        base_model_path = snapshot_download(args.base_model_id)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="gemma3_merged_")

    print(f"Checkpoint path: {args.ckpt}")
    print(f"Base model path: {base_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"LoRA rank: {args.rank}, alpha: {args.alpha}")

    # Load model with LoRA
    print("\nLoading Gemma3 model with LoRA weights...")
    model, mesh = load_gemma3_with_lora_from_orbax(
        ckpt_path=args.ckpt,
        base_model_path=base_model_path,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_module_path=args.lora_module_path,
    )

    # Merge and save
    print("\nMerging LoRA weights and saving as safetensors...")
    gemma3_params.save_lora_merged_model_as_safetensors(
        local_model_path=base_model_path,
        output_dir=output_dir,
        lora_model=model,
        rank=args.rank,
        alpha=args.alpha,
    )

    print(f"\nModel saved to: {output_dir}")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f:<40} {size:>10.2f} MB")

    if not args.skip_push:
        print(f"\nPushing to Hub: {args.hub_name}")
        push_to_hub(output_dir, args.hub_name, private=args.private)


if __name__ == "__main__":
    main()
