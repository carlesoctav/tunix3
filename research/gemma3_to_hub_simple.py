#!/usr/bin/env python3
"""
Simple Gemma3 LoRA merge + upload to HuggingFace.

Usage:
    python research/gemma3_to_hub_simple.py \
        --checkpoint-dir "gs://carles-git-good/tunix-final-sft/27b-generated-Dolci-Instruct-SFT-No-Tools-rank-256-lr-5e7" \
        --hf-repo-id "carlesonai/Dolci-Instruct-1b" \
        --lora-rank 256 \
        --step 15000
"""

import argparse
import dataclasses
import os
import shutil

import jax
from flax import nnx
from huggingface_hub import HfApi, create_repo, snapshot_download

from tunix.cli.utils import model as cli_model_lib
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params_safetensors
from tunix.sft import checkpoint_manager


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument(
        "--step", type=int, default=None, help="Checkpoint step (default: latest)"
    )
    p.add_argument("--lora-rank", type=int, default=256)
    p.add_argument(
        "--lora-alpha", type=float, default=None, help="Default: same as rank"
    )
    p.add_argument("--hf-repo-id", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./exported_model")
    p.add_argument("--mesh-shape", type=str, default="4,1")
    p.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Path to chat template jinja file",
    )
    p.add_argument("--no-upload", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    lora_alpha = args.lora_alpha or float(args.lora_rank)
    mesh_shape = tuple(int(x) for x in args.mesh_shape.split(","))
    mesh = jax.make_mesh(mesh_shape, ("fsdp", "tp"))

    # 1. Download model from HuggingFace
    print("Downloading google/gemma-3-1b-it...")
    local_model_path = snapshot_download(
        repo_id="google/gemma-3-1b-it",
        ignore_patterns=["*.pth"],
    )
    print(f"Model cached at: {local_model_path}")

    # 2. Create model from safetensors
    print("Loading model...")
    config = gemma3_model.ModelConfig.gemma3_1b_it()
    model = params_safetensors.create_model_from_safe_tensors(
        local_model_path, config, mesh
    )

    # 3. Apply LoRA
    print(f"Applying LoRA (rank={args.lora_rank}, alpha={lora_alpha})...")
    lora_config = {
        "rank": args.lora_rank,
        "alpha": lora_alpha,
        "module_path": ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
    }
    model = cli_model_lib.apply_lora_to_model(model, mesh, lora_config)

    # 4. Restore LoRA checkpoint
    print(f"Restoring checkpoint from {args.checkpoint_dir}...")
    ckpt_mgr = checkpoint_manager.CheckpointManager(root_directory=args.checkpoint_dir)
    step = args.step or ckpt_mgr.latest_step()
    print(f"Using step: {step}")
    ckpt_mgr.maybe_restore(model, step=step, restore_only_lora_params=True)

    # 5. Prepare output dir
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 6. Merge LoRA and save
    print(f"Merging LoRA and saving to {output_dir}...")
    params_safetensors.save_lora_merged_model_as_safetensors(
        local_model_path=local_model_path,
        output_dir=output_dir,
        lora_model=model,
        rank=args.lora_rank,
        alpha=lora_alpha,
    )

    # 7. Copy chat template if provided
    if args.chat_template:
        print(f"Copying chat template: {args.chat_template}")
        template_content = open(args.chat_template).read()
        # Update tokenizer_config.json with chat template
        tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            import json

            with open(tokenizer_config_path) as f:
                tokenizer_config = json.load(f)
            tokenizer_config["chat_template"] = template_content
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)

    # 8. Upload
    if not args.no_upload:
        print(f"Uploading to {args.hf_repo_id}...")
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        create_repo(args.hf_repo_id, token=token, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=output_dir, repo_id=args.hf_repo_id, repo_type="model"
        )
        print("Done!")
    else:
        print(f"Skipping upload. Model saved at {output_dir}")


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
