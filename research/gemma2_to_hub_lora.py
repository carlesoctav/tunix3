#!/usr/bin/env python3
"""
Export a Gemma-2 model with merged LoRA weights to SafeTensors and optionally
upload to the Hugging Face Hub.

This script uses the proper safetensors_saver infrastructure:
1. Download base model from HuggingFace (safetensors format)
2. Load model and apply LoRA
3. Restore LoRA params from checkpoint
4. Use save_lora_merged_model_as_safetensors to merge and save
5. Upload to HuggingFace

Typical usage:
    python research/gemma2_to_hub_lora.py \
      --checkpoint-dir "gs://your-bucket/tunix/gemma2-repro/actor" \
      --output-dir "./exported_gemma2_lora" \
      --hf-repo-id "username/gemma2-2b-lora"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

from tunix.cli.utils import model as model_lib
from tunix.models.gemma import params_safetensors as gemma2_params
from tunix.sft import checkpoint_manager

# Hugging Face
try:
    from huggingface_hub import HfApi, create_repo, snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HfApi = None
    create_repo = None
    snapshot_download = None
    HAS_HF_HUB = False

# Transformers for validation
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Load Kaggle credentials from ~/.kaggle/kaggle.json
_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if _kaggle_json.exists():
    with open(_kaggle_json) as f:
        _kaggle_creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = _kaggle_creds.get("username", "")
        os.environ["KAGGLE_KEY"] = _kaggle_creds.get("key", "")
        print(f"Loaded Kaggle credentials for user: {_kaggle_creds.get('username', 'unknown')}")


def create_mesh(mesh_shape: tuple[int, int] = (4, 1)):
    """Create a JAX mesh for the model."""
    return jax.make_mesh(
        mesh_shape,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )


def download_hf_model(hf_model_id: str, cache_dir: str) -> str:
    """Download a HuggingFace model to local cache.
    
    Args:
        hf_model_id: HuggingFace model ID (e.g., 'google/gemma-2-2b-it')
        cache_dir: Local directory to cache the model
        
    Returns:
        Path to the downloaded model directory
    """
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub is not installed")
    
    print(f"Downloading HuggingFace model: {hf_model_id}")
    local_dir = os.path.join(cache_dir, hf_model_id.replace("/", "_"))
    
    if os.path.exists(local_dir) and any(f.endswith('.safetensors') for f in os.listdir(local_dir)):
        print(f"Model already cached at: {local_dir}")
        return local_dir
    
    snapshot_download(
        repo_id=hf_model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {local_dir}")
    return local_dir


def load_base_model_and_apply_lora(
    model_config: Dict[str, Any],
    tokenizer_config: Dict[str, Any],
    lora_config: Dict[str, Any],
    mesh: jax.sharding.Mesh,
):
    """Load base model and apply LoRA adapter."""
    print("Loading base model (this may download from Kaggle)...")
    reference_model, tokenizer_path = model_lib.create_model(
        model_config,
        tokenizer_config,
        mesh,
    )

    print("Applying LoRA adapter to model...")
    actor_model = model_lib.apply_lora_to_model(
        reference_model,
        mesh,
        lora_config,
    )

    model_params = model_lib.obtain_model_params(model_config["model_name"])
    return actor_model, model_params, tokenizer_path


def validate_with_transformers(output_dir: str, prompt: str = "What is 2 + 2?") -> bool:
    """Validate the exported model by loading it with transformers and running inference."""
    if not HAS_TRANSFORMERS:
        print("WARNING: transformers not installed, skipping validation")
        return True
    
    print(f"\n--- Validating exported model with transformers ---")
    print(f"Loading model from: {output_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print(f"Tokenizer loaded successfully: {type(tokenizer).__name__}")
        
        # Load without device_map to avoid accelerate dependency
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"Model loaded successfully: {type(model).__name__}")
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model device: {device}")
        
        print(f"\nTest prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"Input tokens: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated text:\n{generated_text}")
        
        if len(outputs[0]) > len(inputs['input_ids'][0]):
            print("\n✓ Validation PASSED: Model generates output successfully")
            return True
        else:
            print("\n✗ Validation FAILED: Model did not generate any new tokens")
            return False
            
    except Exception as e:
        print(f"\n✗ Validation FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_to_hf(output_dir: str, repo_id: str, token: str | None = None):
    """Upload the contents of output_dir to Hugging Face Hub."""
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub is not installed")
    
    api = HfApi(token=token)
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation note: {e}")
    
    print(f"Uploading folder to {repo_id}...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print("Upload complete!")


def parse_args():
    p = argparse.ArgumentParser(
        description="Export Gemma-2 with merged LoRA weights to HuggingFace format"
    )
    
    # Model config
    p.add_argument(
        "--model-name",
        type=str,
        default="gemma2-2b-it",
        help="Model name (e.g., gemma2-2b-it)",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2/flax/gemma2-2b-it",
        help="Model ID for downloading (Kaggle path)",
    )
    p.add_argument(
        "--model-source",
        type=str,
        default="kaggle",
        choices=["kaggle", "huggingface", "gcs"],
        help="Where to download base model from",
    )
    p.add_argument(
        "--model-download-path",
        type=str,
        default="/tmp/models/gemma2-2b",
        help="Local path to download/cache the base model",
    )
    p.add_argument(
        "--intermediate-ckpt-dir",
        type=str,
        default="/tmp/intermediate_ckpt/export",
        help="Intermediate checkpoint directory for NNX conversion",
    )
    p.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (auto-detected if not specified)",
    )
    
    # LoRA config
    p.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank used during training",
    )
    p.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha used during training",
    )
    p.add_argument(
        "--module-path",
        type=str,
        default=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
        help="Regex pattern for LoRA modules",
    )
    
    # Checkpoint
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to Orbax checkpoint directory with LoRA weights (local or gs://)",
    )
    
    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default="./exported_gemma2_lora",
        help="Directory to write exported files",
    )
    p.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID for upload (e.g., 'username/model-name')",
    )
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--hf-base-model",
        type=str,
        default="google/gemma-2-2b-it",
        help="Base HF model ID for downloading safetensors base model",
    )
    p.add_argument(
        "--hf-model-cache",
        type=str,
        default="/tmp/hf_models",
        help="Cache directory for HuggingFace model downloads",
    )
    p.add_argument(
        "--upload",
        action="store_true",
        default=True,
        help="Upload to HuggingFace Hub (default: True if --hf-repo-id is set)",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace Hub",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate the exported model with transformers before uploading (default: True)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation step",
    )
    p.add_argument(
        "--validation-prompt",
        type=str,
        default="What is 2 + 2? Answer:",
        help="Prompt to use for validation generation test",
    )
    p.add_argument(
        "--mesh-shape",
        type=str,
        default="4,1",
        help="Mesh shape as 'fsdp,tp' (e.g., '4,1')",
    )
    
    return p.parse_args()


def main():
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    should_upload = args.upload and not args.no_upload and (args.hf_repo_id is not None)

    print("=" * 60)
    print("Gemma-2 LoRA Export Tool (Using safetensors_saver)")
    print("=" * 60)
    print("Configuration:")
    print(json.dumps(vars(args), indent=2))
    print("=" * 60)

    # Parse mesh shape
    mesh_shape = tuple(int(x) for x in args.mesh_shape.split(","))
    mesh = create_mesh(mesh_shape)
    print(f"Created mesh: {mesh}")

    # Step 1: Download HuggingFace base model (safetensors format)
    print("\n--- Step 1: Download HuggingFace base model ---")
    hf_base_model_path = download_hf_model(args.hf_base_model, args.hf_model_cache)
    
    # Step 2: Build model config dict for loading from Kaggle (for structure)
    model_config = {
        "model_name": args.model_name,
        "model_id": args.model_id,
        "model_source": args.model_source,
        "model_download_path": args.model_download_path,
        "intermediate_ckpt_dir": args.intermediate_ckpt_dir,
        "model_display": False,
    }
    
    # Tokenizer config
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = os.path.join(args.model_download_path, "tokenizer.model")
    
    tokenizer_config = {
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": "sentencepiece",
        "add_bos": False,
        "add_eos": False,
    }
    
    # LoRA config
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "module_path": args.module_path,
    }

    # Step 3: Load base model and apply LoRA
    print("\n--- Step 2: Load model and apply LoRA ---")
    model, model_params, _ = load_base_model_and_apply_lora(
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        lora_config=lora_config,
        mesh=mesh,
    )

    # Step 4: Restore LoRA params from checkpoint
    print(f"\n--- Step 3: Restore LoRA params from checkpoint ---")
    print(f"Checkpoint: {args.checkpoint_dir}")
    ckpt_mgr = checkpoint_manager.CheckpointManager(root_directory=args.checkpoint_dir)
    step = ckpt_mgr.latest_step()
    if step is None:
        raise RuntimeError(f"No checkpoint found in {args.checkpoint_dir}")
    print(f"Found checkpoint at step {step}")
    
    ckpt_mgr.maybe_restore(model, step=step, restore_only_lora_params=True)
    print("LoRA params restored successfully.")

    # Step 5: Prepare output directory
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.overwrite:
            print(f"\nRemoving existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"\nOutput directory exists, will overwrite: {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 6: Use save_lora_merged_model_as_safetensors
    print(f"\n--- Step 4: Merge LoRA and save as safetensors ---")
    print(f"Base model path: {hf_base_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    
    gemma2_params.save_lora_merged_model_as_safetensors(
        local_model_path=hf_base_model_path,
        output_dir=str(output_dir),
        lora_model=model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print("Model saved successfully!")

    # Step 7: Validate with transformers
    should_validate = args.validate and not args.no_validate
    if should_validate:
        validation_passed = validate_with_transformers(
            str(output_dir), 
            prompt=args.validation_prompt
        )
        if not validation_passed:
            print("\n" + "=" * 60)
            print("WARNING: Validation failed! Model may have incorrect weight mapping.")
            print("Aborting upload. Please check the exported model.")
            print("=" * 60)
            return
    
    # Step 8: Upload to HuggingFace
    if should_upload:
        print(f"\n--- Step 5: Upload to HuggingFace ---")
        print(f"Repo: {args.hf_repo_id}")
        upload_to_hf(str(output_dir), args.hf_repo_id, hf_token)
    else:
        print("\nSkipping upload (use --hf-repo-id to enable)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()