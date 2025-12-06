#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Combine LoRA weights with base model and push to Hugging Face Hub.

Usage:
    python research/combine_lora_push_to_hub.py <ckpt_path> <hub_name> [options]

Example:
    # For Gemma2 with Orbax checkpoint (from Kaggle training):
    python research/combine_lora_push_to_hub.py \\
        /tmp/intermediate_ckpt/1 \\
        carlesoctav/gemma2-gsm8k-base \\
        --model-name gemma2-2b-it \\
        --rank 64 --alpha 64.0

    # For models with safetensors base:
    python research/combine_lora_push_to_hub.py \\
        /path/to/checkpoint \\
        my-org/my-model \\
        --base-model-path /path/to/hf/model \\
        --model-type gemma3
"""

import argparse
import os
import shutil
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import HfApi
import numpy as np
from orbax import checkpoint as ocp
import safetensors.numpy as safe_np


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
    model_type: str = "gemma3",
):
    """Save model with merged LoRA weights as safetensors.

    This function works with models that have a safetensors base model.

    Args:
        local_model_path: Path to the base model safetensors checkpoint directory.
        output_dir: Directory where the merged model will be saved.
        lora_model: Model instance with LoRA weights.
        rank: LoRA rank used during training.
        alpha: LoRA alpha used during training.
        model_type: Model type ('gemma3' or 'qwen3').
    """
    if model_type == "gemma3":
        from tunix.models.gemma3 import params as gemma3_params

        gemma3_params.save_lora_merged_model_as_safetensors(
            local_model_path=local_model_path,
            output_dir=output_dir,
            lora_model=lora_model,
            rank=rank,
            alpha=alpha,
        )
    elif model_type == "qwen3":
        from tunix.models.qwen3 import params as qwen3_params

        qwen3_params.save_lora_merged_model_as_safetensors(
            local_model_path=local_model_path,
            output_dir=output_dir,
            lora_model=lora_model,
            rank=rank,
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_gemma2_model_as_safetensors(
    model: nnx.Module,
    output_dir: str,
    model_name: str = "gemma2-2b-it",
):
    """Save a Gemma2 model (with merged LoRA) as safetensors for HuggingFace.

    Args:
        model: The Gemma2 model with LoRA weights already merged.
        output_dir: Directory where the model will be saved.
        model_name: Model name for config generation.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Get model state
    state = nnx.state(model)
    state_dict = nnx.to_pure_dict(state)

    # Convert to HuggingFace format
    hf_state = _convert_gemma2_to_hf_format(state_dict, model_name)

    # Save as safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    safe_np.save_file(hf_state, safetensors_path)

    # Create config.json
    _create_gemma2_config(output_dir, model_name)

    print(f"Model saved to {output_dir}")


def _convert_gemma2_to_hf_format(state_dict: dict, model_name: str) -> dict:
    """Convert Gemma2 NNX state dict to HuggingFace format."""
    hf_state = {}

    def flatten_dict(d, parent_key=""):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    flat_state = flatten_dict(state_dict)

    for key, value in flat_state.items():
        # Skip non-weight entries
        if not isinstance(value, (np.ndarray, jnp.ndarray)):
            continue

        # Convert to numpy
        if isinstance(value, jnp.ndarray):
            value = np.asarray(value)

        hf_key = _map_gemma2_key_to_hf(key)
        if hf_key:
            # Apply necessary transposes for HF format
            value = _transpose_for_hf(key, value)
            hf_state[hf_key] = value

    return hf_state


def _map_gemma2_key_to_hf(key: str) -> str | None:
    """Map Gemma2 NNX key to HuggingFace key."""
    # Embedder
    if "embedder.input_embedding" in key and key.endswith(".value"):
        return "model.embed_tokens.weight"

    # Final norm
    if "final_norm.scale" in key and key.endswith(".value"):
        return "model.norm.weight"

    # Layer mappings
    import re

    # Attention Q projection
    m = re.match(r"layers\.(\d+)\.attn\.q_einsum\.w\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.q_proj.weight"

    # Attention KV projection (need to split)
    m = re.match(r"layers\.(\d+)\.attn\.kv_einsum\.w\.value", key)
    if m:
        # This needs special handling - returns None, handled separately
        return None

    # Attention output projection
    m = re.match(r"layers\.(\d+)\.attn\.attn_vec_einsum\.w\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.o_proj.weight"

    # MLP projections
    m = re.match(r"layers\.(\d+)\.mlp\.gate_proj\.kernel\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.mlp.gate_proj.weight"

    m = re.match(r"layers\.(\d+)\.mlp\.up_proj\.kernel\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.mlp.up_proj.weight"

    m = re.match(r"layers\.(\d+)\.mlp\.down_proj\.kernel\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.mlp.down_proj.weight"

    # Layer norms
    m = re.match(r"layers\.(\d+)\.pre_attention_norm\.scale\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.input_layernorm.weight"

    m = re.match(r"layers\.(\d+)\.post_attn_norm\.scale\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.post_attention_layernorm.weight"

    m = re.match(r"layers\.(\d+)\.pre_ffw_norm\.scale\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.pre_feedforward_layernorm.weight"

    m = re.match(r"layers\.(\d+)\.post_ffw_norm\.scale\.value", key)
    if m:
        return f"model.layers.{m.group(1)}.post_feedforward_layernorm.weight"

    return None


def _transpose_for_hf(key: str, value: np.ndarray) -> np.ndarray:
    """Apply necessary transposes for HuggingFace format."""
    # MLP kernels need transpose
    if "mlp" in key and "kernel" in key:
        return value.T

    # Attention projections may need reshape
    if "q_einsum" in key or "attn_vec_einsum" in key:
        # Reshape from (num_heads, embed_dim, head_dim) or similar
        if value.ndim == 3:
            d0, d1, d2 = value.shape
            return value.reshape(d0 * d2, d1).T

    return value


def _create_gemma2_config(output_dir: str, model_name: str):
    """Create a config.json for the model."""
    import json

    # Gemma2 2B config
    if "2b" in model_name.lower():
        config = {
            "architectures": ["Gemma2ForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "head_dim": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 2304,
            "initializer_range": 0.02,
            "intermediate_size": 9216,
            "max_position_embeddings": 8192,
            "model_type": "gemma2",
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.42.0",
            "use_cache": True,
            "vocab_size": 256128,
        }
    else:
        # Default/9B config
        config = {
            "architectures": ["Gemma2ForCausalLM"],
            "model_type": "gemma2",
            "torch_dtype": "bfloat16",
        }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def push_to_hub(local_dir: str, hub_name: str, private: bool = False):
    """Push merged model to Hugging Face Hub.

    Args:
        local_dir: Local directory containing the merged model.
        hub_name: Hugging Face Hub repository name (e.g., 'my-org/my-model').
        private: Whether the repository should be private.
    """
    api = HfApi()

    # Create the repository if it doesn't exist
    api.create_repo(repo_id=hub_name, exist_ok=True, private=private)

    # Upload all files in the directory
    api.upload_folder(
        folder_path=local_dir,
        repo_id=hub_name,
        repo_type="model",
    )

    print(f"Model successfully pushed to: https://huggingface.co/{hub_name}")


def load_gemma2_with_lora_from_orbax(
    ckpt_path: str,
    model_name: str = "gemma2-2b-it",
    lora_rank: int = 64,
    lora_alpha: float = 64.0,
    lora_module_path: str = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
) -> nnx.Module:
    """Load a Gemma2 model with LoRA weights from Orbax checkpoint.

    Args:
        ckpt_path: Path to the Orbax checkpoint directory.
        model_name: Model name (e.g., 'gemma2-2b-it').
        lora_rank: LoRA rank used during training.
        lora_alpha: LoRA alpha used during training.
        lora_module_path: Regex pattern for LoRA modules.

    Returns:
        The model with LoRA weights loaded.
    """
    import qwix
    from tunix.cli.utils import model as model_lib
    from tunix.models.gemma import model as gemma_model_lib

    # Create mesh
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices).reshape(2, 2), axis_names=("fsdp", "tp"))

    # Get model config
    model_params = model_lib.obtain_model_params(model_name)

    # Create abstract model
    abs_model = nnx.eval_shape(
        lambda: gemma_model_lib.Gemma(model_params, rngs=nnx.Rngs(42))
    )

    # Apply LoRA to get the structure
    lora_provider = qwix.LoraProvider(
        module_path=lora_module_path,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    model_input = abs_model.get_model_input()
    lora_abs_model = qwix.apply_lora_to_model(abs_model, lora_provider, **model_input)

    # Prepare for restore
    abs_state = nnx.state(lora_abs_model)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )

    # Restore from checkpoint
    checkpointer = ocp.StandardCheckpointer()
    state_path = os.path.join(ckpt_path, "state")
    restored_params = checkpointer.restore(state_path, target=abs_state)

    # Merge graph def with restored params
    graph_def, _ = nnx.split(lora_abs_model)
    model = nnx.merge(graph_def, restored_params)

    return model


def merge_lora_weights(model: nnx.Module, rank: int, alpha: float) -> nnx.Module:
    """Merge LoRA weights into the base model weights.

    Args:
        model: Model with LoRA weights.
        rank: LoRA rank.
        alpha: LoRA alpha.

    Returns:
        Model with merged weights.
    """
    scaling = alpha / rank

    # Iterate through layers and merge LoRA weights
    for layer in model.layers:
        # Merge attention LoRA
        if hasattr(layer, "attn"):
            attn = layer.attn
            for proj_name in ["q_einsum", "kv_einsum", "attn_vec_einsum"]:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    if hasattr(proj, "w_lora_a") and hasattr(proj, "w_lora_b"):
                        lora_a = jnp.asarray(proj.w_lora_a.value)
                        lora_b = jnp.asarray(proj.w_lora_b.value)

                        # Handle different tensor shapes
                        if lora_a.ndim == 3:
                            d0, d1, d2 = lora_a.shape
                            lora_a_flat = lora_a.reshape(d0 * d1, d2)
                        else:
                            lora_a_flat = lora_a

                        if lora_b.ndim == 3:
                            d0, d1, d2 = lora_b.shape
                            lora_b_flat = lora_b.reshape(d0, d1 * d2)
                        else:
                            lora_b_flat = lora_b

                        delta = (lora_a_flat @ lora_b_flat) * scaling

                        # Reshape delta to match original weight shape
                        w_shape = proj.w.value.shape
                        if delta.shape != w_shape:
                            delta = delta.reshape(w_shape)

                        proj.w.value = proj.w.value + delta.astype(proj.w.value.dtype)

        # Merge MLP LoRA
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    if hasattr(proj, "kernel_lora_a") and hasattr(
                        proj, "kernel_lora_b"
                    ):
                        lora_a = jnp.asarray(proj.kernel_lora_a.value)
                        lora_b = jnp.asarray(proj.kernel_lora_b.value)
                        delta = (lora_a @ lora_b) * scaling
                        proj.kernel.value = proj.kernel.value + delta.astype(
                            proj.kernel.value.dtype
                        )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Combine LoRA weights with base model and push to Hub"
    )
    parser.add_argument(
        "ckpt",
        type=str,
        help="Path to the checkpoint directory (Orbax format)",
    )
    parser.add_argument(
        "hub_name",
        type=str,
        help="Hugging Face Hub repository name (e.g., 'my-org/my-model')",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma2-2b-it",
        help="Model name (default: gemma2-2b-it)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Path to the base model safetensors directory (for non-Orbax checkpoints)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gemma2", "gemma3", "qwen3"],
        default="gemma2",
        help="Model type (default: gemma2)",
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

    # Validate checkpoint path
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint path not found: {args.ckpt}")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = tempfile.mkdtemp(prefix="lora_merged_")

    print(f"Checkpoint path: {args.ckpt}")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Model type: {args.model_type}")
    print(f"LoRA rank: {args.rank}, alpha: {args.alpha}")

    if args.model_type == "gemma2":
        print("\nLoading Gemma2 model with LoRA from Orbax checkpoint...")
        model = load_gemma2_with_lora_from_orbax(
            ckpt_path=args.ckpt,
            model_name=args.model_name,
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            lora_module_path=args.lora_module_path,
        )

        print("Merging LoRA weights...")
        model = merge_lora_weights(model, args.rank, args.alpha)

        print("Saving model as safetensors...")
        save_gemma2_model_as_safetensors(model, output_dir, args.model_name)

    elif args.base_model_path:
        # For models with safetensors base (gemma3, qwen3)
        print(f"\nLoading model from safetensors base at {args.base_model_path}...")
        # This path requires the model to be loaded separately
        print(
            "Note: For gemma3/qwen3, please load your model and call "
            "save_lora_merged_model_as_safetensors() directly."
        )
        return
    else:
        print(
            f"\nError: For model type '{args.model_type}', please provide --base-model-path"
        )
        return

    print(f"\nModel saved to: {output_dir}")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f:<30} {size:>10.2f} MB")

    if not args.skip_push:
        print(f"\nPushing to Hub: {args.hub_name}")
        push_to_hub(output_dir, args.hub_name, private=args.private)


if __name__ == "__main__":
    main()
