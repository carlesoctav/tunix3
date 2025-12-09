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

"""Utilities for saving models with merged LoRA weights in safetensors format."""

import json
import os
import shutil
from typing import Any, Callable

import jax.numpy as jnp
import safetensors.numpy as safe_np


def _load_sharded_safetensors(local_model_path: str) -> dict:
  """Load safetensors files, handling both single and sharded formats.
  
  Args:
    local_model_path: Path to the model directory.
    
  Returns:
    Combined state dict from all safetensors files.
  """
  index_path = os.path.join(local_model_path, 'model.safetensors.index.json')
  single_path = os.path.join(local_model_path, 'model.safetensors')
  
  if os.path.exists(single_path):
    # Single file format
    return safe_np.load_file(single_path)
  elif os.path.exists(index_path):
    # Sharded format - load all shards and combine
    with open(index_path, 'r') as f:
      index = json.load(f)
    
    # Get unique shard files
    shard_files = set(index.get('weight_map', {}).values())
    
    combined_state = {}
    for shard_file in sorted(shard_files):
      shard_path = os.path.join(local_model_path, shard_file)
      if os.path.exists(shard_path):
        shard_state = safe_np.load_file(shard_path)
        combined_state.update(shard_state)
      else:
        raise FileNotFoundError(f"Shard file not found: {shard_path}")
    
    return combined_state
  else:
    # Try to find any safetensors file
    safetensor_files = [f for f in os.listdir(local_model_path) if f.endswith('.safetensors')]
    if safetensor_files:
      combined_state = {}
      for sf in sorted(safetensor_files):
        shard_state = safe_np.load_file(os.path.join(local_model_path, sf))
        combined_state.update(shard_state)
      return combined_state
    else:
      raise FileNotFoundError(
          f"No safetensors files found in {local_model_path}. "
          f"Expected 'model.safetensors' or 'model.safetensors.index.json'"
      )


def qwix_path_to_str(qwix_path) -> str:
  return '.'.join([str(field) for field in qwix_path])


def _extract_lora_from_component(
    component: Any, proj_name: str, lora_a_attr: str, lora_b_attr: str
) -> tuple[str, tuple[Any, Any]] | None:
  """Extracts LoRA weights from a component (attn or mlp) if projection exists.

  Args:
    component: The component (e.g., layer.attn or layer.mlp) to check.
    proj_name: Name of the projection to look for (e.g., 'q_proj', 'gate_proj').
    lora_a_attr: Name of the LoRA A matrix attribute (e.g., 'w_lora_a',
      'kernel_lora_a').
    lora_b_attr: Name of the LoRA B matrix attribute (e.g., 'w_lora_b',
      'kernel_lora_b').

  Returns:
    A tuple of (path_str, (lora_a, lora_b)) if the projection exists, None
    otherwise.
  """
  if hasattr(component, proj_name):
    proj = getattr(component, proj_name)
    path = qwix_path_to_str(proj.qwix_path)
    lora_a = getattr(proj, lora_a_attr)
    lora_b = getattr(proj, lora_b_attr)
    return (path, (lora_a, lora_b))
  return None


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
    state_key_transform_fn: Callable[[str], str],
    field_patterns: tuple[str, ...],
    custom_layer_extractor_fn: Callable[[Any], Any] | None = None,
):
  """Saves a model with LoRA weights merged in safetensors format.

  This is a generic function that can be used for any model architecture.
  Model-specific logic is provided via callback functions.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
    state_key_transform_fn: Function that transforms model layer paths to
      safetensors state dict keys.
    field_patterns: Tuple of projection field names to look for in each layer
      (both attn and mlp).
    custom_layer_extractor_fn: Optional function that extracts or updates LoRA
      layers for a given layer; it should accept the current layer and return a
      dict of the new/updated LoRA layers' names as strings to a tuple of the
      corresponding lora pair.
  """

  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  # Extract LoRA layers using the model-specific function
  lora_layers = {}
  for layer in lora_model.layers:
    for proj_name in field_patterns:
      # Check attention layers
      if result := _extract_lora_from_component(
          layer.attn, proj_name, 'w_lora_a', 'w_lora_b'
      ):
        path, lora_params = result
        lora_layers[path] = lora_params

      # Check MLP layers
      if result := _extract_lora_from_component(
          layer.mlp, proj_name, 'kernel_lora_a', 'kernel_lora_b'
      ):
        path, lora_params = result
        lora_layers[path] = lora_params

    if custom_layer_extractor_fn:
      lora_layers |= custom_layer_extractor_fn(layer)

  # Load base model state (handles both single and sharded formats)
  base_state = _load_sharded_safetensors(local_model_path)

  # Apply LoRA deltas
  for lora_name, (lora_a, lora_b) in lora_layers.items():
    state_key = state_key_transform_fn(lora_name)
    assert (
        state_key in base_state
    ), f'LoRA layer {lora_name} not found in base model state dict'

    lora_a_val = jnp.asarray(getattr(lora_a, 'value', lora_a))
    lora_b_val = jnp.asarray(getattr(lora_b, 'value', lora_b))

    # Reshape 3D tensors to 2D if necessary
    if lora_a_val.ndim == 3:
      d0, d1, d2 = lora_a_val.shape
      lora_a_val = lora_a_val.reshape(d0 * d1, d2)
    if lora_b_val.ndim == 3:
      d0, d1, d2 = lora_b_val.shape
      lora_b_val = lora_b_val.reshape(d0, d1 * d2)

    # Compute and apply LoRA delta
    combined_lora = (lora_a_val @ lora_b_val) * (alpha / rank)
    base_state[state_key] += combined_lora.T.astype(base_state[state_key].dtype)

  # Save merged model
  safetensors_path = os.path.join(output_dir, 'model.safetensors')
  safe_np.save_file(base_state, safetensors_path)

  # Copy non-safetensors files (config, tokenizer, etc.)
  # Skip index files since we're saving as a single file
  skip_patterns = ('.safetensors', '.safetensors.index.json')
  for filename in os.listdir(local_model_path):
    if not any(filename.endswith(pat) for pat in skip_patterns):
      src = os.path.join(local_model_path, filename)
      if os.path.isfile(src):
        dst = os.path.join(output_dir, filename)
        shutil.copy(src, dst)
