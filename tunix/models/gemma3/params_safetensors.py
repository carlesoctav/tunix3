# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

"""Loads Gemma3 parameters from safetensors files."""

import re

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.gemma3 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of torch_keys to (nnx_keys, (permute_rule, reshape_rule))."""
  return {
      r"(?:language_model\.)?model\.embed_tokens\.weight": (
          "embedder.input_embedding",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"tmp.layers.\1.attn.q",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"tmp.layers.\1.attn.k",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"tmp.layers.\1.attn.v",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.attn_vec_einsum.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.pre_attention_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.(post_feedforward_layernorm|post_ffn_layernorm|post_ffw_layernorm)\.weight": (
          r"layers.\1.post_ffw_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.norm\.weight": ("final_norm.scale", None),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn._query_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn._key_norm.scale",
          None,
      ),
      r"(?:language_model\.)?lm_head\.weight": ("unused.lm_head.weight", None),
      r"(?:language_model\.)?lm_head\.bias": ("unused.lm_head.bias", None),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.bias": (
          r"unused.layers.\1.attn.\2.bias",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.input_layernorm\.bias": (
          r"unused.layers.\1.input_layernorm.bias",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.post_attention_layernorm\.bias": (
          r"unused.layers.\1.post_attention_layernorm.bias",
          None,
      ),
      r"(?:language_model\.)?model\.rotary_emb\..*": (
          "unused.rotary_emb",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.rotary_emb\..*": (
          r"unused.layers.\1.attn.rotary_emb",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.qkv_proj\.weight": (
          r"unused.layers.\1.attn.qkv_proj.weight",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.(pre_ffn_layernorm|pre_ffw_layernorm)\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
      r"(?:language_model\.)?multi_modal_projector\.*": (
          "unused.multi_modal_projector",
          None,
      ),
      r"(?:language_model\.)?vision_tower\.*": ("unused.vision_tower", None),
  }


def _make_preprocess_fn(cfg: model_lib.ModelConfig):
  """Creates a tensor preprocessing function for Gemma3 safetensors, fusing q, k, and v projections."""
  q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
  k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
  v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")

  fused_qkv = cfg.num_heads == cfg.num_kv_heads
  pending: dict[str, dict[str, jnp.ndarray]] = {}

  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma3 head_dim must be even for RoPE, got {cfg.head_dim}"
    )

  def _to_ndh(q: jnp.ndarray) -> jnp.ndarray:
    # Expected shape: (N, D, H)
    if q.shape == (cfg.num_heads, cfg.embed_dim, cfg.head_dim):
      return q
    if q.shape == (cfg.embed_dim, cfg.num_heads, cfg.head_dim):  # D,N,H
      return jnp.transpose(q, (1, 0, 2))  # -> N,D,H
    if q.shape == (cfg.num_heads, cfg.head_dim, cfg.embed_dim):  # N,H,D
      return jnp.transpose(q, (0, 2, 1))  # -> N,D,H
    raise ValueError(f"[gemma3 preprocess] unexpected q shape: {q.shape}")

  def _to_kdh(x: jnp.ndarray) -> jnp.ndarray:
    if x.shape == (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim):
      return x
    if x.shape == (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim):  # D,K,H
      return jnp.transpose(x, (1, 0, 2))  # -> K,D,H
    if x.shape == (cfg.num_kv_heads, cfg.head_dim, cfg.embed_dim):  # K,H,D
      return jnp.transpose(x, (0, 2, 1))  # -> K,D,H
    raise ValueError(f"[gemma3 preprocess] unexpected kv shape: {x.shape}")

  def preprocess(tensors: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    out = dict(tensors)

    for key in list(out):
      m = q_pat.fullmatch(key) or k_pat.fullmatch(key) or v_pat.fullmatch(key)
      if not m:
        continue
      layer_id = m.group(1)
      arr = out.pop(key)
      slot = "q" if key.endswith(".q") else ("k" if key.endswith(".k") else "v")
      pending.setdefault(layer_id, {})[slot] = arr

    for layer_id, slots in list(pending.items()):
      q = slots.get("q")
      k = slots.get("k")
      v = slots.get("v")

      if fused_qkv:
        if (q is not None) and (k is not None) and (v is not None):
          q = _to_ndh(q)
          k = _to_kdh(k)
          v = _to_kdh(v)
          exp = (cfg.num_heads, cfg.embed_dim, cfg.head_dim)
          if not (q.shape == k.shape == v.shape == exp):
            raise ValueError(
                f"[gemma3 preprocess] layer {layer_id}: fused q/k/v shape"
                f" mismatch: q={getattr(q, 'shape', None)},"
                f" k={getattr(k, 'shape', None)},"
                f" v={getattr(v, 'shape', None)}, expected={exp}"
            )
          out[f"layers.{layer_id}.attn.qkv_einsum.w"] = jnp.stack(
              [q, k, v], axis=0
          )  # (3,N,D,H)
          del pending[layer_id]
      else:
        wrote = False
        if q is not None:
          q = _to_ndh(q)
          out[f"layers.{layer_id}.attn.q_einsum.w"] = q
          slots.pop("q", None)
          wrote = True

        if (k is not None) and (v is not None):
          k = _to_kdh(k)
          v = _to_kdh(v)
          out[f"layers.{layer_id}.attn.kv_einsum.w"] = jnp.stack(
              [k, v], axis=0
          )  # (2,K,D,H)
          slots.pop("k", None)
          slots.pop("v", None)
          wrote = True

        if wrote and not slots:
          del pending[layer_id]

    return out

  return preprocess


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
):
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Gemma3,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_make_preprocess_fn(config),
      dtype=dtype,
  )


def _extract_gemma3_lora_layers(layer) -> dict:
  """Extract LoRA weights from kv_einsum which has fused K and V.

  For Gemma3, kv_einsum has shape (2, num_kv_heads, embed_dim, head_dim) where
  index 0 is K and index 1 is V. We split the LoRA weights accordingly.

  Args:
    layer: A transformer layer with attention module.

  Returns:
    Dict mapping custom extracted layer paths to (lora_a, lora_b) tuples.
  """
  if hasattr(layer.attn, 'kv_einsum'):
    proj = layer.attn.kv_einsum
    if hasattr(proj, 'w_lora_a') and hasattr(proj, 'w_lora_b'):
      path = safetensors_saver.qwix_path_to_str(proj.qwix_path)
      # w_lora_b has shape (rank, 2, num_kv_heads, head_dim)
      # Split along dim 1 for K and V
      return {
          path.replace('kv_einsum', 'k_einsum'): (
              proj.w_lora_a,
              proj.w_lora_b[:, 0],  # K weights
          ),
          path.replace('kv_einsum', 'v_einsum'): (
              proj.w_lora_a,
              proj.w_lora_b[:, 1],  # V weights
          ),
      }
  return {}


def _gemma3_state_key_to_safetensors_key(lora_name: str) -> str:
  """Transform Gemma3 layer path to safetensors state dict key.

  Args:
    lora_name: Internal layer path (e.g., 'layers.0.attn.q_einsum').

  Returns:
    Safetensors state dict key (e.g., 'model.layers.0.self_attn.q_proj.weight').
  """
  return (
      f'model.{lora_name}.weight'.replace('.attn.', '.self_attn.')
      .replace('q_einsum', 'q_proj')
      .replace('k_einsum', 'k_proj')
      .replace('v_einsum', 'v_proj')
      .replace('attn_vec_einsum', 'o_proj')
  )


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Gemma3,
    rank: int,
    alpha: float,
):
  """Saves a Gemma3 model with LoRA weights merged in safetensors format.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Gemma3 model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
  """
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=_gemma3_state_key_to_safetensors_key,
      field_patterns=(
          'q_einsum',
          'attn_vec_einsum',
          'gate_proj',
          'up_proj',
          'down_proj',
      ),
      custom_layer_extractor_fn=_extract_gemma3_lora_layers,
  )
