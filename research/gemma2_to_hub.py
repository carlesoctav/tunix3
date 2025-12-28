import argparse
import os
from pathlib import Path
import jax
import flax
from flax import nnx
from huggingface_hub import snapshot_download, HfApi
import orbax.checkpoint as ocp

from tunix.models.gemma import model as gemma_model
from tunix.models.gemma import params_safetensors as params
from tunix.cli.utils import model as cli_model_lib

def main():
    parser = argparse.ArgumentParser(description="Convert Gemma2 LoRA to SafeTensors and upload to Hub")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the LoRA checkpoint directory (e.g., gs://carles-git-good/tunix/gemma2-op-rank-16/actor)")
    parser.add_argument("--base_model_id", type=str, default="google/gemma-2-2b-it", help="Base model ID on Hugging Face")
    parser.add_argument("--output_dir", type=str, default="exported_model", help="Directory to save the merged model")
    parser.add_argument("--upload_repo_id", type=str, required=True, help="Hugging Face Repo ID to upload to")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    
    args = parser.parse_args()

    print(f"Downloading base model {args.base_model_id}...")
    local_model_path = snapshot_download(repo_id=args.base_model_id)
    print(f"Base model downloaded to {local_model_path}")

    print("Creating model...")
    # Infer config from model name
    model_name_for_config = args.base_model_id.split("/")[-1].replace("-", "_")
    
    try:
        config_fn = getattr(gemma_model.ModelConfig, model_name_for_config)
    except AttributeError:
        # Fallback for known naming discrepancies
        try:
             config_fn = getattr(gemma_model.ModelConfig, model_name_for_config.replace("gemma_2", "gemma2"))
        except AttributeError:
             print(f"Could not find config for {model_name_for_config} in gemma_model.ModelConfig.")
             raise

    config = config_fn()
    
    # Create mesh
    mesh = jax.make_mesh((1, 1), ('fsdp', 'tp'))

    # Load base model
    model = params.create_model_from_safe_tensors(local_model_path, config, mesh)

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = {
        "rank": args.rank,
        "alpha": args.alpha,
        "module_path": ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    }
    model = cli_model_lib.apply_lora_to_model(model, mesh, lora_config)

    # Restore Checkpoint
    print(f"Restoring checkpoint from {args.checkpoint_path}...")
    
    # CheckpointManagerOptions
    options = ocp.CheckpointManagerOptions(max_to_keep=2)
    
    # 'model_params' is the key used in tunix.sft.checkpoint_manager
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path.startswith("gs://"):
        checkpoint_path = Path(checkpoint_path).resolve()

    ckpt_mgr = ocp.CheckpointManager(
        checkpoint_path, 
        options=options, 
        item_names=('model_params', )
    )
    
    step = ckpt_mgr.latest_step()
    if step is None:
        print(f"No checkpoint found at {args.checkpoint_path}")
        return

    print(f"Latest step: {step}")
    
    # Restore logic
    # We only care about LoRA params as base params are frozen/not saved in PeftTrainer usually
    # PeftTrainer uses save_only_lora_params=True
    
    abstract_state = nnx.state(model, nnx.LoRAParam)
    
    # Create restore args
    def map_to_pspec(data):
        return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)

    restore_args_dict = jax.tree_util.tree_map(map_to_pspec, abstract_state)
    checkpoint_args = ocp.args.PyTreeRestore(
        item=abstract_state, restore_args=restore_args_dict
    )
    
    ckpt = ckpt_mgr.restore(
        step,
        args=ocp.args.Composite(
            model_params=checkpoint_args,
        ),
    )
    
    # Update model with restored params
    nnx.update(model, ckpt.model_params)
    print("Checkpoint restored.")

    # Save Merged
    print(f"Saving merged model to {args.output_dir}...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    params.save_lora_merged_model_as_safetensors(
        local_model_path=local_model_path,
        output_dir=args.output_dir,
        lora_model=model,
        rank=args.rank,
        alpha=args.alpha
    )
    print(f"Merged model saved to {args.output_dir}")

    # Upload
    print(f"Uploading to {args.upload_repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=args.upload_repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=args.upload_repo_id,
        repo_type="model"
    )
    print("Upload complete.")

if __name__ == "__main__":
    main()