"""GSM8K evaluation script using tunix sampler and reward function."""

import enum
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import draccus
import jax
from dotenv import load_dotenv
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import Mesh
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from research.reward import gsm8k as gsm8k_reward
from tunix.cli.utils import model as cli_model_lib
from tunix.examples.data import math_dataset
from tunix.generate import sampler as sampler_lib
from tunix.sft import checkpoint_manager


# --- Configuration Classes ---


class ModelFamily(enum.Enum):
    """Model family enum for dispatching to correct model loader."""

    Gemma2 = enum.auto()
    Gemma3 = enum.auto()


@dataclass
class LoraConfig:
    """LoRA configuration."""

    rank: int = 8
    alpha: float = 8.0
    module_path: str = (
        ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "module_path": self.module_path,
        }


@dataclass
class ModelArgs:
    """Model configuration arguments."""

    model_family: ModelFamily = ModelFamily.Gemma3
    model_name: str = "gemma3-1b-it"
    model_id: str = "google/gemma-3-1b-it"
    hf_tokenizer_path: str = "google/gemma-3-1b-it"
    mesh_axis_names: tuple[str, ...] = ("fsdp", "tp")
    mesh_shape: tuple[int, ...] = (1, 1)
    lora_config: LoraConfig | None = None
    lora_checkpoint_path: str | None = (
        None  # GCS path like gs://bucket/path/to/checkpoint
    )
    rng_seed: int = 42

    def create_mesh(self) -> Mesh:
        device_count = jax.device_count()
        mesh_shape = self.mesh_shape
        if device_count < mesh_shape[0] * mesh_shape[1]:
            mesh_shape = (device_count, 1)
            print(f"Adjusted mesh to ({device_count}, 1)")
        return jax.make_mesh(mesh_shape, axis_names=self.mesh_axis_names)

    def make(self) -> tuple[nnx.Module, PreTrainedTokenizerBase, Mesh, Any]:
        """Create model, tokenizer, mesh, and model_config."""
        mesh = self.create_mesh()

        print(f"Downloading/Loading {self.model_id} from Hugging Face...")
        local_model_path = snapshot_download(
            repo_id=self.model_id,
            ignore_patterns=["*.pth"],
        )
        print(f"Model path: {local_model_path}")

        if self.model_family == ModelFamily.Gemma2:
            from tunix.models.gemma import model as gemma_model
            from tunix.models.gemma import params_safetensors as params

            config_method_name = self.model_name.replace("-", "_")
            config_fn = getattr(gemma_model.ModelConfig, config_method_name)
            model_config = config_fn()
            model = params.create_model_from_safe_tensors(
                local_model_path, model_config, mesh
            )
        elif self.model_family == ModelFamily.Gemma3:
            from tunix.models.gemma3 import model as gemma3_model
            from tunix.models.gemma3 import params_safetensors as params3

            config_method_name = self.model_name.replace("-", "_")
            config_fn = getattr(gemma3_model.ModelConfig, config_method_name)
            model_config = config_fn()
            model = params3.create_model_from_safe_tensors(
                local_model_path, model_config, mesh
            )
        else:
            raise NotImplementedError(f"Model family {self.model_family} not supported")

        if self.lora_config:
            print("Applying LoRA to model...")
            model = cli_model_lib.apply_lora_to_model(
                model, mesh, self.lora_config.to_dict()
            )
            print("LoRA applied")

            if self.lora_checkpoint_path:
                print(f"Loading LoRA weights from {self.lora_checkpoint_path}")
                ckpt_mgr = checkpoint_manager.CheckpointManager(
                    root_directory=self.lora_checkpoint_path
                )
                step = ckpt_mgr.latest_step()
                if step is None:
                    print(f"No checkpoint found at {self.lora_checkpoint_path}")
                else:
                    print(f"Restoring from step {step}")
                    ckpt_mgr.maybe_restore(
                        model, step=step, restore_only_lora_params=True
                    )
                    print("LoRA weights restored successfully")

        # Load tokenizer from HuggingFace with original chat template
        tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)
        return model, tokenizer, mesh, model_config


@dataclass
class EvalArgs:
    """Arguments for evaluation script."""

    model: ModelArgs = field(default_factory=ModelArgs)
    data_dir: str = "./data/test"
    output_dir: str = "./eval_results"
    batch_size: int = 4
    num_examples: int | None = None
    temperature: float = 0.0  # Greedy by default
    top_k: int | None = None
    top_p: float | None = None
    max_generation_steps: int = 768


# --- Generation and Evaluation ---


def generate(
    questions: list[str],
    sampler,
    tokenizer,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> tuple[list[str], list[str]]:
    """Generate completions for a batch of questions.

    Returns:
        (prompts, completions)
    """
    input_batch = []
    for q in questions:
        # Use original HF tokenizer chat template with just the question
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_batch.append(prompt)

    out_data = sampler(
        input_strings=input_batch,
        max_generation_steps=768,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
        eos_tokens=[1, 106],
    )

    return input_batch, out_data.text


def evaluate(
    dataset,
    sampler,
    tokenizer,
    output_dir: str,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[int, int, float]:
    """Evaluate model on GSM8K using binary reward (0 or 1).

    Returns:
        (correct, total, accuracy)
    """
    correct = 0
    total = 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output file for rollouts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"rollouts_{timestamp}.jsonl")

    print(f"Writing rollouts to {output_file}")

    with open(output_file, "w") as f:
        print("Starting evaluation...")
        for batch in tqdm(dataset):
            questions = batch["question"]
            answers = batch["answer"]

            # Handle different data formats
            if isinstance(questions, (list, tuple)):
                pass  # Already a list
            elif hasattr(questions, "tolist"):
                questions = questions.tolist()
            else:
                questions = [questions]
                answers = [answers]

            prompts, responses = generate(
                questions, sampler, tokenizer, temperature, top_k, top_p
            )

            for prompt, response, question, answer in zip(
                prompts, responses, questions, answers
            ):
                answer_str = str(answer)

                # Use flexible extraction (same as reward function)
                score = gsm8k_reward.compute_score(
                    response, answer_str, method="flexible"
                )
                extracted = gsm8k_reward.extract_gsm8k_answer(
                    response, method="flexible"
                )

                correct += int(score)
                total += 1

                # Write rollout to file
                rollout_data = {
                    "id": total,
                    "question": question,
                    "prompt": prompt,
                    "response": response,
                    "expected_answer": answer_str,
                    "extracted_answer": extracted,
                    "correct": int(score),
                }
                f.write(json.dumps(rollout_data) + "\n")
                f.flush()

                if total % 10 == 0:
                    acc = correct / total * 100
                    print(f"Progress: {correct}/{total} = {acc:.2f}%")

    accuracy = correct / total * 100 if total > 0 else 0.0

    # Write summary
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
            f,
            indent=2,
        )
    print(f"Summary written to {summary_file}")

    return correct, total, accuracy


@draccus.wrap()
def main(cfg: EvalArgs):
    load_dotenv()

    print(f"Loading model: {cfg.model.model_name}")
    model, tokenizer, mesh, model_config = cfg.model.make()

    MAX_PROMPT_LENGTH = 1024
    TOTAL_GENERATION_STEPS = cfg.max_generation_steps

    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    print("Loading GSM8K test dataset...")
    dataset = math_dataset.get_dataset(cfg.data_dir, "test")
    dataset = dataset.batch(cfg.batch_size)

    if cfg.num_examples:
        num_batches = cfg.num_examples // cfg.batch_size
        dataset = dataset[:num_batches]

    correct, total, accuracy = evaluate(
        dataset,
        sampler,
        tokenizer,
        output_dir=cfg.output_dir,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
    )

    print(f"\n{'=' * 50}")
    print(f"Final Results: {correct}/{total} = {accuracy:.2f}%")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
