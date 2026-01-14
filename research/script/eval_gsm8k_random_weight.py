"""GSM8K evaluation for random weight LoRA."""

import enum
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import draccus
import jax
import jax.numpy as jnp
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


reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


class ModelFamily(enum.Enum):
    Gemma2 = enum.auto()
    Gemma3 = enum.auto()


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    module_path: str = (
        ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "module_path": self.module_path,
        }


def check_ab_zero(model):
    print("Checking if LoRA AB = 0...")
    # Try to get LoRA parameters
    try:
        # Use nnx.state with LoRAParam. If it fails, try Variable.
        state = nnx.state(model, nnx.LoRAParam)
    except Exception as e:
        print(f"Error getting LoRA state: {e}")
        state = nnx.state(model, nnx.Variable)

    if not state:
        # Try getting all variables
        state = nnx.state(model, nnx.Variable)

    # print(f"State keys: {list(state.keys())[:5]}... Total: {len(state)}")

    lora_layers = {}
    for path, var in state.items():
        is_a = any(part in ["w_lora_a", "kernel_lora_a"] for part in path)
        is_b = any(part in ["w_lora_b", "kernel_lora_b"] for part in path)

        if is_a or is_b:
            base_path = []
            for part in path:
                if part in ["w_lora_a", "kernel_lora_a", "w_lora_b", "kernel_lora_b"]:
                    break
                base_path.append(part)
            base_path = tuple(base_path)
            if base_path not in lora_layers:
                lora_layers[base_path] = {}
            if is_a:
                lora_layers[base_path]["a"] = var.value
            else:
                lora_layers[base_path]["b"] = var.value

    if not lora_layers:
        print("No LoRA layers found in state!")
        return True

    all_zero = True
    for path, weights in lora_layers.items():
        if "a" in weights and "b" in weights:
            a = jnp.asarray(weights["a"])
            b = jnp.asarray(weights["b"])

            if a.ndim == 3:
                d0, d1, d2 = a.shape
                a_flat = a.reshape(d0 * d1, d2)
            else:
                a_flat = a

            try:
                # Based on safetensors_saver.py, it's (lora_a_val @ lora_b_val)
                if a_flat.shape[-1] == b.shape[0]:
                    prod = jnp.matmul(a_flat, b)
                elif b.shape[-1] == a_flat.shape[0]:
                    prod = jnp.matmul(b, a_flat)
                else:
                    print(
                        f"Dimension mismatch for {path}: A_flat={a_flat.shape}, B={b.shape}"
                    )
                    all_zero = False
                    continue

                max_val = jnp.max(jnp.abs(prod))
                if max_val > 1e-6:
                    # print(f"  NOT ZERO for {path}: max_val={max_val}")
                    all_zero = False
            except Exception as e:
                print(f"  Error for {path}: {e}")
                all_zero = False

    if all_zero:
        print("Result: All LoRA AB products are zero.")
    else:
        print("Result: Some LoRA AB products are NOT zero.")
    return all_zero


def randomize_lora_weights(model, seed=42):
    print(f"Randomizing LoRA weights with seed {seed}...")
    key = jax.random.PRNGKey(seed)

    # Try to get all variables to be safe
    state = nnx.state(model, nnx.Variable)

    new_flat_state = {}
    for path, var in state.items():
        if any(
            part in ["w_lora_a", "kernel_lora_a", "w_lora_b", "kernel_lora_b"]
            for part in path
        ):
            key, subkey = jax.random.split(key)
            # Use a small scale to avoid complete divergence
            new_val = (
                jax.random.normal(subkey, var.value.shape, dtype=var.value.dtype) * 0.02
            )
            new_flat_state[path] = var.replace(value=new_val)
        else:
            new_flat_state[path] = var

    nnx.update(model, nnx.State.from_flat_path(new_flat_state))


@dataclass
class ModelArgs:
    model_family: ModelFamily = ModelFamily.Gemma3
    model_name: str = "gemma3-1b-it"
    model_id: str = "google/gemma-3-1b-it"
    hf_tokenizer_path: str = "google/gemma-3-1b-it"
    mesh_axis_names: tuple[str, ...] = ("fsdp", "tp")
    mesh_shape: tuple[int, ...] = (1, 1)
    lora_config: LoraConfig = field(default_factory=LoraConfig)
    rng_seed: int = 42

    def create_mesh(self) -> Mesh:
        device_count = jax.device_count()
        mesh_shape = self.mesh_shape
        if device_count < mesh_shape[0] * mesh_shape[1]:
            mesh_shape = (device_count, 1)
            print(f"Adjusted mesh to ({device_count}, 1)")
        return jax.make_mesh(mesh_shape, axis_names=self.mesh_axis_names)

    def make(self) -> tuple[nnx.Module, PreTrainedTokenizerBase, Mesh, Any]:
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

        print("Applying LoRA to model...")
        model = cli_model_lib.apply_lora_to_model(
            model, mesh, self.lora_config.to_dict()
        )
        print(
            f"LoRA applied (rank={self.lora_config.rank}, alpha={self.lora_config.alpha})"
        )

        # Check AB before randomization

        print("Checking AB before randomization:")
        check_ab_zero(model)

        # Randomize LoRA weights to satisfy "random weight" requirement
        randomize_lora_weights(model, self.rng_seed)

        # Check AB after randomization
        print("Checking AB after randomization:")
        check_ab_zero(model)

        tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)
        return model, tokenizer, mesh, model_config


@dataclass
class EvalArgs:
    rank: int = 8
    model: ModelArgs = field(default_factory=ModelArgs)
    data_dir: str = "./data/test"
    output_dir: str = "./eval_results_random_weight"
    batch_size: int = 4
    num_examples: int | None = None
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    max_generation_steps: int = 768


def generate(
    questions: list[str],
    sampler,
    tokenizer,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> tuple[list[str], list[str]]:
    input_batch = []
    for q in questions:
        prompt = TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q)
        print(f"DEBUGPRINT[52]: eval_gsm8k_random_weight.py:247: prompt={prompt}")
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
    correct = 0
    total = 0

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"rollouts_{timestamp}.jsonl")

    print(f"Writing rollouts to {output_file}")

    with open(output_file, "w") as f:
        print("Starting evaluation...")
        for batch in tqdm(dataset):
            questions = batch["question"]
            answers = batch["answer"]

            if isinstance(questions, (list, tuple)):
                pass
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

                score = gsm8k_reward.compute_score(
                    response, answer_str, method="flexible"
                )
                extracted = gsm8k_reward.extract_gsm8k_answer(
                    response, method="flexible"
                )

                correct += int(score)
                total += 1

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
                "lora_type": "random_weight",
            },
            f,
            indent=2,
        )
    print(f"Summary written to {summary_file}")

    return correct, total, accuracy


@draccus.wrap()
def main(cfg: EvalArgs):
    load_dotenv()

    # Sync rank from EvalArgs to LoraConfig
    cfg.model.lora_config.rank = cfg.rank
    # Use alpha = 2 * rank as a common heuristic
    cfg.model.lora_config.alpha = float(2 * cfg.rank)

    print(
        f"Loading model: {cfg.model.model_name} with random weight LoRA (rank={cfg.rank})"
    )
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
    print(
        f"Final Results (Random Weight LoRA, rank={cfg.rank}): {correct}/{total} = {accuracy:.2f}%"
    )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
