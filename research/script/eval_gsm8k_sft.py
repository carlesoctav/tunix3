"""GSM8K evaluation for SFT LoRA with custom think template - configurable exp name."""

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


GEMMA_THINK_TEMPLATE = """{{ bos_token }}{%- set first_user_prefix = 'Think through your approach in <reasoning></reasoning> tags, then provide your complete response in <answer></answer> tags. For creative tasks, briefly plan in reasoning, then write the full creative output in answer.\n\n' -%}{%- set loop_messages = messages -%}{%- for message in loop_messages -%}{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}{{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}{%- endif -%}{%- if (message['role'] == 'assistant') -%}{%- set role = "model" -%}{%- else -%}{%- set role = message['role'] -%}{%- endif -%}{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}{%- if message['role'] == 'assistant' -%}{%- generation -%}{%- if message['content'] is string -%}{{ message['content'] | trim }}{%- elif message['content'] is iterable -%}{%- for item in message['content'] -%}{%- if item['type'] == 'image' -%}{{ '<start_of_image>' }}{%- elif item['type'] == 'text' -%}{{ item['text'] | trim }}{%- endif -%}{%- endfor -%}{%- else -%}{{ raise_exception("Invalid content type") }}{%- endif -%}{{ '<end_of_turn>' }}{%- endgeneration -%}{{ '\n' }}{%- else -%}{%- if message['content'] is string -%}{{ message['content'] | trim }}{%- elif message['content'] is iterable -%}{%- for item in message['content'] -%}{%- if item['type'] == 'image' -%}{{ '<start_of_image>' }}{%- elif item['type'] == 'text' -%}{{ item['text'] | trim }}{%- endif -%}{%- endfor -%}{%- else -%}{{ raise_exception("Invalid content type") }}{%- endif -%}{%- endif -%}{%- if message['role'] != 'assistant' -%}{{ '<end_of_turn>\n' }}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{'<start_of_turn>model\n'}}{%- endif -%}"""


class ModelFamily(enum.Enum):
    Gemma2 = enum.auto()
    Gemma3 = enum.auto()


@dataclass
class LoraConfig:
    rank: int = 256
    alpha: float = 512.0
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

    def make(
        self, lora_checkpoint_path: str
    ) -> tuple[nnx.Module, PreTrainedTokenizerBase, Mesh, Any]:
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

        print(f"Loading LoRA weights from {lora_checkpoint_path}")
        ckpt_mgr = checkpoint_manager.CheckpointManager(
            root_directory=lora_checkpoint_path
        )
        step = ckpt_mgr.latest_step()
        if step is None:
            raise ValueError(f"No checkpoint found at {lora_checkpoint_path}")
        print(f"Restoring from step {step}")
        ckpt_mgr.maybe_restore(model, step=step, restore_only_lora_params=True)
        print("LoRA weights restored successfully")

        tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)
        tokenizer.chat_template = GEMMA_THINK_TEMPLATE
        return model, tokenizer, mesh, model_config


@dataclass
class EvalArgs:
    exp_name: str = "4b-generated-Dolci-Instruct-SFT-No-Tools-rank-256-lr-1e6"
    gcs_bucket: str = "gs://carles-git-good/tunix-final-sft"
    model: ModelArgs = field(default_factory=ModelArgs)
    data_dir: str = "./data/test"
    output_base_dir: str = "./eval_results"
    batch_size: int = 4
    num_examples: int | None = None
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    max_generation_steps: int = 768

    @property
    def lora_checkpoint_path(self) -> str:
        return f"{self.gcs_bucket}/{self.exp_name}"

    @property
    def output_dir(self) -> str:
        return f"{self.output_base_dir}/{self.exp_name}"


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
    exp_name: str,
    lora_checkpoint_path: str,
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
                "exp_name": exp_name,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "lora_checkpoint": lora_checkpoint_path,
                "lora_rank": 256,
                "lora_alpha": 512,
                "prompt_template": "gemma_think (reasoning/answer tags)",
            },
            f,
            indent=2,
        )
    print(f"Summary written to {summary_file}")

    return correct, total, accuracy


@draccus.wrap()
def main(cfg: EvalArgs):
    load_dotenv()

    print(f"=" * 60)
    print(f"Experiment: {cfg.exp_name}")
    print(f"LoRA checkpoint: {cfg.lora_checkpoint_path}")
    print(f"Output dir: {cfg.output_dir}")
    print(f"=" * 60)

    model, tokenizer, mesh, model_config = cfg.model.make(cfg.lora_checkpoint_path)

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
        exp_name=cfg.exp_name,
        lora_checkpoint_path=cfg.lora_checkpoint_path,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
    )

    print(f"\n{'=' * 60}")
    print(f"Final Results ({cfg.exp_name}): {correct}/{total} = {accuracy:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
