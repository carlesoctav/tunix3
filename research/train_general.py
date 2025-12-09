"""
Comprehensive GRPO training pipeline for general-purpose research experiments.

This module provides a structured way to configure and run GRPO training
with support for LLM-as-a-judge reward, LoRA, and flexible model/training configurations.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Load Kaggle credentials from ~/.kaggle/kaggle.json
_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if _kaggle_json.exists():
    with open(_kaggle_json) as f:
        _kaggle_creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = _kaggle_creds["username"]
        os.environ["KAGGLE_KEY"] = _kaggle_creds["key"]

from enum import Enum
from typing import Any, Callable, Iterable

import jax
import optax
from flax import nnx
from jax.sharding import Mesh
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from research.data import DataWithRewardConfig
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.grpo.grpo_learner import GRPOConfig
from tunix.rl.rollout import base_rollout
import orbax.checkpoint as ocp
from tunix.sft import metrics_logger


class RematPolicy(str, Enum):
    """Remat policy for gradient checkpointing."""

    NONE = "none"
    FULL = "full"
    MINIMAL = "minimal"

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

    model_name: str
    model_id: str
    model_source: str
    hf_tokenizer_path: str
    model_download_path: str = "/mnt/carles/models"
    intermediate_ckpt_dir: str = "/mnt/carles/models/intermediate_ckpt"
    _resolved_intermediate_ckpt_dir: str = ""
    mesh_axis_names: tuple[str, ...] = ("fsdp", "tp")
    actor_mesh_shape: tuple[int, ...] = (4, 1)
    rollout_mesh_shape: tuple[int, ...] | None = None  # Defaults to actor_mesh_shape
    lora_config: LoraConfig | None = None
    remat: RematPolicy = RematPolicy.NONE
    rng_seed: int = 42

    def __post_init__(self):
        if self.rollout_mesh_shape is None:
            self.rollout_mesh_shape = self.actor_mesh_shape
        # Make intermediate_ckpt_dir unique per model to avoid conflicts on reruns
        self._resolved_intermediate_ckpt_dir = os.path.join(
            self.intermediate_ckpt_dir, self.model_name
        )

    def _create_mesh(self, mesh_shape: tuple[int, ...]) -> Mesh:
        """Create a JAX mesh with the given shape."""
        return jax.make_mesh(mesh_shape, axis_names=self.mesh_axis_names)

    def create_actor_mesh(self) -> Mesh:
        return self._create_mesh(self.actor_mesh_shape)

    def create_rollout_mesh(self) -> Mesh:
        return self._create_mesh(self.rollout_mesh_shape)

    def _get_model_config(self) -> dict[str, Any]:
        """Get model config dict for tunix model loading."""
        config = {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "model_source": self.model_source,
            "model_download_path": self.model_download_path,
            "intermediate_ckpt_dir": self._resolved_intermediate_ckpt_dir,
            "rng_seed": self.rng_seed,
            "model_display": False,
        }
        if self.lora_config is not None:
            config["lora_config"] = self.lora_config.to_dict()
        return config

    def make(self) -> tuple[nnx.Module, nnx.Module, PreTrainedTokenizerBase]:
        """
        Create actor model, reference model, and tokenizer.

        Returns:
            Tuple of (actor_model, reference_model, tokenizer)
        """
        from tunix.cli.utils import model as model_lib

        mesh = self.create_actor_mesh()

        # Create reference model config (without LoRA)
        ref_config = self._get_model_config()
        ref_config.pop("lora_config", None)

        tokenizer_config = {"tokenizer_path": self.hf_tokenizer_path}

        with mesh:
            reference_model, _ = model_lib.create_model(
                ref_config, tokenizer_config, mesh
            )

        if self.lora_config is not None:
            actor_model = model_lib.apply_lora_to_model(
                reference_model,
                mesh,
                self.lora_config.to_dict(),
            )
        else:
            actor_model = reference_model

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)

        return actor_model, reference_model, tokenizer


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    opt_type: str = "adamw"
    peak_value: float = 3e-6
    init_value: float = 0.0
    end_value: float = 0.0
    warmup_ratio: float = 0.1
    warmup_steps: int | None = None  # Computed from max_steps if None
    decay_steps: int | None = None  # Set to max_steps if None
    b1: float = 0.9
    b2: float = 0.99
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    schedule_type: str = "warmup_cosine_decay_schedule"

    def make(self, max_steps: int) -> optax.GradientTransformation:
        """Create optimizer with learning rate schedule."""
        warmup_steps = self.warmup_steps
        if warmup_steps is None:
            warmup_steps = int(self.warmup_ratio * max_steps)

        decay_steps = self.decay_steps if self.decay_steps is not None else max_steps

        if self.schedule_type == "warmup_cosine_decay_schedule":
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=self.init_value,
                peak_value=self.peak_value,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=self.end_value,
            )
        elif self.schedule_type == "constant":
            learning_rate = self.peak_value
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        if self.opt_type.lower() == "adamw":
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                b1=self.b1,
                b2=self.b2,
                weight_decay=self.weight_decay,
            )
        elif self.opt_type.lower() == "sgd":
            optimizer = optax.sgd(learning_rate=learning_rate)
        elif self.opt_type.lower() == "adam":
            optimizer = optax.adam(
                learning_rate=learning_rate,
                b1=self.b1,
                b2=self.b2,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.opt_type}")

        if self.max_grad_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optimizer,
            )

        return optimizer


@dataclass
class CheckpointingOptions:
    """Checkpointing configuration."""

    save_interval_steps: int = 500
    max_to_keep: int = 4

@dataclass
class TrainingArgs:
    """Training configuration arguments."""

    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    train_micro_batch_size: int | None = None
    rollout_micro_batch_size: int | None = None
    checkpoint_root_directory: str = "gs://carles-git-good/tunix"
    checkpointing_options: CheckpointingOptions = field(
        default_factory=CheckpointingOptions
    )
    eval_every_n_steps: int = 10
    log_dir: str = "/mnt/carles/logs"
    flush_every_n_steps: int = 20

    def make(self, max_steps: int, exp_name: str, batch_size: int) -> rl_cluster_lib.RLTrainingConfig:
        """
        Create RLTrainingConfig.

        Args:
            max_steps: Maximum training steps
            exp_name: Experiment name for checkpoint directory

        Returns:
            RLTrainingConfig instance
        """
        optimizer = self.optimizer_config.make(max_steps)

        checkpoint_dir = os.path.join(self.checkpoint_root_directory, exp_name)

        return rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            max_steps=max_steps,
            mini_batch_size=batch_size,
            train_micro_batch_size=self.train_micro_batch_size,
            rollout_micro_batch_size=self.rollout_micro_batch_size,
            eval_every_n_steps=self.eval_every_n_steps,
            checkpoint_root_directory=checkpoint_dir,
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=self.checkpointing_options.save_interval_steps,
                max_to_keep=self.checkpointing_options.max_to_keep,
            ),
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                log_dir=os.path.join(self.log_dir, exp_name),
                flush_every_n_steps=self.flush_every_n_steps,
            ),
        )

@dataclass
class RolloutArgs:
    """Rollout configuration arguments."""

    max_tokens_to_generate: int = 768
    max_prompt_length: int = 256
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    eos_tokens: list[int] | None = None

    def make(self) -> base_rollout.RolloutConfig:
        """Create RolloutConfig."""
        return base_rollout.RolloutConfig(
            max_tokens_to_generate=self.max_tokens_to_generate,
            max_prompt_length=self.max_prompt_length,
            kv_cache_size=self.max_prompt_length + self.max_tokens_to_generate + 256,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            eos_tokens=self.eos_tokens,
        )


@dataclass
class GRPOArgs:
    """GRPO algorithm configuration."""

    num_generations: int = 4
    num_iterations: int = 1
    beta: float = 0.0
    epsilon: float = 0.2

    def make(self) -> GRPOConfig:
        """Create GRPOConfig."""
        return GRPOConfig(
            num_generations=self.num_generations,
            num_iterations=self.num_iterations,
            beta=self.beta,
            epsilon=self.epsilon,
        )


@dataclass
class DataArgs:
    """Data configuration with multiple sources."""
    sources: list[DataWithRewardConfig] = field(default_factory=list)

    def make(self, batch_size: int) -> list[tuple[Iterable, list[Callable], str]]:
        """
        Create datasets and reward functions for each source.

        Args:
            batch_size: Batch size for datasets

        Returns:
            List of (dataset, reward_functions, name) tuples
        """
        results = []
        for source in self.sources:
            train_ds = source.make(batch_size)
            reward_fns = source.all_reward
            results.append((train_ds, reward_fns, source.name))

        return results


@dataclass
class Args:
    """Complete training arguments."""

    exp_name: str
    batch_size: int
    model_args: ModelArgs
    training_args: TrainingArgs = field(default_factory=TrainingArgs)
    rollout_args: RolloutArgs = field(default_factory=RolloutArgs)
    grpo_args: GRPOArgs = field(default_factory=GRPOArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    rollout_engine: str = "vanilla"
    offload_to_cpu: bool = False


class Pipeline:
    """GRPO Training Pipeline."""

    def __init__(self, args: Args):
        self.args = args
        self._actor_model = None
        self._reference_model = None
        self._tokenizer = None

    def _create_models(self):
        """Create and cache models."""
        if self._actor_model is None:
            self._actor_model, self._reference_model, self._tokenizer = (
                self.args.model_args.make()
            )
        return self._actor_model, self._reference_model, self._tokenizer

    def _create_cluster_config(
        self,
        training_config: rl_cluster_lib.RLTrainingConfig,
    ) -> rl_cluster_lib.ClusterConfig:
        """Create cluster configuration."""
        actor_mesh = self.args.model_args.create_actor_mesh()
        rollout_mesh = self.args.model_args.create_rollout_mesh()

        reference_mesh = actor_mesh

        role_to_mesh = {
            rl_cluster_lib.Role.ACTOR: actor_mesh,
            rl_cluster_lib.Role.REFERENCE: reference_mesh,
            rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
        }

        return rl_cluster_lib.ClusterConfig(
            role_to_mesh=role_to_mesh,
            rollout_engine=self.args.rollout_engine,
            offload_to_cpu=self.args.offload_to_cpu,
            training_config=training_config,
            rollout_config=self.args.rollout_args.make(),
        )

    def _create_rl_cluster(
        self,
        training_config: rl_cluster_lib.RLTrainingConfig,
    ) -> rl_cluster_lib.RLCluster:
        """Create RL cluster."""
        actor_model, reference_model, tokenizer = self._create_models()
        cluster_config = self._create_cluster_config(training_config)

        ref_model = reference_model if self.args.grpo_args.beta > 0 else None

        return rl_cluster_lib.RLCluster(
            actor=actor_model,
            reference=ref_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

    def run(self):
        """Run the training pipeline across all data sources."""
        data_sources = self.args.data_args.make(self.args.batch_size)

        if not data_sources:
            raise ValueError("No data sources configured")
        dataset_names = []

        for dataset, reward_fns, name in data_sources:
            dataset_names.append(name)

            # Create checkpoint directory with all dataset names
            checkpoint_suffix = "_".join(dataset_names)
            full_exp_name = f"{self.args.exp_name}/{checkpoint_suffix}"
            step = len(dataset)

            print(f"\n{'=' * 60}")
            print(f"Training on dataset: {name}")
            print(f"Training step: {len(dataset)}")
            print(f"Checkpoint directory: {full_exp_name}")
            print(f"{'=' * 60}\n")

            training_config = self.args.training_args.make(step, full_exp_name, self.args.batch_size)
            rl_cluster = self._create_rl_cluster(training_config)
            learner = grpo_learner.GRPOLearner(
                rl_cluster=rl_cluster,
                reward_fns=reward_fns,
                algo_config=self.args.grpo_args.make(),
            )
            mesh = self.args.model_args.create_actor_mesh()
            with mesh:
                learner.train(dataset)
            rl_cluster.close()
            print(f"\nCompleted training on dataset: {name}")

        print(f"\n{'=' * 60}")
        print("Training pipeline completed!")
        print(f"{'=' * 60}\n")

def main():
    from research.general import GeneralDataConfig, VLLMJudgeConfig
    args = Args(
        exp_name="general-rl",
        batch_size=1,
        model_args=ModelArgs(
            model_name="gemma2-2b-it",
            model_id="google/gemma-2/flax/gemma2-2b-it",
            model_source="kaggle",
            hf_tokenizer_path="google/gemma-2-2b-it",
            actor_mesh_shape=(4, 1),
            lora_config=LoraConfig(rank=8, alpha=8.0),
        ),
        training_args=TrainingArgs(
            train_micro_batch_size=1,
            optimizer_config=OptimizerConfig(
                peak_value=3e-6,
                warmup_ratio=0.1,
            ),
        ),
        rollout_args=RolloutArgs(
            max_tokens_to_generate=768,
            max_prompt_length=256,
        ),
        grpo_args=GRPOArgs(
            num_generations=3,
            beta=0.0,
        ),
        data_args=DataArgs(
            sources=[
                GeneralDataConfig(
                    name="general-zero-rl",
                    path="allenai/Dolci-RL-Zero-General-7B",
                    tokenizer_path="google/gemma-2-2b-it",
                    prompt_column="prompt",
                    step=3000,
                    ground_truth_column="ground_truth",
                    judge_model="meta-llama/Llama-3.1-8B-Instruct",
                    vllm_config=VLLMJudgeConfig(
                        model_version="meta-llama/Llama-3.1-8B-Instruct",
                        max_model_len=4096,
                        tensor_parallel_size=4,
                        gpu_memory_utilization=0.5,
                    ),
                ),
            ]
        ),
    )
    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
