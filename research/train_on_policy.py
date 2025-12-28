"""
Comprehensive On-Policy training pipeline for research experiments.

This module provides a structured way to configure and run On-Policy training
with support for multiple datasets, LoRA, and flexible model/training configurations.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import jax
import optax
import orbax.checkpoint as ocp
from dotenv import load_dotenv
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import Mesh
from metrax.logging import WandbBackend
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

from research.on_policy import OnPolicyConfig, OnPolicyLearner
from research.on_policy_data import HFSource, OnPolicyData
from tunix.cli.utils import (
    model as cli_model_lib,
)  # Keep for LoRA util if needed or re-implement
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

load_dotenv()

# Load Kaggle credentials from ~/.kaggle/kaggle.json
_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if _kaggle_json.exists():
    with open(_kaggle_json) as f:
        _kaggle_creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = _kaggle_creds["username"]
        os.environ["KAGGLE_KEY"] = _kaggle_creds["key"]


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


class ModelFamily(Enum):
    Gemma2 = auto()
    Gemma3 = auto()


@dataclass
class ModelArgs:
    """Model configuration arguments."""

    model_family: ModelFamily
    model_name: str
    model_id: str
    model_source: str
    hf_tokenizer_path: str

    ref_model_name: str | None = None
    ref_model_id: str | None = None
    ref_model_source: str | None = None

    mesh_axis_names: tuple[str, ...] = ("fsdp", "tp")
    actor_mesh_shape: tuple[int, ...] = (4, 1)
    ref_mesh_shape: tuple[int, ...] = (4, 1)
    rollout_mesh_shape: tuple[int, ...] | None = None  # Defaults to actor_mesh_shape
    lora_config: LoraConfig | None = None
    remat: RematPolicy = RematPolicy.NONE
    rng_seed: int = 42

    def __post_init__(self):
        if self.rollout_mesh_shape is None:
            self.rollout_mesh_shape = self.actor_mesh_shape

    def _create_mesh(self, mesh_shape: tuple[int, ...]) -> Mesh:
        """Create a JAX mesh with the given shape."""
        return jax.make_mesh(mesh_shape, axis_names=self.mesh_axis_names)

    def create_actor_mesh(self) -> Mesh:
        return self._create_mesh(self.actor_mesh_shape)

    def create_ref_mesh(self) -> Mesh:
        return self._create_mesh(self.ref_mesh_shape)

    def remove_binary_tre(self):
        pass

    def create_rollout_mesh(self) -> Mesh:
        if self.rollout_mesh_shape is None:
            return self._create_mesh(self.actor_mesh_shape)
        return self._create_mesh(self.rollout_mesh_shape)

    def make(self) -> tuple[nnx.Module, nnx.Module, PreTrainedTokenizerBase]:
        """
        Create actor model, reference model, and tokenizer.

        Returns:
            Tuple of (actor_model, reference_model, tokenizer)
        """
        actor_mesh = self.create_actor_mesh()
        ref_mesh = self.create_ref_mesh()

        def load_model(model_id: str, model_name: str, mesh: Mesh) -> nnx.Module:
            print(f"Downloading/Loading {model_id} from Hugging Face...")
            local_model_path = snapshot_download(
                repo_id=model_id,
                ignore_patterns=["*.pth"],
            )
            print(f"Model path: {local_model_path}")

            if self.model_family == ModelFamily.Gemma2:
                from tunix.models.gemma import model as gemma_model
                from tunix.models.gemma import params_safetensors as params

                config_method_name = model_name.replace("-", "_")
                config_fn = getattr(gemma_model.ModelConfig, config_method_name)
                config = config_fn()
                model = params.create_model_from_safe_tensors(
                    local_model_path, config, mesh
                )
                return model
            elif self.model_family == ModelFamily.Gemma3:
                from tunix.models.gemma3 import model as gemma_model
                from tunix.models.gemma3 import params_safetensors as params

                config_method_name = model_name.replace("-", "_")
                config_fn = getattr(gemma_model.ModelConfig, config_method_name)
                config = config_fn()
                model = params.create_model_from_safe_tensors(
                    local_model_path, config, mesh
                )
                return model
            else:
                raise NotImplementedError

        # Load Reference Model
        ref_model_id = self.ref_model_id or self.model_id
        ref_model_name = self.ref_model_name or self.model_name
        reference_model = load_model(ref_model_id, ref_model_name, ref_mesh)

        actor_model = load_model(self.model_id, self.model_name, actor_mesh)

        if self.lora_config:
            print("Applying LoRA to actor model...")
            actor_model = cli_model_lib.apply_lora_to_model(
                actor_model,
                actor_mesh,
                self.lora_config.to_dict(),
            )

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

    def make(
        self, max_steps: int, exp_name: str, batch_size: int, factories: Callable
    ) -> rl_cluster_lib.RLTrainingConfig:
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
                backend_factories=factories,
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
class OnPolicyArgs:
    """On-policy algorithm configuration."""

    num_generations: int = 1
    num_iterations: int = 1

    def make(self) -> OnPolicyConfig:
        """Create OnPolicyConfig."""
        return OnPolicyConfig(
            num_generations=self.num_generations,
            num_iterations=self.num_iterations,
        )


@dataclass
class Args:
    """Complete training arguments."""

    project: str
    exp_name: str
    batch_size: int
    model_args: ModelArgs
    training_args: TrainingArgs = field(default_factory=TrainingArgs)
    rollout_args: RolloutArgs = field(default_factory=RolloutArgs)
    on_policy_args: OnPolicyArgs = field(default_factory=OnPolicyArgs)
    data_args: OnPolicyData = field(default_factory=OnPolicyData)
    rollout_engine: str = "vanilla"
    offload_to_cpu: bool = False


class Pipeline:
    """On-Policy Training Pipeline."""

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
        tokenizer: PreTrainedTokenizer = ""
        actor_model, reference_model, tokenizer = self._create_models()
        print("DEBUGPRINT {tokenizer.eos_token_ids}:", tokenizer.eos_token_ids)
        cluster_config = self._create_cluster_config(training_config)

        # We always need the reference model for On-Policy training (for reward computation)
        ref_model = reference_model

        return rl_cluster_lib.RLCluster(
            actor=actor_model,
            reference=ref_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

    def run(self):
        """Run the training pipeline across all data sources."""
        train_ds = self.args.data_args.make(self.args.batch_size)

        if not train_ds:
            raise ValueError("No data sources configured")

        def logging_factories():
            config = asdict(self.args)
            print("DEBUGPRINT {config}:", config)
            return WandbBackend(self.args.project, self.args.exp_name, config=config)

        full_exp_name = f"{self.args.exp_name}"
        step = self.args.data_args.step * self.args.on_policy_args.num_iterations

        print(f"\n{'=' * 60}")
        print(f"Training step: {step}")
        print(f"Checkpoint directory: {full_exp_name}")
        print(f"{'=' * 60}\n")

        training_config = self.args.training_args.make(
            step, full_exp_name, self.args.batch_size, [logging_factories]
        )
        rl_cluster = self._create_rl_cluster(training_config)

        learner = OnPolicyLearner(
            rl_cluster=rl_cluster,
            reward_fns=[],
            algo_config=self.args.on_policy_args.make(),
        )
        mesh = self.args.model_args.create_actor_mesh()

        try:
            with mesh:
                learner.train(train_ds)
        finally:
            rl_cluster.close()

        print(f"\n{'=' * 60}")
        print("Training pipeline completed!")
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--project", type=str)
    script_args = parser.parse_args()

    args = Args(
        exp_name=script_args.exp_name,
        project = script_args.project,
        batch_size=4,
        model_args=ModelArgs(
            model_family=ModelFamily.Gemma3,
            model_name="gemma-3-1b-it",
            model_id="google/gemma-3-1b-it",
            model_source="huggingface",
            ref_model_name="gemma-3-12b-it",
            ref_model_id="google/gemma-3-12b-it",
            ref_model_source="huggingface",
            hf_tokenizer_path="google/gemma-3-1b-it",
            actor_mesh_shape=(4, 1),
            rollout_mesh_shape=(1, 4),
            lora_config=LoraConfig(rank=16, alpha=16),
        ),
        training_args=TrainingArgs(
            train_micro_batch_size=4,
            optimizer_config=OptimizerConfig(
                peak_value=3e-4,
                weight_decay=0.0,
            ),
        ),
        rollout_args=RolloutArgs(
            max_tokens_to_generate=1024,
            max_prompt_length=256,
            eos_tokens=[1, 106],
        ),
        on_policy_args=OnPolicyArgs(
            num_generations=4,
            num_iterations=1,
        ),
        data_args=OnPolicyData(
            sources=[
                HFSource(
                    path="openai/gsm8k",
                    name="main",
                    prompt_column="question",
                    ground_truth_column="answer",
                )
            ],
            tokenizer_path="google/gemma-3-1b-it",
            max_prompt_len=256,
            num_proc=4,
            step=7470 // 4,
        ),
    )
    # Attach project to args dynamically since Args dataclass does not define it.
    setattr(args, "project", script_args.project)
    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
