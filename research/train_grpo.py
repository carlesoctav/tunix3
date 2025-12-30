"""
GRPO training pipeline using draccus for configuration.

This module provides a structured way to configure and run GRPO training
with support for multiple datasets, LoRA, and flexible model/training configurations.
"""

import dataclasses
import draccus
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
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
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from research.on_policy_data import HFSource, OnPolicyData
from tunix.cli.utils import model as cli_model_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.utils import math_rewards

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

    none = "none"
    full = "full"
    minimal = "minimal"


class ModelFamily(str, Enum):
    """Model family enum for dispatching to correct model loader."""

    Gemma2 = "Gemma2"
    Gemma3 = "Gemma3"


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
    model_name: str = "gemma-3-1b-it"
    model_id: str = "google/gemma-3-1b-it"
    model_source: str = "huggingface"
    hf_tokenizer_path: str = "google/gemma-3-1b-it"

    ref_model_name: str | None = None
    ref_model_id: str | None = None
    ref_model_source: str | None = None

    mesh_axis_names: tuple[str, ...] = ("fsdp", "tp")
    actor_mesh_shape: tuple[int, ...] = (4, 1)
    ref_mesh_shape: tuple[int, ...] = (4, 1)
    rollout_mesh_shape: tuple[int, ...] | None = None
    lora_config: LoraConfig | None = None
    remat: RematPolicy = RematPolicy.none
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
                raise NotImplementedError(
                    f"Model family {self.model_family} not supported"
                )

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
    warmup_steps: int | None = None
    decay_steps: int | None = None
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
        self,
        max_steps: int,
        exp_name: str,
        batch_size: int,
        factories: list[Callable],
    ) -> rl_cluster_lib.RLTrainingConfig:
        """
        Create RLTrainingConfig.

        Args:
            max_steps: Maximum training steps
            exp_name: Experiment name for checkpoint directory
            batch_size: Batch size
            factories: Logging backend factories

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
class GRPOArgs:
    """GRPO algorithm configuration."""

    algo_variant: str = "grpo"
    advantage_estimator: str = "grpo"
    policy_loss_fn: str = "grpo"
    loss_agg_mode: str = "sequence-mean-token-mean"
    loss_algo: str = "grpo"
    num_generations: int = 2
    num_iterations: int = 1
    beta: float = 0.04
    epsilon: float = 0.2

    def make(self) -> grpo_learner.GRPOConfig:
        """Create GRPOConfig."""
        return grpo_learner.GRPOConfig(
            algo_variant=self.algo_variant,
            advantage_estimator=self.advantage_estimator,
            policy_loss_fn=self.policy_loss_fn,
            loss_agg_mode=self.loss_agg_mode,
            loss_algo=self.loss_algo,
            num_generations=self.num_generations,
            num_iterations=self.num_iterations,
            beta=self.beta,
            epsilon=self.epsilon,
        )


@dataclass
class DataSourceConfig:
    """Single data source configuration."""

    path: str = ""
    name: str = ""
    prompt_column: str = "prompt"
    ground_truth_column: str = "ground_truth"


@dataclass
class DataArgs:
    """Data configuration for GRPO training."""

    sources: list[DataSourceConfig] = field(default_factory=list)
    tokenizer_path: str = "google/gemma-3-1b-it"
    chat_template_path: str | None = "./template/gemma_think.jinja"
    max_prompt_len: int = 512
    num_proc: int = 8
    step: int = 1000

    def make(self, batch_size: int):
        """Create OnPolicyData and return dataset."""
        # Convert DataSourceConfig to HFSource
        hf_sources = [
            HFSource(
                path=s.path,
                name=s.name,
                prompt_column=s.prompt_column,
                ground_truth_column=s.ground_truth_column,
            )
            for s in self.sources
        ]
        data = OnPolicyData(
            sources=hf_sources,
            tokenizer_path=self.tokenizer_path,
            chat_template_path=self.chat_template_path,
            max_prompt_len=self.max_prompt_len,
            num_proc=self.num_proc,
            step=self.step,
        )
        # Update step from data (may be adjusted based on dataset size)
        self.step = data.step
        return data.make(batch_size)


@dataclass
class RewardConfig:
    """Reward function configuration."""

    use_math_reward: bool = True
    verl_compatible: bool = True


@dataclass
class Args:
    """Complete GRPO training arguments."""

    exp_name: str = "grpo-experiment"
    project: str = "tunix-rl"
    batch_size: int = 4
    model_args: ModelArgs = field(default_factory=ModelArgs)
    training_args: TrainingArgs = field(default_factory=TrainingArgs)
    rollout_args: RolloutArgs = field(default_factory=RolloutArgs)
    grpo_args: GRPOArgs = field(default_factory=GRPOArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
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
        reference_mesh = self.args.model_args.create_ref_mesh()

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
        print(f"Tokenizer EOS token IDs: {tokenizer.eos_token_id}")
        cluster_config = self._create_cluster_config(training_config)

        return rl_cluster_lib.RLCluster(
            actor=actor_model,
            reference=reference_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

    def _create_reward_fns(self) -> list[Callable]:
        """Create reward functions based on configuration."""
        reward_fns = []

        if self.args.reward_config.use_math_reward:
            if self.args.reward_config.verl_compatible:

                def math_reward_fn(prompts, completions, reward_model, **kwargs):
                    del prompts, kwargs
                    ground_truths = reward_model["ground_truth"]
                    return [
                        math_rewards.compute_score(c, gt)
                        for c, gt in zip(completions, ground_truths)
                    ]

                reward_fns.append(math_reward_fn)
            else:
                reward_fns.append(math_rewards.compute_score)

        return reward_fns

    def run(self):
        """Run the training pipeline."""
        print(f"\n{'=' * 60}")
        print(f"GRPO Training: {self.args.exp_name}")
        print(f"{'=' * 60}\n")

        print("Loading datasets...")
        train_ds = self.args.data_args.make(self.args.batch_size)

        if not train_ds:
            raise ValueError("No data sources configured")

        def logging_factory():
            config = asdict(self.args)
            return WandbBackend(self.args.project, self.args.exp_name, config=config)

        step = self.args.data_args.step * self.args.grpo_args.num_iterations

        print(f"Training steps: {step}")
        print(f"Checkpoint directory: {self.args.exp_name}")

        training_config = self.args.training_args.make(
            step, self.args.exp_name, self.args.batch_size, [logging_factory]
        )

        print("Loading models...")
        rl_cluster = self._create_rl_cluster(training_config)

        reward_fns = self._create_reward_fns()
        print(f"Using {len(reward_fns)} reward function(s)")

        learner = grpo_learner.GRPOLearner(
            rl_cluster=rl_cluster,
            reward_fns=reward_fns,
            algo_config=self.args.grpo_args.make(),
        )
        mesh = self.args.model_args.create_actor_mesh()

        try:
            with mesh:
                learner.train(train_ds)
        finally:
            rl_cluster.close()

        print(f"\n{'=' * 60}")
        print("GRPO Training completed!")
        print(f"{'=' * 60}\n")


@draccus.wrap()
def main(cfg: Args):
    """Entry point for GRPO training."""
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
