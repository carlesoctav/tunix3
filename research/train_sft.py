"""
SFT training pipeline using PEFT trainer.

This module provides a structured way to configure and run SFT (Supervised Fine-Tuning)
with support for LoRA, HuggingFace datasets, and flexible model/training configurations.
"""

import dataclasses
import draccus
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterable
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from datasets import load_dataset
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import Mesh
from metrax.logging import WandbBackend
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import jax.numpy as jnp
from jax.typing import ArrayLike

from tunix.cli.utils import model as cli_model_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import profiler
from tunix.sft import utils


def stm_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    ppl_threshold: float = 2.5,
) -> ArrayLike:
    """Selective Token Masking loss function.

    Masks tokens with perplexity > threshold from the loss computation to reduce
    catastrophic forgetting during fine-tuning.

    Based on: "Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token
    Learning" (arXiv:2501.14315).

    The paper found that threshold=2.5 (which masks ~20-24% of tokens) works best.

    Args:
        model: The model to compute loss for.
        input_tokens: Input token IDs [batch, seq_len].
        input_mask: Mask for valid tokens [batch, seq_len] (assistant tokens only).
        positions: Position indices [batch, seq_len].
        attention_mask: Attention mask for causal masking.
        ppl_threshold: Perplexity threshold. Tokens with ppl > threshold are masked.
                       Default 2.5 (from paper).

    Returns:
        The masked cross-entropy loss.
    """
    logits, _ = model(input_tokens, positions, None, attention_mask)

    # Exclude the last step as it does not appear in the targets.
    # logits[:, t, :] predicts input_tokens[:, t+1]
    logits = logits[:, :-1, :]
    target_tokens = input_tokens[:, 1:]
    target_mask = input_mask[:, 1:]  # Shifted mask for targets

    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Get log probability of the target tokens
    # Shape: [batch, seq_len-1]
    target_log_probs = jnp.take_along_axis(
        log_probs, target_tokens[..., None], axis=-1
    ).squeeze(-1)

    # Compute token-level loss (negative log likelihood)
    # loss = -log_prob, and ppl = exp(loss)
    # So: ppl > threshold <==> loss > log(threshold)
    token_losses = -target_log_probs
    loss_threshold = jnp.log(ppl_threshold)

    # STM: KEEP tokens with LOW perplexity (ppl <= threshold)
    # Tokens with loss <= loss_threshold have ppl <= threshold -> KEEP (1)
    # Tokens with loss > loss_threshold have ppl > threshold -> MASK OUT (0)
    stm_mask = (token_losses <= loss_threshold).astype(target_mask.dtype)

    # Combine masks:
    # 1. target_mask: assistant tokens only (from apply_chat_template)
    # 2. stm_mask: tokens with ppl <= threshold
    # Final mask = assistant tokens AND low-ppl tokens
    final_mask = target_mask * stm_mask

    # Convert the target labels to one-hot encoded vectors.
    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

    # Don't update on unwanted tokens.
    one_hot = one_hot * final_mask.astype(one_hot.dtype)[..., None]

    # Normalize by number of valid tokens (assistant * non-pad * ppl <= threshold)
    norm_factor = 1 / (jnp.sum(final_mask) + 1e-8)

    # Return the negative log likelihood (NLL) loss.
    return -jnp.sum(log_probs * one_hot) * norm_factor


# Load Kaggle credentials from ~/.kaggle/kaggle.json
_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if _kaggle_json.exists():
    with open(_kaggle_json) as f:
        _kaggle_creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = _kaggle_creds["username"]
        os.environ["KAGGLE_KEY"] = _kaggle_creds["key"]


@dataclass
class STMConfig:
    """Selective Token Masking configuration.

    Based on: "Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token
    Learning" (arXiv:2501.14315).

    STM masks tokens with perplexity > threshold during training to reduce
    catastrophic forgetting. The paper found threshold=2.5 (masking ~20-24%
    of tokens) works best.
    """

    enabled: bool = True
    ppl_threshold: float = 2.5  # Tokens with ppl > threshold are masked


class ModelFamily(Enum):
    """Model family enum for dispatching to correct model loader."""

    Gemma2 = auto()
    Gemma3 = auto()


class RematPolicy(Enum):
    """Remat policy for gradient checkpointing."""

    NONE = auto()  # No remat, all activations stored in HBM
    BLOCK = auto()  # Remat the entire attention block (saves memory)


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

    model_family: ModelFamily = ModelFamily.Gemma2
    model_name: str = "gemma2-2b-it"
    model_id: str = "google/gemma-2-2b-it"
    hf_tokenizer_path: str = "google/gemma-2-2b-it"
    mesh_axis_names: tuple[str, ...] = ("fsdp", "tp")
    mesh_shape: tuple[int, ...] = (4, 1)
    lora_config: LoraConfig | None = None
    remat: RematPolicy = RematPolicy.NONE
    rng_seed: int = 42

    def _create_mesh(self, mesh_shape: tuple[int, ...]) -> Mesh:
        """Create a JAX mesh with the given shape."""
        return jax.make_mesh(mesh_shape, axis_names=self.mesh_axis_names)

    def create_mesh(self) -> Mesh:
        return self._create_mesh(self.mesh_shape)

    def make(self) -> tuple[nnx.Module, PreTrainedTokenizerBase]:
        """
        Create model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        mesh = self.create_mesh()

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
                # Apply remat config
                if self.remat == RematPolicy.BLOCK:
                    config = dataclasses.replace(
                        config, remat_config=gemma_model.RematConfig.BLOCK
                    )
                    print("Applying BLOCK remat policy for memory optimization")
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
                # Apply remat config
                if self.remat == RematPolicy.BLOCK:
                    config = dataclasses.replace(
                        config, remat_config=gemma_model.RematConfig.BLOCK
                    )
                    print("Applying BLOCK remat policy for memory optimization")
                model = params.create_model_from_safe_tensors(
                    local_model_path, config, mesh
                )
                return model
            else:
                raise NotImplementedError(
                    f"Model family {self.model_family} not supported"
                )

        model = load_model(self.model_id, self.model_name, mesh)

        if self.lora_config:
            print(f"DEBUGPRINT[41]: train_sft.py:254: mesh={mesh}")
            print("Applying LoRA to model...")
            model = cli_model_lib.apply_lora_to_model(
                model,
                mesh,
                self.lora_config.to_dict(),
            )
            # may be skipped if mesh comparison fails
            print("LoRA applied and resharded")

        tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)
        return model, tokenizer


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    opt_type: str = "adamw"
    peak_value: float = 1e-4
    init_value: float = 0.0
    end_value: float = 0.0
    warmup_ratio: float = 0.1
    warmup_steps: int | None = None
    decay_steps: int | None = None
    b1: float = 0.9
    b2: float = 0.99
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
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
    max_steps: int = 1000
    eval_every_n_steps: int = 1000
    gradient_accumulation_steps: int | None = None
    checkpoint_root_directory: str = "gs://carles-git-good/tunix-sft"
    checkpointing_options: CheckpointingOptions = field(
        default_factory=CheckpointingOptions
    )
    log_dir: str = "/mnt/carles/logs"
    flush_every_n_steps: int = 20

    def make(
        self,
        exp_name: str,
        backend_factories: list[Callable] | None = None,
    ) -> tuple[peft_trainer.TrainingConfig, optax.GradientTransformation]:
        """
        Create TrainingConfig and optimizer.

        Args:
            exp_name: Experiment name for checkpoint directory
            backend_factories: Optional list of logging backend factories (e.g., WandbBackend)

        Returns:
            Tuple of (TrainingConfig, optimizer)
        """
        optimizer = self.optimizer_config.make(self.max_steps)
        checkpoint_dir = os.path.join(self.checkpoint_root_directory, exp_name)

        training_config = peft_trainer.TrainingConfig(
            max_steps=self.max_steps,
            eval_every_n_steps=self.eval_every_n_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            checkpoint_root_directory=checkpoint_dir,
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=self.checkpointing_options.save_interval_steps,
                max_to_keep=self.checkpointing_options.max_to_keep,
            ),
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                log_dir=os.path.join(self.log_dir, exp_name),
                flush_every_n_steps=self.flush_every_n_steps,
                backend_factories=backend_factories,
            ),
        )

        return training_config, optimizer


@dataclass
class DataArgs:
    """Data configuration for HuggingFace datasets."""

    tokenizer_path: str
    chat_template_path: str | None = "./gemma_think_new.jinja"
    path: str = "allenai/tulu-3-sft-mixture"
    name: str | None = None
    split: str = "train"
    eval_split: str | None = "test"
    split_ratio: float = 0.05  # Ratio of data to use for eval if splitting from train
    max_seq_length: int = 2048
    num_train_examples: int | None = None
    num_eval_examples: int | None = None
    prompt_column: str = "messages"
    answer_column: str | None = (
        None  # If provided with prompt_column, creates [user, assistant] messages
    )
    batch_size: int = 4
    shuffle: bool = True
    shuffle_seed: int = 42
    epochs: int = 1  # Number of epochs to train

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.chat_template_path:
            self.tokenizer.chat_template = open(self.chat_template_path).read()

    def make(self) -> tuple[Iterable, Iterable | None]:
        load_kwargs = {"path": self.path, "split": self.split}
        if self.name:
            load_kwargs["name"] = self.name

        full_ds = load_dataset(**load_kwargs)

        if self.shuffle:
            full_ds = full_ds.shuffle(seed=self.shuffle_seed)

        if self.num_train_examples:
            full_ds = full_ds.select(range(min(self.num_train_examples, len(full_ds))))

        eval_ds = None
        if self.eval_split:
            # Load separate eval split
            try:
                eval_load_kwargs = {"path": self.path, "split": self.eval_split}
                if self.name:
                    eval_load_kwargs["name"] = self.name
                eval_ds = load_dataset(**eval_load_kwargs)
                if self.num_eval_examples:
                    eval_ds = eval_ds.select(
                        range(min(self.num_eval_examples, len(eval_ds)))
                    )
            except Exception as e:
                print(f"Could not load eval split '{self.eval_split}': {e}")
                eval_ds = None
            train_ds = full_ds
        elif self.split_ratio > 0:
            # Split train data into train/eval using split_ratio
            split_result = full_ds.train_test_split(
                test_size=self.split_ratio, seed=self.shuffle_seed
            )
            train_ds = split_result["train"]
            eval_ds = split_result["test"]
            # Limit eval examples if specified
            if self.num_eval_examples:
                eval_ds = eval_ds.select(
                    range(min(self.num_eval_examples, len(eval_ds)))
                )
            print(f"Split dataset: {len(train_ds)} train, {len(eval_ds)} eval")
        else:
            train_ds = full_ds

        # Repeat train dataset for multiple epochs
        if self.epochs > 1:
            train_ds = train_ds.repeat(self.epochs)
            print(f"Repeating train dataset for {self.epochs} epochs")

        # Create data iterators
        train_iter = self._create_data_iterator(train_ds)
        eval_iter = None
        if eval_ds is not None:
            eval_iter = self._create_data_iterator(eval_ds, is_eval=True)

        return train_iter, eval_iter

    def _create_data_iterator(
        self,
        dataset,
        is_eval: bool = False,
    ) -> Iterable:
        """Create a data iterator that yields TrainingInput batches."""

        def process_example(example):
            """Process a single example into tokens."""
            if self.answer_column is not None:
                prompt = example[self.prompt_column]
                answer = example[self.answer_column]
                messages = [
                    {"role": "user", "content": str(prompt)},
                    {"role": "assistant", "content": str(answer)},
                ]
            else:
                # Use prompt_column as-is (expects messages format)
                messages = example[self.prompt_column]
                if not isinstance(messages, list):
                    messages = [{"role": "user", "content": str(messages)}]
                elif len(messages) > 0 and not isinstance(messages[0], dict):
                    messages = [{"role": "user", "content": " ".join(messages)}]

            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_tensors="np",
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_dict=True,
            )
            input_tokens = encoded["input_ids"][0]

            # assistant_mask marks which tokens are from assistant (1) vs user/system (0)
            # This is used as input_mask so loss is only computed on assistant tokens
            assistant_mask = np.array(encoded["assistant_masks"][0], dtype=np.int32)

            return input_tokens, assistant_mask

        def batch_generator():
            """Generate batches of TrainingInput."""
            batch_tokens = []
            batch_masks = []

            for example in dataset:
                try:
                    input_tokens, assistant_mask = process_example(example)
                    batch_tokens.append(input_tokens)
                    batch_masks.append(assistant_mask)

                    if len(batch_tokens) >= self.batch_size:
                        yield peft_trainer.TrainingInput(
                            input_tokens=np.stack(batch_tokens),
                            input_mask=np.stack(batch_masks),
                        )
                        batch_tokens = []
                        batch_masks = []
                except Exception as e:
                    print(f"Error processing example: {e}")
                    continue

            if batch_tokens:
                yield peft_trainer.TrainingInput(
                    input_tokens=np.stack(batch_tokens),
                    input_mask=np.stack(batch_masks),
                )

        return batch_generator()


@dataclass
class Args:
    """Complete SFT training arguments."""

    exp_name: str = "sft-experiment"
    project: str = "tunix-sft"
    model_args: ModelArgs = field(default_factory=ModelArgs)
    training_args: TrainingArgs = field(default_factory=TrainingArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    stm_config: STMConfig = field(default_factory=STMConfig)


class Pipeline:
    """SFT Training Pipeline using PEFT trainer."""

    def __init__(self, args: Args):
        self.args = args
        self._model = None
        self._tokenizer = None

    def _create_model_and_tokenizer(self):
        """Create and cache model and tokenizer."""
        if self._model is None:
            self._model, self._tokenizer = self.args.model_args.make()
        return self._model, self._tokenizer

    def _gen_model_input_fn(self, x: peft_trainer.TrainingInput) -> dict[str, Any]:
        """Generate model input from training input."""
        # x.input_mask is the assistant_mask from apply_chat_template
        # It marks assistant tokens (1) and user/system/pad tokens (0)
        # We use pad_mask for positions and causal attention
        pad_mask = x.input_tokens != 0
        positions = utils.build_positions_from_mask(pad_mask)
        attention_mask = utils.make_causal_attn_mask(pad_mask)

        # input_mask for loss: use assistant_mask (only train on assistant tokens)
        # Also mask out padding tokens
        loss_mask = x.input_mask * pad_mask.astype(x.input_mask.dtype)

        result = {
            "input_tokens": x.input_tokens,
            "input_mask": loss_mask,
            "positions": positions,
            "attention_mask": attention_mask,
        }

        # Add ppl_threshold for STM loss if enabled
        if self.args.stm_config.enabled:
            result["ppl_threshold"] = self.args.stm_config.ppl_threshold

        return result

    def run(self):
        """Run the SFT training pipeline."""
        print(f"\n{'=' * 60}")
        print(f"SFT Training: {self.args.exp_name}")
        print(f"{'=' * 60}\n")

        # Create model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = self._create_model_and_tokenizer()

        # Create datasets
        print("Loading datasets...")
        train_ds, eval_ds = self.args.data_args.make()

        # Debug: Check first batch shape
        first_batch = next(iter(train_ds))
        print(
            f"First batch shape: input_tokens={first_batch.input_tokens.shape}, input_mask={first_batch.input_mask.shape}"
        )
        # Re-create iterator since we consumed one batch
        train_ds, eval_ds = self.args.data_args.make()

        # Create logging backend factory
        def logging_factory():
            config = asdict(self.args)
            return WandbBackend(self.args.project, self.args.exp_name, config=config)

        # Create training config and optimizer
        training_config, optimizer = self.args.training_args.make(
            self.args.exp_name,
            backend_factories=[logging_factory],
        )

        # Create trainer
        trainer = peft_trainer.PeftTrainer(
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )

        # Set custom model input function
        trainer = trainer.with_gen_model_input_fn(self._gen_model_input_fn)

        # Set STM loss function if enabled (used for both train and eval)
        if self.args.stm_config.enabled:
            print(
                f"Using STM loss with ppl_threshold={self.args.stm_config.ppl_threshold}"
            )
            trainer = trainer.with_loss_fn(stm_loss_fn)

        # Run training
        print(f"\nStarting training for {self.args.training_args.max_steps} steps...")
        mesh = self.args.model_args.create_mesh()

        with mesh:
            # Pass None for eval_ds to skip initial eval, eval runs every eval_every_n_steps
            trainer.train(train_ds, None)

        print(f"\n{'=' * 60}")
        print("SFT Training completed!")
        print(f"{'=' * 60}\n")


@draccus.wrap()
def main(cfg: Args):
    """Entry point for SFT training."""
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
