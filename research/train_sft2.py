import dataclasses
import draccus
import os
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
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
from termcolor import colored



from tunix.cli.utils import model as cli_model_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils

def format_colorized(
    tokens: list[int],
    weights: list[float],
    tokenizer: PreTrainedTokenizerBase,
    draw_newline_arrow: bool = True,
) -> str:
    """Colour-code text according to per-token weights.

    * Green text → weight > 0 (completion/assistant tokens, loss computed)
    * Yellow text → weight = 0 (prompt/padding tokens, no loss)
    """
    if len(tokens) != len(weights):
        raise ValueError("`tokens` and `weights` must be the same length.")

    chunks, current_ids, current_color = [], [], None

    def flush_current_run():
        if not current_ids:
            return
        decoded = tokenizer.decode(current_ids)
        if draw_newline_arrow:
            decoded = decoded.replace("\n", "↵\n")
        chunks.append(colored(decoded, current_color))

    for tok_id, w in zip(tokens, weights, strict=True):
        if w > 0:
            color = "green"
        else:
            color = "yellow"

        if color != current_color and current_ids:
            flush_current_run()
            current_ids = []

        current_ids.append(tok_id)
        current_color = color

    flush_current_run()
    return "".join(chunks)

def visualize_batch(
    batch_tokens: np.ndarray,
    batch_masks: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    num_examples: int = 2,
) -> None:
    """Visualize tokenization with colorized loss mask for a batch.

    Green = assistant tokens (loss computed)
    Yellow = prompt/padding tokens (no loss)
    """
    print("\n" + "=" * 80)
    print("TOKENIZATION VISUALIZATION (Green=loss, Yellow=no loss/padding)")
    print("=" * 80)

    for i in range(min(num_examples, len(batch_tokens))):
        tokens = batch_tokens[i].tolist()
        weights = batch_masks[i].tolist()

        print(f"\n--- Example {i + 1} (total_len={len(tokens)}) ---")
        print(format_colorized(tokens, weights, tokenizer))

    print("\n" + "=" * 80 + "\n")

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
    chat_template_path: str | None 
    path: str 
    name: str | None = None
    split: str = "train"
    eval_split: str | None = "test"
    split_ratio: float = 0.05 
    max_seq_length: int = 2048
    num_train_examples: int | None = None
    num_eval_examples: int | None = None
    prompt_column: str = "messages"
    answer_column: str | None = (
        None  
    )
    batch_size: int = 4
    shuffle: bool = True
    shuffle_seed: int = 42
    epochs: int = 1

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
            split_result = full_ds.train_test_split(
                test_size=self.split_ratio, seed=self.shuffle_seed
            )
            train_ds = split_result["train"]
            eval_ds = split_result["test"]
            if self.num_eval_examples:
                eval_ds = eval_ds.select(
                    range(min(self.num_eval_examples, len(eval_ds)))
                )
            print(f"Split dataset: {len(train_ds)} train, {len(eval_ds)} eval")
        else:
            train_ds = full_ds

        if self.epochs > 1:
            train_ds = train_ds.repeat(self.epochs)
            print(f"Repeating train dataset for {self.epochs} epochs")

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

            # Prepend <reasoning> tag to assistant messages
            for msg in messages:
                if msg.get("role") == "assistant":
                    msg["content"] = "<reasoning>\n" + msg["content"]

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


class Pipeline:
    """SFT Training Pipeline using PEFT trainer."""

    def __init__(self, args: Args):
        self.args = args
        self._model = None
        self._tokenizer = None

    def _create_model_and_tokenizer(self):
        if self._model is None:
            self._model, self._tokenizer = self.args.model_args.make()
        return self._model, self._tokenizer

    def _gen_model_input_fn(self, x: peft_trainer.TrainingInput) -> dict[str, Any]:
        pad_mask = x.input_tokens != 0
        positions = utils.build_positions_from_mask(pad_mask)
        attention_mask = utils.make_causal_attn_mask(pad_mask)

        loss_mask = x.input_mask * pad_mask.astype(x.input_mask.dtype)

        result = {
            "input_tokens": x.input_tokens,
            "input_mask": loss_mask,
            "positions": positions,
            "attention_mask": attention_mask,
        }

        return result

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"SFT Training: {self.args.exp_name}")
        print(f"{'=' * 60}\n")

        print("Loading model and tokenizer...")
        model, tokenizer = self._create_model_and_tokenizer()

        print("Loading datasets...")
        train_ds, eval_ds = self.args.data_args.make()
        first_batch = next(iter(train_ds))
        print(
            f"First batch shape: input_tokens={first_batch.input_tokens.shape}, "
            f"input_mask={first_batch.input_mask.shape}"
        )

        visualize_batch(
            first_batch.input_tokens,
            first_batch.input_mask,
            tokenizer,
            num_examples=2,
        )

        def logging_factory():
            config = asdict(self.args)
            return WandbBackend(self.args.project, self.args.exp_name, config=config)

        training_config, optimizer = self.args.training_args.make(
            self.args.exp_name,
            backend_factories=[logging_factory],
        )

        trainer = peft_trainer.PeftTrainer(
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )

        trainer = trainer.with_gen_model_input_fn(self._gen_model_input_fn)
        print(f"\nStarting training for {self.args.training_args.max_steps} steps...")
        mesh = self.args.model_args.create_mesh()

        with mesh:
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

