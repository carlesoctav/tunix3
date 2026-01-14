### Cell 1: Install tunix and dependencies
# !pip install git+https://github.com/google/tunix.git
# !pip install datasets transformers huggingface_hub termcolor metrax optax orbax-checkpoint

### Cell 2: HuggingFace Token Setup (Kaggle)
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

### Cell 3: Imports
import dataclasses
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from datasets import load_dataset
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import Mesh
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from termcolor import colored

from tunix.cli.utils import model as cli_model_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils

### Cell 4: Chat Template Path
CHAT_TEMPLATE_PATH = "./gemma_think_new.jinja"  # Upload this file to Kaggle


### Cell 5: Loss Function
def mean_loss_fn_with_token_count(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits, _ = model(input_tokens, positions, None, attention_mask)
    logits = logits[:, :-1, :]
    target_tokens = input_tokens[:, 1:]
    target_mask = input_mask[:, 1:]
    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])
    one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]
    num_tokens = jnp.sum(target_mask)
    loss = -jnp.sum(jax.nn.log_softmax(logits) * one_hot) / jnp.maximum(num_tokens, 1)
    return loss, {"num_tokens": num_tokens}


### Cell 6: Visualization Helpers
def format_colorized(
    tokens: list[int],
    weights: list[float],
    tokenizer: PreTrainedTokenizerBase,
    draw_newline_arrow: bool = True,
) -> str:
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
        color = "green" if w > 0 else "yellow"
        if color != current_color and current_ids:
            flush_current_run()
            current_ids = []
        current_ids.append(tok_id)
        current_color = color
    flush_current_run()
    return "".join(chunks)


def visualize_batch(
    batch: dict, tokenizer: PreTrainedTokenizerBase, num_examples: int = 2
):
    print("\n" + "=" * 80)
    print("TOKENIZATION VISUALIZATION (Green=loss, Yellow=no loss/padding)")
    print("=" * 80)
    for i in range(min(num_examples, len(batch.input_tokens))):
        tokens = batch.input_tokens[i].tolist()
        weights = batch.input_mask[i].tolist()
        non_pad = sum(1 for t in tokens if t != 0)
        assistant_tokens = sum(weights)
        print(
            f"\n--- Example {i + 1} (total_len={len(tokens)}, non_pad={non_pad}, assistant_tokens={assistant_tokens}) ---"
        )
        print(format_colorized(tokens, weights, tokenizer))
    print("\n" + "=" * 80 + "\n")


### Cell 7: Enums
class RematPolicy(Enum):
    NONE = auto()
    BLOCK = auto()


### Cell 8: Config Dataclasses
@dataclass
class LoraConfig:
    rank: int
    alpha: float
    module_path: str

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "alpha": self.alpha, "module_path": self.module_path}


@dataclass
class ModelArgs:
    model_id: str
    hf_tokenizer_path: str
    mesh_axis_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]
    lora_config: LoraConfig | None
    remat: RematPolicy
    rng_seed: int

    def create_mesh(self) -> Mesh:
        return jax.make_mesh(self.mesh_shape, axis_names=self.mesh_axis_names)

    def make(self) -> tuple[nnx.Module, PreTrainedTokenizerBase]:
        from tunix.models.gemma3 import model as gemma_model
        from tunix.models.gemma3 import params_safetensors as params

        mesh = self.create_mesh()

        print(f"Downloading/Loading {self.model_id} from Hugging Face...")
        local_model_path = snapshot_download(
            repo_id=self.model_id, ignore_patterns=["*.pth"]
        )
        print(f"Model path: {local_model_path}")

        config = gemma_model.ModelConfig.gemma3_1b_it()

        # Note: BLOCK remat is incompatible with LoRA (nnx.remat bound method limitation)
        if self.remat == RematPolicy.BLOCK and self.lora_config is None:
            config = dataclasses.replace(
                config, remat_config=gemma_model.RematConfig.BLOCK
            )
            print("Applying BLOCK remat policy")
        elif self.remat == RematPolicy.BLOCK and self.lora_config is not None:
            print("Warning: BLOCK remat disabled - incompatible with LoRA")

        model = params.create_model_from_safe_tensors(local_model_path, config, mesh)

        if self.lora_config:
            print("Applying LoRA...")
            model = cli_model_lib.apply_lora_to_model(
                model, mesh, self.lora_config.to_dict()
            )
            print("LoRA applied")

        tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)
        return model, tokenizer


@dataclass
class OptimizerConfig:
    opt_type: str
    peak_value: float
    init_value: float
    end_value: float
    warmup_ratio: float
    warmup_steps: int | None
    decay_steps: int | None
    b1: float
    b2: float
    weight_decay: float
    max_grad_norm: float
    schedule_type: str

    def make(self, max_steps: int) -> optax.GradientTransformation:
        warmup_steps = self.warmup_steps or int(self.warmup_ratio * max_steps)
        decay_steps = self.decay_steps or max_steps

        if self.schedule_type == "warmup_cosine_decay_schedule":
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=self.init_value,
                peak_value=self.peak_value,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=self.end_value,
            )
        else:
            learning_rate = self.peak_value

        if self.opt_type.lower() == "adam":
            optimizer = optax.adam(learning_rate=learning_rate, b1=self.b1, b2=self.b2)
        elif self.opt_type.lower() == "adamw":
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                b1=self.b1,
                b2=self.b2,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optax.sgd(learning_rate=learning_rate)

        if self.max_grad_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm), optimizer
            )
        return optimizer


@dataclass
class CheckpointingOptions:
    save_interval_steps: int
    max_to_keep: int


@dataclass
class TrainingArgs:
    optimizer_config: OptimizerConfig
    max_steps: int
    eval_every_n_steps: int
    gradient_accumulation_steps: int | None
    checkpoint_root_directory: str
    checkpointing_options: CheckpointingOptions
    log_dir: str
    flush_every_n_steps: int

    def make(
        self, exp_name: str
    ) -> tuple[peft_trainer.TrainingConfig, optax.GradientTransformation]:
        optimizer = self.optimizer_config.make(self.max_steps)
        checkpoint_dir = os.path.join(self.checkpoint_root_directory, exp_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

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
            ),
        )
        return training_config, optimizer


@dataclass
class DataArgs:
    tokenizer_path: str
    chat_template_path: str | None
    path: str
    name: str | None
    split: str
    eval_split: str | None
    split_ratio: float
    max_seq_length: int
    num_train_examples: int | None
    num_eval_examples: int | None
    prompt_column: str
    answer_column: str | None
    batch_size: int
    shuffle: bool
    shuffle_seed: int
    epochs: int

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.chat_template_path and os.path.exists(self.chat_template_path):
            with open(self.chat_template_path) as f:
                self.tokenizer.chat_template = f.read()

    def make(self) -> tuple[Iterable, Iterable | None]:
        dataset = load_dataset(self.path, split=self.split)
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.shuffle_seed)
        if self.num_train_examples:
            dataset = dataset.select(range(min(self.num_train_examples, len(dataset))))
        return self._create_data_iterator(dataset), None

    def _create_data_iterator(self, dataset) -> Iterable:
        def process_example(example):
            prompt = example[self.prompt_column]
            answer = example[self.answer_column]
            messages = [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(answer)},
            ]
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
            return encoded["input_ids"][0], np.array(
                encoded["assistant_masks"][0], dtype=np.int32
            )

        def batch_generator():
            batch_tokens, batch_masks = [], []
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
                        batch_tokens, batch_masks = [], []
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            if batch_tokens:
                yield peft_trainer.TrainingInput(
                    input_tokens=np.stack(batch_tokens),
                    input_mask=np.stack(batch_masks),
                )

        return batch_generator()


@dataclass
class Args:
    exp_name: str
    project: str
    model_args: ModelArgs
    training_args: TrainingArgs
    data_args: DataArgs


### Cell 9: Pipeline
class Pipeline:
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
        return {
            "input_tokens": x.input_tokens,
            "input_mask": loss_mask,
            "positions": positions,
            "attention_mask": attention_mask,
        }

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"SFT Training: {self.args.exp_name}")
        print(f"{'=' * 60}\n")

        print("Loading model and tokenizer...")
        model, tokenizer = self._create_model_and_tokenizer()

        print("Loading datasets...")
        train_ds, eval_ds = self.args.data_args.make()

        first_batch = next(train_ds)
        print(
            f"First batch shape: input_tokens={first_batch.input_tokens.shape}, input_mask={first_batch.input_mask.shape}"
        )
        visualize_batch(first_batch, self.args.data_args.tokenizer, num_examples=2)

        training_config, optimizer = self.args.training_args.make(self.args.exp_name)

        trainer = peft_trainer.PeftTrainer(
            model=model,
            optimizer=optimizer,
            training_config=training_config,
        )
        trainer = trainer.with_gen_model_input_fn(self._gen_model_input_fn)
        trainer = trainer.with_loss_fn(mean_loss_fn_with_token_count, has_aux=True)

        print(f"\nStarting training for {self.args.training_args.max_steps} steps...")
        mesh = self.args.model_args.create_mesh()

        with mesh:
            trainer.train(train_ds, None)

        print(f"\n{'=' * 60}")
        print("SFT Training completed!")
        print(f"{'=' * 60}\n")


### Cell 10: Run Training
args = Args(
    exp_name="27b-generated-Dolci-Instruct-SFT-No-Tools-rank-256-lr-5e7",
    project="tunix-finale",
    model_args=ModelArgs(
        model_id="google/gemma-3-1b-it",
        hf_tokenizer_path="google/gemma-3-1b-it",
        mesh_axis_names=("fsdp", "tp"),
        mesh_shape=(4, 1),
        remat=RematPolicy.BLOCK,
        rng_seed=42,
        lora_config=LoraConfig(
            rank=256,
            alpha=512,
            module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
        ),
    ),
    training_args=TrainingArgs(
        max_steps=21218,
        eval_every_n_steps=3000,
        gradient_accumulation_steps=4,
        checkpoint_root_directory="/kaggle/working/checkpoints",
        log_dir="/kaggle/working/logs",
        flush_every_n_steps=20,
        checkpointing_options=CheckpointingOptions(
            save_interval_steps=5000,
            max_to_keep=4,
        ),
        optimizer_config=OptimizerConfig(
            opt_type="adam",
            peak_value=5e-7,
            init_value=0.0,
            end_value=0.0,
            warmup_ratio=0.03,
            warmup_steps=None,
            decay_steps=None,
            b1=0.9,
            b2=0.99,
            weight_decay=0.0,
            max_grad_norm=1.0,
            schedule_type="warmup_cosine_decay_schedule",
        ),
    ),
    data_args=DataArgs(
        tokenizer_path="google/gemma-3-1b-it",
        chat_template_path=CHAT_TEMPLATE_PATH,
        path="carlesoctav/filtered-2048-27b-generated-Dolci-Instruct-SFT-No-Tools",
        name=None,
        split="train",
        eval_split=None,
        split_ratio=0.0,
        max_seq_length=2048,
        num_train_examples=None,
        num_eval_examples=None,
        prompt_column="prompt",
        answer_column="generated",
        batch_size=8,
        shuffle=True,
        shuffle_seed=42,
        epochs=1,
    ),
)

pipeline = Pipeline(args)
pipeline.run()
