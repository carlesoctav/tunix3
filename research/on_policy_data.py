import warnings
from dataclasses import dataclass
from functools import partial

import grain
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

def preprocess(tokenizer, prompt_column, ground_truth_column, x):
    chat = [{"role": "user", "content": x[prompt_column]}]
    prompts = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    return {"prompts": prompts, "ground_truth": x[ground_truth_column]}


def filter_length(tokenizer: PreTrainedTokenizer, length, prompt_column, x):
    return len(tokenizer.encode(x[prompt_column])) < length


@dataclass
class HFSource:
    name: str
    path: str
    prompt_column: str = "prompt"
    ground_truth_column: str = "ground_truth"


@dataclass
class OnPolicyData:
    sources: list[HFSource]
    step: int
    tokenizer_path: str
    chat_template_path: str | None = "./template/gemma_think.jinja"
    max_prompt_len: int = 512
    num_proc: int = 8

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.chat_template_path:
            self.tokenizer.chat_template = open(self.chat_template_path).read()

    def make(self, batch_size: int):
        hf_ds = []
        for source in self.sources:
            ds = load_dataset(source.path, source.name, split="train")
            ds = ds.map(
                partial(
                    preprocess,
                    self.tokenizer,
                    source.prompt_column,
                    source.ground_truth_column,
                ),
                remove_columns=ds.column_names,
            )
            print(f"sanity_check {source.name}:", ds[0])
            hf_ds.append(ds)

        all_ds = concatenate_datasets(hf_ds)
        all_ds = all_ds.filter(
            partial(filter_length, self.tokenizer, self.max_prompt_len, "prompts")
        )
        print(f"after filtering we got {len(all_ds)} samples")
        if len(all_ds) // batch_size < self.step:
            warnings.warn(
                "self.step < len(hf_ds) // batch_size, "
                "changing step to len(hf_ds) // batch_size"
            )

        step = min(self.step, len(all_ds) // batch_size)
        self.step = step
        num_examples = batch_size * step
        grain_ds = grain.MapDataset.source(all_ds.select(range(num_examples))).batch(
            batch_size
        )
        return grain_ds
