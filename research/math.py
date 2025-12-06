"""
Math reward function for R1-style completions with <reasoning>...</reasoning><answer>...</answer> format.

This module provides a reward function that correctly extracts answers from R1-style
model outputs before applying math verification.
"""

import re
from absl import logging
import typing as tp
import warnings
from dataclasses import dataclass
from typing import Any

import grain
import sympy
from datasets import load_dataset
from sympy.parsing.latex import parse_latex
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from research.data import DataWithRewardConfig

DEFAULT_SYSTEM = """You are given a problem. Think about the problem and \
provide your reasoning. Place it between <reasoning> and \
</reasoning>. Then, provide the final answer (i.e., just one numerical \
value) between <answer> and </answer>."""

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def extract_answer_tag(text: str) -> str:
    """
    Extract content from <answer>...</answer> tags.

    Args:
        text: The full model completion text

    Returns:
        The content inside <answer> tags, or the original text if no tags found
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{} or \\fbox{} from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from a string."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def get_unnormalized_answer(text: str) -> str:
    """Extract answer from Minerva format: 'Final Answer: The final answer is X.'"""
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.", text
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def fix_fracs(string: str) -> str:
    """Fix fraction formatting like \\frac12 -> \\frac{1}{2}."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def fix_sqrt(string: str) -> str:
    """Fix sqrt formatting like \\sqrta -> \\sqrt{a}."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def fix_a_slash_b(string: str) -> str:
    """Convert a/b to \\frac{a}{b} for simple integer fractions."""
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def remove_right_units(string: str) -> str:
    """Remove units on the right side of an expression."""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def strip_string(string: str) -> str:
    """Normalize string for Hendrycks comparison."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)
    return string


def is_equiv(x1: str, x2: str) -> bool:
    """Check equivalence using sympy LaTeX parsing."""
    try:
        parsed_x1 = parse_latex(x1)
        parsed_x2 = parse_latex(x2)
    except (
        sympy.parsing.latex.errors.LaTeXParsingError,
        sympy.SympifyError,
        TypeError,
    ):
        return False
    except Exception:
        return False

    try:
        diff = parsed_x1 - parsed_x2
    except TypeError:
        return False

    try:
        return sympy.simplify(diff) == 0
    except (ValueError, Exception):
        return False


def hendrycks_is_equiv(str1: str, str2: str) -> bool:
    """Check equivalence using string normalization (Hendrycks MATH style)."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2


def verify_math_answer(completion: str, label: str) -> float:
    """
    from: open-instruct math rl with some modification
    Verify a single math answer against ground truth.

    Args:
        completion: The model completion (may include <think> and <answer> tags)
        label: The ground truth answer

    Returns:
        1.0 if correct, 0.0 if incorrect
    """

    answer_text = extract_answer_tag(completion)

    all_answers = []

    boxed_answer = last_boxed_only_string(answer_text)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)

    minerva_answer = normalize_final_answer(get_unnormalized_answer(answer_text))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)

    if not all_answers:
        dollars = [m.start() for m in re.finditer(r"\$", answer_text)]
        if len(dollars) > 1:
            answer = normalize_final_answer(answer_text[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)

    if not all_answers:
        all_answers.append(normalize_final_answer(answer_text))
        all_answers.append(answer_text)

    for answer in all_answers:
        if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
            return 1.0

    return 0.0


def math_reward(
    prompts: list[str], completions: list[str], ground_truth: list[str], **kwargs: Any
) -> list[float]:
    """
    Compute math rewards for a batch of completions.

    Supports R1-style completions with <think>...</think><answer>...</answer> format.
    The answer is extracted from <answer> tags before verification.

    Args:
        prompts: List of input prompts (unused, kept for signature compatibility)
        completions: List of model completions to evaluate
        ground_truth: List of ground truth answers
        **kwargs: Additional arguments (unused)

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []

    for completion, label in zip(completions, ground_truth):
        score = verify_math_answer(completion, label)
        rewards.append(score)

    return rewards


@dataclass
class MathDataConfig(DataWithRewardConfig):
    name: str
    path: str
    step: int
    tokenizer_path: str
    prompt_column: str
    ground_truth_column: str = "ground_truth"
    system_prompt: str = DEFAULT_SYSTEM
    max_prompt_len: int = 512
    num_proc: int = 8

    def __post_init__(self):
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path
        )

    def filter_length(self, x):
        return len(self._tokenizer.encode(x["prompts"])) < self.max_prompt_len

    def preprocess(self, x):
        chat = [{"role": "user", "content": self.system_prompt + x[self.prompt_column]}]
        prompts = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        return {"prompts": prompts, "ground_truth": x[self.ground_truth_column]}

    def format_reward_function(self, prompts, completions, **kwargs):
        format_pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"

        rewards = []

        for completion in completions:
            correctness_score = bool(re.search(format_pattern, completion, re.DOTALL))
            rewards.append(float(correctness_score))

        return rewards

    def reward_function(self, prompts, completions, **kwargs):
        ground_truth = kwargs.get("ground_truth", [])
        rewards = []

        # Log first example for debugging
        if len(completions) > 0 and len(ground_truth) > 0:
            extracted = extract_answer_tag(completions[0])
            print("START ============================")
            print(f"Prompt: {prompts[0]}...")
            print(f"Ground Truth: {ground_truth[0]}")
            print(f"Completion: {completions[0]}")
            print(f"Extracted Answer: {extracted}")
            print("END ==============================")

        for completion, label in zip(completions, ground_truth):
            correctness = verify_math_answer(completion, label)
            rewards.append(correctness)

        return rewards

    def make(
        self,
        batch_size: int,
    ) -> tp.Any:
        # Load and preprocess with HuggingFace (convenient map/filter)
        hf_ds = load_dataset(self.path, split="train")
        hf_ds = hf_ds.map(self.preprocess, num_proc=self.num_proc, remove_columns=hf_ds.column_names)
        hf_ds = hf_ds.filter(self.filter_length, num_proc=self.num_proc)
        print(hf_ds)
        print("after filtering number of valid prompts are", len(hf_ds))
        print("Sanity check:", hf_ds[0])

        if len(hf_ds) // batch_size < self.step:
            warnings.warn(
                "self.step < len(hf_ds) // batch_size, "
                "changing step to len(hf_ds) // batch_size"
            )

        step = min(len(hf_ds) // batch_size, self.step)
        num_examples = step * batch_size

        # Convert to Grain for proper batching (same as official gsm8k example)
        train_ds = (
            grain.MapDataset.source(hf_ds.select(range(num_examples)))
            .batch(batch_size)
        )

        return train_ds[:step]


    @property
    def all_reward(self):
        return [self.reward_function, self.format_reward_function]
