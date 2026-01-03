# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GSM8K reward function - simple binary 0/1."""

import re


def extract_gsm8k_answer(text: str, method: str = "strict") -> str | None:
    """Extract answer from GSM8K format (#### <number>).

    Args:
        text: The model completion text.
        method: 'strict' requires #### or \\boxed format, 'flexible' finds last number.

    Returns:
        Extracted answer string or None if not found.
    """
    if method == "strict":
        # Look for #### format (common in ground truth)
        matches = re.findall(r"####\s*(-?[\d,\.]+)", text)
        if matches:
            return matches[-1].replace(",", "")
        # Look for \boxed{...} format (common in model output)
        matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if matches:
            inner_text = matches[-1]
            val_matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", inner_text)
            if val_matches:
                for match in reversed(val_matches):
                    ans = next((m for m in match if m), None)
                    if ans:
                        return ans.replace(",", "").replace("$", "").rstrip(".")
            return inner_text.replace(",", "").replace("$", "")
    else:
        # Flexible extraction - find last number in text
        pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
        matches = re.findall(pattern, text)
        for match in reversed(matches):
            ans = next((m for m in match if m), None)
            if ans and ans not in [".", ",", "$"]:
                return ans.replace(",", "").replace("$", "").rstrip(".")
    return None


def compute_score(
    completion: str,
    ground_truth: str,
    method: str = "strict",
) -> float:
    """Binary reward: 1.0 if correct, 0.0 otherwise.

    Args:
        completion: Model completion text.
        ground_truth: Expected answer.
        method: 'strict' or 'flexible' extraction.

    Returns:
        1.0 if answer matches, 0.0 otherwise.
    """
    answer = extract_gsm8k_answer(completion, method)
    if answer is None:
        return 0.0

    # Extract the correct answer from the ground truth text
    gt_answer = extract_gsm8k_answer(ground_truth, "strict")
    if gt_answer is None:
        # Fallback: assume ground_truth is already the answer value
        gt_answer = ground_truth

    try:
        return 1.0 if float(answer) == float(gt_answer.replace(",", "")) else 0.0
    except ValueError:
        return 1.0 if answer.strip() == gt_answer.strip() else 0.0


def gsm8k_reward_fn(prompts, completions, ground_truth, **kwargs):
    """Verl-compatible reward function for train_grpo.py.

    Args:
        prompts: List of prompts (unused).
        completions: List of model completions.
        reward_model: Dict containing 'ground_truth' key.
        **kwargs: Additional arguments (unused).

    Returns:
        List of float rewards (0.0 or 1.0).
    """
    return [compute_score(c, gt) for c, gt in zip(completions, ground_truth)]
