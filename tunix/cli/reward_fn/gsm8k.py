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
"""Reward functions for the RLHF pipeline."""

import re
from typing import Callable, List
from absl import logging


# Define the expected signature with type hints
ExpectedSignature = Callable[..., List[float]]

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

def check_answer(prompts, completions, answer, **kwargs):
  responses = completions

  print("======= start =======")
  print("DEBUGPRINT {prompts[0]}:", prompts[0])
  print("DEBUGPRINT {completions[0]}:", completions[0])
  print("DEBUGPRINT {answer[0]}:", answer[0])
  print("======= end =======")

  fallback_regex = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
  extracted_responses = []
  for r in responses:
    match = match_format.search(r)
    if match:
      extracted_responses.append(match.group(1))
    else:
      matches = fallback_regex.findall(r)
      if matches:
        extracted_responses.append(max(matches[-1], key=len))
      else:
        extracted_responses.append(None)

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0.0)
      continue

    matched = False
    # 1. Numeric Check
    try:
      # Remove commas for cleaner number parsing (e.g. "1,000")
      val_guess = float(guess.replace(',', ''))
      val_true = float(true_answer.replace(',', ''))
      if val_guess == val_true:
        matched = True
    except ValueError:
      pass

    # 2. String Check (Fallback)
    if not matched:
      if guess == true_answer or guess.strip() == true_answer.strip():
        matched = True

    scores.append(1.0 if matched else 0.0)
  return scores
