"""
General reward function using LLM-as-a-judge for evaluating free-form responses.

This module provides a reward function that uses a vLLM-based judge model
to evaluate the quality of model completions against reference answers.
"""

import json
import re
import typing as tp
import warnings
from dataclasses import dataclass, field
from typing import Any

import grain
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from vllm import LLM, SamplingParams

from research.data import DataWithRewardConfig


DEFAULT_SYSTEM = """You are a helpful AI assistant. Answer the user's question accurately and helpfully."""

JUDGE_TEMPLATE = """### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the conversation history leading up to the answer displayed below.
Judge whether the provided answer is good by comparing it to the reference answer.
Notes:
- Besides comparing to the reference answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable and can get a full score.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.
[Conversation History]
{input}
[AI Answer]
{output}
[Reference Gold Answer]
{label}
[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""

# JSON schema for structured output
JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "REASONING": {
            "type": "string",
            "description": "A short explanation of the evaluation"
        },
        "SCORE": {
            "type": "string",
            "description": "A score from 1 to 10"
        }
    },
    "required": ["REASONING", "SCORE"]
}


@dataclass
class VLLMJudgeConfig:
    """Configuration for the vLLM judge model."""
    
    model_version: str
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    temperature: float = 0.0
    max_tokens: int = 512
    trust_remote_code: bool = True


def extract_score_from_response(response: str) -> float:
    """
    Extract score from judge response, expecting JSON format.
    
    Args:
        response: The judge model's response
        
    Returns:
        Normalized score between 0.0 and 1.0
    """
    try:
        # Try to parse as JSON directly
        parsed = json.loads(response.strip())
        score_str = parsed.get("SCORE", "0")
        # Handle cases like "8" or "8/10" or "<8>"
        score_str = str(score_str).replace("<", "").replace(">", "")
        if "/" in score_str:
            score_str = score_str.split("/")[0]
        score = float(score_str)
        return score / 10.0  # Normalize to 0-1 range
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback: try to extract score using regex
    patterns = [
        r'"SCORE"\s*:\s*"?(\d+(?:\.\d+)?)"?',
        r'SCORE[:\s]+(\d+(?:\.\d+)?)',
        r'score[:\s]+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*/\s*10',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if score > 1:
                    score = score / 10.0
                return min(max(score, 0.0), 1.0)
            except ValueError:
                continue
    
    # Default to 0 if no score found
    return 0.0


@dataclass
class GeneralDataConfig(DataWithRewardConfig):
    """Configuration for general-purpose LLM evaluation with judge model.
    
    Uses the allenai/Dolci-RL-Zero-General-7B dataset and evaluates
    responses using a vLLM-based judge model with structured output.
    """
    
    name: str
    path: str = "allenai/Dolci-RL-Zero-General-7B"
    step: int = 1000
    tokenizer_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    prompt_column: str = "prompt"
    ground_truth_column: str = "ground_truth"
    system_prompt: str = DEFAULT_SYSTEM
    max_prompt_len: int = 1024
    num_proc: int = 8
    
    # Judge model configuration
    judge_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_config: VLLMJudgeConfig = field(default_factory=lambda: VLLMJudgeConfig(
        model_version="meta-llama/Llama-3.1-8B-Instruct"
    ))
    
    # Internal state
    _tokenizer: PreTrainedTokenizer = field(init=False, repr=False)
    _judge_llm: LLM = field(init=False, repr=False, default=None)
    _judge_initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # Update vllm_config model version to match judge_model if not explicitly set
        if self.vllm_config.model_version != self.judge_model:
            self.vllm_config = VLLMJudgeConfig(
                model_version=self.judge_model,
                max_model_len=self.vllm_config.max_model_len,
                tensor_parallel_size=self.vllm_config.tensor_parallel_size,
                gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
                temperature=self.vllm_config.temperature,
                max_tokens=self.vllm_config.max_tokens,
                trust_remote_code=self.vllm_config.trust_remote_code,
            )

    def _init_judge(self):
        """Lazily initialize the judge model."""
        if not self._judge_initialized:
            self._judge_llm = LLM(
                model=self.vllm_config.model_version,
                max_model_len=self.vllm_config.max_model_len,
                tensor_parallel_size=self.vllm_config.tensor_parallel_size,
                gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
                trust_remote_code=self.vllm_config.trust_remote_code,
                guided_decoding_backend="outlines",
            )
            self._judge_initialized = True

    def filter_length(self, x):
        return len(self._tokenizer.encode(x["prompts"])) < self.max_prompt_len

    def preprocess(self, x):
        chat = [{"role": "user", "content": self.system_prompt + "\n\n" + x[self.prompt_column]}]
        prompts = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        
        return {"prompts": prompts, "ground_truth": x[self.ground_truth_column]}

    def _build_judge_prompts(
        self, 
        prompts: list[str], 
        completions: list[str], 
        ground_truth: list[str]
    ) -> list[str]:
        """Build prompts for the judge model."""
        judge_prompts = []
        for prompt, completion, gt in zip(prompts, completions, ground_truth):
            judge_prompt = JUDGE_TEMPLATE.format(
                input=prompt,
                output=completion,
                label=gt
            )
            judge_prompts.append(judge_prompt)
        return judge_prompts

    def reward_function(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """
        Compute rewards using LLM-as-a-judge with vLLM.
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            **kwargs: Must include 'ground_truth' list
            
        Returns:
            List of rewards (normalized scores between 0.0 and 1.0)
        """
        ground_truth = kwargs.get("ground_truth", [])
        
        if not ground_truth:
            raise ValueError("ground_truth must be provided for GeneralDataConfig reward function")
        
        # Initialize judge model if needed
        self._init_judge()
        
        # Build judge prompts
        judge_prompts = self._build_judge_prompts(prompts, completions, ground_truth)
        
        # Configure sampling with structured output (JSON schema)
        sampling_params = SamplingParams(
            temperature=self.vllm_config.temperature,
            max_tokens=self.vllm_config.max_tokens,
            guided_decoding={"json": JUDGE_RESPONSE_SCHEMA},
        )
        
        # Generate judge responses
        outputs = self._judge_llm.generate(judge_prompts, sampling_params)
        
        # Extract scores from responses
        rewards = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text
            score = extract_score_from_response(response_text)
            rewards.append(score)
            
            # Log first example for debugging
            if i == 0:
                print("START JUDGE ============================")
                print(f"Prompt: {prompts[0][:200]}...")
                print(f"Ground Truth: {ground_truth[0][:200]}...")
                print(f"Completion: {completions[0][:200]}...")
                print(f"Judge Response: {response_text}")
                print(f"Extracted Score: {score}")
                print("END JUDGE ==============================")
        
        return rewards

    def format_reward_function(
        self, prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        """
        Simple format reward - checks if response is non-empty and reasonable length.
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of format rewards (1.0 for well-formatted, 0.0 otherwise)
        """
        rewards = []
        for completion in completions:
            # Basic format checks
            stripped = completion.strip()
            is_non_empty = len(stripped) > 10
            is_reasonable_length = 10 < len(stripped) < 10000
            has_content = not stripped.isspace()
            
            score = float(is_non_empty and is_reasonable_length and has_content)
            rewards.append(score)
        
        return rewards

    def make(self, batch_size: int) -> tp.Any:
        """
        Create the training dataset.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Batched dataset ready for training
        """
        # Load and preprocess with HuggingFace
        hf_ds = load_dataset(self.path, split="train")
        hf_ds = hf_ds.map(
            self.preprocess, 
            num_proc=self.num_proc, 
            remove_columns=hf_ds.column_names
        )
        hf_ds = hf_ds.filter(self.filter_length, num_proc=self.num_proc)
        
        print(hf_ds)
        print("After filtering, number of valid prompts:", len(hf_ds))
        print("Sanity check:", hf_ds[0])

        if len(hf_ds) // batch_size < self.step:
            warnings.warn(
                "self.step < len(hf_ds) // batch_size, "
                "changing step to len(hf_ds) // batch_size"
            )

        step = min(len(hf_ds) // batch_size, self.step)
        num_examples = step * batch_size

        # Convert to Grain for proper batching
        train_ds = (
            grain.MapDataset.source(hf_ds.select(range(num_examples)))
            .batch(batch_size)
        )

        return train_ds[:step]

    @property
    def all_reward(self):
        """Return all reward functions for this data config."""
        return [self.reward_function, self.format_reward_function]