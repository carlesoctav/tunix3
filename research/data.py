"""
Base classes for data configuration with reward functions.

This module provides abstract base classes for configuring datasets
that include reward functions for reinforcement learning training.
"""

import abc
from typing import Any, Callable, Iterable


class DataWithRewardConfig(abc.ABC):
    """Abstract base class for data configurations with reward functions.

    Subclasses must implement:
        - reward_function: Primary reward computation
        - make: Dataset creation

    Optionally can implement:
        - format_reward: Additional format-based reward
        - get_reward_functions: Return list of all reward functions

    Attributes:
        num_examples: Maximum number of examples to use (None = use all)
        name: Name for this data source (used for logging/checkpointing)
    """
    name: str
    step: int | None

    # These should be set by subclasses or during initialization

    @property
    @abc.abstractmethod
    def all_reward():
        ...

    @abc.abstractmethod
    def reward_function(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """Compute rewards for completions.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            **kwargs: Additional arguments (e.g., ground_truth)

        Returns:
            List of reward values (one per completion)
        """
        ...

    @abc.abstractmethod
    def make(self) -> tuple[Iterable, Iterable | None]:
        """Create train and optional eval datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            eval_dataset may be None if not available
        """
        ...
