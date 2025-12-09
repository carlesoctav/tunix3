"""On-policy learner implementation."""

from __future__ import annotations

from absl import logging
import dataclasses
from typing import Iterable, List, Sequence, TypeVar, cast, Any, Mapping

import flax
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@flax.struct.dataclass(frozen=True)  # type: ignore
class TrainExample(common.TrainExample):
    prompt_ids: jax.Array
    prompt_mask: jax.Array
    completion_ids: jax.Array
    completion_mask: jax.Array
    advantages: jax.Array
    ref_per_token_logps: jax.Array | None
    old_per_token_logps: jax.Array | None


@dataclasses.dataclass(slots=True, kw_only=True)
class OnPolicyConfig(algo_config_lib.AlgorithmConfig):
    """Configuration for On-Policy algorithms."""

    algo_variant: str = "on_policy"
    policy_loss_fn: str = "on_policy_loss"
    num_generations: int = 1
    num_iterations: int = 1

    def __post_init__(self):
        return


TOnPolicyConfig = TypeVar("TOnPolicyConfig", bound=OnPolicyConfig)


def on_policy_reward(log_prob_student: jax.Array, log_prob_ref: jax.Array) -> jax.Array:
    """Computes the on-policy reward.

    Args:
      log_prob_student: Log probabilities from the student model.
      log_prob_ref: Log probabilities from the reference model.

    Returns:
      The computed reward (log_prob_ref - log_prob_student).
    """
    return log_prob_ref - log_prob_student  # type: ignore


def compute_advantages(rewards: jax.Array) -> jax.Array:
    """Computes advantages from rewards.

    Args:
      rewards: The computed rewards.

    Returns:
      The advantages (same as rewards in this case).
    """
    return rewards


@function_registry.register_policy_loss_fn("on_policy_loss")
def on_policy_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
):
    """On-policy loss function."""
    completion_ids, completion_mask = (
        train_example.completion_ids,
        train_example.completion_mask,
    )

    per_token_logps = common.compute_per_token_logps(
        model,
        prompt_tokens=train_example.prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        stop_gradient=False,
        return_logits=False,
    )

    per_token_logps = cast(jax.Array, per_token_logps)

    if train_example.old_per_token_logps is None:
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
    else:
        old_per_token_logps = train_example.old_per_token_logps

    log_ratio = per_token_logps - old_per_token_logps
    ratio = jnp.exp(log_ratio)

    advantages = train_example.advantages

    # Simple importance sampling without clipping
    per_token_loss = -(ratio * advantages)

    loss = common.aggregate_loss(
        per_token_loss, completion_mask, "sequence-mean-token-mean"
    )

    return loss, {}


class OnPolicyLearner(rl_learner.RLLearner[TOnPolicyConfig]):
    """On-Policy Learner."""

    def __init__(
        self,
        rl_cluster: rl_cluster_lib.RLCluster,
        algo_config: TOnPolicyConfig,
        reward_fns: RewardFn | List[RewardFn],
        metric_fns: Sequence[MetricFn] | None = None,
        data_shuffle_seed: int | None = None,
    ):
        super().__init__(
            rl_cluster=rl_cluster,
            algo_config=algo_config,
            reward_fns=reward_fns,
            metric_fns=metric_fns,
            data_shuffle_seed=data_shuffle_seed,
        )

        policy_loss_fn = function_registry.get_policy_loss_fn(
            self.algo_config.policy_loss_fn
        )

        loss_fn = lambda model, train_example, algo_config: policy_loss_fn(
            model,
            train_example,
            algo_config=self.algo_config,
            pad_id=self.rl_cluster.rollout.pad_id(),
            eos_id=self.rl_cluster.rollout.eos_id(),
        )

        self.rl_cluster.actor_trainer.with_loss_fn(
            loss_fn,
            has_aux=True,
        )

        def gen_model_input(x: Any) -> Any:
            return {
                "train_example": x,
                "algo_config": self.algo_config,
            }

        # Cast to Any to bypass strict type checking
        self.rl_cluster.actor_trainer.with_gen_model_input_fn(
            cast(Any, gen_model_input)
        )  # type: ignore

    def _generate_and_compute_advantage(
        self,
        training_input: TrainingInputT,
        mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
    ) -> TrainExample:
        # Cast to list to satisfy type checker
        prompts = cast(List[str], training_input["prompts"])
        training_input["prompts"] = list(prompts)  # type: ignore

        pad_value = self.rl_cluster.rollout.pad_id()
        eos_value = self.rl_cluster.rollout.eos_id()

        rollout_micro_batch_size = cast(int, self._rollout_micro_batch_size)
        compute_logps_micro_batch_size = cast(int, self._compute_logps_micro_batch_size)

        # 1. Generate
        rollout_output = self.rl_cluster.generate(
            prompts=prompts,
            mode=mode,
            micro_batch_size=(
                rollout_micro_batch_size * self.algo_config.num_generations  # type: ignore
            ),
        )

        if self.rl_cluster.global_steps % 1 == 0:
            print("START ============================")
            logging.info(f"Prompt: {prompts[0]}")
            if "ground_truth" in training_input:
                gt = training_input["ground_truth"]
                gt_val = gt[0] if hasattr(gt, "__getitem__") and len(gt) > 0 else gt
                print(f"Ground Truth: {gt_val}")
            print(f"Completion: {rollout_output.text[0]}")
            print("END ==============================")

        completion_ids = rollout_output.tokens
        prompt_ids = rollout_output.left_padded_prompt_tokens

        prompt_mask = prompt_ids != pad_value
        completion_padding_mask = jnp.not_equal(completion_ids, pad_value)
        completion_mask = common.make_completion_mask(completion_ids, eos_tok=eos_value)
        completion_mask = completion_mask * completion_padding_mask

        devices = self.rl_cluster.r2m[rl_cluster_lib.Role.REFERENCE].devices
        with self.rl_cluster.perf.span("refer_inference", devices):
            ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
                prompt_tokens=prompt_ids,
                completion_tokens=completion_ids,
                pad_id=pad_value,
                eos_id=eos_value,
                micro_batch_size=(
                    compute_logps_micro_batch_size * self.algo_config.num_generations  # type: ignore
                ),
            )

        # Student (Old) logprobs
        devices = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR].devices
        with self.rl_cluster.perf.span("old_actor_inference", devices):
            old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
                prompt_tokens=prompt_ids,
                completion_tokens=completion_ids,
                micro_batch_size=(
                    compute_logps_micro_batch_size * self.algo_config.num_generations  # type: ignore
                ),
            )

        # 3. Compute Rewards
        ref_logps = cast(jax.Array, ref_per_token_logps)
        old_logps = cast(jax.Array, old_per_token_logps)

        rewards = on_policy_reward(old_logps, ref_logps)

        # 4. Compute Advantages
        advantages = compute_advantages(rewards)

        # Log metrics for dense rewards
        valid_rewards = rewards * completion_mask
        mean_reward = valid_rewards.sum() / jnp.clip(completion_mask.sum(), min=1)
        per_seq_reward = valid_rewards.sum(axis=1)

        self.rl_cluster.buffer_metrics(
            {
                "rewards/mean": (mean_reward, np.mean),
                "rewards/per_seq_mean": (per_seq_reward.mean(), np.mean),
                "rewards/per_seq_min": (per_seq_reward.min(), np.min),
                "rewards/per_seq_max": (per_seq_reward.max(), np.max),
            },
            mode=mode,
        )

        # Log completion lengths
        completion_lengths = completion_mask.sum(axis=-1)
        self.rl_cluster.buffer_metrics(
            {
                "completions/mean_length": (np.mean(completion_lengths), np.mean),
                "completions/max_length": (np.max(completion_lengths), np.max),
                "completions/min_length": (np.min(completion_lengths), np.min),
            },
            mode=mode,
        )

        for i, (prompt, completion) in enumerate(zip(prompts, rollout_output.text)):
            self.rl_cluster.buffer_metrics(
                {
                    "prompts": (prompt, None),
                    "completions": (completion, None),
                    "rewards/sum": (per_seq_reward[i], np.mean),
                },
                mode=mode,
            )

        return TrainExample(  # type: ignore
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            ref_per_token_logps=ref_per_token_logps,
            advantages=advantages,
            old_per_token_logps=old_per_token_logps,
        )

    def _compute_trajectory_ids(self, example: TrainingInputT, steps: int) -> List[str]:
        prompts = cast(List[str], example["prompts"])
        batch_size = len(prompts) // self.algo_config.num_generations  # type: ignore
        row_offset = steps * batch_size
        row_offsets = np.repeat(
            np.arange(row_offset, row_offset + batch_size),
            self.algo_config.num_generations,
            axis=0,
        )
        group_offsets = np.tile(
            np.arange(self.algo_config.num_generations),
            batch_size,
        )
        return [f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)]

    def _num_iterations(self) -> int:
        return self.algo_config.num_iterations

    def _num_generations(self) -> int:
        return self.algo_config.num_generations
