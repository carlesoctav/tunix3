"""On-policy learner implementation."""

from __future__ import annotations

from absl import logging
import dataclasses
from typing import List, Sequence, TypeVar, Any

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
    kl_penalty_coef: float = 1.0

    def __post_init__(self):
        return


TOnPolicyConfig = TypeVar("TOnPolicyConfig", bound=OnPolicyConfig)


def on_policy_reward(
    log_prob_student: jax.Array,
    log_prob_ref: jax.Array,
    kl_penalty_coef: float = 1.0,
) -> jax.Array:
    """Computes the on-policy reward (negative reverse KL).

    Args:
      log_prob_student: Log probabilities from the student model.
      log_prob_ref: Log probabilities from the reference model.
      kl_penalty_coef: Coefficient for KL penalty scaling.

    Returns:
      The computed reward: -kl_penalty_coef * (log_student - log_ref).
    """
    # Reverse KL: log_student - log_ref, negate for reward
    return kl_penalty_coef * (log_prob_ref - log_prob_student)  # type: ignore


def compute_advantages(
    rewards: jax.Array,
    num_generations: int,
) -> jax.Array:
    """Computes advantages from rewards with per-group centering.

    Centers advantages within each prompt's generations to ensure
    relative comparison between different completions.

    Args:
      rewards: Per-token rewards of shape [B*G, seq_len].
      num_generations: Number of generations per prompt (G).

    Returns:
      Centered advantages of shape [B*G, seq_len].
    """
    if num_generations <= 1:
        # No centering possible with single generation per prompt
        return rewards

    # rewards shape: [B*G, seq_len]
    batch_times_gen, seq_len = rewards.shape
    batch_size = batch_times_gen // num_generations

    # Reshape to [B, G, seq_len] for per-group centering
    rewards_reshaped = rewards.reshape(batch_size, num_generations, seq_len)

    # Center within each group (across G dimension)
    group_mean = jnp.mean(rewards_reshaped, axis=1, keepdims=True)
    centered = rewards_reshaped - group_mean

    # Flatten back to [B*G, seq_len]
    return centered.reshape(batch_times_gen, seq_len)


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

    if train_example.old_per_token_logps is None:
        old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
    else:
        old_per_token_logps = train_example.old_per_token_logps

    log_ratio = per_token_logps - old_per_token_logps
    ratio = jnp.exp(log_ratio)

    advantages = train_example.advantages  # [B*G, seq_len]

    # PPO Clipping
    clip_eps = 0.2
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    per_token_loss = -jnp.minimum(surr1, surr2)  # [B*G, seq_len]

    loss = common.aggregate_loss(per_token_loss, completion_mask, "token-mean")  # (,)

    metrics = {
        "ppo/ratio_mean": (ratio * completion_mask).sum()
        / (completion_mask.sum() + 1e-8),
        "ppo/clip_fraction": (
            (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps) * completion_mask
        ).sum()
        / (completion_mask.sum() + 1e-8),
    }

    return loss, metrics


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

        self.rl_cluster.actor_trainer.with_gen_model_input_fn(gen_model_input)

    def _generate_and_compute_advantage(
        self,
        training_input: TrainingInputT,
        mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
    ) -> TrainExample:
        # pad_value = self.rl_cluster.rollout.pad_id()
        # eos_value = self.rl_cluster.rollout.eos_id()
        eos_value = 106
        pad_value = 0

        rollout_micro_batch_size = self._rollout_micro_batch_size
        compute_logps_micro_batch_size = self._compute_logps_micro_batch_size

        prompts = list(training_input["prompts"])

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
            print(f"Prompt: {prompts[0]}")
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
            # [B*G, T]
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

        rewards = ref_per_token_logps - old_per_token_logps  # [B*G, seq]
        valid_rewards = rewards * completion_mask
        advantages = valid_rewards  # [B*G, seq]
        per_seq_reward = valid_rewards.sum(axis=1)  # [B*G, Seq] -> [B*G, ]
        per_seq_reward_mean = per_seq_reward / completion_mask.sum(
            axis=1
        )  # [B*G, Seq] -> [B*G, ]

        # Log completion lengths
        completion_lengths = completion_mask.sum(axis=-1)  # [B*G, ]
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
                    "rewards/sum_trajectories_mean": (per_seq_reward[i], np.mean),
                    "rewards/mean_trajectories_mean": (per_seq_reward_mean[i], np.mean),
                },
                mode=mode,
            )

        return TrainExample(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            ref_per_token_logps=ref_per_token_logps,
            advantages=advantages,
            old_per_token_logps=old_per_token_logps,
        )

    def _compute_trajectory_ids(self, example: TrainingInputT, steps: int) -> List[str]:
        prompts = example["prompts"]
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
