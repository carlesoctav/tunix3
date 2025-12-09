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


set -x # Enable xtrace

batch_size=${batch_size:-4}
mini_batch_size=${mini_batch_size:-4}
train_micro_batch_size=${train_micro_batch_size:-1}
num_iterations=${num_iterations:-1}
num_batches=${num_batches:-3738}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-1.0}

# Sanity checks and derived values
if [ $((batch_size % mini_batch_size)) -ne 0 ]; then
  echo "ERROR: batch_size ($batch_size) must be divisible by mini_batch_size ($mini_batch_size)"
  exit 1
fi

if [ $((mini_batch_size % train_micro_batch_size)) -ne 0 ]; then
  echo "ERROR: mini_batch_size ($mini_batch_size) must be divisible by train_micro_batch_size ($train_micro_batch_size)"
  exit 1
fi

if [ "$num_iterations" -le 0 ]; then
  echo "ERROR: num_iterations ($num_iterations) must be greater than 0"
  exit 1
fi

grad_acc_steps=$((mini_batch_size / train_micro_batch_size))

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Mini Batch Size: $mini_batch_size"
echo "  Train Micro Batch Size: $train_micro_batch_size"
echo "  Num Batches: $num_batches"
echo "  Num Iterations: $num_iterations"
echo "  Num Epochs: $num_train_epochs"
echo "  Warmup Ratio: $warmup_ratio"
echo "  Train Fraction: $train_fraction"
echo "  Grad Accumulation Steps: $grad_acc_steps"

max_steps_float=$(awk "BEGIN {print $num_batches * $num_train_epochs * $train_fraction * ($batch_size / $mini_batch_size) * $num_iterations}")

max_steps=$(printf "%.0f" "$max_steps_float")


warmup_steps=$(awk "BEGIN {printf \"%.0f\", $warmup_ratio * $max_steps}")

echo "Max steps (optimizer updates): $max_steps"
echo "Rounded warmup steps: $warmup_steps"
echo "Effective grad accumulation steps: $grad_acc_steps"

python3 -m tunix.cli.grpo_main \
  base_config.yaml \
  reference_model_config.model_name="gemma2-2b-it" \
  reference_model_config.model_id="google/gemma-2/flax/gemma2-2b-it" \
  reference_model_config.model_source="kaggle" \
  reference_model_config.model_download_path="/tmp/models/gemma2-2b" \
  reference_model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/1" \
  reference_model_config.mesh.shape="(4,1)" \
  reference_model_config.mesh.axis_names="('fsdp','tp')" \
  reference_model_config.rng_seed=42 \
  actor_model_config.lora_config.rank=16 \
  actor_model_config.lora_config.alpha=16 \
  actor_model_config.lora_config.module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum" \
  actor_model_config.mesh.shape="(4,1)" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_model_config.mesh.shape="(1,4)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  tokenizer_config.tokenizer_path="/tmp/models/gemma2-2b/models/google/gemma-2/flax/gemma2-2b-it/1/tokenizer.model" \
  tokenizer_config.tokenizer_type="sentencepiece" \
  tokenizer_config.add_bos=false \
  dataset_name="gsm8k" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_test_batches=100 \
  num_train_epochs=$num_train_epochs \
  rl_training_config.actor_optimizer_config.opt_type="adamw" \
  rl_training_config.actor_optimizer_config.peak_value=3e-6 \
  rl_training_config.actor_optimizer_config.schedule_type="warmup_cosine_decay_schedule" \
  rl_training_config.actor_optimizer_config.init_value=0.0 \
  rl_training_config.actor_optimizer_config.end_value=0.0 \
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.actor_optimizer_config.warmup_steps=$warmup_steps \
  rl_training_config.actor_optimizer_config.decay_steps=$max_steps \
  rl_training_config.actor_optimizer_config.b1=0.9 \
  rl_training_config.actor_optimizer_config.b2=0.99 \
  rl_training_config.actor_optimizer_config.weight_decay=0.1 \
  rl_training_config.actor_optimizer_config.max_grad_norm=0.1 \
  rl_training_config.eval_every_n_steps=10 \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/grpo" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=20 \
  rl_training_config.checkpointing_options.save_interval_steps=500 \
  rl_training_config.checkpointing_options.max_to_keep=4 \
  rl_training_config.checkpoint_root_directory="gs://carles-git-good/tunix/gemma2-repro" \
  rl_training_config.mini_batch_size=$mini_batch_size \
  rl_training_config.train_micro_batch_size=$train_micro_batch_size \
  rl_training_config.profiler_options={} \
  rollout_config.total_generation_steps=768 \
  rollout_config.max_prompt_length=256 \
  rollout_config.temperature=0.9 \
  rollout_config.top_p=1.0 \
  rollout_config.top_k=50 \
  rollout_engine="vanilla" \
  offload_to_cpu=false \
  grpo_config.num_generations=4 \
  grpo_config.num_iterations=$num_iterations \
  grpo_config.beta=0.08 \
  grpo_config.epsilon=0.2 \
  reward_functions="['tunix/cli/reward_fn/gsm8k.py']"
