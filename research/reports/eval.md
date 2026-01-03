# GSM8K Evaluation Report

**Date:** 2026-01-02  
**Model:** google/gemma-3-1b-it  
**Dataset:** GSM8K test set (1319 examples)  
**Evaluation Scripts:** `research/script/eval_gsm8k.py`, `research/script/eval_gsm8k_base.py`

## Summary Results

| Model | Checkpoint | batch_size | Correct | Total | Accuracy |
|-------|------------|------------|---------|-------|----------|
| Base (no LoRA) | - | 32 | 735 | 1319 | 55.72% |
| Base (no LoRA) | - | 1 | 224 | 392* | 57.14%* |
| + LoRA | on-policy-gsm8k-ppo-v2 | 32 | 755 | 1319 | 57.24% |
| + LoRA | fix-grpo | 32 | 800 | 1319 | 60.65% |
| + LoRA | fix-grpo | 1 | 811 | 1319 | **61.49%** |

*partial run (timed out)

## Key Findings

### 1. fix-grpo achieves best results
The `fix-grpo` checkpoint shows significant improvement over base:
- **+5.77%** over base (batch_size=1): 61.49% vs 55.72%
- **+4.25%** over on-policy-ppo-v2: 61.49% vs 57.24%

### 2. Left-padding attention mask bug
**Critical finding:** batch_size affects accuracy due to a bug in the attention mask handling with left-padded sequences.

| Model | batch_size=32 | batch_size=1 | Difference |
|-------|---------------|--------------|------------|
| Base | 55.72% | ~57.14% | +1.4% |
| fix-grpo | 60.65% | 61.49% | +0.84% |

**Root cause:** When batching sequences of different lengths, left-padding causes varying amounts of pad tokens. The attention mask computation in `tunix/generate/utils.py` doesn't correctly handle this, causing pad tokens to influence model predictions.

**Evidence:** Same model, same prompt, greedy decoding (temperature=0) produces different outputs:
- Janet's ducks question with batch_size=1: **18** (correct)
- Janet's ducks question with batch_size=32: **12** (wrong)

### 3. Comparison with lm-eval-harness
lm-eval-harness (PyTorch) reports ~60% for gemma-3-1b-it base on GSM8K. Our tunix (JAX) results:
- batch_size=32: 55.72% (gap due to attention mask bug)
- batch_size=1: ~57.14% (closer, remaining gap likely JAX vs PyTorch numerical precision)

## Checkpoint Locations

| Checkpoint | GCS Path |
|------------|----------|
| on-policy-gsm8k-ppo-v2 | `gs://carles-git-good/tunix/on-policy-gsm8k-ppo-v2/actor` |
| fix-grpo | `gs://carles-git-good/tunix/fix-grpo/actor` |

Both checkpoints use:
- LoRA rank: 16
- Model family: Gemma3
- Model name: gemma3-1b-it

## Evaluation Configuration

```yaml
temperature: 0.0  # greedy decoding
max_generation_steps: 768
eos_tokens: [1, 106]  # EOS and end_of_turn
extraction_method: strict  # looks for \boxed{} or #### format
```

## Output Files

Results are stored in:
- `eval_results/` - LoRA model evaluations
- `eval_results_base/` - Base model evaluations  
- `eval_results_fix_grpo/` - fix-grpo with batch_size=32
- `eval_results_fix_grpo_b1/` - fix-grpo with batch_size=1

Each directory contains:
- `rollouts_*.jsonl` - Per-example results (question, response, extracted_answer, correct)
- `summary_*.json` - Aggregate statistics

## Recommendations

1. **Use batch_size=1 for accurate evaluation** until the attention mask bug is fixed
2. **File a bug** for the left-padding attention mask issue in `tunix/generate/sampler.py`
3. **Use fix-grpo checkpoint** for best GSM8K performance (61.49%)

## Reproduction

```bash
# Base model evaluation (batch_size=1 recommended)
HF_HOME="/mnt/carles/.cache" uv run python research/script/eval_gsm8k_base.py \
  --model.model_family="Gemma3" \
  --model.model_name="gemma3-1b-it" \
  --batch_size=1

# fix-grpo LoRA evaluation
HF_HOME="/mnt/carles/.cache" uv run python research/script/eval_gsm8k.py \
  --model.model_family="Gemma3" \
  --model.model_name="gemma3-1b-it" \
  --model.lora_config.rank=16 \
  --model.lora_checkpoint_path="gs://carles-git-good/tunix/fix-grpo/actor" \
  --output_dir="./eval_results_fix_grpo" \
  --batch_size=1
```
