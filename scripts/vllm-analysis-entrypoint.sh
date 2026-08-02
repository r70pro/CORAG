#!/usr/bin/env bash
set -euo pipefail

model=${KIRAG_ANALYSIS_MODEL:?Set KIRAG_ANALYSIS_MODEL}
revision=${KIRAG_ANALYSIS_MODEL_REVISION:?Set KIRAG_ANALYSIS_MODEL_REVISION}

args=(
  serve "$model"
  --revision "$revision"
  --host 0.0.0.0
  --enforce-eager
  --language-model-only
  --gpu-memory-utilization "${KIRAG_ANALYSIS_GPU_MEMORY_UTILIZATION:-0.57}"
  --max-model-len "${KIRAG_ANALYSIS_MAX_MODEL_LEN:-32768}"
  --max-num-batched-tokens "${KIRAG_ANALYSIS_MAX_BATCHED_TOKENS:-8192}"
  --tensor-parallel-size "${KIRAG_VLLM_TENSOR_PARALLEL_SIZE:-1}"
)

# Reasoning parsers are model-specific. Gemma emits ordinary assistant content
# and must never be started with Qwen's parser.
if [[ ${model,,} == qwen/qwen3* ]]; then
  args+=(--reasoning-parser qwen3)
fi

exec vllm "${args[@]}"
