#!/usr/bin/env bash
set -euo pipefail

role=${KIRAG_VLLM_ROLE:?Set KIRAG_VLLM_ROLE}
model=${KIRAG_VLLM_MODEL:?Set KIRAG_VLLM_MODEL}
revision=${KIRAG_VLLM_MODEL_REVISION:?Set KIRAG_VLLM_MODEL_REVISION}

args=(
  serve "$model"
  --revision "$revision"
  --host 0.0.0.0
  --gpu-memory-utilization "${KIRAG_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
  --max-model-len "${KIRAG_VLLM_MAX_MODEL_LEN:?Set KIRAG_VLLM_MAX_MODEL_LEN}"
  --max-num-batched-tokens "${KIRAG_VLLM_MAX_BATCHED_TOKENS:?Set KIRAG_VLLM_MAX_BATCHED_TOKENS}"
  --tensor-parallel-size "${KIRAG_VLLM_TENSOR_PARALLEL_SIZE:-1}"
)

case "$role" in
  ocr)
    [[ "$model" == "allenai/olmOCR-2-7B-1025-FP8" ]] || { echo "Invalid OCR model" >&2; exit 64; }
    args+=(--enforce-eager)
    ;;
  analysis)
    case "$model" in
      Qwen/Qwen3.6-35B-A3B|google/gemma-4-31B-it) ;;
      *) echo "Invalid analysis model" >&2; exit 64 ;;
    esac
    args+=(--language-model-only)
    if [[ ${model,,} == qwen/qwen3* ]]; then
      args+=(--reasoning-parser qwen3)
    fi
    ;;
  *) echo "Invalid KIRAG_VLLM_ROLE: $role" >&2; exit 64 ;;
esac

exec vllm "${args[@]}"
