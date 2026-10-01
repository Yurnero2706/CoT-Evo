#!/usr/bin/env bash
#
# Serve Qwen3-8B offline on a single H100 with vLLM's OpenAI-compatible API,
# so scripts/generate_cot_single.py can drive it unchanged.
#
#   bash scripts/serve_qwen3_local.sh
#
# Override anything via the environment:
#   MODEL=Qwen/Qwen3-14B PORT=8001 bash scripts/serve_qwen3_local.sh
#
# Air-gapped machine? Fetch the weights once on a box with network access
#   huggingface-cli download Qwen/Qwen3-8B --local-dir /data/models/Qwen3-8B
# then point MODEL at that directory and export HF_HUB_OFFLINE=1.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
# The API-visible name. generate_cot_single.py infers the thinking dialect
# from it, and it must match `model_name` in config/models.vllm.yaml, so do
# not drop the "qwen" from it.
SERVED_NAME="${SERVED_NAME:-qwen3-8b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Qwen3-8B is natively 32768 tokens. The DiscourseMT prompts are ~400 tokens,
# so this leaves the whole budget for reasoning. Going beyond 32768 needs YaRN
# scaling, which costs short-prompt quality - not worth it here.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# bf16 Qwen3-8B is ~16 GB of weights on an 80 GB H100; the rest becomes KV
# cache, which is what sets how many requests run concurrently.
GPU_UTIL="${GPU_UTIL:-0.90}"

# Separates the <think> block into the response's `reasoning_content` field,
# which is exactly what OpenAIProvider(capture_reasoning=True) reads. Without
# it the tags stay inline in `content` and the trace is still recovered, just
# less cleanly. The parser was named "deepseek_r1" before vLLM 0.9; if vLLM
# rejects "qwen3", run `vllm serve --help | grep -A5 reasoning-parser` and use
# whatever your build lists.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

echo "model            : ${MODEL}"
echo "served as        : ${SERVED_NAME}"
echo "endpoint         : http://${HOST}:${PORT}/v1"
echo "max_model_len    : ${MAX_MODEL_LEN}"
echo "reasoning parser : ${REASONING_PARSER}"
echo

exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --enable-prefix-caching \
    --max-num-seqs 64 \
    --disable-log-requests
