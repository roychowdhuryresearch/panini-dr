#!/usr/bin/env bash
# Serve jinaai/jina-reranker-v3.5 with vLLM for PANINI-DR's search tool
# (reranker spec "jina"; served model name "jina-rerank", endpoint /v1/rerank).
#
#   CUDA_VISIBLE_DEVICES=2 serve/jina_reranker.sh
#   export JINA_RERANK_URL="http://<host>:<port>/v1/rerank"   # printed below
#
# Notes
# - The paper's reranker ran on vLLM 0.26.0 (torch 2.11.0, transformers 5.15.0),
#   in an environment of its own; serve/README.md has the install lines. Put that
#   environment's bin/ first on PATH, or set VLLM_PYTHON to its python.
# - LICENCE: jina-reranker-v3.5 is CC BY-NC 4.0 (non-commercial). The paper's
#   numbers use it; any other reranker (or none) changes them.
# - The model is listwise: the query and all 40 candidates go into ONE sequence.
#   --max-model-len must hold that whole packed sequence; at 32768 the
#   longest-evidence searches were rejected and silently fell back to fusion
#   order. 131072 is what the paper's server used.
# - Memory. The model is small (about 0.6B parameters); most of its memory is KV
#   cache for the 131072-token context. --gpu-memory-utilization is a fraction of
#   the whole GPU. The default 0.45 is what the paper's runs used: about 22 GB of
#   a 48 GB card, with the query encoder of `panini-dr run` (about 16 GB) on the
#   same card. On a GPU of its own you may raise it; on a smaller card raise it so
#   the absolute amount stays about the same (vLLM refuses to start if the KV
#   cache cannot hold one 131072-token sequence).
# - --trust-remote-code is required: the model's scoring head ships as custom
#   code in its Hugging Face repository. Review it before running.
set -euo pipefail

MODEL="${JINA_RERANK_WEIGHTS:-jinaai/jina-reranker-v3.5}"
HOST="${JINA_RERANK_HOST:-localhost}"
PORT="${JINA_RERANK_PORT:-8090}"
GPU_UTIL="${JINA_RERANK_GPU_UTIL:-0.45}"
MAX_MODEL_LEN="${JINA_RERANK_MAX_MODEL_LEN:-131072}"
PYTHON="${VLLM_PYTHON:-python}"

echo "serving ${MODEL} as 'jina-rerank' (max-model-len=${MAX_MODEL_LEN}, gpu-util=${GPU_UTIL})"
echo "export JINA_RERANK_URL=http://${HOST}:${PORT}/v1/rerank"

exec "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name jina-rerank \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --no-enable-log-requests \
  "$@"
