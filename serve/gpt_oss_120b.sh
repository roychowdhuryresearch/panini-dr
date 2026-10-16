#!/usr/bin/env bash
# Serve openai/gpt-oss-120b with vLLM the way the PANINI-DR paper runs did:
# OpenAI-compatible server, tensor parallel 2, 65,536-token context, OpenAI
# tool-call parser with automatic tool choice. Every PANINI-DR role
# (orchestrator, planner, researcher, reader, verifier, ...) talks to this one
# server.
#
#   CUDA_VISIBLE_DEVICES=0,1 serve/gpt_oss_120b.sh
#   PANINI_LLM_TP=1 CUDA_VISIBLE_DEVICES=0 serve/gpt_oss_120b.sh   # one 96 GB GPU
#   export PANINI_LLM_BASE_URL="http://<host>:<port>/v1"   # printed below
#   export PANINI_LLM_MODEL="openai/gpt-oss-120b"
#
# Notes
# - --max-model-len 65536 is what the paper's server ran with. The reader
#   sub-agent sizes its article window for exactly this context
#   (MODEL_CONTEXT_TOKENS = 65_536 in panini_dr/adapters/ours/_agentic_read.py);
#   a different server context changes behaviour (more or fewer
#   context-length errors), so keep it unless you change both.
# - The paper used vLLM 0.24.0 with tensor parallel 2 on two 96 GB GPUs (MXFP4
#   weights). The model fits on one 96 GB GPU at the 65,536-token context
#   (PANINI_LLM_TP=1); the second GPU only adds throughput. vLLM is not a
#   dependency of panini-dr; install it in its own environment (serve/README.md
#   has the install line) and put that environment's bin/ first on PATH, or set
#   VLLM_BIN to its vllm.
# - Blackwell GPUs: flashinfer JIT-compiles kernels at start-up and needs
#   `ninja` on PATH. If vLLM lives in a venv, put that venv's bin/ on PATH even in
#   a non-interactive shell, or engine start-up fails with
#   FileNotFoundError: 'ninja'.
# - vLLM is not bit-deterministic across runs even at temperature 0 with a seed.
set -euo pipefail

MODEL="${PANINI_LLM_WEIGHTS:-openai/gpt-oss-120b}"        # HF id or local directory
SERVED_NAME="${PANINI_LLM_MODEL:-openai/gpt-oss-120b}"    # what `panini-dr run --model` sends
HOST="${PANINI_LLM_HOST:-localhost}"
PORT="${PANINI_LLM_PORT:-8000}"
TP="${PANINI_LLM_TP:-2}"
MAX_MODEL_LEN="${PANINI_LLM_MAX_MODEL_LEN:-65536}"
VLLM_BIN="${VLLM_BIN:-vllm}"

EXTRA=()
if [[ -n "${PANINI_LLM_GPU_UTIL:-}" ]]; then
  EXTRA+=(--gpu-memory-utilization "${PANINI_LLM_GPU_UTIL}")
fi

echo "serving ${MODEL} as '${SERVED_NAME}' (tp=${TP}, max-model-len=${MAX_MODEL_LEN})"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "export PANINI_LLM_BASE_URL=http://${HOST}:${PORT}/v1"
echo "export PANINI_LLM_MODEL=${SERVED_NAME}"

exec "${VLLM_BIN}" serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
