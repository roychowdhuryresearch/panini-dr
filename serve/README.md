# Serving the models

PANINI-DR needs three model services. The LLM and the reranker are vLLM servers;
vLLM is not a Python dependency of `panini-dr`, so install it in environments
of its own ([Install vLLM](#install-vllm)). The query encoder runs inside
`panini-dr` itself.

| Service | Script | What the paper used | GPU |
|---|---|---|---|
| LLM for every role | `serve/gpt_oss_120b.sh` | `openai/gpt-oss-120b` on vLLM 0.24.0, OpenAI-compatible, tensor parallel 2, `--max-model-len 65536`, `--tool-call-parser openai --enable-auto-tool-choice` | 1 × 96 GB at the 65,536-token context (`PANINI_LLM_TP=1`); the paper used 2 for throughput |
| Reranker | `serve/jina_reranker.sh` | `jinaai/jina-reranker-v3.5` (about 0.6B parameters) on vLLM 0.26.0, served as `jina-rerank`, `--max-model-len 131072`, `--gpu-memory-utilization 0.45` | 0.45 of a 48 GB GPU (about 22 GB), shared with the query encoder |
| Query encoder | none (in-process) | `Qwen/Qwen3-Embedding-8B` via sentence-transformers, fp16, loaded by `panini-dr run` (and `panini-dr prepare`) on the visible GPU with the most free memory ([which GPU](#the-query-encoders-gpu)) | about 14.6 GiB (16 GB) |

The paper's runs used three GPUs: two 96 GB GPUs for the LLM, and one 48 GB GPU
that held both the reranker (0.45 of it, about 22 GB) and the query encoder
(about 16 GB). The LLM fits on one 96 GB GPU at the paper's 65,536-token context
(`PANINI_LLM_TP=1`); the second GPU only adds throughput. Host RAM for the dense
index is about 2.3 GB (FRAMES), 3.3 GB (BrowseComp-Plus) and 20 GB (MoNaCo) on
top of the corpus and BM25.

## Install vLLM

The paper's servers ran two different vLLM versions, each in its own
environment: vLLM 0.24.0 for the LLM, and vLLM 0.26.0 (with torch 2.11.0 and
transformers 5.15.0) for the reranker. Keep them apart from each other and from
the `panini-dr` environment, whose torch and transformers pins differ.

```bash
# LLM server: vLLM 0.24.0
uv venv --python 3.12 vllm-llm
uv pip install --python vllm-llm/bin/python vllm==0.24.0

# Reranker server: vLLM 0.26.0, torch 2.11.0, transformers 5.15.0
uv venv --python 3.12 vllm-rerank
uv pip install --python vllm-rerank/bin/python vllm==0.26.0 torch==2.11.0 transformers==5.15.0
```

Then start each script with its environment's `bin/` first on `PATH` (the
scripts call `vllm` and `python`; `VLLM_BIN` and `VLLM_PYTHON` name them
explicitly instead):

```bash
PATH="$PWD/vllm-llm/bin:$PATH"    CUDA_VISIBLE_DEVICES=0,1 serve/gpt_oss_120b.sh
PATH="$PWD/vllm-rerank/bin:$PATH" CUDA_VISIBLE_DEVICES=2   serve/jina_reranker.sh
```

Putting the environment's `bin/` on `PATH` also makes `ninja` findable, which
Blackwell GPUs need (see [Notes](#notes)).

## Start the services

```bash
# LLM, tensor parallel 2 on two GPUs (the paper's setup and the script's default)
CUDA_VISIBLE_DEVICES=0,1 serve/gpt_oss_120b.sh   # binds localhost; set PANINI_LLM_HOST to expose it
# ...or on one 96 GB GPU
PANINI_LLM_TP=1 CUDA_VISIBLE_DEVICES=0 serve/gpt_oss_120b.sh

# Reranker: takes 0.45 of its GPU by default (about 22 GB of a 48 GB card),
# which leaves room for the query encoder of `panini-dr run` on the same GPU
CUDA_VISIBLE_DEVICES=2 serve/jina_reranker.sh
```

Both scripts print the environment variables to export. Hosts, ports, GPU
memory and the vLLM executable are all variables:

| Variable | Default | Script |
|---|---|---|
| `PANINI_LLM_WEIGHTS` | `openai/gpt-oss-120b` | LLM (HF id or local dir) |
| `PANINI_LLM_MODEL` | `openai/gpt-oss-120b` | LLM served name = what `panini-dr run` sends |
| `PANINI_LLM_HOST` / `PANINI_LLM_PORT` | loopback / 8000 | LLM |
| `PANINI_LLM_TP` | 2 | LLM (1 = one GPU, enough at the 65,536-token context) |
| `PANINI_LLM_MAX_MODEL_LEN` | 65536 | LLM (keep: the reader is sized for it) |
| `PANINI_LLM_GPU_UTIL` | vLLM default | LLM |
| `VLLM_BIN` | `vllm` | LLM |
| `JINA_RERANK_WEIGHTS` | `jinaai/jina-reranker-v3.5` | reranker |
| `JINA_RERANK_HOST` / `JINA_RERANK_PORT` | loopback / 8090 | reranker |
| `JINA_RERANK_GPU_UTIL` | 0.45 | reranker (fraction of the GPU; see below) |
| `JINA_RERANK_MAX_MODEL_LEN` | 131072 | reranker |
| `VLLM_PYTHON` | `python` | reranker |

Then, for `panini-dr run`:

```bash
export PANINI_LLM_BASE_URL="http://<llm-host>:<llm-port>/v1"
export PANINI_LLM_MODEL="openai/gpt-oss-120b"
export JINA_RERANK_URL="http://<reranker-host>:<reranker-port>/v1/rerank"
```

### Reranker memory

jina-reranker-v3.5 is small (about 0.6B parameters); most of its memory is KV
cache for the 131,072-token context. `--gpu-memory-utilization` is a fraction of
the whole GPU: the default 0.45 is about 22 GB on a 48 GB card, which is what the
paper's runs used, with the query encoder (about 16 GB) on the same card. On a
GPU of its own you can raise it; on a smaller card raise it so that the absolute
amount stays about the same. If the KV cache cannot hold one 131,072-token
sequence, vLLM stops at start-up with an error rather than serving a shorter
context.

## The query encoder's GPU

`panini-dr run` (and `panini-dr prepare` when it builds a dense index) loads
Qwen3-Embedding-8B in fp16, about 14.6 GiB, on the visible GPU that has the most
free memory when it starts. Choose the GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=2 panini-dr run ...        # the encoder goes to physical GPU 2
```

With one visible GPU there is no choice to make; restricting the process to one
GPU also keeps it from opening a CUDA context on every other GPU while it picks.
Each `panini-dr` process loads its own copy of the encoder: two runs (or a run
and a `prepare`) on the same GPU need about 30 GB there and can run out of
memory. Give each process its own GPU, or check that the GPU has room for both.
`prepare --device cuda:N` picks the device explicitly.

## Health checks

```bash
curl -s "${PANINI_LLM_BASE_URL}/models"
curl -s "${JINA_RERANK_URL}" -H 'Content-Type: application/json' \
  -d '{"model": "jina-rerank", "query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 2}'
```

The second call must return `results` with document 0 first. If the reranker
fails during a run, the search falls back to fusion order (as the paper's code
did); `panini-dr run` counts every fallback and prints a warning summary, and
`--strict-rerank` aborts on the first failure instead. A run with fallbacks does
not reproduce the paper's configuration.

## Notes

- **Context length.** The reader sub-agent sizes its article window for a
  65,536-token server context (`MODEL_CONTEXT_TOKENS` in
  `panini_dr/adapters/ours/_agentic_read.py`). Serving a longer context does not
  make the reader use it; serving a shorter one causes context-length errors.
- **Blackwell GPUs.** flashinfer JIT-compiles kernels and needs `ninja` on
  `PATH`, including in non-interactive shells (put the vLLM venv's `bin/` first).
- **Determinism.** Runs use temperature 0 and seed 42, but vLLM is not
  bit-deterministic; expect run-to-run variation (the paper reports it).
- **`--trust-remote-code`** (reranker) executes code from the model's Hugging Face
  repository. Review it first.

## Licences

| Model | Licence |
|---|---|
| `openai/gpt-oss-120b` | Apache-2.0 |
| `jinaai/jina-reranker-v3.5` | **CC BY-NC 4.0 (non-commercial use only)** |
| `Qwen/Qwen3-Embedding-8B` | Apache-2.0 |

Swapping the reranker (or running without one) changes the results; the paper's
numbers are for jina-reranker-v3.5.
