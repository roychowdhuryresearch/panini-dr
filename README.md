# PANINI-DR

PANINI-DR is the harness from the paper **"PANINI-DR: Controlled Deep Research
through Generative Semantic Workspaces"** (M. Yigit Turali\*, Shreyas Rajesh\*,
M. Efe Lorasdagi, Chenda Duan, Kartik Sharma, Vwani Roychowdhury; \*equal
contribution; [arXiv:2610.15208](https://arxiv.org/abs/2610.15208)). It answers hard multi-step questions over a fixed
document collection. An Orchestrator agent asks a Planner for a research plan,
sends Researcher agents after the open subquestions, and writes the final
answer. All roles run on one OpenAI-compatible LLM endpoint (the paper uses
`gpt-oss-120b` on vLLM). Search is hybrid (BM25 + Qwen3-Embedding-8B, fused,
then reranked by jina-reranker-v3.5), and an article-reader sub-agent opens
whole documents.

Two ideas set it apart:

- **Explicit research state.** The plan is a shared workspace in the Generative
  Semantic Workspace (GSW) format. It records the subquestions, how they depend
  on each other, the findings so far and the evidence behind each one. The
  agents read and update this workspace instead of passing a long chat history
  around.
- **Verified findings.** A proposed answer to a subquestion is checked against
  that subquestion and its cited evidence before it is committed. A rejected
  finding sends the researcher back; an accepted finding feeds later
  subquestions and the final answer.

This repository contains the harness, the paper configuration, the graders, the
data preparation scripts, and a check (`panini-dr verify`) that this code makes
the same decisions as the code behind the paper's numbers.

- [Requirements](#requirements)
- [Install](#install)
- [Serve the models](#serve-the-models)
- [Quickstart](#quickstart)
- [Run on your own data](#run-on-your-own-data)
- [Reproduce the paper](#reproduce-the-paper)
- [Check this code against the paper's runs](#check-this-code-against-the-papers-runs-panini-dr-verify)
- [Reference](#reference)
- [Licence](#licence) and [citation](#citation)

## Requirements

| What | Paper setup | Notes |
|---|---|---|
| LLM, every role | `openai/gpt-oss-120b` on vLLM 0.24.0 | One 96 GB GPU is enough at the paper's 65,536-token context. The paper used two (tensor parallel 2) for more throughput. |
| Reranker | `jinaai/jina-reranker-v3.5` on vLLM 0.26.0 | A small model (about 0.6B parameters). The paper's runs gave it 0.45 of a 48 GB GPU (about 22 GB) and put the query encoder on the same GPU. **Licence CC BY-NC 4.0: non-commercial use only.** |
| Query encoder | `Qwen/Qwen3-Embedding-8B`, loaded by `panini-dr` | About 14.6 GiB (16 GB) of GPU memory on the machine that runs `panini-dr`, on the visible GPU with the most free memory; choose it with `CUDA_VISIBLE_DEVICES`. Building a dense index for your own documents needs it too. |
| Host RAM | | About 45-60 GB for MoNaCo (dense index + BM25); much less for FRAMES and BrowseComp-Plus. |
| BrowseComp-Plus only | pyserini 1.2.0 + a full JDK 21 | For the published Lucene BM25 index. See [JDK 21](#jdk-21-browsecomp-plus-only) for the install. |
| Python | 3.12 | |

vLLM is not a dependency of `panini-dr`. Install it in environments of its own:
[serve/README.md](serve/README.md#install-vllm) gives the pinned install lines
(vLLM 0.24.0 for the LLM; vLLM 0.26.0 with torch 2.11.0 and transformers 5.15.0
for the reranker, in a separate environment).

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
git clone <this repository> panini-dr
cd panini-dr
uv sync --locked --extra dense --extra bcp --extra eval --extra datasets --extra dev
source .venv/bin/activate          # or put `uv run` in front of each command
```

`uv.lock` pins every package to the version used for the paper's runs.
`--locked` makes uv stop, instead of re-resolving, if the lock and
`pyproject.toml` disagree.

`uv sync` is exact: it installs the extras you name and **removes the packages
of every extra you leave out**. Running `uv sync --locked --extra datasets`
after the line above would uninstall pytest, torch and the rest. To add an extra
to an existing environment, either repeat the full list of extras, or add
`--inexact`, which installs without removing anything:

```bash
uv sync --locked --inexact --extra bcp     # adds bcp, keeps every other extra
```

Pick only the extras you need:

| Extra | What it adds | Needed for |
|---|---|---|
| `dense` | torch, sentence-transformers, transformers, faiss-cpu | the paper configuration (hybrid retrieval), dense indexes for your own data, `panini-dr verify` |
| `bcp` | pyserini (needs a JDK 21 on `JAVA_HOME`) | BrowseComp-Plus |
| `eval` | Bedrock token helper, ijson | the Bedrock judge backend; `score obs` on BrowseComp-Plus |
| `datasets` | huggingface-hub, pyarrow, zstandard | the scripts under `datasets/` |
| `dev` | pytest | the test suite |

Without uv, `pip install -e '.[dense,bcp,eval,datasets,dev]'` installs the same
direct dependencies (pinned in `pyproject.toml`), but not the locked versions of
their own dependencies.

### JDK 21 (BrowseComp-Plus only)

pyserini runs Lucene inside the Python process and needs a full JDK 21 (with
`javac`; a JRE or a trimmed runtime is not enough) on `JAVA_HOME`:

```bash
sudo apt install openjdk-21-jdk                     # Debian/Ubuntu
# or, without root: conda install -c conda-forge openjdk=21
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
"$JAVA_HOME/bin/java" -version                      # should report version 21
```

Put the `export` in your shell profile or in `.env`. If `JAVA_HOME` is unset,
`panini-dr run` also looks for a JDK installed by `install-jdk` under
`~/.jdk/`.

### tiktoken

The article reader counts tokens with tiktoken's `o200k_base` encoding. Fetch it
once while you have network access. Without it the reader falls back to a
character estimate and reads different amounts of text.

```bash
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

## Serve the models

```bash
# LLM on one GPU (the script's default is tensor parallel 2 on two GPUs)
PANINI_LLM_TP=1 CUDA_VISIBLE_DEVICES=0 serve/gpt_oss_120b.sh

# Reranker: 0.45 of its GPU by default (about 22 GB of a 48 GB card), which
# leaves room for the query encoder on the same GPU
CUDA_VISIBLE_DEVICES=1 serve/jina_reranker.sh
```

Run each script with its vLLM environment's `bin/` first on `PATH` (see
[Install vLLM](serve/README.md#install-vllm)).

Each script prints the variables to export. With the defaults:

```bash
export PANINI_LLM_BASE_URL=http://localhost:8000/v1
export JINA_RERANK_URL=http://localhost:8090/v1/rerank
```

Keep the LLM's `--max-model-len` at 65,536: the reader is sized for it. See
[serve/README.md](serve/README.md) for health checks, ports and other options.

`panini-dr run` loads the query encoder itself (about 14.6 GiB), on the visible
GPU with the most free memory. Choose that GPU with `CUDA_VISIBLE_DEVICES`
(`CUDA_VISIBLE_DEVICES=1 panini-dr run ...` puts it next to the reranker, as in
the paper's runs). Every `panini-dr` process loads its own copy, so two
concurrent runs on one GPU can run out of memory; see
[the query encoder's GPU](serve/README.md#the-query-encoders-gpu).

You can also copy `.env.example` to `.env` and set what you need: the two
service URLs above are filled in with the scripts' defaults, every optional
line is commented out. Every `panini-dr` command (`run`, `prepare`, `score`,
`verify`) and the scripts under `datasets/` read `./.env` by themselves (or the
nearest `.env` in a parent directory); variables already set in the shell win.
An empty value in `.env` (`HF_HOME=`) is ignored rather than set: for some tools
an empty variable is not the same as an unset one (an empty `HF_HOME` would move
the Hugging Face cache to `./hub`).

## Quickstart

### 1. Smoke test (CPU, no network, no model servers)

```bash
pytest -q                          # about 2 minutes on CPU
panini-dr run --print-config       # the resolved configuration, its drift from configs/paper.json, the turn limits
```

`tests/e2e` runs the whole paper configuration (planner, orchestrator,
researchers, reader, verifiers, synthesis validator) on a toy corpus. Only the
LLM server, the encoder and the reranker are faked.

`panini-dr prepare` also runs on CPU when you skip the dense index. This turns
the toy example in [`examples/toy/`](examples/toy/) (eight made-up articles,
three questions) into a dataset directory:

```bash
panini-dr prepare --config examples/toy/dataset.yaml --out data/toy-bm25 --no-dense
```

### 2. Run the toy example (needs the model servers and a GPU)

```bash
panini-dr prepare --config examples/toy/dataset.yaml --out data/toy   # builds the dense index on a GPU
panini-dr run --dataset data/toy --output-dir runs/toy
```

The first `prepare` with a dense index downloads Qwen3-Embedding-8B (about
16 GB) from Hugging Face. To grade the answers, see
[docs/own-dataset.md](docs/own-dataset.md#4-grade-the-answers).

## Run on your own data

Write your questions and your documents as two JSONL files, one JSON object per
line:

```json
{"id": "q1", "question": "Which river flows through the birthplace of the composer of The Glass Orchard?", "answer": "Varno River"}
{"id": "q2", "question": "Which towns does the Varno River flow through?", "answer": ["Tessby", "Orlin", "Calder Ford"], "answer_type": "set"}
```

```json
{"id": "d1", "title": "Varno River", "text": "The Varno River is a river in the province of Ardel. ..."}
```

Then:

```bash
panini-dr prepare --questions questions.jsonl --docs docs.jsonl --out data/mydata
panini-dr run --dataset data/mydata --output-dir runs/mydata
panini-dr score answers -b byod --run runs/mydata/<run dir> \
    --questions data/mydata/questions.jsonl --cache judge/mydata_cache.json \
    --out judge/mydata_verdicts.json --backend openai --model <judge model>
```

`answer_type` is `short` (one answer, graded correct or incorrect) or `set` (a
list of items, graded by item-level F1). The full guide, with the schemas, the
options, the GPU needs and grading, is [docs/own-dataset.md](docs/own-dataset.md).

## Reproduce the paper

[REPRODUCE.md](REPRODUCE.md) has the exact commands for each benchmark: prepare
the data, serve the models, run, grade. This configuration reproduces:

| Benchmark | Official score | Obs. (observed support) |
|---|---|---|
| BrowseComp-Plus CF-150 | 73.3% (110/150; easy/medium/hard 46/38/26 of 50 each) | 70.7% |
| FRAMES-824 | 78.5% (647/824) | 66.1% |
| MoNaCo-1315 | mean F1 0.549 | 0.458 |

Expect run-to-run variation. vLLM is not bit-deterministic, even at temperature
0 with a fixed seed: repeated BrowseComp-Plus runs of this configuration scored
between 99 and 110 of 150. REPRODUCE.md lists the sources of variation and the
known issues.

### Grading

The paper grades every answer with an LLM judge, 3 passes per question:
majority vote for BrowseComp-Plus and FRAMES, maximum F1 for MoNaCo. Every
question stays in the denominator. You choose the judge model (`--model` or
`PANINI_JUDGE_MODEL`; there is no default), and a different judge moves the
numbers somewhat.

The graders use the benchmark authors' own grading code, which this repository
does not copy. Fetch it once:

```bash
python scripts/fetch_grader_sources.py --dest third_party   # needs git and network
export BCP_REPO=$PWD/third_party/BrowseComp-Plus             # short answers: BrowseComp-Plus, FRAMES, your own data
export MONACO_JUDGE_DIR=$PWD/third_party/monaco              # set answers: MoNaCo, your own data
```

`--backend openai` talks to any OpenAI-compatible endpoint (`--judge-base-url`
or `PANINI_JUDGE_BASE_URL`; key in `PANINI_JUDGE_API_KEY` or `OPENAI_API_KEY`).
With it, `--judge-api chat` (the default) sends each judge prompt as a Chat
Completions request, and `--judge-api responses` as a Responses API request; use
`responses` when your judge model or endpoint serves only that API.
`--backend bedrock` uses the AWS Bedrock Responses API (region in
`PANINI_BEDROCK_REGION`). The default is `bedrock`, so pass `--backend openai`
for anything else. Verdicts are cached in the `--cache` file, keyed by judge
model and answer; `--offline` grades from that cache only and never calls a
judge.

## Check this code against the paper's runs (`panini-dr verify`)

A verify bundle holds recorded runs of the paper configuration: every response
the code received (LLM completions, query embeddings, reranker scores) and the
outputs it produced. `panini-dr verify` re-runs `panini-dr run` on the same
questions with those responses served from disk. Then it compares the results
strictly: the final answers, the full traces and the run configuration must
match byte for byte.

The bundle is published in the FRAMES corpus repository on Hugging Face, under
`verify/` (8 FRAMES questions in two runs, about 33 MB). It needs the installed
FRAMES data (see [datasets/README.md](datasets/README.md)) and the `dense`
extra, but no GPU, no model server and no network.

```bash
export PANINI_DATA_DIR=$PWD/data
python datasets/frames/prepare.py questions
python datasets/frames/prepare.py corpus
hf download roychowdhuryresearch/panini-dr-frames-corpus --repo-type dataset \
    --include "verify/*" --local-dir panini-dr-verify
panini-dr verify --bundle panini-dr-verify/verify --data-dir "$PANINI_DATA_DIR"
```

It takes about a minute on CPU and ends with a line like:

```
PASS: 8/8 questions reproduce the reference outputs exactly (50 s of replay)
```

- **What it shows.** Given the same model, encoder and reranker outputs, this
  repository makes exactly the same decisions as the code that produced the
  paper's numbers. It does not re-measure accuracy.
- **Why it can demand identical output.** A live run varies for three reasons.
  vLLM is not bit-deterministic. Python's string-hash seed changes the order in
  which some plan steps are shown to the orchestrator (they are kept in a set).
  Parallel researchers in one question share a read cache, so timing decides
  some reuse. The replay removes all three: it serves the recorded responses,
  fixes the hash seed to the recorded one, and releases each question's
  responses in the recorded event order.
- **Exit status.** 0 = PASS. 1 = FAIL, with a per-question report and the first
  request that differs (`--keep-output DIR` keeps the logs and per-miss
  reports). 2 = the bundle or the data cannot be used, for example a corpus
  with different bytes, or tiktoken's `o200k_base` not cached; nothing is
  replayed then.

Maintainers record new bundles against live services with `panini-dr verify
record` and package them with `scripts/build_verify_bundle.py`. That script
removes local paths, hosts and user names, and refuses to write a bundle that
still contains them.

## Reference

### Commands

| Command | What it does |
|---|---|
| `panini-dr run` | Run the paper configuration on a benchmark split (`--subset` + `--articles-cache`) or on a prepared dataset (`--dataset`). |
| `panini-dr prepare` | Turn your own `questions.jsonl` and `docs.jsonl` into a dataset directory. |
| `panini-dr score answers` | Grade a run's final answers with the graders of record. |
| `panini-dr score obs` | Observed support (Obs.) of a benchmark run, from its traces and its verdicts. |
| `panini-dr score import-legacy-cache` | Convert judge verdicts from the research code's cache format. |
| `panini-dr verify` | The record/replay check above. |

Every command has `--help`.

`panini-dr run` refuses to start if the resolved configuration differs from
[`configs/paper.json`](configs/paper.json). You can change a key with
`--set key=value --allow-config-drift`; the run then records the drift in
`run_meta.json` and is no longer the paper configuration. Other mistakes also
stop the run instead of silently changing the experiment: there is no default
corpus, a missing dense index is an error (a run never builds one), and subset
ids missing from the question file are an error. Before the first question,
`run` sends one test request to the LLM server (`GET <base URL>/models`, which
must list the model) and one to the reranker; an unreachable LLM stops the run
with exit status 2, and `--skip-preflight` skips both requests (for offline or
replayed runs). A run in which every question ended with zero turns exits with
status 1. Reranker failures are counted. `--strict-rerank` aborts on the first
one; without it a failed rerank falls back to fusion order and the run prints a
warning.

### What a run writes

Each run creates `<output-dir>/ours_gsw_planner_orchestrator_v1__<model>__<UTC time>/`:

| File | Content |
|---|---|
| `traces/q_<id>.json`, `.md` | The full trajectory of each question, and a readable rendering. `final_answer` is the answer that gets graded. |
| `cell_result.json` | Per-question results and the recorded configuration. Its `exact_match` and `f1` fields are string-match diagnostics, not the paper's metrics; use `panini-dr score`. |
| `run_meta.json` | Provenance: command line (API key redacted), package version, git commit, dataset paths, configuration drift. |
| `run_health.json` | Reranker calls and failures. The paper's runs had no failures. |
| `run.log` | The log. |

### Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `PANINI_LLM_BASE_URL` | run | OpenAI-compatible `/v1` URL of the LLM (or `--base-url`). Required. |
| `PANINI_LLM_MODEL` | run, serve | Served model name (default `openai/gpt-oss-120b`). |
| `PANINI_LLM_API_KEY` | run | Key for the LLM endpoint, if it needs one (or `--api-key`; default: a placeholder). `OPENAI_API_KEY` is never sent to the LLM. |
| `JINA_RERANK_URL` | run | Reranker endpoint, `.../v1/rerank`. Required by the paper configuration. |
| `PANINI_STRICT_RERANK` | run | `1` = same as `--strict-rerank`. |
| `PANINI_DATA_DIR` | all | Root of the prepared data and dense indexes (default `./data`). |
| `BROWSECOMP_QUESTIONS`, `MONACO_QUESTIONS` | run, score | Override the default question files under the data root. |
| `HF_TOKEN`, `HF_HOME`, `HF_HUB_OFFLINE` | datasets, run | Hugging Face access (the MoNaCo data is gated), cache location, offline mode. |
| `JAVA_HOME` | run (BrowseComp-Plus) | A full JDK 21 for pyserini. |
| `PANINI_JUDGE_MODEL` | score | Judge model name (or `--model`). No default. |
| `PANINI_JUDGE_BASE_URL`, `PANINI_JUDGE_API_KEY` | score | OpenAI-compatible judge endpoint and key (fall back to `OPENAI_BASE_URL`, `OPENAI_API_KEY`). |
| `PANINI_BEDROCK_REGION` | score | Region for `--backend bedrock` (falls back to `AWS_REGION`). |
| `BCP_REPO`, `MONACO_JUDGE_DIR` | score | Checkouts of the benchmark authors' grading code (see [Grading](#grading)). |

`GSW_PLANNER_TARGET_HUB_MODE` and `GSW_STRICT_TARGET_HUB_GATE` change the agent's
behaviour. They were unset in the paper's runs, and `panini-dr run` refuses to
start when one is set unless you pass `--allow-config-drift`. Variables that
switched code which has since been removed (`env_vars` in
`configs/removed_features.json`, e.g. `GSW_READ_TITLE_FALLBACK`) are always
rejected. The serve scripts have their own variables (ports,
GPU memory, tensor parallelism); see [serve/README.md](serve/README.md).

### Repository layout

| Path | What |
|---|---|
| `src/panini_dr/adapters/ours/` | The harness: orchestrator, planner, researchers, verifiers, reader, memory. |
| `src/panini_dr/retrieval/` | Corpus and chunking, BM25, dense retrieval, hybrid fusion, reranker, Lucene (BrowseComp-Plus). |
| `src/panini_dr/eval/` | Evaluation loop, dataset loaders, graders, Obs. scorer. |
| `src/panini_dr/config.py`, `configs/` | The paper configuration, and the benchmark splits (question ids only). |
| `src/panini_dr/cli.py`, `runner.py`, `byod.py` | The command line, `panini-dr run` and `panini-dr prepare`. |
| `src/panini_dr/replay/` | Record/replay of external calls, and `panini-dr verify`. |
| `datasets/` | Preparation scripts for the three benchmarks, and the Hugging Face dataset cards. |
| `serve/` | Scripts that serve the LLM and the reranker with vLLM. |
| `scripts/` | Fetch the graders' source code; build verify bundles. |
| `examples/toy/` | A tiny synthetic dataset for `panini-dr prepare`. |
| `docs/` | Guides: [your own dataset](docs/own-dataset.md). |
| `tests/` | Unit and end-to-end tests (offline). |

## Licence

The code is licensed under the [Apache License 2.0](LICENSE). The repository
ships no benchmark questions, answers, corpora, indexes or model weights; the
scripts download them under their own terms, which [NOTICE](NOTICE) summarises.
jina-reranker-v3.5, the reranker of the paper configuration, is licensed for
non-commercial use only.

## Citation

```bibtex
@article{turali2026paninidr,
  title   = {{PANINI-DR}: Controlled Deep Research through Generative Semantic Workspaces},
  author  = {Turali, M. Yigit and Rajesh, Shreyas and Lorasdagi, M. Efe and Duan, Chenda and Sharma, Kartik and Roychowdhury, Vwani},
  journal = {arXiv preprint arXiv:2610.15208},
  url     = {https://arxiv.org/abs/2610.15208},
  year    = {2026}
}
```

M. Yigit Turali and Shreyas Rajesh contributed equally.
