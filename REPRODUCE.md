# Reproducing the paper's numbers

This file gives the exact commands behind each number of PANINI-DR in the
paper: prepare the data, serve the models, run, grade. Set up the environment
first as in the [README](README.md#install):

```bash
uv sync --locked --extra dense --extra bcp --extra eval --extra datasets --extra dev
```

Name every extra you want in one `uv sync`, or add `--inexact`: without it,
`uv sync` removes the extras you leave out.

| Benchmark | Official score | Obs. (observed support) |
|---|---|---|
| BrowseComp-Plus CF-150 | 73.3% (110/150; easy/medium/hard 46/38/26 of 50 each) | 70.7% |
| FRAMES-824 | 78.5% (647/824) | 66.1% |
| MoNaCo-1315 | mean F1 0.549 | 0.458 |

**Expect your numbers to differ somewhat.** A live run is not deterministic (see
[Run-to-run variation](#run-to-run-variation)). To check that this code is the
code behind the numbers, use [`panini-dr verify`](#checking-the-code-instead-of-the-numbers),
which is exact.

- [1. Hardware and time](#1-hardware-and-time)
- [2. Serve the models](#2-serve-the-models)
- [3. Get the grading code and choose a judge](#3-get-the-grading-code-and-choose-a-judge)
- [4. BrowseComp-Plus CF-150](#4-browsecomp-plus-cf-150)
- [5. FRAMES-824](#5-frames-824)
- [6. MoNaCo-1315](#6-monaco-1315)
- [Run-to-run variation](#run-to-run-variation)
- [Checking the code instead of the numbers](#checking-the-code-instead-of-the-numbers)
- [Recomputing scores from existing outputs](#recomputing-scores-from-existing-outputs)

## 1. Hardware and time

The paper's runs used three GPUs: two 96 GB GPUs for the LLM (tensor parallel 2),
and one 48 GB GPU shared by the reranker (vLLM at `--gpu-memory-utilization
0.45`, about 22 GB) and the query encoder (about 16 GB). The LLM also fits on a
single 96 GB GPU at the paper's context length; two give more throughput.

| | Disk (data) | Host RAM | Wall time of the run of record |
|---|---|---|---|
| BrowseComp-Plus CF-150 | ~8.5 GB (~14 GB while preparing) | corpus (3.3 GB of text) and dense index (3.3 GB) in memory, plus a JVM for Lucene | ~4 h (it ran as two legs of 2.4 h and 1.6 h) |
| FRAMES-824 | ~2.4 GB | corpus (0.1 GB), dense index (2.3 GB) and BM25 in memory | ~6 h |
| MoNaCo-1315 | ~21 GB (~27 GB while installing) | 45-60 GB in total | ~21 h |

The data is not all the disk you need. Plan for these as well:

| | Disk | Where |
|---|---|---|
| gpt-oss-120b weights | ~65 GB | Hugging Face cache (`HF_HOME`) of the LLM server, downloaded when it first starts |
| Qwen3-Embedding-8B weights (query encoder) | ~16 GB | Hugging Face cache of the machine that runs `panini-dr` |
| jina-reranker-v3.5 weights | small (about 0.6B parameters) | Hugging Face cache of the reranker server |
| The `panini-dr` environment (`.venv`, all extras) | ~7.5 GB | the repository |
| The two vLLM environments | several GB each | wherever you create them ([serve/README.md](serve/README.md#install-vllm)) |

The benchmark `prepare` steps install precomputed vectors and never load the
query encoder. So unless `panini-dr prepare` (your own data) has already
fetched it, the first `panini-dr run` downloads Qwen3-Embedding-8B (~16 GB)
before its first question. Run it once with network access, or fetch the model
beforehand with `hf download Qwen/Qwen3-Embedding-8B`.

The wall times are for 32 questions in parallel against one LLM server on two
GPUs. They depend mostly on the LLM server's throughput. On BrowseComp-Plus a
question used a median of 1.04 M prompt tokens and 36 k completion tokens; about
two thirds of the submitted prompt tokens are prefixes vLLM has already cached,
so keep vLLM's prefix caching on.

## 2. Serve the models

Install vLLM 0.24.0 (LLM) and vLLM 0.26.0 (reranker) in two environments of
their own, as in [serve/README.md](serve/README.md#install-vllm), and run each
script with its environment's `bin/` first on `PATH`:

```bash
CUDA_VISIBLE_DEVICES=0,1 serve/gpt_oss_120b.sh      # gpt-oss-120b, tensor parallel 2, 65,536-token context
CUDA_VISIBLE_DEVICES=2   serve/jina_reranker.sh     # jina-reranker-v3.5, 0.45 of the GPU (about 22 GB of 48 GB)
export PANINI_LLM_BASE_URL=http://localhost:8000/v1
export JINA_RERANK_URL=http://localhost:8090/v1/rerank
export CUDA_VISIBLE_DEVICES=2                       # panini-dr's query encoder, next to the reranker
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"   # once, with network
```

`panini-dr run` loads the Qwen3-Embedding-8B query encoder itself (about
14.6 GiB), on the visible GPU with the most free memory; `CUDA_VISIBLE_DEVICES`
chooses it. Each `panini-dr` process loads its own copy, so do not start two
runs on one GPU unless it has room for both (see
[the query encoder's GPU](serve/README.md#the-query-encoders-gpu)). See
[serve/README.md](serve/README.md) for health checks.

Settings that change behaviour if you ignore them:

- **Context length.** Keep the LLM server at `--max-model-len 65536`. The
  article reader sizes its window for it.
- **tiktoken.** The reader counts tokens with tiktoken's `o200k_base` encoding.
  If the encoding cannot be loaded (for example offline and not cached), the
  reader falls back to a character estimate and reads different amounts of
  text.
- **Reranker.** If a rerank request fails, search falls back to fusion order,
  which changes results. Every command below passes `--strict-rerank`, which
  aborts the run on the first failure. `run_health.json` counts reranker calls
  and failures; the paper's runs had no failures.
- **Configuration.** `panini-dr run` checks the resolved configuration against
  `configs/paper.json` and refuses to start if they differ. Never pass `--set`
  or `--allow-config-drift` for a reproduction. The environment variables
  `GSW_PLANNER_TARGET_HUB_MODE` and `GSW_STRICT_TARGET_HUB_GATE` must be unset
  (the run refuses to start otherwise). Variables that switched code which has
  since been removed (listed under `env_vars` in `configs/removed_features.json`,
  for example `GSW_READ_TITLE_FALLBACK`) are always rejected.

Every run uses the paper configuration: 32 questions in parallel, seed 42,
temperature 0, medium reasoning effort, up to 12 orchestrator turns and 20
researcher turns, up to 8 researchers dispatched at once, and a
re-attempt cap of 5 (see `configs/paper.json`).

`panini-dr run --print-config` prints all of it. The configuration keys are
under `config` (for example `llm_seed`, `llm_temperature`, `reasoning_effort`,
and `gsw_edge_grounded_cap` for the re-attempt cap). The turn limits are not
configuration keys but the adapter's built-in defaults, printed under
`pinned_adapter_defaults` (the run checks them against the adapter and refuses
to start if one differs):

| Key | Value | Limits |
|---|---|---|
| `orchestrator_max_turns` | 12 | orchestrator turns per question |
| `level_max_turns` | 20 | turns per researcher |
| `reader_max_turns` | 8 | turns per article-reader call |
| `MAX_PARALLEL_RESEARCHERS` | 8 | researchers dispatched at once |

`config.max_turns: 50` is none of these. It limits only the single-agent
fallback that a question falls back to when its plan cannot be built or
ordered; the paper configuration's normal path never reads it.

## 3. Get the grading code and choose a judge

The graders run the benchmark authors' own grading code: the BrowseComp-Plus
grading template (also used for FRAMES) and MoNaCo's official LLM judge. This
repository does not copy it. Fetch it once:

```bash
python scripts/fetch_grader_sources.py --dest third_party    # needs git and network
export BCP_REPO=$PWD/third_party/BrowseComp-Plus              # pinned to the commit the paper used
export MONACO_JUDGE_DIR=$PWD/third_party/monaco
```

The script prints whether the MoNaCo judge matches the version the paper used;
`panini-dr score` refuses a different version unless you pass
`--allow-judge-mismatch`.

The paper grades with an LLM judge, 3 passes per question: majority vote for
BrowseComp-Plus and FRAMES, maximum F1 over the passes for MoNaCo. Every
question stays in the denominator; a question with no answer or no verdict
scores 0. Choose your judge model and endpoint:

```bash
export PANINI_JUDGE_MODEL=<judge model>
export PANINI_JUDGE_BASE_URL=<OpenAI-compatible endpoint>/v1   # or OPENAI_BASE_URL
export PANINI_JUDGE_API_KEY=<key>                               # or OPENAI_API_KEY
```

and pass `--backend openai` (the default, `bedrock`, uses the AWS Bedrock
Responses API with the region in `PANINI_BEDROCK_REGION`). With `--backend
openai`, `--judge-api chat` (the default) sends Chat Completions requests and
`--judge-api responses` sends Responses API requests; pick the one your judge
model and endpoint serve. A different judge
model moves the numbers somewhat. Verdicts are cached in the `--cache` file,
keyed by judge model and graded answer, so re-running `score` on the same run
does not call the judge again.

`panini-dr score` also reads them from `./.env`; variables set in the shell win.

## 4. BrowseComp-Plus CF-150

### Prepare the data

Everything is rebuilt from Tevatron's public releases. The queries and answers
are distributed encrypted, with the request that they never appear as plain
text online. Decrypt them with BrowseComp-Plus's own script, which needs the
Hugging Face `datasets` package (the `datasets` extra installs it, so the
PANINI-DR environment can run it). The checkout from step 3 contains the script:

```bash
cd third_party/BrowseComp-Plus
python scripts_build_index/decrypt_dataset.py --output data/browsecomp_plus_decrypted.jsonl
cd -
```

Then:

```bash
export PANINI_DATA_DIR=$PWD/data
D=$PANINI_DATA_DIR
python datasets/bcp/prepare.py corpus      # browsecomp_plus/HYBRID_BCP.json (100,195 documents)
python datasets/bcp/prepare.py questions --decrypted third_party/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl
python datasets/bcp/prepare.py dense       # browsecomp_plus/bcp_dense_cache/ (Tevatron's Qwen3-Embedding-8B vectors)
python datasets/bcp/prepare.py bm25        # browsecomp_plus/indexes/bm25/ (Tevatron's Lucene index)
```

`questions --bcp-repo third_party/BrowseComp-Plus` runs the decrypt script for
you, with the Python that runs `prepare.py`.
`all` runs the four steps in order. Never commit or publish the decrypted files
or `browsecomp_plus/browsecomp_plus_830.json`. Querying the Lucene index needs a
full JDK 21 on `JAVA_HOME`:

```bash
sudo apt install openjdk-21-jdk          # or, without root: conda install -c conda-forge openjdk=21
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
```

### Run

```bash
panini-dr run --strict-rerank --subset configs/splits/cf_bcp150.json \
    --articles-cache "$D/browsecomp_plus/HYBRID_BCP.json" \
    --dense-cache "$D/browsecomp_plus/bcp_dense_cache" \
    --lucene-index "$D/browsecomp_plus/indexes/bm25" \
    --output-dir runs/bcp
```

The run of record ran as two legs with the same configuration,
`configs/splits/cf_bcp_capped.json` (85 questions) and
`configs/splits/cf_bcp_cap5_rest.json` (65 questions); together they are
`cf_bcp150.json`. To mirror it, run the two split files and pass both run
directories to `score` (`--run` is repeatable).

### Grade

```bash
panini-dr score answers -b bcp --run runs/bcp/<run dir> \
    --subset configs/splits/cf_bcp150.json \
    --cache judge/bcp_cache.json --out judge/bcp_verdicts.json --backend openai

panini-dr score obs -b bcp --run runs/bcp/<run dir> --verdicts judge/bcp_verdicts.json \
    --subset configs/splits/cf_bcp150.json --corpus "$D/browsecomp_plus/HYBRID_BCP.json"
```

`score answers` prints the accuracy over all 150 questions and the counts per
difficulty bucket (`configs/splits/cf_bcp150_buckets.json`, 50 questions each).
`score obs` reads the relevance judgements from `$BCP_REPO/topics-qrels/`.

| | Paper |
|---|---|
| Accuracy | 73.3% (110/150): easy 46/50, medium 38/50, hard 26/50 |
| Obs. | 70.7% |

## 5. FRAMES-824

### Prepare the data

```bash
export PANINI_DATA_DIR=$PWD/data
D=$PANINI_DATA_DIR
python datasets/frames/prepare.py questions   # frames/frames_split.json from google/frames-benchmark
python datasets/frames/prepare.py corpus      # frames/HYBRID_FRAMES824.json + its dense index
```

The questions come from the official release and are converted exactly as for
the paper (ids `"0"` to `"823"`). The corpus is the paper's frozen corpus: the
2,513 Wikipedia articles linked by FRAMES, as fetched in 2026, with their
Qwen3-Embedding-8B vectors. It cannot be rebuilt from today's Wikipedia, so it
is installed from the public Hugging Face dataset
`roychowdhuryresearch/panini-dr-frames-corpus` (0.6 GB download), with every
file checked against a sha256 pinned in the script. See
[datasets/README.md](datasets/README.md).

### Run

```bash
panini-dr run --strict-rerank --subset configs/splits/frames_full824_norm.json \
    --articles-cache "$D/frames/HYBRID_FRAMES824.json" --output-dir runs/frames
```

The dense index is found automatically under `$PANINI_DATA_DIR/retrieval_cache/`
(the directory name is a fingerprint of the corpus).

### Grade

```bash
panini-dr score answers -b frames --run runs/frames/<run dir> \
    --subset configs/splits/frames_full824_norm.json \
    --cache judge/frames_cache.json --out judge/frames_verdicts.json --backend openai

panini-dr score obs -b frames --run runs/frames/<run dir> --verdicts judge/frames_verdicts.json \
    --subset configs/splits/frames_full824_norm.json --corpus "$D/frames/HYBRID_FRAMES824.json"
```

FRAMES is graded with the same template as BrowseComp-Plus (majority of 3).

| | Paper |
|---|---|
| Accuracy | 78.5% (647/824) |
| Obs. | 66.1% |

## 6. MoNaCo-1315

### Prepare the data

MoNaCo's release (`allenai/MoNaCo_Benchmark`) and the paper's MoNaCo corpus are
both gated on Hugging Face. Accept MoNaCo's terms on both dataset pages, then log
in:

```bash
hf auth login                                 # or: export HF_TOKEN=...
export PANINI_DATA_DIR=$PWD/data
D=$PANINI_DATA_DIR
python datasets/monaco/prepare.py questions   # monaco/monaco_full1315.json
python datasets/monaco/prepare.py corpus      # monaco/HYBRID_MONACO_full_v2.json + its dense index
```

`questions` builds the question file from the official release and prints
whether it matches the paper's file byte for byte. If the release archive is
password-protected, pass its password with `--zip-password`. MoNaCo's authors
ask that their data is not redistributed online: never commit or publish
`monaco_full1315.json`.

`corpus` installs the paper's frozen corpus (36,508 Wikipedia pages fetched in
2026, with their vectors) from `roychowdhuryresearch/panini-dr-monaco-corpus`
(5.3 GB download, 21 GB installed), checking every file against a pinned
sha256.

### Run

```bash
panini-dr run --strict-rerank --subset configs/splits/monaco_full1315.json \
    --articles-cache "$D/monaco/HYBRID_MONACO_full_v2.json" --output-dir runs/monaco
```

The host needs about 45-60 GB of RAM (dense index and BM25 in memory).

### Grade

```bash
panini-dr score answers -b monaco --run runs/monaco/<run dir> \
    --subset configs/splits/monaco_full1315.json \
    --cache judge/monaco_cache.json --out judge/monaco_verdicts.json --backend openai

panini-dr score obs -b monaco --run runs/monaco/<run dir> --verdicts judge/monaco_verdicts.json \
    --subset configs/splits/monaco_full1315.json --corpus "$D/monaco/HYBRID_MONACO_full_v2.json"
```

| | Paper |
|---|---|
| Mean F1 | 0.549 |
| Obs. | 0.458 |

The F1 is MoNaCo's official judge, applied item by item, with the maximum over
3 passes per question and the mean over all 1315 questions (blank answers score
0). `score answers` also prints the mean over passes, which is about 0.011
lower on the same verdicts; compare runs on the max-of-3 number.

## Run-to-run variation

Two live runs of the same configuration give different answers on some
questions, for three reasons:

1. **The LLM server.** vLLM is not bit-deterministic, even at temperature 0 with
   a fixed seed. A small difference early in an investigation changes what is
   searched next.
2. **Python's string-hash seed.** It changes between processes, and with it the
   order in which some plan steps are shown to the orchestrator (they are kept
   in a set).
3. **Shared reads.** Parallel researchers in one question share a read cache, so
   timing decides some reuse.

On BrowseComp-Plus CF-150, repeated runs of this configuration scored between 99
and 110 of 150; the paper's run-stability table reports three of them (110, 107
and 110). A fresh run of this release (one 96 GB GPU for the LLM) scored 106 of
150 (easy/medium/hard 46/35/25). Paired with the paper's run, 93 questions were
correct in both, 13 only in the new run and 17 only in the paper's run
(difference −2.7 points, 95% interval −9.3 to +4.7). On FRAMES-824 we estimate a
standard deviation of about 10 questions
(about 1.2 points). The judge adds its own variation, and a different judge
model moves the numbers further. Results within these ranges are consistent
with the paper.

## Known issues

- **Shared code-generation state across concurrent questions.** At the start of
  each question, the orchestrator stores that question's code-generation runtime
  (it keeps the question text for a years-versus-days cue in date arithmetic) and
  its cascade-block list on the adapter object that all concurrent questions
  share. With 32 questions in parallel, a question can therefore use another
  question's runtime and list. The paper's runs had the same behaviour, so this
  release keeps it in order to reproduce them. It will be fixed in a later
  release, together with a measured re-run.
- **Fixed in this release: a planning hand-off race.** In the research code, the
  hand-off from a question's planning phase to its orchestrator loop was also
  stored on the shared adapter, so one question could, in principle, continue
  another question's planning conversation. This release keeps it per question.
  We checked all 2,289 traces behind the paper's numbers: the race never
  occurred in them, so the fix does not change any paper result.

## Checking the code instead of the numbers

`panini-dr verify` removes all three sources of variation: it replays recorded
responses of the LLM, the encoder and the reranker, fixes the hash seed to the
recorded one, and releases each question's responses in the recorded event
order. That is why it can demand byte-identical answers, traces and
configuration. It needs the FRAMES data from section 5 and the `dense` extra,
but no GPU, no model server and no network. It takes about a minute on CPU.

```bash
hf download roychowdhuryresearch/panini-dr-frames-corpus --repo-type dataset \
    --include "verify/*" --local-dir panini-dr-verify
panini-dr verify --bundle panini-dr-verify/verify --data-dir "$PANINI_DATA_DIR"
```

Expected result: `PASS: 8/8 questions reproduce the reference outputs exactly`
and exit status 0. Exit status 1 means a difference (the report shows the first
request that differs; `--keep-output DIR` keeps the replay outputs and
per-miss reports). Exit status 2 means the bundle or the data cannot be used,
for example a corpus that is not byte-identical, or tiktoken's `o200k_base` not
cached; nothing is replayed in that case. See the
[README](README.md#check-this-code-against-the-papers-runs-panini-dr-verify)
for what the check does and does not show.

## Recomputing scores from existing outputs

`panini-dr score answers --offline` grades from the judge cache only: it never
builds a judge backend, and a verdict missing from the cache is an error (exit
status 2). Anyone who has a run's output directory and the judge verdicts for it
can therefore recompute the official scores and Obs. without calling a judge.
`panini-dr score import-legacy-cache` converts verdicts from the research code's
cache format into this repository's cache, after checking each one against the
run's own answer.

We do not distribute benchmark outputs. Traces contain the questions and gold
answers, and BrowseComp-Plus and MoNaCo ask that theirs are not published (see
[NOTICE](NOTICE)); the only recorded outputs we publish are the eight FRAMES
questions of the verify bundle. To reproduce the numbers, take the full path
above: prepare the data, serve the models, run, and grade.
