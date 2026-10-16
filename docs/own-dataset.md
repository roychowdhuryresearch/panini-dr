# Running PANINI-DR on your own dataset

This guide turns your own questions and documents into a dataset that
`panini-dr run` can search, runs the paper configuration on it, and grades the
answers. A tiny made-up example is in [`examples/toy/`](../examples/toy/); the
commands below work on it too.

What you need:

- the LLM and reranker servers from [serve/](../serve/README.md), with
  `PANINI_LLM_BASE_URL` and `JINA_RERANK_URL` set;
- a GPU with about 16 GB free on the machine that runs `panini-dr`, for the
  Qwen3-Embedding-8B encoder (it builds the dense index, and encodes every
  search query during a run). It loads on the visible GPU with the most free
  memory; choose the GPU with `CUDA_VISIBLE_DEVICES`, and do not run two
  `panini-dr` processes on one GPU that has room for only one encoder (see
  [the query encoder's GPU](../serve/README.md#the-query-encoders-gpu));
- the `dense` extra. The README's install line includes it; to add it to an
  environment without removing the other extras, run
  `uv sync --locked --inexact --extra dense` (see the
  [README](../README.md#install): `uv sync` without `--inexact` removes every
  extra you do not name).

## 1. Write the files

### `questions.jsonl`

One JSON object per line:

```json
{"id": "toy-1", "question": "Which river flows through the birthplace of the composer of the opera The Glass Orchard?", "answer": "Varno River"}
{"id": "toy-3", "question": "Which towns does the Varno River flow through?", "answer": ["Tessby", "Orlin", "Calder Ford"], "answer_type": "set"}
```

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Unique id (a number is turned into a string). |
| `question` | yes | The question text. This is the only thing the agent sees. |
| `answer` | yes | The gold answer: a string, or a list. A list can hold strings, or rows (lists) for answers with several columns. |
| `answer_type` | no | `short` or `set`. It picks the grader (see below). Default: `short` for a string answer, `set` for a list. |

Any other keys are copied unchanged into the prepared file. The agent never
sees `answer` or `answer_type`. The run uses `answer` only for the string-match
diagnostics in `cell_result.json`; the real grading is `panini-dr score`.

- **`short`**: one answer, graded correct or incorrect by an LLM judge with the
  BrowseComp-Plus grading template (3 passes, majority vote). This is how the
  paper grades BrowseComp-Plus and FRAMES. A list given as a short answer is
  joined with `"; "`.
- **`set`**: a collection of items, graded with MoNaCo's official LLM judge,
  which matches items and gives precision, recall and F1 (3 passes, maximum
  F1). This is how the paper grades MoNaCo. A string given as a set answer
  becomes a one-item list.

The agent does not see `answer_type`, so write set questions so that they ask
for a list ("Which towns ...", "List all ...").

### `docs.jsonl`

One JSON object per line:

```json
{"id": "d1", "title": "Varno River", "text": "Varno River\n[INFOBOX]\nLength | 212 km\n..."}
```

| Key | Required | Meaning |
|---|---|---|
| `text` | yes | The document text (not empty). |
| `title` | one of `title` / `id` | The document's name. The agent sees it in search results, opens documents by it and cites it. |
| `id` | one of `title` / `id` | A unique id (`doc_id` is accepted too). |

Each document gets a key, which is its name everywhere in the run:

- its `title`, when the title is present and no other document has it;
- its `id`, when it has no title;
- `"<title> [<id>]"`, when several documents share a title (they then need ids).

Plain text works. The prompts were written for Wikipedia-style articles: an
optional `[INFOBOX]` block of `key | value` lines, `## ` section headings and
`- ` list items, with blank lines between paragraphs. Other text works too, but
the paper did not evaluate it.

The lexical search (BM25) keeps only ASCII letters and digits as terms. Words
with other characters (accented letters, non-Latin scripts) are split or lost,
so such documents get few or no BM25 terms (`prepare` warns about documents
with none). Dense search still works for them.

### `dataset.yaml` (optional)

Instead of passing the files on the command line, you can describe the dataset
in a small YAML file. Paths are relative to the YAML file:

```yaml
name: toy
description: Eight synthetic encyclopedia-style articles and three questions about them.
questions: questions.jsonl
docs: docs.jsonl
chunking: paragraph
```

Only these five keys are allowed. Options given on the command line win.

## 2. Prepare the dataset

```bash
panini-dr prepare --questions questions.jsonl --docs docs.jsonl --out data/mydata
# or
panini-dr prepare --config examples/toy/dataset.yaml --out data/toy
```

`prepare` checks both files (a bad line stops it with the file name and line
number), writes the dataset directory, and encodes every chunk with
Qwen3-Embedding-8B.

| Option | Default | Meaning |
|---|---|---|
| `--questions`, `--docs` | from `dataset.yaml` | The two JSONL files. |
| `--out` | (required) | The dataset directory to write. |
| `--config` | | A `dataset.yaml`. |
| `--name` | `name` in the YAML, else the name of `--out` | Dataset name, used as the subset id in run outputs. |
| `--chunking` | `paragraph` | `paragraph`: chunks of at most 2,000 characters with a 200-character overlap, packing whole paragraphs where possible (as for FRAMES and MoNaCo). `whole_doc`: one chunk per document (as for BrowseComp-Plus); suits short documents. |
| `--device` | the GPU with the most free memory | Encoder device, e.g. `cuda:0`. Without a usable GPU the encoder runs on the CPU, which is far slower. |
| `--batch-size` | 16 | Encoder batch size. |
| `--no-dense` | | Skip the dense index (no GPU needed). See below. |

The first run downloads Qwen3-Embedding-8B (about 16 GB) from Hugging Face.
Encoding time grows with the number of chunks; the 616,096-chunk MoNaCo corpus
takes hours. The vectors are float32 with 4,096 dimensions, so each chunk costs
about 16 KB on disk, and about 32 KB of RAM at run time (the vectors plus the
search index).

The dataset directory holds:

| File | Content |
|---|---|
| `corpus.json` | The `{key: text}` corpus that the run searches and reads. |
| `questions.jsonl` | A checked copy of the questions, with `answer_type` filled in. |
| `subset.json` | All question ids. |
| `dense_cache/` | `chunk_embeddings.npy`, `chunk_ids.pkl`, `faiss.index`, and `built_from.json` (the sha256 of the corpus and of the chunk-id list the index was encoded from). Not with `--no-dense`. |
| `dataset.json` | Counts, chunking, dense-index info, and the sha256 of each file. |

`panini-dr run` checks the sha256 of `corpus.json`, `questions.jsonl` and
`subset.json`, and refuses the dataset if one changed after `prepare`. It also
refuses a dense index whose `built_from.json` does not match the corpus and
its chunks. If you change your documents, re-run `prepare` with the same
`--out`: it keeps the dense index only when it was built from the same corpus
and chunking, and rebuilds it otherwise.

`prepare` ends by printing a summary; on the toy example it reports 3
questions, 8 documents and 8 chunks.

## 3. Run

```bash
panini-dr run --dataset data/mydata --output-dir runs/mydata
```

This is the paper configuration, unchanged. Useful options:

- `--limit N` runs only the first N questions (a quick check).
- `--concurrency N` sets how many questions run in parallel (default 32, as in
  the paper). Lower it if your LLM server is slow or small.
- `--strict-rerank` stops the run on the first reranker failure.
- `--model` and `--base-url` point at another OpenAI-compatible server with
  tool calling. The prompts and limits were tuned for `gpt-oss-120b`; other
  models were not evaluated.

The run writes one directory under `--output-dir`; the
[README](../README.md#what-a-run-writes) lists its files.

### Without a dense index (BM25 only)

A dataset prepared with `--no-dense` needs no GPU for the encoder, but it cannot
run the paper configuration, which uses hybrid search. `panini-dr run` refuses
it and names the override:

```bash
panini-dr prepare --config examples/toy/dataset.yaml --out data/toy-bm25 --no-dense
panini-dr run --dataset data/toy-bm25 --set retriever_type=bm25 --allow-config-drift
```

This still needs the LLM and reranker servers. The drift is recorded in the
run's `run_meta.json`; results from it are not comparable with the paper's.

## 4. Grade the answers

The graders use the benchmark authors' grading code. Fetch it once (needs git
and network) and point the graders at it:

```bash
python scripts/fetch_grader_sources.py --dest third_party
export BCP_REPO=$PWD/third_party/BrowseComp-Plus     # needed for short answers
export MONACO_JUDGE_DIR=$PWD/third_party/monaco      # needed for set answers
```

Then grade with a judge model of your choice, served by any OpenAI-compatible
endpoint:

```bash
export PANINI_JUDGE_BASE_URL=http://localhost:8001/v1   # your judge endpoint
export PANINI_JUDGE_API_KEY=...                         # if it needs one
panini-dr score answers -b byod \
    --run runs/mydata/<run dir> \
    --questions data/mydata/questions.jsonl \
    --cache judge/mydata_cache.json --out judge/mydata_verdicts.json \
    --backend openai --model <judge model>
```

- `--questions` is required for your own data: pass the prepared copy.
- `--subset data/mydata/subset.json` (or any file with a `question_ids` list)
  grades a subset; by default every question in `--questions` counts.
- Short and set questions are reported separately: accuracy over the short
  questions, mean F1 over the set questions. A question without an answer, or
  without a trace (for example after `--limit`), scores 0 and stays in the
  denominator.
- Verdicts are cached in `--cache`. Re-running only calls the judge for answers
  it has not seen; `--offline` uses the cache only and never calls a judge.
- `--out` writes every verdict and the summary as JSON.
- Every `panini-dr` command reads `./.env` by itself; variables set in the
  shell win.

On the toy example (two short questions, one set question) the output has
this shape:

```
[byod] short-answer accuracy: <correct>/2 = <percent>%  (blank <n>, no trace <n>, errors <n>)
[byod] set-answer mean F1 (max of 3 passes, over all 1): <F1>  (mean over passes <F1>; judged <n>, blank <n>, no trace <n>, errors <n>)
verdicts written to judge/mydata_verdicts.json
```

The observed-support score (`panini-dr score obs`) is defined for the three
paper benchmarks only.
