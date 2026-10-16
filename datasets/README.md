# Benchmark data

This repository ships **no benchmark questions, answers, gold annotations,
corpora or indexes**. The scripts here download or rebuild them on your machine,
into a data root: `$PANINI_DATA_DIR` (default `./data`, ignored by git), or
`--data-dir <dir>` given before the sub-command
(`python datasets/frames/prepare.py --data-dir <dir> questions`). The paper's
question-id lists (ids only) are in `configs/splits/`.

The scripts need the `datasets` and `dense` extras (and `bcp` for
BrowseComp-Plus). The README's install line already installs them all. To add
them to an environment that lacks them, without removing the extras it already
has (`uv sync` alone is exact; see the [README](../README.md#install)):

```bash
uv sync --locked --inexact --extra datasets --extra dense   # add --extra bcp for BrowseComp-Plus
```

BrowseComp-Plus's Lucene index also needs a full JDK 21 on `JAVA_HOME` at run
time (`apt install openjdk-21-jdk` or `conda install -c conda-forge openjdk=21`;
the [README](../README.md#jdk-21-browsecomp-plus-only) shows how to set
`JAVA_HOME`). The scripts read `./.env` the way `panini-dr` does (variables set
in the shell win).

For your own data, use `panini-dr prepare` instead: see
[docs/own-dataset.md](../docs/own-dataset.md).

## Commands

| Benchmark | Command | Writes (under the data root) |
|---|---|---|
| BrowseComp-Plus | `python datasets/bcp/prepare.py corpus` | `browsecomp_plus/HYBRID_BCP.json` |
| | `python datasets/bcp/prepare.py questions --decrypted <file>` (or `--bcp-repo <checkout>`) | `browsecomp_plus/browsecomp_plus_830.json`, `browsecomp_plus_830_subset.json` |
| | `python datasets/bcp/prepare.py dense` | `browsecomp_plus/bcp_dense_cache/` |
| | `python datasets/bcp/prepare.py bm25` | `browsecomp_plus/indexes/bm25/` |
| | `python datasets/bcp/prepare.py all --decrypted <file>` | all four, in this order |
| FRAMES | `python datasets/frames/prepare.py questions` | `frames/frames_split.json`, `frames/article_index.json` |
| | `python datasets/frames/prepare.py corpus` | `frames/HYBRID_FRAMES824.json`, `retrieval_cache/<fingerprint>/` |
| MoNaCo | `python datasets/monaco/prepare.py questions` | `monaco/monaco_full1315.json` |
| | `python datasets/monaco/prepare.py corpus` | `monaco/HYBRID_MONACO_full_v2.json`, `retrieval_cache/<fingerprint>/` |

Every sub-command has `--help`. [REPRODUCE.md](../REPRODUCE.md) gives the full
sequence for each benchmark, from preparing the data to grading.

## What each paper run needs

| | BrowseComp-Plus CF-150 | FRAMES-824 | MoNaCo-1315 |
|---|---|---|---|
| Split (ids) | `configs/splits/cf_bcp150.json` (= `cf_bcp_capped` 85 + `cf_bcp_cap5_rest` 65, the two legs of the run of record) | `configs/splits/frames_full824_norm.json` | `configs/splits/monaco_full1315.json` |
| Questions (read by the run) | `browsecomp_plus/browsecomp_plus_830.json` (or `$BROWSECOMP_QUESTIONS`) | `frames/frames_split.json` | `monaco/monaco_full1315.json` (or `$MONACO_QUESTIONS`) |
| Corpus (`--articles-cache`) | `HYBRID_BCP.json`: 100,195 documents, one chunk each, ~3.3 GB | `HYBRID_FRAMES824.json`: 2,513 articles, 69,608 chunks, 118 MB | `HYBRID_MONACO_full_v2.json`: 36,508 articles, 616,096 chunks, 1.0 GB |
| Dense index | `--dense-cache browsecomp_plus/bcp_dense_cache` (Tevatron's vectors, 3.3 GB) | `retrieval_cache/Qwen__Qwen3-Embedding-8B__99c7a0f7ddf9__efb84805d53b/` (1.1 GB, found automatically) | `retrieval_cache/Qwen__Qwen3-Embedding-8B__7ea9a4d678f8__de3db946f2e8/` (10.1 GB, found automatically) |
| Lexical index | `--lucene-index browsecomp_plus/indexes/bm25` (Tevatron's Lucene index, 2.1 GB) | Okapi BM25, built in memory at start-up | Okapi BM25, built in memory at start-up |

`panini-dr run` finds the dense index of FRAMES and MoNaCo at
`$PANINI_DATA_DIR/retrieval_cache/<fingerprint>`, where the fingerprint is
computed from the corpus (`src/panini_dr/retrieval/dense.py`). Run with the same
`PANINI_DATA_DIR` you prepared into, or pass `--dense-cache <dir>`. A run never
builds a missing dense index (for MoNaCo that takes hours on a GPU); it stops
with `DenseCacheMissingError` instead.

Example (FRAMES):

```bash
export PANINI_DATA_DIR=$PWD/data
python datasets/frames/prepare.py questions
python datasets/frames/prepare.py corpus
panini-dr run --strict-rerank --subset configs/splits/frames_full824_norm.json \
    --articles-cache "$PANINI_DATA_DIR/frames/HYBRID_FRAMES824.json"
```

## The published FRAMES and MoNaCo corpora

The FRAMES-824 and MoNaCo-1315 corpora are the union of each benchmark's gold
Wikipedia pages, fetched as **live revisions in 2026** (FRAMES revisions up to
2026-07-20; MoNaCo fetched on 2026-07-22/23 and 2026-08-03) and converted to the
`{title: text}` format (`[INFOBOX]` block, `## ` sections, `- ` lists). The
fetch script was not kept and Wikipedia has changed since, so they cannot be
rebuilt byte for byte. They are published instead, with their dense indexes, as
two Hugging Face datasets. Their ids are defined in one place,
`_prep_common.HF_CORPUS_REPOS`:

| | Repo | Access | Download | Installed |
|---|---|---|---|---|
| FRAMES | `roychowdhuryresearch/panini-dr-frames-corpus` | public | 0.6 GB | 2.4 GB |
| MoNaCo | `roychowdhuryresearch/panini-dr-monaco-corpus` | **gated**: accept MoNaCo's terms on the repo page, then `hf auth login` or `export HF_TOKEN=...` | 5.3 GB | 21.2 GB |

Neither contains benchmark questions, answers or gold annotations. Their
dataset cards are in `datasets/hf_cards/`.

Repo layout (the installed files keep the same paths under the data root):

```
<benchmark>/<corpus>.json.zst                            corpus, zstd
retrieval_cache/<fingerprint>/chunk_embeddings.npy.zst   Qwen3-Embedding-8B vectors, zstd
retrieval_cache/<fingerprint>/chunk_ids.pkl
provenance/articles.jsonl   manifest.json   SHA256SUMS   README.md   (not installed)
verify/                     FRAMES only: the `panini-dr verify` bundle (not installed)
```

`corpus` downloads only the three installed files. It checks their size and
sha256 against values pinned in `datasets/<benchmark>/prepare.py` (never against
a checksum file that comes with the download), decompresses the `.zst` files
(with the `zstandard` package if installed, otherwise the `zstd` command),
checks the decompressed bytes again, and reads `chunk_ids.pkl` without allowing
any pickled object other than strings. It then recomputes the dense-index
fingerprint from the corpus with `panini_dr`'s own code and rebuilds
`faiss.index` from the embeddings on CPU (with faiss 1.13.2 the result is
byte-identical to the paper's; the step warns otherwise). Files that are already
installed with the right bytes are kept, so the step can be re-run.

```bash
export PANINI_DATA_DIR=$PWD/data
python datasets/frames/prepare.py corpus                     # public
hf auth login                                                # MoNaCo: after accepting its terms
python datasets/monaco/prepare.py corpus
```

Options of `corpus`: `--from-local DIR` installs from a copy laid out like the
repo (DIR, or DIR/<repo name>/), without network; `--revision` pins a repo
revision; `--download-dir` and `--keep-download` control where the compressed
files go (default `<data root>/_downloads/<repo name>/`, deleted after a
successful install); `--verify-chunks` also chunks the corpus and checks every
chunk id against the dense index.

The verify bundle is downloaded separately:

```bash
hf download roychowdhuryresearch/panini-dr-frames-corpus --repo-type dataset \
    --include "verify/*" --local-dir panini-dr-verify
panini-dr verify --bundle panini-dr-verify/verify --data-dir "$PANINI_DATA_DIR"
```

## Notes per benchmark

**BrowseComp-Plus.** Everything is rebuilt from Tevatron's public releases
(`Tevatron/browsecomp-plus`, `-corpus`, `-indexes`). Queries and answers are
distributed **encrypted**, with the canary "BENCHMARK DATA SHOULD NEVER APPEAR
AS PLAIN TEXT ONLINE". Decrypt them with BrowseComp-Plus's own
`scripts_build_index/decrypt_dataset.py` (texttron/BrowseComp-Plus; the paper
used commit `046949032b03`, which `scripts/fetch_grader_sources.py` checks out)
and pass the result with `--decrypted`. That script needs the Hugging Face
`datasets` package, which the `datasets` extra installs. `--bcp-repo <checkout>`
runs it for you with the Python that runs `prepare.py`. Never commit or publish
the decrypted file or
`browsecomp_plus_830.json`.

Document keys are `"<front-matter title> [<docid>]"` (or the bare docid),
computed by `panini_dr.eval.browsecomp_keys.cache_key` for both the corpus and
the questions' `gold_pages`. The dense vectors are Tevatron's
Qwen3-Embedding-8B document vectors (encoded with `--passage_max_len 4096`),
relabelled `"<key>__0"` because BrowseComp-Plus runs with one chunk per
document. `--parquet <file or dir>` (before the sub-command) uses a local copy of
the corpus parquet instead of downloading it. If the corpus is sharded into
several parquet files, they are read in file-name order (document order does not
affect retrieval).

**FRAMES.** Questions come from `google/frames-benchmark` (`test.tsv`,
Apache-2.0) and are converted exactly as for the paper (ids `"0"` to `"823"`,
`original_prompt`). `questions --tsv <file>` converts a local copy.
`--dev-sample N` also writes `frames_dev.json` for `panini-dr run --split dev`.

**MoNaCo.** Questions come from `allenai/MoNaCo_Benchmark` (gated, ODC-BY; its
card asks that the unzipped data files are not redistributed online, and it
carries a canary). Accept its terms and log in first, or pass a local copy with
`--release`. The script that produced the paper's question file was not kept;
this builder was rebuilt and checked against that file. From
`monaco_version_1_release.json` and the release's `execution_traces/` it writes
the paper's `monaco_full1315.json` byte for byte, and prints whether the sha256
matches. It emits `{id: "monaco_<ex_num>", question, answer: validated_answer,
hop: "<steps>step", gold_pages}`, where `gold_pages` are the English-Wikipedia
titles in the traces' `source_url` fields (pass `--traces` if they are not next
to the release file). It checks that every id of
`configs/splits/monaco_full1315.json` is present. `gold_pages` are not read
during a run; the Obs. scorer uses them only for retrieval diagnostics.

## Sizes and hardware

| | Disk at run time | Notes |
|---|---|---|
| BrowseComp-Plus | ~8.5 GB (corpus 3.3 + dense 3.3 + Lucene 2.1); ~14 GB while building | No GPU needed to prepare (Tevatron's vectors are reused). |
| FRAMES | ~2.4 GB (corpus 0.12 + dense 1.1 + FAISS 1.1) | 0.6 GB download. |
| MoNaCo | ~21 GB (corpus 1.0 + dense 10.1 + FAISS 10.1); ~27 GB while installing | 5.3 GB download; ~20 GB of RAM for the dense index and ~20 GB for BM25. |

At run time every benchmark needs the Qwen3-Embedding-8B query encoder on a GPU
(about 16 GB), plus the LLM and reranker servers ([serve/README.md](../serve/README.md)).

## Licences and terms

- Wikipedia text (FRAMES and MoNaCo corpora): CC BY-SA 4.0. Attribution and
  share-alike apply to the corpora and to anything derived from their text.
- FRAMES (`google/frames-benchmark`): Apache-2.0.
- BrowseComp-Plus (Tevatron): code MIT; queries and answers are distributed
  encrypted and must never be published in plain text; the corpus documents are
  web pages owned by their authors.
- MoNaCo (`allenai/MoNaCo_Benchmark`): ODC-BY, with the request not to
  redistribute the unzipped data files online.

[NOTICE](../NOTICE) at the repository root summarises these.
