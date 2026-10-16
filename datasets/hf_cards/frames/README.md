---
license: cc-by-sa-4.0
language:
- en
pretty_name: PANINI-DR FRAMES-824 corpus
size_categories:
- 1K<n<10K
task_categories:
- text-retrieval
- question-answering
tags:
- wikipedia
- retrieval
- deep-research
- frames
viewer: false
---

# PANINI-DR FRAMES-824 corpus

This is the frozen Wikipedia corpus, and its dense retrieval index, that
PANINI-DR searched in the FRAMES experiments of the paper *PANINI-DR:
Controlled Deep Research through Generative Semantic Workspaces*. It holds the
2,513 English Wikipedia articles linked by the 824 FRAMES questions, as they
were fetched in 2026, so that the paper's numbers can be reproduced without
re-fetching Wikipedia, which has changed since.

**It contains no benchmark questions, answers or gold annotations.** The FRAMES
questions come from the official release (`google/frames-benchmark`,
Apache-2.0); the PANINI-DR repository downloads and converts them.

## Contents

| Path | Size | What it is |
|---|---|---|
| `frames/HYBRID_FRAMES824.json.zst` | 29.7 MB (118.3 MB uncompressed) | The corpus: one JSON object `{title: text}` with 2,513 articles (116.0 M characters), zstd-compressed |
| `retrieval_cache/Qwen__Qwen3-Embedding-8B__99c7a0f7ddf9__efb84805d53b/chunk_embeddings.npy.zst` | 575 MB (1.14 GB uncompressed) | Qwen3-Embedding-8B vectors of the 69,608 chunks: float32, shape (69608, 4096), L2-normalised |
| `retrieval_cache/Qwen__Qwen3-Embedding-8B__99c7a0f7ddf9__efb84805d53b/chunk_ids.pkl` | 1.7 MB | The chunk id of every row (a pickled list of strings) |
| `provenance/articles.jsonl` | 0.7 MB | One line per article: title, URL, page id, revision id, revision permalink, revision timestamp, length |
| `manifest.json` | | Machine-readable description: counts, chunking, sizes and sha256 of every file, build dates |
| `SHA256SUMS` | | sha256 of every file above except this card |
| `verify/` | about 33 MB | The `panini-dr verify` bundle: recorded runs of the paper configuration on 8 FRAMES questions (two cases). `cassettes/` holds every LLM, query-embedding and reranker response of those runs, `reference/` the outputs they wrote, `subsets/` their question lists, and `manifest.json` the sha256 of every bundle file and of the corpus, dense cache and question file it needs. `panini-dr verify` replays the runs offline (no GPU, no model server, no network) against an installation of this corpus and checks that the answers, traces and run configuration match byte for byte. It is not part of the corpus installation: download it with `--include "verify/*"`. |

The sha256 of each file, compressed and uncompressed, is listed in
`manifest.json`, and the PANINI-DR install script checks every byte against
values pinned in its own source.

| File | sha256 |
|---|---|
| `HYBRID_FRAMES824.json` (uncompressed) | `fc3015d31cb98eba5d367165160ba5c227d8801600b7ee7f2b97be227caf7e20` |
| `chunk_embeddings.npy` (uncompressed) | `0a90ad9879b4686368d8cd161cfd7cab5574155cd679f9b84420c3cb6505a47a` |
| `chunk_ids.pkl` | `b488cd4e54c9061a926ef0b1fb4ef1d0fff318c1ebd65926966635832d28b877` |
| `faiss.index` (rebuilt locally, not shipped) | `d7ab3596f41ec9ce9d72db982149050c5a3e50836097ba291192fc4edd0a3eec` |

## How PANINI-DR uses it

Install it with the PANINI-DR repository's preparation script, which downloads
the files, checks them, decompresses them and rebuilds the FAISS index:

```bash
export PANINI_DATA_DIR=$PWD/data
python datasets/frames/prepare.py questions   # FRAMES questions from google/frames-benchmark
python datasets/frames/prepare.py corpus      # this dataset
panini-dr run --strict-rerank --subset configs/splits/frames_full824_norm.json \
    --articles-cache "$PANINI_DATA_DIR/frames/HYBRID_FRAMES824.json" --output-dir runs/frames
```

The corpus lands in `$PANINI_DATA_DIR/frames/` and the dense index in
`$PANINI_DATA_DIR/retrieval_cache/<fingerprint>/`, where `panini-dr run` looks
for it. The directory name is a fingerprint of the corpus (a hash of the sorted
titles and of every article's length), so the run finds the index from the
corpus alone and does not pick up an index built for a different corpus.

At run time PANINI-DR:

- splits each article into chunks of at most 2,000 characters with a
  200-character overlap, packing whole paragraphs where it can (69,608 chunks,
  ids `<title>__<i>`);
- searches them with a hybrid retriever: an Okapi BM25 index built in memory at
  start-up, fused by reciprocal rank with exact inner-product search over these
  vectors, then reranked;
- reads whole articles from the same corpus when the agent opens a page.

`faiss.index` (an exact `IndexFlatIP`) is not shipped because it is a byte copy
of the embedding matrix; the script rebuilds it on CPU in seconds, and with
faiss 1.13.2 the result is identical to the file the paper's runs used.

### Without the script

```bash
zstd -d frames/HYBRID_FRAMES824.json.zst
zstd -d retrieval_cache/*/chunk_embeddings.npy.zst
```

```python
import json, pickle, numpy as np
corpus = json.load(open("frames/HYBRID_FRAMES824.json", encoding="utf-8"))  # {title: text}
vectors = np.load("retrieval_cache/Qwen__Qwen3-Embedding-8B__99c7a0f7ddf9__efb84805d53b/chunk_embeddings.npy", mmap_mode="r")
ids = pickle.load(open("retrieval_cache/Qwen__Qwen3-Embedding-8B__99c7a0f7ddf9__efb84805d53b/chunk_ids.pkl", "rb"))
```

## How it was made

- **Pages.** The English Wikipedia pages linked by the 824 FRAMES questions.
  Links that redirect were resolved to the target page, so a few titles differ
  from the links in FRAMES.
- **Text.** The Parsoid HTML of each page's then-current revision was fetched
  from the Wikimedia REST API and converted to text: the title on the first
  line, then an `[INFOBOX]` block of `key | value` lines, the lead, `## `
  section headings, `- ` list items, `a | b` table rows and a `## References`
  section. The corpus was built on 2026-07-20.
- **Revisions.** These are live revisions, not a dump with a cutoff date:
  revision timestamps run from 2018-03-07 to 2026-07-20 (median 2026-04-29).
  Every article's revision id and permalink are in `provenance/articles.jsonl`.
  The exact fetch times were not recorded; all fetches were on or before
  2026-07-20.
- **Vectors.** Each chunk was encoded with `Qwen/Qwen3-Embedding-8B`
  (sentence-transformers, fp16 on GPU, normalised, no instruction prefix for
  documents; queries use the model's retrieval instruction) and saved as
  float32. Built on 2026-07-20 (US Pacific time).

The fetch-and-parse driver of the original build was not kept, and Wikipedia
has changed since, so the corpus cannot be rebuilt byte for byte; it is
published instead. The revision ids make it possible to re-fetch the same
revisions.

## Licence and attribution

The article text is from Wikipedia and was written by Wikipedia contributors.
It is licensed under the [Creative Commons Attribution-ShareAlike 4.0
licence](https://creativecommons.org/licenses/by-sa/4.0/). The revisions used
are listed in `provenance/articles.jsonl`, each with a permalink; the page
history on Wikipedia lists the authors. The HTML was converted to plain text
automatically, with the markers described above; the text was not edited by
hand. The derived files (chunk ids and embedding vectors) are released under the
same licence. If you redistribute the corpus or something derived from its
text, keep the attribution and the share-alike terms.

FRAMES itself is a separate release by its authors (`google/frames-benchmark`,
Apache-2.0); please cite it if you use this corpus with its questions.

## Citation

```bibtex
@article{turali2026paninidr,
  title   = {{PANINI-DR}: Controlled Deep Research through Generative Semantic Workspaces},
  author  = {Turali, M. Yigit and Rajesh, Shreyas and Lorasdagi, M. Efe and Duan, Chenda and Sharma, Kartik and Roychowdhury, Vwani},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2026}
}
```
