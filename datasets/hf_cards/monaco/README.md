---
license: cc-by-sa-4.0
language:
- en
pretty_name: PANINI-DR MoNaCo-1315 corpus
size_categories:
- 10K<n<100K
task_categories:
- text-retrieval
- question-answering
tags:
- wikipedia
- retrieval
- deep-research
- monaco
viewer: false
extra_gated_heading: Access the PANINI-DR MoNaCo corpus
extra_gated_prompt: >-
  This corpus is English Wikipedia text (CC BY-SA 4.0). The set of pages in it
  was selected by the gold annotations of the MoNaCo benchmark
  (allenai/MoNaCo_Benchmark), whose authors ask that their data is not
  redistributed online. Access is therefore limited to people who accept
  MoNaCo's terms. Please read them on the MoNaCo dataset page
  (https://huggingface.co/datasets/allenai/MoNaCo_Benchmark) before you
  request access.
extra_gated_fields:
  I have read and agree to the terms of the MoNaCo benchmark: checkbox
  I will not redistribute this corpus or its page list: checkbox
extra_gated_button_content: Agree and request access
---

# PANINI-DR MoNaCo-1315 corpus

This is the frozen Wikipedia corpus, and its dense retrieval index, that
PANINI-DR searched in the MoNaCo experiments of the paper *PANINI-DR:
Controlled Deep Research through Generative Semantic Workspaces*. It holds the
English Wikipedia pages named in the execution traces of all 1,315 MoNaCo
questions (36,508 entries, 35,460 distinct pages), as they were fetched in July
and August 2026, so that the paper's numbers can be reproduced without
re-fetching Wikipedia, which has changed since.

**Why it is gated.** The text is ordinary Wikipedia text under CC BY-SA 4.0, but
which pages are in the corpus follows MoNaCo's gold annotations, and MoNaCo's
authors ask that their data is not redistributed online. To respect that, you
must accept MoNaCo's terms (see the MoNaCo dataset page,
`allenai/MoNaCo_Benchmark`) to download this corpus, and you should not
redistribute it or its page list.

**It contains no benchmark questions, answers or gold annotations.** The MoNaCo
questions come from the official release (`allenai/MoNaCo_Benchmark`, itself
gated); the PANINI-DR repository builds its question file from that release on
your machine.

## Contents

| Path | Size | What it is |
|---|---|---|
| `monaco/HYBRID_MONACO_full_v2.json.zst` | 248 MB (1.04 GB uncompressed) | The corpus: one JSON object `{title: text}` with 36,508 entries (1.01 G characters), zstd-compressed |
| `retrieval_cache/Qwen__Qwen3-Embedding-8B__7ea9a4d678f8__de3db946f2e8/chunk_embeddings.npy.zst` | 5.07 GB (10.09 GB uncompressed) | Qwen3-Embedding-8B vectors of the 616,096 chunks: float32, shape (616096, 4096), L2-normalised |
| `retrieval_cache/Qwen__Qwen3-Embedding-8B__7ea9a4d678f8__de3db946f2e8/chunk_ids.pkl` | 15 MB | The chunk id of every row (a pickled list of strings) |
| `provenance/articles.jsonl` | 13 MB | One line per entry: key, Wikipedia title, URL, page id, revision id, revision permalink, revision timestamp, fetch time, length |
| `manifest.json` | | Machine-readable description: counts, chunking, sizes and sha256 of every file, build dates |
| `SHA256SUMS` | | sha256 of every file above except this card |

The sha256 of each file, compressed and uncompressed, is listed in
`manifest.json`, and the PANINI-DR install script checks every byte against
values pinned in its own source.

| File | sha256 |
|---|---|
| `HYBRID_MONACO_full_v2.json` (uncompressed) | `15b7d233da85abaf8667711a9c0467b2e28d2e5d8018a7ab5fa6aa9b668774f8` |
| `chunk_embeddings.npy` (uncompressed) | `80dcc74d99ca6f9e8d5b2c900d23b60c5422bff384b73f46874fd797bc6c59ee` |
| `chunk_ids.pkl` | `449aa7420597ddfdc328a63bb329038b3b82e5df66e9e579035c1fc10bfdcd20` |
| `faiss.index` (rebuilt locally, not shipped) | `a38fe300bcdb56b849b45d300e7406dd0b6c30974499deb8e586d39e24509126` |

## How PANINI-DR uses it

Accept the terms on this page, authenticate (`hf auth login`, or export
`HF_TOKEN`), then run the PANINI-DR repository's preparation script, which
downloads the files, checks them, decompresses them and rebuilds the FAISS
index:

```bash
export PANINI_DATA_DIR=$PWD/data
python datasets/monaco/prepare.py questions   # from allenai/MoNaCo_Benchmark (gated)
python datasets/monaco/prepare.py corpus      # this dataset
panini-dr run --strict-rerank --subset configs/splits/monaco_full1315.json \
    --articles-cache "$PANINI_DATA_DIR/monaco/HYBRID_MONACO_full_v2.json" --output-dir runs/monaco
```

Installation needs about 27 GB of free disk while it runs and 21 GB afterwards
(corpus 1.0 GB, embeddings 10.1 GB, FAISS index 10.1 GB). The corpus lands in
`$PANINI_DATA_DIR/monaco/` and the dense index in
`$PANINI_DATA_DIR/retrieval_cache/<fingerprint>/`, where `panini-dr run` looks
for it. The directory name is a fingerprint of the corpus (a hash of the sorted
titles and of every entry's length), so the run finds the index from the corpus
alone and does not pick up an index built for a different corpus.

At run time PANINI-DR:

- splits each entry into chunks of at most 2,000 characters with a
  200-character overlap, packing whole paragraphs where it can (616,096 chunks,
  ids `<key>__<i>`);
- searches them with a hybrid retriever: an Okapi BM25 index built in memory at
  start-up, fused by reciprocal rank with exact inner-product search over these
  vectors, then reranked;
- reads whole articles from the same corpus when the agent opens a page.

`faiss.index` (an exact `IndexFlatIP`) is not shipped because it is a byte copy
of the embedding matrix; the script rebuilds it on CPU, and with faiss 1.13.2
the result is identical to the file the paper's runs used. A run holds the
embeddings and the FAISS index in memory (about 20 GB of RAM) and builds a BM25
index of roughly the same size at start-up.

### Without the script

```bash
zstd -d monaco/HYBRID_MONACO_full_v2.json.zst
zstd -d retrieval_cache/*/chunk_embeddings.npy.zst
```

```python
import json, pickle, numpy as np
corpus = json.load(open("monaco/HYBRID_MONACO_full_v2.json", encoding="utf-8"))  # {key: text}
vectors = np.load("retrieval_cache/Qwen__Qwen3-Embedding-8B__7ea9a4d678f8__de3db946f2e8/chunk_embeddings.npy", mmap_mode="r")
ids = pickle.load(open("retrieval_cache/Qwen__Qwen3-Embedding-8B__7ea9a4d678f8__de3db946f2e8/chunk_ids.pkl", "rb"))
```

## How it was made

- **Pages.** The English Wikipedia pages named in the `source_url` fields of
  MoNaCo's execution traces for all 1,315 questions: 35,456 of the 35,568
  names (112 could not be fetched), plus 1,052 further entries, mostly the
  target pages of names that redirect.
- **Keys.** Most keys are page names as MoNaCo's traces give them. For 1,092 of
  the 36,508 entries the key is not the page's current Wikipedia title:
  - most are redirects, so the same page can appear under two keys (1,038 pages
    appear more than once, for example `Anwar Sadat` and
    `Muhammad Anwar el-Sadat`);
  - 17 keys are only the part of a title after its last `/`, for example `DC`
    for `AC/DC`, an artefact of the original build.

  The keys are kept exactly as the paper's runs used them, because they are the
  document ids: chunk ids, the fingerprint and the dense index all depend on
  them. `provenance/articles.jsonl` gives the real title of every entry
  (`wikipedia_title`).
- **Text.** The Parsoid HTML of each page's then-current revision was fetched
  from the Wikimedia REST API and converted to text: the title on the first
  line, then an `[INFOBOX]` block of `key | value` lines, the lead, `## `
  section headings, `- ` list items, `a | b` table rows and a `## References`
  section. The corpus was built on 2026-08-03.
- **Revisions and fetch dates.** These are live revisions, not a dump with a
  cutoff date. Pages were fetched on 2026-07-22 and 2026-07-23 (35,525
  entries) and on 2026-08-03 (983 entries); revision timestamps run from
  2018-02-25 to 2026-08-03 (median 2026-06-24). Every entry's revision id,
  permalink and fetch time are in `provenance/articles.jsonl`. For four entries
  two fetched revisions could be the source (two pairs of keys for the same
  page); those rows list both in `revision_candidates`.
- **Vectors.** Each chunk was encoded with `Qwen/Qwen3-Embedding-8B`
  (sentence-transformers, fp16 on GPU, normalised, no instruction prefix for
  documents; queries use the model's retrieval instruction) and saved as
  float32. Built on 2026-08-03.

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
text, keep the attribution and the share-alike terms; the gate above adds
MoNaCo's request that the page selection is not redistributed.

MoNaCo itself is a separate release by its authors (`allenai/MoNaCo_Benchmark`,
ODC-BY); please cite it if you use this corpus with its questions.

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
