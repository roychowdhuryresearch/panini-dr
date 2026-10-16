# Toy dataset

Eight short encyclopedia-style articles about made-up places, people and
buildings, and three questions about them: two `short` questions that each need
three hops, and one `set` question. Nothing in it is real, and nothing comes
from a benchmark.

| File | Content |
|---|---|
| `docs.jsonl` | The documents (`id`, `title`, `text`). |
| `questions.jsonl` | The questions (`id`, `question`, `answer`, optional `answer_type`). |
| `dataset.yaml` | Names the two files for `panini-dr prepare --config`. |

```bash
# CPU only: no dense index (can only run BM25-only, which is not the paper configuration)
panini-dr prepare --config examples/toy/dataset.yaml --out data/toy-bm25 --no-dense

# With the dense index (GPU, about 16 GB), then a run of the paper configuration
panini-dr prepare --config examples/toy/dataset.yaml --out data/toy
panini-dr run --dataset data/toy --output-dir runs/toy
```

See [docs/own-dataset.md](../../docs/own-dataset.md) for the file formats,
running and grading.
