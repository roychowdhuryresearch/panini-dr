"""Offline tests of the published-corpus installer (datasets/_prep_common.py).

Everything runs on a tiny synthetic corpus laid out like the Hugging Face
repos; the Hub client is replaced by fakes, so no test touches the network.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")


def _load(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def frames():
    return _load("datasets/frames/prepare.py", "_panini_prep_frames")


@pytest.fixture(scope="module")
def monaco():
    return _load("datasets/monaco/prepare.py", "_panini_prep_monaco")


@pytest.fixture(scope="module")
def common(frames):
    import _prep_common  # put on sys.path by the prepare scripts

    return _prep_common


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


ARTICLES = {
    "Alpha": "Alpha\n\nAlpha is a synthetic article. " * 3,
    "Beta (band)": "Beta (band)\n\n[INFOBOX]\nOrigin | Nowhere\n\n## History\n" + "Beta played. " * 400,
    "Gamma": "Gamma\n\nShort.",
}


@pytest.fixture
def fake_repo(common, tmp_path):
    """A repo laid out like panini-dr-*-corpus (gzip instead of zstd) and its pinned artefact."""
    from panini_dr.retrieval.corpus import load_frames_corpus

    root = tmp_path / "staging" / "panini-dr-toy-corpus"
    corpus_raw = json.dumps(ARTICLES).encode()
    (tmp_path / "c.json").write_bytes(corpus_raw)
    chunks = load_frames_corpus(cache_path=tmp_path / "c.json").chunks
    ids = [c.chunk_id for c in chunks]
    fp = common.dense_fingerprint(ARTICLES)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((len(ids), 8)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    tmp_npy = tmp_path / "e.npy"
    np.save(tmp_npy, emb)
    npy_raw = tmp_npy.read_bytes()
    pkl_raw = pickle.dumps(ids)
    index = faiss.IndexFlatIP(8)
    index.add(emb)
    faiss_sha = hashlib.sha256(memoryview(faiss.serialize_index(index))).hexdigest()

    cache = f"retrieval_cache/{fp}"
    files = {
        "toy/C.json.gz": gzip.compress(corpus_raw, mtime=0),
        f"{cache}/chunk_embeddings.npy.gz": gzip.compress(npy_raw, mtime=0),
        f"{cache}/chunk_ids.pkl": pkl_raw,
        "provenance/articles.jsonl": b"{}\n",
        "README.md": b"card\n",
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)

    def shipped(rel, raw, install_to):
        data = files[rel]
        kw = {} if raw is None else {"raw_sha256": _sha(raw), "raw_size": len(raw)}
        return common.ShippedFile(path=rel, sha256=_sha(data), size=len(data), install_to=install_to, **kw)

    art = common.CorpusArtefact(
        benchmark="toy", label="TOY", repo_id="example-org/panini-dr-toy-corpus", gated=False,
        fingerprint=fp, n_articles=len(ARTICLES), n_chunks=len(ids), dim=8,
        corpus=shipped("toy/C.json.gz", corpus_raw, "toy/C.json"),
        embeddings=shipped(f"{cache}/chunk_embeddings.npy.gz", npy_raw, f"{cache}/chunk_embeddings.npy"),
        chunk_ids=shipped(f"{cache}/chunk_ids.pkl", None, f"{cache}/chunk_ids.pkl"),
        faiss_sha256=faiss_sha, split_file="configs/splits/toy.json",
    )
    return root, art


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_install_layout_is_what_the_run_looks_for(common, fake_repo, tmp_path, monkeypatch):
    root, art = fake_repo
    data = tmp_path / "data"
    logs: list[str] = []
    placed = common.install_corpus_artefact(art, root, data, verify_chunks=True, log=logs.append)
    assert placed["corpus"] == data / "toy" / "C.json"
    assert json.loads(placed["corpus"].read_text()) == ARTICLES
    assert placed["dense_cache"] == data / "retrieval_cache" / art.fingerprint
    assert {p.name for p in placed["dense_cache"].iterdir()} == {
        "chunk_embeddings.npy", "chunk_ids.pkl", "faiss.index"}
    assert any("identical to the paper's" in line for line in logs)
    assert not list(data.rglob("*.partial")) and not list(data.rglob("*.tmp"))

    # The run's own loader finds the cache from PANINI_DATA_DIR and the corpus alone.
    from panini_dr.retrieval.corpus import load_frames_corpus
    from panini_dr.retrieval.dense import DenseRetriever

    monkeypatch.setenv("PANINI_DATA_DIR", str(data))
    dense = DenseRetriever(load_frames_corpus(cache_path=placed["corpus"]), device="cpu")
    assert dense._cache_dir == placed["dense_cache"]
    assert dense._index.ntotal == art.n_chunks


def test_install_is_idempotent(common, fake_repo, tmp_path):
    root, art = fake_repo
    data = tmp_path / "data"
    common.install_corpus_artefact(art, root, data, log=lambda m: None)
    before = {p: p.stat().st_mtime_ns for p in data.rglob("*") if p.is_file()}
    logs: list[str] = []
    common.install_corpus_artefact(art, root, data, log=logs.append)
    assert sum("already installed" in line for line in logs) == 4  # 3 files + faiss.index
    assert {p: p.stat().st_mtime_ns for p in data.rglob("*") if p.is_file()} == before


def test_local_source_accepts_the_staging_parent(common, fake_repo, tmp_path):
    root, art = fake_repo
    assert common.resolve_local_source(root, art) == root
    assert common.resolve_local_source(root.parent, art) == root
    with pytest.raises(common.PrepareError, match="toy/C.json.gz"):
        common.resolve_local_source(tmp_path / "elsewhere", art)


def test_paper_artefacts_are_pinned_and_consistent(common, frames, monaco):
    assert set(common.HF_CORPUS_REPOS) == {"frames", "monaco"}
    for mod, gated, corpus in ((frames, False, "frames/HYBRID_FRAMES824.json"),
                               (monaco, True, "monaco/HYBRID_MONACO_full_v2.json")):
        art = mod.ARTEFACT
        assert art.repo_id == common.HF_CORPUS_REPOS[art.benchmark]
        assert art.repo_id.startswith("roychowdhuryresearch/")
        assert art.gated is gated
        assert art.fingerprint == mod.PAPER_DENSE_FINGERPRINT and art.n_chunks == mod.PAPER_N_CHUNKS
        assert art.corpus.install_to == corpus
        assert art.corpus.path == corpus + ".zst"
        for f in art.files:
            assert len(f.sha256) == 64 and len(f.installed_sha256) == 64 and f.size > 0
        dense = {str(Path(f.install_to).parent) for f in (art.embeddings, art.chunk_ids)}
        assert dense == {f"retrieval_cache/{art.fingerprint}"}
        assert (REPO / art.split_file).is_file()
        assert sorted(art.allow_patterns()) == sorted(f.path for f in art.files)  # verify/ is never fetched


def _code_strings(path: Path) -> list[str]:
    """String constants of a module, docstrings excluded."""
    import ast

    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def test_repo_ids_are_defined_in_one_place(common):
    for repo_id in common.HF_CORPUS_REPOS.values():
        hits = [p.relative_to(REPO) for p in sorted((REPO / "datasets").rglob("*.py"))
                if any(repo_id in s for s in _code_strings(p))]
        assert hits == [Path("datasets/_prep_common.py")]


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def test_corrupted_download_is_refused(common, fake_repo, tmp_path):
    root, art = fake_repo
    target = root / art.embeddings.path
    data = bytearray(target.read_bytes())
    data[-12] ^= 0xFF  # same size, different bytes
    target.write_bytes(bytes(data))
    with pytest.raises(common.PrepareError, match="sha256 mismatch for .*chunk_embeddings.npy.gz"):
        common.install_corpus_artefact(art, root, tmp_path / "data", log=lambda m: None)
    assert not (tmp_path / "data" / art.embeddings.install_to).exists()
    assert not list((tmp_path / "data").rglob("*.partial"))


def test_truncated_download_is_refused(common, fake_repo, tmp_path):
    root, art = fake_repo
    target = root / art.corpus.path
    target.write_bytes(target.read_bytes()[:-5])
    with pytest.raises(common.PrepareError, match="truncated or different"):
        common.install_corpus_artefact(art, root, tmp_path / "data", log=lambda m: None)


def test_decompressed_bytes_are_checked_too(common, fake_repo, tmp_path):
    root, art = fake_repo
    bad = dataclasses.replace(art, corpus=dataclasses.replace(art.corpus, raw_sha256="0" * 64))
    with pytest.raises(common.PrepareError, match="decompressed to"):
        common.install_corpus_artefact(bad, root, tmp_path / "data", log=lambda m: None)
    assert not (tmp_path / "data" / art.corpus.install_to).exists()
    assert not list((tmp_path / "data").rglob("*.partial"))


def test_missing_shipped_file(common, fake_repo, tmp_path):
    root, art = fake_repo
    (root / art.chunk_ids.path).unlink()
    with pytest.raises(common.PrepareError, match="chunk_ids.pkl is missing"):
        common.install_corpus_artefact(art, root, tmp_path / "data", log=lambda m: None)


def test_wrong_fingerprint_is_refused(common, fake_repo, tmp_path):
    root, art = fake_repo
    other = "Qwen__Qwen3-Embedding-8B__000000000000__000000000000"
    moved = {}
    for name in ("embeddings", "chunk_ids"):
        f = getattr(art, name)
        moved[name] = dataclasses.replace(f, install_to=f.install_to.replace(art.fingerprint, other))
    bad = dataclasses.replace(art, fingerprint=other, **moved)
    with pytest.raises(common.PrepareError, match="would not find it"):
        common.install_corpus_artefact(bad, root, tmp_path / "data", log=lambda m: None)


def test_chunk_ids_pickle_may_not_carry_objects(common, tmp_path):
    good = tmp_path / "ok.pkl"
    good.write_bytes(pickle.dumps(["a__0", "b__0"]))
    assert common.load_chunk_ids(good) == ["a__0", "b__0"]
    evil = tmp_path / "evil.pkl"
    marker = tmp_path / "executed"
    evil.write_bytes(f"cos\nsystem\n(S'touch {marker}'\ntR.".encode())  # calls os.system if unpickled
    with pytest.raises(common.PrepareError, match="references os.system; expected a plain list of strings"):
        common.load_chunk_ids(evil)
    assert not marker.exists()
    other = tmp_path / "set.pkl"
    other.write_bytes(pickle.dumps({"a__0"}))
    with pytest.raises(common.PrepareError, match="not a list of chunk-id strings"):
        common.load_chunk_ids(other)


def test_zstd_files_decompress(common, tmp_path):
    raw = b'{"Alpha": "text"}' * 1000
    src = tmp_path / "c.json.zst"
    try:
        import zstandard

        src.write_bytes(zstandard.ZstdCompressor().compress(raw))
    except ImportError:
        if shutil.which("zstd") is None:
            pytest.skip("neither the zstandard package nor the zstd command is available")
        subprocess.run(["zstd", "-q", "-o", str(src), "-"], input=raw, check=True)
    f = common.ShippedFile(path="c.json.zst", sha256=_sha(src.read_bytes()), size=src.stat().st_size,
                           install_to="c.json", raw_sha256=_sha(raw), raw_size=len(raw))
    common.install_file(src, f, tmp_path / "out" / "c.json")
    assert (tmp_path / "out" / "c.json").read_bytes() == raw


def test_zstd_missing_tool_message(common, tmp_path, monkeypatch):
    src = tmp_path / "c.json.zst"
    src.write_bytes(b"\x28\xb5\x2f\xfd")
    monkeypatch.setitem(sys.modules, "zstandard", None)  # import fails
    monkeypatch.setattr(common.shutil, "which", lambda name: None)
    f = common.ShippedFile(path="c.json.zst", sha256="0" * 64, size=4, install_to="c.json",
                           raw_sha256="0" * 64, raw_size=1)
    with pytest.raises(common.PrepareError, match="install the zstd command-line tool"):
        common.install_file(src, f, tmp_path / "c.json")


# ---------------------------------------------------------------------------
# Hugging Face download
# ---------------------------------------------------------------------------


@pytest.fixture
def hub(monkeypatch):
    """Patch huggingface_hub so nothing reaches the network; returns a call log."""
    import huggingface_hub

    calls: list[dict] = []

    def no_download(**kwargs):
        calls.append(kwargs)
        raise AssertionError("snapshot_download must not be called")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_download)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    return calls


def test_gated_repo_without_token_explains_what_to_do(common, monaco, hub, tmp_path):
    with pytest.raises(common.PrepareError) as exc:
        common.hf_download_artefact(monaco.ARTEFACT, download_dir=tmp_path / "dl")
    msg = str(exc.value)
    assert "no Hugging Face token found" in msg and "gated" in msg
    assert "https://huggingface.co/datasets/roychowdhuryresearch/panini-dr-monaco-corpus" in msg
    assert "hf auth login" in msg and "HF_TOKEN" in msg and "--from-local" in msg
    assert "MoNaCo" in msg
    assert hub == []  # refused before any request


def test_gated_repo_without_token_via_cli(monaco, hub, tmp_path, capsys):
    assert monaco.main(["--data-dir", str(tmp_path), "corpus"]) == 2
    err = capsys.readouterr().err
    assert "no Hugging Face token found" in err and "hf auth login" in err


def test_access_not_granted_yet(common, monaco, monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_test")

    def gated(**kwargs):
        raise GatedRepoError("403 Client Error: access to this repo is restricted")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", gated)
    with pytest.raises(common.PrepareError, match="not been granted access yet"):
        common.hf_download_artefact(monaco.ARTEFACT, download_dir=tmp_path / "dl")


def test_public_repo_errors(common, frames, monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    for err, pattern in ((RepositoryNotFoundError("404"), "was not found on the Hugging Face Hub"),
                         (LocalEntryNotFoundError("offline"), "the Hub is unreachable")):
        def fail(_err=err, **kwargs):
            raise _err

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
        with pytest.raises(common.PrepareError, match=pattern) as exc:
            common.hf_download_artefact(frames.ARTEFACT, download_dir=tmp_path / "dl")
        assert "--from-local" in str(exc.value)


def test_download_then_install_via_cli(frames, fake_repo, monkeypatch, tmp_path, capsys):
    """`prepare.py corpus` with a fake Hub: only the three files are requested, then installed."""
    import huggingface_hub

    root, art = fake_repo
    calls: list[dict] = []

    def fake_snapshot(*, repo_id, repo_type, revision, allow_patterns, local_dir, **kw):
        calls.append({"repo_id": repo_id, "repo_type": repo_type, "revision": revision,
                      "allow_patterns": list(allow_patterns)})
        for rel in allow_patterns:
            (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, Path(local_dir) / rel)
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(frames, "ARTEFACT", art)
    data = tmp_path / "data"
    assert frames.main(["--data-dir", str(data), "corpus", "--revision", "v1"]) == 0
    assert calls == [{"repo_id": art.repo_id, "repo_type": "dataset", "revision": "v1",
                      "allow_patterns": art.allow_patterns()}]
    assert (data / "toy" / "C.json").is_file()
    assert (data / "retrieval_cache" / art.fingerprint / "faiss.index").is_file()
    assert not (data / "_downloads").exists()  # removed after a successful install
    out = capsys.readouterr().out
    assert (f"PANINI_DATA_DIR={data} panini-dr run --strict-rerank --subset configs/splits/toy.json "
            f"--articles-cache {data / 'toy' / 'C.json'} --output-dir runs/{art.benchmark}") in out

    # --repo overrides the repo id; --keep-download keeps the compressed files
    data2 = tmp_path / "data2"
    assert frames.main(["--data-dir", str(data2), "corpus", "--repo", "mirror/panini-dr-toy-corpus",
                        "--keep-download"]) == 0
    assert calls[-1]["repo_id"] == "mirror/panini-dr-toy-corpus"
    assert (data2 / "_downloads" / art.repo_name / art.corpus.path).is_file()


def test_from_local_via_cli_and_alias(frames, fake_repo, monkeypatch, tmp_path, hub):
    root, art = fake_repo
    monkeypatch.setattr(frames, "ARTEFACT", art)
    assert frames.main(["--data-dir", str(tmp_path / "a"), "corpus", "--from-local", str(root.parent)]) == 0
    assert frames.main(["--data-dir", str(tmp_path / "b"), "corpus", "--from-dir", str(root)]) == 0
    for d in ("a", "b"):
        assert (tmp_path / d / "retrieval_cache" / art.fingerprint / "chunk_embeddings.npy").is_file()
    assert hub == []  # no download attempted


def test_from_local_with_a_corrupted_file_exits_2(frames, fake_repo, monkeypatch, tmp_path, capsys):
    root, art = fake_repo
    (root / art.chunk_ids.path).write_bytes(pickle.dumps(["x__0"]))
    monkeypatch.setattr(frames, "ARTEFACT", art)
    assert frames.main(["--data-dir", str(tmp_path / "d"), "corpus", "--from-local", str(root)]) == 2
    assert "chunk_ids.pkl" in capsys.readouterr().err
