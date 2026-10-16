"""Corpus provenance and eligibility contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class CorpusScope(str, Enum):
    """High-level corpus scope for validity checks."""

    FULL_WIKIPEDIA = "full_wikipedia"
    BENCHMARK_DERIVED = "benchmark_derived"
    CUSTOM = "custom"


class CorpusEligibilityError(RuntimeError):
    """Raised by future official-evaluation gates for invalid corpora."""


class CorpusManifest(BaseModel):
    """Stable manifest describing a retrieval corpus."""

    manifest_schema_version: str = "1.0"
    corpus_id: str
    corpus_protocol: str = ""
    source: str
    source_artifact_id: str = ""
    snapshot_date: str
    build_timestamp: str
    article_count: int
    document_count: int | None = None
    structured_unit_count: int
    typed_unit_counts: dict[str, int] = Field(default_factory=dict)
    content_hash_or_manifest_hash: str
    artifact_checksum: str = ""
    output_checksum: str = ""
    parser_name: str = ""
    parser_version: str
    build_configuration: dict[str, Any] = Field(default_factory=dict)
    scope: CorpusScope
    benchmark_name_if_derived: str | None = None
    benchmark_name: str | None = None
    dataset_fingerprint: str = ""
    membership_source: str = ""
    membership_question_count: int = 0
    raw_gold_reference_count: int = 0
    normalized_unique_title_count: int = 0
    canonical_document_count: int = 0
    alias_count: int = 0
    unresolved_reference_count: int = 0
    contains_hyperlink_targets: bool = False
    contains_article_structure: bool = False
    contains_question_to_article_mapping: bool = False
    hyperlink_target_policy: str = ""
    eligible_for_open_corpus_evaluation: bool = False
    eligible_for_frames_gold_union_evaluation: bool = False
    eligible_for_official_evaluation: bool = False
    official_evaluation_eligible: bool = Field(
        False,
        description="Compatibility alias used in run summaries.",
    )

    @model_validator(mode="after")
    def _align_eligibility_aliases(self) -> "CorpusManifest":
        if self.official_evaluation_eligible != self.eligible_for_official_evaluation:
            self.official_evaluation_eligible = self.eligible_for_official_evaluation
        if self.benchmark_name is None:
            self.benchmark_name = self.benchmark_name_if_derived
        if self.canonical_document_count == 0:
            self.canonical_document_count = self.document_count or self.article_count
        if self.document_count is None:
            self.document_count = self.article_count
        if not self.artifact_checksum:
            self.artifact_checksum = self.content_hash_or_manifest_hash
        if not self.output_checksum:
            self.output_checksum = self.content_hash_or_manifest_hash
        return self


def assert_official_evaluation_eligible(manifest: CorpusManifest) -> None:
    """Future enforcement hook for official benchmark runs."""

    if not manifest.eligible_for_official_evaluation:
        raise CorpusEligibilityError(
            f"Corpus {manifest.corpus_id!r} is not eligible for official evaluation"
        )


def manifest_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


#: Benchmark name (as split files and ``panini-dr run`` spell it) -> (label, display name).
BENCHMARK_LABELS: dict[str, tuple[str, str]] = {
    "frames": ("frames", "FRAMES"),
    "monaco": ("monaco", "MoNaCo"),
    "browsecomp_plus": ("browsecomp_plus", "BrowseComp-Plus"),
    "browsecomp-plus": ("browsecomp_plus", "BrowseComp-Plus"),
    "bcp": ("browsecomp_plus", "BrowseComp-Plus"),
}
#: The benchmark name of a dataset written by ``panini-dr prepare``.
USER_DATASET = "byod"


def build_corpus_manifest(
    *,
    path: Path,
    articles: dict[str, str],
    structured_unit_count: int,
    benchmark: str = "frames",
    dataset_name: str | None = None,
) -> CorpusManifest:
    """Manifest for a ``{title: text}`` corpus file.

    ``benchmark`` says what the corpus is for: a paper benchmark (``frames``,
    ``monaco``, ``browsecomp_plus``; scope ``benchmark_derived``, corpus id
    ``<benchmark>-derived:<file>:<hash>``) or ``byod``, a dataset written by
    ``panini-dr prepare`` (scope ``custom``, corpus id
    ``dataset:<name>:<hash>``).
    """

    digest = manifest_hash(articles)
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        build_timestamp = mtime.isoformat()
    except OSError:
        build_timestamp = datetime.now(timezone.utc).isoformat()

    short = digest.removeprefix("sha256:")[:12]
    bench = (benchmark or "frames").strip().lower()
    if bench == USER_DATASET:
        name = dataset_name or path.parent.name or path.name
        corpus_id = f"dataset:{name}:{short}"
        scope, derived_from = CorpusScope.CUSTOM, None
    else:
        label, derived_from = BENCHMARK_LABELS.get(bench, (bench, bench))
        corpus_id = f"{label}-derived:{path.name}:{short}"
        scope = CorpusScope.BENCHMARK_DERIVED
    return CorpusManifest(
        corpus_id=corpus_id,
        source=str(path),
        snapshot_date="unknown",
        build_timestamp=build_timestamp,
        article_count=len(articles),
        structured_unit_count=structured_unit_count,
        content_hash_or_manifest_hash=digest,
        parser_version="frames_plaintext_cache_v1",
        scope=scope,
        benchmark_name_if_derived=derived_from,
        contains_hyperlink_targets=False,
        contains_article_structure=False,
        eligible_for_official_evaluation=False,
    )


def build_frames_cache_manifest(
    *,
    path: Path,
    articles: dict[str, str],
    structured_unit_count: int,
) -> CorpusManifest:
    """Manifest for a FRAMES-derived article cache."""

    return build_corpus_manifest(
        path=path, articles=articles, structured_unit_count=structured_unit_count, benchmark="frames"
    )


__all__ = [
    "CorpusEligibilityError",
    "CorpusManifest",
    "CorpusScope",
    "BENCHMARK_LABELS",
    "USER_DATASET",
    "assert_official_evaluation_eligible",
    "build_corpus_manifest",
    "build_frames_cache_manifest",
    "manifest_hash",
]
