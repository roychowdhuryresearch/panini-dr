"""The paper configuration: preset expansion and the identity check.

This is a faithful port of how the research runner (``run_substitution.py``)
turned ``--full-champion-agentic-run-latest --gsw-edge-grounded-cap 5 ...`` into
the adapter config that the paper runs recorded in ``cell_result.json``:

1. every runner option starts at its runner default (:data:`RUNNER_DEFAULTS`);
2. values typed on the command line replace them (:data:`PAPER_COMMAND_LINE`,
   plus any ``--set key=value`` override);
3. ``--full-champion-agentic-run-latest`` switches on :data:`AGENTIC_OVERLAY`
   (except keys typed explicitly);
4. ``gsw_champion`` overlays ``configs/presets/champion_82.json`` (except keys
   typed explicitly);
5. :func:`build_run_extra` drops the disabled options listed in
   :data:`GSW_OMIT_IF_FALSY`.

Options whose code the paper configuration could never execute were removed
(``configs/removed_features.json``): they are no longer runner options, and
setting one is an error. Their paper values are the only behaviour left; the
recorded ones are added back by :func:`recorded_config`.

``recorded_config(result)`` must equal ``configs/paper.json["config"]``;
``panini-dr run`` refuses to start otherwise (unless ``--allow-config-drift``). The adapter
defaults that are not config keys but govern the run (researcher turns 20,
orchestrator turns 12, ...) are pinned in ``paper.json["pinned_adapter_defaults"]``
and checked against the constructed adapter by :func:`check_pinned_defaults`.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config files: in a source checkout they live in <repo>/configs; an installed
# wheel carries copies under panini_dr/_configs (pyproject force-include).
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent


def configs_dir() -> Path:
    """Directory holding ``paper.json`` and ``presets/``."""
    repo = _PKG_DIR.parent.parent / "configs"
    if (repo / "paper.json").is_file():
        return repo
    packaged = _PKG_DIR / "_configs"
    if (packaged / "paper.json").is_file():
        return packaged
    raise FileNotFoundError(
        "cannot find configs/paper.json (looked in the source checkout and in the "
        "installed package)"
    )


def paper_config_path() -> Path:
    return configs_dir() / "paper.json"


def champion_preset_path() -> Path:
    return configs_dir() / "presets" / "champion_82.json"


def load_paper_config(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or paper_config_path()).read_text())


def removed_features_path() -> Path:
    return configs_dir() / "removed_features.json"


@functools.lru_cache(maxsize=1)
def load_removed_features() -> dict[str, Any]:
    """``configs/removed_features.json``: options, values and environment
    variables whose code was removed because the paper never executed it."""
    data = json.loads(removed_features_path().read_text())
    data.setdefault("keys", {})
    data.setdefault("values", {})
    data.setdefault("env_vars", {})
    return data


def removed_keys() -> dict[str, dict[str, Any]]:
    """``{key: {"paper_value", "recorded", "reason"}}`` for every removed key."""
    return dict(load_removed_features()["keys"])


def recorded_removed_values() -> dict[str, Any]:
    """Removed keys the paper's recorded config carries, with their paper value."""
    return {k: v["paper_value"] for k, v in removed_keys().items() if v.get("recorded")}


# ---------------------------------------------------------------------------
# 1. Runner defaults: the value every option of run_substitution.py had when it
# was not typed. Non-gsw options use their adapter-config key names
# (retriever -> retriever_type, temperature -> llm_temperature, seed -> llm_seed).
# ---------------------------------------------------------------------------

#: Non-gsw options and their runner defaults.
NONGSW_RUNNER_DEFAULTS: dict[str, Any] = {
    "retriever_type": "hybrid",
    "llm_temperature": 0.0,
    "reasoning_effort": "medium",
    "synthesis_validator_mode": "repair",
    "max_turns": 50,
    "max_completion_tokens": 16384,
    "reranker": "jina",
    # runner `--seed -1` meant "send no seed"
    "llm_seed": None,
}

#: Every gsw_ option of the runner and its default. Kept complete so that a
#: ``--set gsw_<key>=...`` override behaves exactly like the old flag.
GSW_RUNNER_DEFAULTS: dict[str, Any] = {
    "gsw_evidence_state": True,
    "gsw_evidence_gate": "measure",
    "gsw_hard_type_filter": True,
    "gsw_evidence_grounder": "llm",
    "gsw_typed_blank_resolver": True,
    "gsw_react_fallback": True,
    "gsw_block_unverified_cascade": False,
    "gsw_blank_qualifier": True,
    "gsw_blank_hints": True,
    "gsw_codegen_compute": True,
    "gsw_investigation_memory": False,
    "gsw_investigation_memory_enforce": False,
    "gsw_investigation_memory_finalshot": False,
    "gsw_memory_dossier": False,
    "gsw_memory_scoped_dispatch": False,
    "gsw_context_compaction": True,
    "gsw_plan_verifier": False,
    "gsw_verifier_scoped": True,
    "gsw_verifier_reasoning": True,
    "gsw_edge_grounded_cap": 1,
    "gsw_plan_update_guard": "enforce",
    "gsw_plan_quality_checks": True,
    "gsw_plan_issue_surface": "advisory",
    "gsw_max_plan_updates": 5,
    "gsw_plan_request_contract": False,
    "gsw_read_raw_fallback": True,
    "gsw_champion": False,
}

RUNNER_DEFAULTS: dict[str, Any] = {**NONGSW_RUNNER_DEFAULTS, **GSW_RUNNER_DEFAULTS}

#: Keys that are not runner options but may be set with ``--set``; they go
#: straight into the adapter's ``extra`` (the runner set them only when non-default).
PASSTHROUGH_KEYS: frozenset[str] = frozenset(
    {"level_max_turns", "orchestrator_max_turns", "synthesis_validator_guided_repair"}
)

#: Values the paper runs typed on the command line (besides
#: --full-champion-agentic-run-latest, --model, --base-url, --subset, ...).
PAPER_COMMAND_LINE: dict[str, Any] = {
    "gsw_edge_grounded_cap": 5,
    "llm_seed": 42,
    "llm_temperature": 0.0,
    "reranker": "jina",
}

#: What --full-champion-agentic-run-latest switched on.
AGENTIC_OVERLAY: dict[str, Any] = {
    "gsw_champion": True,
    "gsw_investigation_memory": True,
    "gsw_investigation_memory_enforce": True,
    "gsw_investigation_memory_finalshot": True,
    "gsw_memory_dossier": True,
    "gsw_memory_scoped_dispatch": True,
    "gsw_read_raw_fallback": True,
    "gsw_plan_verifier": True,
    "gsw_plan_request_contract": True,
}

# ---------------------------------------------------------------------------
# 5. Which disabled options are left out of `extra` (verbatim from the runner).
# ---------------------------------------------------------------------------

# gsw_* options written to `extra` ONLY when enabled (everything else is
# written unconditionally).
GSW_OMIT_IF_FALSY: frozenset[str] = frozenset(
    {
        "gsw_codegen_compute",
        "gsw_investigation_memory",
        "gsw_investigation_memory_enforce",
        "gsw_investigation_memory_finalshot",
        "gsw_memory_dossier",
        "gsw_memory_scoped_dispatch",
        "gsw_champion",
        "gsw_plan_verifier",
        "gsw_plan_request_contract",
    }
)


def build_gsw_extra(gsw_opts: Mapping[str, Any]) -> dict[str, Any]:
    """The gsw_* slice of the adapter ``extra`` (verbatim port of the runner's)."""
    out: dict[str, Any] = {}
    for key, value in gsw_opts.items():
        if key in GSW_OMIT_IF_FALSY and not value:
            continue
        out[key] = value
    return out


def build_run_extra(
    *,
    retriever: str,
    temperature: float,
    reasoning_effort: str,
    synthesis_validator_mode: str,
    gsw_opts: Mapping[str, Any],
) -> dict[str, Any]:
    """The deterministic config slice of ``extra`` (port of the runner's, minus
    the removed options)."""
    out: dict[str, Any] = {
        "retriever_type": retriever,
        "llm_temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    if synthesis_validator_mode:
        out["synthesis_validator_mode"] = synthesis_validator_mode
    out.update(build_gsw_extra(gsw_opts))
    return out


def expand_gsw_champion_preset(explicit: Mapping[str, Any]) -> dict[str, Any]:
    """The champion preset (nongsw + gsw sections), with ``explicit`` on top."""
    fixture = json.loads(champion_preset_path().read_text())
    out: dict[str, Any] = {}
    out.update(fixture.get("nongsw_config", {}))
    out.update(fixture.get("gsw_config", {}))
    out.update(explicit)
    return out


def apply_gsw_champion_preset(gsw_opts: Mapping[str, Any], *, explicit: set) -> dict[str, Any]:
    """Overlay the preset's gsw_ keys onto ``gsw_opts`` except keys typed explicitly.

    The preset file is the research fixture, byte for byte, so it still names
    removed options; those are skipped (their paper value is the only behaviour).
    """
    removed = removed_keys()
    out = dict(gsw_opts)
    for key, value in expand_gsw_champion_preset({}).items():
        if key.startswith("gsw_") and key not in explicit and key not in removed:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

#: AdapterContext.max_prompt_tokens: never set by the runner (dataclass default).
DEFAULT_MAX_PROMPT_TOKENS = 32000

#: Keys of the resolved config that live on AdapterContext rather than in `extra`.
CONTEXT_KEYS = ("max_turns", "max_prompt_tokens", "max_completion_tokens")


@dataclass
class ResolvedConfig:
    """What the runner hands the adapter: AdapterContext caps + ``extra``."""

    max_turns: int
    max_prompt_tokens: int
    max_completion_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        """The shape ``cell_result.json`` records (and ``paper.json`` stores)."""
        return {
            "max_turns": self.max_turns,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            **self.extra,
        }


class UnknownConfigKeyError(KeyError):
    pass


class RemovedConfigError(UnknownConfigKeyError):
    """A key (or value, or environment variable) whose code was removed."""


def check_not_removed(overrides: Mapping[str, Any]) -> None:
    """Raise :class:`RemovedConfigError` for an override naming removed code."""
    features = load_removed_features()
    for key, value in overrides.items():
        if key in features["keys"]:
            info = features["keys"][key]
            raise RemovedConfigError(
                f"config key {key!r} was removed: {info['reason']} The paper value "
                f"({info['paper_value']!r}) is the only behaviour "
                "(see configs/removed_features.json)."
            )
        gone = features["values"].get(key)
        if gone and value in gone.get("removed", []):
            raise RemovedConfigError(
                f"{key}={value!r} was removed: {gone['reason']} "
                "(see configs/removed_features.json)."
            )


def removed_env_vars_set(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Removed environment variables that are set, ``{name: reason}``."""
    import os

    env = os.environ if environ is None else environ
    return {
        name: reason
        for name, reason in load_removed_features()["env_vars"].items()
        if str(env.get(name, "")).strip()
    }


def resolve_config(
    overrides: Mapping[str, Any] | None = None,
    *,
    full_champion_agentic_run_latest: bool = True,
    command_line: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    """Expand the preset the way the research runner did.

    ``overrides`` are treated exactly like flags typed on the command line: they
    win over the agentic overlay and the champion preset. ``command_line``
    defaults to :data:`PAPER_COMMAND_LINE`, so ``resolve_config()`` is the paper
    configuration.
    """
    typed: dict[str, Any] = dict(PAPER_COMMAND_LINE if command_line is None else command_line)
    check_not_removed(overrides or {})
    for key, value in (overrides or {}).items():
        if key not in RUNNER_DEFAULTS and key not in PASSTHROUGH_KEYS:
            raise UnknownConfigKeyError(
                f"unknown config key {key!r}: not a runner option "
                f"(see panini_dr.config.RUNNER_DEFAULTS / PASSTHROUGH_KEYS)"
            )
        typed[key] = value

    opts = dict(RUNNER_DEFAULTS)
    opts.update({k: v for k, v in typed.items() if k in RUNNER_DEFAULTS})
    gsw_opts = {k: v for k, v in opts.items() if k.startswith("gsw_")}

    if full_champion_agentic_run_latest:
        for key, value in AGENTIC_OVERLAY.items():
            if key not in typed:
                gsw_opts[key] = value

    nongsw = {k: opts[k] for k in NONGSW_RUNNER_DEFAULTS}
    if gsw_opts.get("gsw_champion"):
        preset = expand_gsw_champion_preset({})
        gsw_opts = apply_gsw_champion_preset(
            gsw_opts, explicit={k for k in gsw_opts if k in typed}
        )
        for key in (
            "retriever_type",
            "llm_temperature",
            "reasoning_effort",
            "synthesis_validator_mode",
            "max_turns",
            "max_completion_tokens",
        ):
            if key not in typed:
                nongsw[key] = preset.get(key, nongsw[key])

    extra = build_run_extra(
        retriever=nongsw["retriever_type"],
        temperature=nongsw["llm_temperature"],
        reasoning_effort=nongsw["reasoning_effort"],
        synthesis_validator_mode=nongsw["synthesis_validator_mode"],
        gsw_opts=gsw_opts,
    )
    # Added on top by the runner, in this order.
    if nongsw["reranker"]:
        extra["reranker"] = nongsw["reranker"]
    for key in sorted(PASSTHROUGH_KEYS):
        if key in typed:
            extra[key] = typed[key]
    seed = nongsw["llm_seed"]
    if seed is not None and int(seed) >= 0:
        extra["llm_seed"] = int(seed)

    return ResolvedConfig(
        max_turns=int(nongsw["max_turns"]),
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
        max_completion_tokens=int(nongsw["max_completion_tokens"]),
        extra=extra,
    )


def recorded_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """The config a run records: ``resolved`` (``ResolvedConfig.flat()``) plus the
    removed keys the paper's recorded config carries, at their paper values.

    This is what ``configs/paper.json["config"]`` is compared against, and what
    a run writes into ``cell_result.json``.
    """
    clash = set(resolved) & set(removed_keys())
    if clash:
        raise RemovedConfigError(f"resolved config names removed keys: {sorted(clash)}")
    return {**resolved, **recorded_removed_values()}


def config_drift(
    resolved: Mapping[str, Any], paper: Mapping[str, Any] | None = None
) -> dict[str, tuple[Any, Any]]:
    """``{key: (paper value, resolved value)}`` for every key where
    ``recorded_config(resolved)`` differs from the paper's recorded config.

    Dataset keys (corpus, dense cache, Lucene index) are ignored.
    """
    paper = paper or load_paper_config()
    expected = dict(paper["config"])
    resolved = recorded_config(resolved)
    ignore = set(paper.get("dataset_keys", ()))
    missing = "<absent>"
    out: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(expected) | set(resolved)):
        if key in ignore:
            continue
        a, b = expected.get(key, missing), resolved.get(key, missing)
        if a != b or type(a) is not type(b):
            out[key] = (a, b)
    return out


# ---------------------------------------------------------------------------
# Pinned adapter defaults
# ---------------------------------------------------------------------------


def observed_adapter_defaults(adapter: Any) -> dict[str, Any]:
    """Read the non-config values that govern a run off a constructed adapter."""
    import inspect

    from panini_dr.adapters.ours import _agentic_read, _investigation_memory, _plan_updater
    from panini_dr.adapters.ours import gsw_planner_orchestrator_v1 as orch
    from panini_dr.retrieval import dense, rerank

    extra = adapter.ctx.extra
    hybrid_sig = inspect.signature(dense.HybridRetriever.__init__).parameters
    search_sig = inspect.signature(orch.ReActResearcher._tool_search).parameters
    return {
        "level_max_turns": adapter.level_max_turns,
        # read in _run_llm_orchestrator as int(extra.get("orchestrator_max_turns", 12))
        "orchestrator_max_turns": int(extra.get("orchestrator_max_turns", _ORCHESTRATOR_TURNS_DEFAULT)),
        "MAX_PLANNER_TOOL_CALLS": adapter.MAX_PLANNER_TOOL_CALLS,
        "MAX_PLAN_UPDATES_PER_RUN": adapter.MAX_PLAN_UPDATES_PER_RUN,
        "MAX_PARALLEL_RESEARCHERS": adapter.MAX_PARALLEL_RESEARCHERS,
        "MAX_ESCALATIONS_PER_BLANK": adapter.MAX_ESCALATIONS_PER_BLANK,
        "MAX_VALIDATOR_RETRIES": adapter.MAX_VALIDATOR_RETRIES,
        "MAX_VALIDATOR_GUIDED_REPAIRS_PER_RUN": adapter.MAX_VALIDATOR_GUIDED_REPAIRS_PER_RUN,
        "synthesis_validator_guided_repair": adapter.synthesis_validator_guided_repair,
        "search_top_k_default": search_sig["top_k"].default,
        "rerank_candidate_k": inspect.signature(rerank.RerankedRetriever.__init__)
        .parameters["candidate_k"].default,
        "rrf_k": hybrid_sig["k"].default,
        "rrf_per_retriever_top_k": hybrid_sig["per_retriever_top_k"].default,
        "reader_max_turns": _agentic_read.MAX_READER_TURNS,
        "reader_model_context_tokens": _agentic_read.MODEL_CONTEXT_TOKENS,
        "raw_read_window_chars": orch._READ_WINDOW,
        "invmem_R_Q": _investigation_memory.R_Q,
        "invmem_R_STALL": _investigation_memory.R_STALL,
        "invmem_K_DRY": _investigation_memory.K_DRY,
        "plan_updater_max_revision_attempts": _plan_updater.MAX_REVISION_ATTEMPTS,
    }


#: The literal default in ``_run_llm_orchestrator`` (a unit test reads the source
#: to keep this in step).
_ORCHESTRATOR_TURNS_DEFAULT = 12


def check_pinned_defaults(adapter: Any, paper: Mapping[str, Any] | None = None) -> dict[str, tuple[Any, Any]]:
    """``{name: (pinned, observed)}`` for every pinned default the adapter breaks."""
    paper = paper or load_paper_config()
    pinned = dict(paper.get("pinned_adapter_defaults", {}))
    observed = observed_adapter_defaults(adapter)
    return {
        k: (v, observed.get(k, "<not observed>"))
        for k, v in pinned.items()
        if observed.get(k, "<not observed>") != v
    }


__all__ = [
    "AGENTIC_OVERLAY",
    "GSW_OMIT_IF_FALSY",
    "PAPER_COMMAND_LINE",
    "RUNNER_DEFAULTS",
    "RemovedConfigError",
    "ResolvedConfig",
    "apply_gsw_champion_preset",
    "build_gsw_extra",
    "build_run_extra",
    "check_not_removed",
    "check_pinned_defaults",
    "config_drift",
    "configs_dir",
    "expand_gsw_champion_preset",
    "load_paper_config",
    "load_removed_features",
    "observed_adapter_defaults",
    "recorded_config",
    "recorded_removed_values",
    "removed_env_vars_set",
    "removed_keys",
    "resolve_config",
]
