"""``panini-dr verify``: the record/replay equivalence check.

    panini-dr verify --bundle DIR [--data-dir DIR] [--keep-output DIR] [--case NAME ...]
    panini-dr verify record --cassette DIR [--source-repo PATH] -- <panini-dr run arguments>

The second form is for maintainers: it records a new cassette against LIVE services
(the LLM server and the reranker must be up); ``scripts/build_verify_bundle.py`` then
turns cassettes and their runs into a bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

verify_app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="Replay a verify bundle offline and check that this code reproduces the recorded "
    "runs exactly (exit 0 PASS, 1 FAIL, 2 unusable bundle/data).",
)


@verify_app.callback()
def verify(
    ctx: typer.Context,
    bundle: Optional[Path] = typer.Option(None, "--bundle", help="A verify bundle directory (manifest.json, cassettes/, reference/)."),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir",
        help="Data root holding the bundle's corpus, dense index and question file (else $PANINI_DATA_DIR, else ./data).",
    ),
    keep_output: Optional[Path] = typer.Option(
        None, "--keep-output", help="Keep the replay outputs, logs and miss reports here (a new or empty dir)."),
    case: list[str] = typer.Option([], "--case", help="Replay only this case of the bundle (repeatable)."),
    gate_timeout: float = typer.Option(60.0, "--gate-timeout", help="Seconds a response may wait for its recorded turn."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if bundle is None:
        typer.echo("error: pass --bundle DIR (or use `panini-dr verify record` to make a cassette)", err=True)
        raise typer.Exit(2)
    from panini_dr.replay.verify import verify_bundle

    raise typer.Exit(verify_bundle(bundle, data_dir=data_dir, keep_output=keep_output, only=case or None,
                                   gate_timeout=gate_timeout, echo=typer.echo))


@verify_app.command(
    "record",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def record(
    ctx: typer.Context,
    cassette: Path = typer.Option(..., "--cassette", help="New cassette directory (must not hold a recording)."),
    source_repo: str = typer.Option("", "--source-repo", help="Git repo whose HEAD is recorded as the source commit."),
    runner_script: str = typer.Option(
        "", "--runner-script",
        help="Record another harness: run this script with the arguments after -- (default: panini-dr run).",
    ),
    pkg: str = typer.Option("panini_dr", "--pkg", help="Package of the harness whose LLMClient is hooked."),
    hashseed: str = typer.Option("0", "--hashseed", help="PYTHONHASHSEED for the recorded process."),
) -> None:
    """Maintainers: record a cassette against LIVE services.

    Everything after -- is passed to `panini-dr run` (or to --runner-script), for example:

    \b
    panini-dr verify record --cassette cas/frames -- \\
        --subset s.json --articles-cache c.json --output-dir runs/rec

    The process re-execs itself with PYTHONHASHSEED fixed and HF offline; a connection made
    outside the LLM / encoder / reranker hooks is logged as a leak (a bundle refuses it).
    The run gets --skip-preflight (its test requests are not recorded; verify replays
    with it too).
    """
    from panini_dr.replay.launcher import main as launcher_main

    args = list(ctx.args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        typer.echo("error: give the run arguments after --", err=True)
        raise typer.Exit(2)
    own = ["--mode", "record", "--cassette", str(cassette), "--pkg", pkg, "--hashseed", hashseed]
    if source_repo:
        own += ["--source-repo", source_repo]
    if runner_script:
        own += ["--runner-script", runner_script]
    runner = args if runner_script else ["run", *args]
    raise typer.Exit(launcher_main([*own, "--", *runner]))


__all__ = ["verify_app"]
