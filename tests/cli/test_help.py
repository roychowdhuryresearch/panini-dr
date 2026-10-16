"""Every command's --help shows its help text in full (no Rich markup eating [brackets])."""

from __future__ import annotations

import click
import pytest
import typer
from typer.testing import CliRunner

from panini_dr.cli import app

runner = CliRunner()


def _commands(cmd: click.Command, path: tuple[str, ...] = ()):
    yield path, cmd
    if isinstance(cmd, click.Group):
        for name, sub in cmd.commands.items():
            yield from _commands(sub, (*path, name))


ALL = list(_commands(typer.main.get_command(app)))


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("path, cmd", ALL, ids=[" ".join(p) or "panini-dr" for p, _ in ALL])
def test_help_shows_every_option_text(path, cmd):
    res = runner.invoke(app, [*path, "--help"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    for param in cmd.params:
        if isinstance(param, click.Option) and param.help:
            assert _flat(param.help) in out, (param.name, param.help)


def test_bracketed_help_text_survives():
    out = _flat(runner.invoke(app, ["run", "--help"]).output)
    assert "Served model name [env PANINI_LLM_MODEL; default: the paper's]." in out
    assert "OpenAI-compatible base URL of the LLM [env PANINI_LLM_BASE_URL]." in out
    out = _flat(runner.invoke(app, ["score", "answers", "--help"]).output)
    assert "{bucket: [qid]} for per-bucket counts" in out
