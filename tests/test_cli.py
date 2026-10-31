"""Tests for the command-line interface."""

import sys

from typer.testing import CliRunner

from pyevm.cli import app

runner = CliRunner()


def test_dashboard_without_streamlit_suggests_app_extra(monkeypatch):
    monkeypatch.setenv("PYEVM_LOG_LEVEL", "INFO")
    monkeypatch.setitem(sys.modules, "streamlit", None)

    result = runner.invoke(app, ["dashboard"])

    assert result.exit_code == 1
    assert 'pip install "pyeulervid[app]"' in result.output
