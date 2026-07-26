"""Unit coverage for the installed ``kirag`` command-line interface."""

from __future__ import annotations

import argparse
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cli


def test_main_dispatches_a_regular_command(monkeypatch):
    handler = Mock()
    monkeypatch.setattr(cli, "cmd_settings_show", handler)
    monkeypatch.setattr(sys, "argv", ["kirag", "settings", "show"])

    cli.main()

    handler.assert_called_once()


def test_main_dispatches_rag_infrastructure_command(monkeypatch):
    handler = Mock()
    monkeypatch.setattr(cli, "cmd_rag_infra_status", handler)
    monkeypatch.setattr(sys, "argv", ["kirag", "rag", "infra", "status"])

    cli.main()

    handler.assert_called_once()


@pytest.mark.parametrize("argv", [["kirag"], ["kirag", "rag", "infra"]])
def test_main_prints_help_for_incomplete_commands(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert "usage:" in capsys.readouterr().out


def test_pipeline_handlers(monkeypatch, capsys):
    monkeypatch.setattr(
        sys.modules["settings_manager"],
        "get_available_runs",
        lambda: [("Case One", "/workspace/run_one")],
        raising=False,
    )
    cli.cmd_pipeline_runs(argparse.Namespace())
    assert "Case One" in capsys.readouterr().out

    monkeypatch.setattr(sys.modules["settings_manager"], "get_available_runs", lambda: [])
    cli.cmd_pipeline_runs(argparse.Namespace())
    assert "No completed runs" in capsys.readouterr().out


def test_pipeline_status_states(monkeypatch, capsys):
    process_state = SimpleNamespace(active_runs_lock=threading.Lock(), active_runs={})
    monkeypatch.setitem(sys.modules, "process_state", process_state)

    cli.cmd_pipeline_status(argparse.Namespace(run_id="missing"))
    assert "not found" in capsys.readouterr().out

    process_state.active_runs["running"] = {"proc": SimpleNamespace(poll=lambda: None)}
    cli.cmd_pipeline_status(argparse.Namespace(run_id="running"))
    assert "RUNNING" in capsys.readouterr().out

    process_state.active_runs["stopped"] = {"proc": None, "stop": True}
    cli.cmd_pipeline_status(argparse.Namespace(run_id="stopped"))
    assert "STOPPED" in capsys.readouterr().out

    process_state.active_runs["complete"] = {"proc": None}
    cli.cmd_pipeline_status(argparse.Namespace(run_id="complete"))
    assert "COMPLETED" in capsys.readouterr().out


def test_settings_show_masks_token(monkeypatch, capsys):
    settings = {"hf_token": "do-not-print", "docker_port": 8000}
    monkeypatch.setattr(sys.modules["settings_manager"], "load_settings", lambda: settings)

    cli.cmd_settings_show(argparse.Namespace())

    output = capsys.readouterr().out
    assert "do-not-print" not in output
    assert "********" in output


@pytest.mark.parametrize(
    "existing,raw,expected",
    [(True, "no", False), (3, "4", 4), (1.5, "2.5", 2.5), ("old", "new", "new")],
)
def test_settings_set_casts_existing_types(monkeypatch, capsys, existing, raw, expected):
    settings = {"value": existing}
    monkeypatch.setattr(sys.modules["settings_manager"], "load_settings", lambda: settings)
    monkeypatch.setattr(
        sys.modules["settings_manager"], "save_settings", lambda updated: "saved"
    )

    cli.cmd_settings_set(argparse.Namespace(key="value", value=raw))

    assert settings["value"] == expected
    assert "saved" in capsys.readouterr().out
