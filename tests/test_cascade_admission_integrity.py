"""Real legacy CLI admission rejects planning declarations before geometry/dispatch."""

from __future__ import annotations

import json

import pytest

from Libraries.cochem_torq_cli import main


@pytest.mark.parametrize(
    "declaration",
    [
        {"schema_version": "cochem.torq.legacy-cascade-planning/1"},
        {"status": "planning_only"},
        {"dispatch_authorized": False},
    ],
)
def test_planning_or_prohibited_dispatch_cannot_use_cli_runtime_defaults(
    tmp_path, capsys, declaration
):
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(declaration))
    output = tmp_path / "results"
    status = main(
        [
            "--config",
            str(config),
            "--input",
            str(tmp_path / "absent.xyz"),
            "--output-dir",
            str(output),
            "--json",
        ]
    )
    response = json.loads(capsys.readouterr().out)
    assert status == 1
    assert response["status"] == "FAILED"
    assert response["error_type"] == "DispatchNotAuthorized"
    assert not output.exists()


def test_configuration_array_rejects_before_runtime_default_selection(tmp_path, capsys):
    config = tmp_path / "array.json"
    config.write_text("[]")
    status = main(["--config", str(config), "--json"])
    response = json.loads(capsys.readouterr().out)
    assert status == 1
    assert response["error_type"] == "InvalidConfiguration"
