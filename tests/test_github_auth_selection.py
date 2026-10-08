"""Authentication selection tests never obtain a real credential."""

import pytest

from cochem_torq.github_auth import private_gh_environment


@pytest.mark.parametrize("mode", ["auto", "stored-cli"])
def test_codespaces_selects_native_cli_without_mutating_parent(mode):
    inherited = {
        "CODESPACES": "true",
        "GH_TOKEN": "injected-session",
        "GITHUB_TOKEN": "project-session",
        "COCHEM_PRIVATE_GH_AUTH": mode,
        "GH_CONFIG_DIR": "/student/native-cli",
        "PATH": "/usr/bin",
    }
    selected = private_gh_environment(inherited)
    assert "GH_TOKEN" not in selected and "GITHUB_TOKEN" not in selected
    assert selected["GH_CONFIG_DIR"] == inherited["GH_CONFIG_DIR"]
    assert inherited["GH_TOKEN"] == "injected-session"


@pytest.mark.parametrize("mode", ["auto", "environment"])
def test_local_or_explicit_environment_auth_preserves_selected_session(mode):
    values = {"COCHEM_PRIVATE_GH_AUTH": mode, "GH_TOKEN": "account-session"}
    assert private_gh_environment(values) == values


@pytest.mark.parametrize("mode", ["auto", "stored-cli", "environment", "invalid"])
def test_actions_always_keeps_owning_project_authentication(mode):
    values = {
        "GITHUB_ACTIONS": "true",
        "CODESPACES": "true",
        "COCHEM_PRIVATE_GH_AUTH": mode,
        "GITHUB_TOKEN": "job-session",
    }
    assert private_gh_environment(values) == values


def test_invalid_local_authentication_selection_fails_explicitly():
    with pytest.raises(ValueError, match="COCHEM_PRIVATE_GH_AUTH"):
        private_gh_environment({"COCHEM_PRIVATE_GH_AUTH": "invalid"})


def test_explicit_codespaces_environment_mode_preserves_injected_session():
    values = {
        "CODESPACES": "true",
        "COCHEM_PRIVATE_GH_AUTH": "environment",
        "GH_TOKEN": "explicit-session",
    }
    assert private_gh_environment(values) == values
