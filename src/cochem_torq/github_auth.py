"""Select GitHub CLI authentication without reading stored credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping


def private_gh_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Use a student's native gh login in Codespaces unless explicitly overridden.

    Actions keeps its owning project's injected authentication. No mode obtains
    a stored credential; authentication remains the official CLI's responsibility.
    """
    selected = dict(os.environ if environment is None else environment)
    if selected.get("GITHUB_ACTIONS", "").lower() == "true":
        return selected
    mode = selected.get("COCHEM_PRIVATE_GH_AUTH", "auto")
    if mode not in {"auto", "stored-cli", "environment"}:
        raise ValueError(
            "COCHEM_PRIVATE_GH_AUTH must be auto, stored-cli, or environment."
        )
    if mode == "stored-cli" or (
        mode == "auto" and selected.get("CODESPACES", "").lower() == "true"
    ):
        selected.pop("GH_TOKEN", None)
        selected.pop("GITHUB_TOKEN", None)
    return selected
