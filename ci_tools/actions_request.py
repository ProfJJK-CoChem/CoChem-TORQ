"""Strict request transport and additional public classroom limits.

Input values arrive through environment variables, never through shell-code
interpolation. The JSON SHA-256 identifies the exact submitted bytes.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

MAX_REQUEST_BYTES = 16 * 1024
MAX_APPROVAL_BYTES = 32 * 1024
ALLOWED_RECIPES = {"hf-sto-3g-education", "hf-cc-pvdz-research"}
ALLOWED_ELEMENTS = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"}


def decode_request(
    encoded: str, expected_sha256: str, request_id: str, engine: str = "pyscf"
) -> tuple[bytes, dict]:
    if engine != "pyscf":
        raise ValueError(
            "This public workflow qualifies the PySCF profile only; "
            "licensed engine profiles require separate provisioned qualification"
        )
    if not isinstance(encoded, str) or not 0 < len(encoded) <= 4 * (
        (MAX_REQUEST_BYTES + 2) // 3
    ):
        raise ValueError("Base64 request exceeds the 16 KiB decoded input limit")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError(
            "Supply the exact lowercase SHA-256 of the submitted UTF-8 JSON bytes"
        )
    try:
        identifier = uuid.UUID(request_id)
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "Request ID must be a UUID and the transport must be strict base64"
        ) from exc
    if (
        not raw
        or len(raw) > MAX_REQUEST_BYTES
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError(
            "Request is empty, oversized, or differs from its submitted SHA-256"
        )

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    data = json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite
    )
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != "cochem.torq.request/1"
    ):
        raise ValueError("Supply an explicit cochem.torq.request/1 object")
    if uuid.UUID(str(data.get("request_id"))) != identifier:
        raise ValueError("Embedded request_id differs from the workflow request ID")
    if data.get("recipe") not in ALLOWED_RECIPES:
        raise ValueError(
            "This classroom workflow accepts only its explicitly enabled "
            "HF teaching/validation recipes"
        )
    molecule = data.get("molecule")
    if not isinstance(molecule, dict):
        raise ValueError("Embed a complete molecule object")
    symbols = molecule.get("symbols")
    if (
        not isinstance(symbols, list)
        or not 1 <= len(symbols) <= 12
        or any(
            not isinstance(symbol, str) or symbol not in ALLOWED_ELEMENTS
            for symbol in symbols
        )
    ):
        raise ValueError("Classroom jobs require 1–12 atoms from H through Ne")
    coordinates = molecule.get("geometry_bohr")
    if not isinstance(coordinates, list) or len(coordinates) != len(symbols):
        raise ValueError("geometry_bohr must match the atom count")
    for position in coordinates:
        if (
            not isinstance(position, list)
            or len(position) != 3
            or any(
                type(value) not in (float, int)
                or not math.isfinite(value)
                or abs(value) > 1e4
                for value in position
            )
        ):
            raise ValueError(
                "Coordinates must be three finite bounded numbers per atom"
            )
    if (
        type(molecule.get("charge")) is not int
        or abs(molecule["charge"]) > 4
        or type(molecule.get("multiplicity")) is not int
        or molecule.get("multiplicity") != 1
    ):
        raise ValueError(
            "This workflow requires an explicit bounded charge "
            "and closed-shell singlet multiplicity"
        )
    resources = data.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("Resources must be an object")
    for field, default, lower, upper in (
        ("cores", 1, 1, 2),
        ("memory_mb", 2048, 256, 2048),
        ("wall_seconds", 600, 1, 600),
    ):
        value = resources.get(field, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(
                f"Classroom {field} must be an integer in [{lower}, {upper}]"
            )
    if not isinstance(data.get("products"), list) or not data["products"]:
        raise ValueError("Request at least one explicit scientific product")
    return raw, data


def write_request(output: Path, raw: bytes, data: dict, *, source_commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Execution requires an immutable source commit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        target.write(raw)
    evidence = {
        "schema_version": "cochem.actions-request/1",
        "request_id": data["request_id"],
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": source_commit,
        "request_bytes": len(raw),
        "engine_profile": "pyscf-linux-cpu",
        "validated_transport": True,
        "scientific_calculation_performed": False,
    }
    with output.with_suffix(".transport.json").open("x") as target:
        json.dump(evidence, target, indent=2, allow_nan=False)
        target.write("\n")


def decode_approval(
    encoded: str, expected_sha256: str, request: dict, *, source_commit: str
) -> tuple[bytes, dict]:
    """Verify approval transport; the worker validates scientific semantics."""
    if not isinstance(encoded, str) or not 0 < len(encoded) <= 4 * (
        (MAX_APPROVAL_BYTES + 2) // 3
    ):
        raise ValueError(
            "The complete reviewed approval must fit the 32 KiB transport limit."
        )
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("Supply the exact reviewed approval byte SHA-256.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("The reviewed approval must use strict base64.") from exc
    if (
        not raw
        or len(raw) > MAX_APPROVAL_BYTES
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("Reviewed approval bytes differ from their submitted SHA-256.")

    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate approval field: {key}")
            value[key] = item
        return value

    def nonfinite(value):
        raise ValueError(f"Nonfinite approval value: {value}")

    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite
    )
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "cochem.torq.approved-plan/1"
    ):
        raise ValueError("Supply the complete versioned reviewed plan approval.")
    plan = value.get("plan")
    approval = value.get("approval")
    identity = value.get("source_identity")
    if (
        not isinstance(plan, dict)
        or not isinstance(approval, dict)
        or not isinstance(identity, dict)
    ):
        raise ValueError(
            "Reviewed approval requires plan, approval and source identities."
        )
    if plan.get("request") != request or approval.get("plan_sha256") != plan.get(
        "plan_sha256"
    ):
        raise ValueError("The reviewed plan does not bind this complete request.")
    if (
        identity.get("git_commit") is not None
        and identity["git_commit"] != source_commit
    ):
        raise ValueError(
            "The reviewed source commit differs from this worker checkout."
        )
    return raw, value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-plan-output", type=Path, required=True)
    args = parser.parse_args()
    raw, request = decode_request(
        os.environ["TORQ_REQUEST_B64"],
        os.environ["TORQ_REQUEST_SHA256"],
        os.environ["TORQ_REQUEST_ID"],
        os.environ.get("TORQ_ENGINE", "pyscf"),
    )
    expected_commit = os.environ["TORQ_EXPECTED_SOURCE_SHA"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", expected_commit)
        or expected_commit != os.environ["GITHUB_SHA"]
    ):
        raise ValueError(
            "The workflow source differs from the immutable commit "
            "approved at submission"
        )
    scientific_commit = os.environ.get("TORQ_SCIENTIFIC_SOURCE_SHA", expected_commit)
    if not re.fullmatch(r"[0-9a-f]{40}", scientific_commit):
        raise ValueError("The scientific worker requires an immutable source commit")
    approval_raw, approval = decode_approval(
        os.environ["TORQ_APPROVED_PLAN_B64"],
        os.environ["TORQ_APPROVED_PLAN_SHA256"],
        request,
        source_commit=scientific_commit,
    )
    write_request(args.output, raw, request, source_commit=os.environ["GITHUB_SHA"])
    with args.approved_plan_output.open("xb") as stream:
        stream.write(approval_raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": "transport_validated",
                "request_id": request["request_id"],
                "request_bytes": len(raw),
            }
        )
    )
