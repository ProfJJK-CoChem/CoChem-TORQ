"""Transport rejection checks on actual serialized molecular requests.

These checks perform no electronic calculation and do not qualify an engine.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from pathlib import Path

import pytest

from ci_tools.actions_request import decode_request, write_request


def request_bytes() -> tuple[bytes, str]:
    identifier = str(uuid.uuid4())
    value = {
        "schema_version": "cochem.torq.request/1",
        "request_id": identifier,
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            "charge": 0,
            "multiplicity": 1,
        },
        "recipe": "hf-sto-3g-education",
        "products": ["geometry"],
        "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 60},
    }
    return json.dumps(value, allow_nan=False).encode(), identifier


def transport(raw: bytes, identifier: str, *, engine="pyscf"):
    return decode_request(
        base64.b64encode(raw).decode(),
        hashlib.sha256(raw).hexdigest(),
        identifier,
        engine,
    )


def test_exact_submitted_bytes_and_immutable_transport_record(tmp_path: Path):
    raw, identifier = request_bytes()
    decoded, data = transport(raw, identifier)
    assert decoded == raw
    output = tmp_path / "request.json"
    write_request(
        output, decoded, data, source_commit="d7a4739a5f7d6f22ed659b32eeb4706bef16225e"
    )
    assert output.read_bytes() == raw
    evidence = json.loads(output.with_suffix(".transport.json").read_text())
    assert evidence["request_sha256"] == hashlib.sha256(raw).hexdigest()
    assert evidence["scientific_calculation_performed"] is False
    with pytest.raises(FileExistsError):
        write_request(output, decoded, data, source_commit=evidence["source_commit"])


def test_changed_digest_rejected():
    raw, identifier = request_bytes()
    with pytest.raises(ValueError, match="SHA-256"):
        decode_request(
            base64.b64encode(raw).decode(),
            hashlib.sha256(raw + b" ").hexdigest(),
            identifier,
        )


def test_different_request_identity_rejected():
    raw, _ = request_bytes()
    with pytest.raises(ValueError, match="request_id"):
        transport(raw, str(uuid.uuid4()))


@pytest.mark.parametrize("engine", ["orca", "cfour", "mpqc", "gpu4pyscf"])
def test_unqualified_or_different_engine_cannot_route_to_pyscf(engine):
    raw, identifier = request_bytes()
    with pytest.raises(ValueError, match="profile only"):
        transport(raw, identifier, engine=engine)


def test_duplicate_json_fields_rejected():
    raw, identifier = request_bytes()
    duplicate = raw[:-1] + b', "recipe": "hf-sto-3g-education"}'
    with pytest.raises(ValueError, match="Duplicate"):
        transport(duplicate, identifier)


def test_nonfinite_json_rejected():
    raw, identifier = request_bytes()
    changed = raw.replace(b"1.4", b"NaN")
    with pytest.raises(ValueError, match="Nonfinite"):
        transport(changed, identifier)


def test_boolean_spin_is_not_a_physical_multiplicity():
    raw, identifier = request_bytes()
    data = json.loads(raw)
    data["molecule"]["multiplicity"] = True
    with pytest.raises(ValueError, match="multiplicity"):
        transport(json.dumps(data).encode(), identifier)


@pytest.mark.parametrize(
    "field,value",
    [("cores", 3), ("memory_mb", 4096), ("wall_seconds", 601), ("cores", True)],
)
def test_classroom_resource_excess_rejected(field, value):
    raw, identifier = request_bytes()
    data = json.loads(raw)
    data["resources"][field] = value
    with pytest.raises(ValueError, match=field):
        transport(json.dumps(data).encode(), identifier)


def test_oversized_request_rejected_before_engine_installation():
    raw, identifier = request_bytes()
    with pytest.raises(ValueError, match="limit"):
        transport(raw + b" " * (16 * 1024), identifier)


def test_source_commit_cannot_be_a_moving_branch(tmp_path):
    raw, identifier = request_bytes()
    decoded, data = transport(raw, identifier)
    with pytest.raises(ValueError, match="immutable source"):
        write_request(tmp_path / "request.json", decoded, data, source_commit="main")
