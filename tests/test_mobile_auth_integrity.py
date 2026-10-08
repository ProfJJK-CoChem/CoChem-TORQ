"""Actual HMAC, file, database and loopback HTTP contracts; no engine substitutes."""

import hashlib
import hmac
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path

import pytest

from cochem.mobile.airgap_receiver import (
    AirGapReceiverHTTPRequestHandler,
    ThreadedHTTPServer,
    make_airgap_receiver_server,
    verify_hmac_signature,
)
from cochem.mobile.job_state import ExecutionPayload, ExecutionTier, ManifestReference
from cochem.mobile.payload_serializer import (
    canonical_json_dumps,
    canonical_serialize,
    ensure_tripartite_dirs,
    get_hmac_secret,
    get_job_artifact_dir,
    get_job_lock_path,
    get_job_status_path,
    load_staged_payload,
    sign_payload,
    stage_or_inline_payload,
    strict_json_loads,
    validate_xyz_structure_dynamic,
    verify_payload_signature,
)
from Libraries.cochem_isotopes import isotope_mass


def independent_signature(body, secret):
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def test_core_imports_do_not_load_optional_molecular_implementations():
    script = """
import importlib.util
import sys
import cochem.mobile as mobile
from cochem.mobile import airgap_receiver, payload_serializer
assert "rdkit" not in sys.modules
assert "cochem.mobile.assembly" not in sys.modules
assert "cochem.mobile.conformer_engine" not in sys.modules
assert mobile.sign_payload is payload_serializer.sign_payload
assert mobile.make_airgap_receiver_server is airgap_receiver.make_airgap_receiver_server
assert "generate_3d_conformer" in dir(mobile)
try:
    mobile.no_such_mobile_api
except AttributeError:
    pass
else:
    raise AssertionError("Unknown public API was accepted")
if importlib.util.find_spec("rdkit") is None:
    try:
        mobile.generate_3d_conformer
    except ImportError as error:
        assert "rdkit" in str(error)
    else:
        raise AssertionError("Missing RDKit was silently accepted")
else:
    from cochem.mobile.conformer_engine import generate_3d_conformer
    assert mobile.generate_3d_conformer is generate_3d_conformer
from cochem.mobile import inorganic
from cochem.mobile.inorganic import models
assert "cochem.mobile.inorganic.ui" not in sys.modules
assert "anywidget" not in sys.modules
assert "ipywidgets" not in sys.modules
assert inorganic.MetalCenter is models.MetalCenter
assert mobile.calculate_formula_weight is models.calculate_formula_weight
assert models.calculate_formula_weight("H2O") > 0
notebook_dependencies = ("anywidget", "ipywidgets", "traitlets")
if any(importlib.util.find_spec(name) is None for name in notebook_dependencies):
    try:
        inorganic.InorganicBuilderWidget
    except ImportError as error:
        assert any(name in str(error) for name in notebook_dependencies)
    else:
        raise AssertionError("Missing notebook dependencies were silently accepted")
else:
    from cochem.mobile.inorganic.ui import InorganicBuilderWidget
    assert inorganic.InorganicBuilderWidget is InorganicBuilderWidget
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def execution_payload(tmp_path, *, staged=False):
    # An explicit coordinate input tests serialization, not molecular qualification.
    return ExecutionPayload(
        job_id="job-source",
        created_at_utc="2026-10-07T00:00:00+00:00",
        tier=ExecutionTier.LOCAL_TIER_1,
        workflow_type="SERIALIZATION_CONTRACT",
        molecule_xyz="2\n\nH 0 0 0\nH 0 0 1\n",
        parameters={"input_text": "x" * 65536} if staged else {},
        output_artifact_dir=str(tmp_path / "output"),
    )


@contextmanager
def running_receiver(tmp_path, secret):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    server, port = make_airgap_receiver_server(
        scratch_dir=tmp_path / "scratch",
        src_dir=source,
        db_path=tmp_path / "ledger.sqlite",
        secret_key=secret,
    )
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def request(port, body, signature=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        headers = {"Content-Type": "application/json"}
        if signature is not None:
            headers["X-CoChem-Signature"] = signature
        connection.request("POST", "/", body, headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_signatures_use_exact_explicit_key_and_canonical_bytes():
    secret = " " + secrets.token_hex(32) + " "
    data = {"z": "Å", "a": {"number": 1}, "hmac_sha256": "excluded"}
    body = canonical_serialize(data)
    assert body == '{"a":{"number":1},"z":"Å"}'.encode("utf-8")
    signature = independent_signature(body, secret)
    assert get_hmac_secret(secret) == secret
    assert sign_payload(body, secret) == signature
    assert verify_payload_signature(body, signature, secret)
    assert verify_hmac_signature(body, "sha256=" + signature.upper(), secret)
    assert not verify_payload_signature(body + b" ", signature, secret)
    assert not verify_hmac_signature(body, signature, secrets.token_hex(32))


@pytest.mark.parametrize("signature", [None, "", "é", "0" * 63, "g" * 64])
def test_malformed_signatures_reject_without_crypto_type_errors(signature):
    secret = secrets.token_hex(32)
    assert not verify_payload_signature(b"{}", signature, secret)
    assert not verify_hmac_signature(b"{}", signature, secret)


def test_missing_configuration_fails_before_socket_or_file_creation(tmp_path):
    environment = dict(os.environ)
    environment.pop("COCHEM_HMAC_SECRET", None)
    script = """
import sys
from pathlib import Path
from cochem.mobile.airgap_receiver import make_airgap_receiver_server
from cochem.mobile.job_state import ManifestReference
from cochem.mobile.payload_serializer import get_hmac_secret, sign_payload, verify_payload_signature, load_staged_payload
root = Path(sys.argv[1])
manifest = ManifestReference(job_id="unconfigured", manifest_uri=str(root / "missing.json"), file_size_bytes=0, hmac_sha256="0" * 64, created_at_utc="2026-10-07T00:00:00+00:00")
operations = [lambda: get_hmac_secret(), lambda: sign_payload(b"{}"), lambda: verify_payload_signature(b"{}", ""), lambda: load_staged_payload(manifest), lambda: make_airgap_receiver_server(scratch_dir=root / "scratch", src_dir=root / "source", db_path=root / "state" / "ledger.sqlite")]
for operation in operations:
    try:
        operation()
    except ValueError:
        pass
    else:
        raise AssertionError("Unconfigured authentication accepted")
assert not (root / "scratch").exists()
assert not (root / "state").exists()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_environment_key_is_explicit_and_blank_override_does_not_fall_back():
    environment = dict(os.environ)
    environment["COCHEM_HMAC_SECRET"] = " " + secrets.token_hex(32) + " "
    script = """
import hashlib, hmac, os
from cochem.mobile.payload_serializer import get_hmac_secret, sign_payload
key = os.environ["COCHEM_HMAC_SECRET"]
assert get_hmac_secret() == key
assert sign_payload(b"{}") == hmac.new(key.encode(), b"{}", hashlib.sha256).hexdigest()
for invalid in ("", " ", 0):
    try:
        get_hmac_secret(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid explicit key accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_cannot_be_serialized_or_signed(value):
    with pytest.raises(ValueError):
        canonical_json_dumps({"nested": [value]})
    with pytest.raises(ValueError):
        canonical_serialize({"nested": [value]})


@pytest.mark.parametrize(
    "raw",
    [
        '{"nested":{"x":1,"x":2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e999}',
    ],
)
def test_nonstandard_json_and_duplicate_keys_are_rejected(raw):
    with pytest.raises(ValueError):
        strict_json_loads(raw)


@pytest.mark.parametrize(
    "xyz",
    [
        "2\n\nH 0 0 0\nH 0 0 1\n",
        "2\ncomment\nH 0 0 0\nH 0 0 1\n",
    ],
)
def test_xyz_preserves_all_atoms_and_blank_comment(xyz):
    atoms = validate_xyz_structure_dynamic(xyz)
    actual_weight = isotope_mass("1H")
    assert atoms == [
        ("1H", 0.0, 0.0, 0.0, actual_weight),
        ("1H", 0.0, 0.0, 1.0, actual_weight),
    ]


@pytest.mark.parametrize(
    "xyz",
    [
        "",
        "0\n\n",
        "-1\n\n",
        "1",
        "2\n\nH 0 0 0\n",
        "1\n\nH 0 0 0\nH 0 0 1\n",
        "2\n\nH 0 0 0\n\n",
        "H 0 0",
        "H 0 0 0 ignored",
        "H1 0 0 0",
        "13C 0 0 0",
        "Xx 0 0 0",
        "H NaN 0 0",
        "H inf 0 0",
        "H 1e999 0 0",
    ],
)
def test_invalid_xyz_never_drops_an_atom_or_invents_a_mass(xyz):
    with pytest.raises(ValueError):
        validate_xyz_structure_dynamic(xyz)


def test_xyz_retains_explicit_isotope_identity_and_actual_mass():
    atoms = validate_xyz_structure_dynamic("2\nIsotope input\n13C 0 0 0\nD 0 0 1\n")
    assert atoms == [
        ("13C", 0.0, 0.0, 0.0, isotope_mass("13C")),
        ("2H", 0.0, 0.0, 1.0, isotope_mass("2H")),
    ]
    with pytest.raises(ValueError, match="No tabulated mass"):
        validate_xyz_structure_dynamic("1\nUnknown isotope\n999C 0 0 0\n")


@pytest.mark.parametrize(
    "atom_row", ["H nan 0 0", "H inf 0 0", "H 1e999 0 0", "Xx 0 0 0"]
)
def test_invalid_atom_inside_standard_frame_is_rejected(atom_row):
    with pytest.raises(ValueError):
        validate_xyz_structure_dynamic(f"1\nActual frame\n{atom_row}\n")


def test_invalid_key_rejects_before_binding_an_actual_occupied_port(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        with pytest.raises(ValueError, match="authentication key"):
            make_airgap_receiver_server(
                port=occupied.getsockname()[1],
                scratch_dir=tmp_path / "scratch",
                src_dir=tmp_path / "source",
                db_path=tmp_path / "ledger.sqlite",
                secret_key="",
            )
    assert list(tmp_path.iterdir()) == []


def test_actual_inline_and_staged_files_round_trip(tmp_path):
    secret = secrets.token_hex(32)
    payload = execution_payload(tmp_path)
    staged, inline = stage_or_inline_payload(payload, secret, tmp_path / "inline")
    assert not staged and payload.hmac_sha256 is None
    assert inline.hmac_sha256 == independent_signature(
        canonical_serialize(payload), secret
    )
    assert not (tmp_path / "inline").exists()
    payload = execution_payload(tmp_path, staged=True)
    staged, manifest = stage_or_inline_payload(payload, secret, tmp_path / "staged")
    assert staged
    raw = Path(manifest.manifest_uri).read_bytes()
    assert raw == canonical_serialize(payload)
    assert manifest.file_size_bytes == len(raw)
    assert manifest.hmac_sha256 == independent_signature(raw, secret)
    loaded = load_staged_payload(manifest, secret)
    assert canonical_serialize(loaded) == raw
    assert loaded.hmac_sha256 == manifest.hmac_sha256
    assert not (tmp_path / "staged" / "payload.json.tmp").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "other-job"),
        ("created_at_utc", "2026-10-08T00:00:00+00:00"),
    ],
)
def test_manifest_identity_cannot_relabel_an_authentic_staged_file(
    tmp_path, field, value
):
    secret = secrets.token_hex(32)
    _, manifest = stage_or_inline_payload(
        execution_payload(tmp_path, staged=True), secret, tmp_path
    )
    relabeled = manifest.model_copy(update={field: value})
    with pytest.raises(ValueError, match="manifest identity"):
        load_staged_payload(relabeled, secret)


def test_staged_file_size_signature_and_missing_file_are_checked(tmp_path):
    secret = secrets.token_hex(32)
    _, manifest = stage_or_inline_payload(
        execution_payload(tmp_path, staged=True), secret, tmp_path
    )
    with pytest.raises(ValueError, match="integrity verification"):
        load_staged_payload(manifest, secrets.token_hex(32))
    target = Path(manifest.manifest_uri)
    raw = target.read_bytes()
    target.write_bytes(raw.replace(b"job-source", b"job-tamper"))
    assert target.stat().st_size == manifest.file_size_bytes
    with pytest.raises(ValueError, match="integrity verification"):
        load_staged_payload(manifest, secret)
    target.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        load_staged_payload(manifest, secret)
    target.unlink()
    with pytest.raises(FileNotFoundError):
        load_staged_payload(manifest, secret)


@pytest.mark.parametrize("addition", [',"parameters":{"x":NaN}', ',"job_id":"other"'])
def test_authenticated_malformed_staged_json_is_still_rejected(tmp_path, addition):
    secret = secrets.token_hex(32)
    payload = execution_payload(tmp_path)
    raw = canonical_serialize(payload)[:-1] + addition.encode() + b"}"
    target = tmp_path / "payload.json"
    target.write_bytes(raw)
    manifest = ManifestReference(
        job_id=payload.job_id,
        created_at_utc=payload.created_at_utc,
        manifest_uri=str(target),
        file_size_bytes=len(raw),
        hmac_sha256=independent_signature(raw, secret),
    )
    with pytest.raises(ValueError):
        load_staged_payload(manifest, secret)


@pytest.mark.parametrize("job_id", ["../escape", "/absolute", "a/b", "a\\b", "", 1])
def test_job_ids_cannot_escape_storage_paths(tmp_path, job_id):
    for operation in (
        lambda: get_job_artifact_dir(job_id, tmp_path),
        lambda: get_job_status_path(job_id, tmp_path),
        lambda: get_job_lock_path(job_id, tmp_path),
        lambda: ensure_tripartite_dirs(
            job_id, tmp_path / "src", tmp_path / "artifacts", tmp_path / "state"
        ),
    ):
        with pytest.raises(ValueError):
            operation()
    assert list(tmp_path.iterdir()) == []


def test_two_actual_receivers_keep_keys_files_and_ledgers_isolated(tmp_path):
    secret_a, secret_b = secrets.token_hex(32), secrets.token_hex(32)
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    body = b'{ "job_id" : "request-source", "parameters" : {"x": 1} }'
    with (
        running_receiver(root_a, secret_a) as port_a,
        running_receiver(root_b, secret_b) as port_b,
    ):
        signature_a = independent_signature(body, secret_a)
        signature_b = independent_signature(body, secret_b)
        assert request(port_a, body, signature_b)[0] == 401
        assert request(port_b, body, signature_a)[0] == 401
        assert request(port_a, body)[0] == 401
        assert request(port_a, body + b" ", signature_a)[0] == 401
        for port, root, signature in (
            (port_a, root_a, signature_a),
            (port_b, root_b, signature_b),
        ):
            status, response = request(port, body, "sha256=" + signature)
            assert status == 200 and response["job_id"] == "request-source"
            artifact = Path(response["scratch_file"])
            assert artifact.parent == root / "scratch"
            assert artifact.read_bytes() == body
            assert list((root / "source").iterdir()) == []
            with sqlite3.connect(root / "ledger.sqlite") as connection:
                assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
                rows = connection.execute(
                    "SELECT job_id, signature, scratch_file, payload_size, status FROM airgap_transactions"
                ).fetchall()
            assert rows == [
                (
                    "request-source",
                    "sha256=" + signature,
                    str(artifact),
                    len(body),
                    "COMMITTED",
                )
            ]


@pytest.mark.parametrize(
    "body",
    [
        b'{"job_id":"../escape"}',
        b'{"job_id":null}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b'{"job_id":"first","job_id":"second"}',
        b"[]",
        b'{"molecule_xyz":"H 0 0"}',
    ],
)
def test_authenticated_invalid_requests_write_no_payload_or_ledger(tmp_path, body):
    secret = secrets.token_hex(32)
    with running_receiver(tmp_path, secret) as port:
        assert request(port, body, independent_signature(body, secret))[0] == 422
    assert list((tmp_path / "scratch").iterdir()) == []
    assert not (tmp_path / "ledger.sqlite").exists()


def test_source_destination_and_symlink_are_rejected_over_http(tmp_path):
    secret = secrets.token_hex(32)
    with running_receiver(tmp_path, secret) as port:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path / "source", target_is_directory=True)
        for destination in (tmp_path / "source" / "write.json", alias / "write.json"):
            body = canonical_json_dumps(
                {"job_id": "blocked", "output_artifact_dir": str(destination)}
            ).encode()
            assert request(port, body, independent_signature(body, secret))[0] == 403
    assert list((tmp_path / "source").iterdir()) == []
    assert list((tmp_path / "scratch").iterdir()) == []
    assert not (tmp_path / "ledger.sqlite").exists()


def test_receiver_cannot_configure_storage_inside_protected_source(tmp_path):
    secret = secrets.token_hex(32)
    source = tmp_path / "source"
    for scratch, ledger in (
        (source / "scratch", tmp_path / "ledger.sqlite"),
        (tmp_path / "scratch", source / "ledger.sqlite"),
    ):
        with pytest.raises(ValueError, match="outside the protected source"):
            make_airgap_receiver_server(
                scratch_dir=scratch, src_dir=source, db_path=ledger, secret_key=secret
            )
    assert list(tmp_path.iterdir()) == []


def test_actual_unconfigured_handler_returns_unavailable_without_writes(tmp_path):
    # This is the genuine base handler on an actual HTTP socket, with no config.
    server = ThreadedHTTPServer(("127.0.0.1", 0), AirGapReceiverHTTPRequestHandler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        body = b'{"job_id":"unconfigured"}'
        assert (
            request(
                server.server_address[1],
                body,
                independent_signature(body, secrets.token_hex(32)),
            )[0]
            == 503
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert list(tmp_path.iterdir()) == []
