"""Verified TOPOS members as new TORQ inputs across separate interpreters.

The producer owns ensemble verification. TORQ preserves the original handoff
and starts a separately reviewed calculation; source results are never reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .domain import CalculationRequest, canonical_json, read_json
from .ecosystem import MAX_ARTIFACT_BYTES, AtomIdentity, MoleculeHandoff

_VERIFY = """
import hashlib, importlib.metadata, json, sys
from pathlib import Path
from topos import review
from topos.review import verify_torq_handoff
p = Path(sys.argv[1])
original = p.read_bytes()
handoff = verify_torq_handoff(p)
if p.read_bytes() != original:
    raise ValueError('TOPOS source changed during verification')
print(json.dumps({
    'handoff': handoff,
    'source_sha256': hashlib.sha256(original).hexdigest(),
    'producer': {
        'python': sys.executable,
        'distribution': 'CoChem-TOPOS',
        'version': importlib.metadata.version('CoChem-TOPOS'),
        'verifier_file': str(Path(review.__file__).absolute()),
        'verifier_sha256': hashlib.sha256(
            Path(review.__file__).read_bytes()).hexdigest(),
    },
}, allow_nan=False, separators=(',', ':')))
"""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_reviewed_topos(
    handoff_path: str | Path,
    *,
    producer_python: str | Path,
    expected_handoff_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify exact bytes using the explicitly selected installed producer.

    This is integrity/contract verification, not sender authentication, source
    certification, engine execution, or a TORQ consumption receipt.
    """
    source = Path(handoff_path).expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError("Supply a regular, nonsymlink TOPOS handoff JSON file.")
    original = source.read_bytes()
    if not 0 < len(original) <= MAX_ARTIFACT_BYTES:
        raise ValueError("A TOPOS handoff must be nonempty and at most 16 MiB.")
    # Reject ambiguous/nonfinite JSON before invoking the trusted producer.
    read_json(source)
    python = Path(producer_python).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Select the installed TOPOS environment's Python executable.")
    if expected_handoff_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_handoff_sha256
    ):
        raise ValueError("Expected handoff identity must be a lowercase SHA-256.")
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP"):
        environment.pop(name, None)
    try:
        with tempfile.TemporaryDirectory(prefix="cochem_exec_topos_verify_") as work:
            completed = subprocess.run(
                [str(python), "-I", "-B", "-c", _VERIFY, str(source)],
                cwd=work,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            "Installed TOPOS verification exceeded its 60-second limit."
        ) from error
    if completed.returncode:
        diagnostic = completed.stderr.strip()[-2000:]
        raise ValueError(f"Installed TOPOS producer rejected the handoff: {diagnostic}")
    if len(completed.stdout.encode("utf-8")) > MAX_ARTIFACT_BYTES * 2:
        raise ValueError(
            "Installed producer returned an oversized verification record."
        )
    observation = json.loads(completed.stdout)
    if not isinstance(observation, dict) or not isinstance(
        observation.get("producer"), dict
    ):
        raise ValueError("Installed producer returned an invalid verification record.")
    if (
        source.read_bytes() != original
        or observation.get("source_sha256") != _digest(original)
        or observation.get("producer", {}).get("python") != str(python)
    ):
        raise ValueError(
            "Producer interpreter or handoff bytes changed during verification."
        )
    handoff = observation["handoff"]
    if (
        expected_handoff_sha256 is not None
        and handoff.get("handoff_sha256") != expected_handoff_sha256
    ):
        raise ValueError("Verified TOPOS ensemble differs from the selected handoff.")
    return observation


def import_reviewed_topos_member(
    handoff_path: str | Path,
    destination: str | Path,
    *,
    producer_python: str | Path,
    member_id: str,
    recipe: str,
    products: list[str],
    resources: dict[str, int],
    expected_handoff_sha256: str | None = None,
) -> dict[str, Any]:
    """Retain the original producer evidence and return a new typed request.

    The target recipe and budget are explicit independent choices. Original
    topology, fragments, constraints and calculated quantities remain in the
    retained source; this geometry entry does not import them as target results.
    """
    observation = inspect_reviewed_topos(
        handoff_path,
        producer_python=producer_python,
        expected_handoff_sha256=expected_handoff_sha256,
    )
    handoff = observation["handoff"]
    matching = [m for m in handoff["members"] if m["member_id"] == member_id]
    if len(matching) != 1:
        raise ValueError("Select exactly one member ID from the verified ensemble.")
    member = matching[0]
    source_molecule = member["molecule"]
    if source_molecule.get("coordinate_units") != "angstrom":
        raise ValueError(
            "The reviewed TOPOS geometry contract requires angstrom units."
        )
    molecule = MoleculeHandoff(
        molecule_id=member_id,
        atoms=tuple(
            AtomIdentity(atom_id=atom_id, symbol=symbol, isotope_mass_number=isotope)
            for atom_id, symbol, isotope in zip(
                source_molecule["atom_ids"],
                source_molecule["symbols"],
                source_molecule["isotopes"],
                strict=True,
            )
        ),
        geometry=source_molecule["coordinates"],
        geometry_unit="angstrom",
        charge=source_molecule["charge"],
        multiplicity=source_molecule["multiplicity"],
    )
    attempts = [
        a
        for a in handoff["record"]["attempts"]
        if a["attempt_id"] == member["attempt_id"]
    ]
    if len(attempts) != 1:
        raise ValueError("Selected member must identify one exact producer attempt.")
    attempt = attempts[0]
    source_request = handoff["request"]
    source_recipe = {
        name: source_request.get(name)
        for name in (
            "purpose",
            "engine",
            "engine_version",
            "method",
            "basis",
            "auxiliary_basis",
            "dispersion",
            "solvent",
            "constraints",
            "profile_id",
            "matrix_revision",
            "matrix_row_id",
            "matrix_product",
            "matrix_inputs",
            "search_algorithm",
            "sampler_profile",
            "sampler_nci",
            "seed",
        )
    }
    from .registry import get_profile

    get_profile(recipe)  # Named target availability remains a separate plan check.
    provenance = {
        "reviewed_topos": {
            "schema_version": "cochem.torq-reviewed-topos-input/1",
            "scope": "verified_geometry_as_new_calculation_input",
            "source_file": "source-handoff.json",
            "source_sha256": observation["source_sha256"],
            "source_bytes": Path(handoff_path).stat().st_size,
            "run_id": handoff["run_id"],
            "handoff_sha256": handoff["handoff_sha256"],
            "ensemble_sha256": handoff["ensemble_sha256"],
            "source_record_sha256": handoff["source_record_sha256"],
            "source_snapshot_sha256": handoff["source_snapshot_sha256"],
            "member_id": member_id,
            "geometry_sha256": member["geometry_sha256"],
            "attempt_id": member["attempt_id"],
            "comparison_protocol": member["comparison_protocol"],
            "source_recipe": source_recipe,
            "observed_attempt": {
                name: attempt.get(name)
                for name in (
                    "engine",
                    "engine_version",
                    "method",
                    "status",
                    "converged",
                )
            },
            "producer_verification": observation["producer"],
            "geometry_status": "imported_input",
            "computation_performed": False,
            "engine_checkpoint_reused": False,
            "source_result_reused": False,
            "limitations": [
                "Only geometry, atom/isotope identities and electronic state "
                "enter the new request.",
                "Original topology, fragments, environment, constraints and "
                "quantities remain source observations.",
                "Review the target method, physical capabilities and budget "
                "before any calculation.",
            ],
        }
    }
    request = CalculationRequest.model_validate(
        {
            "molecule": molecule.to_application_molecule(),
            "recipe": recipe,
            "products": products,
            "resources": resources,
            "source_provenance": provenance,
        }
    ).model_dump(mode="json")
    target = Path(destination).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise ValueError(
            "A TOPOS import requires a new destination; "
            "originals are never overwritten."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    original = Path(handoff_path).read_bytes()
    if _digest(original) != observation["source_sha256"]:
        raise ValueError("TOPOS source changed before canonical import.")
    temporary = Path(tempfile.mkdtemp(prefix=".topos-import-", dir=target.parent))
    try:
        payloads = {
            "source-handoff.json": original,
            "request.json": canonical_json(request) + b"\n",
            "verification.json": canonical_json(
                {
                    "source_sha256": observation["source_sha256"],
                    "producer": observation["producer"],
                    "handoff_sha256": handoff["handoff_sha256"],
                    "member_id": member_id,
                    "computation_performed": False,
                    "source_result_reused": False,
                }
            )
            + b"\n",
        }
        inventory = {}
        for name, payload in payloads.items():
            with (temporary / name).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            (temporary / name).chmod(0o444)
            inventory[name] = {"bytes": len(payload), "sha256": _digest(payload)}
        with (temporary / "manifest.json").open("xb") as stream:
            stream.write(
                canonical_json(
                    {
                        "schema_version": "cochem.torq-reviewed-topos-import/1",
                        "files": inventory,
                    }
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        (temporary / "manifest.json").chmod(0o444)
        directory_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.rename(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return request
