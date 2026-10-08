"""Pinned BASE process producer and an independently verified TORQ file receiver.

The supported operation transfers geometry/state; it does not execute a chain or
claim electronic-structure results. BASE never enters TORQ's Python namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .domain import PrerequisiteError, read_json
from .ecosystem import (
    AtomIdentity,
    ConformerHandoff,
    MoleculeHandoff,
    parse_xyz,
    read_base_handoff,
    write_conformer_handoff,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_REVISION = "83462724849f1ef0be8c70ffcad6265d6af99388"
# Actual reviewed installer bytes at this immutable BASE source revision.
BASE_INSTALLER_SHA256 = (
    "7a22fc5fe91cbae865d3781f3899b38acc69f8cd9fb66648b8ea9c3eb54d9432"
)


def _environment(destination: Path) -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "COCHEM_REPO_DIR": str(ROOT),
            "COCHEM_ARTIFACT_DIR": str(destination.parent),
        }
    )
    return environment


def _run(command: list[str], destination: Path) -> dict[str, Any]:
    try:
        process = subprocess.run(
            command, env=_environment(destination), capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrerequisiteError(
            "The genuine isolated BASE process could not finish; "
            "no local substitute was executed."
        ) from exc
    if process.returncode:
        raise PrerequisiteError(
            f"The genuine BASE process rejected the operation "
            f"(exit {process.returncode}); inspect its reviewed source "
            "and installation receipts."
        )
    if not 0 < len(process.stdout) <= 1024 * 1024:
        raise PrerequisiteError("BASE returned an absent or oversized process record.")
    value: dict[str, Any] = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise PrerequisiteError("BASE returned an unsupported process record.")
    return value


class BaseHandoffBridge:
    """Run BASE's genuine producer in its verified separate environment."""

    def __init__(
        self,
        module_root: str | Path | None = None,
        module_manifest: str | Path | None = None,
    ):
        configured = module_root or os.environ.get("COCHEM_MODULE_ROOT")
        if not configured:
            raise PrerequisiteError(
                "Set COCHEM_MODULE_ROOT to the separate verified BASE installation; "
                "run scripts/student_setup.py setup first."
            )
        self.module_root = Path(configured).absolute()
        self.module_manifest = Path(
            module_manifest
            or os.environ.get("COCHEM_MODULE_MANIFEST")
            or ROOT / "ci_tools/ecosystem-modules.json"
        ).absolute()
        if not self.module_manifest.is_file():
            raise PrerequisiteError(
                "Set COCHEM_MODULE_MANIFEST to the reviewed source manifest. "
                "An installed wheel cannot infer a repository checkout."
            )

    def verify(self, destination: Path) -> dict[str, Any]:
        manifest = read_json(self.module_manifest)
        specification = manifest.get("modules", {}).get("base", {})
        if (
            manifest.get("schema_version") != "cochem.module-distribution/1"
            or specification.get("repository") != "ProfJJK-CoChem/CoChem-BASE"
            or specification.get("revision") != BASE_REVISION
        ):
            raise PrerequisiteError(
                "This BASE bridge requires its reviewed immutable installer "
                "revision; changed sources require adapter review."
            )
        source = self.module_root / "base" / BASE_REVISION / "source"
        installer = source / "scripts/manage_modules.py"
        if (
            not installer.is_file()
            or installer.is_symlink()
            or hashlib.sha256(installer.read_bytes()).hexdigest()
            != BASE_INSTALLER_SHA256
        ):
            raise PrerequisiteError(
                "The reviewed BASE installer is missing or changed; "
                "no provider code was executed."
            )
        verification = _run(
            [
                sys.executable,
                "-I",
                "-B",
                str(installer),
                "verify",
                "--manifest",
                str(self.module_manifest),
                "--root",
                str(self.module_root),
                "--modules",
                "base",
                "--json",
            ],
            destination,
        )
        receipts = verification.get("modules", [])
        if (
            len(receipts) != 1
            or receipts[0].get("module_id") != "base"
            or receipts[0].get("revision") != BASE_REVISION
        ):
            raise PrerequisiteError(
                "The genuine BASE installer returned inconsistent source identity."
            )
        receipt: dict[str, Any] = receipts[0]
        return receipt

    def prepare_geometry(
        self,
        geometry_xyz: str | Path,
        destination: str | Path,
        *,
        molecule_id: str,
        charge: int,
        multiplicity: int,
        atom_ids: list[str] | tuple[str, ...],
    ) -> ConformerHandoff:
        """Preserve explicit geometry/state/row identity without a calculation."""
        source, target = Path(geometry_xyz).absolute(), Path(destination).absolute()
        if (
            source.is_symlink()
            or not source.is_file()
            or not 0 < source.stat().st_size <= 16 * 1024 * 1024
        ):
            raise ValueError("Supply one bounded regular XYZ geometry file.")
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                "A handoff package cannot overwrite existing artifacts."
            )
        symbols, coordinates = parse_xyz(source.read_text(encoding="utf-8"))
        molecule = MoleculeHandoff(
            molecule_id=molecule_id,
            atoms=tuple(
                AtomIdentity(atom_id=identifier, symbol=symbol)
                for identifier, symbol in zip(atom_ids, symbols, strict=True)
            ),
            geometry=tuple(
                (float(row[0]), float(row[1]), float(row[2])) for row in coordinates
            ),
            geometry_unit="angstrom",
            charge=charge,
            multiplicity=multiplicity,
        )
        receipt = self.verify(target)
        program = (
            "import json,sys\nfrom cochem_base.interfaces.artifact_handoff import "
            "prepare_module_handoff,load_module_handoff\n"
            "from pathlib import Path\nstate=json.loads(sys.argv[3])\n"
            "value=prepare_module_handoff('torq',sys.argv[1],sys.argv[2],operation='geometry_analysis',options=state)\n"
            "assert load_module_handoff(Path(sys.argv[2])/'handoff.json')==value\n"
            "print(value.model_dump_json())\n"
        )
        observed = _run(
            [
                receipt["python_path"],
                "-I",
                "-B",
                "-c",
                program,
                str(source),
                str(target),
                json.dumps(molecule.model_dump(mode="json"), allow_nan=False),
            ],
            target,
        )
        if observed.get("scientific_execution_performed") is not False:
            raise ValueError(
                "A BASE geometry handoff cannot claim scientific execution."
            )
        handoff = read_base_handoff(
            target / "handoff.json",
            molecule_id=molecule_id,
            charge=charge,
            multiplicity=multiplicity,
            atom_ids=atom_ids,
            repository_revision=BASE_REVISION,
        )
        write_conformer_handoff(handoff, target / "torq-conformer.json")
        return handoff
