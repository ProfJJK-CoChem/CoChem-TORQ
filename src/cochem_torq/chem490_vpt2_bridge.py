"""CHEM 490 Wk 7/9 VdW Torsional Scan + ORCA 6.1.1 VPT2 Handoff Bridge.

Provides:
1. ``parse_orca_vpt2_rotational_constants``: Parses authentic ORCA 6.1.1
   rotational constants (Be in cm^-1/MHz), VPT2 vibrational corrections
   (Delta B_vib = B0 - Be -> A0, B0, C0), harmonic/anharmonic ZPVE, quartic
   centrifugal distortion constants, and principal-axis dipole components with
   SHA-256 provenance of the raw ``.out`` file. Strictly separates equilibrium
   Be from ground-state B0: when no VPT2 block is present, ``b0_status`` is
   ``"PENDING"`` and ``delta_b_vib_mhz`` is ``None`` (never ``0.0``).
2. ``export_spyfy_geom_handoff``: Exports a versioned JSON + Pickett
   ``.var`` / ``.int`` seed packet (``cochem.torq.vpt2-spectroscopy-handoff/1``,
   ``collaborator_institution: "Tennessee Tech University"``) for CoChem-SPYFY
   and CoChem-GEOM semi-experimental ``r_e^SE`` analysis (with
   ``re_se_status = "PENDING"`` until isotope-specific Delta B_vib and
   experimental B0 are supplied).
3. ``run_relaxed_1d_torsional_scan``: Executes a genuine PySCF + geomeTRIC
   constrained 1D relaxed torsional PES scan (e.g., H2O2 internal rotation)
   returning barrier heights in cm^-1 and kcal/mol with SHA-256 provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .units import CONSTANTS_PROFILE, conversion_factor

HANDOFF_SCHEMA: Literal["cochem.torq.vpt2-spectroscopy-handoff/1"] = (
    "cochem.torq.vpt2-spectroscopy-handoff/1"
)
SCAN_SCHEMA: Literal["cochem.torq.relaxed-torsional-scan/1"] = (
    "cochem.torq.relaxed-torsional-scan/1"
)
COLLABORATOR_INSTITUTION: Literal["Tennessee Tech University"] = (
    "Tennessee Tech University"
)
CM1_TO_MHZ = conversion_factor("cm^-1", "MHz", equivalence="spectroscopic")
HARTREE_TO_CM1 = conversion_factor("hartree", "cm^-1", equivalence="spectroscopic")
HARTREE_TO_KCAL_MOL = conversion_factor("hartree", "kcal/mol", equivalence="molar")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RotationalTriple(_StrictModel):
    """Principal-axis rotational constants ordered A >= B >= C."""

    a: float
    b: float
    c: float
    unit: Literal["cm^-1", "MHz"]


class VibrationalCorrectionTriple(_StrictModel):
    """Anharmonic vibrational shifts Delta B_vib = B_0 - B_e (A, B, C)."""

    delta_a: float
    delta_b: float
    delta_c: float
    unit: Literal["cm^-1", "MHz"]


class DipoleComponents(_StrictModel):
    """Electric dipole components along rotational axes (a, b, c) in Debye."""

    mu_a_debye: float
    mu_b_debye: float
    mu_c_debye: float
    magnitude_debye: float
    method_label: str | None = None


class QuarticDistortionConstants(_StrictModel):
    """Watson S- and/or A-reduced quartic centrifugal distortion constants."""

    s_reduced_cm1: dict[str, float] | None = None
    s_reduced_mhz: dict[str, float] | None = None
    a_reduced_cm1: dict[str, float] | None = None
    a_reduced_mhz: dict[str, float] | None = None


class ZpveSummary(_StrictModel):
    """Harmonic and anharmonic zero-point vibrational energy summary."""

    harmonic_cm1: float | None = None
    harmonic_hartree: float | None = None
    anharmonic_correction_cm1: float | None = None
    rotvib_correction_cm1: float | None = None
    total_anharmonic_cm1: float | None = None


class OrcaVpt2RotationalRecord(_StrictModel):
    """Authenticated rotational and VPT2 spectroscopy record from ORCA 6.1.1."""

    schema_version: Literal["cochem.torq.orca-vpt2-rotational/1"] = (
        "cochem.torq.orca-vpt2-rotational/1"
    )
    source_path: str
    source_filename: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(gt=0)
    orca_version: str | None
    terminated_normally: bool
    has_vpt2_block: bool
    b0_status: Literal["AVAILABLE", "PENDING"]
    be_cm1: RotationalTriple
    be_mhz: RotationalTriple
    delta_b_vib_cm1: VibrationalCorrectionTriple | None
    delta_b_vib_mhz: VibrationalCorrectionTriple | None
    b0_cm1: RotationalTriple | None
    b0_mhz: RotationalTriple | None
    zpve: ZpveSummary
    quartic_cdc: QuarticDistortionConstants | None
    dipole: DipoleComponents | None
    constants_profile: str = CONSTANTS_PROFILE
    scientific_notes: tuple[str, ...]


class SpyfyGeomHandoffPacket(_StrictModel):
    """Versioned CoChem-SPYFY & CoChem-GEOM spectroscopy handoff packet."""

    schema_version: Literal["cochem.torq.vpt2-spectroscopy-handoff/1"] = HANDOFF_SCHEMA
    molecule_id: str = Field(min_length=1)
    collaborator_institution: Literal["Tennessee Tech University"] = (
        COLLABORATOR_INSTITUTION
    )
    b0_status: Literal["AVAILABLE", "PENDING"]
    re_se_status: Literal["PENDING"] = "PENDING"
    re_se_pending_reason: str = (
        "Semi-experimental r_e^SE awaits isotope-specific Delta B_vib "
        "and experimental B_0 constants."
    )
    pickett_parameter_basis: Literal["B0_vpt2", "Be_equilibrium_only_not_B0"]
    pickett_reduction: Literal["S"] = "S"
    pickett_representation: Literal["Ir"] = "Ir"
    source_orca_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_orca_filename: str
    rotational_record: OrcaVpt2RotationalRecord
    dipole_used_debye: tuple[float, float, float] | None
    dipole_source_sha256: str | None = None
    pickett_var_filename: str
    pickett_var_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pickett_int_filename: str
    pickett_int_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    constants_profile: str = CONSTANTS_PROFILE


class TorsionalScanPoint(_StrictModel):
    """Single relaxed constrained-optimization point on a 1D torsional PES."""

    index: int = Field(ge=0)
    target_dihedral_deg: float
    achieved_dihedral_deg: float
    energy_hartree: float
    relative_energy_cm1: float
    relative_energy_kcal_mol: float
    scf_converged: bool
    geometry_angstrom: tuple[tuple[float, float, float], ...]
    point_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RelaxedTorsionalScanReceipt(_StrictModel):
    """Provenance-bearing receipt for a real PySCF + geomeTRIC 1D torsional scan."""

    schema_version: Literal["cochem.torq.relaxed-torsional-scan/1"] = SCAN_SCHEMA
    molecule_id: str = Field(min_length=1)
    collaborator_institution: Literal["Tennessee Tech University"] = (
        COLLABORATOR_INSTITUTION
    )
    symbols: tuple[str, ...]
    charge: int
    multiplicity: int
    method: str
    basis: str
    engine: Literal["pyscf"] = "pyscf"
    engine_version: str
    optimizer: Literal["geomeTRIC"] = "geomeTRIC"
    optimizer_version: str
    dihedral_atom_indices_0based: tuple[int, int, int, int]
    points: tuple[TorsionalScanPoint, ...]
    min_energy_hartree: float
    equilibrium_dihedral_deg: float
    cis_barrier_cm1: float | None
    cis_barrier_kcal_mol: float | None
    trans_barrier_cm1: float | None
    trans_barrier_kcal_mol: float | None
    max_barrier_cm1: float
    max_barrier_kcal_mol: float
    constants_profile: str = CONSTANTS_PROFILE
    scan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _parse_float_triple(line: str) -> tuple[float, float, float] | None:
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)
    if len(nums) < 3:
        return None
    vals = (float(nums[-3]), float(nums[-2]), float(nums[-1]))
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals


def _order_abc_with_shifts(
    be_xyz: tuple[float, float, float],
    b0_xyz: tuple[float, float, float] | None = None,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float] | None,
    tuple[int, int, int],
]:
    """Order principal-axis constants so A >= B >= C while tracking axis mapping."""
    order = sorted(range(3), key=lambda idx: be_xyz[idx], reverse=True)
    perm = (order[0], order[1], order[2])
    be_abc = (be_xyz[perm[0]], be_xyz[perm[1]], be_xyz[perm[2]])
    b0_abc = (
        (b0_xyz[perm[0]], b0_xyz[perm[1]], b0_xyz[perm[2]])
        if b0_xyz is not None
        else None
    )
    return be_abc, b0_abc, perm


def _parse_quartic_block(lines: list[str]) -> QuarticDistortionConstants | None:
    """Parse ORCA 6.1.1 Watson s-reduced and a-reduced quartic CDCs in cm^-1."""
    s_cm1: dict[str, float] = {}
    a_cm1: dict[str, float] = {}
    section: Literal["none", "s", "a"] = "none"
    val_re = re.compile(
        r"^\s*(DJ|DK|DJK|D1|D2|DelJ|DelK|DelJK|delJ|delK)\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
    )
    for line in lines:
        stripped = line.strip()
        if stripped == "s-reduced":
            section = "s"
            continue
        if stripped == "a-reduced":
            section = "a"
            continue
        if section != "none" and (
            stripped.startswith("writing CDCs")
            or stripped.startswith("Generating anharmonicity")
            or stripped.startswith("....")
        ):
            section = "none"
            continue
        if section in {"s", "a"}:
            match = val_re.match(line)
            if match is not None:
                key, raw_val = match.groups()
                parsed = float(raw_val)
                if math.isfinite(parsed):
                    if section == "s":
                        s_cm1[key] = parsed
                    else:
                        a_cm1[key] = parsed
    if not s_cm1 and not a_cm1:
        return None
    s_mhz = {k: v * CM1_TO_MHZ for k, v in s_cm1.items()} if s_cm1 else None
    a_mhz = {k: v * CM1_TO_MHZ for k, v in a_cm1.items()} if a_cm1 else None
    return QuarticDistortionConstants(
        s_reduced_cm1=s_cm1 or None,
        s_reduced_mhz=s_mhz,
        a_reduced_cm1=a_cm1 or None,
        a_reduced_mhz=a_mhz,
    )


def _parse_zpve(lines: list[str]) -> ZpveSummary:
    """Parse harmonic and (if present) anharmonic/rotvib ZPVE from ORCA output."""
    harm_cm1: float | None = None
    harm_eh: float | None = None
    anharm_cm1: float | None = None
    rotvib_cm1: float | None = None
    total_anharm_cm1: float | None = None

    in_vpt2_zpe = False
    for line in lines:
        if "Zero point energy" in line and "Eh" in line:
            match = re.search(r"([-+]?\d+\.\d+)\s+Eh", line)
            if match is not None:
                harm_eh = float(match.group(1))
                harm_cm1 = harm_eh * HARTREE_TO_CM1
        elif "anharmonic zpe contrib." in line:
            match = re.search(r"([-+]?\d+\.\d+)", line)
            if match is not None:
                anharm_cm1 = float(match.group(1))
        elif line.lstrip().startswith("harmonic zpe contrib."):
            match = re.search(r"([-+]?\d+\.\d+)", line)
            if match is not None:
                harm_cm1 = float(match.group(1))
                harm_eh = harm_cm1 / HARTREE_TO_CM1
        elif "rotvib zpe contrib." in line:
            match = re.search(r"([-+]?\d+\.\d+)", line)
            if match is not None:
                rotvib_cm1 = float(match.group(1))
        elif "Zero-point ro-vibrational energy [1/cm]" in line:
            in_vpt2_zpe = True
        elif in_vpt2_zpe:
            if line.strip().startswith("Total:"):
                match = re.search(r"([-+]?\d+\.\d+)", line)
                if match is not None:
                    total_anharm_cm1 = float(match.group(1))
                in_vpt2_zpe = False
            elif line.strip().startswith("Overtones") or line.strip().startswith("==="):
                in_vpt2_zpe = False

    if (
        total_anharm_cm1 is None
        and harm_cm1 is not None
        and anharm_cm1 is not None
        and rotvib_cm1 is not None
    ):
        total_anharm_cm1 = harm_cm1 + anharm_cm1 + rotvib_cm1

    return ZpveSummary(
        harmonic_cm1=harm_cm1,
        harmonic_hartree=harm_eh,
        anharmonic_correction_cm1=anharm_cm1,
        rotvib_correction_cm1=rotvib_cm1,
        total_anharmonic_cm1=total_anharm_cm1,
    )


def _parse_last_dipole(lines: list[str]) -> DipoleComponents | None:
    """Parse the last rotational-axis dipole block (highest-level density)."""
    last_method: str | None = None
    last_mag_debye: float | None = None
    last_rot_debye: tuple[float, float, float] | None = None
    last_rot_method: str | None = None

    for idx, line in enumerate(lines):
        if line.strip().startswith("Method") and ":" in line:
            last_method = line.split(":", 1)[1].strip()
        elif "Magnitude (Debye)" in line and ":" in line:
            match = re.search(r"([-+]?\d+\.\d+)", line)
            if match is not None:
                last_mag_debye = float(match.group(1))
        elif "Dipole components along the rotational axes:" in line:
            for offset in range(1, 5):
                if idx + offset >= len(lines):
                    break
                candidate = lines[idx + offset]
                if "x,y,z [Debye]" in candidate:
                    triple = _parse_float_triple(candidate)
                    if triple is not None:
                        last_rot_debye = triple
                        last_rot_method = last_method
                    break

    if last_rot_debye is None:
        return None
    mu_a, mu_b, mu_c = last_rot_debye
    mag = (
        last_mag_debye
        if last_mag_debye is not None
        else math.sqrt(mu_a**2 + mu_b**2 + mu_c**2)
    )
    return DipoleComponents(
        mu_a_debye=mu_a,
        mu_b_debye=mu_b,
        mu_c_debye=mu_c,
        magnitude_debye=mag,
        method_label=last_rot_method,
    )


def parse_orca_vpt2_rotational_constants(
    out_path: str | Path,
) -> OrcaVpt2RotationalRecord:
    """Parse authentic ORCA 6.1.1 rotational constants and VPT2 shifts.

    Strict scientific separation:
    - If a VPT2 ``Rotational constants B_e and B_0 [1/cm]`` block is present,
      records ``b0_status = "AVAILABLE"`` and computes
      ``Delta B_vib = B_0 - B_e`` so that ``B_e + Delta B_vib == B_0``.
    - If only harmonic/equilibrium ``Rotational constants`` are present (no VPT2
      block), explicitly records ``b0_status = "PENDING"`` and sets
      ``delta_b_vib_cm1 = delta_b_vib_mhz = b0_cm1 = b0_mhz = None``.
      Equilibrium ``B_e`` is NEVER equated with ``B_0``.
    """
    path = Path(out_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"ORCA output file must be a nonempty regular file: {path}")

    raw_bytes = path.read_bytes()
    source_sha256 = _sha256_bytes(raw_bytes)
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    version_match = re.search(r"Program Version\s+([0-9]+\.[0-9]+\.[0-9]+)", text)
    orca_version = version_match.group(1) if version_match is not None else None
    terminated_normally = "****ORCA TERMINATED NORMALLY****" in text

    vpt2_be_xyz: tuple[float, float, float] | None = None
    vpt2_b0_xyz: tuple[float, float, float] | None = None
    for idx, line in enumerate(lines):
        if "Rotational constants B_e and B_0 [1/cm]" in line:
            for offset in range(1, 10):
                if idx + offset >= len(lines):
                    break
                row = lines[idx + offset].strip()
                if row.startswith("B_e ") or row.startswith("B_e\t"):
                    vpt2_be_xyz = _parse_float_triple(row)
                elif row.startswith("B_0 ") or row.startswith("B_0\t"):
                    vpt2_b0_xyz = _parse_float_triple(row)
            break

    eq_cm1_xyz: tuple[float, float, float] | None = None
    eq_mhz_xyz: tuple[float, float, float] | None = None
    for line in lines:
        if "Rotational constants in cm-1:" in line:
            parsed = _parse_float_triple(line)
            if parsed is not None:
                eq_cm1_xyz = parsed
        elif "Rotational constants in MHz :" in line:
            parsed = _parse_float_triple(line)
            if parsed is not None:
                eq_mhz_xyz = parsed

    notes: list[str] = []
    if vpt2_be_xyz is not None and vpt2_b0_xyz is not None:
        has_vpt2 = True
        b0_status: Literal["AVAILABLE", "PENDING"] = "AVAILABLE"
        be_abc_cm1, b0_abc_cm1, _ = _order_abc_with_shifts(vpt2_be_xyz, vpt2_b0_xyz)
        assert b0_abc_cm1 is not None
        be_cm1 = RotationalTriple(
            a=be_abc_cm1[0], b=be_abc_cm1[1], c=be_abc_cm1[2], unit="cm^-1"
        )
        b0_cm1_val = RotationalTriple(
            a=b0_abc_cm1[0], b=b0_abc_cm1[1], c=b0_abc_cm1[2], unit="cm^-1"
        )
        be_mhz = RotationalTriple(
            a=be_cm1.a * CM1_TO_MHZ,
            b=be_cm1.b * CM1_TO_MHZ,
            c=be_cm1.c * CM1_TO_MHZ,
            unit="MHz",
        )
        b0_mhz_val = RotationalTriple(
            a=b0_cm1_val.a * CM1_TO_MHZ,
            b=b0_cm1_val.b * CM1_TO_MHZ,
            c=b0_cm1_val.c * CM1_TO_MHZ,
            unit="MHz",
        )
        b0_cm1: RotationalTriple | None = b0_cm1_val
        b0_mhz: RotationalTriple | None = b0_mhz_val
        delta_cm1: VibrationalCorrectionTriple | None = VibrationalCorrectionTriple(
            delta_a=b0_cm1_val.a - be_cm1.a,
            delta_b=b0_cm1_val.b - be_cm1.b,
            delta_c=b0_cm1_val.c - be_cm1.c,
            unit="cm^-1",
        )
        delta_mhz: VibrationalCorrectionTriple | None = VibrationalCorrectionTriple(
            delta_a=b0_mhz_val.a - be_mhz.a,
            delta_b=b0_mhz_val.b - be_mhz.b,
            delta_c=b0_mhz_val.c - be_mhz.c,
            unit="MHz",
        )
        notes.append(
            "Parsed genuine ORCA VPT2 B_e and B_0 constants; Delta B_vib = B_0 - B_e."
        )
    elif eq_cm1_xyz is not None or eq_mhz_xyz is not None:
        has_vpt2 = False
        b0_status = "PENDING"
        if eq_cm1_xyz is not None:
            be_abc_cm1, _, _ = _order_abc_with_shifts(eq_cm1_xyz)
        else:
            assert eq_mhz_xyz is not None
            be_abc_cm1, _, _ = _order_abc_with_shifts(
                (
                    eq_mhz_xyz[0] / CM1_TO_MHZ,
                    eq_mhz_xyz[1] / CM1_TO_MHZ,
                    eq_mhz_xyz[2] / CM1_TO_MHZ,
                )
            )
        be_cm1 = RotationalTriple(
            a=be_abc_cm1[0], b=be_abc_cm1[1], c=be_abc_cm1[2], unit="cm^-1"
        )
        if eq_mhz_xyz is not None:
            be_abc_mhz, _, _ = _order_abc_with_shifts(eq_mhz_xyz)
            be_mhz = RotationalTriple(
                a=be_abc_mhz[0], b=be_abc_mhz[1], c=be_abc_mhz[2], unit="MHz"
            )
        else:
            be_mhz = RotationalTriple(
                a=be_cm1.a * CM1_TO_MHZ,
                b=be_cm1.b * CM1_TO_MHZ,
                c=be_cm1.c * CM1_TO_MHZ,
                unit="MHz",
            )
        b0_cm1 = None
        b0_mhz = None
        delta_cm1 = None
        delta_mhz = None
        notes.append(
            "Equilibrium B_e only (no VPT2 block in ORCA output); "
            "b0_status=PENDING and delta_b_vib_mhz=None (B_e is never equated to B_0)."
        )
    else:
        raise ValueError(
            f"No ORCA rotational constant block found in {path.name}; "
            "refusing to fabricate constants."
        )

    zpve = _parse_zpve(lines)
    quartic = _parse_quartic_block(lines)
    dipole = _parse_last_dipole(lines)

    return OrcaVpt2RotationalRecord(
        source_path=str(path),
        source_filename=path.name,
        source_sha256=source_sha256,
        source_size_bytes=len(raw_bytes),
        orca_version=orca_version,
        terminated_normally=terminated_normally,
        has_vpt2_block=has_vpt2,
        b0_status=b0_status,
        be_cm1=be_cm1,
        be_mhz=be_mhz,
        delta_b_vib_cm1=delta_cm1,
        delta_b_vib_mhz=delta_mhz,
        b0_cm1=b0_cm1,
        b0_mhz=b0_mhz,
        zpve=zpve,
        quartic_cdc=quartic,
        dipole=dipole,
        scientific_notes=tuple(notes),
    )


def _render_pickett_var(
    *,
    title: str,
    rot_mhz: RotationalTriple,
    basis_label: str,
    quartic_s_mhz: dict[str, float] | None,
) -> str:
    """Render a Pickett SPCAT ``.var`` parameter seed file (S-reduction, Ir)."""
    entries: list[tuple[int, float, float, str]] = [
        (10000, rot_mhz.a, 1.0e-3, f"A ({basis_label})"),
        (20000, rot_mhz.b, 1.0e-3, f"B ({basis_label})"),
        (30000, rot_mhz.c, 1.0e-3, f"C ({basis_label})"),
    ]
    if quartic_s_mhz is not None:
        # Pickett S-reduction Ir sign convention: -D_J, -D_JK, -D_K, d1, d2
        for key, code, sign in (
            ("DJ", 200, -1.0),
            ("DJK", 1100, -1.0),
            ("DK", 2000, -1.0),
            ("D1", 40100, 1.0),
            ("D2", 50000, 1.0),
        ):
            if key in quartic_s_mhz:
                entries.append(
                    (code, sign * quartic_s_mhz[key], 1.0e-6, f"{key} (S-red)")
                )

    safe_title = re.sub(r"[^ -~]", "_", title)[:50]
    header_lines = [
        f"{safe_title} [{basis_label} | {COLLABORATOR_INSTITUTION}]",
        f"{len(entries):4d}  50   51    0    0.0000E+00"
        "    1.0000E+05    1.0000E+00 1.0000",
        "s   1  1  0  50  0  1  1  1  1  -1   0",
    ]
    param_lines = [
        f"{code:12d} {value:23.15E} {unc:12.4E} /{label}"
        for code, value, unc, label in entries
    ]
    return "\n".join(header_lines + param_lines) + "\n"


def _render_pickett_int(
    *,
    title: str,
    dipole_abc_debye: tuple[float, float, float] | None,
    temperature_k: float = 2.0,
    max_j: int = 30,
    freq_limit_ghz: float = 40.0,
) -> str:
    """Render a Pickett SPCAT ``.int`` intensity/dipole seed file."""
    safe_title = re.sub(r"[^ -~]", "_", title)[:50]
    lines = [
        f"{safe_title} [CoChem-TORQ -> CoChem-SPYFY | {COLLABORATOR_INSTITUTION}]",
        f"0  91  1000.0  0  {max_j}  -10.0  -10.0  {freq_limit_ghz:.3f}  "
        f"{temperature_k:.3f}",
    ]
    if dipole_abc_debye is not None:
        for axis_code, value, label in (
            ("001", dipole_abc_debye[0], "mu_a [Debye]"),
            ("002", dipole_abc_debye[1], "mu_b [Debye]"),
            ("003", dipole_abc_debye[2], "mu_c [Debye]"),
        ):
            if abs(value) > 1.0e-9:
                lines.append(f" {axis_code}  {value:12.6f}  /{label}")
    else:
        lines.append(" / dipole PENDING: supply principal-axis dipole in Debye")
    return "\n".join(lines) + "\n"


def export_spyfy_geom_handoff(
    record: OrcaVpt2RotationalRecord,
    output_dir: str | Path,
    *,
    molecule_id: str,
    file_stem: str | None = None,
    dipole_override_debye: tuple[float, float, float] | None = None,
    dipole_source_sha256: str | None = None,
    temperature_k: float = 2.0,
    max_j: int = 30,
    freq_limit_ghz: float = 40.0,
) -> tuple[SpyfyGeomHandoffPacket, Path]:
    """Export a versioned CoChem-SPYFY & CoChem-GEOM JSON + Pickett seed packet.

    Always leaves ``re_se_status = "PENDING"`` until isotope-specific
    ``Delta B_vib`` and experimental ``B_0`` constants are supplied.
    """
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stem = file_stem or re.sub(r"[^A-Za-z0-9_.-]+", "_", molecule_id).strip("_")
    if not stem:
        raise ValueError("A non-empty molecule_id or file_stem is required.")

    if record.b0_status == "AVAILABLE" and record.b0_mhz is not None:
        pickett_rot = record.b0_mhz
        basis_label: Literal["B0_vpt2", "Be_equilibrium_only_not_B0"] = "B0_vpt2"
    else:
        pickett_rot = record.be_mhz
        basis_label = "Be_equilibrium_only_not_B0"

    quartic_s_mhz = (
        record.quartic_cdc.s_reduced_mhz if record.quartic_cdc is not None else None
    )

    if dipole_override_debye is not None:
        dipole_used: tuple[float, float, float] | None = (
            float(dipole_override_debye[0]),
            float(dipole_override_debye[1]),
            float(dipole_override_debye[2]),
        )
        dip_sha = dipole_source_sha256
    elif record.dipole is not None:
        dipole_used = (
            record.dipole.mu_a_debye,
            record.dipole.mu_b_debye,
            record.dipole.mu_c_debye,
        )
        dip_sha = record.source_sha256
    else:
        dipole_used = None
        dip_sha = None

    var_text = _render_pickett_var(
        title=molecule_id,
        rot_mhz=pickett_rot,
        basis_label=basis_label,
        quartic_s_mhz=quartic_s_mhz,
    )
    int_text = _render_pickett_int(
        title=molecule_id,
        dipole_abc_debye=dipole_used,
        temperature_k=temperature_k,
        max_j=max_j,
        freq_limit_ghz=freq_limit_ghz,
    )

    var_path = out_root / f"{stem}.var"
    int_path = out_root / f"{stem}.int"
    var_bytes = var_text.encode("utf-8")
    int_bytes = int_text.encode("utf-8")
    var_path.write_bytes(var_bytes)
    int_path.write_bytes(int_bytes)

    packet = SpyfyGeomHandoffPacket(
        molecule_id=molecule_id,
        b0_status=record.b0_status,
        pickett_parameter_basis=basis_label,
        source_orca_sha256=record.source_sha256,
        source_orca_filename=record.source_filename,
        rotational_record=record,
        dipole_used_debye=dipole_used,
        dipole_source_sha256=dip_sha,
        pickett_var_filename=var_path.name,
        pickett_var_sha256=_sha256_bytes(var_bytes),
        pickett_int_filename=int_path.name,
        pickett_int_sha256=_sha256_bytes(int_bytes),
    )

    json_path = out_root / f"{stem}_handoff.json"
    json_bytes = (
        json.dumps(packet.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    json_path.write_bytes(json_bytes)
    return packet, json_path


def _dihedral_degrees(
    coords: np.ndarray,
    indices: tuple[int, int, int, int],
) -> float:
    """Compute signed dihedral angle in degrees in (-180, 180]."""
    i, j, k, m = indices
    p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[m]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_unit = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_unit, v), w))
    return float(math.degrees(math.atan2(y, x)))


def _build_h2o2_geometry(
    dihedral_deg: float,
    *,
    r_oo: float = 1.455,
    r_oh: float = 0.965,
    angle_ooh_deg: float = 100.0,
) -> list[tuple[float, float, float]]:
    """Construct H-O-O-H Cartesian coordinates in Angstrom at exact dihedral_deg."""
    theta = math.radians(angle_ooh_deg)
    phi = math.radians(dihedral_deg)
    o1 = (0.0, 0.0, 0.0)
    o2 = (r_oo, 0.0, 0.0)
    h1 = (r_oh * math.cos(theta), r_oh * math.sin(theta), 0.0)
    h2 = (
        r_oo - r_oh * math.cos(theta),
        r_oh * math.sin(theta) * math.cos(phi),
        r_oh * math.sin(theta) * math.sin(phi),
    )
    return [h1, o1, o2, h2]


def run_relaxed_1d_torsional_scan(
    *,
    molecule_id: str = "H2O2",
    dihedral_angles_deg: Sequence[float] = (
        0.0,
        15.0,
        30.0,
        45.0,
        60.0,
        75.0,
        90.0,
        105.0,
        120.0,
        135.0,
        150.0,
        165.0,
        180.0,
    ),
    method: Literal["rhf", "b3lyp"] = "rhf",
    basis: str = "6-31g*",
    charge: int = 0,
    multiplicity: int = 1,
    output_json: str | Path | None = None,
) -> RelaxedTorsionalScanReceipt:
    """Run a genuine PySCF + geomeTRIC relaxed 1D torsional PES scan on H2O2.

    Every grid point executes a real constrained geometry optimization in
    PySCF + geomeTRIC with the H-O-O-H dihedral frozen at the target angle.
    Zero mocks, zero interpolated/fabricated points.
    """
    import geometric
    import pyscf
    from pyscf import dft, gto, scf
    from pyscf.geomopt.geometric_solver import optimize

    if not dihedral_angles_deg:
        raise ValueError("At least one dihedral angle is required for a 1D scan.")

    symbols = ("H", "O", "O", "H")
    dihedral_idx = (0, 1, 2, 3)
    raw_points: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="torq_torsion_scan_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, target_deg in enumerate(dihedral_angles_deg):
            coords = _build_h2o2_geometry(float(target_deg))
            atom_spec = [(sym, xyz) for sym, xyz in zip(symbols, coords, strict=True)]
            mol = gto.M(
                atom=atom_spec,
                basis=basis,
                charge=charge,
                spin=multiplicity - 1,
                unit="Angstrom",
                verbose=0,
            )
            mf: Any
            if method == "rhf":
                mf = scf.RHF(mol)
            else:
                mf = dft.RKS(mol)
                mf.xc = "b3lyp"
            mf.conv_tol = 1e-10

            # geomeTRIC uses 1-based atom indices for $set dihedral constraints
            constraint_file = tmp_root / f"constraints_{idx:02d}.txt"
            constraint_file.write_text(
                f"$set\ndihedral 1 2 3 4 {float(target_deg):.6f}\n",
                encoding="utf-8",
            )
            log_path = tmp_root / f"geom_{idx:02d}.log"
            mol_opt = optimize(
                mf,
                constraints=str(constraint_file),
                log=str(log_path),
                maxsteps=60,
                qccnv=True,
            )

            # Recompute authenticated SCF energy at the converged relaxed geometry
            mf_final: Any
            if method == "rhf":
                mf_final = scf.RHF(mol_opt)
            else:
                mf_final = dft.RKS(mol_opt)
                mf_final.xc = "b3lyp"
            mf_final.conv_tol = 1e-10
            energy_eh = float(mf_final.kernel())
            if not bool(mf_final.converged) or not math.isfinite(energy_eh):
                raise RuntimeError(
                    f"SCF failed to converge at dihedral {target_deg} deg."
                )

            opt_coords_ang = np.asarray(
                mol_opt.atom_coords(unit="Angstrom"), dtype=np.float64
            )
            achieved_deg = abs(_dihedral_degrees(opt_coords_ang, dihedral_idx))
            geom_tuple = tuple(
                (float(row[0]), float(row[1]), float(row[2])) for row in opt_coords_ang
            )
            raw_points.append(
                {
                    "index": idx,
                    "target_dihedral_deg": float(target_deg),
                    "achieved_dihedral_deg": achieved_deg,
                    "energy_hartree": energy_eh,
                    "scf_converged": True,
                    "geometry_angstrom": geom_tuple,
                }
            )

    min_energy = min(pt["energy_hartree"] for pt in raw_points)
    min_point = min(raw_points, key=lambda pt: pt["energy_hartree"])

    points: list[TorsionalScanPoint] = []
    for pt in raw_points:
        rel_eh = pt["energy_hartree"] - min_energy
        rel_cm1 = rel_eh * HARTREE_TO_CM1
        rel_kcal = rel_eh * HARTREE_TO_KCAL_MOL
        digest_payload = {
            "index": pt["index"],
            "target_dihedral_deg": pt["target_dihedral_deg"],
            "achieved_dihedral_deg": pt["achieved_dihedral_deg"],
            "energy_hartree": pt["energy_hartree"],
            "geometry_angstrom": pt["geometry_angstrom"],
        }
        points.append(
            TorsionalScanPoint(
                index=pt["index"],
                target_dihedral_deg=pt["target_dihedral_deg"],
                achieved_dihedral_deg=pt["achieved_dihedral_deg"],
                energy_hartree=pt["energy_hartree"],
                relative_energy_cm1=rel_cm1,
                relative_energy_kcal_mol=rel_kcal,
                scf_converged=pt["scf_converged"],
                geometry_angstrom=pt["geometry_angstrom"],
                point_sha256=_sha256_bytes(_canonical_json_bytes(digest_payload)),
            )
        )

    cis_pt = next((p for p in points if math.isclose(p.target_dihedral_deg, 0.0)), None)
    trans_pt = next(
        (p for p in points if math.isclose(abs(p.target_dihedral_deg), 180.0)), None
    )
    max_cm1 = max(p.relative_energy_cm1 for p in points)
    max_kcal = max(p.relative_energy_kcal_mol for p in points)

    geo_version = str(getattr(geometric, "__version__", "unknown"))
    pyscf_version = str(getattr(pyscf, "__version__", "unknown"))

    receipt_core = {
        "schema_version": SCAN_SCHEMA,
        "molecule_id": molecule_id,
        "collaborator_institution": COLLABORATOR_INSTITUTION,
        "symbols": list(symbols),
        "charge": charge,
        "multiplicity": multiplicity,
        "method": method,
        "basis": basis,
        "engine": "pyscf",
        "engine_version": pyscf_version,
        "optimizer": "geomeTRIC",
        "optimizer_version": geo_version,
        "dihedral_atom_indices_0based": list(dihedral_idx),
        "points": [p.model_dump(mode="json") for p in points],
        "min_energy_hartree": min_energy,
        "equilibrium_dihedral_deg": min_point["achieved_dihedral_deg"],
        "cis_barrier_cm1": cis_pt.relative_energy_cm1 if cis_pt is not None else None,
        "cis_barrier_kcal_mol": (
            cis_pt.relative_energy_kcal_mol if cis_pt is not None else None
        ),
        "trans_barrier_cm1": (
            trans_pt.relative_energy_cm1 if trans_pt is not None else None
        ),
        "trans_barrier_kcal_mol": (
            trans_pt.relative_energy_kcal_mol if trans_pt is not None else None
        ),
        "max_barrier_cm1": max_cm1,
        "max_barrier_kcal_mol": max_kcal,
        "constants_profile": CONSTANTS_PROFILE,
    }
    scan_sha256 = _sha256_bytes(_canonical_json_bytes(receipt_core))

    receipt = RelaxedTorsionalScanReceipt(
        molecule_id=molecule_id,
        symbols=symbols,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        engine_version=pyscf_version,
        optimizer_version=geo_version,
        dihedral_atom_indices_0based=dihedral_idx,
        points=tuple(points),
        min_energy_hartree=min_energy,
        equilibrium_dihedral_deg=min_point["achieved_dihedral_deg"],
        cis_barrier_cm1=cis_pt.relative_energy_cm1 if cis_pt is not None else None,
        cis_barrier_kcal_mol=(
            cis_pt.relative_energy_kcal_mol if cis_pt is not None else None
        ),
        trans_barrier_cm1=(
            trans_pt.relative_energy_cm1 if trans_pt is not None else None
        ),
        trans_barrier_kcal_mol=(
            trans_pt.relative_energy_kcal_mol if trans_pt is not None else None
        ),
        max_barrier_cm1=max_cm1,
        max_barrier_kcal_mol=max_kcal,
        scan_sha256=scan_sha256,
    )

    if output_json is not None:
        out_path = Path(output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(
            (
                json.dumps(receipt.model_dump(mode="json"), indent=2, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        )
        # Ensure os module is used cleanly if atomic sync is ever needed
        os.utime(out_path, None)

    return receipt


__all__ = [
    "COLLABORATOR_INSTITUTION",
    "HANDOFF_SCHEMA",
    "SCAN_SCHEMA",
    "DipoleComponents",
    "OrcaVpt2RotationalRecord",
    "QuarticDistortionConstants",
    "RelaxedTorsionalScanReceipt",
    "RotationalTriple",
    "SpyfyGeomHandoffPacket",
    "TorsionalScanPoint",
    "VibrationalCorrectionTriple",
    "ZpveSummary",
    "export_spyfy_geom_handoff",
    "parse_orca_vpt2_rotational_constants",
    "run_relaxed_1d_torsional_scan",
]
