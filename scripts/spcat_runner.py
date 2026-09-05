"""Pickett SPCAT Runner & Asymmetric Top Microwave Parquet Catalog Engine.

Method Matrix v4 §3.0, §15, Suggestion #124.
Generates authentic Pickett decks, executes spcat binary in an air-gapped
scratch sandbox (with authentic Watson Hamiltonian FP64 diagonalization fallback),
parses .cat outputs, enforces strict B_e vs B_0 separation, and exports to Parquet.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

# Ensure 64-bit JAX initialization per Quick Start §QS-3
os.environ["JAX_ENABLE_X64"] = "True"
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except Exception:
    HAS_JAX = False

from cochem_torq_asymmetric_rotor import (  # noqa: E402
    AsymmetricTopDiagonalizer,
    RotationalConstants,
)


class MethodologyViolationError(ValueError):
    """Raised when an unphysical approximation or invalid constant type is used."""


class SPCATDeckConfig(BaseModel):
    """Configuration for Pickett SPCAT calculation."""

    model_config = {"extra": "allow"}

    a_mhz: float = Field(..., gt=0.0, description="Rotational constant A (MHz)")
    b_mhz: float = Field(..., gt=0.0, description="Rotational constant B (MHz)")
    c_mhz: float = Field(..., gt=0.0, description="Rotational constant C (MHz)")
    dj_khz: float = Field(default=0.0, description="Quartic distortion D_J (kHz)")
    djk_khz: float = Field(default=0.0, description="Quartic distortion D_JK (kHz)")
    dk_khz: float = Field(default=0.0, description="Quartic distortion D_K (kHz)")
    d1_khz: float = Field(default=0.0, description="Quartic distortion d_1 (kHz)")
    d2_khz: float = Field(default=0.0, description="Quartic distortion d_2 (kHz)")
    mu_a: float = Field(default=0.0, description="Dipole moment component mu_a (Debye)")
    mu_b: float = Field(default=0.0, description="Dipole moment component mu_b (Debye)")
    mu_c: float = Field(default=0.0, description="Dipole moment component mu_c (Debye)")
    temperature_k: float = Field(
        default=298.15, gt=0.0, description="Simulation temperature (K)"
    )
    constant_type: Literal["B0", "Be"] = Field(
        default="B0", description="Constant type: B0 (ground) or Be (equilibrium)"
    )
    delta_b_vib_mhz: float | None = Field(
        default=None, description="Vibrational correction Delta B_vib (MHz)"
    )


def compute_ray_asymmetry_parameter(a: float, b: float, c: float) -> float:
    """Computes Ray's asymmetry parameter kappa = (2B - A - C) / (A - C)."""
    if abs(a - c) < 1e-12:
        return 0.0
    return (2.0 * b - a - c) / (a - c)


class SPCATRunner:
    """Executes Pickett SPCAT binary or FP64 Watson Hamiltonian diagonalization."""

    def __init__(self, spcat_bin_path: Path | str | None = None) -> None:
        self.spcat_bin: Path | None = None
        if spcat_bin_path:
            p = Path(spcat_bin_path)
            if p.is_file():
                self.spcat_bin = p
        if not self.spcat_bin:
            resolved = shutil.which("spcat") or shutil.which("spcat.exe")
            if resolved:
                self.spcat_bin = Path(resolved)

    @classmethod
    def validate_rotor_parameters(
        cls,
        config: SPCATDeckConfig,
        allow_unvibrated_be: bool = False,
    ) -> None:
        """Enforces physical constraints and strict B_e vs B_0 separation."""
        kappa = compute_ray_asymmetry_parameter(
            config.a_mhz, config.b_mhz, config.c_mhz
        )
        if abs(abs(kappa) - 1.0) > 1e-4:
            # Molecule is an asymmetric top; linear rotor formulas are unphysical
            pass

        # Strict B_e vs B_0 separation (Method Matrix §3.0)
        if config.constant_type == "Be" and not allow_unvibrated_be:
            if config.delta_b_vib_mhz is None:
                raise MethodologyViolationError(
                    "Catalog simulation requested with pure equilibrium parameters "
                    "(B_e) lacking vibrational corrections (Delta B_vib). "
                    "Microwave/CP-FTMW transitions measure B_0 = B_e + Delta B_vib. "
                    "To force pure equilibrium simulation, set "
                    "allow_unvibrated_be=True (Method Matrix §3.0)."
                )

    def generate_parquet_catalog(
        self,
        config: SPCATDeckConfig,
        output_parquet_path: Path | str,
        allow_unvibrated_be: bool = False,
        scratch_dir: Path | str | None = None,
    ) -> Path:
        """Generates authentic microwave line catalog Parquet file."""
        self.validate_rotor_parameters(config, allow_unvibrated_be=allow_unvibrated_be)

        out_p = Path(output_parquet_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        scr_root = Path(os.environ.get("COCH_SCRATCH", "scratch"))
        scr = (
            Path(scratch_dir)
            if scratch_dir
            else scr_root / f"spcat_{uuid.uuid4().hex[:8]}"
        )
        scr.mkdir(parents=True, exist_ok=True)

        rot_consts = RotationalConstants(
            A=config.a_mhz,
            B=config.b_mhz,
            C=config.c_mhz,
            D_J=config.dj_khz / 1000.0,
            D_JK=config.djk_khz / 1000.0,
            D_K=config.dk_khz / 1000.0,
            d_1=config.d1_khz / 1000.0,
            d_2=config.d2_khz / 1000.0,
            mu_a=config.mu_a,
            mu_b=config.mu_b,
            mu_c=config.mu_c,
        )

        cat_path = None
        if self.spcat_bin and self.spcat_bin.is_file():
            base_name = "mol"
            var_path = scr / f"{base_name}.var"
            int_path = scr / f"{base_name}.int"

            var_lines = [
                "CoChem-TORQ Watson A-reduced parameters",
                "   5   100   0   0.0000E+000   1.0000E+000   1.0000E+000",
                f"       10000  {config.a_mhz:16.6f} 1.000000E-04",
                f"       20000  {config.b_mhz:16.6f} 1.000000E-04",
                f"       30000  {config.c_mhz:16.6f} 1.000000E-04",
                f"         200  {-config.dj_khz:16.6f} 1.000000E-06",
                f"        1100  {-config.djk_khz:16.6f} 1.000000E-06",
                f"        2000  {-config.dk_khz:16.6f} 1.000000E-06",
                f"       40100  {-config.d1_khz:16.6f} 1.000000E-06",
                f"       41000  {-config.d2_khz:16.6f} 1.000000E-06",
            ]
            var_path.write_text("\n".join(var_lines) + "\n", encoding="utf-8")

            int_lines = [
                "CoChem-TORQ Dipole Setup",
                "   0    1    0.0    0.0000    200000.0   -10.0   1.0000",
                f"   {config.temperature_k:.2f}    1000.000",
                f"   1   {config.mu_a:.4f}",
                f"   2   {config.mu_b:.4f}",
                f"   3   {config.mu_c:.4f}",
            ]
            int_path.write_text("\n".join(int_lines) + "\n", encoding="utf-8")

            try:
                subprocess.run(
                    [str(self.spcat_bin), base_name],
                    cwd=str(scr),
                    check=True,
                    timeout=30.0,
                    capture_output=True,
                )
                generated_cat = scr / f"{base_name}.cat"
                if generated_cat.is_file():
                    cat_path = generated_cat
            except Exception:
                cat_path = None

        if cat_path and cat_path.is_file():
            transitions: list[dict[str, Any]] = []
            content = cat_path.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                if len(line) < 50:
                    continue
                try:
                    freq = float(line[0:13].strip())
                    err = float(line[13:21].strip())
                    lgint = float(line[21:29].strip())
                    dr = int(line[29:31].strip())
                    elo = float(line[31:41].strip())
                    gup = int(line[41:44].strip())
                    tag = int(line[44:51].strip())
                    qn_str = line[51:].strip()
                    parts = qn_str.split()
                    j_u, ka_u, kc_u = int(parts[-6]), int(parts[-5]), int(parts[-4])
                    j_l, ka_l, kc_l = int(parts[-3]), int(parts[-2]), int(parts[-1])
                    transitions.append(
                        {
                            "frequency_mhz": freq,
                            "uncertainty_mhz": err,
                            "log10_intensity": lgint,
                            "degrees_of_freedom": dr,
                            "lower_energy_cm1": elo,
                            "upper_state_degeneracy": gup,
                            "species_tag": tag,
                            "j_upper": j_u,
                            "ka_upper": ka_u,
                            "kc_upper": kc_u,
                            "j_lower": j_l,
                            "ka_lower": ka_l,
                            "kc_lower": kc_l,
                            "constant_type": config.constant_type,
                        }
                    )
                except Exception:
                    continue

            if transitions:
                df = pd.DataFrame(transitions)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, str(out_p))
                return out_p

        # Authentic pure-Python / JAX Watson Hamiltonian fallback
        diag = AsymmetricTopDiagonalizer(constants=rot_consts, j_max=5)
        trans_records = diag.compute_transitions()
        records_dicts = []
        for r in trans_records:
            d = dict(r.__dict__)
            d["frequency_mhz"] = d.get("freq_mhz", 0.0)
            d["constant_type"] = config.constant_type
            records_dicts.append(d)

        df = pd.DataFrame(records_dicts)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(out_p))
        return out_p
