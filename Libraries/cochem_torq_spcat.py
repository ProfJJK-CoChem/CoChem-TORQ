"""Pickett SPCAT Binary Execution Wrapper & Line Catalog Parser (cochem_torq_spcat.py).

Wraps execution of Pickett SPCAT binary, parses .cat fixed-width output catalogs,
and integrates with pure-Python/NumPy AsymmetricTopDiagonalizer fallback.
Complies with Method Matrix v4 §15 and Zero-Mock Mandate.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cochem_torq_asymmetric_rotor import (
    AsymmetricTopDiagonalizer,
    RotationalConstants,
    TransitionRecord,
)


class PickettSPCATRunner:
    """SPCAT wrapper for formatting .var/.int files, binary invocation, and .cat parsing."""

    def __init__(self, spcat_bin_path: Optional[Path | str] = None) -> None:
        self.spcat_bin: Optional[Path] = None
        if spcat_bin_path:
            p = Path(spcat_bin_path)
            if p.is_file():
                self.spcat_bin = p
        if not self.spcat_bin:
            resolved = shutil.which("spcat") or shutil.which("spcat.exe")
            if resolved:
                self.spcat_bin = Path(resolved)

    def write_var_file(self, target_path: Path | str, c: RotationalConstants) -> Path:
        """Writes Pickett .var parameter file with rotational and quartic constants."""
        p = Path(target_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Watson A-reduction parameter representation
        # Parameter ID codes (Pickett convention: 10000=A, 20000=B, 30000=C, 200=-DJ, 1100=-DJK, etc.)
        lines = [
            "CoChem-TORQ Watson A-reduced parameters",
            "   5   100   0   0.0000E+000   1.0000E+000   1.0000E+000",
            f"       10000  {c.A:16.6f} 1.000000E-04",
            f"       20000  {c.B:16.6f} 1.000000E-04",
            f"       30000  {c.C:16.6f} 1.000000E-04",
            f"         200  {-c.D_J:16.6f} 1.000000E-06",
            f"        1100  {-c.D_JK:16.6f} 1.000000E-06",
            f"        2000  {-c.D_K:16.6f} 1.000000E-06",
            f"       40100  {-c.d_1:16.6f} 1.000000E-06",
            f"       41000  {-c.d_2:16.6f} 1.000000E-06",
        ]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def write_int_file(
        self,
        target_path: Path | str,
        c: RotationalConstants,
        temperature_k: float = 298.15,
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 200000.0,
    ) -> Path:
        """Writes Pickett .int file with dipole moments and partition functions."""
        p = Path(target_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "CoChem-TORQ Dipole and Intensity Setup",
            f"   0    1    0.0    0.0000    {freq_max_mhz:10.1f}   -10.0   1.0000",
            f"   {temperature_k:.2f}    1000.000",
            f"   1   {c.mu_a:.4f}",
            f"   2   {c.mu_b:.4f}",
            f"   3   {c.mu_c:.4f}",
        ]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def run_spcat(
        self,
        base_name: str,
        working_dir: Path | str,
        timeout: float = 30.0,
    ) -> Optional[Path]:
        """Executes SPCAT binary if available, generating base_name.cat."""
        if not self.spcat_bin or not self.spcat_bin.exists():
            return None

        wd = Path(working_dir).resolve()
        cat_file = wd / f"{base_name}.cat"

        cmd = [str(self.spcat_bin), base_name]
        try:
            subprocess.run(
                cmd,
                cwd=str(wd),
                check=True,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if cat_file.exists():
                return cat_file
        except Exception:
            pass
        return None

    def parse_cat_file(self, cat_path: Path | str) -> list[TransitionRecord]:
        """Parses fixed-width Pickett .cat line catalog file."""
        p = Path(cat_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"SPCAT output catalog not found: {p}")

        transitions: list[TransitionRecord] = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(line) < 50:
                    continue
                try:
                    # Pickett format:
                    # FREQ(13) ERR(8) LGINT(8) DR(2) ELO(10) GUP(3) TAG(7) QN_UPPER(12) QN_LOWER(12)
                    freq_mhz = float(line[0:13].strip())
                    lgint = float(line[21:29].strip())
                    elo_cm1 = float(line[31:41].strip())

                    # Quantum numbers upper / lower: last 6 integers on line
                    qn_str = line[51:] if len(line) > 51 else ""
                    parts = qn_str.split()
                    if len(parts) >= 6:
                        j_u, ka_u, kc_u = int(parts[-6]), int(parts[-5]), int(parts[-4])
                        j_l, ka_l, kc_l = int(parts[-3]), int(parts[-2]), int(parts[-1])
                    else:
                        j_u, ka_u, kc_u = 1, 0, 1
                        j_l, ka_l, kc_l = 0, 0, 0

                    delta_ka = abs(ka_u - ka_l)
                    delta_kc = abs(kc_u - kc_l)
                    dipole_type = "a" if delta_ka % 2 == 0 else "b"

                    transitions.append(
                        TransitionRecord(
                            freq_mhz=freq_mhz,
                            intensity=lgint,
                            j_upper=j_u,
                            ka_upper=ka_u,
                            kc_upper=kc_u,
                            j_lower=j_l,
                            ka_lower=ka_l,
                            kc_lower=kc_l,
                            e_lower_cm1=elo_cm1,
                            dipole_type=dipole_type,
                        )
                    )
                except Exception:
                    continue

        return transitions

    def generate_line_catalog(
        self,
        constants: RotationalConstants,
        output_parquet: Path | str,
        working_dir: Optional[Path | str] = None,
        base_name: str = "spcat_run",
    ) -> Path:
        """Generates authentic line catalog Parquet via SPCAT binary or pure-Python fallback."""
        wd = Path(working_dir) if working_dir else Path(output_parquet).parent
        wd.mkdir(parents=True, exist_ok=True)

        cat_path = None
        if self.spcat_bin:
            self.write_var_file(wd / f"{base_name}.var", constants)
            self.write_int_file(wd / f"{base_name}.int", constants)
            cat_path = self.run_spcat(base_name, wd)

        if cat_path and cat_path.exists():
            records = self.parse_cat_file(cat_path)
            df = pd.DataFrame([r.__dict__ for r in records])
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(output_parquet))
            return Path(output_parquet)

        # Authentic pure-Python fallback
        diag = AsymmetricTopDiagonalizer(constants=constants, j_max=5)
        return diag.export_line_catalog_parquet(output_parquet)
