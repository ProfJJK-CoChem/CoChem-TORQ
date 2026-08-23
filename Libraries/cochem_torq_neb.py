"""
CoChem-TORQ: Nudged Elastic Band (NEB) and Transition State Search Module
Compliant with Method Matrix v4 (§4.4, §8B.3).
Enforces 5-threshold tight convergence criteria and proper initial Hessian models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

ARTIFACTS_DIR = os.environ.get('COCHEM_ARTIFACTS_DIR', str(Path.home() / 'cochem_artifacts'))
from typing import Any
import numpy as np

from Libraries.cochem_torq_mpqc import TorqMpqcExecutor

logger = logging.getLogger("TorqNeb")


def generate_neb_input(
    job_name: str,
    initial_coords: list[list[str | float]],
    final_coords: list[list[str | float]],
    n_images: int = 8,
    inhess: str | None = None,
    xtb_hessian_file: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "R2SCAN-3c",
) -> str:
    """
    Generates an ORCA/MPQC NEB calculation input string adhering to Method Matrix v4.
    Enforces 5 tight convergence thresholds:
      TolE 1e-7
      TolRMSG 3e-6
      TolMaxG 1e-5
      TolRMSD 5e-5
      TolMaxD 1e-4
    Configures initial Hessian to Lindh or imported xTB matrix, strictly avoiding legacy Calc_Hess true.
    """
    if xtb_hessian_file:
        inhess_line = f'  InHess Name "{xtb_hessian_file}"'
    elif inhess:
        inhess_line = f"  InHess {inhess}"
    else:
        inhess_line = "  InHess XTB2"

    initial_atom_block = "\n".join(
        f"{c[0]:>2} {float(c[1]):>12.8f} {float(c[2]):>12.8f} {float(c[3]):>12.8f}"
        for c in initial_coords
    )
    final_atom_block = "\n".join(
        f"{c[0]:>2} {float(c[1]):>12.8f} {float(c[2]):>12.8f} {float(c[3]):>12.8f}"
        for c in final_coords
    )

    neb_input = f"""! {method} NEB-CI
%maxcore 2000

%neb
  Nimages {n_images}
  Product
{final_atom_block}
  end
end

%geom
{inhess_line}
  TolE 1e-7
  TolRMSG 3e-6
  TolMaxG 1e-5
  TolRMSD 5e-5
  TolMaxD 1e-4
end

* xyz {charge} {multiplicity}
{initial_atom_block}
*
"""
    return neb_input


compute_kabsch_rmsd = TorqMpqcExecutor.compute_kabsch_rmsd


async def run_ts_optimization(
    job_name: str,
    atom_coords: list[list[str | float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "R2SCAN-3c",
    basis_set: str = "",
    output_dir: str | None = None,
    timeout: int = 3600,
) -> tuple[str, bool, dict[str, float | list | dict]]:
    """
    Executes transition state optimization via TorqMpqcExecutor.
    """
    executor = TorqMpqcExecutor()
    return await executor.run_ts_optimization(
        job_name=job_name,
        atom_coords=atom_coords,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis_set=basis_set,
        output_dir=output_dir,
        timeout=timeout,
    )


async def _run_irc_validation(
    job_name: str,
    ts_coords: list[list[str | float]],
    reactant_coords: list[list[str | float]],
    product_coords: list[list[str | float]],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "R2SCAN-3c",
    basis_set: str = "",
    output_dir: str | None = None,
    timeout: int = 3600,
) -> tuple[bool, float, float]:
    """
    Executes IRC validation via TorqMpqcExecutor.
    """
    executor = TorqMpqcExecutor()
    return await executor._run_irc_validation(
        job_name=job_name,
        ts_coords=ts_coords,
        reactant_coords=reactant_coords,
        product_coords=product_coords,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis_set=basis_set,
        output_dir=output_dir,
        timeout=timeout,
    )
