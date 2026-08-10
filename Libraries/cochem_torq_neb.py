"""
CoChem-TORQ 0.0.11
Stage 4.1: Transition State Optimization & NEB Module
------------------------------------------------------
Interfaces ORCA transition state optimizations with imaginary frequency validation,
NEB path generation with initial Hessian preconditioning (InHess Lindh or xTB pre-calculated Hessian matrix import),
and Kabsch RMSD alignment across IRC endpoints.
"""

from Libraries.cochem_torq_mpqc import TorqMpqcExecutor

_executor = TorqMpqcExecutor()

async def run_ts_optimization(
    job_name: str,
    atom_coords: list,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "R2SCAN-3c",
    basis_set: str = "",
    output_dir: str = ".",
    timeout: int = 3600,
    inhess: str = "XTB2",
):
    """Executes ORCA transition state optimization with R2SCAN-3c and initial Hessian preconditioning (InHess Lindh or InHess XTB2). Prohibits Calc_Hess true."""
    return await _executor.run_ts_optimization(
        job_name=job_name,
        atom_coords=atom_coords,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis_set=basis_set,
        output_dir=output_dir,
        timeout=timeout,
    )

def generate_neb_input(
    job_name: str,
    initial_coords: list,
    final_coords: list,
    n_images: int = 8,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "R2SCAN-3c",
    basis_set: str = "",
    inhess: str = "Lindh",
    xtb_hessian_file: str = None,
) -> str:
    """
    Generates ORCA NEB / CI-NEB calculation input with initial Hessian preconditioning
    using 'InHess Lindh' or importing an xTB pre-calculated Hessian matrix (§8B.3).
    Prohibits legacy Calc_Hess true.
    """
    inhess_line = f"  InHess {inhess}\n"
    if xtb_hessian_file:
        inhess_line = f'  InHess Name "{xtb_hessian_file}"\n'

    neb_opts = (
        f"! {method} {basis_set} NEB-TS TightSCF\n"
        "%neb\n"
        f"  Nimages {n_images}\n"
        f"{inhess_line}"
        "  TolE 1e-7\n"
        "  TolRMSG 3e-6\n"
        "  TolMaxG 1e-5\n"
        "  TolRMSD 5e-5\n"
        "  TolMaxD 1e-4\n"
        "end"
    )
    return neb_opts

async def _run_irc_validation(
    job_name: str,
    ts_coords: list,
    reactant_coords: list,
    product_coords: list,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "R2SCAN-3c",
    basis_set: str = "",
    output_dir: str = ".",
    timeout: int = 3600,
):
    """Executes ORCA IRC calculation and verifies Kabsch RMSD against targets (< 0.5 A)."""
    return await _executor._run_irc_validation(
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

validate_imaginary_frequencies = _executor.validate_imaginary_frequencies
compute_kabsch_rmsd = _executor.compute_kabsch_rmsd
