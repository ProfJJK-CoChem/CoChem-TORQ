"""
CoChem-TORQ 0.0.11
Stage 4.1: Transition State Optimization & IRC Module
------------------------------------------------------
Interfaces ORCA transition state optimizations with imaginary frequency validation
and Kabsch RMSD alignment across IRC endpoints.
"""

from Libraries.cochem_torq_orca import TorqOrcaExecutor

_executor = TorqOrcaExecutor()

async def run_ts_optimization(
    job_name: str,
    atom_coords: list,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "R2SCAN-3c",
    basis_set: str = "",
    output_dir: str = ".",
    timeout: int = 3600,
):
    """Executes ORCA transition state optimization with R2SCAN-3c and Calc_Hess true."""
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
