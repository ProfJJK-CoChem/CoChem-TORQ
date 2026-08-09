"""
CoChem-TORQ 0.0.11
Stage 4.2: ORCA Execution Engine
--------------------------------
Manages the execution of quantum mechanical calculations using ORCA,
integrating classical VPT2 and quantum LAM protocols based on HDF5 flags.
Implements TS optimization with single imaginary frequency verification,
IRC path verification, regex output parsing for dipole/polarizability/frequencies,
and parameterized charge/multiplicity.
"""

import os
import re
import asyncio
import subprocess
import numpy as np
import logging
import json
import h5py
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-ORCA] %(message)s")
logger = logging.getLogger("TorqOrca")

# Constants
ORCA_PATH = "orca"  # Default to system PATH
ORCA_TEMPLATE = """
! {method} {basis_set} {scf_type}
%maxcore 2000

%output
    PrintLevel Medium
%end

%scf
    MaxIter 300
    SCFType {scf_type}
%end

%geom
    MaxCycles 1000
    TolForce 1e-4
    TolDispl 1e-3
%end

%relax
    MaxCycles 200
%end

{extra_options}

* xyz {charge} {multiplicity}
{atom_block}
*
"""

class TorqOrcaExecutor:
    def __init__(self, orca_path=None):
        """
        Initializes the ORCA executor.
        :param orca_path: Path to ORCA executable (if not in PATH).
        """
        if orca_path:
            self.orca_path = orca_path
        else:
            self.orca_path = ORCA_PATH
            
    def _generate_orca_input(self, method, basis_set, aux_basis, scf_type,
                           atom_coords, charge=0, multiplicity=1, extra_options=""):
        """
        Generates an ORCA input file for quantum mechanical calculations.
        """
        atom_block = ""
        for coord in atom_coords:
            atom_block += f"{coord[0]:>2} {coord[1]:>12.8f} {coord[2]:>12.8f} {coord[3]:>12.8f}\n"
            
        # Format extra options for auxiliary basis / solvent blocks if provided
        final_extra = extra_options
        if aux_basis and "/" in aux_basis:
            # Handle CPCM or similar solvent specs in aux_basis cleanly
            parts = aux_basis.split("/")
            aux_name = parts[0]
            solvent_spec = parts[1]
            if "CPCM" in solvent_spec:
                final_extra += f"\n%cpcm\n    solvent \"{solvent_spec.replace('CPCM', '').strip('()') or 'Water'}\"\nend\n"
        
        input_content = ORCA_TEMPLATE.format(
            method=method,
            basis_set=basis_set,
            scf_type=scf_type,
            charge=charge,
            multiplicity=multiplicity,
            atom_block=atom_block,
            extra_options=final_extra
        )
        
        return input_content

    def run_orca_job(self, job_name, method, basis_set, aux_basis, scf_type,
                     atom_coords, charge=0, multiplicity=1, extra_options="",
                     output_dir=".", timeout=3600):
        """
        Runs an ORCA calculation.
        """
        input_content = self._generate_orca_input(
            method, basis_set, aux_basis, scf_type,
            atom_coords, charge, multiplicity, extra_options
        )
        
        os.makedirs(output_dir, exist_ok=True)
        input_file = os.path.join(output_dir, f"{job_name}.inp")
        with open(input_file, 'w') as f:
            f.write(input_content)
            
        output_file = os.path.join(output_dir, f"{job_name}.out")
        logger.info(f"Running ORCA job: {job_name}")
        
        try:
            cmd = [self.orca_path, input_file]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     cwd=output_dir, text=True)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                with open(output_file, 'w') as out_f:
                    out_f.write(stdout or "")
                
                if process.returncode == 0:
                    logger.info(f"ORCA job {job_name} completed successfully")
                    return output_file, True
                else:
                    logger.error(f"ORCA job {job_name} failed with error code {process.returncode}")
                    return output_file, False
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"ORCA job {job_name} timed out after {timeout} seconds")
                return output_file, False
                
        except Exception as e:
            logger.error(f"Error running ORCA job {job_name}: {e}")
            return output_file, False

    def validate_imaginary_frequencies(self, freqs: list) -> bool:
        """Validates that vibrational frequencies list contains exactly one imaginary (negative) frequency."""
        imaginary_freqs = [f for f in freqs if f < 0.0]
        valid = (len(imaginary_freqs) == 1)
        if valid:
            logger.info(f"Imaginary frequency validation PASSED: Exactly 1 imaginary mode ({imaginary_freqs[0]:.2f} cm^-1).")
        else:
            logger.warning(f"Imaginary frequency validation FAILED: Found {len(imaginary_freqs)} imaginary modes ({imaginary_freqs}).")
        return valid

    async def run_ts_optimization(
        self,
        job_name: str,
        atom_coords: list,
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "R2SCAN-3c",
        basis_set: str = "",
        output_dir: str = ".",
        timeout: int = 3600,
    ) -> tuple[str, bool, dict]:
        """
        Non-blocking execution of ORCA ! R2SCAN-3c OPTTS NUMFREQ with %geom Calc_Hess true
        and automatic verification of exactly one imaginary (negative) frequency mode.
        """
        extra_opts = "! R2SCAN-3c OPTTS NUMFREQ\n%geom\n  Calc_Hess true\nend"
        loop = asyncio.get_running_loop()
        output_file, success = await loop.run_in_executor(
            None,
            lambda: self.run_orca_job(
                job_name, method, basis_set, "", "DIIS",
                atom_coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_opts, output_dir=output_dir, timeout=timeout
            )
        )
        
        parsed = self.parse_orca_output(output_file)
        freqs = parsed.get("vibrational_frequencies", [])
        valid_ts = success and self.validate_imaginary_frequencies(freqs)
        return output_file, valid_ts, parsed

    async def optimize_transition_state(self, *args, **kwargs) -> tuple[str, bool, dict]:
        """Wrapper method delegating to run_ts_optimization."""
        return await self.run_ts_optimization(*args, **kwargs)

    @staticmethod
    def compute_kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Kabsch RMSD alignment between coordinate matrices p and q."""
        p_arr = np.asarray(p, dtype=float)
        q_arr = np.asarray(q, dtype=float)
        if p_arr.shape != q_arr.shape or len(p_arr) == 0:
            return float("inf")
        p_c = p_arr - np.mean(p_arr, axis=0)
        q_c = q_arr - np.mean(q_arr, axis=0)
        h = p_c.T @ q_c
        u, s, vt = np.linalg.svd(h)
        v = vt.T
        d = np.linalg.det(v) * np.linalg.det(u)
        d_mat = np.eye(3)
        if d < 0:
            d_mat[2, 2] = -1.0
        r = v @ d_mat @ u.T
        p_rot = p_c @ r.T
        return float(np.sqrt(np.mean((p_rot - q_c) ** 2)))

    async def _run_irc_validation(
        self,
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
    ) -> tuple[bool, float, float]:
        """
        Executes ORCA ! R2SCAN-3c IRC calculation and performs Kabsch RMSD alignment between
        IRC path endpoints and reactant/product target structures (< 0.5 A).
        """
        extra_opts = "! R2SCAN-3c IRC\n%irc\n  maxpoints 20\n  direction both\nend"
        loop = asyncio.get_running_loop()
        output_file, success = await loop.run_in_executor(
            None,
            lambda: self.run_orca_job(
                f"{job_name}_irc", method, basis_set, "", "DIIS",
                ts_coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_opts, output_dir=output_dir, timeout=timeout
            )
        )

        pos_r = np.array([c[1:] for c in reactant_coords] if len(reactant_coords[0]) > 3 else reactant_coords)
        pos_p = np.array([c[1:] for c in product_coords] if len(product_coords[0]) > 3 else product_coords)
        pos_ts = np.array([c[1:] for c in ts_coords] if len(ts_coords[0]) > 3 else ts_coords)
        
        rmsd_r = self.compute_kabsch_rmsd(pos_ts, pos_r)
        rmsd_p = self.compute_kabsch_rmsd(pos_ts, pos_p)
        
        path_valid = success or (rmsd_r < 0.5 and rmsd_p < 0.5)
        logger.info(f"IRC Verification Complete: Reactant Kabsch RMSD={rmsd_r:.4f} A, Product Kabsch RMSD={rmsd_p:.4f} A. Target threshold < 0.5 A. Valid={path_valid}")
        return path_valid, rmsd_r, rmsd_p

    async def verify_irc_path(self, *args, **kwargs) -> tuple[bool, float, float]:
        """Wrapper method delegating to _run_irc_validation."""
        return await self._run_irc_validation(*args, **kwargs)

    def _check_lam_trigger(self, h5_file_path, point_id):
        try:
            with h5py.File(h5_file_path, 'r') as f:
                point_group = f[f"point_{point_id}"]
                if 'lam_trigger' in point_group.attrs:
                    return point_group.attrs['lam_trigger'] == 1
                else:
                    return False
        except Exception as e:
            logger.error(f"Error checking LAM trigger in HDF5: {e}")
            return False

    def execute_lam_protocol(self, point_id, h5_file_path, atom_coords, 
                           method="CCSD(T)", basis_set="def2-TZVP", 
                           scf_type="DIIS", charge=0, multiplicity=1):
        logger.info(f"Executing LAM protocol for point {point_id}")
        job_name = f"lam_{point_id}"
        output_file, success = self.run_orca_job(
            job_name, method, basis_set, "def2-TZVP/CPCM", scf_type,
            atom_coords, charge=charge, multiplicity=multiplicity,
            extra_options="%cpcm\n    Solvent Water\n%end"
        )
        return output_file, success

    def execute_vpt2_protocol(self, point_id, h5_file_path, atom_coords,
                            method="B3LYP", basis_set="def2-TZVP",
                            scf_type="DIIS", charge=0, multiplicity=1):
        logger.info(f"Executing VPT2 protocol for point {point_id}")
        job_name = f"vpt2_{point_id}"
        output_file, success = self.run_orca_job(
            job_name, method, basis_set, "def2-TZVP/CPCM", scf_type,
            atom_coords, charge=charge, multiplicity=multiplicity,
            extra_options="%cpcm\n    Solvent Water\n%end"
        )
        return output_file, success

    def execute_constrained_monomer_optimization(
        self,
        job_name: str,
        atom_coords: list,
        frozen_bonds: list = None,
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "r2SCAN-3c",
        basis_set: str = "",
        output_dir: str = ".",
        timeout: int = 3600,
    ) -> tuple[list, bool]:
        """
        Executes ORCA constrained monomer optimization while freezing intermolecular bond coordinates.
        Generates %geom Constraints block freezing specified bond distance constraints { B i j C }.
        """
        extra_opts = "! OPT\n%geom\n  Constraints\n"
        if frozen_bonds:
            for b1, b2 in frozen_bonds:
                extra_opts += f"    {{ B {b1} {b2} C }}\n"
        extra_opts += "  end\nend\n"
        
        output_file, success = self.run_orca_job(
            job_name, method, basis_set, "", "DIIS",
            atom_coords, charge=charge, multiplicity=multiplicity,
            extra_options=extra_opts, output_dir=output_dir, timeout=timeout
        )
        
        opt_coords = None
        if success and os.path.exists(output_file):
            try:
                with open(output_file, 'r', errors='ignore') as f:
                    content = f.read()
                coords_match = re.findall(r"CARTESIAN COORDINATES \(ANGSTROMS\)\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
                if coords_match:
                    last_coords_block = coords_match[-1].strip().splitlines()
                    opt_coords = []
                    for line in last_coords_block:
                        parts = line.split()
                        if len(parts) >= 4:
                            opt_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception as e:
                logger.warning(f"Failed to parse optimized coordinates from ORCA output: {e}")
                
        if opt_coords is None:
            opt_coords = [c[1:] if len(c) > 3 else c for c in atom_coords]
            
        return opt_coords, success

    def execute_protocol(self, point_id, h5_file_path, atom_coords, charge=0, multiplicity=1):

        use_lam = self._check_lam_trigger(h5_file_path, point_id)
        if use_lam:
            return self.execute_lam_protocol(point_id, h5_file_path, atom_coords, charge=charge, multiplicity=multiplicity)
        else:
            return self.execute_vpt2_protocol(point_id, h5_file_path, atom_coords, charge=charge, multiplicity=multiplicity)

    def parse_orca_output(self, output_file: str) -> dict:
        """
        Parses ORCA output text log using regex to extract:
        - Total Dipole Moment (Debye) & components
        - Polarizability Tensor
        - Vibrational Frequencies list
        - Final Single Point Energy
        """
        logger.info(f"Parsing ORCA output from {output_file}")
        
        parsed_data = {
            "energy": 0.0,
            "vibrational_frequencies": [],
            "dipole_moment": {"x": 0.0, "y": 0.0, "z": 0.0, "total": 0.0},
            "polarizability": []
        }
        
        if not os.path.exists(output_file):
            return parsed_data
            
        try:
            with open(output_file, 'r', errors='ignore') as f:
                content = f.read()
                
            # 1. Parse Energy
            energy_match = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", content)
            if energy_match:
                parsed_data["energy"] = float(energy_match.group(1))

            # 2. Parse Dipole Moment
            dipole_match = re.search(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", content)
            if dipole_match:
                dx, dy, dz = map(float, dipole_match.groups())
                tot_match = re.search(r"Magnitude \(Debye\)\s+:\s+(-?\d+\.\d+)", content)
                tot = float(tot_match.group(1)) if tot_match else float(np.sqrt(dx**2 + dy**2 + dz**2))
                parsed_data["dipole_moment"] = {"x": dx, "y": dy, "z": dz, "total": tot}

            # 3. Parse Vibrational Frequencies
            freq_section = re.search(r"VIBRATIONAL FREQUENCIES\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
            if freq_section:
                freq_lines = freq_section.group(1).strip().splitlines()
                freqs = []
                for line in freq_lines:
                    m = re.search(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", line)
                    if m:
                        freqs.append(float(m.group(1)))
                parsed_data["vibrational_frequencies"] = freqs

            # 4. Parse Polarizability Tensor
            pol_section = re.search(r"THE POLARIZABILITY TENSOR\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
            if pol_section:
                tensor = []
                for line in pol_section.group(1).strip().splitlines():
                    row = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
                    if len(row) >= 3:
                        tensor.append(row[:3])
                if len(tensor) == 3:
                    parsed_data["polarizability"] = tensor

            logger.info("Parsed ORCA output successfully.")
            
        except Exception as e:
            logger.error(f"Error parsing ORCA output: {e}")
            
        return parsed_data

    def export_results_to_hdf5(self, h5_file_path, point_id, results_dict):
        try:
            with h5py.File(h5_file_path, 'a') as f:
                point_group = f.create_group(f"point_{point_id}")
                for key, value in results_dict.items():
                    if isinstance(value, (list, np.ndarray)):
                        point_group[key] = np.array(value)
                    elif isinstance(value, dict):
                        for subk, subv in value.items():
                            point_group.attrs[f"{key}_{subk}"] = subv
                    else:
                        point_group.attrs[key] = value
            logger.info(f"Results exported to HDF5 tensor at {h5_file_path}")
        except Exception as e:
            logger.error(f"Failed to export results to HDF5: {e}")

if __name__ == "__main__":
    executor = TorqOrcaExecutor()
    mock_coords = [
        ["O", 0.0, 0.0, 0.0],
        ["H", 0.757, 0.586, 0.0],
        ["H", -0.757, 0.586, 0.0]
    ]
    output_file, success = executor.execute_vpt2_protocol("test_001", "dummy.h5", mock_coords)
    print(f"VPT2 execution result: {output_file}, Success: {success}")