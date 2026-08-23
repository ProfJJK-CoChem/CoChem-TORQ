import os
import re
import asyncio
import subprocess
import numpy as np
import logging
import h5py
import hashlib
import atexit
import psutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
from pydantic import BaseModel, Field

def cleanup_zombies() -> None:
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'orca' in str(proc.info['name']).lower():
                for child in proc.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
atexit.register(cleanup_zombies)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-ORCA] %(message)s")
logger = logging.getLogger("TorqOrca")

ORCA_PATH = os.environ.get("ORCA_PATH", "orca")
ARTIFACTS_DIR = os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts"))

ORCA_TEMPLATE = """! {method_line}
%maxcore 2000

%output
    PrintLevel Medium
%end

%scf
    MaxIter 300
%end

{extra_options}

* xyz {charge} {multiplicity}
{atom_block}*
"""

class DipoleMoment(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    total: float = 0.0

class ZFS(BaseModel):
    D_cm1: float = 0.0
    E_cm1: float = 0.0
    E_over_D: float = 0.0
    D_tensor: List[List[float]] = Field(default_factory=lambda: [[0.0]*3]*3)

class GTensor(BaseModel):
    g_x: float = 2.0023
    g_y: float = 2.0023
    g_z: float = 2.0023
    g_iso: float = 2.0023
    delta_g: float = 0.0
    matrix: List[List[float]] = Field(default_factory=lambda: [[2.0023, 0.0, 0.0], [0.0, 2.0023, 0.0], [0.0, 0.0, 2.0023]])

class HyperfineA(BaseModel):
    nucleus_idx: int
    element: str
    A_iso_MHz: float

class SpinHamiltonian(BaseModel):
    zfs: ZFS = Field(default_factory=ZFS)
    g_tensor: GTensor = Field(default_factory=GTensor)
    hyperfine_A: List[HyperfineA] = Field(default_factory=list)
    soc_matrix_cm1: List[List[float]] = Field(default_factory=list)

class ParsedOrcaOutput(BaseModel):
    energy: float = 0.0
    vibrational_frequencies: List[float] = Field(default_factory=list)
    dipole_moment: DipoleMoment = Field(default_factory=DipoleMoment)
    polarizability: List[List[float]] = Field(default_factory=list)
    spin_hamiltonian: SpinHamiltonian = Field(default_factory=SpinHamiltonian)
    s_squared: Optional[float] = None
    s_squared_ideal: Optional[float] = None


class TorqOrcaExecutor:
    def __init__(self, orca_path: str | None = None) -> None:
        self.orca_path = orca_path if orca_path else ORCA_PATH

    def _generate_orca_input(self, method: str, basis_set: str, aux_basis: str, scf_type: str, coords: list | None = None, charge: int = 0, multiplicity: int = 1, extra_options: str = "", atom_coords: list | None = None, is_complex: bool = False) -> str:
        if is_complex:
            m_upper = method.upper() if method else ""
            e_upper = extra_options.upper() if extra_options else ""
            is_dft = "DFT" in m_upper or any(func in m_upper for func in ["B3LYP", "PBE", "SCAN", "M06", "W06", "OLYP", "OPBE"])
            has_dispersion = any(d in m_upper or d in e_upper for d in ["D3", "D4", "-V", "VV10", "3C"])
            if is_dft and not has_dispersion:
                raise ValueError("[ERR_METHOD_MATRIX] Dispersion correction (D3/D4) is strictly required for DFT optimization of weak complexes.")

        if coords is None and atom_coords is not None:
            coords = atom_coords
        
        atom_block = ""
        for coord in (coords or []):
            sym, x, y, z = coord[0], float(coord[1]), float(coord[2]), float(coord[3])
            atom_block += f"{sym:>2} {x:>12.8f} {y:>12.8f} {z:>12.8f}\n"

        final_extra = extra_options
        if aux_basis and "/" in aux_basis:
            parts = aux_basis.split("/")
            solvent_spec = parts[1]
            if "CPCM" in solvent_spec:
                solvent_name = solvent_spec.replace('CPCM', '').strip('()') or 'Water'
                final_extra += f"\n%cpcm\n    solvent \"{solvent_name}\"\nend\n"

        
        # Enforce Method Matrix: dynamic grid tightening
        if "opt" in (method or "").lower() or "opt" in final_extra.lower():
            if "defgrid1" in (method or "").lower() or "defgrid1" in final_extra.lower():
                if "defgrid3" not in (method or "").lower() and "defgrid3" not in final_extra.lower():
                    final_extra += "\n! defgrid3\n%geom AutoGrid true end\n"
        
        
        
        method_parts = []
        if method:
            method_parts.append(method)
        if basis_set:
            method_parts.append(basis_set)
        if scf_type:
            method_parts.append(scf_type)
        method_line = " ".join(method_parts)

        input_content = ORCA_TEMPLATE.format(
            method_line=method_line,
            charge=charge,
            multiplicity=multiplicity,
            atom_block=atom_block,
            extra_options=final_extra
        )

        return input_content

    def _hash_artifact(self, filepath: str) -> str:
        sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError:
            return ""

    def run_orca_job(
        self,
        job_name: str,
        method: str,
        basis_set: str,
        aux_basis: str,
        scf_type: str,
        coords: list,
        charge: int = 0,
        multiplicity: int = 1,
        extra_options: str = "",
        output_dir: str | None = None,
        timeout: int = 3600,
        is_complex: bool = False
    ) -> tuple[str, bool]:
        if not output_dir or output_dir == ".":
            output_dir = ARTIFACTS_DIR
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        input_file = str(out_path / f"{job_name}.inp")
        output_file = str(out_path / f"{job_name}.out")
        gbw_file = str(out_path / f"{job_name}.gbw")
        
        try:
            input_content = self._generate_orca_input(
                method, basis_set, aux_basis, scf_type,
                coords=coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_options, is_complex=is_complex
            )
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(input_content)

            cmd = [self.orca_path, input_file]
            with open(output_file, 'w', encoding='utf-8') as out:
                try:
                    subprocess.run(
                        cmd,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=timeout
                    )

                    out_hash = self._hash_artifact(output_file)
                    gbw_hash = self._hash_artifact(gbw_file)
                    logger.info(f"ORCA job {job_name} completed successfully. SHA-256 [M]: OUT={out_hash[:8]}, GBW={gbw_hash[:8]}")

                    return output_file, True
                except subprocess.TimeoutExpired as exc:
                    logger.error(f"ORCA job {job_name} timed out after {timeout} seconds: {exc}")
                    return output_file, False
                except subprocess.CalledProcessError as exc:
                    logger.error(f"ORCA job {job_name} failed with error code {exc.returncode}")
                    return output_file, False

        except Exception as e:
            logger.error(f"Error running ORCA job {job_name}: {e}")
            raise

    def validate_imaginary_frequencies(self, freqs: list) -> bool:
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
        output_dir: str | None = None,
        timeout: int = 3600,
    ) -> tuple[str, bool, ParsedOrcaOutput]:
        extra_opts = (
            f"! {method} OPTTS NUMFREQ\n"
            "%geom\n"
            "  InHess XTB2\n"
            "  TolE 1e-7\n"
            "  TolRMSG 3e-6\n"
            "  TolMaxG 1e-5\n"
            "  TolRMSD 5e-5\n"
            "  TolMaxD 1e-4\n"
            "end"
        )
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
        freqs = parsed.vibrational_frequencies
        valid_ts = success and self.validate_imaginary_frequencies(freqs)
        return output_file, valid_ts, parsed

    async def optimize_transition_state(self, *args: object, **kwargs: object) -> tuple[str, bool, ParsedOrcaOutput]:
        return await self.run_ts_optimization(*args, **kwargs)

    @staticmethod
    def compute_kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
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
        output_dir: str | None = None,
        timeout: int = 3600,
    ) -> tuple[bool, float, float]:
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

    async def verify_irc_path(self, *args: object, **kwargs: object) -> tuple[bool, float, float]:
        return await self._run_irc_validation(*args, **kwargs)

    def _check_lam_trigger(self, h5_file_path: str, point_id: str | int) -> bool:
        try:
            with h5py.File(h5_file_path, 'r') as f:
                group_name = f"point_{point_id}"
                if group_name in f:
                    point_group = f[group_name]
                    if 'lam_trigger' in point_group.attrs:
                        return bool(point_group.attrs['lam_trigger'] == 1)
                return False
        except Exception as e:
            logger.error(f"Error checking LAM trigger in HDF5: {e}")
            raise

    def execute_lam_protocol(
        self,
        point_id: str | int,
        h5_file_path: str,
        atom_coords: list,
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "r2SCAN-3c",
        basis_set: str = "",
        output_dir: str | None = None,
        timeout: int = 3600,
        frozen_bonds: list | None = None,
    ) -> tuple[list, bool]:
        job_name = f"lam_opt_point_{point_id}"
        extra_opts = (
            f"! {method} TightOPT TightSCF\n"
            "%geom\n"
            "  InHess XTB2\n"
            "  TolE 1e-7\n"
            "  TolRMSG 3e-6\n"
            "  TolMaxG 1e-5\n"
            "  TolRMSD 5e-5\n"
            "  TolMaxD 1e-4\n"
            "  Constraints\n"
        )
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
                coords_match = re.findall(
                    r"CARTESIAN COORDINATES \(ANGSTROMS\)\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)",
                    content, re.DOTALL
                )
                if coords_match:
                    last_coords_block = coords_match[-1].strip().splitlines()
                    opt_coords = []
                    for line in last_coords_block:
                        parts = line.split()
                        if len(parts) >= 4:
                            opt_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception as e:
                logger.warning(f"Failed to parse optimized coordinates from ORCA output: {e}")
                raise

        if opt_coords is None:
            opt_coords = [c[1:] if len(c) > 3 else c for c in atom_coords]

        return opt_coords, success

    def execute_vpt2_protocol(
        self,
        point_id: str | int,
        h5_file_path: str,
        atom_coords: list,
        charge: int = 0,
        multiplicity: int = 1,
        output_dir: str | None = None,
        timeout: int = 3600
    ) -> tuple[str, bool]:
        job_name = f"vpt2_point_{point_id}"
        extra_opts = "! FREQ Anfreq\n"
        return self.run_orca_job(
            job_name, "r2SCAN-3c", "def2-mSVP", "", "DIIS",
            atom_coords, charge=charge, multiplicity=multiplicity,
            extra_options=extra_opts, output_dir=output_dir, timeout=timeout
        )

    def execute_protocol(self, point_id: str | int, h5_file_path: str, atom_coords: list, charge: int = 0, multiplicity: int = 1) -> tuple[list | str, bool]:
        use_lam = self._check_lam_trigger(h5_file_path, point_id)
        if use_lam:
            return self.execute_lam_protocol(point_id, h5_file_path, atom_coords, charge=charge, multiplicity=multiplicity)
        else:
            return self.execute_vpt2_protocol(point_id, h5_file_path, atom_coords, charge=charge, multiplicity=multiplicity)

    def parse_orca_output(self, output_file: str) -> ParsedOrcaOutput:
        logger.info(f"Parsing ORCA output from {output_file}")

        parsed_data = ParsedOrcaOutput()

        if not os.path.exists(output_file):
            return parsed_data

        try:
            with open(output_file, 'r', errors='ignore') as f:
                content = f.read()

            energy_match = re.search(r"(?:FINAL SINGLE POINT ENERGY|TOTAL ENERGY)\s+(-?\d+\.\d+)", content)
            if energy_match:
                parsed_data.energy = float(energy_match.group(1))

            dipole_match = re.search(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", content)
            if dipole_match:
                dx, dy, dz = map(float, dipole_match.groups())
                tot_match = re.search(r"Magnitude \(Debye\)\s+:\s+(-?\d+\.\d+)", content)
                tot = float(tot_match.group(1)) if tot_match else float(np.sqrt(dx**2 + dy**2 + dz**2))
                parsed_data.dipole_moment = DipoleMoment(x=dx, y=dy, z=dz, total=tot)

            freq_section = re.search(r"VIBRATIONAL FREQUENCIES\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
            if freq_section:
                freq_lines = freq_section.group(1).strip().splitlines()
                freqs = []
                for line in freq_lines:
                    m = re.search(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", line)
                    if m:
                        freqs.append(float(m.group(1)))
                parsed_data.vibrational_frequencies = freqs

            pol_section = re.search(r"THE POLARIZABILITY TENSOR\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
            if pol_section:
                tensor = []
                for line in pol_section.group(1).strip().splitlines():
                    row = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
                    if len(row) >= 3:
                        tensor.append(row[:3])
                if len(tensor) == 3:
                    parsed_data.polarizability = tensor

            parsed_data.spin_hamiltonian = self.extract_spin_hamiltonian(output_file)

            s2_match = re.search(r"Expectation value of <S\*\*2>\s+:\s+([\d\.]+)", content)
            s2_ideal_match = re.search(r"Ideal value s\*\(s\+1\)\s+for\s+S=\S+\s+:\s+([\d\.]+)", content)
            if s2_match and s2_ideal_match:
                s2_val = float(s2_match.group(1))
                s2_ideal = float(s2_ideal_match.group(1))
                parsed_data.s_squared = s2_val
                parsed_data.s_squared_ideal = s2_ideal
                
                if s2_ideal > 0:
                    deviation = abs(s2_val - s2_ideal) / s2_ideal
                    if deviation > 0.10:
                        raise ValueError(f"[ERR_SPIN_CONTAMINATION] S**2 deviation exceeds 10% (Expected: {s2_ideal}, Found: {s2_val}). Halting calculation.")

            logger.info("Parsed ORCA output successfully.")

        except Exception as e:
            logger.error(f"Error parsing ORCA output: {e}")
            raise

        return parsed_data

    def extract_spin_hamiltonian(self, output_file: str) -> SpinHamiltonian:
        spin_data = SpinHamiltonian()
        if not os.path.exists(output_file):
            return spin_data

        try:
            with open(output_file, 'r', errors='ignore') as f:
                content = f.read()

            zfs = ZFS()
            zfs_d_match = re.search(r"D\s*=\s*([-\d\.]+)\s*cm\*\*-1", content)
            zfs_e_match = re.search(r"E/D\s*=\s*([-\d\.]+)", content)
            if zfs_d_match:
                d_val = float(zfs_d_match.group(1))
                e_over_d = float(zfs_e_match.group(1)) if zfs_e_match else 0.0
                e_val = d_val * e_over_d
                zfs.D_cm1 = d_val
                zfs.E_cm1 = e_val
                zfs.E_over_D = e_over_d
                zfs.D_tensor = [[-1/3*d_val+e_val, 0.0, 0.0], [0.0, -1/3*d_val-e_val, 0.0], [0.0, 0.0, 2/3*d_val]]
            spin_data.zfs = zfs

            g_tensor = GTensor()
            g_mat_match = re.search(r"The g-matrix:\s*([-\d\.\s]+)", content)
            if g_mat_match:
                try:
                    vals = [float(x) for x in g_mat_match.group(1).split()[:9]]
                    if len(vals) == 9:
                        g_mat = np.array(vals).reshape(3, 3)
                        evals = np.sort(np.linalg.eigvalsh(0.5*(g_mat + g_mat.T)))
                        gx, gy, gz = evals[0], evals[1], evals[2]
                        g_iso = (gx + gy + gz) / 3.0
                        delta_g = gz - 0.5 * (gx + gy)
                        g_tensor.g_x = float(gx)
                        g_tensor.g_y = float(gy)
                        g_tensor.g_z = float(gz)
                        g_tensor.g_iso = float(g_iso)
                        g_tensor.delta_g = float(delta_g)
                        g_tensor.matrix = g_mat.tolist()
                except Exception as e:
                    logger.error(f"Error parsing g-matrix: {e}")
                    raise
            spin_data.g_tensor = g_tensor

            a_matches = re.finditer(r"Nucleus\s+(\d+)\s+([A-Za-z]+).*?A_iso\s*=\s*([-\d\.]+)", content, re.DOTALL)
            for m in a_matches:
                spin_data.hyperfine_A.append(HyperfineA(
                    nucleus_idx=int(m.group(1)),
                    element=m.group(2),
                    A_iso_MHz=float(m.group(3))
                ))

            soc_block = re.search(r"SPIN-ORBIT COUPLING MATRIX ELEMENTS\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
            if soc_block:
                soc_matrix = []
                for line in soc_block.group(1).strip().splitlines():
                    row = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
                    if row:
                        soc_matrix.append(row)
                if soc_matrix:
                    spin_data.soc_matrix_cm1 = soc_matrix

        except Exception as e:
            logger.error(f"Error parsing Spin Hamiltonian: {e}")
            raise

        return spin_data

    def export_results_to_hdf5(self, h5_file_path: str, point_id: str | int, results_dict: dict) -> None:
        try:
            with h5py.File(h5_file_path, 'a') as f:
                group_name = f"point_{point_id}"
                if group_name in f:
                    point_group = f[group_name]
                else:
                    point_group = f.create_group(group_name)
                for key, value in results_dict.items():
                    if isinstance(value, (list, np.ndarray)):
                        if key in point_group:
                            del point_group[key]
                        point_group[key] = np.array(value)
                    elif isinstance(value, dict):
                        for subk, subv in value.items():
                            point_group.attrs[f"{key}_{subk}"] = subv
                    else:
                        point_group.attrs[key] = value
            logger.info(f"Results exported to HDF5 tensor at {h5_file_path}")
        except Exception as e:
            logger.error(f"Failed to export results to HDF5: {e}")
            raise
