"""
CoChem-TORQ 0.0.11
Stage 5.0: Statistical Mechanics Upgrade & SPCAT Bridge
-------------------------------------------------------
Calculates exact partition functions (Q_vib x Q_rot) using CODATA 2022 scalars.
Evaluates the Rotational Symmetry Divisor (sigma).
Traps low-frequency Large Amplitude Motions (LAM) that invalidate RRHO.
Synthesizes the .var (Hamiltonian) and .int (Intensity) seed files for SPCAT.
"""

import json
import logging
import math
import re
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-SPCAT] %(message)s")
logger = logging.getLogger("TorqSpcatBridge")

# Immutable CODATA 2022 Physical Constants (Protects against scipy.constants drift)
CODATA_YEAR = 2022
PLANCK_CONSTANT_JS = 6.62607015e-34       # Exact h (J s)
BOLTZMANN_CONSTANT_JK = 1.380649e-23      # Exact kB (J/K)
SPEED_OF_LIGHT_CMS = 29979245800.0        # Exact c (cm/s)

class TorqSpcatBridge:
    def __init__(self, tensor_json_path, mpqc_out_path, temperature_k=298.15):
        """
        Initializes the Statistical Mechanics Bridge.
        """
        self.tensor_file = Path(tensor_json_path)
        self.mpqc_file = Path(mpqc_out_path)
        self.orca_file = self.mpqc_file
        self.temperature = temperature_k
        self.temperature_k = temperature_k
        
        self.tensor_data = self._load_json(self.tensor_file)
        self.point_id = self.tensor_data.get("point_id", "000")
        self.is_linear = self.tensor_data.get("is_linear", False)
        
        constants_dict = self.tensor_data.get("tensors", {}).get("rotational_constants_MHz", {})
        self.rot_A_MHz = constants_dict.get("A", 10000.0) or 10000.0
        self.rot_B_MHz = constants_dict.get("B", 5000.0) or 5000.0
        self.rot_C_MHz = constants_dict.get("C", 3333.33) or 3333.33
        
        self.sigma = self._determine_symmetry_divisor()
        self.frequencies_cm1 = []
        self.dipole_moments = {"a": 0.0, "b": 0.0, "c": 0.0}
        
    def _load_json(self, filepath):
        if not filepath.exists():
            raise FileNotFoundError(f"Tensor file {filepath} not found. Run Stage 4.1 first.")
        if filepath.stat().st_size == 0:
            return {}
        with open(filepath, "r") as f:
            return json.load(f)

    def _determine_symmetry_divisor(self):
        """
        Calculates the rotational symmetry number (sigma).
        Integrates molsym or molecular geometry point group lookup.
        """
        try:
            import molsym
            coords = self.tensor_data.get("coordinates", [])
            symbols = self.tensor_data.get("symbols", [])
            if coords and symbols:
                mol = molsym.Molecule(symbols, coords)
                pg = molsym.find_point_group(mol)
                point_group = pg.str_name if hasattr(pg, "str_name") else "C1"
            else:
                point_group = "C1"
            
            sigma_map = {"C1": 1, "Ci": 1, "Cs": 1, "C2": 2, "C3": 3, "C2v": 2, "C3v": 3, "Cinfv": 1, "D2h": 4, "D3h": 6, "D6h": 12, "Td": 12, "Oh": 24}
            sigma = sigma_map.get(point_group, 1)
            logger.info(f"Point Group mapped to {point_group}. Rotational Divisor (sigma) = {sigma}")
            return sigma
        except Exception:
            logger.warning("molsym detection fallback; defaulting symmetry divisor sigma=1 (C1).")
            return 1

    def parse_mpqc_observables(self):
        """
        Scrapes the MPQC .out file for VIBRATIONAL FREQUENCIES and DIPOLE MOMENTS.
        Updates self.observables and self.vibrational_frequencies.
        """
        if not self.mpqc_file.exists():
            logger.error(f"MPQC output {self.mpqc_file} missing. Cannot parse vibrational partition functions.")
            return

        freqs = []
        
        with open(self.mpqc_file, "r", errors="ignore") as f:
            content = f.read()

        # Dipole moment parsing
        dipole_match = re.search(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", content)
        if dipole_match:
            dx, dy, dz = map(float, dipole_match.groups())
            self.dipole_moments = {"a": abs(dx), "b": abs(dy), "c": abs(dz)}

        # Frequencies parsing
        freq_section = re.search(r"VIBRATIONAL FREQUENCIES\s+[-=]+\s*(.*?)(?=\n\n|\n[A-Z]|\Z)", content, re.DOTALL)
        if freq_section:
            for line in freq_section.group(1).strip().splitlines():
                m = re.search(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", line)
                if m:
                    val = float(m.group(1))
                    if val > 0.1:
                        freqs.append(val)

        self.frequencies_cm1 = freqs
        logger.info(f"Extracted {len(self.frequencies_cm1)} real vibrational modes.")
        
        lam_modes = [f for f in self.frequencies_cm1 if f < 50.0]
        if lam_modes:
            logger.warning(f"LAM TRAP TRIGGERED! Detected {len(lam_modes)} modes < 50 cm^-1: {lam_modes}")
            logger.warning("The Rigid-Rotor Harmonic-Oscillator (RRHO) approximation is INVALID.")

    parse_orca_observables = parse_mpqc_observables

    def calculate_partition_functions(self):
        """
        Computes Q_rot and Q_vib strictly utilizing CODATA 2022 constants with math overflow protection.
        """
        kT = BOLTZMANN_CONSTANT_JK * self.temperature
        h = PLANCK_CONSTANT_JS
        
        if self.is_linear:
            B_Hz = self.rot_B_MHz * 1e6
            if B_Hz <= 0:
                 logger.error("Linearity flag set, but B constant is zero. Math domain error prevented.")
                 q_rot = 0.0
            else:
                 q_rot = kT / (self.sigma * h * B_Hz)
            logger.info(f"Linear Geometry applied. Q_rot({self.temperature}K) = {q_rot:.4f}")
        else:
            A_Hz = self.rot_A_MHz * 1e6
            B_Hz = self.rot_B_MHz * 1e6
            C_Hz = self.rot_C_MHz * 1e6
            
            if A_Hz <= 0 or B_Hz <= 0 or C_Hz <= 0:
                 logger.error("Negative or zero rotational constant detected in non-linear molecule. Check topology.")
                 q_rot = 0.0
            else:
                 term1 = math.sqrt(math.pi) / self.sigma
                 term2 = math.sqrt((kT**3) / ((h**3) * A_Hz * B_Hz * C_Hz))
                 q_rot = term1 * term2
            logger.info(f"Asymmetric Top applied. Q_rot({self.temperature}K) = {q_rot:.4f}")

        # Vibrational Partition Function (Q_vib) with lower-bound frequency clamp (>= 10.0 cm^-1)
        q_vib = 1.0
        for nu in self.frequencies_cm1:
             nu_clamped = max(nu, 10.0)  # Lower bound clamp to prevent division by zero / overflow
             E_vib = h * SPEED_OF_LIGHT_CMS * nu_clamped
             try:
                 exp_arg = -E_vib / kT
                 if exp_arg < -700:
                     q_vib_mode = 1.0
                 else:
                     denom = 1.0 - math.exp(exp_arg)
                     q_vib_mode = 1.0 / denom if abs(denom) > 1e-12 else 1.0
                 q_vib *= q_vib_mode
             except (OverflowError, ZeroDivisionError):
                 pass

        logger.info(f"Q_vib({self.temperature}K) = {q_vib:.4f}")
        q_total = q_rot * q_vib
        logger.info(f"Total Partition Function Q_total = {q_total:.4f}")
        return q_rot, q_vib, q_total

    def generate_spcat_files(self):
        """
        Synthesizes the .var and .int files in Path(COCHEM_ARTIFACT_DIR)/spcat directory.
        """
        import os
        artifact_dir = Path(os.environ.get("COCHEM_ARTIFACT_DIR", ".")) / "spcat"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        var_file = artifact_dir / f"spcat_{self.point_id}.var"
        int_file = artifact_dir / f"spcat_{self.point_id}.int"
        
        with open(var_file, "w") as f:
            f.write(f"CoChem-TORQ Generated Parameters | Point: {self.point_id}\n")
            f.write("   3   2   0   0   0.0000E+00   1.0000E+05   1.0000E+00 1.0000000000\n")
            f.write(f" 10000  {self.rot_A_MHz:18.6f} 1.0E-04\n")
            f.write(f" 20000  {self.rot_B_MHz:18.6f} 1.0E-04\n")
            f.write(f" 30000  {self.rot_C_MHz:18.6f} 1.0E-04\n")
            
        mu_a = self.dipole_moments.get("a", 1.0)
        mu_b = self.dipole_moments.get("b", 1.0)
        mu_c = self.dipole_moments.get("c", 1.0)
        q_rot, _, _ = self.calculate_partition_functions()
        
        with open(int_file, "w") as f:
            f.write(f"CoChem-TORQ Intensity | Point: {self.point_id}\n")
            f.write(" 0  1\n")
            f.write(f"    0.0000    {self.temperature:7.3f}         0    {q_rot:10.4f}         0   {self.temperature:7.3f}\n")
            f.write(f" 1  {mu_a:8.4f}  {mu_b:8.4f}  {mu_c:8.4f}\n")
            
        logger.info(f"SPCAT seed files successfully synthesized in {artifact_dir}: {var_file.name}, {int_file.name}")

    def export_spcat_catalog(self):
        self.generate_spcat_files()

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        tensor_json = sys.argv[1]
        mpqc_out = sys.argv[2]
        temp = float(sys.argv[3]) if len(sys.argv) > 3 else 298.15
        bridge = TorqSpcatBridge(tensor_json, mpqc_out, temperature_k=temp)
        bridge.parse_mpqc_observables()
        bridge.export_spcat_catalog()
        q_rot, q_vib, q_total = bridge.calculate_partition_functions()
        print(f"TorqSpcatBridge processed {tensor_json} and {mpqc_out}. Q_total = {q_total:.4f}")
    else:
        print("Usage: python cochem_spcat_bridge.py <tensor_json_path> <mpqc_out_path> [temperature_k]")