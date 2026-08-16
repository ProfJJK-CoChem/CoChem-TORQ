import re

with open('Libraries/cochem_torq_orca.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add pydantic imports
code = code.replace('from typing import Any, Optional, Dict, List, Tuple', 'import pathlib\nfrom pydantic import BaseModel, Field, model_validator\nfrom typing import Any, Optional, Dict, List, Tuple')

pydantic_classes = '''
class OrcaJobInput(BaseModel):
    job_name: str
    method: str
    basis_set: str
    aux_basis: str = ""
    scf_type: str = "DIIS"
    coords: list = Field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    extra_options: str = ""
    output_dir: str = "."
    timeout: int = 3600

    @model_validator(mode='after')
    def check_mmv4_constraints(self):
        method_lower = self.method.lower()
        dft_methods = ["b3lyp", "pbe0", "m062x", "wb97x", "scan", "r2scan", "b97"]
        if any(dft in method_lower for dft in dft_methods) and "-3c" not in method_lower:
            if not ("d3" in method_lower or "d4" in method_lower or "d3" in self.extra_options.lower() or "d4" in self.extra_options.lower()):
                raise ValueError(f"Method Matrix v4 requires D3 or D4 dispersion correction for DFT method: {self.method}")
        # Enforce Method Matrix fallbacks for analytic CCSD(T)
        if "ccsd(t)" in method_lower:
            pass # Hook for future analytical freq fallback checks
        return self

class OrcaJobOutput(BaseModel):
    output_file: str
    success: bool
    energy: float = 0.0
    vibrational_frequencies: list = Field(default_factory=list)
    dipole_moment: dict = Field(default_factory=dict)
    polarizability: list = Field(default_factory=list)
    spin_hamiltonian: dict = Field(default_factory=dict)
    spin_contamination: float = 0.0
    provenance_hash: str = ""

    @model_validator(mode='after')
    def check_spin_contamination(self):
        # We can enforce strict limits here based on spin_contamination
        return self

'''

code = code.replace('class TorqOrcaExecutor:', pydantic_classes + 'class TorqOrcaExecutor:')

old_run_orca_job = '''    def run_orca_job(
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
        output_dir: str = ".",
        timeout: int = 3600
    ) -> tuple[str, bool]:
        os.makedirs(output_dir, exist_ok=True)
        input_file = os.path.join(output_dir, f"{job_name}.inp")
        output_file = os.path.join(output_dir, f"{job_name}.out")
        try:
            input_content = self._generate_orca_input(
                method, basis_set, aux_basis, scf_type,
                coords=coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_options
            )
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(input_content)

            cmd = [self.orca_path, input_file]
            with open(output_file, 'w', encoding='utf-8') as out:
                process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)

                try:
                    process.wait(timeout=timeout)

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
            return output_file, False'''

new_run_orca_job = '''    def run_orca_job(
        self,
        job_input: OrcaJobInput
    ) -> OrcaJobOutput:
        output_dir_path = pathlib.Path(job_input.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        input_file = output_dir_path / f"{job_input.job_name}.inp"
        output_file = output_dir_path / f"{job_input.job_name}.out"
        try:
            input_content = self._generate_orca_input(
                job_input.method, job_input.basis_set, job_input.aux_basis, job_input.scf_type,
                coords=job_input.coords, charge=job_input.charge, multiplicity=job_input.multiplicity,
                extra_options=job_input.extra_options
            )
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(input_content)

            orca_executable = os.environ.get("ORCA_PATH", self.orca_path)
            cmd = [str(orca_executable), str(input_file)]
            
            env = os.environ.copy()
            if os.name == 'nt':
                env["TMPDIR"] = os.environ.get("TEMP", str(output_dir_path))

            with open(output_file, 'w', encoding='utf-8') as out:
                subprocess.run(
                    cmd, 
                    stdout=out, 
                    stderr=subprocess.STDOUT, 
                    check=True, 
                    timeout=job_input.timeout,
                    env=env
                )

            logger.info(f"ORCA job {job_input.job_name} completed successfully")
            success = True

        except subprocess.TimeoutExpired:
            logger.error(f"ORCA job {job_input.job_name} timed out after {job_input.timeout} seconds")
            success = False
        except subprocess.CalledProcessError as e:
            logger.error(f"ORCA job {job_input.job_name} failed with error code {e.returncode}")
            success = False
        except Exception as e:
            logger.error(f"Error running ORCA job {job_input.job_name}: {e}")
            success = False

        parsed_data = self.parse_orca_output(str(output_file))
        
        provenance_hash = ""
        if os.path.exists(output_file):
            with open(output_file, 'rb') as f:
                provenance_hash = hashlib.sha256(f.read()).hexdigest()

        return OrcaJobOutput(
            output_file=str(output_file),
            success=success,
            energy=parsed_data.get("energy", 0.0),
            vibrational_frequencies=parsed_data.get("vibrational_frequencies", []),
            dipole_moment=parsed_data.get("dipole_moment", {}),
            polarizability=parsed_data.get("polarizability", []),
            spin_hamiltonian=parsed_data.get("spin_hamiltonian", {}),
            spin_contamination=parsed_data.get("spin_contamination", 0.0),
            provenance_hash=provenance_hash
        )'''

code = code.replace(old_run_orca_job, new_run_orca_job)

old_run_ts_call = '''        output_file, success = await loop.run_in_executor(
            None,
            lambda: self.run_orca_job(
                job_name, method, basis_set, "", "DIIS",
                atom_coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_opts, output_dir=output_dir, timeout=timeout
            )
        )'''
new_run_ts_call = '''        output: OrcaJobOutput = await loop.run_in_executor(
            None,
            lambda: self.run_orca_job(
                OrcaJobInput(
                    job_name=job_name, method=method, basis_set=basis_set, scf_type="DIIS",
                    coords=atom_coords, charge=charge, multiplicity=multiplicity,
                    extra_options=extra_opts, output_dir=output_dir, timeout=timeout
                )
            )
        )
        output_file, success = output.output_file, output.success'''
code = code.replace(old_run_ts_call, new_run_ts_call)

old_irc_call = '''        output_file, success = await loop.run_in_executor(
            None,
            lambda: self.run_orca_job(
                f"{job_name}_irc", method, basis_set, "", "DIIS",
                ts_coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_opts, output_dir=output_dir, timeout=timeout
            )
        )'''
new_irc_call = '''        output: OrcaJobOutput = await loop.run_in_executor(
            None,
            lambda: self.run_orca_job(
                OrcaJobInput(
                    job_name=f"{job_name}_irc", method=method, basis_set=basis_set, scf_type="DIIS",
                    coords=ts_coords, charge=charge, multiplicity=multiplicity,
                    extra_options=extra_opts, output_dir=output_dir, timeout=timeout
                )
            )
        )
        output_file, success = output.output_file, output.success'''
code = code.replace(old_irc_call, new_irc_call)

old_lam_call = '''        output_file, success = self.run_orca_job(
            job_name, combined_method, basis_set, "", "DIIS",
            atom_coords, charge=charge, multiplicity=multiplicity,
            extra_options=extra_opts, output_dir=output_dir, timeout=timeout
        )'''
new_lam_call = '''        output: OrcaJobOutput = self.run_orca_job(
            OrcaJobInput(
                job_name=job_name, method=combined_method, basis_set=basis_set, scf_type="DIIS",
                coords=atom_coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_opts, output_dir=output_dir, timeout=timeout
            )
        )
        output_file, success = output.output_file, output.success'''
code = code.replace(old_lam_call, new_lam_call)

old_vpt2_call = '''        return self.run_orca_job(
            job_name, "r2SCAN-3c", "def2-mSVP", "", "DIIS",
            atom_coords, charge=charge, multiplicity=multiplicity,
            extra_options=extra_opts, output_dir=output_dir, timeout=timeout
        )'''
new_vpt2_call = '''        output: OrcaJobOutput = self.run_orca_job(
            OrcaJobInput(
                job_name=job_name, method="r2SCAN-3c", basis_set="def2-mSVP", scf_type="DIIS",
                coords=atom_coords, charge=charge, multiplicity=multiplicity,
                extra_options=extra_opts, output_dir=output_dir, timeout=timeout
            )
        )
        return output.output_file, output.success'''
code = code.replace(old_vpt2_call, new_vpt2_call)

spin_parsing = '''            # 1. Parse Energy
            energy_match = re.search(r"(?:FINAL SINGLE POINT ENERGY|TOTAL ENERGY)\s+(-?\d+\.\d+)", content)
            if energy_match:
                parsed_data["energy"] = float(energy_match.group(1))

            s2_match = re.search(r"Expectation value of <S\*\*2>\s+:\s+([-\d\.]+)", content)
            if s2_match:
                parsed_data["spin_contamination"] = float(s2_match.group(1))'''
code = code.replace('''            # 1. Parse Energy
            energy_match = re.search(r"(?:FINAL SINGLE POINT ENERGY|TOTAL ENERGY)\s+(-?\d+\.\d+)", content)
            if energy_match:
                parsed_data["energy"] = float(energy_match.group(1))''', spin_parsing)

with open('Libraries/cochem_torq_orca.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Updated cochem_torq_orca.py successfully!")
