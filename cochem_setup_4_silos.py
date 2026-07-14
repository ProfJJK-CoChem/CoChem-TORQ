#!/usr/bin/env python3
"""
CoChem Setup Phase 4: Dynamic Orchestration & Silo Generation
Implements aggressive dependency upgrading for standard libraries and high-risk C++ bindings.
Enforces MolSym extraction and maps pre-compiled GPU4PySCF binary wheels.
"""
import os
import sys
import subprocess
import shutil
import json
import logging
import glob
import urllib.request
import tarfile

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status: str = "info") -> None:
    if status == "success": print(f"  {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
    elif status == "warning": print(f"  {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")
    elif status == "fail": print(f"  {Colors.FAIL}❌ {msg}{Colors.ENDC}")
    else: print(f"  {Colors.OKCYAN}➡️ {msg}{Colors.ENDC}")

log = logging.getLogger("CoChem")

class DynamicSiloManager:
    def __init__(self, project_name: str):
        self.main_env = f"CoChem-{project_name}"
        self.conda = shutil.which("conda")
        self.silo_registry = {}
        
        print_status(f"Initializing Orchestrator in {self.main_env}...", "info")
        env_list = subprocess.check_output([self.conda, "env", "list"]).decode("utf-8")
        
        # Enforce Python 3.11 upgrades so we aren't trapped on older 3.10 builds
        if self.main_env not in env_list:
            print_status("Creating main environment with Python 3.11 baseline...", "info")
            subprocess.check_call([self.conda, "create", "-y", "-n", self.main_env, "python=3.11", "ipykernel"], stdout=subprocess.DEVNULL)
        else:
            print_status("Environment found. Upgrading and enforcing Python 3.11 baseline...", "info")
            subprocess.run([self.conda, "install", "-y", "-n", self.main_env, "python=3.11"], stdout=subprocess.DEVNULL)
            
        raw_py = subprocess.check_output(
            [self.conda, "run", "-n", self.main_env, "python", "-c", "import sys; print(sys.executable)"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        self.main_py = [line for line in raw_py.splitlines() if line.strip()][-1].strip()

        # Force upgrade Pip in the main environment before matrix installs begin
        subprocess.run([self.main_py, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)

    def aggressive_install(self, pkg_name: str, import_test: str) -> None:
        print_status(f"Attempting native aggressive install: {pkg_name}", "info")
        subprocess.run([self.main_py, "-m", "pip", "install", "--upgrade", pkg_name], capture_output=True)
        
        res = subprocess.run([self.main_py, "-c", import_test], capture_output=True, text=True)
        if res.returncode == 0:
            print_status(f"✅ {pkg_name} stable natively.", "success")
            self.silo_registry[pkg_name] = "native"
            return
            
        print_status(f"❌ {pkg_name} broke the main environment (ABI Conflict). Isolating into silo...", "warning")
        silo_base_name = f"{self.main_env}-{pkg_name.split('[')[0].lower()}"
        
        for py_v in ["3.11", "3.10", "3.9"]:
            print_status(f"  -> Testing Silo Build with Python {py_v}...", "info")
            silo_name = f"{silo_base_name}-py{py_v.replace('.', '')}"
            subprocess.run([self.conda, "create", "-y", "-n", silo_name, f"python={py_v}"], stdout=subprocess.DEVNULL)
            
            raw_silo_py = subprocess.check_output([self.conda, "run", "-n", silo_name, "python", "-c", "import sys; print(sys.executable)"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            silo_py = [line for line in raw_silo_py.splitlines() if line.strip()][-1].strip()
            
            subprocess.run([silo_py, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
            subprocess.run([silo_py, "-m", "pip", "install", pkg_name], capture_output=True)
            
            if subprocess.run([silo_py, "-c", import_test], capture_output=True).returncode == 0:
                print_status(f"✅ {pkg_name} successfully stabilized in silo running Python {py_v}.", "success")
                self.silo_registry[pkg_name] = silo_py
                return
            subprocess.run([self.conda, "env", "remove", "-n", silo_name, "-y"], stdout=subprocess.DEVNULL)
                
        print_status(f"CRITICAL: {pkg_name} failed across all Python backwards steps.", "fail")
        
    def enforce_molsym(self) -> None:
        print_status("Enforcing MolSym Installation...", "info")
        lib_dir = os.path.expanduser("~/.cochem/libraries")
        os.makedirs(lib_dir, exist_ok=True)
        molsym_dir = os.path.join(lib_dir, "MolSym")

        if os.path.exists(os.path.join(molsym_dir, "molsym")):
            self.silo_registry["molsym_path"] = molsym_dir
            print_status("MolSym verified.", "success")
            return

        tarballs = glob.glob("cochem_setup/MolSym*.tar.gz")
        target = tarballs[0] if tarballs else ""

        if not target:
            print_status("Attempting to fetch MolSym tarball...", "info")
            try:
                os.makedirs("cochem_setup", exist_ok=True)
                download_url = "https://github.com/NASymmetry/MolSym/archive/refs/tags/1.0.0.tar.gz"
                target = "cochem_setup/MolSym-1.0.0.tar.gz"
                urllib.request.urlretrieve(download_url, target)
            except Exception: pass

        if target and os.path.exists(target):
            print_status(f"Extracting {target}...", "info")
            try:
                with tarfile.open(target, "r:gz") as tar:
                    tar.extractall(path=lib_dir)
                
                extracted_folder = glob.glob(os.path.join(lib_dir, "MolSym*"))[0]
                if extracted_folder != molsym_dir:
                    os.rename(extracted_folder, molsym_dir)

                req_file = os.path.join(molsym_dir, "requirements.txt")
                if os.path.exists(req_file):
                    subprocess.run([self.main_py, "-m", "pip", "install", "-r", req_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                self.silo_registry["molsym_path"] = molsym_dir
                print_status("MolSym successfully installed.", "success")
                return
            except Exception as e:
                print_status(f"MolSym extraction failed: {e}", "warning")

        print_status("CRITICAL: MolSym missing and automatic installation failed.", "fail")
        sys.exit(1)

    def enforce_gpu4pyscf(self) -> None:
        """Installs GPU4PySCF using pre-compiled binary wheels matching the CUDA driver."""
        print_status("Enforcing GPU4PySCF (CUDA-Accelerated PySCF)...", "info")
        
        cuda_v = "12x"
        cu_suffix = "12"
        try:
            smi = subprocess.check_output(["nvidia-smi"], text=True)
            if "CUDA Version: 11" in smi:
                cuda_v = "11x"
                cu_suffix = "11"
        except Exception:
            print_status("No NVIDIA GPU detected. Skipping GPU4PySCF.", "warning")
            return

        pypi_pkg = f"gpu4pyscf-cuda{cuda_v}"
        cutensor_pkg = f"cutensor-cu{cu_suffix}"
        cupy_pkg = f"cupy-cuda{cuda_v}"

        print_status(f"Mapping binary dependencies: {pypi_pkg}, {cupy_pkg}, {cutensor_pkg}...", "info")

        subprocess.run([self.main_py, "-m", "pip", "install", "pyscf", cupy_pkg, cutensor_pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([self.main_py, "-m", "pip", "install", pypi_pkg], capture_output=True)
        
        if subprocess.run([self.main_py, "-c", "import gpu4pyscf"], capture_output=True).returncode == 0:
            print_status("✅ GPU4PySCF successfully stabilized natively.", "success")
            self.silo_registry["gpu4pyscf"] = "native"
            return
            
        print_status("❌ GPU4PySCF broke the main environment. Isolating into silo...", "warning")
        silo_base_name = f"{self.main_env}-gpu4pyscf"
        
        for py_v in ["3.11", "3.10", "3.9"]:
            print_status(f"  -> Testing Silo Build with Python {py_v}...", "info")
            silo_name = f"{silo_base_name}-py{py_v.replace('.', '')}"
            subprocess.run([self.conda, "create", "-y", "-n", silo_name, f"python={py_v}"], stdout=subprocess.DEVNULL)
            
            raw_silo_py = subprocess.check_output([self.conda, "run", "-n", silo_name, "python", "-c", "import sys; print(sys.executable)"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            silo_py = [line for line in raw_silo_py.splitlines() if line.strip()][-1].strip()
            
            subprocess.run([silo_py, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
            subprocess.run([silo_py, "-m", "pip", "install", "pyscf", cupy_pkg, cutensor_pkg], stdout=subprocess.DEVNULL)
            subprocess.run([silo_py, "-m", "pip", "install", pypi_pkg], capture_output=True)
            
            if subprocess.run([silo_py, "-c", "import gpu4pyscf"], capture_output=True).returncode == 0:
                print_status(f"✅ GPU4PySCF successfully stabilized in silo running Python {py_v}.", "success")
                self.silo_registry["gpu4pyscf"] = silo_py
                return
            subprocess.run([self.conda, "env", "remove", "-n", silo_name, "-y"], stdout=subprocess.DEVNULL)
                
        print_status("CRITICAL: GPU4PySCF failed across all Python backwards steps.", "fail")
        print_status(f"ACTION REQUIRED: Ensure your network allows pip to download: {pypi_pkg}", "warning")
        sys.exit(1)

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 4: Dynamic Orchestration & Silos ---{Colors.ENDC}")
    
    try:
        with open("cochem_setup/cochem_state_p3.json", "r") as f: state = json.load(f)
    except FileNotFoundError:
        print_status("Missing cochem_setup/cochem_state_p3.json.", "fail")
        sys.exit(1)
        
    # Dynamically read project name from command line (passed by setup.py)
    project_arg = sys.argv[1] if len(sys.argv) > 1 else "TORQ"
    manager = DynamicSiloManager(project_name=project_arg)
    
    base_pkgs = [
        "numpy", "scipy", "matplotlib", "cclib", "networkx", 
        "ipywidgets", "tqdm", "psutil", "ase", "py3Dmol", 
        "emcee", "qcelemental", "IPython"
    ]
    for pkg in base_pkgs:
        manager.aggressive_install(pkg, f"import {pkg}")
        
    manager.enforce_molsym()
    manager.enforce_gpu4pyscf()
        
    manager.aggressive_install("spglib", "import spglib")
    manager.aggressive_install("mace-torch", "import mace")
    manager.aggressive_install("cuequivariance", "import cuequivariance")
    manager.aggressive_install("cuequivariance-torch", "import cuequivariance_torch")
    
    state["silos"] = manager.silo_registry
    
    os.makedirs("cochem_setup", exist_ok=True)
    with open("cochem_setup/cochem_state_p4.json", "w") as f: json.dump(state, f, indent=4)
    print_status("Phase 4 Complete. Silos isolated and mapped.", "success")

if __name__ == "__main__":
    main()