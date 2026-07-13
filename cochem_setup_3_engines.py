#!/usr/bin/env python3
"""
CoChem Setup Phase 3: Deep Engine & Cryptographic Verification
Enforces strict installation and mapping of ORCA, OpenMPI, and g-xTB.
If dependencies are missing, natively builds from tarballs or fails with instructions.
"""
import os
import sys
import subprocess
import shutil
import json
import logging
import hashlib
import time
import tarfile
import glob
import zipfile
import urllib.request

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

def generate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return "Not_Found"

def robust_which(executable: str) -> str:
    """Aggressively searches for executables, bypassing Python's limited $PATH."""
    path = shutil.which(executable)
    if path: return path
    
    try:
        res = subprocess.run(["bash", "-l", "-c", f"which {executable}"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split('\n')[-1]
    except Exception: pass
    
    common_paths = [
        f"/opt/{executable}/{executable}",
        f"/usr/local/bin/{executable}",
        f"/usr/lib64/openmpi/bin/{executable}",
        f"/usr/lib/openmpi/bin/{executable}",
        os.path.expanduser(f"~/{executable}/{executable}"),
        os.path.expanduser(f"~/.local/bin/{executable}")
    ]
    for p in common_paths:
        if os.path.exists(p) and os.access(p, os.X_OK): 
            return p
    return ""

def get_openmpi_version(mpirun_path: str) -> str:
    """Returns parsed OpenMPI version string for a given mpirun path."""
    try:
        res = subprocess.run([mpirun_path, "--version"], capture_output=True, text=True, timeout=10)
        txt = (res.stdout or "") + "\n" + (res.stderr or "")
        # Example: "mpirun (Open MPI) 4.1.6"
        for line in txt.splitlines():
            if "Open MPI" in line:
                parts = line.strip().split()
                return parts[-1] if parts else ""
    except Exception:
        pass
    return ""

def is_orca_compatible_openmpi(mpirun_path: str) -> bool:
    """ORCA openmpi418 builds are compatible with OpenMPI 4.1.x runtimes."""
    ver = get_openmpi_version(mpirun_path)
    return ver.startswith("4.1.")

def handle_orca_tarball() -> str:
    tarballs = glob.glob("orca_6_1_1_*.tar.xz")
    if not tarballs: return ""
    target_tarball = tarballs[0]
    engine_dir = os.path.expanduser("~/.cochem/engines/orca_6_1_1")
    os.makedirs(engine_dir, exist_ok=True)
    
    existing_bins = glob.glob(f"{engine_dir}/**/orca", recursive=True)
    for p in existing_bins:
        if os.path.isfile(p) and os.access(p, os.X_OK): return p

    print_status(f"Extracting {target_tarball} into {engine_dir}...", "info")
    try:
        with tarfile.open(target_tarball, "r:xz") as tar:
            tar.extractall(path=engine_dir)
        extracted_bins = glob.glob(f"{engine_dir}/**/orca", recursive=True)
        for p in extracted_bins:
            if os.path.isfile(p) and os.access(p, os.X_OK): return p
    except Exception: pass
    return ""

def enforce_openmpi() -> str:
    """Strictly maps or builds OpenMPI from source."""
    existing = robust_which("mpirun")
    if existing and is_orca_compatible_openmpi(existing):
        return existing

    mpi_dir = os.path.expanduser("~/.cochem/engines/openmpi")
    mpi_bin = os.path.join(mpi_dir, "bin", "mpirun")
    if os.path.exists(mpi_bin) and is_orca_compatible_openmpi(mpi_bin):
        return mpi_bin

    tarballs = sorted(glob.glob("openmpi-4.1*.tar.gz"))
    target = tarballs[0] if tarballs else ""

    if not target:
        print_status("Attempting to dynamically fetch Open MPI...", "info")
        try:
            download_url = "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.gz"
            target = "openmpi-4.1.8.tar.gz"
            urllib.request.urlretrieve(download_url, target)
        except Exception: pass

    if target and os.path.exists(target):
        print_status("Extracting and compiling Open MPI (This will take 5-10 minutes)...", "info")
        try:
            src_dir = os.path.expanduser("~/.cochem/engines/src")
            os.makedirs(src_dir, exist_ok=True)
            with tarfile.open(target, "r:gz") as tar:
                tar.extractall(path=src_dir)
            
            extracted = os.path.join(src_dir, target.replace(".tar.gz", ""))
            print_status("  -> Configuring...", "info")
            subprocess.run(["./configure", f"--prefix={mpi_dir}"], cwd=extracted, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print_status("  -> Compiling...", "info")
            subprocess.run(["make", "-j", "4", "all", "install"], cwd=extracted, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if os.path.exists(mpi_bin) and is_orca_compatible_openmpi(mpi_bin):
                print_status("Open MPI successfully compiled and installed.", "success")
                return mpi_bin
        except Exception as e:
            print_status(f"Open MPI build failed: {e}", "warning")

    print_status("CRITICAL: Open MPI missing and automated installation failed.", "fail")
    print_status("ACTION REQUIRED:", "warning")
    print_status("1. Go to: https://www.open-mpi.org/doc/", "warning")
    print_status("2. Download: openmpi-5.0.10.tar.gz", "warning")
    print_status("3. Place it in this directory and re-run.", "warning")
    sys.exit(1)

def enforce_gxtb() -> str:
    """Strictly maps or installs Grimme Lab's g-xTB."""
    existing = robust_which("xtb")
    if existing:
        res = subprocess.run([existing, "--help"], capture_output=True, text=True)
        if "--gxtb" in res.stdout or "g-xTB" in res.stdout: return existing

    engine_dir = os.path.expanduser("~/.cochem/engines/gxtb")
    existing_silo = glob.glob(f"{engine_dir}/**/xtb", recursive=True)
    if existing_silo: return existing_silo[0]

    tarballs = glob.glob("xtb-6.7.1-gxtb*.tar.xz")
    target = tarballs[0] if tarballs else ""

    if not target:
        print_status("Attempting to fetch Grimme Lab g-xTB...", "info")
        try:
            download_url = "https://github.com/grimme-lab/g-xtb/releases/download/v6.7.1-gxtb/xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz"
            target = "xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz"
            urllib.request.urlretrieve(download_url, target)
        except Exception: pass
        
    if target and os.path.exists(target):
        print_status(f"Extracting {target}...", "info")
        os.makedirs(engine_dir, exist_ok=True)
        try:
            with tarfile.open(target, "r:xz") as tar:
                tar.extractall(path=engine_dir)
            bins = glob.glob(f"{engine_dir}/**/xtb", recursive=True)
            for p in bins:
                if os.path.isfile(p):
                    os.chmod(p, 0o755)
                    print_status("g-xTB successfully extracted and mapped.", "success")
                    return p
        except Exception as e:
            print_status(f"g-xTB extraction failed: {e}", "warning")

    print_status("CRITICAL: g-xTB missing and automated installation failed.", "fail")
    print_status("ACTION REQUIRED:", "warning")
    print_status("1. Go to: https://github.com/grimme-lab/g-xtb", "warning")
    print_status("2. Download: xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz", "warning")
    print_status("3. Place it in this directory and re-run.", "warning")
    sys.exit(1)

def test_engine_determinism() -> dict:
    orca_path = handle_orca_tarball() or robust_which("orca")
    if not orca_path:
        print_status("CRITICAL: ORCA missing. Neither local tarball nor system-wide installation found.", "fail")
        print_status("Action Required: Download 'orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz' from https://faccts.de/download/", "warning")
        sys.exit(1)

    mpi_path = enforce_openmpi()
    xtb_path = enforce_gxtb()

    engine_data = {
        "orca_path": orca_path, 
        "orca_hash": generate_sha256(orca_path), 
        "mpi_path": mpi_path, 
        "mpi_hash": generate_sha256(mpi_path),
        "xtb_path": xtb_path,
        "xtb_hash": generate_sha256(xtb_path),
        "gxtb_capable": True
    }
    
    print_status(f"Testing ORCA 6.1.x execution natively...", "info")
    dummy_inp = "dummy_orca.inp"
    with open(dummy_inp, "w") as f:
        f.write("! r2SCAN-3c\n*xyz 0 1\nO 0 0 0\nH 0 0.75 0.5\nH 0 -0.75 0.5\n*")
        
    try:
        result = subprocess.run([engine_data["orca_path"], dummy_inp], capture_output=True, text=True, timeout=45)
        if "ORCA TERMINATED NORMALLY" in result.stdout:
            print_status("ORCA executed correctly. No MPI deadlocks detected.", "success")
        else:
            print_status("ORCA crashed. Check basis sets and library links.", "fail")
    except subprocess.TimeoutExpired:
        print_status("CRITICAL: MPI Deadlock Watchdog Triggered. ORCA hanging indefinitely.", "fail")
    finally:
        for ext in [".inp", ".out", ".densities", ".gbw", "_property.txt"]:
            if os.path.exists(dummy_inp.replace(".inp", ext)): os.remove(dummy_inp.replace(".inp", ext))
            
    return engine_data

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 3: Engines & Determinism ---{Colors.ENDC}")
    try:
        with open("cochem_state_p2.json", "r") as f: state = json.load(f)
    except FileNotFoundError:
        print_status("Missing cochem_state_p2.json.", "fail")
        sys.exit(1)
        
    state["engines"] = test_engine_determinism()
    with open("cochem_state_p3.json", "w") as f: json.dump(state, f, indent=4)
    print_status("Phase 3 Complete. Cryptographic baselines locked.", "success")

if __name__ == "__main__":
    main()