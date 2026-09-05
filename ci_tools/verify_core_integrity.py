import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class HashringModel(BaseModel):
    hashes: Dict[str, str]

def get_root_dir() -> Path:
    root_env = os.environ.get("COCHEM_ROOT")
    if not root_env:
        logger.error("[MISSING DATA] COCHEM_ROOT environment variable is required.")
        sys.exit(1)
    root_dir = Path(root_env).resolve()
    if not Path(__file__).resolve().is_relative_to(root_dir):
        logger.error(f"[INTEGRITY FAILURE] Path injection detected. COCHEM_ROOT {root_dir} must contain {Path(__file__).name}")
        sys.exit(1)
    return root_dir

def get_hashring_file() -> Path:
    artifacts_env = os.environ.get("COCHEM_ARTIFACTS")
    if artifacts_env:
        artifacts_dir = Path(artifacts_env).resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        art_hashring = artifacts_dir / ".core_infrastructure_hashring.json"
        if art_hashring.exists():
            return art_hashring
            
    root = get_root_dir()
    for candidate in [
        root / ".core_infrastructure_hashring.json",
        root / "ci_tools" / ".core_infrastructure_hashring.json",
        root / "artifacts" / ".core_infrastructure_hashring.json",
    ]:
        if candidate.exists():
            return candidate
            
    if artifacts_env:
        return Path(artifacts_env).resolve() / ".core_infrastructure_hashring.json"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir / ".core_infrastructure_hashring.json"

def get_core_dirs() -> List[Path]:
    root = get_root_dir()
    return [root / "ci_tools", root / "core"]

def calculate_hash(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def generate_hashring() -> None:
    hashring_data: Dict[str, str] = {}
    root = get_root_dir()
    for dp in get_core_dirs():
        if dp.exists() and dp.is_dir():
            for p in dp.rglob("*.py"):
                rel_path = p.relative_to(root).as_posix()
                hashring_data[rel_path] = calculate_hash(p)
    
    hashring = HashringModel(hashes=hashring_data)
    data_dict: Dict[str, Any] = hashring.model_dump() if hasattr(hashring, "model_dump") else hashring.dict()
    
    primary_hashring_file = get_hashring_file()
    with open(primary_hashring_file, "w") as f:
        json.dump(data_dict, f, indent=4)
    logger.info(f"Generated hashring at {primary_hashring_file}")

    # Synchronize standard repository locations
    for target in [
        root / ".core_infrastructure_hashring.json",
        root / "ci_tools" / ".core_infrastructure_hashring.json",
    ]:
        if target != primary_hashring_file:
            with open(target, "w") as f:
                json.dump(data_dict, f, indent=4)

    artifacts_env = os.environ.get("COCHEM_ARTIFACTS")
    if artifacts_env:
        art_target = Path(artifacts_env).resolve() / ".core_infrastructure_hashring.json"
        if art_target != primary_hashring_file:
            with open(art_target, "w") as f:
                json.dump(data_dict, f, indent=4)

def verify_hashring() -> None:
    hashring_file = get_hashring_file()
    if not hashring_file.exists():
        logger.error("Hashring file not found!")
        sys.exit(1)
        
    try:
        with open(hashring_file, "r") as f:
            data = json.load(f)
        
        if "hashes" not in data:
            data = {"hashes": data}
            
        hashring = HashringModel.model_validate(data) if hasattr(HashringModel, "model_validate") else HashringModel.parse_obj(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Invalid hashring file: {e}")
        sys.exit(1)
        
    root = get_root_dir()
    for rel_path_str, expected_hash in hashring.hashes.items():
        p = root / rel_path_str
        if not p.exists():
            logger.error(f"[INTEGRITY FAILURE] File missing: {p.as_posix()}")
            sys.exit(1)
        actual_hash = calculate_hash(p)
        if actual_hash != expected_hash:
            logger.error(f"[INTEGRITY FAILURE] Hash mismatch in: {p.as_posix()}")
            sys.exit(1)
            
    for dp in get_core_dirs():
        if dp.exists() and dp.is_dir():
            for p in dp.rglob("*.py"):
                rel_path = p.relative_to(root).as_posix()
                if rel_path not in hashring.hashes:
                    logger.error(f"[INTEGRITY FAILURE] Untracked file detected: {p.as_posix()}")
                    sys.exit(1)
            
    logger.info("Core infrastructure integrity verified.")
    
    # AST Linting for Spoofed Imports (Parallelism)
    linter_path = root / "ci_tools" / "anti_spoof_linter.py"
    if linter_path.exists():
        import subprocess
        logger.info("Running AST anti-spoof linter on core infrastructure...")
        for dp in get_core_dirs():
            if dp.exists():
                res = subprocess.run([sys.executable, str(linter_path), str(dp)], capture_output=True, text=True)
                if res.returncode != 0:
                    logger.error(f"[INTEGRITY FAILURE] Spoofed imports detected in {dp.as_posix()}:\n{res.stdout}\n{res.stderr}")
                    sys.exit(1)
        logger.info("AST Linting passed. No spoofed parallel imports detected.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_hashring()
    else:
        verify_hashring()
