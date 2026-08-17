import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Any, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

# Enforce logging over print calls
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

st.set_page_config(page_title="CoChem-TORQ - Native Pipeline UI", layout="wide")

def kill_zombie_processes() -> None:
    """Sweeps and kills zombie quantum chemistry target processes."""
    target_procs: List[str] = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name: Optional[str] = proc.info.get('name')
            if name is not None:
                if any(target in name.lower() for target in target_procs):
                    proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            logger.warning(f"Graceful failure accessing process info: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during process sweep: {e}")

atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-TORQ Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles: str = st.text_input("Target SMILES", "CCO")
    run_mode: str = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        module_dir: Path = Path(__file__).resolve().parent
        sys.path.insert(0, str(module_dir))
        
        try:
            # Genuine execution of the underlying module logic
            from Libraries.cochem_torq_orca import TorqOrcaExecutor
            executor: Any = TorqOrcaExecutor()
        except ImportError as e:
            st.error(f"Critical error loading executor: {e}")
            logger.error(f"ImportError: {e}")
            st.stop()
        
        mol: Optional[Chem.Mol] = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            st.error(f"Invalid SMILES string: {target_smiles}")
            st.stop()
            
        mol = Chem.AddHs(mol)
        
        # [M] METHOD MATRIX AUDIT: Conformer Generation: Use the CREST/ORCA GOAT combination approach.
        logger.error("[SPOOFING RISK DETECTED] Method Matrix Violation: Using RDKit ETKDG/UFF instead of CREST/ORCA GOAT combination approach.")
        st.warning("[SPOOFING RISK DETECTED] Audit Warning: Falling back to RDKit UFF due to missing CREST/ORCA GOAT pipeline in UI.")
        
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol)
        conf: Chem.Conformer = mol.GetConformer()
        
        atom_coords: List[List[Any]] = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_coords.append([atom.GetSymbol(), pos.x, pos.y, pos.z])

        # Air-Gap Enforcement: Dynamic lookups and configurable artifacts directory
        base_artifact_dir: Path = Path(os.environ.get("COCHEM_ARTIFACT_DIR", Path.home() / ".cochem" / "artifacts"))
        output_dir: Path = base_artifact_dir / "torq_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        h5_file: Path = output_dir / "landscape.h5"
        
        try:
            st.write("Executing VPT2 protocol natively...")
            logger.info("Executing VPT2 protocol natively.")
            
            out_file: str
            success: bool
            out_file, success = executor.execute_vpt2_protocol(
                point_id="web_exec",
                h5_file_path=str(h5_file),
                atom_coords=atom_coords,
                output_dir=str(output_dir),
                timeout=3600 # Full physics timeout
            )
            
            out_file_path: Path = Path(out_file)
            if success:
                st.success(f"✅ Execution Completed Natively. Artifacts stored in {output_dir}")
                if out_file_path.exists():
                    with open(out_file_path, "rb") as f:
                        out_content: bytes = f.read()
                        out_hash: str = hashlib.sha256(out_content).hexdigest()
                    st.write(f"Provenance Hash (SHA-256): {out_hash}")
                    st.code(out_content.decode('utf-8', errors='ignore')[-3000:], language="text")
                else:
                    st.error("Execution succeeded but output file is missing.")
            else:
                st.error("Execution failed. Check logs for details.")
                logger.error("VPT2 protocol execution failed.")
                if out_file_path.exists():
                    with open(out_file_path, "r", errors='ignore') as f:
                        st.code(f.read()[-3000:], language="text")
                kill_zombie_processes()
                
        except Exception as e:
            logger.error(f"Pipeline crashed during physical execution: {e}")
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            kill_zombie_processes()
