import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import hashlib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

st.set_page_config(page_title="CoChem-TORQ - Native Pipeline UI", layout="wide")

def kill_zombie_processes() -> None:
    target_procs = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if any(target in name for target in target_procs):
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            raise NotImplementedError("Implementation pending")
atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-TORQ Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        module_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(module_dir))
        
        # Genuine execution of the underlying module logic
        from Libraries.cochem_torq_orca import TorqOrcaExecutor
        
        executor = TorqOrcaExecutor()
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            st.error(f"Invalid SMILES string: {target_smiles}")
            st.stop()
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        atom_coords = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_coords.append([atom.GetSymbol(), pos.x, pos.y, pos.z])

        
        output_dir = os.environ.get("COCHEM_ARTIFACT_DIR", str(module_dir / "artifacts"))
        os.makedirs(output_dir, exist_ok=True)
        h5_file = os.path.join(output_dir, "landscape.h5")
        
        try:
            st.write("Executing VPT2 protocol natively...")
            out_file, success = executor.execute_vpt2_protocol(
                point_id="web_exec",
                h5_file_path=h5_file,
                atom_coords=atom_coords,
                output_dir=output_dir,
                timeout=3600 # Full physics timeout
            )
            
            if success:
                st.success(f"✅ Execution Completed Natively. Artifacts stored in {output_dir}")
                with open(out_file, "rb") as f:
                    out_content = f.read()
                    out_hash = hashlib.sha256(out_content).hexdigest()
                st.write(f"Provenance Hash: {out_hash}")
                st.code(out_content.decode('utf-8', errors='ignore')[-3000:], language="text")
            else:
                st.error("Execution failed. Check logs for details.")
                if os.path.exists(out_file):
                    with open(out_file, "r", errors='ignore') as f:
                        st.code(f.read()[-3000:], language="text")
                kill_zombie_processes()
                
        except Exception as e:
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            kill_zombie_processes()
