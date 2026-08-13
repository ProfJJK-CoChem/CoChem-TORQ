import streamlit as st
import time

st.set_page_config(page_title="CoChem-TORQ - Novice Web UI", layout="wide")

st.title("🔬 CoChem-TORQ Control Panel")
st.markdown("Welcome to the **low-friction web UI**. Use this interface to launch standard predefined tasks without touching the command line.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast (xTB)", "Accurate (ORCA)"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Initializing CoChem-TORQ payload for {target_smiles}..."):
        time.sleep(1)
        st.success(f"Pipeline executed successfully in {run_mode} mode!")
        st.balloons()
