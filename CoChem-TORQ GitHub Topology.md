# **CoChem-TORQ GitHub Repository Topology & Manifest**

Repository URL: https://github.com/CoChem/CoChem-TORQ

Below is the definitive structural manifest for the CoChem-TORQ GitHub repository. All source files, configuration manifests, and execution environments are accounted for, while runtime-generated numerical artifacts (such as Parquet databases, SPCAT binary outputs, and scratch wavefunctions) are excluded in compliance with the Filesystem Air-Gap Policy.

## **Complete File Manifest**

CoChem-TORQ/  
├── .github/  
│   └── workflows/  
│       └── cochem\_offload.yml          \# Automated CI/CD and cloud runner offload workflow  
├── .devcontainer/  
│   └── devcontainer.json               \# Container configuration for VS Code & Codespaces  
├── Dockerfile                          \# Container environment specification (Py3.11, CUDA, OpenMPI)  
├── .gitignore                          \# Air-gap enforcer blocking .h5, .parquet, .out, .gbw  
├── README.md                           \# Top-level deployment & ecosystem integration manifest  
├── requirements.txt                    \# Pinned Python package requirements  
├── LICENSE                             \# Apache 2.0 Open-Source License  
│  
├── Libraries/                          \# Backend Computational Modules  
│   ├── cochem\_torq\_topology.py         \# \[CODED\] Stage 1.0: Torsional Topology & Dihedral Matrix  
│   ├── cochem\_torq\_grid.py             \# \[CODED\] Stage 1.2: Grid Definition & Torsional Mesh  
│   ├── cochem\_torq\_mace.py             \# \[CODED\] Stage 2.0: MACE-OFF23 Hierarchical Triage  
│   ├── cochem\_torq\_orca.py             \# \[CODED\] Stage 3.0: High-Fidelity ORCA Execution & Memory Backoff  
│   ├── cochem\_tensor\_extractor.py      \# \[CODED\] Stage 4.1: Tensor Extraction & Linearity Protections  
│   ├── cochem\_spcat\_bridge.py          \# \[CODED\] Stage 5.0: Statistical Mechanics & SPCAT Bridge  
│   ├── cochem\_catalog\_compiler.py      \# \[CODED\] Stage 5.4: PyArrow Streaming Parquet Serialization  
│   └── cochem\_torq\_export.py           \# \[CODED\] Stage 5.5: SpycFit Payload Synthesizer & FAIR Export  
│  
├── cochem\_setup/                       \# Stage 0 Initialization Routines  
│   ├── setup.py                        \# Master bootstrapper script  
│   └── cochem\_setup\_torq.py            \# Environment silo configuration for TORQ  
│  
└── Workflow\_Master.ipynb               \# Master Jupyter execution notebook for students/collaborators

## **Air-Gap Exclusions ( Enforced via .gitignore )**

The following generated paths and files are permanently barred from staging or committing to this repository:

* torq\_grid.json (Ephemeral Cartesian mesh files)  
* torq\_mace\_surface.json (Temporary MACE triage outputs)  
* \*.parquet (Gigabyte-scale spectroscopic transition catalogs)  
* spcat\_\*.var, spcat\_\*.int, \*.cat (Pickett execution files)  
* \*.out, \*.gbw, \*.tmp (ORCA quantum scratch wavefunctions and logs)  
* CoChem\_\*\_SpycFit\_Payload\_\*.zip (Generated FAIR delivery bundles)