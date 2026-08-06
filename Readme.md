# **CoChem-TORQ (Torsional Optimization & Rotational Quantification)**

Repository URL: https://github.com/CoChem/CoChem-TORQ

**CoChem-TORQ** is the high-performance torsional scanning, quantum tensor extraction, and microwave spectral prediction module of the CoChem ecosystem. It bridges automated geometry discovery (CoChem-TOPOS) and ab initio thermodynamics (CoChem-Cascade) directly to experimental spectral fitting tools (CoChem-SpycFit).

## **🏗️ Repository Topology & File Inventory**

CoChem-TORQ/  
├── .devcontainer/  
│   └── devcontainer.json         \# Standardized OS & container mapping  
├── Dockerfile                    \# Python 3.11, OpenMPI, and MACE/PyTorch environment  
├── .gitignore                    \# Filesystem Air-Gap enforcer (blocks .h5, .out, .gbw, .json)  
├── README.md                     \# Master deployment manifest & architecture guide  
├── requirements.txt              \# Pinned Python dependencies (numpy, scipy, networkx, pyarrow)  
├── LICENSE                       \# Apache 2.0 License  
│  
├── Libraries/                    \# Core Execution Modules (Coded in Batches)  
│   ├── cochem\_torq\_topology.py   \# Stage 1.0: 5-Option Dihedral Engine & Covalent Graph  
│   ├── cochem\_torq\_grid.py       \# Stage 1.2: 1D/2D Mesh Generator & Rodrigues Rotation  
│   ├── cochem\_torq\_mace.py       \# Stage 2.0: MACE-OFF23 Triage & VRAM Governor  
│   ├── cochem\_torq\_orca.py       \# Stage 3.0: ORCA 6.1.1 Escalation & OOM Memory Backoff  
│   ├── cochem\_tensor\_extractor.py\# Stage 4.1: Inertia Tensor Diagonalization & Linearity Trap  
│   ├── cochem\_spcat\_bridge.py    \# Stage 5.0: CODATA 2022 RRHO Partitions & SPCAT Bridge  
│   ├── cochem\_catalog\_compiler.py\# Stage 5.4: Streaming PyArrow Parquet Catalog Compiler  
│   └── cochem\_torq\_export.py     \# Stage 5.5: SpycFit Payload Synthesizer & FAIR Export  
│  
└── Workflow\_Master.ipynb         \# Single-cell execution notebook for students/collaborators

## **🚀 Installation & Bootstrapping**

We strongly recommend deploying CoChem-TORQ within a GitHub Codespace or a local Docker DevContainer to ensure base-layer Linux dependencies and micro-silos are correctly configured.

\# 1\. Clone the repository  
git clone \[https://github.com/CoChem/CoChem-TORQ.git\](https://github.com/CoChem/CoChem-TORQ.git)  
cd CoChem-TORQ

\# 2\. Run the atomic setup orchestrator via CoChem-CORE  
python3 cochem\_setup/setup.py

## **🔗 Integration into the CoChem Ecosystem**

CoChem-TORQ strictly obeys the **Registry-First** data access policy. It reads baseline hardware limits and paths from $HOME/CoChem\_Artifacts/cochem\_system\_config.json and writes its massive output tensors and Parquet databases into the isolated dynamic artifact tier ($HOME/CoChem\_Artifacts/).

It maintains clean separation from version control via strict .gitignore rules, ensuring gigabyte-scale wavefunctions (.gbw), scratch logs (.out), and raw binary caches never pollute the public Git tree.