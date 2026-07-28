# **CoChem-TORQ User Manual: Part 1 — Introduction, System Philosophy & Prerequisites**

## **1.1 Preface & Ecosystem Context**

Welcome to the official User Manual for **CoChem-TORQ (Torsional Optimization & Rotational Quantification)**. CoChem-TORQ is engineered to automate the arduous workflow of predicting microwave and far-infrared rotational spectra for complex, flexible molecular systems.

Unlike traditional manual setup of conformational scans, CoChem-TORQ operates under a strict **"Registry-First"** philosophy. No script is permitted to access raw user coordinate files directly; all hardware limits, binary paths, and execution environments are governed by the authoritative cochem\_system\_config.json file established during Stage 0 initialization.

## **1.2 System Requirements & Hardware Assumptions**

* **Operating System**: Linux Mint 21/22, Ubuntu 22.04+ LTS, or WSL2 (Windows Subsystem for Linux).  
* **Hardware Profile**:  
  * Minimum: 16 GB RAM, 4 CPU cores.  
  * Recommended: 64 GB RAM, 16+ CPU cores, and an NVIDIA GPU (RTX 3090/4090 or A100) with CUDA 12.x support for accelerated MACE-OFF23 structural screening.  
* **Quantum Chemistry Engine**: ORCA 6.1.1 installed and correctly path-linked in your system environment.