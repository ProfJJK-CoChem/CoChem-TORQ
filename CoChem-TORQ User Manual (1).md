# **CoChem-TORQ User Manual: Part 2 — Installation & Stage 0 Configuration**

## **2.1 Cloning and Bootstrapping**

CoChem-TORQ follows the distributed modular repository topology. It resides in its own dedicated GitHub repository and interfaces with the core operating system via CoChem-CORE.

\# 1\. Clone the repository  
git clone \[https://github.com/CoChem/CoChem-TORQ.git\](https://github.com/CoChem/CoChem-TORQ.git)  
cd CoChem-TORQ

\# 2\. Execute the setup orchestrator  
python3 cochem\_setup/setup.py

## **2.2 The Single-Cell Jupyter Integration Pattern**

To keep your working notebooks clean, version-controlled, and debuggable, do not paste monolithic script code into your cells. Instead, import the modular architecture directly from the library path:

\# %% \[Cell 0: Import CoChem-TORQ Pipeline\]  
import sys  
from pathlib import Path

\# Add local libraries to path  
sys.path.append(str(Path.cwd() / "Libraries"))

from cochem\_torq\_topology import TorqTopology  
from cochem\_torq\_grid import TorqGrid  
from cochem\_torq\_mace import TorqMACETriage  
from cochem\_torq\_orca import TorqOrcaExecution  
from cochem\_tensor\_extractor import TorqTensorExtractor  
from cochem\_spcat\_bridge import TorqSpcatBridge  
from cochem\_catalog\_compiler import TorqCatalogCompiler  
from cochem\_torq\_export import TorqPayloadSynthesizer

print("✅ CoChem-TORQ 0.0.11 modules successfully loaded into kernel.")  
