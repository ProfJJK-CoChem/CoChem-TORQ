# **CoChem-TORQ User Manual: Part 3 — Running Torsional Scans & MACE Triage**

## **3.1 Initializing Torsional Topology (Stage 1.0)**

Once your initial geometry (aligned via CoChem-TOPOS) is loaded into memory, you must initialize the topology engine to map covalent connectivity and configure the Method Matrix Cascade:

\# %% \[Cell 1: Initialize Topology & Cascade Parameters\]  
symbols \= \["C", "C", "H", "H", "H", "H", "H", "H"\] \# Ethane mock  
coordinates \= \[  
    \[0.0, 0.0, 0.0\], \[1.5, 0.0, 0.0\],  
    \[-0.5, 1.0, 0.0\], \[-0.5, \-0.5, 0.8\], \[-0.5, \-0.5, \-0.8\],  
    \[2.0, 1.0, 0.0\], \[2.0, \-0.5, 0.8\], \[2.0, \-0.5, \-0.8\]  
\]

\# Instantiate topology engine  
topology \= TorqTopology(symbols, coordinates, is\_complex=False)

\# Configure the Method Matrix Cascade (wB97X-D4 / def2-TZVP tier)  
cascade\_params \= topology.generate\_cascade\_parameters(tier="medium")  
print(f"Cascade Configured: {cascade\_params\['keywords'\]}")

## **3.2 Generating the Torsional Mesh (Stage 1.2)**

To map the potential energy surface across internal rotational degrees of freedom, build the 1D or 2D grid:

\# %% \[Cell 2: Generate 1D Torsional Grid\]  
gridder \= TorqGrid(symbols, coordinates, topology.graph)  
grid\_points \= gridder.generate\_1d\_grid(dihedral=(2, 0, 1, 5), resolution\_deg=15)  
gridder.export\_grid("torq\_grid.json")

## **3.3 Executing MACE-OFF23 Screening (Stage 2.0)**

Screen the generated grid using machine learning force fields while actively managing VRAM:

\# %% \[Cell 3: Execute MACE Triage\]  
triage \= TorqMACETriage(grid\_filepath="torq\_grid.json", model\_size="medium")  
surface\_results \= triage.execute\_surface\_scan(vram\_flush\_interval=50)  
triage.extract\_topographic\_extrema(energy\_threshold\_kcal=1.0)  
triage.export\_triage\_surface("torq\_mace\_surface.json")  
