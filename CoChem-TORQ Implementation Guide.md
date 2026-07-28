# **CoChem-TORQ 0.0.11: Comprehensive Implementation & Interfacing Guide**

## **1\. Executive Summary & Operational Scope**

**CoChem-TORQ** (Torsional Optimization & Rotational Quantification) is the specialized rotational spectroscopy and conformational routing module of the CoChem ecosystem. It operates immediately downstream of **CoChem-TOPOS** (Topology and Geometry Discovery) and **CoChem-Cascade** (Method Matrix refinement), taking raw minimum-energy configurations and scaling them into dense, high-fidelity torsional potential energy surfaces (PES), partition functions, and quantum mechanical spectral catalogs.

This guide provides the complete programmatic specifications, directory pathways, active dependencies, and interfacing protocols required to operate CoChem-TORQ 0.0.11 inside the unified Jupyter execution environment.

## **2\. System Architecture & Module Map**

The CoChem-TORQ architecture is structured into a rigorous 5-stage processing pipeline that respects context-window safety and hardware resource bounds:

\[CoChem-TOPOS / Cascade\] \---\> landscape.h5 \---\> \[Stage 1.0: Topology & Dihedral Matrix\]  
                                                          │  
                                                          ▼  
                                              \[Stage 1.2: Grid Mesh Generation\]  
                                                          │  
                                                          ▼  
                                              \[Stage 2.0: MACE-OFF23 Triage\]  
                                                          │  
                                                          ▼  
                                              \[Stage 3.0: ORCA 6.1.1 Escalation\]  
                                                          │  
                                                          ▼  
                                              \[Stage 4.1: Tensor & Inertia Engine\]  
                                                          │  
                                                          ▼  
                                              \[Stage 5.0: Partition & SPCAT Bridge\]  
                                                          │  
                                                          ▼  
                                              \[Stage 5.4: PyArrow Parquet Compiler\]  
                                                          │  
                                                          ▼  
                                              \[Stage 5.5: SpycFit Payload Synthesizer\]

### **Script Inventory & Responsibilities**

1. **cochem\_torq\_topology.py (Stage 1.0)**: Ingests molecular coordinates, builds the covalent bonding graph via Covalent Radii Summation (with a 1.15x breathing tolerance), executes the 5-Option Dihedral Detection Engine, and injects Method Matrix Cascade runtime parameters (torq\_run\_params.json).  
2. **cochem\_torq\_grid.py (Stage 1.2)**: Isolates rotating fragments using NetworkX graph cutting and generates 1D and 2D dense torsional meshes using Rodrigues' rotation formula.  
3. **cochem\_torq\_mace.py (Stage 2.0)**: Ingests Cartesian grids and processes them through the MACE-OFF23 Machine Learning Potential. Enforces active VRAM flushing (vram\_flush\_interval=50) to prevent PyTorch memory leaks.  
4. **cochem\_torq\_orca.py (Stage 3.0)**: Takes topographic extrema, generates strict ORCA 6.1.1 input blocks with constrained dihedrals, and executes dynamic memory backoff (%maxcore halving) upon detecting out-of-memory errors.  
5. **cochem\_tensor\_extractor.py (Stage 4.1)**: Translates coordinates to the Center of Mass (COM), computes principal moments of inertia, evaluates rotational constants (![][image1] in MHz) using immutable CODATA 2022 constants, and catches ![][image2] linear geometries via SVD rank testing.  
6. **cochem\_spcat\_bridge.py (Stage 5.0)**: Computes rigid-rotor harmonic-oscillator (RRHO) partition functions (![][image3]), evaluates rotational symmetry divisors (![][image4]), detects low-frequency Large Amplitude Motions (LAM ![][image5]), and synthesizes Pickett .var and .int seed files.  
7. **cochem\_catalog\_compiler.py (Stage 5.4)**: Streams legacy Fortran SPCAT ASCII .cat files and serializes them into compressed columnar Apache Parquet format using PyArrow.  
8. **cochem\_torq\_export.py (Stage 5.5)**: Bundles parquet catalogs, variance/intensity files, tensor provenance, and a PGOPHER .pgo skeleton into a cryptographic, version-controlled ZIP archive for seamless CoChem-SpycFit ingestion.

## **3\. Dependency Matrix & Environment Integration**

CoChem-TORQ relies on the centralized micro-silo configuration defined by cochem\_system\_config.json (generated during Stage 0).

### **Required Python Packages**

* numpy, scipy (Numerical and spatial matrix processing)  
* networkx (Graph-theoretic bond severing and fragment isolation)  
* torch, ase, mace-torch (High-speed machine learning force field triage)  
* pyarrow, pandas (Out-of-core columnar catalog serialization)  
* molsym (Rigorous point-group symmetry number evaluation)

### **Environment Silo Routing Rule**

High-risk computational packages must never be imported directly into the master notebook kernel if they conflict with base dependencies. They are dynamically isolated within their respective Conda silos:

{  
  "silo\_registry": {  
    "torq\_engine": "$HOME/.cochem/silos/torq/bin/python",  
    "mace\_backend": "$HOME/.cochem/silos/mace/bin/python"  
  }  
}

## **4\. Operational Workflows & Execution Guide**

To run CoChem-TORQ within your local Jupyter environment (Workflow\_Master.ipynb), execute scripts sequentially via module imports or designated cell blocks.

### **Step-by-Step Execution Sequence**

1. **Initialize System Registry**: Ensure cochem\_system\_config.json is present in $HOME/CoChem\_Artifacts/.  
2. **Topological Mapping**: Run cochem\_torq\_topology.py to identify rotatable dihedrals and configure calculation tiers (r2SCAN-3c \-\> wB97X-D4 \-\> DLPNO-CCSD(T)).  
3. **Mesh Generation**: Call cochem\_torq\_grid.py to pre-compute rotated Cartesian coordinates.  
4. **ML Screening**: Pass coordinates to cochem\_torq\_mace.py to isolate local minima and barrier peaks.  
5. **Ab Initio Escalation**: Execute cochem\_torq\_orca.py for high-fidelity gradient optimizations and harmonic frequency calculations.  
6. **Spectroscopic Handoff**: Run cochem\_tensor\_extractor.py, cochem\_spcat\_bridge.py, and cochem\_catalog\_compiler.py to generate the .parquet spectral line lists.  
7. **SpycFit Packaging**: Execute cochem\_torq\_export.py to create the final FAIR-compliant delivery archive.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAAC3ElEQVR4Xu2XPYjUUBSFZ1kFBUFERlhmJm9+RBBBhUFEUFRQsPEHLbWy0cLKQkGblcXKbhpB1sJCBNdOtlAsrES0FWx1kQUREQQFF3b1nJkbeR6TTJLNQop8cGFy7rv3vdy83Lyp1SoqKoqg2WxudM6tqF5m2u32Eaz5HmzQ7/fXmzyB6/P/DMwKEryD/VY9C4i/GQTBAvPAlmGfeG3aoulvNS4LKMBh5HhvuW5AmqRBfwM7Y/N+kLD0YLG7LPmqikGQa4vl+u/pQDtKHxY9rb40IHbOcr9QH4H+jX6s4br6UoPFPUaSz0UUAzl2Mk+n09kT4Zuym5nHnBvUn0S3291ssXP8rX4C3xPYCopxTH2paDQaW7Gw50hym5PV6/VNOiYLyDGIKyoWeZC+PDvDCvFDdR/4z+Yp9BA0ze0IXqxZ07EJp3RcFhD/FfZddeJG7/OS6uNAzCzXxtdZfT4YczX3rkDwT1TxMn8ziU3Y13FZsILOhte9Xm8b5jht+h1/bFosdu2+dLjpC7Dj4TXfcSvGSX9cFpz1i1ardciN+sPQwtywgcaMw4udV19R8LX44gu28FV1Yt4sc6hOkLebJz8fDuOwuy6prxCQ/CnsvhsdWELjNZ/AQx2fFpfQL9iYLf9L9SURFmPcjkWxTmDcAdXH4mKaWJ7F+lh85KEKX62m+TNt97SvL8Y84ila9UQQNBPXJG2xH1XnU/WOu5EEdtiK2848IdLPT7n62GRV87F1vVY9BAVzmP+u6okgaDcTqx5ik+q3fJ3psXHE2aeZRREX+9Pw+IyC7BXf37OHSzhCw3eOY9CYT6kP2j7keKZ6EpPhDUXdGBb5QP2B973G9SvYr6gDGfSLGiu2jPy3NM6DxVriWHX44NS5Q/IO/+sg934du+a40dlhQvWicBGvZ2nBTllQrSjsL8G06qWEDQrMqF4UKPQ1/jVQvZS4HCfHLCD/FdUqKioqKkrGH36z79cyNRQWAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAZCAYAAACy0zfoAAAB1ElEQVR4Xu1UPUvEQBCNhWChiMj5wXHZJAce2ugRtPIPWNjcNX79ArG3shL/gNpZiIIIFiJcYSsWItZWco0WFjZW11jc+YbbgcmSvSSF2OyDYfe9mZ28bHbjeQ4O/wyl1Jbv+/umzkD+GNHTcY/aBbOGNFHTC4JgU+bBl6BfYtygETEv8wmgWSyb2cwh15KNwjBcAf9G1EVNnXowJ6DfG7RDUXMq8yZPAG8QoGC3VCqNZpjrpmh7sjnm14hno2Yb0WGO/utGviG5FRnmaEeGpFapVJahPzBPWw9tVusTmuffOYm05gzKUWCjZ4hXq9Up8Fc6FsRrtdqYXp/YGf4iiDXi+sydEVdZZ04iw1xDGHzB2KFDzXmsiwaZs/XNjawmyB+wQR1PnOOLZTOHF7mQemEMMuf3b12T5hhP2CAe+kgabu/iIHO2vrlha8IP8MSFAG+zQc354P+dubTtV/2f75Wpl8vlSVpDBrBuRJtN/Bqwo9Pa3KrUC6OoOQLvnJ53EUcyz5+bzEu9MGzmoigap1wcx8NSh9ZEtJhjd3aU+OES0O+GbrfUckPps4L4QvMPCs3fKcd1MDjHdYhPPb+VvQhYfw79B+Od6p/Ltlnj4ODg4OB5v8X8spIEClDDAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAaCAYAAACn4zKhAAAEhklEQVR4Xu1YO2hUQRTdEAXF/ydu/rObRBY/iLJoiIoIKmihiI2fYGWhhSJYKGJhRG1SSbAKQbFQQQQb01kEK1HQxkIkARUhYAhikUIliee8d8dcb3bf7qq7MfIODG/uPXfeuzN35s7MSyRixIgR499HQ0PDCudcD0pvc3Pzfsv/D0ilUvNQrrCPLPX19U3WZkYAp57DoS8Y+LNKdxG6yXQ67bTtbEVbW9ti9gd9vA2xStTVkD9A/03bVhTZbHYuHcOAd1mOgIM3yCemnJ6VQB8OST9rLUeAG0d5bfUVgSzJSQbDcgS4DvJNTU3rLTebwEHGhHpr9R7gn8lkqyzg1B6ZHe2W80gmkwtog3LXcmVEtVVY5Js0Fq2tras4w2WA865m8Odogz2x0XJlhQzuZ6s3mCN2A5YoFyRF5s3RzOEtLS1LrD4X8J4x8f+05TTAd9IO785armzA7D/Cj/JpOQ3Y1FU6CIQEYtzqoRspIQBp8Z2rYI7lNTAOdyoeBHxwiB+tqalZaDkNP0Pg5EnLlRuNjY3zdSBKCQAB+z4JwpjlLGAzLP2cZ7mygY7JDIkEbD6KXeRMKhckEBzIcdSXWz4KaDMgbZ9YzsCn3I+WyAcEayPsX6BaxcCh/gll2NpFwhURBLm4+XP1H0E62WP1hYA2W9HJp3jew/Ow5aPgJAhMNZbTgM1VsUtZLh8wJrvQ5rqXmSm0XBTQYLRQEPDiU7T5G5c1CUKn1RcC2nxmCpI94nsi4oRjAft++e5Dy2mAH3E59p9SwG8gMLutPhIY4C46yOVOGfUv4nBwWmLHRZ52qkDboyjtLty07eyuhq4b5YTY7oRzZyAP4nkc32sw9nlBX/QeoAJRFJzccXyfePwU2R9Xq+DTecp2r5HbdTcmYIYy7LZ5G+n7MT92BP2CvNqF41H0RGFD3hKfSMMg54ujwe0SF7QDpgnbwK90kjyeW1w4i/YJx3a9rHOwnZyo+H5y6jUF4UwAPEoNBC9oytf3oq5C/RL1LHowPS9ZgH4H2QLyA7zrAso6uXtwIx8Q+2BP4Z2Kbem7f1ExYINBceaxqr8iRwN7KWJqSoUb0S8DAfm+d5iAs8ucLHE8RzOZzKIp62jAfk2uAHjQJ/ixw+rzQa8AF/4h+IYypgdf/8RTgznBflHHUySPryhr5bDwFaWDHCbrZtT7WZexiUzzBeHCFPPOy3jpU0UHYO5z5gYtHRxSMo+2PD0E3JTlvwHtk6yum5pHUFZCN4y+tlBOqftUKkzlP9O0C4/CQTaA/XbURz33W5AgcEB9mZZGoOvnx4yOp61gNoj8VfaVNU6CwVxa6F5SKZg+TjuCc9an1KkKNrdUfSIRpqAOWRV9Pliov4auzdv+Fpg2tHM2HRHQv+NM0TrmTOhHWEeq2ODkbO7CIDwEX4vnI91mJqH7yH3D8rLvveQxnSkY2OQ5tpHBD7IB+raXRXQl7Qd5Aacu42XPcmxYAWwANNCuLmF+wMHBpbmCOZOATwfh6xuUa5bzkL0naX2nbFc0ZfZT62LEiBEjRowYswQ/AOHKYs7w+fORAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAApUlEQVR4XmNgGAVDH6ioqPDJy8s/BeL/6FhBQeEQimKgQCVUYiaQ7gPihyC+nJxcKBCHANmSyIrNQZLGxsasSGYwQDW4IIvBJH4C8RQs4iANvujiYAmgLR5owoxQDTYookABQZCEqKgoD5q4C0gcyGREFgcBFqgEC0xAVlbWFurMYiR1CACUmAXES6HsaCD+C7QhH10dCgA5CRR8QL9woMuNgmEAADpILV3MjgILAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF0AAAAZCAYAAABTuCK5AAADCUlEQVR4Xu1YO4hTQRTNsisoio2GuPm9JARFEJv4wVZsFvGDWAhaq4UoCOoKdouVjYUghLXQThAbsQ9YbLGFjYt1tlDYQhE2goHd9ZzkTvbm5uUHyfKic+AyM+fe+Z03k5lJLObhMYnI5XLziURir+U9xoAgCOYg+C2kW/F4fJ/1e4wRkRGdA4F9zufzp5DOYkVcQ1q3cel0eg/43xK/lUql0jYm6oia6G0G4Q+ZsBkR+oCUp1jGhyi2RUUcURV9A4KfDok5Q7/hlmGLmttpoP/z2Wz2ajfD7g1M/HhELxaL+9H4L3T63PrCgNgaYkuWd0gmkwcRswKrah51HtkPEXWMXHQ0mGejEOOe9fVCP9Hhv8J2YRXNo84F8oNcwfDhMogt0wqFwmHHozxL46pkuVQq7WI+k8mc2K7d7As78DKyU5ofFqMSnb+tS7AqD0DrHARB83BcgK2jjbccGNKc8j/tJTpF07wGD1v4lxG7yjInLHXuoIvdSF/BqsKV8XNwnHHI/4R9gX1F3GtyaOMm45CdUV0MBDVWZxUbMwimOSDYUmwEXx8Tu+3KWGXHyLkVwUmHDdRNpM8uaUwyJkI50dHHRRfD+jKG1qJB+brUzTtOeNY9qbmxg9sPHa/B3lvfKCETfif5RSlXdEy/lQ7+qPhr1qeBdgqM09veta3jCHL0WX6skO3KibStgFFD+lhhPisHZjC86HPi7yk66zMusqI7yAOGW7LjejcMMIGXIswHzQvXuK0g5hzym7BPJua+CBP6Gxv8a6I7UHQOBOklFKetvx9QtyLCtESXA47cC1X+CPu+XbP1s7OpOQ15wbKdDvHQ5rzLu907MaI74NQ/ggG9ga1xstbfDTzEIMDjmDqMwT2UCbdWMF+ilkP5D2JvuHIYguY1kcKXHce/G3Q9jmEiRdfAwJ7A6urJ3hOYxF3Gw75Jyud9x4cDf1YE5DVvA/bMxnQBr7W8abEu7QFJdX2sYwyrbFNuSjXYD+EaddxHEG49MA+1yGDHr1YeHh4eHh4eHh4e/x/+Av03IlevJKsMAAAAAElFTkSuQmCC>