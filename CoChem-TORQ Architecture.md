# **CoChem-TORQ Architecture: Part 1 — Torsional Topology & Graph Discovery (Stages 1.0 \- 1.2)**

## **1\. Architectural Intent**

The primary engineering challenge in automated torsional spectroscopy is identifying active internal rotational axes without manual user intervention or breaking molecular connectivity during coordinate transformations. CoChem-TORQ solves this via an algorithmic **5-Option Dihedral Detection Engine** backed by Covalent Radii Summation graph theory.

## **2\. Module Responsibilities**

* **TorqTopology (cochem\_torq\_topology.py)**: Constructs the molecular adjacency graph using standard covalent radii adjusted by a ![][image1] breathing tolerance multiplier. This guarantees that weak interactions (such as hydrogen bonds and halogen contacts) are correctly mapped as topological edges.  
* **Dihedral Identification Engine**: Implements five distinct strategies to locate moving fragments:  
  1. Z-Matrix internal coordinate distance diffing.  
  2. Kabsch RMSD alignment mapping with an SVD reflection trap.  
  3. Graph-theoretic bridge bond severing (networkx.connected\_components).  
  4. Coulomb Matrix eigenspectrum variance analysis.  
  5. Manual user index override.

## **3\. Data Flow & Inter-Process Communication**

\[User XYZ / TOPOS Handoff\]   
       │  
       ▼  
\[Stage 1.0: Adjacency Graph Construction (NetworkX)\]  
       │  
       ├───► Covalent Radii Summation (r\_i \+ r\_j) \* 1.15  
       │  
       ▼  
\[Stage 1.2: Spinning Top Isolation & Rodrigues' Rotation Matrix\]  
       │  
       ▼  
\[Output: torq\_grid.json & torq\_run\_params.json\]

## **4\. Key Dependencies & Failure Points**

* **Dependencies**: numpy, scipy, networkx.  
* **Failure Points**:  
  * Unbonded atomic configurations throwing ValueError during distance matrix calculation.  
  * Collinear atomic arrays causing reflection matrix inversion failures in the Kabsch SVD loop.  
* **Validation Checkpoints**: Strict verification that severed central bonds isolate the spinning top into exactly two distinct connected subgraphs.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAYCAYAAABa1LWYAAABwUlEQVR4Xu2VsUoDQRCGI8ZCEBQlgsldLpfUCnJqYWVroa1gYy92Fj6BYKsIQjoLbWJjowgW6XwCIc9gY6OFkRj/MTOyjPH2Ii4W7gfD7f0zezv/XXaTy3k8HqdEUbRZLpf3tJ4FzLvE/Bmtgzz0TqVS2aJ8HMdLuG4j3nThr4FmEizQlRjEFBo95Xkdvn4xVSgUxsznc7STJBnRtU4Y1JRAZiymmrVabZrGOu8cF6YIMqW1fqCulfYFkW+gvxutp+LSFNdcIHaLxWKoawTkX9HDSR+9QftR61YcmuqioTka80HxiKjrOmYIe3XDzGN8/SNDhCtTQRDMmvdhGC5yfWzqJmKMDCGWdT4zrkxpjPojnTNB/g5xrvWBoIXomNa6DWmS/h765fDMFa3Z1sLXnYx6P9M2xqM6nxnbQt9hMfVCOaXJl9o3dUEMyT0ZSzsVU0kzhdwt4lnrhMXUPWLe1LCn1qm+VCpNmTrB63waMvTsxmBiIuptyCYtxFFHgweUkzrJYZg3tFWe+8D5j1ONdKmpVqvjUe9lDPOcM67dkRoB2hUML2hdQP4Jcaz1v4KO6UM01MLLWsv8xj0ej8fzn3kH7JCfLc9zfjUAAAAASUVORK5CYII=>