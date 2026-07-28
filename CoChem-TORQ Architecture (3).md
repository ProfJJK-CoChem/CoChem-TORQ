# **CoChem-TORQ Architecture: Part 4 — High-Performance Serialization & SpycFit Handoff (Stages 5.4 \- 5.5)**

## **1\. Architectural Intent**

High-temperature spectroscopic simulations generate tens of millions of rotational transitions. Standard in-memory dataframes (Pandas) will instantly crash Jupyter kernels via Out-Of-Memory (OOM) exceptions. CoChem-TORQ solves this with a streaming, chunk-based PyArrow Parquet compiler and a cryptographic payload packager designed for seamless handoff to CoChem-SpycFit.

## **2\. Module Responsibilities**

* **TorqCatalogCompiler (cochem\_catalog\_compiler.py)**: Ingests legacy Fortran-77 SPCAT ASCII .cat outputs using fixed-width slicing (line\[0:13\], etc.) and streams them into compressed Apache Parquet tables.  
* **TorqPayloadSynthesizer (cochem\_torq\_export.py)**: Bundles Parquet catalogs, SPCAT parameter files, tensor provenance, and PGOPHER XML skeletons into a version-controlled ZIP archive. Generates spycfit\_manifest.json embedded with SHA-256 file hashes to prevent downstream corruption.

## **3\. Data Flow & Inter-Process Communication**

\[SPCAT .cat Output\] ──► \[Stage 5.4: Chunked Fortran Slicing (100,000 rows/chunk)\]  
                              │  
                              ▼  
                       \[PyArrow Parquet Serialization (Snappy Compression)\]  
                              │  
                              ▼  
                       \[Stage 5.5: Cryptographic Hashing & ZIP Packaging\]  
                              │  
                              ▼  
                       \[Output: CoChem\_{Project}\_SpycFit\_Payload.zip\]

## **4\. Key Dependencies & Failure Points**

* **Dependencies**: pyarrow, pandas, zipfile, hashlib.  
* **Failure Points**:  
  * Fortran string overflow asterisks (\*\*\*\*\*\*\*) breaking standard float parsers.  
  * RAM exhaustion when handling uncompressed multi-gigabyte spectral line lists.  
* **Validation Checkpoints**: Strict schema validation, chunk-based disk flushing, and automatic metadata row-count extraction without loading raw table arrays into memory.