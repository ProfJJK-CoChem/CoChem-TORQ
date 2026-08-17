"""
CoChem-TORQ 0.0.11
Stage 5.4: Catalog Compilation & PyArrow Serialization
------------------------------------------------------
Parses legacy Fortran-77 SPCAT ASCII outputs (.cat files).
Bypasses Pandas MemoryErrors by utilizing chunked, streaming ingestion.
Serializes massive transition inventories directly into highly compressed, 
columnar Apache Parquet databases for downstream spectroscopic visualization.
"""

import os
import logging
from pathlib import Path
from typing import Any
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    raise ImportError("Critical dependency 'pyarrow' missing. Ensure the CoChem environment silo is active.") from e

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-CatCompile] %(message)s")
logger = logging.getLogger("TorqCatCompiler")

class TorqCatalogCompiler:
    def __init__(self, cat_filepath, point_id="000") -> None:
        """
        Initialize the PyArrow streaming compiler.
        :param cat_filepath: Path to the SPCAT generated .cat file.
        :param point_id: Topographic identifier for provenance tracking.
        """
        self.cat_filepath = Path(cat_filepath)
        self.point_id = point_id
        self.parquet_outpath = Path(f"torq_catalog_{self.point_id}.parquet")

        # SPCAT .cat Fixed-Width Format Definitions
        # Ref: Pickett, H.M. J. Mol. Spectrosc. 148, 371 (1991)
        self.col_widths = [13, 8, 8, 2, 10, 3, 7, 12, 12]
        self.col_names = [
            "Frequency_MHz", "Error_MHz", "Log_Intensity", "DOF", 
            "E_Lower_cm1", "G_Up", "Tag", "QNs_Up", "QNs_Low"
        ]

    def _parse_chunk(self, raw_lines) -> Any:
        """
        Strictly slices Fortran-77 fixed-width strings.
        Avoids the `.split()` method, which fails when large numbers run together
        without spaces (e.g., negative signs fusing with previous columns).
        """
        parsed_data = {col: [] for col in self.col_names}
        
        for line in raw_lines:
            if not line.strip():
                continue
            
            try:
                # Fixed-width slicing based on SPCAT standards
                parsed_data["Frequency_MHz"].append(float(line[0:13].strip()))
                parsed_data["Error_MHz"].append(float(line[13:21].strip()))
                parsed_data["Log_Intensity"].append(float(line[21:29].strip()))
                parsed_data["DOF"].append(int(line[29:31].strip()))
                parsed_data["E_Lower_cm1"].append(float(line[31:41].strip()))
                parsed_data["G_Up"].append(int(line[41:44].strip()))
                parsed_data["Tag"].append(int(line[44:51].strip()))
                parsed_data["QNs_Up"].append(line[51:63].strip())
                parsed_data["QNs_Low"].append(line[63:75].strip())
            except ValueError:
                # Handle Fortran asterisk overflow (e.g., '*******' when bounds exceeded)
                if "*" in line:
                    logger.debug(f"Skipping line due to Fortran overflow: {line.strip()}")
                else:
                    logger.warning(f"Malformed line encountered and skipped: {line.strip()}")
                continue

        return pd.DataFrame(parsed_data)

    def compile_to_parquet(self, chunk_size=100000) -> Any:
        """
        Executes the out-of-core streaming read/write loop.
        Flushes to disk every `chunk_size` rows to guarantee constant O(1) RAM footprint.
        """
        if not self.cat_filepath.exists():
            logger.error(f"Catalog file {self.cat_filepath} not found. SPCAT execution may have failed.")
            return False

        logger.info(f"Initiating out-of-core Parquet compilation for {self.cat_filepath}")
        
        # PyArrow schema definition for strict type enforcement
        schema = pa.schema([
            ('Frequency_MHz', pa.float64()),
            ('Error_MHz', pa.float64()),
            ('Log_Intensity', pa.float64()),
            ('DOF', pa.int32()),
            ('E_Lower_cm1', pa.float64()),
            ('G_Up', pa.int32()),
            ('Tag', pa.int32()),
            ('QNs_Up', pa.string()),
            ('QNs_Low', pa.string())
        ])

        total_rows = 0
        writer = None

        try:
            with open(self.cat_filepath, 'r') as f:
                chunk = []
                for line in f:
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        df_chunk = self._parse_chunk(chunk)
                        table_chunk = pa.Table.from_pandas(df_chunk, schema=schema)
                        
                        if writer is None:
                            writer = pq.ParquetWriter(self.parquet_outpath, schema, compression='snappy')
                        
                        writer.write_table(table_chunk)
                        total_rows += len(df_chunk)
                        chunk = [] # Clear memory
                        logger.info(f"Processed and flushed {total_rows} transitions...")

                # Process remaining lines
                if chunk:
                    df_chunk = self._parse_chunk(chunk)
                    if not df_chunk.empty:
                        table_chunk = pa.Table.from_pandas(df_chunk, schema=schema)
                        if writer is None:
                            writer = pq.ParquetWriter(self.parquet_outpath, schema, compression='snappy')
                        writer.write_table(table_chunk)
                        total_rows += len(df_chunk)

            if writer:
                writer.close()
                
            file_size_mb = os.path.getsize(self.parquet_outpath) / (1024 * 1024)
            logger.info(f"Compilation Complete! {total_rows} transitions secured.")
            logger.info(f"Parquet Payload: {self.parquet_outpath} ({file_size_mb:.2f} MB)")
            return True

        except Exception as e:
            logger.error(f"Catastrophic failure during Parquet serialization: {e}")
            if writer:
                writer.close()
            return False

    def compute_temperature_dependent_partition_function(self, temp_k: float, A_MHz: float = 10000.0, B_MHz: float = 2000.0, C_MHz: float = 1500.0, sigma: int = 1) -> float:
        """
        Computes temperature-dependent rotational partition function Q_rot(T).
        Q_rot(T) = (sqrt(pi) / sigma) * sqrt( (k_B * T)^3 / (h^3 * A * B * C) )
        """
        import math
        kB = 1.380649e-23
        h = 6.62607015e-34
        kT = kB * temp_k
        
        A_Hz = max(abs(A_MHz), 1e-6) * 1e6
        B_Hz = max(abs(B_MHz), 1e-6) * 1e6
        C_Hz = max(abs(C_MHz), 1e-6) * 1e6
        
        q_rot = (math.sqrt(math.pi) / max(sigma, 1)) * math.sqrt((kT**3) / ((h**3) * A_Hz * B_Hz * C_Hz))
        logger.info(f"Q_rot({temp_k} K) = {q_rot:.4f}")
        return q_rot


if __name__ == "__main__":
    # Self-test block: Testing a 3-line SPCAT output to verify fixed-width slicing
    test_cat_content = (
        "    22557.5181  0.0039 -8.8475 3    3.7661  3 13002 1 1 0 1 0 1\n"
        "    22650.0000  0.0010 -7.1234 3   15.1000  5 13002 2 1 1 2 0 2\n"
        "   122650.0000  0.0010 -7.1234 3 1015.1000  5 1300215 11414 014\n" # Intentional spacing squeeze test
    )
    
    with open("test_spcat.cat", "w") as f:
        f.write(test_cat_content)
        
    compiler = TorqCatalogCompiler("test_spcat.cat", point_id="test_001")
    compiler.compile_to_parquet(chunk_size=2) # Force a chunking boundary during test
