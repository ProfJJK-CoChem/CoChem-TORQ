"""
CoChem-TORQ 0.0.11
Stage 5.5: SpycFit Payload Synthesizer & FAIR Export
----------------------------------------------------
Constructs the terminal execution block of CoChem-TORQ.
Harvests the massive PyArrow .parquet catalogs, Pickett .var/.int files, 
and JSON provenance trackers. Bundles them into a mathematically locked, 
version-controlled ZIP payload specifically designed for CoChem-SpycFit.
Includes OOM-proof PyArrow metadata inspection for PGOPHER skeleton generation.
"""

import os
import json
import zipfile
import hashlib
import logging
from pathlib import Path
from datetime import datetime

try:
    import pyarrow.parquet as pq
except ImportError as e:
    raise ImportError("Critical dependency 'pyarrow' missing. Ensure the CoChem silo is active.") from e

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Export] %(message)s")
logger = logging.getLogger("TorqExport")

class TorqPayloadSynthesizer:
    def __init__(self, project_name="Default", point_id="000"):
        """
        Initializes the payload synthesizer.
        :param project_name: Name of the active CoChem project.
        :param point_id: Topographic identifier to gather files for.
        """
        self.project_name = project_name
        self.point_id = point_id
        
        # Expected files from Stages 1.0 - 5.4
        self.target_files = {
            "catalog": Path(f"torq_catalog_{self.point_id}.parquet"),
            "variance": Path(f"spcat_{self.point_id}.var"),
            "intensity": Path(f"spcat_{self.point_id}.int"),
            "tensors": Path(f"torq_tensors_{self.point_id}.json"),
            "landscape": Path("landscape.h5") # Optional, might be massive
        }
        
        self.payload_filename = f"CoChem_{self.project_name}_SpycFit_Payload_{self.point_id}.zip"
        self.manifest_filename = "spycfit_manifest.json"
        self.pgo_filename = f"skeleton_{self.point_id}.pgo"

    def _generate_file_hash(self, filepath, chunk_size=8192):
        """Generates a SHA-256 checksum utilizing low-memory chunking."""
        sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError:
            return None

    def generate_pgopher_skeleton(self):
        """
        Creates a .pgo compatible XML skeleton for external visualization.
        Extracts structural parameters from the tensor JSON without loading 
        the massive .parquet catalog into RAM.
        """
        if not self.target_files["tensors"].exists():
            logger.warning("Tensor JSON missing. Cannot build PGOPHER skeleton.")
            return False

        with open(self.target_files["tensors"], "r") as f:
            tensor_data = json.load(f)

        rc = tensor_data.get("tensors", {}).get("rotational_constants_MHz", {})
        A, B, C = rc.get("A", 0), rc.get("B", 0), rc.get("C", 0)

        # Basic PGOPHER XML Structure for an Asymmetric Top
        pgo_xml = f"""<?xml version="1.0"?>
<PGOPHER>
  <AsymmetricTop Name="Point_{self.point_id}">
    <Parameter Name="A" Value="{A}" />
    <Parameter Name="B" Value="{B}" />
    <Parameter Name="C" Value="{C}" />
    <!-- Transitions dynamically appended by SpycFit during ingestion -->
  </AsymmetricTop>
</PGOPHER>"""

        with open(self.pgo_filename, "w") as f:
            f.write(pgo_xml)
            
        logger.info(f"PGOPHER skeleton generated: {self.pgo_filename}")
        return True

    def build_manifest(self):
        """
        Constructs the strict cryptographic manifest. If SpycFit detects a mismatch 
        in these hashes upon ingestion, it will refuse to fit against corrupted data.
        """
        logger.info("Building cryptographic manifest...")
        manifest = {
            "project": self.project_name,
            "point_id": self.point_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cochem_version": "TORQ 0.0.11",
            "files": {}
        }

        for key, filepath in self.target_files.items():
            if filepath.exists():
                file_hash = self._generate_file_hash(filepath)
                file_size = os.path.getsize(filepath)
                manifest["files"][filepath.name] = {
                    "type": key,
                    "sha256": file_hash,
                    "size_bytes": file_size
                }
                
                # OOM-Safe PyArrow Row Count Extraction for the Parquet Catalog
                if key == "catalog":
                    try:
                        parquet_meta = pq.read_metadata(filepath)
                        manifest["files"][filepath.name]["total_transitions"] = parquet_meta.num_rows
                    except Exception as e:
                        logger.warning(f"Failed to read parquet metadata: {e}")
            else:
                if key != "landscape":  # Landscape is optional
                    logger.error(f"Critical workflow failure: Missing {filepath}")
                    raise FileNotFoundError(f"Missing required artifact: {filepath}")

        # Add the PGO skeleton to the manifest
        if Path(self.pgo_filename).exists():
             manifest["files"][self.pgo_filename] = {
                 "type": "pgopher_skeleton",
                 "sha256": self._generate_file_hash(self.pgo_filename),
                 "size_bytes": os.path.getsize(self.pgo_filename)
             }

        with open(self.manifest_filename, "w") as f:
            json.dump(manifest, f, indent=4)
            
        logger.info("Manifest built successfully.")
        return True

    def package_payload(self):
        """
        Compresses the manifest and all verified artifacts into the final FAIR zip payload.
        """
        logger.info(f"Packaging artifacts into {self.payload_filename}...")
        
        with zipfile.ZipFile(self.payload_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            # Add manifest
            zipf.write(self.manifest_filename)
            
            # Add target files
            for key, filepath in self.target_files.items():
                if filepath.exists():
                    zipf.write(filepath)
                    
            # Add PGO skeleton
            if Path(self.pgo_filename).exists():
                zipf.write(self.pgo_filename)

        file_size_mb = os.path.getsize(self.payload_filename) / (1024 * 1024)
        logger.info(f"Payload secured! Output: {self.payload_filename} ({file_size_mb:.2f} MB)")
        return self.payload_filename

if __name__ == "__main__":
    # Self-test block: Mocking files to ensure the packager logic executes
    mock_id = "test_000"
    Path(f"torq_catalog_{mock_id}.parquet").touch()
    Path(f"spcat_{mock_id}.var").touch()
    Path(f"spcat_{mock_id}.int").touch()
    
    with open(f"torq_tensors_{mock_id}.json", "w") as f:
        json.dump({"tensors": {"rotational_constants_MHz": {"A": 100, "B": 200, "C": 300}}}, f)

    synthesizer = TorqPayloadSynthesizer(project_name="IntegrationTest", point_id=mock_id)
    synthesizer.generate_pgopher_skeleton()
    synthesizer.build_manifest()
    synthesizer.package_payload()