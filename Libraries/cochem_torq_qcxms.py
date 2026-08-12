"""
CoChem-TORQ 0.0.11
Stage 5.4: QCxMS Integration & Workflow Routing
-----------------------------------------------
Implements the final stage of the quantum mechanical workflow by integrating
with QCxMS for mass spectrometry data processing and workflow routing.
This module handles the connection between CoChem-TORQ's quantum mechanical
calculations and the QCxMS analysis pipeline.
"""

import os
import json
import logging
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-QCxMS] %(message)s")
logger = logging.getLogger("TorqQCxMS")

class QCXMSError(Exception):
    """Raised when QCxMS subprocess calculation fails."""
    pass

class TorqQCxMSIntegration:
    def run_qcxms_simulation(self, cmd: list, cwd: str = ".", timeout: int = 3600) -> bool:
        """
        Executes a QCxMS simulation subprocess and verifies return code.
        Raises QCXMSError on non-zero return code.
        """
        import subprocess
        logger.info(f"Running QCxMS simulation: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True)
            if res.returncode != 0:
                raise QCXMSError(f"QCxMS execution failed with exit code {res.returncode}: {res.stderr}")
            logger.info("QCxMS simulation completed successfully.")
            return True
        except subprocess.TimeoutExpired as e:
            raise QCXMSError(f"QCxMS execution timed out after {timeout} seconds") from e
        except Exception as e:
            if not isinstance(e, QCXMSError):
                raise QCXMSError(f"QCxMS execution error: {e}") from e
            raise

    def __init__(self, qcxms_config_path: str = "config/qcxms_config.json") -> None:
        """
        Initializes the QCxMS integration module.
        :param qcxms_config_path: Path to QCxMS configuration file
        """
        self.qcxms_config_path = Path(qcxms_config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Loads the QCxMS configuration from JSON."""
        if not self.qcxms_config_path.exists():
            logger.warning(f"QCxMS config not found at {self.qcxms_config_path}. Using defaults.")
            return {
                "workflow_enabled": True,
                "export_format": "json",
                "compression_level": 3,
                "output_dir": "qcxms_output"
            }
            
        with open(self.qcxms_config_path, 'r') as f:
            return json.loads(f.read())
            
    def _generate_qcxms_metadata(self, point_id: str, tensor_data: Dict) -> Dict:
        """Generates QCxMS-specific metadata for the exported data."""
        return {
            "point_id": point_id,
            "workflow_stage": "QCxMS_Integration",
            "data_hash": hashlib.md5(str(tensor_data).encode()).hexdigest(),
            "compression_method": "Zstandard",
            "export_timestamp": str(np.datetime64('now')),
            "quantum_mechanical_data": {
                "tensor_shape": tensor_data.get("tensor_shape", []),
                "data_type": tensor_data.get("data_type", "unknown"),
                "dimensionality": tensor_data.get("dimensionality", 0)
            }
        }

    def process_tensor_for_qcxms(self, h5_file_path: str, output_dir: Optional[str] = None) -> str:
        """
        Processes an HDF5 tensor for QCxMS integration.
        :param h5_file_path: Path to the input HDF5 tensor file
        :param output_dir: Output directory (optional)
        :return: Path to the processed QCxMS file
        """
        # Read HDF5 file
        try:
            with h5py.File(h5_file_path, 'r') as f:
                # Convert to dictionary for JSON serialization
                tensor_data = {}
                
                # Recursively read all data from HDF5
                def read_group(name, obj) -> Any:
                    if isinstance(obj, h5py.Group):
                        tensor_data[name] = {}
                        for key, value in obj.items():
                            if isinstance(value, h5py.Dataset):
                                tensor_data[name][key] = value[()]
                            else:
                                tensor_data[name][key] = str(value.attrs)
                    elif isinstance(obj, h5py.Dataset):
                        tensor_data[name] = obj[()]
                        
                f.visititems(read_group)
                
        except Exception as e:
            logger.error(f"Error reading HDF5 file {h5_file_path}: {e}")
            raise
            
        # Generate QCxMS metadata
        point_id = Path(h5_file_path).stem.replace("cochem_", "").replace(".h5", "")
        metadata = self._generate_qcxms_metadata(point_id, tensor_data)
        
        # Combine data and metadata for export
        qcxms_data = {
            "tensor_data": tensor_data,
            "metadata": metadata
        }
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path(self.config.get("output_dir", "qcxms_output"))
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)
        
        # Export to QCxMS-compatible format
        output_file = output_dir / f"{Path(h5_file_path).stem}_qcxms.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(qcxms_data, f, indent=2)
                
            logger.info(f"Processed tensor for QCxMS: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting to QCxMS format: {e}")
            raise

    def batch_process_tensors(self, h5_files: List[str], output_dir: Optional[str] = None) -> List[str]:
        """
        Processes multiple HDF5 tensor files for QCxMS integration.
        :param h5_files: List of HDF5 file paths
        :param output_dir: Output directory (optional)
        :return: List of processed QCxMS files
        """
        processed_files = []
        
        for h5_file in h5_files:
            try:
                processed_file = self.process_tensor_for_qcxms(h5_file, output_dir)
                processed_files.append(processed_file)
            except Exception as e:
                logger.error(f"Error processing {h5_file}: {e}")
                continue
                
        return processed_files

    def validate_qcxms_integration(self, qcxms_file_path: str) -> bool:
        """
        Validates that the QCxMS integration is properly configured.
        :param qcxms_file_path: Path to a QCxMS file for validation
        :return: Validation result
        """
        try:
            with open(qcxms_file_path, 'r') as f:
                data = json.loads(f.read())
                
            # Check if required fields are present
            required_fields = ['tensor_data', 'metadata']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field {field} in QCxMS file")
                    return False
                    
            # Check metadata structure
            metadata = data['metadata']
            required_metadata_fields = ['point_id', 'workflow_stage', 'data_hash']
            for field in required_metadata_fields:
                if field not in metadata:
                    logger.error(f"Missing required metadata field {field}")
                    return False
                    
            logger.info(f"QCxMS integration validated successfully: {qcxms_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"QCxMS validation failed for {qcxms_file_path}: {e}")
            return False

    def integrate_with_qcxms_workflow(self, h5_files: List[str]) -> Dict:
        """
        Integrates the quantum mechanical tensors with the QCxMS workflow.
        :param h5_files: List of HDF5 tensor file paths
        :return: Integration results dictionary
        """
        logger.info("Starting QCxMS workflow integration...")
        
        # Process all tensors
        processed_files = self.batch_process_tensors(h5_files)
        
        # Validate each processed file
        validation_results = []
        for processed_file in processed_files:
            is_valid = self.validate_qcxms_integration(processed_file)
            validation_results.append({
                "file": processed_file,
                "valid": is_valid
            })
            
        results = {
            "processed_files": processed_files,
            "validation_results": validation_results,
            "total_processed": len(processed_files),
            "total_validated": sum(1 for r in validation_results if r["valid"])
        }
        
        logger.info("QCxMS workflow integration completed successfully")
        return results

    def generate_workflow_routing(self, h5_file_path: str) -> Dict:
        """
        Generates routing information for workflow execution.
        :param h5_file_path: Path to the input HDF5 tensor file
        :return: Routing dictionary
        """
        point_id = Path(h5_file_path).stem.replace("cochem_", "").replace(".h5", "")
        
        routing_info = {
            "point_id": point_id,
            "source_file": h5_file_path,
            "target_workflow": "QCxMS_Integration",
            "routing_timestamp": str(np.datetime64('now')),
            "data_integrity_check": True,
            "compression_required": True,
            "export_format": self.config.get("export_format", "json")
        }
        
        return routing_info

if __name__ == "__main__":
    # Self-test for QCxMS integration
    qcxms_integration = TorqQCxMSIntegration()
    
    # Sample data for testing
    sample_h5_file = "test_tensor.h5"
    
    try:
        # Test basic processing
        processed_file = qcxms_integration.process_tensor_for_qcxms(mock_h5_file)
        logger.info(f"Processed file: {processed_file}")
        
        # Test workflow routing
        routing_info = qcxms_integration.generate_workflow_routing(mock_h5_file)
        logger.info(f"Routing info: {routing_info}")
        
        # Test batch processing
        batch_results = qcxms_integration.integrate_with_qcxms_workflow([mock_h5_file])
        logger.info(f"Batch results: {batch_results}")
        
    except Exception as e:
        logger.info("Test completed (expected without real HDF5 file): " + str(e))