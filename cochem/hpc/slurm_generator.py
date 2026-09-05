"""Slurm dry-run batch script generator and preflight resource validator.

Implements preflight partition bound validation, Method Matrix v4 solver resource
distribution (%maxcore, OpenMP threads), and Tripartite Air-Gapped staging scripts.
"""

import math
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

from src.cochem.hpc.models import SlurmDryRunResult, SlurmJobDirectiveSpec


class SlurmDryRunGenerator:
    """Preflight validator and generator for HPC Slurm batch job submissions."""

    def __init__(self) -> None:
        self._walltime_regex = re.compile(r"^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$")

    def validate_directives(
        self,
        spec: SlurmJobDirectiveSpec,
        partition_limits: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Validate job directive specifications against Slurm syntax and cluster partition bounds."""
        errors: List[str] = []

        # 1. Walltime format and component range validation
        match = self._walltime_regex.match(spec.walltime_str)
        if not match:
            errors.append(
                f"Invalid walltime_str '{spec.walltime_str}'. Expected 'D-HH:MM:SS' or 'HH:MM:SS'."
            )
        else:
            days_str, hours_str, mins_str, secs_str = match.groups()
            hours = int(hours_str)
            mins = int(mins_str)
            secs = int(secs_str)
            if mins >= 60:
                errors.append(f"Walltime minutes ({mins}) must be strictly less than 60.")
            if secs >= 60:
                errors.append(f"Walltime seconds ({secs}) must be strictly less than 60.")
            if days_str is not None and hours >= 24:
                errors.append(f"Walltime hours ({hours}) must be less than 24 when days are specified.")

        # 2. POSIX path verification (Windows path letters or backslashes rejected)
        scratch_posix = PurePosixPath(spec.scratch_dir)
        if not scratch_posix.is_absolute() or "\\" in spec.scratch_dir or ":" in spec.scratch_dir:
            errors.append(
                f"scratch_dir '{spec.scratch_dir}' must be a valid absolute POSIX path starting with '/'."
            )

        artifact_posix = PurePosixPath(spec.artifact_dir)
        if not artifact_posix.is_absolute() or "\\" in spec.artifact_dir or ":" in spec.artifact_dir:
            errors.append(
                f"artifact_dir '{spec.artifact_dir}' must be a valid absolute POSIX path starting with '/'."
            )

        # 3. Partition bounds validation if supplied
        if partition_limits:
            allowed_partitions = partition_limits.get("allowed_partitions")
            if allowed_partitions and spec.partition not in allowed_partitions:
                errors.append(
                    f"Partition '{spec.partition}' is not in allowed partition list: {allowed_partitions}."
                )

            max_nodes = partition_limits.get("max_nodes")
            if max_nodes is not None and spec.nodes > max_nodes:
                errors.append(
                    f"Requested nodes ({spec.nodes}) exceeds partition max_nodes limit ({max_nodes})."
                )

            max_cpus = partition_limits.get("max_cpus_per_task")
            if max_cpus is not None and spec.cpus_per_task > max_cpus:
                errors.append(
                    f"Requested cpus_per_task ({spec.cpus_per_task}) exceeds partition limit ({max_cpus})."
                )

            max_mem = partition_limits.get("max_mem_mb")
            if max_mem is not None and spec.memory_per_node_mb > max_mem:
                errors.append(
                    f"Requested memory ({spec.memory_per_node_mb} MB) exceeds partition limit ({max_mem} MB)."
                )

            max_gpus = partition_limits.get("max_gpus_per_node")
            if max_gpus is not None and spec.gpus_per_node is not None and spec.gpus_per_node > max_gpus:
                errors.append(
                    f"Requested gpus_per_node ({spec.gpus_per_node}) exceeds partition limit ({max_gpus})."
                )

        return errors

    def generate_sbatch_script(
        self,
        spec: SlurmJobDirectiveSpec,
        solver: str = "orca",
        input_filename: str = "input.inp",
        partition_limits: Optional[Dict[str, Any]] = None,
    ) -> SlurmDryRunResult:
        """Compile verified directives into a structured, executable SBATCH submission script."""
        validation_errors = self.validate_directives(spec, partition_limits=partition_limits)
        validation_warnings: List[str] = []

        estimated_mem_per_rank = spec.memory_per_node_mb // spec.ntasks_per_node
        orca_maxcore_mb: Optional[int] = None
        solver_lower = solver.lower().strip()

        # Compute ORCA %maxcore per rank: floor((spec.memory_per_node_mb * 0.75) / spec.ntasks_per_node)
        if solver_lower == "orca":
            orca_maxcore_mb = int(math.floor((spec.memory_per_node_mb * 0.75) / spec.ntasks_per_node))

        if validation_errors:
            return SlurmDryRunResult(
                is_valid=False,
                generated_script_content="",
                estimated_memory_per_rank_mb=estimated_mem_per_rank,
                orca_maxcore_mb=orca_maxcore_mb,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
            )

        # Build SBATCH header lines
        header_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.job_name}",
            f"#SBATCH --partition={spec.partition}",
            f"#SBATCH --nodes={spec.nodes}",
            f"#SBATCH --ntasks-per-node={spec.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
            f"#SBATCH --time={spec.walltime_str}",
            f"#SBATCH --mem={spec.memory_per_node_mb}M",
        ]

        if spec.gpus_per_node is not None and spec.gpus_per_node > 0:
            header_lines.append(f"#SBATCH --gpus-per-node={spec.gpus_per_node}")
        if spec.account:
            header_lines.append(f"#SBATCH --account={spec.account}")
        if spec.qos:
            header_lines.append(f"#SBATCH --qos={spec.qos}")

        scratch_path = PurePosixPath(spec.scratch_dir).as_posix()
        artifact_path = PurePosixPath(spec.artifact_dir).as_posix()

        # Solver execution logic
        if solver_lower == "orca":
            solver_block = (
                f"# Solver execution: ORCA\n"
                f"# Calculated %maxcore: {orca_maxcore_mb} MB\n"
                f"orca {input_filename} > orca_output.out"
            )
        elif solver_lower == "crest":
            solver_block = (
                f"# Solver execution: CREST\n"
                f"export OMP_NUM_THREADS={spec.cpus_per_task}\n"
                f"crest {input_filename} --nci --nocross --noreftopo -T {spec.cpus_per_task} > crest_output.out"
            )
        elif solver_lower == "xtb":
            solver_block = (
                f"# Solver execution: xTB\n"
                f"export OMP_NUM_THREADS={spec.cpus_per_task}\n"
                f"xtb {input_filename} > xtb_output.out"
            )
        else:
            solver_block = f"# Generic solver execution\n{solver} {input_filename}"

        script_content = (
            "\n".join(header_lines)
            + "\n\n"
            + "# Tripartite Air-Gapped Staging Directives\n"
            + "export COCH_SRC=\"${COCH_SRC:-/opt/cochem/src}\"\n"
            + f"export COCH_ARTIFACTS=\"{artifact_path}\"\n"
            + f"export COCH_SCRATCH=\"{scratch_path}\"\n"
            + "export SLURM_TMPDIR=\"${COCH_SCRATCH}\"\n\n"
            + "mkdir -p \"${COCH_SCRATCH}\"\n"
            + "mkdir -p \"${COCH_ARTIFACTS}\"\n\n"
            + f"cp \"${{COCH_SRC}}/{input_filename}\" \"${{COCH_SCRATCH}}/\"\n"
            + "cd \"${COCH_SCRATCH}\"\n\n"
            + f"{solver_block}\n\n"
            + "# Synchronize final outputs to append-only artifacts storage\n"
            + "cp -r \"${COCH_SCRATCH}\"/* \"${COCH_ARTIFACTS}/\"\n"
        )

        return SlurmDryRunResult(
            is_valid=True,
            generated_script_content=script_content,
            estimated_memory_per_rank_mb=estimated_mem_per_rank,
            orca_maxcore_mb=orca_maxcore_mb,
            validation_errors=[],
            validation_warnings=validation_warnings,
        )
