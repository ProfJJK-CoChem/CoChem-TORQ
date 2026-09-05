"""
Authentic HPC / Slurm Dispatch Controller and Shell Injection Defense Engine.
Method Matrix v4: §8A, §13, and SRS Chunk 4 Suggestion #33.
"""
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional


SLURM_PARAM_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.:@/]+$")
SLURM_WALLTIME_REGEX = re.compile(r"^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$")


def sanitize_slurm_parameter(param_name: str, value: str) -> str:
    """Sanitizes user-provided Slurm parameters against shell metacharacters and command injection.

    Raises ValueError if malicious characters or spaces are detected. [M]
    """
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Slurm parameter '{param_name}' cannot be empty.")

    # Reject forbidden shell metacharacters immediately
    forbidden_tokens = [";", "&", "|", "$", "`", "\n", "\r", "(", ")", "<", ">", "!", "{", "}"]
    for token in forbidden_tokens:
        if token in cleaned:
            raise ValueError(f"Shell injection detected in {param_name}: '{cleaned}' (forbidden token '{token}')")

    if not SLURM_PARAM_REGEX.match(cleaned):
        raise ValueError(f"Shell injection detected in {param_name}: '{cleaned}' (failed character whitelist)")

    return cleaned


def validate_slurm_walltime(walltime_str: str) -> str:
    """Validates Slurm walltime format (D-HH:MM:SS or HH:MM:SS), time component bounds,
    and enforces maximum allowable walltime cap of 48:00:00 per Method Matrix §8A. [M]
    """
    walltime_str = walltime_str.strip()
    match = SLURM_WALLTIME_REGEX.match(walltime_str)
    if not match:
        raise ValueError(
            f"Invalid walltime format '{walltime_str}'. Expected 'D-HH:MM:SS' or 'HH:MM:SS'."
        )
    days_str, hours_str, mins_str, secs_str = match.groups()
    hours = int(hours_str)
    mins = int(mins_str)
    secs = int(secs_str)
    if mins >= 60:
        raise ValueError(f"Walltime minutes ({mins}) must be strictly less than 60.")
    if secs >= 60:
        raise ValueError(f"Walltime seconds ({secs}) must be strictly less than 60.")
    if days_str is not None and hours >= 24:
        raise ValueError(f"Walltime hours ({hours}) must be less than 24 when days are specified.")

    days = int(days_str) if days_str is not None else 0
    total_seconds = days * 86400 + hours * 3600 + mins * 60 + secs
    if total_seconds <= 0:
        raise ValueError(f"Walltime '{walltime_str}' must be strictly greater than zero.")

    max_seconds = 48 * 3600  # Strict 48:00:00 cap per Method Matrix §8A
    if total_seconds > max_seconds:
        raise ValueError(
            f"Requested walltime '{walltime_str}' ({total_seconds / 3600:.2f}h) exceeds "
            f"maximum allowable limit of 48:00:00 (48 hours) per Method Matrix §8A."
        )

    return walltime_str


def generate_slurm_script(
    job_name: str = "cochem_job",
    partition: str = "standard",
    nodes: int = 1,
    ntasks_per_node: int = 16,
    cpus_per_task: int = 1,
    mem: str = "32GB",
    walltime: str = "04:00:00",
    engine: str = "orca",
    input_deck_path: str = "input.inp",
    scratch_dir: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """Generates an authentic Slurm SBATCH submission script adhering to Method Matrix §8A."""
    sanitized_job_name = sanitize_slurm_parameter("job_name", job_name)
    sanitized_partition = sanitize_slurm_parameter("partition", partition)
    sanitized_mem = sanitize_slurm_parameter("mem", mem)
    validated_walltime = validate_slurm_walltime(walltime)
    sanitized_engine = sanitize_slurm_parameter("engine", engine.lower())

    if nodes < 1:
        raise ValueError(f"Nodes must be >= 1, got {nodes}")
    if ntasks_per_node < 1:
        raise ValueError(f"Tasks per node must be >= 1, got {ntasks_per_node}")

    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={sanitized_job_name}",
        f"#SBATCH --partition={sanitized_partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={validated_walltime}",
        f"#SBATCH --mem={sanitized_mem}",
    ]

    if email:
        sanitized_email = sanitize_slurm_parameter("email", email)
        sbatch_lines.append(f"#SBATCH --mail-user={sanitized_email}")
        sbatch_lines.append("#SBATCH --mail-type=END,FAIL")

    # Environment setup and scratch handling
    sbatch_lines.extend([
        "",
        "# Environment Module Loading per Method Matrix §8A",
        f"module load {sanitized_engine}",
        "",
        "# Ephemeral Scratch Setup",
        'SCRATCH_DIR="${SLURM_TMPDIR:-/tmp/cochem_${SLURM_JOB_ID}}"',
        'mkdir -p "$SCRATCH_DIR"',
        'cd "$SCRATCH_DIR"',
        "",
        "# Physical Binary Execution",
    ])

    if sanitized_engine == "orca":
        sbatch_lines.append(f"orca {input_deck_path} > orca.out 2>&1")
    elif sanitized_engine == "cfour":
        sbatch_lines.append(f"xcfour > cfour.out 2>&1")
    elif sanitized_engine == "xtb":
        sbatch_lines.append(f"xtb {input_deck_path} --opt > xtb.out 2>&1")
    else:
        sbatch_lines.append(f"{sanitized_engine} {input_deck_path}")

    sbatch_lines.append("")
    return "\n".join(sbatch_lines)


def submit_slurm_job(script_path: Path) -> str:
    """Submits an sbatch script or returns a structured pending message when on non-HPC systems."""
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Slurm script not found at '{script_path}'")

    sbatch_bin = shutil.which("sbatch")
    if sbatch_bin is None:
        return f"PENDING_LOCAL_STAGED: Script synthesized at {script_path.as_posix()}; sbatch binary unavailable on local environment."

    res = subprocess.run(
        [sbatch_bin, str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"Submitted batch job (\d+)", res.stdout)
    if match:
        return match.group(1)
    return res.stdout.strip()


class SlurmSubmissionController:
    """Controller orchestrating Slurm validation, synthesis, and submission for the GUI."""

    def __init__(self, default_partition: str = "standard") -> None:
        self.default_partition = default_partition
        self.last_submitted_job_id: Optional[str] = None

    def validate_and_generate(self, **kwargs: Any) -> str:
        return generate_slurm_script(**kwargs)

    def dispatch(self, script_path: Path) -> str:
        return submit_slurm_job(script_path)
