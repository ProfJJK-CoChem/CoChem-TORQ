"""CoChem CLI Subsystem.

Provides interactive and headless command-line interfaces, workspace scaffolding,
and environment initialization wizards.
"""

from src.cochem.cli.init_wizard import (
    InitWizardError,
    InsufficientDiskSpaceError,
    WorkspacePermissionError,
    ExistingProjectError,
    BinaryAuditRecord,
    ProjectManifest,
    InitWizardResult,
    run_init_wizard,
)

__all__ = [
    "InitWizardError",
    "InsufficientDiskSpaceError",
    "WorkspacePermissionError",
    "ExistingProjectError",
    "BinaryAuditRecord",
    "ProjectManifest",
    "InitWizardResult",
    "run_init_wizard",
]
