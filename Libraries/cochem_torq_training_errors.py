"""Domain-specific typed exceptions for CoChem-TORQ Training Dynamics.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Absolutely no stubs or empty pass blocks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


try:
    from cochem.topos.exceptions import CoChemError  # type: ignore[import-not-found]
except ImportError:
    class CoChemError(Exception):  # type: ignore[no-redef]
        """Root fallback exception for all CoChem operations. [M]"""

        def __init__(self, message: str = "") -> None:
            super().__init__(message)
            self.message = message


class CoChemTorqError(CoChemError):
    """Base exception for TORQ potential backbones and dynamics. [M]"""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message


class TorqTrainingError(CoChemTorqError):
    """Base exception for all TORQ training dynamics errors. [M]"""

    def __init__(
        self,
        message: str,
        error_code: str = "TORQ_TRAIN_GENERIC",
        component: str = "training_dynamics",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.diagnostics = diagnostics if diagnostics is not None else {}


class HDF5LockTimeoutError(TorqTrainingError):
    """Raised when acquiring advisory HDF5 filelock exceeds timeout ceiling. [M]"""

    def __init__(
        self,
        message: str = "HDF5 advisory filelock acquisition timed out.",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TORQ_TRAIN_HDF5_LOCK_TIMEOUT",
            component="storage",
            diagnostics=diagnostics,
        )


class PrecisionDivergenceError(TorqTrainingError):
    """Raised when AMP GradScaler encounters unrecoverable NaN/Inf gradients. [M]"""

    def __init__(
        self,
        message: str = "Unrecoverable NaN/Inf gradients detected across consecutive scaling cycles.",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TORQ_TRAIN_PRECISION_DIVERGENCE",
            component="amp_trainer",
            diagnostics=diagnostics,
        )


class CheckpointCorruptionError(TorqTrainingError):
    """Raised when checkpoint state or SHA-256 checksum validation fails. [M]"""

    def __init__(
        self,
        message: str = "Checkpoint integrity check or SHA-256 checksum verification failed.",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TORQ_TRAIN_CHECKPOINT_CORRUPT",
            component="persistence",
            diagnostics=diagnostics,
        )


class DistributedSyncError(TorqTrainingError):
    """Raised when distributed all-reduce or rank synchronization deadlocks. [M]"""

    def __init__(
        self,
        message: str = "Distributed metric synchronization or collective communication failed.",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TORQ_TRAIN_DISTRIBUTED_SYNC_DEADLOCK",
            component="distributed",
            diagnostics=diagnostics,
        )


class UnsupportedElementError(TorqTrainingError):
    """Raised when input molecular batch contains unmapped atomic species. [M]"""

    def __init__(
        self,
        message: str = "Input molecular batch contains unregistered or unsupported atomic species.",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TORQ_TRAIN_UNSUPPORTED_ELEMENT",
            component="transfer_learning",
            diagnostics=diagnostics,
        )


class ParityVerificationError(TorqTrainingError):
    """Raised when TorchScript compiled model violates eager numerical parity. [M]"""

    def __init__(
        self,
        message: str = "TorchScript compiled model failed numerical parity check against eager model.",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TORQ_TRAIN_PARITY_VIOLATION",
            component="torchscript_export",
            diagnostics=diagnostics,
        )


class DiscontinuousForceError(TorqTrainingError):
    """Raised when graph pruning or switching function violates C^2 continuity. [D]"""

    def __init__(
        self,
        message: str = "Graph pruning or switching function violates C^2 continuity.",
        error_code: str = "TORQ_TRAIN_DISCONTINUOUS_FORCE",
        component: str = "graph_pruning",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class NonReciprocalGraphError(TorqTrainingError):
    """Raised when asymmetric edge pruning breaks Newton's Third Law (F_ij != -F_ji). [D]"""

    def __init__(
        self,
        message: str = "Asymmetric edge configuration detected; breaks Newton's Third Law.",
        error_code: str = "TORQ_TRAIN_NON_RECIPROCAL_GRAPH",
        component: str = "graph_pruning",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class OOMRecoveryError(TorqTrainingError):
    """Raised when dynamic batch-size scaler fails to recover after max step-down attempts. [M]"""

    def __init__(
        self,
        message: str = "Dynamic batch scaler failed to recover after maximum step-down attempts.",
        error_code: str = "TORQ_TRAIN_OOM_RECOVERY_EXHAUSTED",
        component: str = "dynamic_batch",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class EquivarianceBreakError(TorqTrainingError):
    """Raised when geometric tensor operations violate E(3) or SO(3) rotational covariance. [D]"""

    def __init__(
        self,
        message: str = "Geometric tensor operations violate E(3) or SO(3) rotational covariance.",
        error_code: str = "TORQ_TRAIN_EQUIVARIANCE_BREAK",
        component: str = "geometric_tensors",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class SchedulerDivergenceError(TorqTrainingError):
    """Raised when learning rate scheduler encounters NaN or infinite parameter norms. [D]"""

    def __init__(
        self,
        message: str = "Learning rate scheduler encountered NaN or infinite parameter norms.",
        error_code: str = "TORQ_TRAIN_SCHEDULER_DIVERGENCE",
        component: str = "scheduler",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )

