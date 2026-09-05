"""Domain-specific typed exceptions for CoChem-TORQ Inference, Active Learning, and Export.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Absolutely no stubs or empty pass blocks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CoChemError(Exception):
    """Root exception for CoChem framework. [M]"""

    def __init__(self, message: str = "Generic CoChem error") -> None:
        super().__init__(message)
        self.message = message


class CoChemTorqError(CoChemError):
    """Base exception for all TORQ sub-framework operations. [M]"""

    def __init__(
        self,
        message: str = "Generic TORQ error",
        error_code: str = "TORQ_GENERIC_ERROR",
        component: str = "inference_export",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.component = component
        self.diagnostics = diagnostics or {}


class PhysicsDivergenceError(CoChemTorqError):
    """Raised when fundamental physical invariants are violated (non-positive gap, negative mass, unphysical ZPVE) [M]."""

    def __init__(
        self,
        message: str = "Physical invariant violation detected",
        error_code: str = "TORQ_PHYSICS_DIVERGENCE",
        component: str = "physical_invariants",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class NumericalParityError(CoChemTorqError):
    """Raised when finite-difference gradients diverge from analytical forces beyond acceptance tolerance or float32 is detected [M]."""

    def __init__(
        self,
        message: str = "Numerical parity tolerance exceeded or unsupported precision",
        error_code: str = "TORQ_NUMERICAL_PARITY_ERROR",
        component: str = "finite_difference",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class ConcurrencyLockError(CoChemTorqError):
    """Raised when filesystem locking fails or HPC distributed lock prohibitions are violated on shared filesystems [M]."""

    def __init__(
        self,
        message: str = "Filesystem locking failure or distributed lock prohibition violated",
        error_code: str = "TORQ_CONCURRENCY_LOCK_ERROR",
        component: str = "environment_concurrency",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class OpsetUnsupportedError(CoChemTorqError):
    """Raised when an unsupported ONNX opset (< 17) or invalid autograd export pipeline is invoked [M]."""

    def __init__(
        self,
        message: str = "Unsupported ONNX opset or invalid autograd export pipeline",
        error_code: str = "TORQ_OPSET_UNSUPPORTED",
        component: str = "onnx_export",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class TorqInferenceError(CoChemTorqError):
    """Base exception for all TORQ inference and export errors. [M]"""

    def __init__(
        self,
        message: str,
        error_code: str = "TORQ_INF_GENERIC",
        component: str = "inference_engine",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            component=component,
            diagnostics=diagnostics,
        )


class ActiveLearningSelectionError(TorqInferenceError):
    """Raised when active learning selection or pool deduplication fails. [M]"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="TORQ_AL_SELECTION_ERR",
            component="active_learning",
            **kwargs,
        )


class HDF5DataModuleLockError(TorqInferenceError):
    """Raised when multi-worker HDF5 handle acquisition encounters lock timeout. [M]"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="TORQ_HDF5_LOCK_ERR",
            component="hdf5_datamodule",
            **kwargs,
        )


class EnsembleConsensusError(TorqInferenceError):
    """Raised when committee ensemble encounters dimension mismatch or consensus divergence. [D]"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="TORQ_ENSEMBLE_ERR",
            component="committee_ensemble",
            **kwargs,
        )


class CutoffContinuityError(TorqInferenceError):
    """Raised when radial cutoff function violates C^2 smoothness or boundary zero condition. [D]"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message, error_code="TORQ_CUTOFF_ERR", component="c2_cutoff", **kwargs
        )


class GradientExplosionError(TorqInferenceError):
    """Raised when GNN message passing activations or gradients exceed stability threshold. [D]"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="TORQ_GRAD_EXPLOSION_ERR",
            component="gnn_debugger",
            **kwargs,
        )


class VanishingGradientWarning(UserWarning):
    """Emitted when message passing gradients drop below minimum threshold across consecutive layers. [D]"""

    def __init__(
        self, message: str = "GNN message passing gradients vanishing"
    ) -> None:
        super().__init__(message)
        self.message = message


class PBCGraphError(TorqInferenceError):
    """Raised when periodic boundary condition graph construction or virial calculation fails. [D]"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="TORQ_PBC_GRAPH_ERR",
            component="pbc_graph",
            **kwargs,
        )


class AirGapViolationError(TorqInferenceError):
    """Raised when inference or active learning loops attempt forbidden outbound networking or direct subprocess execution. [M]"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="TORQ_AIRGAP_ERR",
            component="airgap_sandbox",
            **kwargs,
        )


class HardwareDispatchError(TorqInferenceError):
    """Raised if hardware accelerator encounters unrecoverable runtime states without a valid fallback. [M]"""

    def __init__(
        self, message: str, diagnostics: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message,
            error_code="TORQ_HW_DISPATCH_FAIL",
            component="hardware_dispatcher",
            diagnostics=diagnostics,
        )


class AirGapIntegrityError(TorqInferenceError):
    """Raised if network sockets are opened during inference or if Ring 2 data SHA-256 hashes mismatch. [M]"""

    def __init__(
        self, message: str, diagnostics: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message,
            error_code="TORQ_AIRGAP_INTEGRITY_FAIL",
            component="airgap_enforcer",
            diagnostics=diagnostics,
        )


class ClashDetectedError(TorqInferenceError):
    """Raised during L-BFGS line-search if any interatomic distance drops below 0.7 Angstroms. [E]"""

    def __init__(
        self,
        message: str,
        min_distance: float,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        diags = dict(diagnostics or {})
        diags["min_distance_angstrom"] = float(min_distance)
        super().__init__(
            message,
            error_code="TORQ_GEOM_CLASH_DETECTED",
            component="lbfgs_optimizer",
            diagnostics=diags,
        )


class ConvergenceError(TorqInferenceError):
    """Raised if geometry optimization fails to reach Method Matrix force thresholds within maximum iterations. [M]"""

    def __init__(
        self,
        message: str,
        iterations: int,
        final_force: float,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        diags = dict(diagnostics or {})
        diags["iterations"] = int(iterations)
        diags["final_max_force"] = float(final_force)
        super().__init__(
            message,
            error_code="TORQ_LBFGS_NON_CONVERGENCE",
            component="lbfgs_optimizer",
            diagnostics=diags,
        )


class CalibrationSizeError(TorqInferenceError):
    """Raised in strict initialization mode if conformal calibration dataset size n < ceil((1 - alpha) / alpha). [M]"""

    def __init__(self, message: str, n_samples: int, n_required: int) -> None:
        super().__init__(
            message,
            error_code="TORQ_CONFORMAL_INSUFFICIENT_CALIBRATION",
            component="conformal_predictor",
            diagnostics={"n_samples": int(n_samples), "n_required": int(n_required)},
        )


class BaselineExecutionError(TorqInferenceError):
    """Raised when Delta-ML baseline calculation fails or returns non-physical values. [M]"""

    def __init__(
        self,
        message: str,
        method: str = "GFN2-xTB",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        diags = dict(diagnostics or {})
        diags["baseline_method"] = method
        super().__init__(
            message,
            error_code="TORQ_DELTA_BASELINE_FAIL",
            component="delta_ml_engine",
            diagnostics=diags,
        )


class DispersionParameterError(TorqInferenceError):
    """Raised when dispersion damping parameters or C6/C8 tables fail SHA-256 verification. [M]"""

    def __init__(
        self, message: str, expected_sha: str, calculated_sha: str
    ) -> None:
        super().__init__(
            message,
            error_code="TORQ_DISPERSION_PARAM_CORRUPT",
            component="dispersion_layer",
            diagnostics={
                "expected_sha256": expected_sha,
                "calculated_sha256": calculated_sha,
            },
        )

