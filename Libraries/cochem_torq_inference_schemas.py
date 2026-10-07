"""Pydantic v2 schemas and data contracts for CoChem-TORQ Inference, Active Learning, and Export.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Strongly typed, validated configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ActiveLearningOrchestratorConfig(BaseModel):
    """Configuration contract for the active learning acquisition and deduplication engine. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_capacity_k: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Active learning batch selection capacity K [E]",
    )
    force_uncertainty_threshold_ev_per_angstrom: float = Field(
        default=0.05,
        gt=0.0,
        description="Epistemic force standard deviation threshold theta_force in eV/Angstrom [E]",
    )
    energy_uncertainty_threshold_ev2_per_atom: float = Field(
        default=0.001,
        gt=0.0,
        description="QBC energy variance threshold theta_energy in eV^2/atom [E]",
    )
    stage_a_rmsd_threshold_angstrom: float = Field(
        default=0.125,
        gt=0.0,
        description="Method Matrix v4 Stage A geometric RMSD threshold delta_RMSD [M]",
    )
    stage_b_rotational_threshold: float = Field(
        default=0.001,
        gt=0.0,
        description="Method Matrix v4 Stage B relative rotational constant invariance delta_B/B (--bthr) [M]",
    )
    triage_tier: Literal["T3-10s", "T3-1h", "T3-12h", "T3O-1h", "T3O-12h"] = Field(
        default="T3-10s",
        description="Initial baseline QM routing tier [M]",
    )
    staging_manifest_dir: Path = Field(
        ...,
        description="Air-gapped manifest export directory in Ring 3 [M]",
    )

    @model_validator(mode="before")
    @classmethod
    def remap_legacy_field_names(cls, data: Any) -> Any:
        """Support field name aliases across specification revisions. [D]"""
        if isinstance(data, dict):
            mapped = dict(data)
            if "force_uncertainty_threshold" in mapped and "force_uncertainty_threshold_ev_per_angstrom" not in mapped:
                mapped["force_uncertainty_threshold_ev_per_angstrom"] = mapped.pop("force_uncertainty_threshold")
            if "energy_uncertainty_threshold" in mapped and "energy_uncertainty_threshold_ev2_per_atom" not in mapped:
                mapped["energy_uncertainty_threshold_ev2_per_atom"] = mapped.pop("energy_uncertainty_threshold")
            if "rmsd_dedup_threshold_angstrom" in mapped and "stage_a_rmsd_threshold_angstrom" not in mapped:
                mapped["stage_a_rmsd_threshold_angstrom"] = mapped.pop("rmsd_dedup_threshold_angstrom")
            return mapped
        return data

    @property
    def force_uncertainty_threshold(self) -> float:
        """Alias for force_uncertainty_threshold_ev_per_angstrom. [D]"""
        return self.force_uncertainty_threshold_ev_per_angstrom

    @property
    def energy_uncertainty_threshold(self) -> float:
        """Alias for energy_uncertainty_threshold_ev2_per_atom. [D]"""
        return self.energy_uncertainty_threshold_ev2_per_atom

    @property
    def rmsd_dedup_threshold_angstrom(self) -> float:
        """Alias for stage_a_rmsd_threshold_angstrom. [D]"""
        return self.stage_a_rmsd_threshold_angstrom


class ChunkedHDF5DataModuleConfig(BaseModel):
    """Configuration contract for chunked HDF5 PyTorch Lightning DataModule. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    h5_path: Path = Field(..., description="Path to chunked HDF5 dataset [M]")
    batch_size: int = Field(default=32, ge=1, description="Batch size per device [M]")
    num_workers: int = Field(
        default=2, ge=0, description="Number of dataloader worker processes [M]"
    )
    chunk_cache_bytes: int = Field(
        default=16 * 1024 * 1024,
        ge=1024 * 1024,
        description="HDF5 raw data chunk cache size rdcc_nbytes [E]",
    )
    chunk_cache_slots: int = Field(
        default=10007,
        ge=1009,
        description="Prime number of chunk cache hash slots rdcc_nslots [D]",
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin host memory for non-blocking GPU transfer [M]",
    )
    train_val_test_split: List[float] = Field(
        default=[0.8, 0.1, 0.1],
        description="Train, validation, and test split ratios [M]",
    )

    @model_validator(mode="after")
    def validate_splits(self) -> "ChunkedHDF5DataModuleConfig":
        """Verify train, val, and test splits sum to unity within tolerance. [D]"""
        if abs(sum(self.train_val_test_split) - 1.0) > 1e-5:
            raise ValueError("train_val_test_split must sum to 1.0")
        return self


class CommitteeEnsembleConfig(BaseModel):
    """Configuration contract for Committee Ensemble uncertainty wrapper. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_models_m: int = Field(
        default=8, ge=2, le=32, description="Number of committee models M [E]"
    )
    max_concurrent_models_vram: int = Field(
        default=2,
        ge=1,
        description="Maximum models loaded into active VRAM simultaneously [E]",
    )
    synchronize_cuda_streams: bool = Field(
        default=True,
        description="Execute stream synchronization between model forward passes [D]",
    )
    vectorized: bool = Field(
        default=True,
        description="Enable vectorized inference via torch.vmap [M]",
    )
    concurrency_mode: Literal["vmap", "cuda_streams", "serial"] = Field(
        default="vmap",
        description="Concurrency strategy for ensemble execution [M]",
    )
    vram_headroom_threshold_mb: float = Field(
        default=2048.0,
        ge=512.0,
        description="Minimum VRAM headroom threshold in MB before throttling [E]",
    )
    max_batch_size: int = Field(
        default=128,
        ge=1,
        description="Maximum batch size for vectorized inference [E]",
    )


class C2SmoothCutoffConfig(BaseModel):
    """Configuration contract for C^2-smooth radial cutoff envelope. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cutoff_radius_rc: float = Field(
        default=5.0,
        gt=1.0,
        description="Radial cutoff boundary radius r_c in Angstroms [M]",
    )
    polynomial_degree: Literal[5] = Field(
        default=5,
        description="Degree of C^2 continuous switching polynomial [D]",
    )


class GNNGradientDebuggerConfig(BaseModel):
    """Configuration contract for message-passing GNN gradient health debugger. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exploding_grad_threshold: float = Field(
        default=1e3, gt=0.0, description="L2 gradient norm explosion threshold [D]"
    )
    vanishing_grad_threshold: float = Field(
        default=1e-7, gt=0.0, description="L2 gradient norm vanishing threshold [D]"
    )
    consecutive_vanishing_blocks: int = Field(
        default=3,
        ge=1,
        description="Consecutive layers below vanishing threshold before alert [D]",
    )
    enabled: bool = Field(
        default=True, description="Enable active hook monitoring [M]"
    )


class PBCRadialGraphConfig(BaseModel):
    """Configuration contract for periodic boundary condition graph engine. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cutoff_radius_rc: float = Field(
        default=5.0,
        gt=1.0,
        description="Radial graph cutoff radius in Angstroms [M]",
    )
    compute_virial_stress: bool = Field(
        default=True,
        description="Compute analytical unit cell virial stress tensor [D]",
    )


class HPORunConfig(BaseModel):
    """Configuration contract for automated hyperparameter optimization. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_name: str = Field(
        ..., description="Unique identifier for the HPO study [M]"
    )
    n_trials: int = Field(
        default=100, ge=1, description="Total optimization trials [E]"
    )
    pruner: Literal["ASHA", "MedianPruner", "Hyperband"] = Field(
        default="ASHA", description="Pruning strategy [E]"
    )
    grace_period: int = Field(
        default=10, ge=1, description="Epochs before pruning evaluation [E]"
    )
    storage_uri: str = Field(
        ..., description="Storage backend URI (sqlite:///... or h5py) [M]"
    )
    w_energy: float = Field(
        default=1.0, ge=0.0, description="Potential energy loss weight [E]"
    )
    w_force: float = Field(
        default=10.0, ge=0.0, description="Atomic force loss weight [E]"
    )
    lr_min: float = Field(
        default=1e-5, gt=0.0, description="Lower bound for learning rate [E]"
    )
    lr_max: float = Field(
        default=1e-2, gt=0.0, description="Upper bound for learning rate [E]"
    )
    cutoff_min: float = Field(
        default=4.0, ge=1.0, description="Minimum cutoff radius in Angstroms [E]"
    )
    cutoff_max: float = Field(
        default=6.5, ge=1.0, description="Maximum cutoff radius in Angstroms [E]"
    )
    rbf_options: List[int] = Field(
        default=[16, 32, 64], description="Candidate RBF basis counts [E]"
    )
    depth_options: List[int] = Field(
        default=[3, 4, 5, 6], description="Candidate interaction depths [E]"
    )
    embedding_dim_options: List[int] = Field(
        default=[64, 128, 256],
        description="Candidate feature embedding dimensions [E]",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "HPORunConfig":
        """Verify learning rate and cutoff interval boundaries. [D]"""
        if self.lr_min >= self.lr_max:
            raise ValueError("lr_min must be strictly less than lr_max")
        if self.cutoff_min >= self.cutoff_max:
            raise ValueError("cutoff_min must be strictly less than cutoff_max")
        return self


class DeltaMLConfig(BaseModel):
    """Configuration contract for Delta-Learning architecture. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_method: Literal["GFN2-xTB", "PM6", "LennardJones", "EMT"] = Field(
        default="GFN2-xTB", description="Baseline physical engine [M]"
    )
    qm_target_method: str = Field(
        default="wB97M-V/def2-TZVP",
        description="High-level QM target benchmark [M]",
    )
    energy_unit: Literal["eV", "Hartree", "kcal/mol"] = Field(
        default="eV", description="Internal standard energy unit [D]"
    )
    length_unit: Literal["Angstrom", "Bohr"] = Field(
        default="Angstrom", description="Internal standard length unit [D]"
    )
    use_d3_dispersion: bool = Field(
        default=True,
        description="Automatically augment semi-empirical baselines with D3 dispersion [M]",
    )
    dispersion_damping: Literal["bj", "zero"] = Field(
        default="bj",
        description="D3 dispersion damping scheme (Becke-Johnson or zero-damping) [M]",
    )


class ConformalPredictorConfig(BaseModel):
    """Configuration contract for inductive conformal prediction uncertainty. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Target miscoverage significance level [D]",
    )
    regularization_energy: float = Field(
        default=1e-6,
        gt=0.0,
        description="Numerical regularizer epsilon_E in eV [E]",
    )
    regularization_force: float = Field(
        default=1e-6,
        gt=0.0,
        description="Numerical regularizer epsilon_F in eV/Angstrom [E]",
    )
    strict_calibration_size: bool = Field(
        default=True,
        description=(
            "Raise CalibrationSizeError if calibration sample size is"
            " insufficient [M]"
        ),
    )
    apply_bonferroni: bool = Field(
        default=False,
        description=(
            "Apply Bonferroni correction for simultaneous joint 3N force bounds"
            " [D]"
        ),
    )


class LBFGSOptimizerConfig(BaseModel):
    """Configuration contract for L-BFGS geometry optimizer with Eckart projection. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_iterations: int = Field(
        default=500, ge=1, description="Maximum optimization iterations [M]"
    )
    history_size: int = Field(
        default=10, ge=1, description="Two-loop recursion memory depth [D]"
    )
    dtype: Literal["float64"] = Field(
        default="float64",
        description="Mandatory precision for geometry optimization [M]",
    )
    tol_max_g: float = Field(
        default=0.00051422,
        gt=0.0,
        description="Max force convergence threshold in eV/Angstrom [M]",
    )
    tol_rms_g: float = Field(
        default=0.00034453,
        gt=0.0,
        description="RMS force convergence threshold in eV/Angstrom [M]",
    )
    tol_max_d: float = Field(
        default=5.29177e-5,
        gt=0.0,
        description="Max displacement threshold in Angstroms [M]",
    )
    tol_rms_d: float = Field(
        default=3.54549e-5,
        gt=0.0,
        description="RMS displacement threshold in Angstroms [M]",
    )
    tol_energy: float = Field(
        default=2.72114e-5,
        gt=0.0,
        description="Energy change convergence threshold in eV [M]",
    )
    max_step: float = Field(
        default=0.1,
        gt=0.0,
        description="Maximum Cartesian step displacement in Angstroms [E]",
    )
    clash_distance: float = Field(
        default=0.7,
        gt=0.0,
        description="Clash distance abort threshold in Angstroms [E]",
    )
    c1: float = Field(
        default=1e-4,
        gt=0.0,
        lt=0.5,
        description="Strong Wolfe Armijo sufficient decrease parameter [D]",
    )
    c2: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Strong Wolfe curvature condition parameter [D]",
    )


class DispersionD3Config(BaseModel):
    """Configuration contract for Grimme D3 empirical dispersion layer. [M]"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    functional: str = Field(
        default="wB97M-V", description="Underlying DFT functional [M]"
    )
    damping: Literal["BJ", "zero"] = Field(
        default="BJ", description="Dispersion damping variant [D]"
    )
    s6: float = Field(default=1.0, ge=0.0, description="Dipole scale factor [E]")
    s8: float = Field(
        default=1.0, ge=0.0, description="Quadrupole scale factor [E]"
    )
    a1: float = Field(
        default=0.5, ge=0.0, description="Becke-Johnson damping parameter a1 [E]"
    )
    a2: float = Field(
        default=3.0, ge=0.0, description="Becke-Johnson damping parameter a2 [E]"
    )
    c9_cutoff: float = Field(
        default=16.0,
        gt=0.0,
        description="Three-body dispersion cutoff in Angstroms [E]",
    )
    pair_cutoff: float = Field(
        default=25.0,
        gt=0.0,
        description="Pairwise dispersion cutoff in Angstroms [E]",
    )
    data_manifest_sha256: str = Field(
        default="7dc504e705bf220fc0a014f21f18e02ad205cb68e714ca3aaec2c8b48fc7e327",
        description="SHA-256 checksum of Ring 2 dispersion table [M]",
    )



class NeighborListResult(NamedTuple):
    """Container for spatial neighbor list search results. [M]"""

    edge_index: torch.Tensor  # Shape: [2, num_edges], dtype: torch.int64
    edge_vector: torch.Tensor  # Shape: [num_edges, 3], dtype matches coordinates (float32/float64)
    edge_distance: torch.Tensor  # Shape: [num_edges], dtype matches coordinates (float32/float64)


class ConformalInterval(NamedTuple):
    """Computed intervals; coverage requires independently justified calibration assumptions."""

    energy_lower: Optional[float]  # Unit: eV; None for force-only requests
    energy_upper: Optional[float]  # Unit: eV; None for force-only requests
    force_lower: torch.Tensor  # Shape: [N, 3], Unit: eV/Angstrom
    force_upper: torch.Tensor  # Shape: [N, 3], Unit: eV/Angstrom
    confidence_level: float  # 1 - alpha, e.g., 0.95


class LBFGSOptimizationState(BaseModel):
    """Telemetry and state snapshot of L-BFGS geometry optimization. [M]"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    converged: bool
    iterations: int
    final_energy: float  # Unit: eV
    max_force: float  # Unit: eV/Angstrom
    rms_force: float  # Unit: eV/Angstrom
    max_displacement: Optional[float] = None  # Unit: Angstrom
    rms_displacement: Optional[float] = None  # Unit: Angstrom
    energy_change: Optional[float] = None  # Unit: eV
    final_coordinates: Optional[torch.Tensor] = (
        None  # Shape: [N, 3], Unit: Angstrom
    )


class MultiTaskPrediction(BaseModel):
    """Encapsulates multi-task predictions for potential energy, forces, and HOMO-LUMO gap [M]."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    energy: float = Field(
        ..., description="Molecular potential energy in eV [M]"
    )
    forces: List[List[float]] = Field(
        ...,
        description="Atomic Cartesian forces [N, 3] in eV/Angstrom [M]",
    )
    homo_lumo_gap: float = Field(
        ...,
        gt=0.0,
        description="Fundamental HOMO-LUMO electronic gap in eV [M]",
    )
    energy_log_variance: float = Field(
        ..., description="Task log-variance s_E = log(sigma_E^2) [D]"
    )
    gap_log_variance: float = Field(
        ..., description="Task log-variance s_G = log(sigma_G^2) [D]"
    )

    @field_validator("forces")
    @classmethod
    def validate_forces_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if not v or len(v) == 0:
            raise ValueError("Forces tensor cannot be empty.")
        for row in v:
            if len(row) != 3:
                raise ValueError(
                    f"Each force vector must be 3D Cartesian [x, y, z], got dimension {len(row)}."
                )
        return v


class FiniteDiffVerificationResult(BaseModel):
    """Validation report certifying agreement between analytic and finite-difference forces [M]."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    max_absolute_error: float = Field(
        ...,
        description="Maximum absolute force component error (L_infinity) in eV/Angstrom [M]",
    )
    relative_frobenius_error: float = Field(
        ..., description="Relative Frobenius norm error [M]"
    )
    step_size: float = Field(
        ..., description="Displacement step size h in Angstrom [E]"
    )
    passed: bool = Field(
        ...,
        description="True if within acceptance criteria (L_inf < 1e-4 eV/A) [M]",
    )
    dtype: str = Field(
        ...,
        description="Execution tensor precision, strictly torch.float64 [M]",
    )
    atom_count: int = Field(
        ...,
        ge=1,
        description="Number of atoms in the evaluated structure [M]",
    )


class VibrationalModes(BaseModel):
    """Full normal mode report containing projected harmonic frequencies and zero-point energy [D]."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    frequencies_cm1: List[float] = Field(
        ...,
        description="Signed harmonic vibrational frequencies in cm^-1 (nu < 0 for imaginary modes) [D]",
    )
    zero_point_energy_ev: float = Field(
        ...,
        ge=0.0,
        description="Harmonic Zero-Point Vibrational Energy (ZPVE) in eV [D]",
    )
    imaginary_mode_count: int = Field(
        ...,
        ge=0,
        description="Count of transition-state imaginary normal modes (nu < 0) [D]",
    )
    eigenvalues: List[float] = Field(
        ...,
        description="Mass-weighted Hessian eigenvalues in eV/(Angstrom^2 * u) [D]",
    )
    projected_degrees_of_freedom: int = Field(
        ...,
        description="Number of projected translational and rotational degrees of freedom (5 or 6) [D]",
    )
    mass_weighting_standard: str = Field(
        default="CIAAW_MONOISOTOPIC",
        description="Governing standard for isotopic masses [M]",
    )

    zpe_scale_factor: float = Field(default=1.0, ge=0.5, le=1.5)
    unscaled_zero_point_energy_ev: Optional[float] = Field(default=None, ge=0.0)
    scaled_frequencies_cm1: Optional[List[float]] = None
    method: Optional[str] = None
    basis: Optional[str] = None
    scaling_provenance: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_mode_consistency(self) -> "VibrationalModes":
        if self.zpe_scale_factor != 1.0:
            provenance = self.scaling_provenance or {}
            if not all(provenance.get(k) for k in ("source", "method", "basis", "domain", "target_observable")):
                raise ValueError("Nonunit ZPE scaling requires complete calibration provenance.")
            if (not self.method or not self.basis or provenance["target_observable"] != "zpe"
                    or str(provenance["method"]).strip().lower() != self.method.strip().lower()
                    or str(provenance["basis"]).strip().lower() != self.basis.strip().lower()):
                raise ValueError("ZPE calibration does not match the result method, basis, and observable.")
        if self.scaled_frequencies_cm1 is not None:
            raise ValueError("Scaled fundamental frequencies require a separate validated observable contract.")
        actual_imaginary = sum(1 for f in self.frequencies_cm1 if f < 0.0)
        if self.imaginary_mode_count != actual_imaginary:
            raise ValueError(
                f"imaginary_mode_count mismatch: reported {self.imaginary_mode_count}, "
                f"but frequencies_cm1 contains {actual_imaginary} negative modes."
            )
        if len(self.frequencies_cm1) != len(self.eigenvalues):
            raise ValueError(
                f"Dimension mismatch between frequencies ({len(self.frequencies_cm1)}) "
                f"and eigenvalues ({len(self.eigenvalues)})."
            )
        return self


class ONNXExportSpec(BaseModel):
    """Configuration specification governing TorchDynamo ONNX compilation and dynamic axes [D]."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    opset_version: int = Field(
        ..., ge=17, description="Target ONNX opset version (>= 17) [E]"
    )
    export_mechanism: str = Field(
        ...,
        description="Export engine: dynamo_export_aot_autograd or direct_analytical_force_head [D]",
    )
    dynamic_axes: Dict[str, Dict[int, str]] = Field(
        ...,
        description="Dynamic axes dictionary covering 3D tensors and edge_index [D]",
    )
    precision: str = Field(
        ..., description="Model numerical precision: float32 or float64 [E]"
    )

    @field_validator("opset_version")
    @classmethod
    def validate_opset(cls, v: int) -> int:
        if v < 17:
            from Libraries.cochem_torq_inference_errors import OpsetUnsupportedError

            raise OpsetUnsupportedError(
                f"Target ONNX opset {v} is unsupported. CoChem-TORQ requires opset >= 17 for PyTorch 2.0+ Dynamo export."
            )
        return v


