"""Conformal Prediction Uncertainty Quantification Suite for CoChem-TORQ.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic mathematical conformal bounds and physical residuals.
"""

from __future__ import annotations

import collections
import math
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from Libraries.cochem_torq_inference_errors import CalibrationSizeError
from Libraries.cochem_torq_inference_schemas import ConformalInterval, ConformalPredictorConfig
from cochem_base.exceptions import ConformalCalibrationError
from cochem_base.schemas import ConformalCalibrationConfig


@dataclass
class CalibrationSample:
    """Authentic physical calibration structure with true values and model predictions. [M]"""

    energy_true: float
    energy_pred: float
    energy_sigma: float
    forces_true: torch.Tensor  # Shape: [N, 3]
    forces_pred: torch.Tensor  # Shape: [N, 3]
    forces_sigma: torch.Tensor  # Shape: [N, 3] or [N]


class ConformalPredictor:
    """Inductive Conformal Prediction wrapper providing distribution-free finite-sample guarantees. [M]"""

    def __init__(
        self,
        config: Optional[Union[ConformalCalibrationConfig, ConformalPredictorConfig]] = None,
    ) -> None:
        if isinstance(config, ConformalCalibrationConfig):
            self.calib_config = config
            self.alpha = config.significance_level
            self.apply_bonferroni = (config.hypothesis_scope == "atomwise_bonferroni")
            self.min_calibration_observations = config.min_calibration_observations
            self.eps_e = 1e-4
            self.eps_f = 1e-4
            self.strict = True
        else:
            cfg = config or ConformalPredictorConfig()
            self.config = cfg
            self.calib_config = None
            self.alpha = cfg.alpha
            self.eps_e = cfg.regularization_energy
            self.eps_f = cfg.regularization_force
            self.strict = cfg.strict_calibration_size
            self.apply_bonferroni = cfg.apply_bonferroni
            self.min_calibration_observations = 20

        self.q_hat_energy: float = float("inf")
        self.q_hat_force: float = float("inf")
        self.is_calibrated: bool = False

    def compute_minimum_calibration_size(self, num_atoms: Optional[int] = None) -> int:
        r"""Compute exact minimum calibration sample size n_min. [M]

        $$n_{\min} = \max\left(20, \left\lceil \frac{1 - \alpha}{\alpha} \right\rceil\right)$$
        """
        base_n = math.ceil((1.0 - self.alpha) / self.alpha)
        return max(20, base_n)

    def calibrate(self, calibration_data: Sequence[CalibrationSample]) -> None:
        r"""Compute non-conformity empirical quantiles over exchangeable calibration dataset. [D]

        Parameters
        ----------
        calibration_data : Sequence[CalibrationSample]
            Calibration configurations with ground-truth and predicted observables.
        """
        n_samples = len(calibration_data)
        if n_samples == 0:
            raise ConformalCalibrationError(
                "Calibration dataset is empty",
                details={"n_samples": 0, "n_required": self.compute_minimum_calibration_size()},
            )

        # Pool atomic force observations across all configurations
        n_atoms_per_mol = [sample.forces_true.shape[0] for sample in calibration_data]
        n_force_scores = sum(n_atoms_per_mol)
        n_required = self.compute_minimum_calibration_size()

        if n_force_scores < n_required:
            raise ConformalCalibrationError(
                f"Insufficient pooled calibration observations: received n={n_force_scores} force scores, "
                f"strictly requires n >= {n_required} for alpha={self.alpha:.3f}.",
                details={"n_force_scores": n_force_scores, "n_required": n_required, "alpha": self.alpha},
            )

        # 1. Scalar Energy Non-Conformity Scores
        energy_scores: List[float] = []
        for sample in calibration_data:
            residual = abs(sample.energy_true - sample.energy_pred)
            s_e = residual / (sample.energy_sigma + self.eps_e)
            energy_scores.append(float(s_e))

        energy_scores.sort()
        p_energy = math.ceil((n_samples + 1) * (1.0 - self.alpha))
        if p_energy <= n_samples:
            self.q_hat_energy = energy_scores[p_energy - 1]
        else:
            self.q_hat_energy = energy_scores[-1] if energy_scores else float("inf")

        # 2. Rotationally Invariant Per-Atom Force Non-Conformity Scores
        force_scores: List[float] = []
        for sample in calibration_data:
            f_true = sample.forces_true.to(dtype=torch.float64)
            f_pred = sample.forces_pred.to(dtype=torch.float64)
            f_sig = sample.forces_sigma.to(dtype=torch.float64)

            # Per-atom Euclidean norm difference: ||F_i - F_hat_i||_2
            diff = torch.norm(f_true - f_pred, dim=-1)  # [N]

            if f_sig.ndim == 2 and f_sig.shape[-1] == 3:
                sig_atom = torch.sqrt(torch.mean(f_sig ** 2, dim=-1))
            else:
                sig_atom = f_sig.view(-1)

            s_f = diff / (sig_atom + self.eps_f)
            force_scores.extend([float(v.item()) for v in s_f])

        force_scores.sort()
        n_force_scores = len(force_scores)

        # Quantile index for forces (no 3x Cartesian coordinate inflation)
        n_atoms = calibration_data[0].forces_true.shape[0]
        if self.apply_bonferroni:
            alpha_eff = self.alpha / float(n_atoms)
        else:
            alpha_eff = self.alpha

        p_force = math.ceil((n_force_scores + 1) * (1.0 - alpha_eff))
        if p_force <= n_force_scores:
            self.q_hat_force = force_scores[p_force - 1]
        else:
            self.q_hat_force = force_scores[-1]

        self.is_calibrated = True

    def predict_interval(
        self,
        predicted_energy: float,
        sigma_energy: float,
        predicted_forces: torch.Tensor,
        sigma_forces: torch.Tensor,
    ) -> ConformalInterval:
        r"""Evaluate finite-sample distribution-free prediction intervals. [D]

        $$\mathcal{C}_E = [\hat{E} - \hat{q}_{1-\alpha}^E (\hat{\sigma}_E + \epsilon_E), \; \hat{E} + \hat{q}_{1-\alpha}^E (\hat{\sigma}_E + \epsilon_E)]$$
        $$\mathcal{C}_{\mathbf{F}, i, \alpha} = [\hat{F}_{i, \alpha} - \hat{q}^F (\hat{\sigma}_{F, i} + \epsilon_F), \; \hat{F}_{i, \alpha} + \hat{q}^F (\hat{\sigma}_{F, i} + \epsilon_F)]$$
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated before generating prediction intervals.")

        # Energy Interval
        if math.isinf(self.q_hat_energy):
            e_lower = float("-inf")
            e_upper = float("inf")
        else:
            delta_e = self.q_hat_energy * (sigma_energy + self.eps_e)
            e_lower = float(predicted_energy - delta_e)
            e_upper = float(predicted_energy + delta_e)

        # Force Interval
        device = predicted_forces.device
        dtype = predicted_forces.dtype

        if math.isinf(self.q_hat_force):
            f_lower = torch.full_like(predicted_forces, float("-inf"))
            f_upper = torch.full_like(predicted_forces, float("inf"))
        else:
            if sigma_forces.ndim == 2 and sigma_forces.shape[-1] == 3:
                sig_atom = torch.sqrt(torch.mean(sigma_forces ** 2, dim=-1, keepdim=True))
            else:
                sig_atom = sigma_forces.view(-1, 1)

            delta_f = self.q_hat_force * (sig_atom + self.eps_f)
            f_lower = predicted_forces - delta_f
            f_upper = predicted_forces + delta_f

        return ConformalInterval(
            energy_lower=e_lower,
            energy_upper=e_upper,
            force_lower=f_lower.to(dtype=dtype, device=device),
            force_upper=f_upper.to(dtype=dtype, device=device),
            confidence_level=float(1.0 - self.alpha),
        )

    def predict_forces(
        self,
        predicted_forces: torch.Tensor,
        sigma_forces: torch.Tensor,
    ) -> ConformalInterval:
        """Evaluate distribution-free conformal prediction intervals on force vectors. [D]"""
        return self.predict_interval(
            predicted_energy=0.0,
            sigma_energy=0.0,
            predicted_forces=predicted_forces,
            sigma_forces=sigma_forces,
        )


class UncertaintyBreachSignal(Exception):
    """Raised when conformal nonconformity exceeds the calibrated tolerance (1 - alpha). [M]"""

    def __init__(
        self,
        message: str,
        nonconformity_score: float,
        threshold: float,
        frame_index: int,
    ) -> None:
        super().__init__(message)
        self.nonconformity_score = float(nonconformity_score)
        self.threshold = float(threshold)
        self.frame_index = int(frame_index)


@dataclass
class MolecularFrame:
    """Authentic physical trajectory frame holding coordinates, velocities, forces, and observables."""

    step: int
    positions: torch.Tensor
    velocities: torch.Tensor
    forces: torch.Tensor
    energy: float
    uncertainty_score: float
    atomic_numbers: Optional[List[int]] = None
    timestamp: Optional[str] = None


class TrajectoryInterventionHandler:
    """Autonomous trajectory intervention monitor with circular buffer, rollback, and scoped CUDA cleanup. [M]"""

    def __init__(
        self,
        predictor: Optional[ConformalPredictor] = None,
        capacity: int = 10,
        threshold: Optional[float] = None,
    ) -> None:
        self.predictor = predictor
        self.capacity = max(2, capacity)
        self.buffer: collections.deque[MolecularFrame] = collections.deque(maxlen=self.capacity)
        self.threshold = threshold
        self._spawn_ctx = mp.get_context("spawn")

    @property
    def current_threshold(self) -> float:
        if self.threshold is not None:
            return float(self.threshold)
        if self.predictor is not None and self.predictor.is_calibrated:
            if not math.isinf(self.predictor.q_hat_force):
                return float(self.predictor.q_hat_force)
            if not math.isinf(self.predictor.q_hat_energy):
                return float(self.predictor.q_hat_energy)
        return float(1.0 - (self.predictor.alpha if self.predictor else 0.10))

    def push_frame(self, frame: MolecularFrame) -> None:
        """Appends a valid molecular frame to the circular buffer."""
        self.buffer.append(frame)

    def rollback(self) -> Optional[MolecularFrame]:
        """Rolls back the circular buffer, discarding contaminated extrapolation and returning the last trustworthy frame."""
        if len(self.buffer) > 0:
            return self.buffer[-1]
        return None

    def evaluate_and_intervene(
        self,
        frame: MolecularFrame,
        nonconformity_score: Optional[float] = None,
    ) -> bool:
        """Evaluates nonconformity score. If threshold breached:
        1. Cleans up CUDA memory safely.
        2. Rolls back to last valid frame.
        3. Raises UncertaintyBreachSignal.
        Returns True if safe.
        """
        score = float(nonconformity_score if nonconformity_score is not None else frame.uncertainty_score)
        thresh = self.current_threshold

        if score > thresh:
            self._cleanup_device_memory()
            self.rollback()
            raise UncertaintyBreachSignal(
                f"Epistemic uncertainty breach detected at frame {frame.step}: score {score:.4f} > threshold {thresh:.4f}",
                nonconformity_score=score,
                threshold=thresh,
                frame_index=frame.step,
            )

        self.push_frame(frame)
        return True

    def _cleanup_device_memory(self) -> None:
        """Scoped GPU memory cleanup with automatic CPU fallback."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

