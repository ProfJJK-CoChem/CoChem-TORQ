"""Explicit D3(BJ) adapter to the actual simple-dftd3 library.

An atomic-number power law and geometric C6/C8 mixing do not implement Grimme
D3. This adapter delegates reference coordination, interpolation, damping and
three-body treatment to DFTD3. D4 is a different model and is never named here.
The caller must select the functional explicitly; old guessed-parameter configs
are rejected. Actual library parameters and their source-table digest are retained.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.constants import physical_constants
from torch.autograd.function import once_differentiable

from Libraries.cochem_torq_inference_schemas import DispersionD3Config

HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
BOHR_TO_ANGSTROM = physical_constants["Bohr radius"][0] * 1e10


class DispersionUnavailableError(RuntimeError):
    """The requested dispersion recipe or actual library is unavailable."""


def _geometry(coordinates: torch.Tensor, atomic_numbers: Sequence[int]):
    numbers = tuple(atomic_numbers)
    if not numbers or any(
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not 1 <= int(value) <= 118
        for value in numbers
    ):
        raise ValueError("Supply actual atomic numbers 1..118 for a nonempty molecule.")
    if not coordinates.is_floating_point() or coordinates.shape != (len(numbers), 3):
        raise ValueError(
            "Coordinates must be a floating-point [N,3] tensor in angstrom."
        )
    if not torch.isfinite(coordinates).all():
        raise ValueError("Nonfinite coordinates cannot be evaluated.")
    positions = coordinates.detach().cpu().double().numpy()
    if len(numbers) > 1:
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < 1e-8:
            raise ValueError(
                "Coincident nuclei cannot be replaced by invented coordinates."
            )
    return np.asarray(numbers, dtype=np.int32), positions


def compute_coordination_numbers(
    coordinates: torch.Tensor, atomic_numbers: Sequence[int], k1: float = 16.0
) -> torch.Tensor:
    """Geometric radius-based descriptor; not the actual DFTD3 coordination model.

    This public historical helper implements its stated logistic sum with actual
    tabulated Pyykko radii. Missing radii fail explicitly. DispersionD3Layer does
    not use this descriptor to replace the reference library's coordination model.
    """
    from mendeleev import element

    numbers, _ = _geometry(coordinates, atomic_numbers)
    if not np.isfinite(k1) or k1 <= 0:
        raise ValueError("Descriptor steepness must be finite and positive.")
    radii = []
    for number in numbers:
        radius = element(int(number)).covalent_radius_pyykko
        if radius is None or not np.isfinite(radius) or radius <= 0:
            raise DispersionUnavailableError(
                f"No tabulated Pyykko radius for Z={number}."
            )
        radii.append(float(radius) / 100)
    if len(numbers) == 1:
        return torch.zeros(1, dtype=coordinates.dtype, device=coordinates.device)
    radius_tensor = torch.tensor(
        radii, dtype=coordinates.dtype, device=coordinates.device
    )
    distances = torch.linalg.vector_norm(
        coordinates[:, None] - coordinates[None, :], dim=-1
    )
    diagonal = torch.eye(len(numbers), dtype=torch.bool, device=coordinates.device)
    safe_distances = torch.where(diagonal, torch.ones_like(distances), distances)
    contributions = torch.sigmoid(
        k1 * ((radius_tensor[:, None] + radius_tensor[None, :]) / safe_distances - 1)
    )
    return torch.where(diagonal, torch.zeros_like(contributions), contributions).sum(
        dim=1
    )


class _ReferenceD3Energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coordinates, calculator, numbers):
        energy, gradient = calculator._evaluate(coordinates, numbers)
        derivative = torch.tensor(
            gradient, dtype=coordinates.dtype, device=coordinates.device
        )
        ctx.save_for_backward(derivative)
        forces = -derivative
        ctx.mark_non_differentiable(forces)
        return torch.tensor(
            energy, dtype=coordinates.dtype, device=coordinates.device
        ), forces

    @staticmethod
    @once_differentiable
    def backward(ctx, energy_weight, force_weight):
        (derivative,) = ctx.saved_tensors
        # No unvalidated Hessian or force-loss derivative is inferred.
        return energy_weight * derivative, None, None


class DispersionD3Layer:
    """Actual functional-specific D3(BJ), optionally with reference ATM treatment.

    Coordinates are angstrom; returned energy is eV and force is eV/angstrom.
    First energy derivatives use actual DFTD3 gradients. Higher derivatives are
    deliberately not exposed by the Torch bridge. This is a dispersion correction,
    not a complete electronic energy or a calibrated spectroscopy method.
    """

    def __init__(
        self,
        config: DispersionD3Config | None = None,
        *,
        method: str | None = None,
        include_atm: bool = True,
    ) -> None:
        if config is not None:
            raise DispersionUnavailableError(
                "Legacy guessed atomic/damping configs are unqualified. Select an "
                "explicit reference functional with DispersionD3Layer(method=...)."
            )
        if not isinstance(method, str) or not method.strip():
            raise DispersionUnavailableError(
                "D3(BJ) requires an explicitly selected reference functional."
            )
        if type(include_atm) is not bool:
            raise ValueError("include_atm must be explicitly boolean.")
        try:
            from dftd3 import parameters
            from dftd3.interface import RationalDampingParam

            resolved = parameters.get_damping_param(method.lower(), defaults=["bj"])
            self.param = RationalDampingParam(method=method, atm=include_atm)
            table = Path(parameters.get_data_file_name())
            library_version = version("dftd3")
        except (ImportError, KeyError, RuntimeError, ValueError) as exc:
            raise DispersionUnavailableError(
                f"Actual DFTD3 D3(BJ) parameters unavailable for {method!r}."
            ) from exc
        self.method = method
        self.include_atm = include_atm
        self.provenance = {
            "engine": "simple-dftd3",
            "version": library_version,
            "model": "D3(BJ)-ATM" if include_atm else "D3(BJ)",
            "functional": method,
            "library_parameters": resolved,
            "include_atm": include_atm,
            "parameter_table_sha256": sha256(table.read_bytes()).hexdigest(),
            "energy_unit": "eV",
            "force_unit": "eV/angstrom",
            "qualification": (
                "reference implementation; chemical accuracy separately uncalibrated"
            ),
        }

    def _evaluate(self, coordinates, atomic_numbers):
        from dftd3.interface import DispersionModel

        numbers, positions = _geometry(coordinates, atomic_numbers)
        model = DispersionModel(numbers, positions / BOHR_TO_ANGSTROM)
        observed = model.get_dispersion(self.param, grad=True)
        energy = float(observed["energy"]) * HARTREE_TO_EV
        gradient = np.asarray(observed["gradient"]) * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        if (
            not np.isfinite(energy)
            or gradient.shape != positions.shape
            or not np.isfinite(gradient).all()
        ):
            raise DispersionUnavailableError(
                "Actual DFTD3 returned invalid energy/gradient data."
            )
        return energy, gradient

    def compute_energy_and_forces(
        self, coordinates: torch.Tensor, atomic_numbers: Sequence[int]
    ):
        """Evaluate the actual reference library and preserve its first derivative."""
        _geometry(coordinates, atomic_numbers)
        return _ReferenceD3Energy.apply(coordinates, self, tuple(atomic_numbers))
