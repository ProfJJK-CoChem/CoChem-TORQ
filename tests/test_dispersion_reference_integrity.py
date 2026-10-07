"""Actual DFTD3 computations and independent derivative/sign/unit checks.

These qualify this adapter's numerical contract, not a complete electronic
method, a chemical accuracy target, D4, or a rotational-identification profile.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch
from dftd3 import parameters
from dftd3.interface import DispersionModel, RationalDampingParam

from Libraries.cochem_torq_dispersion_d3 import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    DispersionD3Layer,
    DispersionUnavailableError,
    compute_coordination_numbers,
)
from Libraries.cochem_torq_inference_schemas import DispersionD3Config

pytestmark = [pytest.mark.cpu_ml, pytest.mark.real_engine]


def water_input():
    # Starting coordinates are inputs; no optimized-structure claim is made.
    return torch.tensor(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )


@pytest.mark.parametrize("include_atm", [True, False])
def test_actual_library_energy_gradient_and_parameter_provenance(include_atm):
    positions = water_input()
    layer = DispersionD3Layer(method="b3lyp", include_atm=include_atm)
    energy, forces = layer.compute_energy_and_forces(positions, [8, 1, 1])
    direct = DispersionModel(
        np.array([8, 1, 1], dtype=np.int32),
        positions.detach().numpy() / BOHR_TO_ANGSTROM,
    ).get_dispersion(RationalDampingParam(method="b3lyp", atm=include_atm), grad=True)
    np.testing.assert_allclose(
        energy.detach().numpy(), direct["energy"] * HARTREE_TO_EV
    )
    np.testing.assert_allclose(
        forces.numpy(), -direct["gradient"] * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    )
    derivative = torch.autograd.grad(energy, positions)[0]
    torch.testing.assert_close(derivative, -forces)
    assert not forces.requires_grad
    assert layer.provenance["engine"] == "simple-dftd3"
    assert "D4" not in layer.provenance["model"]
    assert layer.provenance["library_parameters"] == parameters.get_damping_param(
        "b3lyp", defaults=["bj"]
    )
    assert (
        layer.provenance["parameter_table_sha256"]
        == sha256(Path(parameters.get_data_file_name()).read_bytes()).hexdigest()
    )


def test_actual_force_matches_independently_displaced_energy():
    positions = water_input().detach()
    layer = DispersionD3Layer(method="b3lyp")
    _, forces = layer.compute_energy_and_forces(positions, [8, 1, 1])
    step = 1e-4
    finite_difference = np.empty((3, 3))
    for atom in range(3):
        for component in range(3):
            plus, minus = positions.clone(), positions.clone()
            plus[atom, component] += step
            minus[atom, component] -= step
            e_plus = layer.compute_energy_and_forces(plus, [8, 1, 1])[0].item()
            e_minus = layer.compute_energy_and_forces(minus, [8, 1, 1])[0].item()
            finite_difference[atom, component] = (e_plus - e_minus) / (2 * step)
    np.testing.assert_allclose(
        finite_difference, -forces.numpy(), atol=2e-10, rtol=2e-5
    )
    np.testing.assert_allclose(forces.sum(dim=0).numpy(), np.zeros(3), atol=1e-15)


def test_higher_derivative_is_unavailable_instead_of_zero():
    positions = water_input()
    energy, _ = DispersionD3Layer(method="b3lyp").compute_energy_and_forces(
        positions, [8, 1, 1]
    )
    derivative = torch.autograd.grad(energy, positions, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(derivative.sum(), positions)


@pytest.mark.parametrize("method", [None, "", "unqualified-new-functional"])
def test_unknown_or_unspecified_recipe_does_not_guess_parameters(method):
    with pytest.raises(DispersionUnavailableError):
        DispersionD3Layer(method=method)


def test_legacy_guessed_config_cannot_activate_dispersion():
    with pytest.raises(DispersionUnavailableError, match="Legacy guessed"):
        DispersionD3Layer(DispersionD3Config(), method="b3lyp")


def test_invalid_geometry_cannot_be_substituted():
    layer = DispersionD3Layer(method="b3lyp")
    with pytest.raises(ValueError, match="Coincident"):
        layer.compute_energy_and_forces(torch.zeros((2, 3)), [1, 1])
    with pytest.raises(ValueError, match="actual atomic numbers"):
        layer.compute_energy_and_forces(water_input(), [8, 1, 119])
    invalid = water_input().detach()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        layer.compute_energy_and_forces(invalid, [8, 1, 1])


def test_geometric_descriptor_uses_named_real_radii():
    from mendeleev import element

    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=torch.float64)
    radius = float(element("H").covalent_radius_pyykko) / 100
    expected = torch.sigmoid(torch.tensor(16 * (2 * radius / 0.74 - 1)))
    torch.testing.assert_close(
        compute_coordination_numbers(coordinates, [1, 1]),
        torch.full((2,), expected.item(), dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
