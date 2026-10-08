"""CoChem-TORQ: Unit Tests for Multi-Fidelity Spline Router & WKB Tunneling Estimator.

================================================================================
Phase 4 (Stage 3.0 / 6.0) Validation Suite
------------------------------------------
Validates 1D and 2D continuous SciPy spline interpolation, stationary point
extraction (minima, saddle points, maxima), barrier height quantification,
and semiclassical Wentzel-Kramers-Brillouin (WKB) tunneling probabilities
and splitting estimators for light rotors (-CH3, -OH, -NH2, -SH) vs heavy rotors.

Adheres strictly to Anti-Spoofing Protocol v2 and Zero-Mock Physical Execution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Libraries.cochem_torq_slicer import (
    HARTREE_TO_CM1,
    HARTREE_TO_KCAL_MOL,
    StationaryPoint,
    StationaryPoint2D,
    WKBTunnelingResult,
    evaluate_wkb_action_integral,
    fit_continuous_2d_splines,
    fit_continuous_splines,
    get_dynamic_reduced_inertia,
    wkb_tunneling_estimator,
)


class TestSplineFitting1D:
    """Test suite for 1D continuous spline interpolation & node extraction."""

    def test_fit_continuous_splines_threefold_potential(self) -> None:
        """Verifies 1D periodic cubic spline fitting on a 3-fold torsional PES."""
        v0_hartree = 0.005  # ~3.14 kcal/mol (standard methyl barrier)
        angles = np.linspace(0.0, 360.0, 24, endpoint=False)
        energies = [
            0.5 * v0_hartree * (1.0 - math.cos(math.radians(3.0 * a))) - 150.0
            for a in angles
        ]

        res = fit_continuous_splines(angles, energies, periodic=True)

        assert "stationary_points" in res
        assert "global_minimum" in res
        assert len(res["minima"]) >= 3
        assert len(res["maxima"]) >= 3

        expected_barrier_kcal = v0_hartree * HARTREE_TO_KCAL_MOL
        assert res["max_barrier_kcal_mol"] == pytest.approx(
            expected_barrier_kcal, rel=0.05
        )
        assert res["max_barrier_cm1"] == pytest.approx(
            v0_hartree * HARTREE_TO_CM1, rel=0.05
        )

        # Validate Pydantic model serialization of stationary points
        for pt in res["stationary_points"]:
            validated_pt = StationaryPoint(**pt)
            assert validated_pt.angle_deg >= 0.0
            assert validated_pt.type in ("MINIMUM", "MAXIMUM", "INFLECTION")

    def test_fit_continuous_splines_non_periodic(self) -> None:
        """Verifies non-periodic 1D cubic spline fitting across a restricted scan."""
        angles = np.linspace(30.0, 150.0, 10)
        energies = [0.002 * (math.radians(a) - math.pi / 2.0) ** 2 for a in angles]

        res = fit_continuous_splines(angles, energies, periodic=False)
        assert len(res["minima"]) >= 1
        assert res["global_minimum"]["angle_deg"] == pytest.approx(90.0, abs=5.0)

    def test_fit_continuous_splines_closed_endpoints(self) -> None:
        """Verifies periodic spline fitting when input data explicitly spans [0, 360]."""
        angles = np.linspace(0.0, 360.0, 13)
        energies = [0.005 * (1.0 - math.cos(math.radians(2.0 * a))) for a in angles]
        # Simulate slight floating point discrepancy at end
        energies[-1] += 1e-12

        with pytest.raises(ValueError, match="explicit finite closure tolerance"):
            fit_continuous_splines(angles, energies, periodic=True)
        res = fit_continuous_splines(
            angles, energies, periodic=True, periodic_endpoint_tolerance_hartree=2e-12
        )
        assert res["input_energies_hartree"][-1] == energies[-1]
        assert (
            res["endpoint_closure"]["input_difference_hartree"]
            == energies[-1] - energies[0]
        )
        assert len(res["minima"]) >= 2
        assert len(res["maxima"]) >= 2

    def test_fit_continuous_splines_insufficient_points(self) -> None:
        """Ensures ValueError is raised when fewer than 4 points are supplied."""
        with pytest.raises(ValueError, match="At least 4 points required"):
            fit_continuous_splines([0.0, 60.0, 120.0], [-10.0, -9.9, -9.8])

    def test_fit_continuous_splines_mismatched_lengths(self) -> None:
        """Ensures ValueError is raised when angles and energies lengths diverge."""
        with pytest.raises(ValueError, match="must equal energies count"):
            fit_continuous_splines([0.0, 60.0, 120.0, 180.0], [-10.0, -9.9])


class TestSplineFitting2D:
    """Test suite for 2D coupled rotor potential surface fitting & saddles."""

    def test_fit_continuous_2d_splines_coupled_rotors(self) -> None:
        """Verifies 2D bivariate spline fitting on a 2-rotor coupled potential."""
        deg1 = np.linspace(0.0, 360.0, 16, endpoint=False)
        deg2 = np.linspace(0.0, 360.0, 16, endpoint=False)
        r1, r2 = np.meshgrid(np.radians(deg1), np.radians(deg2), indexing="ij")

        # 2D coupled potential
        v1 = 0.003
        v2 = 0.002
        v12 = 0.0005
        pes_2d = (
            0.5 * v1 * (1.0 - np.cos(3.0 * r1))
            + 0.5 * v2 * (1.0 - np.cos(2.0 * r2))
            + 0.5 * v12 * np.cos(3.0 * r1 - 2.0 * r2)
        )

        res = fit_continuous_2d_splines(deg1, deg2, pes_2d)

        assert "spline" in res
        assert "minima" in res
        assert "saddles" in res
        assert "maxima" in res
        assert res["max_barrier_kcal_mol"] > 0.0
        assert res["max_barrier_cm1"] > 0.0

        for pt in res["stationary_points"]:
            validated_pt = StationaryPoint2D(**pt)
            assert validated_pt.type in ("MINIMUM", "SADDLE", "MAXIMUM", "INFLECTION")
            assert len(validated_pt.hessian_eigenvalues) == 2

    def test_fit_continuous_2d_splines_unsorted_grid(self) -> None:
        """Verifies 2D spline interpolation handles permuted/unsorted grid coordinates."""
        deg1 = np.array([180.0, 0.0, 270.0, 90.0])
        deg2 = np.array([90.0, 270.0, 0.0, 180.0])
        # Function V = cos(theta1) + sin(theta2)
        r1, r2 = np.meshgrid(np.radians(deg1), np.radians(deg2), indexing="ij")
        pes = np.cos(r1) + np.sin(r2)

        res = fit_continuous_2d_splines(deg1, deg2, pes)
        assert "spline" in res
        assert "global_minimum" in res


class TestWKBTunnelingEstimator:
    """Test suite for Wentzel-Kramers-Brillouin (WKB) quantum tunneling estimation."""

    def test_wkb_tunneling_estimator_ch3(self) -> None:
        """Evaluate a declared cosine potential; this is an uncalibrated WKB estimate."""
        res = wkb_tunneling_estimator(
            rotor_type="-CH3",
            barrier_height_cm1=1000.0,
            reduced_moment_inertia_amu_ang2=3.1,
            periodicity=3,
            potential_model="cosine",
        )
        assert res["is_light_rotor"] is True
        assert res["tunneling_probability"] > 0.0
        assert res["tunneling_splitting_mhz"] >= 0.0
        assert res["quantum_treatment_required"] is True

        validated = WKBTunnelingResult(**res)
        assert validated.is_light_rotor is True

    def test_wkb_tunneling_estimator_hydroxyl(self) -> None:
        """Verifies light hydroxyl (-OH) rotor tunneling estimation."""
        res = wkb_tunneling_estimator(
            rotor_type="OH",
            barrier_height_cm1=800.0,
            reduced_moment_inertia_amu_ang2=0.95,
            periodicity=1,
            potential_model="cosine",
        )
        assert res["is_light_rotor"] is True
        assert res["tunneling_probability"] > 0.0
        assert res["quantum_treatment_required"] is True

    def test_wkb_tunneling_estimator_heavy_rotor(self) -> None:
        """A small WKB estimate still requires independent quantum validation."""
        res = wkb_tunneling_estimator(
            rotor_type="Phenyl",
            barrier_height_cm1=5000.0,
            reduced_moment_inertia_amu_ang2=120.0,
            periodicity=2,
            potential_model="cosine",
        )
        assert res["is_light_rotor"] is False
        assert res["quantum_treatment_required"] is True
        assert "uncalibrated" in res["quality_flags"]
        assert res["tunneling_splitting_mhz"] < 1e-10

    @pytest.mark.parametrize("name", ["CH3", "CD3", "-OH", "OD", "NH2", "-SH"])
    def test_name_cannot_define_reduced_inertia(self, name: str) -> None:
        """An axis, geometry, isotope masses and rotor/frame model are indispensable."""
        with pytest.raises(ValueError, match="provide geometry, isotope masses"):
            get_dynamic_reduced_inertia(name)

    def test_wkb_requires_declared_potential_shape(self) -> None:
        with pytest.raises(ValueError, match="explicitly selected cosine"):
            wkb_tunneling_estimator("CH3", 1000.0, 3.1, 3)

    def test_evaluate_wkb_action_integral(self) -> None:
        """Verifies numerical general WKB phase integral across turning points."""
        barrier_joules = 1000.0 / 5.034116567e22
        energy_joules = 0.2 * barrier_joules
        i_red_kg_m2 = 3.1 * 1.66053906892e-27 * (1.0e-10**2)

        def potential(theta: float) -> float:
            return 0.5 * barrier_joules * (1.0 - math.cos(3.0 * theta))

        # Turning points where V(theta) = E
        theta_t1 = math.acos(1.0 - 2.0 * energy_joules / barrier_joules) / 3.0
        theta_t2 = (2.0 * math.pi / 3.0) - theta_t1

        action = evaluate_wkb_action_integral(
            potential_func=potential,
            energy_joules=energy_joules,
            theta_turning_1_rad=theta_t1,
            theta_turning_2_rad=theta_t2,
            reduced_moment_inertia_kg_m2=i_red_kg_m2,
        )
        assert action > 0.0
        assert np.isfinite(action)


def test_monotonic_scan_does_not_fabricate_stationary_curvature():
    angles = np.linspace(0.0, 90.0, 8)
    energies = np.radians(angles)
    result = fit_continuous_splines(angles, energies, periodic=False)
    assert result["stationary_points"] == []
    assert result["global_minimum"] is None
    assert result["sampled_minimum"]["type"] == "SAMPLED_POINT"
    assert "curvature" not in result["sampled_minimum"]
    assert result["sampled_minimum"]["energy_hartree"] == energies.min()
    assert result["max_barrier_kcal_mol"] is None
    assert result["sampled_energy_span_hartree"] == pytest.approx(np.ptp(energies))


def test_monotonic_2d_surface_does_not_fabricate_hessian_eigenvalues():
    angles = np.linspace(0.0, 90.0, 8)
    x, y = np.meshgrid(np.radians(angles), np.radians(angles), indexing="ij")
    energies = x + y
    result = fit_continuous_2d_splines(angles, angles, energies)
    assert result["stationary_points"] == []
    assert result["global_minimum"] is None
    assert result["sampled_minimum"]["type"] == "SAMPLED_POINT"
    assert "hessian_eigenvalues" not in result["sampled_minimum"]
    assert result["max_barrier_kcal_mol"] is None
    assert result["sampled_energy_span_hartree"] == pytest.approx(np.ptp(energies))


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j])
def test_spline_never_truncates_complex_or_nonfinite_energies(dimensions, bad):
    angles = np.linspace(0.0, 90.0, 8)
    shape = (8,) if dimensions == 1 else (8, 8)
    energies = np.zeros(shape, dtype=complex if isinstance(bad, complex) else float)
    energies.flat[0] = bad
    with pytest.raises(ValueError, match="real, not complex|finite"):
        if dimensions == 1:
            fit_continuous_splines(angles, energies, periodic=False)
        else:
            fit_continuous_2d_splines(angles, angles, energies)
