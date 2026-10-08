"""Independent SI/dimensional identities and strict physical-coordinate contracts.

Inputs here are exact mathematical examples and published constants, never
substitute engine observations or experimental-accuracy evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import numpy as np
import pytest
import scipy.constants as reference

from cochem_torq import units
from cochem_torq.ecosystem import AtomIdentity, MoleculeHandoff
from cochem_torq.spectroscopy import harmonic, rotational


def test_si_defining_constants_and_codata_source_are_explicit() -> None:
    expected = {
        "Planck constant": 6.62607015e-34,
        "elementary charge": 1.602176634e-19,
        "Boltzmann constant": 1.380649e-23,
        "Avogadro constant": 6.02214076e23,
        "speed of light in vacuum": 299792458.0,
    }
    for name, value in expected.items():
        record = units.CONSTANT_DEFINITIONS[name]
        assert record.value == value
        assert record.standard_uncertainty == 0.0
        assert record.source == "https://physics.nist.gov/cuu/Constants/"
    provenance = units.constants_provenance()
    assert provenance["codata_release"] == "2022"
    assert provenance["profile"] == units.CONSTANTS_PROFILE
    assert provenance["scipy_version"]
    assert len(provenance["definition_sha256"]) == 64
    assert units.CONSTANT_DEFINITIONS["Bohr radius"].standard_uncertainty > 0


def test_published_definitions_cannot_be_mutated_through_records() -> None:
    with pytest.raises(TypeError):
        units.CONSTANT_DEFINITIONS["Bohr radius"] = units.CONSTANT_DEFINITIONS[
            "Hartree energy"
        ]
    with pytest.raises(FrozenInstanceError):
        units.CONSTANT_DEFINITIONS["Bohr radius"].value = 1.0
    with pytest.raises(TypeError):
        units.UNITS["bohr"] = units.UNITS["angstrom"]
    with pytest.raises(FrozenInstanceError):
        units.UNITS["bohr"].scale_si = 1.0
    provenance = units.constants_provenance()
    digest = provenance["definition_sha256"]
    provenance["constants"]["Bohr radius"]["value"] = -1.0
    assert units.constants_provenance()["definition_sha256"] == digest
    assert units.BOHR_METRE > 0


def test_migrated_constants_retain_original_numeric_values_and_operation_order() -> (
    None
):
    bohr = reference.physical_constants["Bohr radius"][0]
    hartree = reference.physical_constants["Hartree energy"][0]
    assert units.BOHR_METRE == harmonic.BOHR_METRE == bohr
    assert units.HARTREE_JOULE == harmonic.HARTREE_JOULE == hartree
    assert (
        units.ATOMIC_MASS_ELECTRON
        == harmonic.ATOMIC_MASS_ELECTRON
        == (reference.atomic_mass / reference.physical_constants["electron mass"][0])
    )
    assert (
        units.HARTREE_CM1
        == harmonic.HARTREE_CM1
        == hartree / (reference.h * reference.c * 100)
    )
    assert units.DEBYE_COULOMB_METRE == rotational.DEBYE_COULOMB_METRE == 3.33564e-30
    assert units.ANGSTROM_BOHR == 1.0 / (bohr * 1e10)
    assert units.BOHR_ANGSTROM == bohr * 1e10


def test_molecular_interchange_conversion_matches_previous_bytes() -> None:
    molecule = MoleculeHandoff(
        molecule_id="mathematical-row-mapping",
        atoms=(
            AtomIdentity(atom_id="h-1", symbol="H"),
            AtomIdentity(atom_id="h-2", symbol="H"),
        ),
        geometry=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)),
        geometry_unit="angstrom",
        charge=0,
        multiplicity=1,
    )
    old_factor = 1.0 / (reference.physical_constants["Bohr radius"][0] * 1e10)
    previous = np.asarray(molecule.geometry) * old_factor
    assert np.array_equal(previous, molecule.to_application_molecule()["geometry_bohr"])


@pytest.mark.parametrize(
    "source,target",
    [
        ("bohr", "angstrom"),
        ("hartree", "eV"),
        ("MHz", "Hz"),
        ("debye", "C*m"),
        ("u", "kg"),
        ("degree", "radian"),
        ("hartree/bohr", "joule/m"),
        ("hartree/bohr^2", "joule/m^2"),
        ("bohr*sqrt(electron_mass)", "m*sqrt(kg)"),
    ],
)
def test_same_dimension_roundtrip_preserves_signed_mathematical_values(
    source: str, target: str
) -> None:
    values = np.array([-3.5, 0.0, 2.0], dtype=np.float64)
    changed = units.convert(values, source, target)
    recovered = units.convert(changed, target, source)
    np.testing.assert_allclose(recovered, values, rtol=3e-16, atol=0)
    np.testing.assert_array_equal(values, [-3.5, 0.0, 2.0])
    assert changed.dtype == np.float64


def test_frequency_and_wavenumber_identity_uses_cycles_not_angular_frequency() -> None:
    assert units.convert(1.0, "MHz", "Hz") == 1e6
    assert (
        units.convert(1.0, "cm^-1", "Hz", equivalence="spectroscopic") == 29979245800.0
    )
    assert units.convert(
        1.0, "Hz", "rad/s", equivalence="spectroscopic"
    ) == pytest.approx(2 * np.pi)
    assert units.convert(
        1.0, "hartree", "cm^-1", equivalence="spectroscopic"
    ) == pytest.approx(units.HARTREE_CM1, rel=2e-16)
    assert units.convert(
        1.0, "angular_frequency_au", "cm^-1", equivalence="spectroscopic"
    ) == pytest.approx(units.HARTREE_CM1, rel=2e-16)


def test_per_particle_molar_energy_conversion_is_explicit_and_independent() -> None:
    with localcontext() as context:
        context.prec = 50
        expected_kj = float(
            Decimal("4.359744722206e-18") * Decimal("6.02214076e23") / Decimal(1000)
        )
        expected_kcal = float(
            Decimal("4.359744722206e-18") * Decimal("6.02214076e23") / Decimal(4184)
        )
    assert units.convert(
        1.0, "hartree", "kJ/mol", equivalence="molar"
    ) == pytest.approx(expected_kj, rel=2e-16)
    assert units.convert(
        1.0, "hartree", "kcal/mol", equivalence="molar"
    ) == pytest.approx(expected_kcal, rel=2e-16)
    assert units.convert(
        expected_kj, "kJ/mol", "hartree", equivalence="molar"
    ) == pytest.approx(1.0, rel=2e-16)


def test_rounded_debye_convention_remains_named_and_does_not_upgrade_silently() -> None:
    assert units.convert(1.0, "debye", "C*m") == 3.33564e-30
    assert units.convert(1.0, "debye_exact", "C*m") == 1e-21 / 299792458.0
    assert units.DEBYE_EXACT_COULOMB_METRE != units.DEBYE_COULOMB_METRE
    assert "rounded" in units.constants_provenance()["debye_convention"]
    assert (
        units.convert(1.0, "atomic_dipole", "C*m")
        == units.ELEMENTARY_CHARGE_COULOMB * units.BOHR_METRE
    )


@pytest.mark.parametrize(
    "source,target,equivalence",
    [
        ("hartree", "MHz", None),
        ("hartree", "kJ/mol", None),
        ("radian", "dimensionless", None),
        ("bohr", "bohr*sqrt(electron_mass)", None),
        ("hartree/bohr", "hartree/bohr^2", None),
        ("Hz", "rad/s", None),
        ("bohr", "hartree", "spectroscopic"),
        ("MHz", "kJ/mol", "molar"),
    ],
)
def test_cross_dimension_or_implicit_equivalence_is_rejected(
    source: str, target: str, equivalence: str | None
) -> None:
    with pytest.raises(units.UnitError):
        units.convert(1.0, source, target, equivalence=equivalence)


@pytest.mark.parametrize(
    "value",
    [
        True,
        [True, 1.0],
        [[1.0, False]],
        "1.0",
        ["1", 2],
        complex(1.0, 0.0),
        [1.0, complex(0.0, 1e-30)],
        np.array([1.0], dtype=object),
        float("nan"),
        float("inf"),
        -float("inf"),
        [[1.0], [1.0, 2.0]],
        None,
    ],
)
def test_type_complex_nonfinite_and_ragged_values_fail_without_coercion(value) -> None:
    with pytest.raises(units.UnitError):
        units.convert(value, "bohr", "angstrom")


def test_unsupported_units_equivalence_and_overflow_fail_explicitly() -> None:
    with pytest.raises(units.UnitError):
        units.convert(1.0, "guessed", "angstrom")
    with pytest.raises(units.UnitError):
        units.convert(1.0, "bohr", "angstrom", equivalence=["molar"])
    with pytest.raises(units.UnitError, match="finite float64"):
        units.convert(1e308, "hartree", "kJ/mol", equivalence="molar")


@pytest.fixture
def explicit_frame() -> units.CartesianFrame:
    return units.CartesianFrame(
        "mathematical-frame",
        "reference-frame",
        [0.0, 0.0, 0.0],
        "bohr",
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )


def test_cartesian_frame_arrays_are_genuinely_immutable_and_right_handed(
    explicit_frame,
) -> None:
    assert np.linalg.det(explicit_frame.axes_columns) == 1.0
    with pytest.raises(ValueError):
        explicit_frame.origin.setflags(write=True)
    with pytest.raises(ValueError):
        explicit_frame.axes_columns[0, 0] = 1.0
    record = explicit_frame.to_record()
    assert record["origin_unit"] == "bohr"
    assert "columns" in record["convention"]


@pytest.mark.parametrize(
    "axes", [np.diag([1.0, 1.0, -1.0]), np.diag([2.0, 1.0, 1.0]), np.ones((2, 2))]
)
def test_unphysical_or_wrong_shape_cartesian_axes_are_rejected(axes) -> None:
    with pytest.raises(units.UnitError):
        units.CartesianFrame("frame", "reference", [0.0, 0.0, 0.0], "bohr", axes)


def test_gradient_force_sign_is_explicit_and_unit_conversion_never_flips_it(
    explicit_frame,
) -> None:
    gradient = units.PhysicalQuantity(
        "gradient", "hartree/bohr", [[0.0, -1.0, 2.0]], explicit_frame, ("atom-one",)
    )
    converted = gradient.in_unit("joule/m")
    assert converted.kind == "gradient" and converted.values[0, 1] < 0
    force = units.gradient_to_force(gradient)
    assert force.kind == "force"
    np.testing.assert_array_equal(force.values, -gradient.values)
    assert force.frame == gradient.frame
    with pytest.raises(units.UnitError):
        units.gradient_to_force(force)
    with pytest.raises(ValueError):
        converted.values.setflags(write=True)


@pytest.mark.parametrize(
    "kind,unit,values,ids",
    [
        ("cartesian_coordinates", "bohr", [[0.0, 0.0, 0.0]], ("atom-one",)),
        ("gradient", "hartree/bohr", [[1.0, 0.0, 0.0]], ("atom-one",)),
        ("hessian", "hartree/bohr^2", np.eye(3), ("atom-one",)),
        ("dipole", "debye", [1.0, 0.0, 0.0], None),
    ],
)
def test_cartesian_quantity_cannot_hide_missing_frame(kind, unit, values, ids) -> None:
    with pytest.raises(units.UnitError, match="explicit validated frame"):
        units.PhysicalQuantity(kind, unit, values, atom_ids=ids)


def test_atom_mapping_and_quantity_dimensions_are_bound_to_actual_arrays(
    explicit_frame,
) -> None:
    with pytest.raises(units.UnitError, match="row IDs"):
        units.PhysicalQuantity(
            "gradient", "hartree/bohr", [[1.0, 0.0, 0.0]], explicit_frame
        )
    with pytest.raises(units.UnitError, match="atom mapping"):
        units.PhysicalQuantity(
            "hessian", "hartree/bohr^2", np.eye(2), explicit_frame, ("atom-one",)
        )
    with pytest.raises(units.UnitError, match="dimensions disagree"):
        units.PhysicalQuantity(
            "gradient",
            "hartree/bohr^2",
            [[1.0, 0.0, 0.0]],
            explicit_frame,
            ("atom-one",),
        )
    with pytest.raises(units.UnitError, match="non-Cartesian"):
        units.PhysicalQuantity("energy", "hartree", -1.0, explicit_frame)


def test_normal_coordinates_keep_mass_weighting_and_dimensionless_basis_distinct() -> (
    None
):
    basis_digest = hashlib.sha256(np.eye(2).tobytes()).hexdigest()
    with pytest.raises(units.UnitError, match="mode-basis"):
        units.PhysicalQuantity(
            "mass_weighted_normal_coordinate", "bohr*sqrt(electron_mass)", [1.0]
        )
    weighted = units.PhysicalQuantity(
        "mass_weighted_normal_coordinate",
        "bohr*sqrt(electron_mass)",
        [1.0, -1.0],
        normal_basis_sha256=basis_digest,
    )
    assert weighted.to_record()["normal_basis_sha256"] == basis_digest
    with pytest.raises(units.UnitError, match="dimensions"):
        weighted.in_unit("bohr")
    dimensionless = units.PhysicalQuantity(
        "dimensionless_normal_coordinate",
        "dimensionless",
        [1.0],
        normal_basis_sha256=basis_digest,
    )
    assert dimensionless.to_record()["frame"] is None
    with pytest.raises(units.UnitError, match="dimensions disagree"):
        units.PhysicalQuantity(
            "mass_weighted_normal_coordinate",
            "dimensionless",
            [1.0],
            normal_basis_sha256=basis_digest,
        )
