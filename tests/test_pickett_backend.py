"""Real externally provisioned SPCAT/SPFIT and independent mathematical checks.

Native regression targets come unchanged from actual SPCAT calculations; they
are explicitly not laboratory measurements. There are no emulator binaries,
mocked engines, injected providers or invented observed reference values.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_torq.artifacts import file_digest
from cochem_torq.domain import (
    CalculationRequest,
    PrerequisiteError,
    canonical_json,
    read_json,
)
from cochem_torq.publication import export_publication_bundle
from cochem_torq.spectroscopy.pickett_backend import (
    CatalogSettings,
    FitLine,
    PickettModel,
    _inventory,
    _parameter_deck,
    compare_rigid_rotor_reference,
    configured_pickett_installation,
    decode_quantum_number,
    encode_quantum_number,
    load_pickett_installation,
    run_spcat,
    run_spfit,
    verify_native_run,
)
from cochem_torq.spectroscopy.rotational import (
    rigid_rotor_catalog,
    rigid_rotor_levels,
)


def mathematical_model(**changes):
    fields = {
        "title": "TORQ mathematical model; not experiment",
        "reduction": "A",
        "constant_observable": "mathematical_model",
        "parameter_origin": "mathematical_model",
        "constants_mhz": (10000.0, 5000.0, 3000.0),
        "dipole_abc_debye": (1.0, 1.0, 1.0),
    }
    fields.update(changes)
    return PickettModel(**fields)


def finite_j_settings(model, **changes):
    rotor = rigid_rotor_catalog(
        model.constants_mhz,
        model.dipole_abc_debye,
        temperature_kelvin=10.0,
        J_max=5,
        constant_observable="Be",
    )
    fields = {
        "temperature_kelvin": 10.0,
        "maximum_j": 5,
        "maximum_frequency_ghz": 100.0,
        "partition_function": rotor.partition_function,
        "partition_origin": "Independent finite-J=0..5 sum; spin weights excluded.",
    }
    fields.update(changes)
    return CatalogSettings(**fields)


@pytest.fixture(scope="module")
def native_installation():
    # Absence is a genuine provisioning failure, never a skipped scientific pass.
    return configured_pickett_installation()


@pytest.fixture(scope="module")
def actual_native_catalog(native_installation, tmp_path_factory):
    root = tmp_path_factory.mktemp("actual-native-pickett-catalog")
    model = mathematical_model()
    settings = finite_j_settings(model)
    target = root / "spcat"
    receipt = run_spcat(native_installation, model, settings, target)
    assert receipt["status"] == "completed_native_unqualified", receipt["failure"]
    verify_native_run(target, expected_manifest_sha256=receipt["manifest_sha256"])
    return target, receipt, model, settings


@pytest.mark.parametrize(
    "integer,encoded",
    [
        (-269, "z9"),
        (-260, "z0"),
        (-19, "a9"),
        (-10, "a0"),
        (-9, "-9"),
        (0, " 0"),
        (9, " 9"),
        (99, "99"),
        (100, "A0"),
        (109, "A9"),
        (110, "B0"),
        (359, "Z9"),
    ],
)
def test_exact_native_two_column_quantum_number_math(integer, encoded):
    assert encode_quantum_number(integer) == encoded
    assert decode_quantum_number(encoded) == integer


@pytest.mark.parametrize("value", [-270, 360, True, 1.0])
def test_quantum_overflow_and_coercion_are_rejected(value):
    with pytest.raises(ValueError, match="span"):
        encode_quantum_number(value)


@pytest.mark.parametrize("encoded", ["**", "1", " 1 ", "aX", "é1"])
def test_malformed_or_overflowed_quantum_fields_are_rejected(encoded):
    with pytest.raises(ValueError):
        decode_quantum_number(encoded)


def test_missing_actual_provisioning_cannot_select_a_substitute_binary(tmp_path):
    with pytest.raises(PrerequisiteError, match="provisioning"):
        load_pickett_installation(
            tmp_path / "missing-provisioning.json", expected_manifest_sha256="0" * 64
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"constants_mhz": (3000.0, 5000.0, 10000.0)},
        {"constants_mhz": (10000.0, 5000.0, math.nan)},
        {"constant_observable": "B0"},
        {"parameter_origin": "estimated"},
        {"dipole_abc_debye": (0.0, 0.0, 0.0)},
        {"representation": "IIIl"},
        {"parameter_unit": "GHz"},
        {"quartic_mhz": {"D_J": 0.001}},
    ],
)
def test_undefined_mislabeled_estimated_or_unsupported_models_are_rejected(changes):
    with pytest.raises(ValidationError):
        mathematical_model(**changes)


def test_unknown_covariance_remains_unavailable_and_nonpositive_covariance_fails():
    model = mathematical_model()
    assert model.parameter_covariance_mhz2 is None
    with pytest.raises(ValidationError, match="provenance"):
        mathematical_model(
            parameter_covariance_mhz2=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
    with pytest.raises(ValidationError, match="positive definite"):
        mathematical_model(
            parameter_covariance_mhz2=(
                (1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ),
            covariance_origin="Mathematical singular matrix rejection case.",
        )


def test_parameter_identifiers_and_operator_signs_follow_explicit_reduction():
    for reduction, names in [
        ("A", ("Delta_J", "Delta_JK", "Delta_K", "delta_J", "delta_K")),
        ("S", ("D_J", "D_JK", "D_K", "d1", "d2")),
    ]:
        model = mathematical_model(
            reduction=reduction, quartic_mhz={name: 0.001 for name in names}
        )
        deck = _parameter_deck(model).decode().splitlines()
        records = [line.split("/", 1)[0].split() for line in deck[3:]]
        assert [int(row[0]) for row in records] == (
            [10000, 20000, 30000, 200, 1100, 2000, 40100, 41000]
            if reduction == "A"
            else [10000, 20000, 30000, 200, 1100, 2000, 40100, 50000]
        )
        assert [float(row[1]) for row in records[3:]] == (
            [-0.001] * 5 if reduction == "A" else [-0.001] * 3 + [0.001] * 2
        )
        assert all(len(line) <= 79 for line in deck)


@pytest.mark.real_engine
@pytest.mark.research
def test_actual_source_binary_pins_and_missing_distribution_permission(
    native_installation,
):
    native_installation.verify()
    assert native_installation.distribution_permission == "unverified_not_bundled"
    with pytest.raises(PrerequisiteError, match="pinned"):
        load_pickett_installation(
            native_installation.manifest_path, expected_manifest_sha256="0" * 64
        )


@pytest.mark.real_engine
@pytest.mark.research
def test_genuine_spcat_catalog_matches_independent_finite_j_math(actual_native_catalog):
    target, receipt, model, settings = actual_native_catalog
    rows = receipt["analysis"]["lines"]
    comparison = compare_rigid_rotor_reference(rows, model, settings)
    assert (
        comparison["native_line_count"]
        == comparison["independent_in_band_line_count"]
        == 229
    )
    assert comparison["maximum_absolute_frequency_residual_mhz"] <= 0.000051
    assert (
        comparison["maximum_absolute_log10_integrated_intensity_residual"] <= 0.000051
    )
    assert not comparison["experimental_accuracy_established"]
    assert all(row["frequency_standard_uncertainty_mhz"] is None for row in rows)
    assert all(row["native_error_column_mhz"] == 0.0 for row in rows)
    assert (target / "inputs/molecule.var").read_bytes() == (
        target / "molecule.var"
    ).read_bytes()


@pytest.mark.real_engine
@pytest.mark.research
def test_genuine_artifacts_are_immutable_and_input_output_corruption_is_rejected(
    actual_native_catalog, native_installation, tmp_path
):
    target, receipt, model, settings = actual_native_catalog
    original = file_digest(target / "manifest.json")
    with pytest.raises(FileExistsError, match="overwritten"):
        run_spcat(native_installation, model, settings, target)
    assert file_digest(target / "manifest.json") == original
    changed = tmp_path / "damaged"
    shutil.copytree(target, changed)
    with (changed / "molecule.cat").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="inventory/hash"):
        verify_native_run(changed, expected_manifest_sha256=receipt["manifest_sha256"])


@pytest.mark.real_engine
@pytest.mark.research
def test_resealed_summary_cannot_change_actual_native_observations(
    actual_native_catalog, tmp_path
):
    target, _, _, _ = actual_native_catalog
    changed = tmp_path / "deliberately-corrupted-summary"
    shutil.copytree(target, changed)
    analysis = read_json(changed / "analysis.json")
    analysis["lines"][0]["frequency_mhz"] += 1.0
    (changed / "analysis.json").write_bytes(canonical_json(analysis) + b"\n")
    manifest = read_json(changed / "manifest.json")
    manifest["files"] = _inventory(changed)
    (changed / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="actual retained output"):
        verify_native_run(
            changed, expected_manifest_sha256=file_digest(changed / "manifest.json")
        )


@pytest.mark.real_engine
@pytest.mark.research
def test_resealed_request_cannot_relabel_actual_native_parameter_decks(
    actual_native_catalog, tmp_path
):
    target, _, _, _ = actual_native_catalog
    changed = tmp_path / "deliberately-corrupted-request"
    shutil.copytree(target, changed)
    request = read_json(changed / "request.json")
    request["model"]["constants_mhz"][0] += 1.0
    (changed / "request.json").write_bytes(canonical_json(request) + b"\n")
    manifest = read_json(changed / "manifest.json")
    manifest["files"] = _inventory(changed)
    (changed / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="input decks differ"):
        verify_native_run(
            changed, expected_manifest_sha256=file_digest(changed / "manifest.json")
        )


def native_regression_lines(receipt, *, only_delta_j_zero=False):
    return [
        FitLine(
            upper_j_ka_kc=row["upper_j_ka_kc"],
            lower_j_ka_kc=row["lower_j_ka_kc"],
            frequency_mhz=row["frequency_mhz"],
            standard_uncertainty_or_regression_tolerance_mhz=0.0001,
            source_id="actual-spcat-regression",
            origin="verified_native_regression_prediction",
        )
        for row in receipt["analysis"]["lines"]
        if not only_delta_j_zero or row["upper_j_ka_kc"][0] == row["lower_j_ka_kc"][0]
    ]


@pytest.mark.real_engine
@pytest.mark.research
def test_genuine_spfit_spcat_roundtrip_preserves_covariance_without_experiment_label(
    actual_native_catalog, native_installation, tmp_path
):
    target, receipt, _, _ = actual_native_catalog
    initial = mathematical_model(constants_mhz=(10001.0, 5001.0, 3001.0))
    lines = native_regression_lines(receipt)
    fitted = run_spfit(
        native_installation,
        initial,
        lines,
        tmp_path / "fit",
        rejection_limit=1e9,
        native_catalog_sources={
            "actual-spcat-regression": (target, receipt["manifest_sha256"])
        },
    )
    assert fitted["status"] == "completed_native_unqualified", fitted["failure"]
    result = fitted["analysis"]
    assert np.allclose(
        result["constants_mhz"], (10000.0, 5000.0, 3000.0), atol=0.000002, rtol=0
    )
    assert result["covariance_status"] == "full_rank_native_local_linearization"
    assert np.linalg.eigvalsh(result["parameter_covariance_mhz2"]).min() > 0
    assert result["native_microwave_rms_mhz"] <= 0.00005
    assert result["native_final_rejection_count"] == 0
    assert (
        result["fit_origin"] == "verified_native_prediction_regression_not_experiment"
    )
    verify_native_run(
        tmp_path / "fit", expected_manifest_sha256=fitted["manifest_sha256"]
    )
    fit_request = read_json(tmp_path / "fit/request.json")
    source = fit_request["native_catalog_sources"]["actual-spcat-regression"]
    retained_source = tmp_path / "fit" / source["retained_path"]
    assert source["manifest_sha256"] == receipt["manifest_sha256"]
    assert (retained_source / "molecule.cat").read_bytes() == (
        target / "molecule.cat"
    ).read_bytes()
    verify_native_run(
        retained_source, expected_manifest_sha256=source["manifest_sha256"]
    )
    corrupted_fit = tmp_path / "fit-with-corrupted-retained-source"
    shutil.copytree(tmp_path / "fit", corrupted_fit)
    corrupted_catalog = corrupted_fit / source["retained_path"] / "molecule.cat"
    with corrupted_catalog.open("ab") as stream:
        stream.write(b"\n")
    changed_manifest = read_json(corrupted_fit / "manifest.json")
    changed_manifest["files"] = _inventory(corrupted_fit)
    (corrupted_fit / "manifest.json").write_bytes(
        canonical_json(changed_manifest) + b"\n"
    )
    with pytest.raises(ValueError, match="inventory/hash"):
        verify_native_run(
            corrupted_fit,
            expected_manifest_sha256=file_digest(corrupted_fit / "manifest.json"),
        )
    assert (tmp_path / "fit/inputs/molecule.par").read_bytes() != (
        tmp_path / "fit/molecule.par"
    ).read_bytes()
    fitted_model = mathematical_model(
        constants_mhz=tuple(result["constants_mhz"]),
        parameter_covariance_mhz2=tuple(
            tuple(row) for row in result["parameter_covariance_mhz2"]
        ),
        covariance_origin="Native SPFIT regression covariance; "
        "not experimental uncertainty.",
    )
    recatalog = run_spcat(
        native_installation,
        fitted_model,
        finite_j_settings(fitted_model),
        tmp_path / "fitted-catalog",
    )
    assert recatalog["status"] == "completed_native_unqualified", recatalog["failure"]
    for old, new in zip(receipt["analysis"]["lines"], recatalog["analysis"]["lines"]):
        assert (
            old["upper_j_ka_kc"] == new["upper_j_ka_kc"]
            and old["lower_j_ka_kc"] == new["lower_j_ka_kc"]
        )
        assert abs(old["frequency_mhz"] - new["frequency_mhz"]) <= 0.00011
        assert new["frequency_standard_uncertainty_mhz"] is not None


@pytest.mark.real_engine
@pytest.mark.research
def test_underidentified_genuine_fit_retains_rank_deficiency(
    actual_native_catalog, native_installation, tmp_path
):
    target, receipt, _, _ = actual_native_catalog
    fitted = run_spfit(
        native_installation,
        mathematical_model(constants_mhz=(10001.0, 5001.0, 3001.0)),
        native_regression_lines(receipt, only_delta_j_zero=True),
        tmp_path / "underidentified",
        rejection_limit=1e9,
        native_catalog_sources={
            "actual-spcat-regression": (target, receipt["manifest_sha256"])
        },
    )
    assert fitted["status"] == "completed_native_unqualified", fitted["failure"]
    assert fitted["analysis"]["covariance_status"] == "rank_deficient_native_factor"
    assert fitted["analysis"]["covariance_rank"] < 3
    assert not fitted["analysis"]["experimental_accuracy_established"]


@pytest.mark.real_engine
@pytest.mark.research
def test_invented_regression_frequency_is_rejected_before_native_execution(
    actual_native_catalog, native_installation, tmp_path
):
    target, receipt, model, _ = actual_native_catalog
    line = native_regression_lines(receipt)[0]
    changed = FitLine.model_validate(
        {**line.model_dump(), "frequency_mhz": line.frequency_mhz + 1.0}
    )
    with pytest.raises(ValueError, match="actual verified native catalog"):
        run_spfit(
            native_installation,
            model,
            [changed],
            tmp_path / "rejected",
            native_catalog_sources={
                "actual-spcat-regression": (target, receipt["manifest_sha256"])
            },
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.real_engine
@pytest.mark.research
def test_actual_native_timeout_retains_owned_input_output_and_explicit_failure(
    native_installation, tmp_path
):
    model = mathematical_model()
    receipt = run_spcat(
        native_installation,
        model,
        finite_j_settings(model, maximum_j=30, wall_seconds=0.000001),
        tmp_path / "timed-out",
    )
    assert receipt["status"] == "failed" and receipt["analysis"] is None
    assert "wall-time" in receipt["failure"]
    verify_native_run(
        tmp_path / "timed-out", expected_manifest_sha256=receipt["manifest_sha256"]
    )
    assert (tmp_path / "timed-out/inputs/molecule.var").is_file()


@pytest.mark.real_engine
@pytest.mark.research
def test_explicit_full_covariance_matches_independent_energy_derivatives(
    native_installation, tmp_path
):
    covariance = np.array(
        [[1e-4, 2e-5, -1e-5], [2e-5, 2e-4, 3e-5], [-1e-5, 3e-5, 3e-4]]
    )
    model = mathematical_model(
        parameter_covariance_mhz2=tuple(
            tuple(float(x) for x in row) for row in covariance
        ),
        covariance_origin="Specified mathematical parameter covariance.",
    )
    settings = finite_j_settings(model)
    receipt = run_spcat(
        native_installation, model, settings, tmp_path / "full-covariance"
    )
    assert receipt["status"] == "completed_native_unqualified", receipt["failure"]
    levels = rigid_rotor_levels(model.constants_mhz, settings.maximum_j)
    independent = rigid_rotor_catalog(
        model.constants_mhz,
        model.dipole_abc_debye,
        temperature_kelvin=10.0,
        J_max=5,
        constant_observable="Be",
    )
    level_map = {(level.J, level.eigenstate_index): level for level in levels}

    def derivative(level):
        j = level.J
        ks = np.asarray(level.K_basis, dtype=float)
        da = np.diag(ks**2)
        db = np.diag(0.5 * (j * (j + 1) - ks**2))
        dc = db.copy()
        for index, kval in enumerate(ks[:-2]):
            value = (
                math.sqrt(
                    (j * (j + 1) - kval * (kval + 1))
                    * (j * (j + 1) - (kval + 1) * (kval + 2))
                )
                / 4
            )
            db[index, index + 2] = db[index + 2, index] = value
            dc[index, index + 2] = dc[index + 2, index] = -value
        v = level.coefficients
        return np.asarray([v @ matrix @ v for matrix in (da, db, dc)])

    errors = []
    for row in receipt["analysis"]["lines"]:
        candidates = [
            line
            for line in independent.lines
            if line.upper_J == row["upper_j_ka_kc"][0]
            and line.lower_J == row["lower_j_ka_kc"][0]
        ]
        match = min(
            candidates, key=lambda line: abs(line.frequency_mhz - row["frequency_mhz"])
        )
        gradient = derivative(
            level_map[(match.upper_J, match.upper_eigenstate_index)]
        ) - derivative(level_map[(match.lower_J, match.lower_eigenstate_index)])
        errors.append(
            abs(
                math.sqrt(float(gradient @ covariance @ gradient))
                - row["frequency_standard_uncertainty_mhz"]
            )
        )
    assert max(errors) <= 0.000052


@pytest.mark.real_engine
@pytest.mark.research
@pytest.mark.parametrize("reduction", ["A", "S"])
def test_quartic_native_operators_match_independent_angular_momentum_matrices(
    native_installation, tmp_path, reduction
):
    names = (
        ("Delta_J", "Delta_JK", "Delta_K", "delta_J", "delta_K")
        if reduction == "A"
        else ("D_J", "D_JK", "D_K", "d1", "d2")
    )
    values = (0.001, 0.002, 0.003, 0.0004, 0.0005)
    model = mathematical_model(
        reduction=reduction, quartic_mhz=dict(zip(names, values))
    )
    spectra = {}
    for j in range(6):
        ks = np.arange(-j, j + 1, dtype=float)
        ja2 = np.diag(ks**2)
        plus = np.zeros((2 * j + 1, 2 * j + 1))
        for index, kval in enumerate(ks[:-1]):
            plus[index + 1, index] = math.sqrt(j * (j + 1) - kval * (kval + 1))
        minus = plus.T
        s2 = plus @ plus + minus @ minus
        s4 = np.linalg.matrix_power(plus, 4) + np.linalg.matrix_power(minus, 4)
        j2 = j * (j + 1)
        a, b, c = model.constants_mhz
        matrix = a * ja2 + (b + c) / 2 * (j2 * np.eye(len(ks)) - ja2)
        matrix += (b - c) / 4 * s2
        matrix -= values[0] * j2**2 * np.eye(len(ks))
        matrix -= values[1] * j2 * ja2 + values[2] * ja2 @ ja2
        if reduction == "A":
            matrix -= values[3] * j2 * s2 + values[4] / 2 * (ja2 @ s2 + s2 @ ja2)
        else:
            matrix += values[3] * j2 * s2 + values[4] * s4
        spectra[j] = np.linalg.eigvalsh(matrix)
    from scipy.constants import h, k

    partition = sum(
        (2 * j + 1) * float(np.exp(-h * 1e6 * energy / (k * 10.0)))
        for j, energies in spectra.items()
        for energy in energies
    )
    settings = finite_j_settings(
        model,
        partition_function=partition,
        partition_origin="Independent distorted finite-J angular-momentum matrix sum.",
    )
    receipt = run_spcat(native_installation, model, settings, tmp_path / "quartic")
    assert receipt["status"] == "completed_native_unqualified", receipt["failure"]
    errors = []
    for row in receipt["analysis"]["lines"]:
        upper = spectra[row["upper_j_ka_kc"][0]]
        lower = spectra[row["lower_j_ka_kc"][0]]
        separations = (upper[:, None] - lower[None, :]).ravel()
        positive = separations[separations > 0]
        errors.append(float(np.min(np.abs(positive - row["frequency_mhz"]))))
    assert max(errors) <= 0.000051


@pytest.mark.real_engine
@pytest.mark.research
def test_actual_water_quantum_geometry_and_dipole_export_keep_be_label(
    native_installation, tmp_path
):
    from cochem_torq.application import execute_request

    request = CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["O", "H", "H"],
                "geometry_bohr": [[0.0, 0.0, 0.0], [1.4, 0.0, 1.1], [-1.4, 0.0, 1.1]],
                "charge": 0,
                "multiplicity": 1,
                "isotopes": [16, 1, 1],
            },
            "recipe": "hf-sto-3g-education",
            "products": [
                "geometry",
                "harmonic",
                "equilibrium_constants",
                "rigid_rotor_catalog",
            ],
            "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
            "catalog": {"temperature_kelvin": 10.0, "max_j": 5},
        }
    )
    result = execute_request(request, tmp_path / "quantum-shard")
    assert result["status"] == "complete", result["errors"]
    export = export_publication_bundle(
        tmp_path / "quantum-shard", tmp_path / "quantum-publication"
    )
    rotor = result["stages"]["equilibrium_constants"]["value"]
    dipole = np.asarray(result["native_result"]["dipole_debye"]) @ np.asarray(
        rotor["principal_axes_columns"]
    )
    model = PickettModel(
        title="Actual HF water equilibrium screening",
        reduction="A",
        constant_observable="Be",
        parameter_origin="ab_initio",
        constants_mhz=tuple(rotor["constants_mhz"]),
        dipole_abc_debye=tuple(float(x) for x in dipole),
        source_bundle=str(tmp_path / "quantum-publication"),
        source_manifest_sha256=export["manifest_sha256"],
    )
    settings = finite_j_settings(model, maximum_frequency_ghz=10000.0)
    catalog = run_spcat(
        native_installation, model, settings, tmp_path / "actual-water-spcat"
    )
    assert catalog["status"] == "completed_native_unqualified", catalog["failure"]
    assert catalog["analysis"]["constant_observable"] == "Be"
    assert catalog["analysis"]["parameter_origin"] == "ab_initio"
    assert not catalog["analysis"]["identification_ready"]
    wrong = PickettModel.model_validate(
        {
            **model.model_dump(),
            "constants_mhz": (model.constants_mhz[0] + 1.0, *model.constants_mhz[1:]),
        }
    )
    with pytest.raises(ValueError, match="actual typed available"):
        run_spcat(
            native_installation, wrong, settings, tmp_path / "cannot-substitute-water"
        )
