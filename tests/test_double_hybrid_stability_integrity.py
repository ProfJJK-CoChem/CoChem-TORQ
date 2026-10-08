"""Genuine orbital-reference diagnostics; no exact-revDSD qualification claim."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from cochem_torq.engines import revdsd
from cochem_torq.engines.revdsd import ExperimentalDoubleHybrid, ResearchRecipe

pytestmark = [pytest.mark.research, pytest.mark.real_engine]


@pytest.fixture(scope="module")
def actual_reference_cases(tmp_path_factory):
    from pyscf import dft, gto, lib, mp, scf

    root = tmp_path_factory.mktemp("cochem_exec_actual_double_hybrid_stability")
    records = {}
    for xc in ("HF", "PBE0"):
        for separation in (1.4, 4.0):
            coords = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, separation]])
            recipe = ResearchRecipe(
                "explicit orbital stability research " + xc, xc, xc, 0.4, 0.2
            )
            calculator = ExperimentalDoubleHybrid(
                recipe,
                basis="sto-3g",
                grid_level=2,
                stability_tolerance=1e-8,
                stability_nroots=1,
            )
            result = calculator.evaluate(
                ["H", "H"], coords, charge=0, multiplicity=1, artifact_directory=root
            )
            # A genuinely separate invocation of the same native API checks its
            # classification and retained original constrained-reference energy.
            # It is not independent-engine or exact-method qualification.
            with lib.with_omp_threads(1):
                directory = root / f"direct-{xc}-{separation}"
                directory.mkdir()
                mol = gto.M(
                    atom=list(zip(["H", "H"], coords)),
                    unit="Bohr",
                    basis="sto-3g",
                    verbose=4,
                    output=str(directory / "pyscf.log"),
                )
                try:
                    mf = scf.RHF(mol) if xc == "HF" else dft.RKS(mol)
                    if xc != "HF":
                        mf.xc = xc
                        mf.grids.level = 2
                    mf.chkfile = str(directory / "orbitals.chk")
                    mf.run(conv_tol=1e-12, conv_tol_grad=1e-8)
                    assert mf.converged
                    before = np.array(mf.mo_coeff, copy=True)
                    _, _, stable_i, stable_e = mf.stability(
                        internal=True,
                        external=True,
                        return_status=True,
                        tol=1e-8,
                        nroots=1,
                    )
                    np.testing.assert_array_equal(mf.mo_coeff, before)
                    pt2 = mp.mp2.MP2(mf).run()
                    expected = mf.e_tot + 0.4 * pt2.e_corr_os + 0.2 * pt2.e_corr_ss
                finally:
                    mol.stdout.close()
            records[xc, separation] = (result, bool(stable_i), bool(stable_e), expected)
    return records


@pytest.mark.parametrize("xc", ["HF", "PBE0"])
@pytest.mark.parametrize("separation", [1.4, 4.0])
def test_actual_stable_and_unstable_hf_gks_reference_is_retained(
    actual_reference_cases, xc, separation
):
    result, expected_internal, expected_external, expected_energy = (
        actual_reference_cases[xc, separation]
    )
    diagnostic = result["reference_stability"]
    assert diagnostic["internal_stable"] is expected_internal
    assert diagnostic["restricted_to_unrestricted_stable"] is expected_external
    assert expected_internal is True
    assert expected_external is (separation == 1.4)
    assert diagnostic["status"] == ("stable" if separation == 1.4 else "unstable")
    assert diagnostic["scope"] == "orbital-generating HF/GKS reference only"
    assert diagnostic["full_correlated_energy_stability"] == "not_evaluated"
    assert diagnostic["real_to_complex_status"] == "not_separately_extracted"
    assert diagnostic["returned_candidate_orbitals_applied"] is False
    assert diagnostic["orbital_reference_unchanged"] is True
    assert (
        diagnostic["orbital_state_sha256_before"]
        == diagnostic["orbital_state_sha256_after"]
    )
    assert result["energy_hartree"] == pytest.approx(expected_energy, abs=1e-10)
    assert result["qualification"] == "experimental_unqualified"
    assert result["independent_method_validation"] == "unavailable"
    assert result["method"].startswith("custom-research/")
    assert result["quality_flags"] == (
        [] if separation == 1.4 else ["orbital_reference_unstable"]
    )
    directory = Path(result["artifact_directory"])
    diagnostic_path = directory / "reference-stability.json"
    actual = json.loads(diagnostic_path.read_text())
    assert actual == {
        key: value for key, value in diagnostic.items() if key != "artifact_sha256"
    }
    assert (
        sha256(diagnostic_path.read_bytes()).hexdigest()
        == diagnostic["artifact_sha256"]
    )
    assert (
        diagnostic["artifact_sha256"] == result["artifact_hashes"][diagnostic_path.name]
    )
    assert (
        sha256((directory / "orbitals.chk").read_bytes()).hexdigest()
        == diagnostic["checkpoint_sha256_before"]
        == diagnostic["checkpoint_sha256_after"]
    )
    assert "stability" in (directory / "pyscf.log").read_text().lower()
    assert diagnostic["controls"] == {
        "internal": True,
        "external": True,
        "return_status": True,
        "nroots": 1,
        "tol": 1e-8,
    }


def test_explicitly_unchecked_reference_is_not_declared_stable(tmp_path):
    recipe = ResearchRecipe("explicit unchecked HF research", "HF", "HF", 0.4, 0.2)
    result = ExperimentalDoubleHybrid(
        recipe, basis="sto-3g", check_reference_stability=False
    ).evaluate(
        ["H", "H"],
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
    )
    diagnostic = result["reference_stability"]
    assert diagnostic["status"] == "not_requested"
    assert diagnostic["internal_stable"] is None
    assert diagnostic["restricted_to_unrestricted_stable"] is None
    assert diagnostic["requested"] is False
    assert diagnostic["orbital_reference_unchanged"] is True
    assert result["quality_flags"] == ["orbital_reference_not_requested"]
    assert result["recipe_sha256"] == recipe.fingerprint


@pytest.mark.parametrize("separation", [1.4, 4.0])
def test_every_actual_displaced_gradient_reference_retains_its_diagnostic(
    tmp_path, separation
):
    recipe = ResearchRecipe("displaced-reference research", "HF", "HF", 0.4, 0.2)
    calculator = ExperimentalDoubleHybrid(recipe, basis="sto-3g", stability_nroots=1)
    result = calculator.numerical_gradient(
        ["H", "H"],
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, separation]]),
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
        steps_bohr=(0.002, 0.001),
        max_energy_calculations=24,
    )
    status = "stable" if separation == 1.4 else "unstable"
    summary = result["reference_stability"]
    assert summary["status"] == status
    assert summary["evaluation_counts"][status] == 24
    assert summary["evaluations"] == result["executed_energy_calculations"] == 24
    assert summary["all_evaluations_stable_within_checked_scope"] is (separation == 1.4)
    assert summary["full_correlated_energy_stability"] == "not_evaluated"
    assert summary["state_continuity"] == "not_qualified"
    assert result["convergence_qualified"] is False
    for parent in result["parents"]:
        directory = Path(parent["artifact_directory"])
        native = json.loads((directory / "result.json").read_text())
        assert (
            native["reference_stability"]["status"]
            == parent["reference_stability_status"]
            == status
        )
        assert (
            sha256((directory / "reference-stability.json").read_bytes()).hexdigest()
            == (parent["reference_stability_artifact_sha256"])
        )
        assert (
            sha256((directory / "result.json").read_bytes()).hexdigest()
            == parent["result_sha256"]
        )


def test_named_exact_recipe_stays_blocked_after_reference_diagnostics():
    with pytest.raises(revdsd.RecipeNotQualifiedError, match="original article/SI"):
        revdsd.exact_revdsd_pbep86_d4_recipe()
