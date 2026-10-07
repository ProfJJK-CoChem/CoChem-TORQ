"""Real CPU calculations test research machinery, never qualify revDSD itself."""

from pathlib import Path

import numpy as np
import pytest

from cochem_torq.engines.revdsd import (
    ExperimentalDoubleHybrid,
    RecipeNotQualifiedError,
    ResearchCalculationError,
    ResearchRecipe,
    exact_revdsd_pbep86_d4_recipe,
    resolved_d4_parameters,
    restricted_pt2_components,
)

WATER_BOHR = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
H2_BOHR = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])


def test_exact_revdsd_recipe_is_blocked_without_evidence():
    with pytest.raises(RecipeNotQualifiedError, match="original article/SI"):
        exact_revdsd_pbep86_d4_recipe()


def test_user_coefficients_cannot_impersonate_revdsd():
    with pytest.raises(RecipeNotQualifiedError, match="accepted recipe registry"):
        ResearchRecipe("revDSD-PBEP86-D4", "HF", "HF", 1.0, 1.0)


@pytest.fixture(scope="module")
def pyscf_modules():
    return pytest.importorskip(
        "pyscf", reason="Actual PySCF required for engine evidence"
    )


def test_canonical_mp2_components_against_pyscf(pyscf_modules, tmp_path):
    from pyscf import gto, lib, mp, scf

    recipe = ResearchRecipe(
        "canonical restricted MP2 verification", "HF", "HF", 1.0, 1.0
    )
    result = ExperimentalDoubleHybrid(recipe, basis="sto-3g", grid_level=1).evaluate(
        ["O", "H", "H"],
        WATER_BOHR,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
    )
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["O", "H", "H"], WATER_BOHR)),
            unit="Bohr",
            basis="sto-3g",
            verbose=0,
        )
        hf = scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-8)
        correlated = mp.MP2(hf).run()
    assert result["energy_hartree"] == pytest.approx(correlated.e_tot, abs=1e-10)
    c = result["components"]
    assert c["opposite_spin_pt2_hartree"] == pytest.approx(
        correlated.e_corr_os, abs=1e-11
    )
    assert c["same_spin_pt2_hartree"] == pytest.approx(correlated.e_corr_ss, abs=1e-11)
    assert c["same_spin_pt2_hartree"] < 0
    assert result["qualification"] == "experimental_unqualified"
    assert result["independent_method_validation"] == "unavailable"
    artifact = Path(result["artifact_directory"])
    assert (artifact / "pyscf.log").stat().st_size > 100
    assert (artifact / "orbitals.chk").stat().st_size > 100
    assert len(result["provenance"]["basis_sha256"]) == 64
    assert "basis.json" in result["artifact_hashes"]


def test_frozen_core_components_match_explicit_pyscf_indices(pyscf_modules):
    from pyscf import gto, lib, mp, scf

    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["O", "H", "H"], WATER_BOHR)),
            unit="Bohr",
            basis="sto-3g",
            verbose=0,
        )
        hf = scf.RHF(mol).run(conv_tol=1e-12)
        ref = mp.MP2(hf, frozen=[0]).run()
        components = restricted_pt2_components(
            mol, hf.mo_coeff, hf.mo_energy, hf.mo_occ, frozen_occupied_orbitals=(0,)
        )
    assert components["opposite_spin_pt2_hartree"] == pytest.approx(
        ref.e_corr_os, abs=1e-11
    )
    assert components["same_spin_pt2_hartree"] == pytest.approx(
        ref.e_corr_ss, abs=1e-11
    )


def test_gks_orbitals_and_denominators_are_preserved(pyscf_modules, tmp_path):
    from pyscf import dft, gto, lib, mp

    # Explicit custom PBE0+PT2 model; this does not activate a named revDSD recipe.
    recipe = ResearchRecipe(
        "PBE0 plus explicitly scaled PT2 research model", "PBE0", "PBE0", 0.4, 0.2
    )
    result = ExperimentalDoubleHybrid(recipe, basis="sto-3g", grid_level=2).evaluate(
        ["O", "H", "H"],
        WATER_BOHR,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
    )
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["O", "H", "H"], WATER_BOHR)),
            unit="Bohr",
            basis="sto-3g",
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = "PBE0"
        mf.grids.level = 2
        mf.run(conv_tol=1e-12, conv_tol_grad=1e-8)
        correlation = mp.mp2.MP2(mf).run()  # Retains the actual RKS reference.
    c = result["components"]
    assert c["hybrid_reference_hartree"] == pytest.approx(mf.e_tot, abs=1e-9)
    assert c["opposite_spin_pt2_hartree"] == pytest.approx(
        correlation.e_corr_os, abs=1e-10
    )
    assert c["same_spin_pt2_hartree"] == pytest.approx(correlation.e_corr_ss, abs=1e-10)
    expected = mf.e_tot + 0.4 * correlation.e_corr_os + 0.2 * correlation.e_corr_ss
    assert result["energy_hartree"] == pytest.approx(expected, abs=1e-9)


def test_d4_resolved_parameters_and_direct_dispersion_agree(pyscf_modules, tmp_path):
    pytest.importorskip("dftd4", reason="Actual DFTD4 required for dispersion evidence")
    from dftd4.interface import DampingParam, DispersionModel

    resolved = resolved_d4_parameters("revdsdpbep86")
    assert resolved["parameters"] == {
        "s6": 0.5132,
        "s8": 0.0,
        "s9": 1.0,
        "a1": 0.44,
        "a2": 3.6,
        "alp": 16.0,
    }
    recipe = ResearchRecipe(
        "explicit HF/PT2/D4 component test", "HF", "HF", 1.0, 1.0, d4_method="pbe0"
    )
    result = ExperimentalDoubleHybrid(recipe, basis="sto-3g", grid_level=1).evaluate(
        ["H", "H"], H2_BOHR, charge=0, multiplicity=1, artifact_directory=tmp_path
    )
    model = DispersionModel(np.asarray([1, 1]), H2_BOHR, charge=0.0)
    reference = float(
        model.get_dispersion(DampingParam(method="pbe0", atm=True), grad=False)[
            "energy"
        ]
    )
    assert result["components"]["dispersion_hartree"] == pytest.approx(
        reference, abs=1e-13
    )
    assert result["dispersion"]["variant"] == "BJ-EEQ-ATM"
    assert result["provenance"]["dftd4"]


def test_numerical_full_energy_gradient_against_real_mp2_derivative(
    pyscf_modules, tmp_path
):
    from pyscf import gto, lib, mp, scf

    recipe = ResearchRecipe(
        "canonical MP2 numerical-derivative verification", "HF", "HF", 1.0, 1.0
    )
    evaluator = ExperimentalDoubleHybrid(recipe, basis="sto-3g", grid_level=1)
    result = evaluator.numerical_gradient(
        ["H", "H"],
        H2_BOHR,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
        steps_bohr=(0.001, 0.0005),
    )
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["H", "H"], H2_BOHR)), unit="Bohr", basis="sto-3g", verbose=0
        )
        hf = scf.RHF(mol).run(conv_tol=1e-12)
        expected = mp.MP2(hf).run().nuc_grad_method().kernel()
    numerical = np.asarray(result["gradient"])
    np.testing.assert_allclose(numerical, expected, atol=2e-7, rtol=0)
    np.testing.assert_allclose(numerical.sum(axis=0), 0.0, atol=1e-9, rtol=0)
    assert result["executed_energy_calculations"] == 24
    assert result["max_absolute_step_difference"] < 1e-6
    assert result["convergence_qualified"] is False
    assert all(
        Path(p["artifact_directory"], "result.json").is_file()
        for p in result["parents"]
    )


def test_nonconverged_scf_retains_failure_artifact(pyscf_modules, tmp_path):
    recipe = ResearchRecipe("SCF failure evidence", "HF", "HF", 1.0, 1.0)
    evaluator = ExperimentalDoubleHybrid(recipe, basis="sto-3g", max_cycle=1)
    with pytest.raises(ResearchCalculationError, match="SCF failed"):
        evaluator.evaluate(
            ["O", "H", "H"],
            WATER_BOHR,
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
        )
    directories = list(tmp_path.iterdir())
    assert len(directories) == 1
    assert (directories[0] / "failure.json").is_file()
    assert not (directories[0] / "result.json").exists()


def test_unsupported_spin_and_resource_limit_fail_explicitly(pyscf_modules, tmp_path):
    from pyscf import gto, lib, scf

    evaluator = ExperimentalDoubleHybrid(
        ResearchRecipe("guard test", "HF", "HF", 1.0, 1.0), basis="sto-3g"
    )
    with pytest.raises(ValueError, match="multiplicity 1"):
        evaluator.evaluate(
            ["H"],
            [[0.0, 0.0, 0.0]],
            charge=0,
            multiplicity=2,
            artifact_directory=tmp_path,
        )
    with pytest.raises(ValueError, match="two distinct"):
        evaluator.numerical_gradient(
            ["H", "H"],
            H2_BOHR,
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
            steps_bohr=(0.001,),
        )
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["H", "H"], H2_BOHR)), unit="Bohr", basis="sto-3g", verbose=0
        )
        hf = scf.RHF(mol).run()
        with pytest.raises(ResearchCalculationError, match="memory budget"):
            restricted_pt2_components(
                mol, hf.mo_coeff, hf.mo_energy, hf.mo_occ, max_tensor_bytes=1
            )


def test_numerical_composite_gradient_includes_native_d4_response(
    pyscf_modules, tmp_path
):
    pytest.importorskip("dftd4", reason="Actual DFTD4 required for derivative evidence")
    from dftd4.interface import DampingParam, DispersionModel
    from pyscf import gto, lib, mp, scf

    # A deliberately named component check, with no revDSD or accuracy claim.
    recipe = ResearchRecipe(
        "HF-MP2 plus explicit D4 derivative check",
        "HF",
        "HF",
        1.0,
        1.0,
        d4_method="pbe0",
    )
    evaluator = ExperimentalDoubleHybrid(recipe, basis="sto-3g", grid_level=1)
    result = evaluator.numerical_gradient(
        ["H", "H"],
        H2_BOHR,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
        steps_bohr=(0.001, 0.0005),
    )
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["H", "H"], H2_BOHR)), unit="Bohr", basis="sto-3g", verbose=0
        )
        hf = scf.RHF(mol).run(conv_tol=1e-12)
        mp2_gradient = mp.MP2(hf).run().nuc_grad_method().kernel()
    d4_gradient = DispersionModel(
        np.asarray([1, 1]), H2_BOHR, charge=0.0
    ).get_dispersion(DampingParam(method="pbe0", atm=True), grad=True)["gradient"]
    np.testing.assert_allclose(
        result["gradient"], mp2_gradient + d4_gradient, atol=2e-7, rtol=0
    )
    assert result["executed_energy_calculations"] == 24
