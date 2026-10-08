"""Explicit, unqualified restricted double-hybrid research calculations.

This module implements conventional PT2 on the *specified* GKS orbitals. It
never constructs an alleged revDSD recipe from unrelated coefficients. The
exact revDSD-PBEP86-D4 entry remains closed until its complete definition and
independent reference evidence have been reconciled.
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import numpy as np


class RecipeNotQualifiedError(ValueError):
    """The named scientific method has no accepted complete recipe."""


class ResearchCalculationError(RuntimeError):
    """A calculation did not supply a scientifically usable research result."""


class _ResearchSettings(TypedDict):
    basis: str
    grid_level: int
    scf_energy_tolerance: float
    scf_gradient_tolerance: float
    max_cycle: int
    max_memory_mb: int
    threads: int
    check_reference_stability: bool
    stability_tolerance: float
    stability_nroots: int


EXACT_REVDSD_GAPS = (
    "Complete revDSD-PBEP86-D4 exchange, correlation and OS/SS coefficients "
    "have not been reconciled with original article/SI.",
    "The orbital-generating functional and precise P86 implementation "
    "(including FT/VWN variants) need independent verification.",
    "The published frozen-core convention and any 2019/2021 recipe "
    "distinction need independent verification.",
    "No matched independent native revDSD reference bundle is available.",
)

D4_SOURCE = {
    "url": "https://github.com/dftd4/dftd4/blob/"
    "7b2ff85a71a3630808fd8a2d972933f75b743f3c/assets/parameters.toml",
    "sha256": "8e44a7fed51e511c9e2ab0938a94f0a21bacc4c70f470a915224239937c1d61f",
    "scope": "DFTD4 3.7.0 dispersion parameters only; not the electronic recipe",
}


def exact_revdsd_pbep86_d4_recipe() -> ResearchRecipe:
    """Fail explicitly until DH0 and independent DH1 evidence exist."""
    raise RecipeNotQualifiedError(" ".join(EXACT_REVDSD_GAPS))


@dataclass(frozen=True)
class ResearchRecipe:
    """An explicitly named user research model, without accuracy certification.

    ``orbital_xc`` determines the self-consistent orbitals and denominators;
    ``energy_xc`` determines the hybrid part evaluated on those orbitals.
    Both use the PySCF/LibXC expression syntax. All weights must be explicit.
    A D4 term is optional and must name a released library parametrization.
    """

    name: str
    orbital_xc: str
    energy_xc: str
    opposite_spin_scale: float
    same_spin_scale: float
    frozen_occupied_orbitals: tuple[int, ...] = ()
    d4_method: str | None = None
    source_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(x, str) for x in (self.name, self.orbital_xc, self.energy_xc)
        ):
            raise ValueError("Name and XC expressions must be strings")
        normalized = self.name.lower().replace("-", "").replace("_", "")
        if "revdsd" in normalized or "revdod" in normalized:
            raise RecipeNotQualifiedError(
                "Named revDSD/revDOD recipes must be supplied by the accepted "
                "recipe registry; custom research coefficients cannot activate them."
            )
        if not self.name.strip() or not self.orbital_xc or not self.energy_xc:
            raise ValueError("Explicit name, orbital XC and energy XC are required")
        scales = (self.opposite_spin_scale, self.same_spin_scale)
        if any(not np.isfinite(x) or x < 0 for x in scales):
            raise ValueError("PT2 scaling coefficients must be finite and nonnegative")
        if not isinstance(self.frozen_occupied_orbitals, tuple) or not isinstance(
            self.source_evidence, tuple
        ):
            raise ValueError(
                "Recipe indices and source evidence must be immutable tuples"
            )
        if any(not isinstance(x, str) or not x for x in self.source_evidence):
            raise ValueError("Source evidence must contain nonempty locators")
        frozen = self.frozen_occupied_orbitals
        if any(type(x) is not int or x < 0 for x in frozen) or len(set(frozen)) != len(
            frozen
        ):
            raise ValueError(
                "Frozen occupied indices must be distinct nonnegative integers"
            )

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _orbital_state_digest(mf: Any) -> str:
    """Hash the actual orbital coefficients, energies and occupations unchanged."""
    digest = sha256()
    for name in ("mo_coeff", "mo_energy", "mo_occ"):
        values = np.asarray(getattr(mf, name))
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ResearchCalculationError("Finite real reference orbitals required")
        digest.update(
            json.dumps(
                {"name": name, "shape": values.shape, "dtype": values.dtype.str},
                sort_keys=True,
            ).encode()
        )
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _reference_stability(
    mf: Any,
    settings: _ResearchSettings,
    recipe: ResearchRecipe,
    directory: Path,
) -> dict[str, Any]:
    """Diagnose only the supplied HF/GKS reference without replacing orbitals.

    PySCF's returned external flag describes RHF/RKS -> UHF/UKS. Its native
    routine also logs real-to-complex checks, but does not return their separate
    flag; the boolean here must not be described as complete external stability.
    No stability of the full nonvariational correlated energy is established.
    """
    from pyscf.scf import stability

    before = _orbital_state_digest(mf)
    checkpoint_before = _file_hash(directory / "orbitals.chk")
    record: dict[str, Any] = {
        "status": "not_requested",
        "internal_stable": None,
        "restricted_to_unrestricted_stable": None,
        "scope": "orbital-generating HF/GKS reference only",
        "internal_scope": "restricted real-orbital variations",
        "external_scope": "native returned RHF/RKS -> UHF/UKS status",
        "real_to_complex_status": "not_separately_extracted",
        "full_correlated_energy_stability": "not_evaluated",
        "solver": "PySCF RHF/RKS stability",
        "pyscf_version": importlib.metadata.version("pyscf"),
        "solver_source_sha256": _file_hash(Path(stability.__file__)),
        "orbital_xc": recipe.orbital_xc,
        "recipe_sha256": recipe.fingerprint,
        "requested": settings["check_reference_stability"],
        "controls": {
            "internal": True,
            "external": True,
            "return_status": True,
            "nroots": settings["stability_nroots"],
            "tol": settings["stability_tolerance"],
        },
        "orbital_state_sha256_before": before,
        "checkpoint_sha256_before": checkpoint_before,
        "native_log": "pyscf.log",
        "error": None,
    }
    if settings["check_reference_stability"]:
        try:
            _, _, internal, external = mf.stability(**record["controls"])
            if not all(
                isinstance(value, (bool, np.bool_)) for value in (internal, external)
            ):
                raise ValueError("Native stability solver did not return both flags")
            record.update(
                status="stable" if internal and external else "unstable",
                internal_stable=bool(internal),
                restricted_to_unrestricted_stable=bool(external),
            )
        except (
            NotImplementedError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            record.update(
                status="unavailable",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
    after = _orbital_state_digest(mf)
    checkpoint_after = _file_hash(directory / "orbitals.chk")
    unchanged = before == after and checkpoint_before == checkpoint_after
    record.update(
        orbital_state_sha256_after=after,
        checkpoint_sha256_after=checkpoint_after,
        orbital_reference_unchanged=unchanged,
        returned_candidate_orbitals_applied=False,
    )
    if not unchanged:
        record.update(
            status="failed", error={"message": "Reference changed during diagnostic"}
        )
    path = directory / "reference-stability.json"
    _write_json(path, record)
    if not unchanged:
        raise ResearchCalculationError("Reference changed during stability diagnostic")
    return {**record, "artifact_sha256": _file_hash(path)}


def _summarize_reference_stability(parents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actual displaced diagnoses without inferring state continuity."""
    statuses = ("stable", "unstable", "unavailable", "not_requested")
    counts = {
        status: sum(
            parent["reference_stability_status"] == status for parent in parents
        )
        for status in statuses
    }
    if sum(counts.values()) != len(parents):
        raise ResearchCalculationError("A derivative parent lacks a stability status")
    status = (
        "unstable"
        if counts["unstable"]
        else "unavailable"
        if counts["unavailable"]
        else "not_requested"
        if counts["not_requested"]
        else "stable"
        if parents
        else "not_evaluated"
    )
    return {
        "status": status,
        "scope": "orbital-generating HF/GKS references at actually evaluated nodes",
        "evaluation_counts": counts,
        "evaluations": len(parents),
        "all_evaluations_stable_within_checked_scope": bool(parents)
        and counts["stable"] == len(parents),
        "full_correlated_energy_stability": "not_evaluated",
        "real_to_complex_status": "not_separately_extracted",
        "state_continuity": "not_qualified",
    }


def _derivative_steps(steps_bohr: Sequence[float]) -> tuple[float, ...]:
    steps = tuple(float(h) for h in steps_bohr)
    if (
        len(steps) < 2
        or len(set(steps)) != len(steps)
        or any(not np.isfinite(h) or h <= 0 for h in steps)
    ):
        raise ValueError(
            "At least two distinct finite positive displacement sizes are required"
        )
    return steps


def _derivative_budget(required: int, maximum: int) -> None:
    if type(maximum) is not int or maximum < 1:
        raise ValueError("The numerical derivative energy budget must be positive")
    if required > maximum:
        raise ResearchCalculationError(
            f"Numerical derivative needs {required} complete energy calculations; "
            f"the explicit budget permits {maximum}"
        )


def restricted_pt2_components(
    mol: Any,
    coefficients: np.ndarray,
    orbital_energies: np.ndarray,
    occupations: np.ndarray,
    *,
    frozen_occupied_orbitals: Sequence[int] = (),
    max_tensor_bytes: int = 1_000_000_000,
) -> dict[str, float]:
    """Conventional real-orbital OS/SS PT2 with unchanged GKS denominators.

    OS = sum (ia|jb)^2 / (eps_i + eps_j - eps_a - eps_b).
    SS = sum (ia|jb)[(ia|jb) - (ib|ja)] / the same denominator.
    Uses no HF conversion, density fitting, level shift or denominator repair.
    """
    from pyscf import ao2mo

    if np.iscomplexobj(orbital_energies) or np.iscomplexobj(occupations):
        raise ValueError("Real orbital energies and occupations are required")
    c = np.asarray(coefficients)
    eps = np.asarray(orbital_energies, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    if (
        np.iscomplexobj(c)
        or c.ndim != 2
        or eps.shape != occ.shape
        or c.shape[1] != len(eps)
    ):
        raise ValueError(
            "Consistent real restricted orbitals and orbital energies are required"
        )
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(eps)):
        raise ValueError("Nonfinite orbitals or energies")
    if not np.all(np.isin(occ, [0.0, 2.0])):
        raise ValueError("Only integer closed-shell occupations are supported")
    occupied = np.flatnonzero(occ == 2.0)
    virtual = np.flatnonzero(occ == 0.0)
    frozen = tuple(frozen_occupied_orbitals)
    if any(type(i) is not int or i not in occupied for i in frozen) or len(
        set(frozen)
    ) != len(frozen):
        raise ValueError("Frozen indices must identify distinct occupied orbitals")
    occupied = np.asarray([i for i in occupied if i not in frozen], dtype=int)
    if not len(occupied) or not len(virtual):
        raise ResearchCalculationError("PT2 needs active occupied and virtual orbitals")
    gap = float(np.min(eps[virtual]) - np.max(eps[occupied]))
    if gap <= 0:
        raise ResearchCalculationError(
            "The selected PT2 reference has a nonpositive orbital gap"
        )
    nocc, nvir = len(occupied), len(virtual)
    if nocc * nvir * nocc * nvir * 8 > max_tensor_bytes:
        raise ResearchCalculationError(
            "Conventional PT2 tensor exceeds the explicit memory budget"
        )
    co, cv = c[:, occupied], c[:, virtual]
    integrals = ao2mo.general(mol, (co, cv, co, cv), compact=False)
    integrals = np.asarray(integrals).reshape(nocc, nvir, nocc, nvir)
    eia = eps[occupied, None] - eps[None, virtual]
    os_energy, ss_energy = 0.0, 0.0
    for i in range(nocc):
        g = integrals[i].transpose(1, 0, 2)
        denominator = eia[:, None, :] + eia[i][None, :, None]
        if not np.all(np.isfinite(denominator)) or np.any(denominator >= 0):
            raise ResearchCalculationError(
                "Invalid PT2 denominators; no level shift is applied"
            )
        amplitudes = g / denominator
        os_energy += float(np.einsum("jab,jab->", amplitudes, g))
        ss_energy += float(np.einsum("jab,jab->", amplitudes, g - g.transpose(0, 2, 1)))
    if not np.isfinite(os_energy + ss_energy):
        raise ResearchCalculationError("Nonfinite conventional PT2 correlation")
    return {
        "opposite_spin_pt2_hartree": os_energy,
        "same_spin_pt2_hartree": ss_energy,
        "orbital_gap_hartree": gap,
    }


def resolved_d4_parameters(method: str) -> dict[str, Any]:
    """Resolve an explicit BJ-EEQ-ATM entry from the installed DFTD4 table."""
    from dftd4 import parameters

    resolved = parameters.get_damping_param(method.lower(), defaults=["bj-eeq-atm"])
    required = {"s6", "s8", "s9", "a1", "a2", "alp"}
    if required != set(resolved) or not all(np.isfinite(v) for v in resolved.values()):
        raise ResearchCalculationError("Incomplete D4 BJ-EEQ-ATM parametrization")
    return {
        "method": method,
        "variant": "BJ-EEQ-ATM",
        "charge_model": "EEQ",
        "parameters": resolved,
        "parameter_table_sha256": _file_hash(Path(parameters.get_data_file_name())),
    }


class ExperimentalDoubleHybrid:
    """A reproducible restricted nonrelativistic research evaluator.

    Numerical gradients include orbital response by rerunning the complete
    energy at every displacement. No analytic double-hybrid derivative is
    exposed. Results have qualification ``experimental_unqualified``.
    """

    def __init__(
        self,
        recipe: ResearchRecipe,
        *,
        basis: str,
        grid_level: int = 5,
        scf_energy_tolerance: float = 1e-12,
        scf_gradient_tolerance: float = 1e-8,
        max_cycle: int = 150,
        max_memory_mb: int = 2000,
        threads: int = 1,
        check_reference_stability: bool = True,
        stability_tolerance: float = 1e-4,
        stability_nroots: int = 3,
    ) -> None:
        if not basis or type(grid_level) is not int or not 0 <= grid_level <= 9:
            raise ValueError("Explicit basis and a valid grid level are required")
        if (
            not np.isfinite(scf_energy_tolerance)
            or scf_energy_tolerance <= 0
            or not np.isfinite(scf_gradient_tolerance)
            or scf_gradient_tolerance <= 0
            or isinstance(stability_tolerance, (bool, np.bool_))
            or not np.isfinite(stability_tolerance)
            or stability_tolerance <= 0
        ):
            raise ValueError("SCF and stability tolerances must be finite and positive")
        if any(
            type(x) is not int or x <= 0
            for x in (max_cycle, max_memory_mb, threads, stability_nroots)
        ):
            raise ValueError("Calculation limits must be positive integers")
        if type(check_reference_stability) is not bool:
            raise ValueError("check_reference_stability must be boolean")
        self.recipe = recipe
        self.settings: _ResearchSettings = {
            "basis": basis,
            "grid_level": grid_level,
            "scf_energy_tolerance": scf_energy_tolerance,
            "scf_gradient_tolerance": scf_gradient_tolerance,
            "max_cycle": max_cycle,
            "max_memory_mb": max_memory_mb,
            "threads": threads,
            "check_reference_stability": check_reference_stability,
            "stability_tolerance": stability_tolerance,
            "stability_nroots": stability_nroots,
        }

    def evaluate(
        self,
        symbols: Sequence[str],
        coordinates_bohr: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        artifact_directory: Path | str,
    ) -> dict[str, Any]:
        from pyscf import dft, gto, lib, scf

        if np.iscomplexobj(coordinates_bohr):
            raise ValueError("Real Cartesian coordinates are required")
        coords = np.asarray(coordinates_bohr, dtype=float)
        if (
            coords.shape != (len(symbols), 3)
            or not len(symbols)
            or not np.all(np.isfinite(coords))
        ):
            raise ValueError(
                "Finite atom-ordered N by 3 coordinates in bohr are required"
            )
        if (
            type(charge) is not int
            or type(multiplicity) is not int
            or multiplicity != 1
        ):
            raise ValueError(
                "This research implementation supports integer charge "
                "and multiplicity 1"
            )
        if any(gto.charge(s) < 1 or gto.charge(s) > 10 for s in symbols):
            raise ValueError(
                "Initial research domain contains first-row elements H through Ne only"
            )
        run_dir = Path(artifact_directory).resolve() / ("double-hybrid-" + uuid4().hex)
        run_dir.mkdir(parents=True)
        input_record = {
            "symbols": list(symbols),
            "coordinates": coords.tolist(),
            "coordinate_unit": "bohr",
            "charge": charge,
            "multiplicity": multiplicity,
            "recipe": asdict(self.recipe),
            "recipe_sha256": self.recipe.fingerprint,
            "settings": self.settings,
        }
        _write_json(run_dir / "input.json", input_record)
        try:
            with lib.with_omp_threads(self.settings["threads"]):
                mol = gto.M(
                    atom=list(zip(symbols, coords)),
                    unit="Bohr",
                    charge=charge,
                    spin=0,
                    basis=self.settings["basis"],
                    verbose=4,
                    output=str(run_dir / "pyscf.log"),
                    max_memory=self.settings["max_memory_mb"],
                )
                try:
                    if self.recipe.orbital_xc.upper() == "HF":
                        mf = scf.RHF(mol)
                    else:
                        mf = dft.RKS(mol)
                        mf.xc = self.recipe.orbital_xc
                        mf.grids.level = self.settings["grid_level"]
                    for expression in (self.recipe.orbital_xc, self.recipe.energy_xc):
                        if dft.libxc.xc_type(expression) not in ("HF", "LDA", "GGA"):
                            raise ValueError(
                                "Only HF and LDA/GGA hybrid expressions are supported"
                            )
                        if dft.libxc.rsh_coeff(expression)[0] != 0:
                            raise ValueError(
                                "Range-separated functionals are outside "
                                "this research domain"
                            )
                    mf.conv_tol = self.settings["scf_energy_tolerance"]
                    mf.conv_tol_grad = self.settings["scf_gradient_tolerance"]
                    mf.max_cycle = self.settings["max_cycle"]
                    mf.chkfile = str(run_dir / "orbitals.chk")
                    orbital_energy = float(mf.kernel())
                    if not mf.converged or not np.isfinite(orbital_energy):
                        raise ResearchCalculationError(
                            "Orbital-generating SCF failed to converge"
                        )
                    reference_stability = _reference_stability(
                        mf, self.settings, self.recipe, run_dir
                    )
                    density = mf.make_rdm1()
                    j, k = mf.get_jk(mol, density)
                    one = float(np.einsum("ij,ji->", mf.get_hcore(), density))
                    coulomb = float(0.5 * np.einsum("ij,ji->", j, density))
                    exchange = float(-0.25 * np.einsum("ij,ji->", k, density))
                    hybrid_scale = float(dft.libxc.hybrid_coeff(self.recipe.energy_xc))
                    if dft.libxc.xc_type(self.recipe.energy_xc) == "HF":
                        semilocal = (
                            0.0  # Exactly absent in the declared HF energy expression.
                        )
                    else:
                        grids = dft.gen_grid.Grids(mol)
                        grids.level = self.settings["grid_level"]
                        grids.build()
                        _, semilocal, _ = dft.numint.NumInt().nr_rks(
                            mol,
                            grids,
                            self.recipe.energy_xc,
                            density,
                            max_memory=self.settings["max_memory_mb"],
                        )
                        semilocal = float(semilocal)
                    components = restricted_pt2_components(
                        mol,
                        mf.mo_coeff,
                        mf.mo_energy,
                        mf.mo_occ,
                        frozen_occupied_orbitals=self.recipe.frozen_occupied_orbitals,
                        max_tensor_bytes=self.settings["max_memory_mb"]
                        * 1_000_000
                        // 2,
                    )
                    components.update(
                        {
                            "nuclear_repulsion_hartree": float(mol.energy_nuc()),
                            "one_electron_hartree": one,
                            "coulomb_hartree": coulomb,
                            "unscaled_exact_exchange_hartree": exchange,
                            "weighted_exact_exchange_hartree": hybrid_scale * exchange,
                            "semilocal_xc_hartree": semilocal,
                        }
                    )
                    components["hybrid_reference_hartree"] = (
                        components["nuclear_repulsion_hartree"]
                        + one
                        + coulomb
                        + hybrid_scale * exchange
                        + semilocal
                    )
                    components["scaled_pt2_hartree"] = (
                        self.recipe.opposite_spin_scale
                        * components["opposite_spin_pt2_hartree"]
                        + self.recipe.same_spin_scale
                        * components["same_spin_pt2_hartree"]
                    )
                    d4_record = None
                    dispersion = (
                        0.0  # Absent only when the declared recipe has no dispersion.
                    )
                    if self.recipe.d4_method is not None:
                        from dftd4.interface import DampingParam, DispersionModel

                        d4_record = resolved_d4_parameters(self.recipe.d4_method)
                        model = DispersionModel(
                            np.asarray(mol.atom_charges(), dtype=int),
                            coords,
                            charge=float(charge),
                        )
                        dispersion = float(
                            model.get_dispersion(
                                DampingParam(**d4_record["parameters"]),
                                grad=False,
                            )["energy"]
                        )
                    components["dispersion_hartree"] = dispersion
                    total = (
                        components["hybrid_reference_hartree"]
                        + components["scaled_pt2_hartree"]
                        + dispersion
                    )
                    if not np.all(
                        np.isfinite(list(components.values()))
                    ) or not np.isfinite(total):
                        raise ResearchCalculationError("Nonfinite energy component")
                    _write_json(run_dir / "basis.json", mol._basis)
                    result = {
                        "status": "success",
                        "qualification": "experimental_unqualified",
                        "method": "custom-research/" + self.recipe.name,
                        "recipe_sha256": self.recipe.fingerprint,
                        "energy_hartree": total,
                        "components": components,
                        "orbital_generating_energy_hartree": orbital_energy,
                        "scf_converged": True,
                        "reference_stability": reference_stability,
                        "quality_flags": []
                        if reference_stability["status"] == "stable"
                        else ["orbital_reference_" + reference_stability["status"]],
                        "gradient_availability": "explicit_numerical_only",
                        "hessian_availability": "explicit_numerical_only",
                        "independent_method_validation": "unavailable",
                        "dispersion": d4_record,
                        "artifact_directory": str(run_dir),
                        "provenance": {
                            "pyscf": importlib.metadata.version("pyscf"),
                            "libxc": dft.libxc.__version__,
                            "dftd4": importlib.metadata.version("dftd4")
                            if d4_record
                            else None,
                            "basis_sha256": _file_hash(run_dir / "basis.json"),
                            "pt2_approximation": (
                                "conventional_real_restricted_GKS_orbitals"
                            ),
                            "frozen_occupied_orbitals": list(
                                self.recipe.frozen_occupied_orbitals
                            ),
                            "effective_settings": self.settings,
                        },
                    }
                finally:
                    mol.stdout.close()
            result["artifact_hashes"] = {
                p.name: _file_hash(p) for p in run_dir.iterdir() if p.is_file()
            }
            _write_json(run_dir / "result.json", result)
            return result
        except Exception as exc:
            _write_json(
                run_dir / "failure.json",
                {"status": "failed", "error": str(exc), "type": type(exc).__name__},
            )
            raise

    def numerical_gradient(
        self,
        symbols: Sequence[str],
        coordinates_bohr: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        artifact_directory: Path | str,
        steps_bohr: Sequence[float],
        max_energy_calculations: int = 1000,
    ) -> dict[str, Any]:
        """Centered finite differences with explicit multi-step evidence."""
        steps = _derivative_steps(steps_bohr)
        if np.iscomplexobj(coordinates_bohr):
            raise ValueError("Real Cartesian coordinates are required")
        coords = np.asarray(coordinates_bohr, dtype=float)
        if (
            not len(symbols)
            or coords.shape != (len(symbols), 3)
            or not np.all(np.isfinite(coords))
        ):
            raise ValueError("Finite N by 3 coordinates in bohr are required")
        _derivative_budget(6 * len(symbols) * len(steps), max_energy_calculations)
        gradients, parents = [], []
        for step in steps:
            gradient = np.empty_like(coords)
            for atom in range(len(symbols)):
                for axis in range(3):
                    displaced = []
                    for sign in (1, -1):
                        geometry = coords.copy()
                        geometry[atom, axis] += sign * step
                        r = self.evaluate(
                            symbols,
                            geometry,
                            charge=charge,
                            multiplicity=multiplicity,
                            artifact_directory=artifact_directory,
                        )
                        displaced.append(r["energy_hartree"])
                        parents.append(
                            {
                                "atom": atom,
                                "axis": axis,
                                "step_bohr": sign * step,
                                "artifact_directory": r["artifact_directory"],
                                "result_sha256": _file_hash(
                                    Path(r["artifact_directory"]) / "result.json"
                                ),
                                "reference_stability_status": r["reference_stability"][
                                    "status"
                                ],
                                "reference_stability_artifact_sha256": r[
                                    "reference_stability"
                                ]["artifact_sha256"],
                            }
                        )
                    gradient[atom, axis] = (displaced[0] - displaced[1]) / (2 * step)
            gradients.append(gradient)
        chosen = int(np.argmin(steps))
        result = {
            "qualification": "experimental_unqualified",
            "derivative": "centered_numerical_energy_gradient",
            "unit": "hartree/bohr",
            "steps_bohr": list(steps),
            "gradient": gradients[chosen].tolist(),
            "selected_step_bohr": steps[chosen],
            "gradients_by_step": [g.tolist() for g in gradients],
            "max_absolute_step_difference": float(
                max(np.max(np.abs(g - gradients[chosen])) for g in gradients)
            ),
            "convergence_qualified": False,
            "reference_stability": _summarize_reference_stability(parents),
            "planned_energy_calculations": 6 * len(symbols) * len(steps),
            "energy_calculation_budget": max_energy_calculations,
            "executed_energy_calculations": len(parents),
            "parents": parents,
            "recipe_sha256": self.recipe.fingerprint,
        }
        _write_json(
            Path(artifact_directory) / ("numerical-gradient-" + uuid4().hex + ".json"),
            result,
        )
        return result

    def numerical_hessian(
        self,
        symbols: Sequence[str],
        coordinates_bohr: np.ndarray,
        *,
        charge: int,
        multiplicity: int,
        artifact_directory: Path | str,
        steps_bohr: Sequence[float],
        max_energy_calculations: int = 1000,
    ) -> dict[str, Any]:
        """Full Cartesian centered energy Hessian, with no harmonic qualification.

        Diagonals use (E(+h)-2E(0)+E(-h))/h**2; mixed derivatives use
        four complete energy evaluations at (+/-h,+/-h). Every requested
        scale is evaluated separately. The reference energy is shared across
        scales, and the exact cost is 1 + 2*(3*N)**2*number_of_scales.
        Symmetry is imposed by this stencil, so it is not a validation result.
        """
        steps = _derivative_steps(steps_bohr)
        if np.iscomplexobj(coordinates_bohr):
            raise ValueError("Real Cartesian coordinates are required")
        coords = np.asarray(coordinates_bohr, dtype=float)
        if (
            not len(symbols)
            or coords.shape != (len(symbols), 3)
            or not np.all(np.isfinite(coords))
        ):
            raise ValueError("Finite nonempty N by 3 coordinates in bohr are required")
        dimension = coords.size
        required = 1 + 2 * dimension**2 * len(steps)
        _derivative_budget(required, max_energy_calculations)
        directory = Path(artifact_directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        derivative_path = directory / ("numerical-hessian-" + uuid4().hex + ".json")
        parents: list[dict[str, Any]] = []
        gaps: list[float] = []

        def energy_at(displacements: Sequence[tuple[int, float]]) -> float:
            geometry = coords.copy().ravel()
            for coordinate, displacement in displacements:
                geometry[coordinate] += displacement
            run = self.evaluate(
                symbols,
                geometry.reshape(coords.shape),
                charge=charge,
                multiplicity=multiplicity,
                artifact_directory=directory,
            )
            parents.append(
                {
                    "displacements": [
                        {"atom": q // 3, "axis": q % 3, "step_bohr": delta}
                        for q, delta in displacements
                    ],
                    "artifact_directory": run["artifact_directory"],
                    "result_sha256": _file_hash(
                        Path(run["artifact_directory"]) / "result.json"
                    ),
                    "reference_stability_status": run["reference_stability"]["status"],
                    "reference_stability_artifact_sha256": run["reference_stability"][
                        "artifact_sha256"
                    ],
                }
            )
            gaps.append(run["components"]["orbital_gap_hartree"])
            return float(run["energy_hartree"])

        try:
            reference_energy = energy_at(())
            matrices, gradients = [], []
            for step in steps:
                matrix = np.empty((dimension, dimension))
                coordinate_gradient = np.empty(dimension)
                for q in range(dimension):
                    plus = energy_at(((q, step),))
                    minus = energy_at(((q, -step),))
                    matrix[q, q] = (plus - 2 * reference_energy + minus) / step**2
                    coordinate_gradient[q] = (plus - minus) / (2 * step)
                    for r in range(q):
                        mixed = 0.0
                        for sq, sr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                            mixed += (
                                sq * sr * energy_at(((q, sq * step), (r, sr * step)))
                            )
                        matrix[q, r] = matrix[r, q] = mixed / (4 * step**2)
                if not np.all(np.isfinite(matrix)) or not np.all(
                    np.isfinite(coordinate_gradient)
                ):
                    raise ResearchCalculationError("Nonfinite numerical derivative")
                matrices.append(matrix)
                gradients.append(coordinate_gradient.reshape(coords.shape))
            chosen = int(np.argmin(steps))
            matrix, gradient = matrices[chosen], gradients[chosen]
            centered = coords - coords.mean(axis=0)
            translation_residuals, rotation_residuals = [], []
            for axis in np.eye(3):
                translation = np.tile(axis, (len(symbols), 1)).ravel()
                rotation = np.cross(axis, centered).ravel()
                expected = np.cross(axis, gradient).ravel()
                translation_residuals.append(
                    float(np.max(np.abs(matrix @ translation)))
                )
                rotation_residuals.append(
                    float(np.max(np.abs(matrix @ rotation - expected)))
                )
            result = {
                "status": "success",
                "qualification": "experimental_unqualified",
                "derivative": "centered_numerical_energy_hessian",
                "unit": "hartree/bohr^2",
                "atom_order": list(symbols),
                "coordinates_bohr": coords.tolist(),
                "charge": charge,
                "multiplicity": multiplicity,
                "steps_bohr": list(steps),
                "selected_step_bohr": steps[chosen],
                "hessian": matrix.tolist(),
                "hessians_by_step": [m.tolist() for m in matrices],
                "gradients_by_step_hartree_bohr": [g.tolist() for g in gradients],
                "max_absolute_step_difference": float(
                    max(np.max(np.abs(m - matrix)) for m in matrices)
                ),
                "symmetry_by_construction": True,
                "invariance": {
                    "translation_residual_hartree_bohr2": translation_residuals,
                    "rotation_covariance_residual_hartree_bohr": rotation_residuals,
                    "net_gradient_hartree_bohr": gradient.sum(axis=0).tolist(),
                    "gradient_torque_hartree": np.cross(centered, gradient)
                    .sum(axis=0)
                    .tolist(),
                },
                "minimum_orbital_gap_hartree": min(gaps),
                "maximum_orbital_gap_hartree": max(gaps),
                "reference_stability": _summarize_reference_stability(parents),
                "state_continuity": "not_qualified",
                "convergence_qualified": False,
                "harmonic_characterization": "not_performed",
                "independent_method_validation": "unavailable",
                "planned_energy_calculations": required,
                "executed_energy_calculations": len(parents),
                "energy_calculation_budget": max_energy_calculations,
                "parents": parents,
                "recipe_sha256": self.recipe.fingerprint,
            }
            _write_json(derivative_path, result)
            return result
        except Exception as exc:
            _write_json(
                derivative_path,
                {
                    "status": "failed",
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "planned_energy_calculations": required,
                    "executed_energy_calculations": len(parents),
                    "parents": parents,
                    "recipe_sha256": self.recipe.fingerprint,
                },
            )
            raise
