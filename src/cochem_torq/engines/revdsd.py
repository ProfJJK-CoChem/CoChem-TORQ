"""Explicit, unqualified restricted double-hybrid research calculations.

This module implements conventional PT2 on the *specified* GKS orbitals. It
never constructs an alleged revDSD recipe from unrelated coefficients. The
exact revDSD-PBEP86-D4 entry remains closed until its complete definition and
independent reference evidence have been reconciled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np


class RecipeNotQualifiedError(ValueError):
    """The named scientific method has no accepted complete recipe."""


class ResearchCalculationError(RuntimeError):
    """A calculation did not supply a scientifically usable research result."""


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
    ) -> None:
        if not basis or type(grid_level) is not int or not 0 <= grid_level <= 9:
            raise ValueError("Explicit basis and a valid grid level are required")
        if (
            not np.isfinite(scf_energy_tolerance)
            or scf_energy_tolerance <= 0
            or not np.isfinite(scf_gradient_tolerance)
            or scf_gradient_tolerance <= 0
        ):
            raise ValueError("SCF tolerances must be finite and positive")
        if any(
            type(x) is not int or x <= 0 for x in (max_cycle, max_memory_mb, threads)
        ):
            raise ValueError("Calculation limits must be positive integers")
        self.recipe = recipe
        self.settings = {
            "basis": basis,
            "grid_level": grid_level,
            "scf_energy_tolerance": scf_energy_tolerance,
            "scf_gradient_tolerance": scf_gradient_tolerance,
            "max_cycle": max_cycle,
            "max_memory_mb": max_memory_mb,
            "threads": threads,
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
                "This research implementation supports integer charge and multiplicity 1"
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
                                "Range-separated functionals are outside this research domain"
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
                        "reference_stability": "not_evaluated",
                        "gradient_availability": "explicit_numerical_only",
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
                            "pt2_approximation": "conventional_real_restricted_GKS_orbitals",
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
    ) -> dict[str, Any]:
        """Two-sided finite differences with explicit multi-step convergence evidence."""
        steps = tuple(float(h) for h in steps_bohr)
        if (
            len(steps) < 2
            or len(set(steps)) != len(steps)
            or any(not np.isfinite(h) or h <= 0 for h in steps)
        ):
            raise ValueError(
                "At least two distinct finite positive displacement sizes are required"
            )
        coords = np.asarray(coordinates_bohr, dtype=float)
        if coords.shape != (len(symbols), 3) or not np.all(np.isfinite(coords)):
            raise ValueError("Finite N by 3 coordinates in bohr are required")
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
            "executed_energy_calculations": len(parents),
            "parents": parents,
            "recipe_sha256": self.recipe.fingerprint,
        }
        _write_json(
            Path(artifact_directory) / ("numerical-gradient-" + uuid4().hex + ".json"),
            result,
        )
        return result
