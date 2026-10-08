"""Explicit bounded native UHF/UKS observations without state projection.

This adapter retains the requested electron count and M_S sector. An unrestricted
determinant is not generally an S^2 eigenfunction: actual spin contamination and
native reference-stability observations remain visible. No lowest-state,
electronic-term, Hessian or spectroscopy qualification is inferred.
"""

from __future__ import annotations

import json
import platform
import shutil
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ..domain import ATOMIC_NUMBERS, Molecule
from .pyscf_backend import (
    BackendCalculationError,
    BackendInputError,
    PySCFBackend,
    _d4,
    _engine_fingerprint,
    _environment,
    _json,
    _mapping,
    _seal,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _real(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise BackendInputError(
            f"{label} must be real; complex values are unsupported."
        )
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise BackendInputError(f"{label} must be finite with shape {shape}.")
    return result


def _normalize(request: Any) -> dict[str, Any]:
    """Validate the explicit unrestricted tuple before starting any engine."""
    data = _mapping(request)
    allowed = {"molecule", "method", "properties", "settings", "optimization"}
    if set(data) - allowed:
        raise BackendInputError("Unsupported open-shell request fields.")
    molecule = _mapping(data.get("molecule", {}))
    symbols = molecule.get("symbols")
    if not isinstance(symbols, (tuple, list)) or not 1 <= len(symbols) <= 12:
        raise BackendInputError("The bounded open-shell profile requires 1–12 atoms.")
    if any(not isinstance(s, str) or s not in ATOMIC_NUMBERS for s in symbols):
        raise BackendInputError("Use canonical element symbols; isotopes are separate.")
    if any(ATOMIC_NUMBERS[s] > 10 for s in symbols):
        raise BackendInputError("Initial open-shell domain contains H through Ne only.")
    geometry = _real(molecule.get("geometry_bohr"), (len(symbols), 3), "geometry_bohr")
    charge, multiplicity = molecule.get("charge"), molecule.get("multiplicity")
    if type(charge) is not int or not -2 <= charge <= 2:
        raise BackendInputError(
            "Open-shell charge must be an explicit integer in [-2,2]."
        )
    if type(multiplicity) is not int or not 2 <= multiplicity <= 7:
        raise BackendInputError(
            "An explicit multiplicity 2–7 is required; broken-symmetry singlets "
            "need a separate state-preparation model."
        )
    molecule["symbols"] = list(symbols)
    molecule["geometry_bohr"] = geometry.tolist()
    try:
        molecular_model = Molecule.model_validate(molecule)
    except ValueError as exc:
        raise BackendInputError(
            "Invalid molecular electron/state identity: " + str(exc)
        ) from exc
    if molecular_model.isotopes is not None:
        from Libraries.cochem_isotopes import isotope_record

        for symbol, mass in zip(molecular_model.symbols, molecular_model.isotopes):
            if mass is not None:
                try:
                    isotope_record(f"{mass}{symbol}")
                except ValueError as exc:
                    raise BackendInputError(
                        "Unsupported requested isotope: " + str(exc)
                    ) from exc

    method = _mapping(data.get("method", {}))
    method_fields = {
        "name",
        "basis",
        "reference",
        "dispersion",
        "frozen_core",
        "density_fitting",
        "ecp",
        "solvent",
        "relativity",
        "settings",
    }
    if set(method) - method_fields:
        raise BackendInputError("Unsupported open-shell method-definition fields.")
    name = method.get("name")
    if not isinstance(name, str):
        raise BackendInputError("An explicit UHF, PBE or B3LYP method is required.")
    name = name.lower()
    dispersion = method.get("dispersion")
    if name.endswith("-d4"):
        name = name[:-3]
        if dispersion not in (None, "d4", "D4"):
            raise BackendInputError("Method suffix and explicit dispersion disagree.")
        dispersion = "d4"
    name = {"uhf": "hf"}.get(name, name)
    if name not in {"hf", "pbe", "b3lyp"}:
        raise BackendInputError(
            "Only explicit unrestricted HF/PBE/B3LYP are supported."
        )
    if dispersion in ("none", "None", False):
        dispersion = None
    if isinstance(dispersion, str):
        dispersion = dispersion.lower()
    if dispersion not in (None, "d4") or (dispersion and name == "hf"):
        raise BackendInputError("Named D4(BJ)-EEQ-ATM is supported for PBE/B3LYP only.")
    reference = method.get("reference")
    if reference not in ("unrestricted", "uhf" if name == "hf" else "uks"):
        raise BackendInputError("The matching unrestricted reference must be explicit.")
    basis = method.get("basis")
    if not isinstance(basis, str) or not basis.strip():
        raise BackendInputError("An explicitly named all-electron basis is required.")
    for key in ("frozen_core", "density_fitting", "ecp", "solvent", "relativity"):
        if method.get(key) not in (None, False, "none", "None"):
            raise BackendInputError(
                f"{key} is outside this unrestricted native domain."
            )
    properties = data.get("properties", ["energy", "gradient"])
    if (
        not isinstance(properties, (list, tuple))
        or not properties
        or any(not isinstance(p, str) for p in properties)
        or set(properties) - {"energy", "gradient"}
    ):
        raise BackendInputError(
            "Only energy/analytic-gradient properties are implemented; "
            "open-shell Hessians and response spectroscopy are unavailable."
        )
    settings = _mapping(method.get("settings", {}))
    outer = _mapping(data.get("settings", {}))
    if any(settings[key] != outer[key] for key in set(settings) & set(outer)):
        raise BackendInputError("Conflicting nested/top-level engine settings.")
    settings.update(outer)
    defaults: dict[str, Any] = {
        "threads": 1,
        "memory_mb": 1500,
        "scf_max_cycle": 150,
        "scf_energy_tolerance": 1e-11,
        "scf_gradient_tolerance": 1e-7,
        "dft_grid_level": 4,
        "check_stability": True,
        "stability_tolerance": 1e-4,
        "stability_nroots": 3,
        "spin_contamination_tolerance": 0.1,
    }
    if set(settings) - set(defaults):
        raise BackendInputError("Unsupported unrestricted numerical settings.")
    defaults.update(settings)
    for key, upper in (
        ("threads", 8),
        ("memory_mb", 32000),
        ("scf_max_cycle", 1000),
        ("dft_grid_level", 9),
        ("stability_nroots", 10),
    ):
        if type(defaults[key]) is not int or not 1 <= defaults[key] <= upper:
            raise BackendInputError(f"{key} must be an integer in [1,{upper}].")
    if type(defaults["check_stability"]) is not bool:
        raise BackendInputError("check_stability must be boolean.")
    for key in (
        "scf_energy_tolerance",
        "scf_gradient_tolerance",
        "stability_tolerance",
        "spin_contamination_tolerance",
    ):
        value = defaults[key]
        if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
            raise BackendInputError(f"{key} must be finite, real and positive.")
        try:
            valid = bool(np.isfinite(value) and value > 0)
        except TypeError as exc:
            raise BackendInputError(f"{key} must be numeric.") from exc
        if not valid:
            raise BackendInputError(f"{key} must be finite and positive.")
    return {
        "molecule": molecular_model.model_dump(mode="json"),
        "method": {
            "name": name,
            "basis": basis,
            "reference": "unrestricted",
            "dispersion": dispersion,
            "frozen_core": False,
            "density_fitting": False,
            "ecp": None,
            "solvent": None,
            "relativity": "none",
        },
        "properties": sorted(set(properties) | {"energy"}),
        "settings": defaults,
        "optimization": _mapping(data.get("optimization", {})),
    }


def _orbital_digest(mf: Any) -> str:
    digest = sha256()
    for name in ("mo_coeff", "mo_energy", "mo_occ"):
        array = np.asarray(getattr(mf, name))
        if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
            raise BackendCalculationError("Finite real unrestricted orbitals required.")
        digest.update(
            json.dumps(
                {"name": name, "shape": array.shape, "dtype": array.dtype.str}
            ).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def _stability(mf: Any, settings: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    from pyscf.scf import stability

    checkpoint = directory / "wavefunction.chk"
    before, checkpoint_before = _orbital_digest(mf), _hash(checkpoint)
    record: dict[str, Any] = {
        "status": "not_requested",
        "internal_stable": None,
        "external_stable": None,
        "internal_scope": "unrestricted real alpha/beta orbital variations",
        "external_scope": "native returned UHF/UKS -> GHF/GKS status only",
        "real_to_complex_status": "native_log_only_not_separately_extracted",
        "requested_state_preserved": True,
        "candidate_orbitals_applied": False,
        "electronic_branch_continuity": "not_qualified",
        "solver_source_sha256": _hash(Path(stability.__file__)),
        "controls": {
            "nroots": settings["stability_nroots"],
            "tol": settings["stability_tolerance"],
        },
        "orbital_sha256_before": before,
        "checkpoint_sha256_before": checkpoint_before,
        "diagnostics": {},
    }
    if settings["check_stability"]:
        for label in ("internal", "external"):
            try:
                returned = mf.stability(
                    internal=label == "internal",
                    external=label == "external",
                    return_status=True,
                    **record["controls"],
                )
                flag = returned[2 if label == "internal" else 3]
                if not isinstance(flag, (bool, np.bool_)):
                    raise BackendCalculationError(
                        "Native stability flag is unavailable."
                    )
                record[label + "_stable"] = bool(flag)
                record["diagnostics"][label] = {
                    "status": "stable" if flag else "unstable"
                }
            except (
                NotImplementedError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                record["diagnostics"][label] = {
                    "status": "unavailable",
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
        flags = (record["internal_stable"], record["external_stable"])
        record["status"] = (
            "unstable"
            if False in flags
            else "stable"
            if all(v is True for v in flags)
            else "unavailable"
        )
    after, checkpoint_after = _orbital_digest(mf), _hash(checkpoint)
    record.update(
        orbital_sha256_after=after,
        checkpoint_sha256_after=checkpoint_after,
        reference_unchanged=before == after and checkpoint_before == checkpoint_after,
    )
    _json(directory / "reference-stability.json", record)
    if not record["reference_unchanged"]:
        raise BackendCalculationError(
            "Native stability diagnosis changed the requested reference."
        )
    return record


@lru_cache(maxsize=1)
def _d4_fingerprint() -> dict[str, Any]:
    import dftd4
    from dftd4 import parameters

    root = Path(dftd4.__file__).parent
    files = {
        str(path.relative_to(root)): _hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and (path.suffix in {".py", ".toml"} or ".so" in path.name)
    }
    return {
        "version": version("dftd4"),
        "files": files,
        "digest": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "parameter_table_sha256": _hash(Path(parameters.get_data_file_name())),
    }


def _build(data: dict[str, Any], directory: Path) -> tuple[Any, Any, float]:
    from pyscf import dft, gto, scf

    molecule, method, settings = data["molecule"], data["method"], data["settings"]
    mol = gto.Mole()
    mol.atom = list(zip(molecule["symbols"], molecule["geometry_bohr"]))
    mol.unit, mol.charge, mol.spin = (
        "Bohr",
        molecule["charge"],
        molecule["multiplicity"] - 1,
    )
    mol.basis, mol.symmetry, mol.verbose = method["basis"], False, 4
    mol.max_memory, mol.output = settings["memory_mb"], str(directory / "pyscf.log")
    try:
        mol.build()
        mf = scf.UHF(mol) if method["name"] == "hf" else dft.UKS(mol)
        if method["name"] != "hf":
            mf.xc, mf.grids.level = method["name"], settings["dft_grid_level"]
        mf.conv_tol, mf.conv_tol_grad = (
            settings["scf_energy_tolerance"],
            settings["scf_gradient_tolerance"],
        )
        mf.max_cycle, mf.max_memory = settings["scf_max_cycle"], settings["memory_mb"]
        mf.chkfile = str(directory / "wavefunction.chk")
        if method["dispersion"] == "d4":
            from dftd4.pyscf import energy as d4_energy

            mf = d4_energy(mf)
            mf.with_dftd4.stdout = mol.stdout
        return mol, mf, float(mf.kernel())
    except Exception:
        if getattr(mol, "stdout", None) and not mol.stdout.closed:
            mol.stdout.close()
        raise


class OpenShellPySCFBackend:
    """Bounded UHF/UKS energy and analytic-gradient adapter, unprojected."""

    probe = staticmethod(PySCFBackend.probe)

    @staticmethod
    def capabilities() -> dict[str, Any]:
        return {
            "engine": "PySCF",
            "reference": "explicit unrestricted open shell",
            "methods": {
                name: {"energy": "analytic", "gradient": "analytic", "hessian": None}
                for name in ("hf", "pbe", "b3lyp")
            },
            "dispersion": {"pbe-d4": "D4(BJ)-EEQ-ATM", "b3lyp-d4": "D4(BJ)-EEQ-ATM"},
            "domain": {
                "max_atoms": 12,
                "elements": "H through Ne",
                "charge": [-2, 2],
                "multiplicity": [2, 7],
            },
            "unavailable": [
                "broken_symmetry_singlet",
                "spin_projection",
                "Hessian",
                "electronic_term_identification",
                "lowest_state_certification",
                "revDSD",
            ],
            "qualification": (
                "Experimental native observations; independent "
                "chemical accuracy uncalibrated."
            ),
        }

    def evaluate(self, request: Any, workspace: str | Path) -> dict[str, Any]:
        data = _normalize(request)
        directory = Path(workspace).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise BackendInputError(
                "Open-shell workspace must be empty; stale artifacts are forbidden."
            )
        _json(directory / "request.json", data)
        with (
            _environment(data["settings"], directory),
            (directory / "launcher.log").open("w") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            record = self._evaluate(data, directory)
        return _seal(directory, record)

    def _evaluate(self, data: dict[str, Any], directory: Path) -> dict[str, Any]:
        import pyscf
        from pyscf.dft import libxc

        record: dict[str, Any] = {
            "schema_version": "cochem-torq.pyscf-open-shell-result.v1",
            "engine": "PySCF",
            "engine_version": pyscf.__version__,
            "python_version": platform.python_version(),
            "libxc_version": libxc.__version__,
            "adapter_source_sha256": _hash(Path(__file__)),
            "shared_backend_source_sha256": _hash(
                Path(__file__).with_name("pyscf_backend.py")
            ),
            "engine_installation_sha256": _engine_fingerprint()["digest"],
            "method": data["method"],
            "settings": data["settings"],
            "molecule": data["molecule"],
            "geometry_bohr": data["molecule"]["geometry_bohr"],
            "status": "failed",
            "energy_hartree": None,
            "gradient_hartree_bohr": None,
            "hessian_hartree_bohr2": None,
            "dipole_debye": None,
            "unavailable_properties": {
                "hessian": {
                    "status": "unsupported",
                    "value": None,
                    "reason": (
                        "Unrestricted Hessian derivatives are not "
                        "validated in this adapter."
                    ),
                },
                "spin_projected_energy": {
                    "status": "unsupported",
                    "value": None,
                    "reason": "No Yamaguchi or other spin projection is applied.",
                },
            },
            "units": {
                "geometry": "bohr",
                "energy": "hartree",
                "gradient": "hartree/bohr",
                "spin_squared": "<S^2>/hbar^2, dimensionless",
            },
            "errors": [],
            "quality_flags": [
                "electronic_model_error_uncalibrated",
                "electronic_branch_continuity_unqualified",
            ],
            "independent_accuracy_validation": "unavailable",
            "requested_state_preserved": None,
            "state_definition": (
                "Requested electron count and alpha/beta M_S sector; "
                "no exact-S or electronic-term identification."
            ),
            "lowest_electronic_state_certified": False,
        }
        mol = None
        try:
            mol, mf, energy = _build(data, directory)
            _json(directory / "engine-installation.json", _engine_fingerprint())
            _json(directory / "basis-definition.json", mol._basis)
            record["basis_definition_sha256"] = _hash(
                directory / "basis-definition.json"
            )
            np.save(
                directory / "geometry-bohr.npy",
                mol.atom_coords(unit="Bohr"),
                allow_pickle=False,
            )
            expected_electrons = (
                sum(ATOMIC_NUMBERS[s] for s in data["molecule"]["symbols"])
                - data["molecule"]["charge"]
            )
            occupations = np.asarray(mf.mo_occ)
            actual_spin_counts = [int(np.sum(channel)) for channel in occupations]
            record["scf"] = {
                "converged": bool(mf.converged),
                "cycles": int(mf.cycles),
                "electron_count": int(mol.nelectron),
                "alpha_electrons": actual_spin_counts[0],
                "beta_electrons": actual_spin_counts[1],
                "requested_multiplicity": data["molecule"]["multiplicity"],
                "requested_two_Ms": data["molecule"]["multiplicity"] - 1,
                "native_molecular_spin": int(mol.spin),
                "energy_hartree": energy,
            }
            if (
                int(mol.nelectron) != expected_electrons
                or not np.all(np.isin(occupations, [0.0, 1.0]))
                or sum(actual_spin_counts) != expected_electrons
                or actual_spin_counts[0] - actual_spin_counts[1] != mol.spin
                or mol.spin != data["molecule"]["multiplicity"] - 1
            ):
                raise BackendCalculationError(
                    "Native alpha/beta occupations changed the requested state."
                )
            record["requested_state_preserved"] = True
            if not mf.converged or not np.isfinite(energy):
                raise BackendCalculationError(
                    "Unrestricted SCF did not converge to a finite energy."
                )
            record["energy_hartree"] = energy
            spin2, inferred_multiplicity = (float(value) for value in mf.spin_square())
            if not np.isfinite(spin2) or not np.isfinite(inferred_multiplicity):
                raise BackendCalculationError(
                    "Native spin-square expectation is unavailable."
                )
            requested_spin = mol.spin / 2
            expected_spin2 = requested_spin * (requested_spin + 1)
            record["spin"] = {
                "definition": "Unprojected unrestricted determinant expectation <S^2>",
                "spin_squared": spin2,
                "requested_spin_squared": expected_spin2,
                "spin_contamination": spin2 - expected_spin2,
                "effective_multiplicity_from_spin_expectation": inferred_multiplicity,
                "requested_multiplicity": data["molecule"]["multiplicity"],
                "effective_multiplicity_is_state_label": False,
                "spin_projection_applied": False,
                "declared_contamination_tolerance": data["settings"][
                    "spin_contamination_tolerance"
                ],
            }
            _json(directory / "spin-diagnostics.json", record["spin"])
            if (
                abs(spin2 - expected_spin2)
                > data["settings"]["spin_contamination_tolerance"]
            ):
                record["quality_flags"].append(
                    "spin_contamination_exceeds_declared_tolerance"
                )
            if data["method"]["name"] != "hf":
                parsed = libxc.parse_xc(mf.xc)
                record["density_functional"] = {
                    "libxc_alias": mf.xc,
                    "resolved_components": [[int(c), float(w)] for c, w in parsed[1]],
                    "hybrid_coefficients": list(parsed[0]),
                    "references": list(libxc.xc_reference(mf.xc)),
                    "gradient_grid_response": True,
                }
            if data["method"]["dispersion"] == "d4":
                record["dispersion"] = _d4(data, np.asarray(record["geometry_bohr"]))
                record["dispersion"]["implementation"] = _d4_fingerprint()
                _json(directory / "dftd4-result.json", record["dispersion"])
                record["scf"]["electronic_energy_excluding_dispersion_hartree"] = (
                    energy - record["dispersion"]["energy_hartree"]
                )
            elif data["method"]["name"] != "hf":
                record["quality_flags"].append(
                    "bare_dft_validation_profile_not_dispersion_compliant_production_recipe"
                )
            record["stability"] = _stability(mf, data["settings"], directory)
            if record["stability"]["status"] != "stable":
                record["quality_flags"].append(
                    "unrestricted_reference_stability_" + record["stability"]["status"]
                )
            if "gradient" in data["properties"]:
                gradient_method = mf.nuc_grad_method()
                if data["method"]["name"] != "hf":
                    gradient_method.grid_response = True
                gradient = _real(
                    gradient_method.kernel(), (mol.natm, 3), "native gradient"
                )
                record["gradient_hartree_bohr"] = gradient.tolist()
                record["gradient_definition"] = (
                    "Analytic nuclear derivative of the unprojected declared "
                    "unrestricted energy, including actual D4 when requested."
                )
                record["gradient_translation_residual"] = float(
                    np.linalg.norm(gradient.sum(axis=0))
                )
                np.save(
                    directory / "gradient-hartree-bohr.npy",
                    gradient,
                    allow_pickle=False,
                )
            record["status"] = "complete"
            if (
                data["settings"]["check_stability"]
                and record["stability"]["status"] != "stable"
            ):
                record["status"] = "partial"
                record["errors"].append(
                    {
                        "code": "UNRESTRICTED_STATE_NOT_QUALIFIED",
                        "message": (
                            "Requested reference stability is "
                            + record["stability"]["status"]
                            + "; actual unprojected observations are retained."
                        ),
                    }
                )
        except (
            BackendCalculationError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            record["errors"].append(
                {"code": "OPEN_SHELL_CALCULATION_FAILED", "message": str(exc)}
            )
            if record["energy_hartree"] is not None:
                record["status"] = "partial"
        finally:
            if (
                mol is not None
                and getattr(mol, "stdout", None)
                and not mol.stdout.closed
            ):
                mol.stdout.flush()
                mol.stdout.close()
        return record

    def optimize(self, request: Any, workspace: str | Path) -> dict[str, Any]:
        """Optimize the declared unprojected state with genuine geomeTRIC.

        Actual initial, per-evaluation and final native evidence is retained.
        Optimizer convergence does not classify a minimum without a validated
        Hessian, and fixed alpha/beta counts do not identify an electronic term.
        """
        data = _normalize(request)
        parameters: dict[str, Any] = {
            "maxsteps": 80,
            "convergence_energy": 1e-8,
            "convergence_grms": 1e-5,
            "convergence_gmax": 1.5e-5,
            "convergence_drms": 1e-4,
            "convergence_dmax": 1.5e-4,
        }
        if set(data["optimization"]) - set(parameters):
            raise BackendInputError("Unsupported unrestricted optimizer parameters.")
        parameters.update(data["optimization"])
        for key, value in parameters.items():
            if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
                raise BackendInputError(f"Optimizer {key} must be real and positive.")
            try:
                valid = bool(np.isfinite(value) and value > 0)
            except TypeError as exc:
                raise BackendInputError(f"Optimizer {key} must be numeric.") from exc
            if not valid:
                raise BackendInputError(f"Optimizer {key} must be finite and positive.")
        if type(parameters["maxsteps"]) is not int or parameters["maxsteps"] > 300:
            raise BackendInputError("Optimizer maxsteps must be an integer in [1,300].")
        directory = Path(workspace).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise BackendInputError("Open-shell optimization workspace must be empty.")
        _json(directory / "request.json", data)
        trajectory: list[dict[str, Any]] = []
        mol = None
        initial_reference: dict[str, Any] | None = None
        record: dict[str, Any] = {
            "schema_version": "cochem-torq.pyscf-open-shell-result.v1",
            "engine": "PySCF",
            "engine_version": version("pyscf"),
            "adapter_source_sha256": _hash(Path(__file__)),
            "method": data["method"],
            "molecule": data["molecule"],
            "status": "failed",
            "energy_hartree": None,
            "geometry_bohr": None,
            "gradient_hartree_bohr": None,
            "hessian_hartree_bohr2": None,
            "errors": [],
            "optimization": {"converged": False},
        }
        with (
            _environment(data["settings"], directory),
            (directory / "launcher.log").open("w") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            try:
                from pyscf import dft, scf
                from pyscf.geomopt import geometric_solver

                initial_directory = directory / "initial"
                initial_directory.mkdir()
                initial_data = deepcopy(data)
                initial_data["properties"] = ["energy", "gradient"]
                _json(initial_directory / "request.json", initial_data)
                initial_result = self._evaluate(initial_data, initial_directory)
                _seal(initial_directory, initial_result)
                initial_reference = {
                    "artifact_directory": str(initial_directory),
                    "result_sha256": _hash(initial_directory / "result.json"),
                    "manifest_sha256": _hash(initial_directory / "manifest.json"),
                    "status": initial_result["status"],
                }
                if initial_result["status"] != "complete":
                    raise BackendCalculationError(
                        "Initial unrestricted calculation failed."
                    )
                if (
                    data["settings"]["check_stability"]
                    and initial_result["stability"]["status"] != "stable"
                ):
                    raise BackendCalculationError(
                        "Initial requested unrestricted state "
                        "is not stably characterized."
                    )
                initial_hashes = {
                    path.name: _hash(path)
                    for path in initial_directory.iterdir()
                    if path.is_file()
                }
                checkpoint = initial_directory / "wavefunction.chk"
                saved = scf.chkfile.load(str(checkpoint), "scf")
                mol = scf.chkfile.load_mol(str(checkpoint))
                mol.output = str(directory / "optimization-native.log")
                mol.build(False, False)
                if (
                    mol.spin != data["molecule"]["multiplicity"] - 1
                    or mol.charge != data["molecule"]["charge"]
                ):
                    raise BackendCalculationError(
                        "Initial native checkpoint state differs."
                    )
                settings, method = data["settings"], data["method"]
                mf = scf.UHF(mol) if method["name"] == "hf" else dft.UKS(mol)
                if method["name"] != "hf":
                    mf.xc, mf.grids.level = method["name"], settings["dft_grid_level"]
                mf.conv_tol, mf.conv_tol_grad = (
                    settings["scf_energy_tolerance"],
                    settings["scf_gradient_tolerance"],
                )
                mf.max_cycle, mf.max_memory = (
                    settings["scf_max_cycle"],
                    settings["memory_mb"],
                )
                if method["dispersion"] == "d4":
                    from dftd4.pyscf import energy as d4_energy

                    mf = d4_energy(mf)
                    mf.with_dftd4.stdout = mol.stdout
                mf.mo_coeff, mf.mo_energy, mf.mo_occ = (
                    saved["mo_coeff"],
                    saved["mo_energy"],
                    saved["mo_occ"],
                )
                mf.e_tot, mf.converged = (
                    saved["e_tot"],
                    initial_result["scf"]["converged"],
                )
                mf.chkfile = str(directory / "optimization-wavefunction.chk")
                gradient_method = mf.nuc_grad_method()
                if method["name"] != "hf":
                    gradient_method.grid_response = True
                step_directory = directory / "optimization-evaluations"
                step_directory.mkdir()

                def callback(environment: dict[str, Any]) -> None:
                    scanner = environment["g_scanner"]
                    native = scanner.base
                    occupations = np.asarray(native.mo_occ)
                    counts = [int(np.sum(channel)) for channel in occupations]
                    if (
                        counts[0] - counts[1] != data["molecule"]["multiplicity"] - 1
                        or native.mol.charge != data["molecule"]["charge"]
                    ):
                        raise BackendCalculationError(
                            "Optimization changed the requested state."
                        )
                    cycle = int(environment["self"].cycle)
                    destination = step_directory / f"evaluation-{cycle:04d}"
                    destination.mkdir()
                    native_checkpoint = Path(native.chkfile)
                    shutil.copyfile(native_checkpoint, destination / "wavefunction.chk")
                    spin_squared, effective_multiplicity = native.spin_square()
                    snapshot = {
                        "evaluation": cycle,
                        "geometry_bohr": np.asarray(environment["coords"]).tolist(),
                        "energy_hartree": float(environment["energy"]),
                        "gradient_hartree_bohr": np.asarray(
                            environment["gradients"]
                        ).tolist(),
                        "scf_converged": bool(scanner.converged),
                        "alpha_electrons": counts[0],
                        "beta_electrons": counts[1],
                        "requested_multiplicity": data["molecule"]["multiplicity"],
                        "spin_squared": float(spin_squared),
                        "effective_multiplicity_from_spin_expectation": float(
                            effective_multiplicity
                        ),
                        "spin_projection_applied": False,
                        "checkpoint_sha256": _hash(destination / "wavefunction.chk"),
                        "checkpoint_artifact_directory": str(destination),
                    }
                    _json(destination / "observation.json", snapshot)
                    trajectory.append(snapshot)

                with (directory / "geometric.log").open("w") as geometric_log:
                    with redirect_stdout(geometric_log), redirect_stderr(geometric_log):
                        converged, final_molecule = geometric_solver.kernel(
                            gradient_method,
                            assert_convergence=True,
                            callback=callback,
                            **parameters,
                        )
                if any(
                    _hash(initial_directory / name) != digest
                    for name, digest in initial_hashes.items()
                ):
                    raise BackendCalculationError(
                        "Optimization changed an initial native artifact."
                    )
                final_data = deepcopy(data)
                final_data["molecule"]["geometry_bohr"] = final_molecule.atom_coords(
                    unit="Bohr"
                ).tolist()
                final_data["properties"] = ["energy", "gradient"]
                final_directory = directory / "final"
                final_directory.mkdir()
                _json(final_directory / "request.json", final_data)
                record = self._evaluate(final_data, final_directory)
                _seal(final_directory, record)
                gradient_ok = False
                if record["gradient_hartree_bohr"] is not None:
                    gradient = np.asarray(record["gradient_hartree_bohr"])
                    gradient_ok = bool(
                        np.sqrt(np.mean(gradient**2)) <= parameters["convergence_grms"]
                        and np.max(np.abs(gradient)) <= parameters["convergence_gmax"]
                    )
                stability_ok = (
                    not settings["check_stability"]
                    or record.get("stability", {}).get("status") == "stable"
                )
                record["optimization"] = {
                    "engine": "geomeTRIC",
                    "version": version("geometric"),
                    "converged": bool(converged),
                    "final_gradient_verified": gradient_ok,
                    "final_reference_stability_passed": stability_ok,
                    "parameters": parameters,
                    "evaluations": len(trajectory),
                    "initial_geometry_bohr": data["molecule"]["geometry_bohr"],
                    "initial_artifacts_unchanged": True,
                    "initial_result": initial_reference,
                    "final_geometry_source": (
                        "Actual geomeTRIC coordinates, fresh final "
                        "unrestricted energy/gradient evaluation."
                    ),
                    "stationary_point": (
                        "unclassified_until_validated_unrestricted_hessian"
                    ),
                    "electronic_branch_continuity": "not_qualified",
                    "requested_state_preserved": True,
                    "spin_projection_applied": False,
                }
                if not converged or not gradient_ok or not stability_ok:
                    record["status"] = (
                        "partial" if record["energy_hartree"] is not None else "failed"
                    )
                    record["errors"].append(
                        {
                            "code": "OPEN_SHELL_OPTIMIZATION_NOT_QUALIFIED",
                            "message": (
                                "Native convergence, final gradient and "
                                "requested stability gates must pass."
                            ),
                        }
                    )
            except (
                BackendCalculationError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                record["errors"].append(
                    {"code": "OPEN_SHELL_OPTIMIZATION_FAILED", "message": str(exc)}
                )
                record["initial_valid_result"] = initial_reference
            finally:
                if (
                    mol is not None
                    and getattr(mol, "stdout", None)
                    and not mol.stdout.closed
                ):
                    mol.stdout.flush()
                    mol.stdout.close()
                _json(directory / "optimization-trajectory.json", trajectory)
        record.pop("artifacts", None)
        record.pop("manifest_path", None)
        record.pop("manifest_sha256", None)
        return _seal(directory, record)
