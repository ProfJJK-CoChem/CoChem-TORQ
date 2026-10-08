"""Real, bounded CPU PySCF calculations for the open-source TORQ profile.

The adapter never substitutes an engine or method. Nuclear coordinates,
energies, gradients and Cartesian Hessians use atomic units. Dipoles are
reported in debye in the input Cartesian frame, with origin at (0, 0, 0).
DFT and MP2 Hessians are explicitly numerical derivatives of full analytic gradients;
their displacement calculations and step-convergence evidence are retained.
No MP2 dipole is advertised: an unrelaxed density expectation is not silently
substituted for a correlated response property.

PySCF and geomeTRIC are imported only when called, so domain/UI imports do not
require a quantum engine. Each backend call needs an isolated process if
concurrent execution is desired: PySCF thread/scratch settings are global.
"""

from __future__ import annotations

import json
import platform
import re
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


class BackendInputError(ValueError):
    """A requested molecular state, method or property is unsupported."""


class BackendCalculationError(RuntimeError):
    """An actual calculation failed a numerical or scientific check."""


_METHODS = {"hf", "pbe", "b3lyp", "mp2"}
_PROPERTIES = {"energy", "gradient", "hessian", "dipole"}
_ALIASES = {"rhf": "hf", "rmp2": "mp2"}
_SYMBOL = re.compile(r"^(?:[0-9]+)?([A-Z][a-z]?)$")
_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _engine_fingerprint() -> dict[str, Any]:
    """Hash the installed engine's Python implementation and native libraries."""
    import pyscf

    root = Path(pyscf.__file__).parent
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix == ".py" or ".so" in path.name):
            hashes[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
    summary = sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "scheme": "sha256 installed PySCF Python and native shared-library files",
        "digest": summary,
        "files": hashes,
    }


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise BackendInputError("Request must be a mapping or a typed model.")
    return dict(value)


def _finite(array: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(array, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise BackendInputError(f"{label} must have shape {shape} and finite values.")
    return result


def _normalize(request: Any) -> dict[str, Any]:
    """Validate the exact tuple before acquiring computational resources."""
    data = _mapping(request)
    molecule = _mapping(data.get("molecule", {}))
    raw_symbols = molecule.get("symbols")
    if not isinstance(raw_symbols, (list, tuple)) or not raw_symbols:
        raise BackendInputError("A nonempty symbols list is required.")
    symbols = []
    for symbol in raw_symbols:
        match = _SYMBOL.fullmatch(str(symbol))
        if not match:
            raise BackendInputError(f"Unsupported atom symbol {symbol!r}.")
        symbols.append(match.group(1))
    geometry = _finite(
        molecule.get("geometry_bohr"), (len(symbols), 3), "geometry_bohr"
    )
    if len(symbols) > 1:
        distance = np.linalg.norm(geometry[:, None, :] - geometry[None, :, :], axis=2)
        distance[np.diag_indices_from(distance)] = np.inf
        if np.min(distance) < 1e-6:
            raise BackendInputError(
                "Coincident nuclei are not a molecular calculation."
            )
    charge = molecule.get("charge", 0)
    multiplicity = molecule.get("multiplicity", 1)
    if isinstance(charge, bool) or not isinstance(charge, int):
        raise BackendInputError("Charge must be an integer.")
    if multiplicity != 1 or isinstance(multiplicity, bool):
        raise BackendInputError(
            "This CPU profile currently supports closed-shell singlets only."
        )

    method = _mapping(data.get("method", {}))
    method_keys = {
        "name",
        "basis",
        "reference",
        "frozen_core",
        "dispersion",
        "ecp",
        "solvent",
        "density_fitting",
        "relativity",
        "settings",
    }
    if set(method) - method_keys:
        raise BackendInputError(
            "Unsupported method-definition fields: "
            f"{sorted(set(method) - method_keys)}."
        )
    name = str(method.get("name", "")).lower()
    dispersion = method.get("dispersion")
    if name.endswith("-d4"):
        name = name[:-3]
        if dispersion not in (None, "D4", "d4"):
            raise BackendInputError("Method suffix and requested dispersion disagree.")
        dispersion = "d4"
    if dispersion in ("none", "None", False):
        dispersion = None
    if isinstance(dispersion, str):
        dispersion = dispersion.lower()
    if dispersion not in (None, "d4"):
        raise BackendInputError(
            "Only the explicitly named D4(BJ)-EEQ-ATM dispersion model is implemented."
        )
    name = _ALIASES.get(name, name)
    if name not in _METHODS:
        raise BackendInputError(
            f"Unsupported PySCF method {name!r}; supported: {sorted(_METHODS)}."
        )
    if dispersion and name not in ("pbe", "b3lyp"):
        raise BackendInputError(
            "D4 is implemented for PBE and B3LYP, with their "
            "library-resolved parameters."
        )
    basis = method.get("basis")
    if not isinstance(basis, str) or not basis.strip():
        raise BackendInputError("An explicitly named basis is required.")
    reference = str(method.get("reference", "restricted")).lower()
    if reference not in ("restricted", "rhf", "rks"):
        raise BackendInputError(
            "Only restricted closed-shell references are implemented."
        )
    if reference == "rks" and name not in ("pbe", "b3lyp"):
        raise BackendInputError("An RKS reference does not define this HF/MP2 recipe.")
    if reference == "rhf" and name in ("pbe", "b3lyp"):
        raise BackendInputError(
            "An RHF reference does not define this Kohn-Sham DFT recipe."
        )
    if method.get("frozen_core", False) is not False:
        raise BackendInputError(
            "This qualified profile uses all-electron, unfrozen-core MP2."
        )
    forbidden = {"ecp", "solvent", "density_fitting", "relativity"}
    for key in forbidden:
        if key in method and method[key] not in (None, False, "none", "None"):
            raise BackendInputError(
                f"{key} requires a separately implemented method profile."
            )
    properties = data.get("properties", ["energy", "gradient"])
    if not isinstance(properties, (list, tuple)) or not properties:
        raise BackendInputError("properties must be a nonempty list.")
    properties = sorted(set(properties) | {"energy"})
    unknown = set(properties) - _PROPERTIES
    if unknown:
        raise BackendInputError(f"Unsupported requested properties: {sorted(unknown)}.")
    if name == "mp2" and "dipole" in properties:
        raise BackendInputError(
            "A validated relaxed-response MP2 dipole is not implemented."
        )
    settings = _mapping(method.get("settings", {}))
    outer_settings = _mapping(data.get("settings", {}))
    for key in set(settings) & set(outer_settings):
        if settings[key] != outer_settings[key]:
            raise BackendInputError(f"Conflicting nested/top-level setting {key!r}.")
    settings.update(outer_settings)
    defaults = {
        "threads": 1,
        "memory_mb": 1500,
        "scf_max_cycle": 100,
        "scf_energy_tolerance": 1e-11,
        "scf_gradient_tolerance": 1e-7,
        "dft_grid_level": 4,
        "check_stability": True,
        "hessian_step_bohr": 0.002,
        "hessian_convergence_tolerance": 5e-5,
        "hessian_symmetry_tolerance": 5e-6,
    }
    for key in settings:
        if key not in defaults:
            raise BackendInputError(f"Unknown PySCF setting {key!r}.")
    defaults.update(settings)
    for key in ("threads", "memory_mb", "scf_max_cycle", "dft_grid_level"):
        value = defaults[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise BackendInputError(f"{key} must be a positive integer.")
    if defaults["threads"] > 8 or defaults["memory_mb"] > 32000:
        raise BackendInputError("Resource request exceeds the bounded CPU profile.")
    for key in (
        "scf_energy_tolerance",
        "scf_gradient_tolerance",
        "hessian_step_bohr",
        "hessian_convergence_tolerance",
        "hessian_symmetry_tolerance",
    ):
        if not np.isfinite(defaults[key]) or defaults[key] <= 0:
            raise BackendInputError(f"{key} must be finite and positive.")
    if not isinstance(defaults["check_stability"], bool):
        raise BackendInputError("check_stability must be boolean.")
    return {
        "molecule": {
            "symbols": symbols,
            "isotope_symbols": list(raw_symbols),
            "atom_ids": molecule.get("atom_ids"),
            "isotopes": molecule.get("isotopes"),
            "geometry_bohr": geometry.tolist(),
            "charge": charge,
            "multiplicity": 1,
        },
        "method": {
            "name": name,
            "basis": basis,
            "reference": "restricted",
            "frozen_core": False,
            "density_fitting": False,
            "dispersion": dispersion,
        },
        "properties": properties,
        "settings": defaults,
        "optimization": _mapping(data.get("optimization", {})),
    }


@contextmanager
def _environment(settings: Mapping[str, Any], directory: Path):
    from pyscf import lib

    old_threads = lib.num_threads()
    old_tmp = lib.param.TMPDIR
    scratch = directory / "scratch"
    scratch.mkdir()
    lib.num_threads(settings["threads"])
    lib.param.TMPDIR = str(scratch)
    try:
        yield
    finally:
        lib.num_threads(old_threads)
        lib.param.TMPDIR = old_tmp


def _build(data: dict[str, Any], directory: Path):
    from pyscf import dft, gto, mp, scf

    molecule, method, settings = data["molecule"], data["method"], data["settings"]
    mol = gto.Mole()
    mol.atom = list(zip(molecule["symbols"], molecule["geometry_bohr"]))
    mol.unit = "Bohr"
    mol.charge = molecule["charge"]
    mol.spin = 0
    mol.basis = method["basis"]
    mol.symmetry = False
    mol.verbose = 4
    mol.max_memory = settings["memory_mb"]
    mol.output = str(directory / "pyscf.log")
    mol.build()
    if mol.nelectron <= 0 or mol.nelectron % 2:
        mol.stdout.close()
        raise BackendInputError(
            "The requested singlet must contain a positive even number of electrons."
        )
    if method["name"] in ("pbe", "b3lyp"):
        mf = dft.RKS(mol)
        mf.xc = method["name"]
        mf.grids.level = settings["dft_grid_level"]
        # Full grid response is required for the derivative of moving atom grids.
    else:
        mf = scf.RHF(mol)
    mf.conv_tol = settings["scf_energy_tolerance"]
    mf.conv_tol_grad = settings["scf_gradient_tolerance"]
    mf.max_cycle = settings["scf_max_cycle"]
    mf.chkfile = str(directory / "wavefunction.chk")
    if method["dispersion"] == "d4":
        from dftd4.pyscf import energy as d4_energy

        mf = d4_energy(mf)
        # The upstream dispersion logger otherwise falls back to sys.__stdout__,
        # contaminating structured CLI output despite Python redirection.
        mf.with_dftd4.stdout = mol.stdout
    energy = float(mf.kernel())
    calculator = mf
    if mf.converged and method["name"] == "mp2":
        calculator = mp.MP2(mf, frozen=None)
        calculator.kernel()
        energy = float(calculator.e_tot)
    return mol, mf, calculator, energy


def _stability(mf: Any, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "not_requested",
            "internal_stable": None,
            "external_stable": None,
            "qualification": "wavefunction_stability_unchecked",
        }
    _, _, internal, external = mf.stability(
        internal=True, external=True, return_status=True
    )
    return {
        "status": "stable" if internal and external else "unstable",
        "internal_stable": bool(internal),
        "external_stable": bool(external),
        "scope": "restricted real-orbital internal and unrestricted external stability",
    }


def _d4(data: dict[str, Any], coordinates: np.ndarray) -> dict[str, Any]:
    """Evaluate actual D4 and resolve every inherited damping parameter."""
    from dftd4.interface import DampingParam, DispersionModel
    from dftd4.parameters import get_damping_param
    from pyscf import gto

    name = data["method"]["name"]
    parameters = get_damping_param(name, defaults=["bj-eeq-atm"])
    molecule = data["molecule"]
    model = DispersionModel(
        np.asarray([gto.charge(symbol) for symbol in molecule["symbols"]]),
        coordinates,
        charge=molecule["charge"],
        model="d4",
    )
    result = model.get_dispersion(DampingParam(method=name, atm=True), grad=True)
    return {
        "energy_hartree": float(result["energy"]),
        "gradient_hartree_bohr": np.asarray(result["gradient"]).tolist(),
        "parameters": parameters,
        "variant": "D4(BJ)-EEQ-ATM",
        "library_version": version("dftd4"),
        "charge": molecule["charge"],
    }


def _seal(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Publish hashes after actual raw files and parsed results are durable."""
    _json(directory / "result.json", record)
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "scratch" in path.relative_to(directory).parts:
            continue
        if path.name == "manifest.json":
            continue
        relative = str(path.relative_to(directory))
        artifacts.append(
            {
                "path": relative,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "cochem-torq.engine-artifacts.v1",
        "artifacts": artifacts,
    }
    _json(directory / "manifest.json", manifest)
    record["artifacts"] = manifest
    record["manifest_path"] = str(directory / "manifest.json")
    record["manifest_sha256"] = sha256(
        (directory / "manifest.json").read_bytes()
    ).hexdigest()
    return record


class PySCFBackend:
    """All-electron closed-shell HF, named DFT and canonical MP2 CPU adapter."""

    @staticmethod
    def probe() -> dict[str, Any]:
        try:
            import pyscf
            from pyscf.dft import libxc

            dependencies = {}
            for name in ("geometric", "dftd4"):
                try:
                    dependencies[name] = {"available": True, "version": version(name)}
                except PackageNotFoundError:
                    dependencies[name] = {"available": False, "version": None}
            return {
                "available": True,
                "engine": "PySCF",
                "version": pyscf.__version__,
                "libxc_version": libxc.libxc_version(),
                "dependencies": dependencies,
                "geometric_version": dependencies["geometric"]["version"],
                "dftd4_version": dependencies["dftd4"]["version"],
            }
        except ImportError as exc:
            return {"available": False, "engine": "PySCF", "reason": str(exc)}

    @staticmethod
    def capabilities() -> dict[str, Any]:
        return {
            "engine": "PySCF",
            "platform": "linux-cpu",
            "reference": "closed-shell restricted",
            "methods": {
                name: {
                    "energy": "analytic",
                    "gradient": "analytic",
                    "hessian": "analytic" if name == "hf" else "numerical_gradient",
                    "dipole": None if name == "mp2" else "density_expectation",
                }
                for name in sorted(_METHODS)
            },
            "not_implemented": [
                "open_shell",
                "ECP",
                "solvent",
                "D3",
                "density_fitting",
            ],
            "dispersion": {"pbe-d4": "D4(BJ)-EEQ-ATM", "b3lyp-d4": "D4(BJ)-EEQ-ATM"},
            "qualification": (
                "See actual per-method integration and derivative "
                "evidence; no universal chemical accuracy guarantee."
            ),
        }

    def evaluate(self, request: Any, workspace: str | Path) -> dict[str, Any]:
        data = _normalize(request)
        directory = Path(workspace).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise BackendInputError(
                "Engine workspace must be empty to prevent stale artifact reuse."
            )
        _json(directory / "request.json", data)
        with (
            _environment(data["settings"], directory),
            (directory / "launcher.log").open("w") as log,
        ):
            with redirect_stdout(log), redirect_stderr(log):
                record = self._evaluate(data, directory)
        return _seal(directory, record)

    def _evaluate(
        self, data: dict[str, Any], directory: Path, *, stability: bool | None = None
    ) -> dict[str, Any]:
        import pyscf
        from pyscf.dft import libxc

        record: dict[str, Any] = {
            "schema_version": "cochem-torq.pyscf-result.v1",
            "engine": "PySCF",
            "engine_version": pyscf.__version__,
            "python_version": platform.python_version(),
            "adapter_source_sha256": _SOURCE_SHA256,
            "engine_installation_sha256": _engine_fingerprint()["digest"],
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "libxc_version": libxc.libxc_version(),
            "method": data["method"],
            "settings": data["settings"],
            "molecule": data["molecule"],
            "geometry_bohr": data["molecule"]["geometry_bohr"],
            "status": "failed",
            "energy_hartree": None,
            "gradient_hartree_bohr": None,
            "hessian_hartree_bohr2": None,
            "dipole_debye": None,
            "units": {
                "geometry": "bohr",
                "energy": "hartree",
                "gradient": "hartree/bohr",
                "hessian": "hartree/bohr^2",
                "dipole": "debye",
            },
            "errors": [],
            "quality_flags": ["electronic_model_error_uncalibrated"],
        }
        mol = None
        try:
            mol, mf, calculator, energy = _build(data, directory)
            _json(directory / "engine-installation.json", _engine_fingerprint())
            _json(directory / "basis-definition.json", mol._basis)
            record["basis_definition_sha256"] = sha256(
                (directory / "basis-definition.json").read_bytes()
            ).hexdigest()
            np.save(
                directory / "geometry-bohr.npy",
                mol.atom_coords(unit="Bohr"),
                allow_pickle=False,
            )
            record["scf"] = {
                "converged": bool(mf.converged),
                "energy_hartree": float(mf.e_tot),
                "cycles": int(mf.cycles),
                "electron_count": int(mol.nelectron),
            }
            if not mf.converged:
                raise BackendCalculationError(
                    "SCF did not converge; no accepted energy or derivative exists."
                )
            if not np.isfinite(energy):
                raise BackendCalculationError("The calculated energy is nonfinite.")
            record["energy_hartree"] = energy
            if data["method"]["name"] in ("pbe", "b3lyp"):
                record["density_functional"] = {
                    "libxc_alias": mf.xc,
                    "resolved_components": [
                        [int(component), float(coefficient)]
                        for component, coefficient in libxc.parse_xc(mf.xc)[1]
                    ],
                    "hybrid_coefficients": list(libxc.parse_xc(mf.xc)[0]),
                    "references": list(libxc.xc_reference(mf.xc)),
                    "gradient_grid_response": True,
                }
            if data["method"]["dispersion"] == "d4":
                record["dispersion"] = _d4(data, np.asarray(record["geometry_bohr"]))
                _json(directory / "dftd4-result.json", record["dispersion"])
            elif data["method"]["name"] in ("pbe", "b3lyp"):
                record["quality_flags"].append(
                    "bare_dft_validation_profile_not_dispersion_compliant_production_recipe"
                )
            if data["method"]["name"] == "mp2":
                record["mp2"] = {
                    "correlation_energy_hartree": float(calculator.e_corr),
                    "same_spin_hartree": float(calculator.e_corr_ss),
                    "opposite_spin_hartree": float(calculator.e_corr_os),
                    "frozen_orbitals": 0,
                }
            enabled = (
                data["settings"]["check_stability"] if stability is None else stability
            )
            record["stability"] = _stability(mf, enabled)
            if record["stability"]["status"] == "unstable":
                raise BackendCalculationError(
                    "Restricted SCF is unstable; this profile does not "
                    "change the electronic state automatically."
                )
            if record["stability"]["status"] == "not_requested":
                record["quality_flags"].append("wavefunction_stability_unchecked")
            if "gradient" in data["properties"] or "hessian" in data["properties"]:
                grad = calculator.nuc_grad_method()
                if data["method"]["name"] in ("pbe", "b3lyp"):
                    grad.grid_response = True
                raw_gradient = grad.kernel()
                np.save(
                    directory / "gradient-hartree-bohr.npy",
                    raw_gradient,
                    allow_pickle=False,
                )
                gradient = _finite(raw_gradient, (mol.natm, 3), "calculated gradient")
                record["gradient_hartree_bohr"] = gradient.tolist()
                record["gradient_translation_residual"] = float(
                    np.linalg.norm(gradient.sum(axis=0))
                )
            if "dipole" in data["properties"]:
                dipole = _finite(
                    mf.dip_moment(unit="Debye", verbose=0), (3,), "calculated dipole"
                )
                np.save(directory / "dipole-debye.npy", dipole, allow_pickle=False)
                record["dipole_debye"] = dipole.tolist()
                record["dipole_origin_bohr"] = [0.0, 0.0, 0.0]
                record["dipole_definition"] = (
                    "total electronic plus nuclear density expectation, "
                    "input Cartesian axes"
                )
            if "hessian" in data["properties"]:
                if data["method"]["name"] != "hf":
                    hessian, evidence = self._numerical_hessian(data, directory)
                    record["hessian_evidence"] = evidence
                else:
                    hess = calculator.Hessian()
                    raw = np.asarray(hess.kernel(), dtype=float)
                    hessian = raw.transpose(0, 2, 1, 3).reshape(
                        3 * mol.natm, 3 * mol.natm
                    )
                    record["hessian_evidence"] = {
                        "derivative": "analytic",
                        "engine_tensor_order": "atom_i,atom_j,cart_i,cart_j",
                    }
                np.save(
                    directory / "hessian-hartree-bohr2.npy", hessian, allow_pickle=False
                )
                hessian = _finite(
                    hessian, (3 * mol.natm, 3 * mol.natm), "calculated Hessian"
                )
                asymmetry = float(np.max(np.abs(hessian - hessian.T)))
                record["hessian_evidence"]["max_asymmetry_hartree_bohr2"] = asymmetry
                norm = float(np.linalg.norm(hessian))
                record["hessian_evidence"]["relative_frobenius_asymmetry"] = (
                    float(np.linalg.norm(hessian - hessian.T)) / norm if norm else 0.0
                )
                record["hessian_evidence"][
                    "absolute_symmetry_tolerance_hartree_bohr2"
                ] = data["settings"]["hessian_symmetry_tolerance"]
                if asymmetry > data["settings"]["hessian_symmetry_tolerance"]:
                    raise BackendCalculationError(
                        "Hessian asymmetry exceeds the declared tolerance; "
                        "original values were retained in raw artifacts."
                    )
                record["hessian_hartree_bohr2"] = hessian.tolist()
                record["hessian_evidence"]["coordinate_order"] = (
                    "atom-major xyz, unweighted Cartesian"
                )
            record["status"] = "complete"
        except (
            BackendCalculationError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            record["errors"].append({"code": "CALCULATION_FAILED", "message": str(exc)})
            if record["energy_hartree"] is not None:
                record["status"] = "partial"
        finally:
            if mol is not None and getattr(mol, "stdout", None):
                mol.stdout.flush()
                mol.stdout.close()
        return record

    def _numerical_hessian(
        self, data: dict[str, Any], directory: Path
    ) -> tuple[np.ndarray, dict[str, Any]]:
        geometry = np.asarray(data["molecule"]["geometry_bohr"])
        ncart = geometry.size
        estimates = []
        displacement_dir = directory / "hessian-displacements"
        displacement_dir.mkdir()
        steps = (
            data["settings"]["hessian_step_bohr"],
            data["settings"]["hessian_step_bohr"] / 2,
        )
        for level, step in enumerate(steps):
            hessian = np.empty((ncart, ncart))
            for coordinate in range(ncart):
                gradients = []
                for sign in (-1, 1):
                    displaced = deepcopy(data)
                    coords = geometry.copy().ravel()
                    coords[coordinate] += sign * step
                    displaced["molecule"]["geometry_bohr"] = coords.reshape(
                        geometry.shape
                    ).tolist()
                    displaced["properties"] = ["energy", "gradient"]
                    child = (
                        displacement_dir
                        / f"level-{level}-coordinate-{coordinate:03d}-sign-{sign:+d}"
                    )
                    child.mkdir()
                    _json(child / "request.json", displaced)
                    result = self._evaluate(displaced, child, stability=False)
                    _seal(child, result)
                    if result["status"] != "complete":
                        raise BackendCalculationError(
                            f"{data['method']['name']} Hessian displacement "
                            f"{child.name} failed."
                        )
                    gradients.append(
                        np.asarray(result["gradient_hartree_bohr"]).ravel()
                    )
                hessian[:, coordinate] = (gradients[1] - gradients[0]) / (2 * step)
            estimates.append(hessian)
            np.save(
                directory / f"numerical-hessian-level-{level}.npy",
                hessian,
                allow_pickle=False,
            )
        residual = float(np.max(np.abs(estimates[0] - estimates[1])))
        if residual > data["settings"]["hessian_convergence_tolerance"]:
            raise BackendCalculationError(
                "Numerical Hessian step-convergence residual "
                f"{residual:g} exceeds tolerance."
            )
        return estimates[1], {
            "derivative": "centered_difference_of_analytic_gradient",
            "steps_bohr": list(steps),
            "returned_step_bohr": steps[1],
            "max_step_difference_hartree_bohr2": residual,
            "displacement_count": 4 * ncart,
            "symmetrized": False,
        }

    def optimize(self, request: Any, workspace: str | Path) -> dict[str, Any]:
        """Use genuine geomeTRIC convergence, then evaluate the actual geometry."""
        data = _normalize(request)
        directory = Path(workspace).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise BackendInputError(
                "Engine workspace must be empty to prevent stale artifact reuse."
            )
        _json(directory / "request.json", data)
        parameters = {
            "maxsteps": 80,
            "convergence_energy": 1e-8,
            "convergence_grms": 1e-5,
            "convergence_gmax": 1.5e-5,
            "convergence_drms": 1e-4,
            "convergence_dmax": 1.5e-4,
        }
        unknown = set(data["optimization"]) - set(parameters)
        if unknown:
            raise BackendInputError(
                f"Unsupported optimizer parameters: {sorted(unknown)}."
            )
        parameters.update(data["optimization"])
        for name, value in parameters.items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise BackendInputError(
                    f"Optimizer {name} must be finite and positive."
                )
        if not isinstance(parameters["maxsteps"], int) or parameters["maxsteps"] > 300:
            raise BackendInputError(
                "Optimizer maxsteps must be an integer no greater than 300."
            )
        trajectory: list[dict[str, Any]] = []

        def callback(envs: dict[str, Any]) -> None:
            trajectory.append(
                {
                    "step": int(envs["self"].cycle),
                    "geometry_bohr": np.asarray(envs["coords"]).tolist(),
                    "energy_hartree": float(envs["energy"]),
                    "gradient_hartree_bohr": np.asarray(envs["gradients"]).tolist(),
                }
            )

        mol = None
        with (
            _environment(data["settings"], directory),
            (directory / "launcher.log").open("w") as launcher,
            redirect_stdout(launcher),
            redirect_stderr(launcher),
        ):
            try:
                from pyscf.geomopt import geometric_solver

                initial_dir = directory / "initial"
                initial_dir.mkdir()
                mol, mf, calculator, _ = _build(data, initial_dir)
                if not mf.converged:
                    raise BackendCalculationError(
                        "Initial SCF failed before optimization."
                    )
                initial_stability = _stability(mf, data["settings"]["check_stability"])
                if initial_stability["status"] == "unstable":
                    raise BackendCalculationError(
                        "Initial restricted wavefunction is unstable."
                    )
                # The geomeTRIC scanner may update its checkpoint at each
                # geometry. Keep the initial wavefunction immutable and give
                # the mutable optimization checkpoint its own explicit role.
                mf.chkfile = str(directory / "optimization-wavefunction.chk")
                grad = calculator.nuc_grad_method()
                if data["method"]["name"] in ("pbe", "b3lyp"):
                    grad.grid_response = True
                with (directory / "geometric.log").open("w") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        converged, optimized_mol = geometric_solver.kernel(
                            grad,
                            assert_convergence=True,
                            callback=callback,
                            **parameters,
                        )
                final_data = deepcopy(data)
                final_data["molecule"]["geometry_bohr"] = optimized_mol.atom_coords(
                    unit="Bohr"
                ).tolist()
                final_data["properties"] = sorted(
                    set(data["properties"]) | {"gradient"}
                )
                final_dir = directory / "final"
                final_dir.mkdir()
                _json(final_dir / "request.json", final_data)
                record = self._evaluate(final_data, final_dir)
                _seal(final_dir, record)
                final_gradient = record["gradient_hartree_bohr"]
                gradient_ok = False
                if final_gradient is not None:
                    g = np.asarray(final_gradient)
                    gradient_ok = (
                        float(np.sqrt(np.mean(g * g))) <= parameters["convergence_grms"]
                        and float(np.max(np.abs(g))) <= parameters["convergence_gmax"]
                    )
                record["optimization"] = {
                    "engine": "geomeTRIC",
                    "version": version("geometric"),
                    "converged": bool(converged),
                    "final_gradient_verified": gradient_ok,
                    "parameters": parameters,
                    "evaluations": len(trajectory),
                    "initial_geometry_bohr": data["molecule"]["geometry_bohr"],
                    "final_geometry_source": (
                        "geomeTRIC returned molecular coordinates; independent "
                        "final PySCF evaluation"
                    ),
                    "stationary_point": "unclassified_until_hessian_analysis",
                }
                if not converged or not gradient_ok:
                    record["status"] = (
                        "partial" if record["energy_hartree"] is not None else "failed"
                    )
                    record["errors"].append(
                        {
                            "code": "OPTIMIZATION_NOT_CONVERGED",
                            "message": (
                                "Optimizer convergence and final gradient gates must "
                                "both pass."
                            ),
                        }
                    )
            except (BackendCalculationError, RuntimeError, ValueError) as exc:
                record = {
                    "schema_version": "cochem-torq.pyscf-result.v1",
                    "engine": "PySCF",
                    "status": "failed",
                    "energy_hartree": None,
                    "geometry_bohr": None,
                    "optimization": {"converged": False},
                    "errors": [{"code": "OPTIMIZATION_FAILED", "message": str(exc)}],
                }
            finally:
                if (
                    mol is not None
                    and getattr(mol, "stdout", None)
                    and not mol.stdout.closed
                ):
                    mol.stdout.flush()
                    mol.stdout.close()
                _json(directory / "optimization-trajectory.json", trajectory)
        # The outer manifest covers initial/final native checkpoints and logs.
        record.pop("artifacts", None)
        record.pop("manifest_path", None)
        record.pop("manifest_sha256", None)
        return _seal(directory, record)
