"""Consume an authenticated PySCF checkpoint as a fresh SCF initial guess.

This is density/orbital continuation, not restoration of an interrupted SCF
iteration, DIIS history, optimization, or a previously calculated final result.
Preparation alone never establishes consumption. The first actual SCF callback
must expose the exact supplied density before a consumed claim is available.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ..capabilities import (
    MAX_ARTIFACT_BYTES,
    RestartArtifact,
    RestartFingerprint,
    ReusePolicy,
    _actual_file,
    _copy_new,
    _file_sha256,
    _native_bundle,
    assess_restart_reuse,
    prepare_initial_guess,
    register_restart_artifact,
)
from ..domain import canonical_json, read_json
from ..registry import list_method_profiles
from .pyscf_backend import (
    _SOURCE_SHA256,
    BackendCalculationError,
    BackendInputError,
    _d4,
    _engine_fingerprint,
    _environment,
    _finite,
    _json,
    _normalize,
    _seal,
    _stability,
)


def _array_sha256(array: np.ndarray) -> str:
    return sha256(np.asarray(array, dtype="<f8").tobytes(order="C")).hexdigest()


def _bind_recipe(
    data: dict[str, Any], recipe_sha256: str, *, optimized_source: bool = False
) -> dict[str, Any]:
    """Resolve an actual registered definition rather than trusting a label."""
    profiles = [
        profile
        for profile in list_method_profiles()
        if profile["recipe_sha256"] == recipe_sha256
    ]
    if len(profiles) != 1 or profiles[0].get("runnable") is not True:
        raise BackendInputError("Checkpoint continuation needs a registered recipe.")
    profile = profiles[0]
    method = data["method"]
    named_method = method["name"] + ("-d4" if method["dispersion"] == "d4" else "")
    if (
        profile["engine"] != "PySCF"
        or profile["engine_version"] != version("pyscf")
        or profile["method"] != named_method
        or profile["basis"] != method["basis"]
        or profile["reference"] != method["reference"]
        or profile["frozen_core"] != method["frozen_core"]
        or profile["density_fitting"] != method["density_fitting"]
        or profile["dispersion"] != method["dispersion"]
        or any(
            data["settings"].get(name) != value
            for name, value in profile["numerical"].items()
        )
        or len(data["molecule"]["symbols"]) > profile["max_atoms"]
        or set(data["molecule"]["symbols"]) - set(profile["elements"])
        or data["molecule"]["multiplicity"] != profile["multiplicity"]
    ):
        raise BackendInputError("Native request differs from its registered recipe.")
    if (
        method["dispersion"] == "d4"
        and version("dftd4") != profile["dispersion_version"]
    ):
        raise BackendInputError("Registered dispersion installation differs.")
    if data["optimization"] and not optimized_source:
        raise BackendInputError("This initial-density route is a fresh single point.")
    if method["name"] != "hf" and "hessian" in data["properties"]:
        raise BackendInputError(
            "Checkpoint continuation currently exposes analytic HF Hessians only."
        )
    if profile["id"] in {
        "hf-sto-3g-pes-validation",
        "hf-sto-3g-internal-pes-validation",
    } and set(data["properties"]) - {
        "energy",
        "gradient",
    }:
        raise BackendInputError("The bounded PES recipe exposes energy/gradient only.")
    return profile


def _validate_checkpoint_orbitals(path: Path, source_result: dict[str, Any]) -> None:
    """Reject malformed orbital/state records even if bytes were re-inventoried."""
    from pyscf.lib import chkfile

    molecule = chkfile.load_mol(str(path))
    data = chkfile.load(str(path), "scf")
    if not isinstance(data, dict):
        raise BackendInputError("Native checkpoint has no real SCF record.")
    arrays = {
        name: np.asarray(data.get(name)) for name in ("mo_coeff", "mo_occ", "mo_energy")
    }
    nao = molecule.nao_nr()
    if any(np.iscomplexobj(array) for array in arrays.values()):
        raise BackendInputError(
            "This checkpoint route requires real restricted orbitals."
        )
    coefficients = _finite(arrays["mo_coeff"], (nao, nao), "checkpoint orbitals")
    occupations = _finite(arrays["mo_occ"], (nao,), "checkpoint occupations")
    _finite(arrays["mo_energy"], (nao,), "checkpoint orbital energies")
    if (
        not np.all(np.isin(occupations, [0.0, 2.0]))
        or abs(float(occupations.sum()) - molecule.nelectron) > 1e-8
        or molecule.spin != 0
    ):
        raise BackendInputError(
            "Checkpoint occupations contradict its restricted state."
        )
    orthogonality = (
        coefficients.T @ molecule.intor_symmetric("int1e_ovlp") @ coefficients
    )
    if not np.allclose(orthogonality, np.eye(nao), atol=1e-7, rtol=0):
        raise BackendInputError("Checkpoint orbitals violate their source AO metric.")
    energy = data.get("e_tot")
    if (
        not isinstance(energy, (int, float, np.integer, np.floating))
        or not np.isfinite(energy)
        or abs(float(energy) - source_result["scf"]["energy_hartree"]) > 1e-8
    ):
        raise BackendInputError("Checkpoint energy differs from its actual native SCF.")


def _new_mean_field(data: dict[str, Any], directory: Path) -> tuple[Any, Any]:
    """Build the requested real engine without running a default-guess SCF."""
    from pyscf import dft, gto, scf

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
            "A positive even-electron restricted state is required."
        )
    if method["name"] in ("pbe", "b3lyp"):
        mf = dft.RKS(mol)
        mf.xc = method["name"]
        mf.grids.level = settings["dft_grid_level"]
    else:
        mf = scf.RHF(mol)
    mf.conv_tol = settings["scf_energy_tolerance"]
    mf.conv_tol_grad = settings["scf_gradient_tolerance"]
    mf.max_cycle = settings["scf_max_cycle"]
    mf.chkfile = str(directory / "wavefunction.chk")
    if method["dispersion"] == "d4":
        from dftd4.pyscf import energy

        mf = energy(mf)
        mf.with_dftd4.stdout = mol.stdout
    return mol, mf


def _snapshot_source(root: Path, directory: Path) -> dict[str, Any]:
    """Copy actual inventoried bytes with independent read-only file inodes."""
    manifest_path = _actual_file(root / "manifest.json")
    manifest = read_json(manifest_path)
    entries = manifest["artifacts"]
    if sum(entry["size_bytes"] for entry in entries) > MAX_ARTIFACT_BYTES:
        raise BackendInputError("Native restart snapshot exceeds the bounded budget.")
    expected = {entry["path"] for entry in entries}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "scratch" not in path.relative_to(root).parts
    }
    if actual != expected:
        raise BackendInputError("Native snapshot inventory omits or adds raw files.")
    snapshot = directory / "source-snapshot"
    for entry in entries:
        _copy_new(
            root / entry["path"],
            snapshot / entry["path"],
            entry["sha256"],
            mode=0o440,
        )
    # Retain the original manifest as an indexed raw artifact of the new run.
    _copy_new(
        manifest_path,
        snapshot / "source-manifest.json",
        _file_sha256(manifest_path),
        mode=0o440,
    )
    # Nested optimizer manifests are not inventoried in the native parent format.
    # Their authentic inventories have already been verified by _native_bundle.
    for path in sorted(root.rglob("manifest.json")):
        if path == manifest_path or "scratch" in path.relative_to(root).parts:
            continue
        target = snapshot / path.relative_to(root).parent / "source-manifest.json"
        _copy_new(path, target, _file_sha256(path), mode=0o440)
    for path in sorted(snapshot.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o550)
    snapshot.chmod(0o550)
    return {
        "directory": str(snapshot),
        "source_manifest_sha256": _file_sha256(manifest_path),
        "inventory": entries,
    }


@dataclass
class PreparedCheckpointDensity:
    """Actual engine plus loaded density; the caller owns its isolated process.

    run_scf performs a fresh calculation. A caller may obtain energy/gradients
    from mf afterwards, but must retain consumption_evidence and authenticate
    the resulting output; preparation is not a completed native calculation.
    """

    mol: Any
    mf: Any
    initial_density: np.ndarray
    normalized_request: dict[str, Any]
    artifact: RestartArtifact
    target: RestartFingerprint
    policy: ReusePolicy
    workspace: Path
    recipe_profile: dict[str, Any]
    snapshot: dict[str, Any]
    project_orbitals: bool
    observations: list[dict[str, Any]] = field(default_factory=list)
    kernel_called: bool = False
    _prepared_density_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._prepared_density_sha256 = _array_sha256(self.initial_density)

    def run_scf(self) -> float:
        """Pass the actual authenticated density to PySCF's genuine kernel."""
        if self.kernel_called:
            raise BackendInputError(
                "A prepared checkpoint starts exactly one fresh SCF."
            )
        if self.mf.callback is not None:
            raise BackendInputError("Checkpoint consumption owns its observation hook.")
        self.verify_preserved_inputs()
        expected = _array_sha256(self.initial_density)
        if expected != self._prepared_density_sha256 or expected != _file_sha256(
            self.workspace / "initial-density.raw"
        ):
            raise BackendCalculationError("The prepared density changed before use.")

        def observe(environment: Mapping[str, Any]) -> None:
            if not self.observations:
                actual = np.asarray(environment["dm_last"], dtype=float)
                if _array_sha256(actual) != expected:
                    raise BackendCalculationError(
                        "The first SCF iteration did not consume the prepared density."
                    )
                self.observations.append(
                    {
                        "first_cycle": int(environment["cycle"]) + 1,
                        "first_cycle_input_density_sha256": _array_sha256(actual),
                        "initial_energy_hartree": float(environment["last_hf_e"]),
                        "first_cycle_energy_hartree": float(environment["e_tot"]),
                    }
                )

        self.mf.callback = observe
        self.kernel_called = True
        try:
            energy = float(self.mf.kernel(dm0=self.initial_density))
        finally:
            self.verify_preserved_inputs()
        if not self.observations:
            raise BackendCalculationError("No actual startup consumption was observed.")
        return energy

    def verify_preserved_inputs(self) -> None:
        """Reauthenticate original, sealed, working input, and snapshot bytes."""
        assess_restart_reuse(self.artifact, self.target, self.policy)
        if (
            not np.allclose(
                self.mol.atom_coords(unit="Bohr"),
                self.target.geometry_bohr,
                atol=1e-12,
                rtol=0,
            )
            or self.mol.charge != self.target.charge
            or self.mol.spin + 1 != self.target.multiplicity
            or self.mf.chkfile != str(self.workspace / "wavefunction.chk")
        ):
            raise BackendCalculationError(
                "Prepared engine geometry/state/output changed."
            )
        if (
            _file_sha256(self.workspace / "restart-input.chk")
            != self.artifact.artifact_sha256
        ):
            raise BackendCalculationError("The separate restart input was modified.")
        snapshot = Path(self.snapshot["directory"])
        if (
            _file_sha256(snapshot / "source-manifest.json")
            != self.snapshot["source_manifest_sha256"]
        ):
            raise BackendCalculationError(
                "The retained native snapshot manifest changed."
            )
        for entry in self.snapshot["inventory"]:
            path = snapshot / entry["path"]
            if _file_sha256(path) != entry["sha256"] or path.stat().st_mode & 0o222:
                raise BackendCalculationError("The read-only native snapshot changed.")

    def consumption_evidence(self) -> dict[str, Any]:
        """Expose observed consumption, including false when only prepared."""
        self.verify_preserved_inputs()
        assessment = assess_restart_reuse(self.artifact, self.target, self.policy)
        return {
            "schema_version": "cochem.torq.native-checkpoint-consumption/1",
            "purpose": "initial_guess",
            "engine_checkpoint_reused": bool(self.observations),
            "new_physical_calculation_required": True,
            "fresh_scf_kernel_called": self.kernel_called,
            "scf_iteration_state_resumed": False,
            "final_result_reused": False,
            "chemical_accuracy_established": False,
            "registered_recipe_id": self.recipe_profile["id"],
            "recipe_sha256": self.recipe_profile["recipe_sha256"],
            "source_manifest_sha256": self.artifact.native_manifest_sha256,
            "source_checkpoint_sha256": self.artifact.artifact_sha256,
            "source_fingerprint_sha256": self.artifact.fingerprint.identity_sha256,
            "target_fingerprint_sha256": self.target.identity_sha256,
            "changed_fields": assessment["changed_fields"],
            "initial_density_sha256": _array_sha256(self.initial_density),
            "initial_density_definition": (
                "real PySCF chkfile orbitals/occupations; init_guess_by_chkfile; "
                "restricted AO density; little-endian float64 row-major digest"
            ),
            "project_orbitals": self.project_orbitals,
            "initial_electron_count": float(
                np.einsum("ij,ji->", self.initial_density, self.mf.get_ovlp())
            ),
            "startup_observations": list(self.observations),
            "immutable_inputs_verified": True,
            "checkpoint_adapter_source_sha256": _file_sha256(Path(__file__)),
            "scope_flags": [
                "no_scf_root_continuity_proof",
                "no_speedup_or_chemical_accuracy_claim",
                "no_interrupted_iteration_or_optimizer_resume",
            ],
        }

    def close(self) -> None:
        if getattr(self.mol, "stdout", None) and not self.mol.stdout.closed:
            self.mol.stdout.flush()
            self.mol.stdout.close()


def prepare_checkpoint_density(
    request: Any,
    workspace: str | Path,
    *,
    source_native_directory: str | Path,
    recipe_sha256: str,
    allow_geometry_change: bool = False,
    expected_source_manifest_sha256: str | None = None,
    project_orbitals: bool = True,
) -> PreparedCheckpointDensity:
    """Authenticate and load native orbitals; this function does not run SCF.

    Use a distinct, empty workspace. The same registered recipe, ordered atom
    identifiers, isotope labels, state, basis and installation are required.
    Only geometry may change, explicitly, for initial-guess use. No source or
    sealed checkpoint is passed as the new SCF output checkpoint.
    """
    if type(allow_geometry_change) is not bool or type(project_orbitals) is not bool:
        raise BackendInputError("Checkpoint continuation permissions must be boolean.")
    data = _normalize(request)
    profile = _bind_recipe(data, recipe_sha256)
    source = Path(source_native_directory).absolute()
    source_request, source_result = _native_bundle(source)
    _bind_recipe(
        source_request, recipe_sha256, optimized_source="optimization" in source_result
    )
    if source_request["molecule"].get("isotopes") != data["molecule"].get("isotopes"):
        raise BackendInputError("Native isotope declarations differ from the target.")
    manifest_identity = _file_sha256(source / "manifest.json")
    if expected_source_manifest_sha256 is not None and (
        manifest_identity != expected_source_manifest_sha256
    ):
        raise BackendInputError("The supplied source manifest identity differs.")
    directory = Path(workspace).absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise BackendInputError("Checkpoint output paths cannot follow symlinks.")
    if directory.is_relative_to(source) or source.is_relative_to(directory):
        raise BackendInputError(
            "Source and fresh checkpoint workspaces must be separate."
        )
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise BackendInputError(
            "Checkpoint workspace must be empty; stale reuse rejected."
        )
    _json(directory / "request.json", data)
    mol = None
    try:
        with (directory / "preparation.log").open("w") as log:
            with redirect_stdout(log), redirect_stderr(log):
                mol, mf = _new_mean_field(data, directory)
        _json(directory / "basis-definition.json", mol._basis)
        _json(directory / "engine-installation.json", _engine_fingerprint())
        checkpoint_relative = (
            "final/wavefunction.chk"
            if "optimization" in source_result
            else "wavefunction.chk"
        )
        _validate_checkpoint_orbitals(source / checkpoint_relative, source_result)
        artifact = register_restart_artifact(
            source,
            checkpoint_relative,
            directory / "sealed-input.chk",
            artifact_kind="wavefunction",
            recipe_sha256=recipe_sha256,
        )
        target_fields = artifact.fingerprint.model_dump(mode="json")
        molecule = data["molecule"]
        target_fields.update(
            geometry_bohr=molecule["geometry_bohr"],
            symbols=molecule["symbols"],
            isotope_symbols=molecule["isotope_symbols"],
            atom_ids=molecule["atom_ids"]
            or [f"atom-{index}" for index in range(len(molecule["symbols"]))],
            charge=molecule["charge"],
            multiplicity=molecule["multiplicity"],
            electronic_reference=data["method"]["reference"],
            basis=data["method"]["basis"],
            basis_definition_sha256=_file_sha256(directory / "basis-definition.json"),
            method_definition_json=canonical_json(data["method"]).decode(),
            numerical_settings_json=canonical_json(data["settings"]).decode(),
        )
        target = RestartFingerprint.model_validate(target_fields)
        policy = ReusePolicy(
            purpose="initial_guess",
            allowed_initial_guess_changes=("geometry_bohr",)
            if allow_geometry_change
            else (),
        )
        prepare_initial_guess(
            artifact, target, directory / "restart-input.chk", policy=policy
        )
        snapshot = _snapshot_source(source, directory)
        density = _finite(
            mf.init_guess_by_chkfile(
                str(directory / "restart-input.chk"), project=project_orbitals
            ),
            (mol.nao_nr(), mol.nao_nr()),
            "native checkpoint density",
        )
        if not np.allclose(density, density.T, atol=1e-10, rtol=0):
            raise BackendInputError(
                "The authentic checkpoint density is not symmetric."
            )
        electrons = float(np.einsum("ij,ji->", density, mf.get_ovlp()))
        if abs(electrons - mol.nelectron) > 1e-7:
            raise BackendInputError(
                "Checkpoint density electron count differs in the target AO metric; "
                "an explicit orbital projection or a fresh guess is required."
            )
        np.save(directory / "initial-density.npy", density, allow_pickle=False)
        (directory / "initial-density.raw").write_bytes(
            np.asarray(density, dtype="<f8").tobytes(order="C")
        )
        prepared = PreparedCheckpointDensity(
            mol,
            mf,
            density,
            data,
            artifact,
            target,
            policy,
            directory,
            profile,
            snapshot,
            project_orbitals,
        )
        prepared.verify_preserved_inputs()
        _json(
            directory / "checkpoint-preparation.json", prepared.consumption_evidence()
        )
        return prepared
    except Exception:
        if mol is not None and getattr(mol, "stdout", None):
            mol.stdout.close()
        raise


def evaluate_with_checkpoint(
    request: Any,
    workspace: str | Path,
    *,
    source_native_directory: str | Path,
    recipe_sha256: str,
    allow_geometry_change: bool = False,
    expected_source_manifest_sha256: str | None = None,
    project_orbitals: bool = True,
) -> dict[str, Any]:
    """Run and seal a genuine fresh native calculation using verified orbitals."""
    import pyscf
    from pyscf import mp
    from pyscf.dft import libxc

    prepared = prepare_checkpoint_density(
        request,
        workspace,
        source_native_directory=source_native_directory,
        recipe_sha256=recipe_sha256,
        allow_geometry_change=allow_geometry_change,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        project_orbitals=project_orbitals,
    )
    data, directory = prepared.normalized_request, prepared.workspace
    record: dict[str, Any] = {
        "schema_version": "cochem-torq.pyscf-result.v1",
        "engine": "PySCF",
        "engine_version": pyscf.__version__,
        "python_version": platform.python_version(),
        "adapter_source_sha256": _SOURCE_SHA256,
        "engine_installation_sha256": _engine_fingerprint()["digest"],
        "basis_definition_sha256": _file_sha256(directory / "basis-definition.json"),
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
        "quality_flags": [
            "electronic_model_error_uncalibrated",
            "checkpoint_initial_guess_no_scf_root_continuity_proof",
        ],
    }
    try:
        with (
            _environment(data["settings"], directory),
            (directory / "launcher.log").open("w") as log,
        ):
            with redirect_stdout(log), redirect_stderr(log):
                log.write(
                    "Starting fresh PySCF SCF with authenticated native checkpoint "
                    "density; source_manifest_sha256="
                    f"{prepared.artifact.native_manifest_sha256}\n"
                )
                log.flush()
                energy = prepared.run_scf()
                mf, mol = prepared.mf, prepared.mol
                record["checkpoint_consumption"] = prepared.consumption_evidence()
                record["scf"] = {
                    "converged": bool(mf.converged),
                    "energy_hartree": float(mf.e_tot),
                    "cycles": int(mf.cycles),
                    "electron_count": int(mol.nelectron),
                }
                if not mf.converged or not np.isfinite(energy):
                    raise BackendCalculationError(
                        "The fresh checkpoint-guess SCF did not converge."
                    )
                calculator = mf
                if data["method"]["name"] == "mp2":
                    calculator = mp.MP2(mf, frozen=None)
                    calculator.kernel()
                    energy = float(calculator.e_tot)
                    record["mp2"] = {
                        "correlation_energy_hartree": float(calculator.e_corr),
                        "same_spin_hartree": float(calculator.e_corr_ss),
                        "opposite_spin_hartree": float(calculator.e_corr_os),
                        "frozen_orbitals": 0,
                    }
                if not np.isfinite(energy):
                    raise BackendCalculationError(
                        "The fresh correlated total energy is nonfinite."
                    )
                record["energy_hartree"] = energy
                record["stability"] = _stability(
                    mf, data["settings"]["check_stability"]
                )
                if record["stability"]["status"] != "stable":
                    raise BackendCalculationError(
                        "Fresh restricted SCF stability was not accepted."
                    )
                np.save(
                    directory / "geometry-bohr.npy",
                    mol.atom_coords(unit="Bohr"),
                    allow_pickle=False,
                )
                if data["method"]["name"] in ("pbe", "b3lyp"):
                    record["density_functional"] = {
                        "libxc_alias": mf.xc,
                        "resolved_components": [
                            [int(i), float(c)] for i, c in libxc.parse_xc(mf.xc)[1]
                        ],
                        "hybrid_coefficients": list(libxc.parse_xc(mf.xc)[0]),
                        "references": list(libxc.xc_reference(mf.xc)),
                        "gradient_grid_response": True,
                    }
                if data["method"]["dispersion"] == "d4":
                    record["dispersion"] = _d4(
                        data, np.asarray(record["geometry_bohr"])
                    )
                    _json(directory / "dftd4-result.json", record["dispersion"])
                if "gradient" in data["properties"] or "hessian" in data["properties"]:
                    grad = calculator.nuc_grad_method()
                    if data["method"]["name"] in ("pbe", "b3lyp"):
                        grad.grid_response = True
                    gradient = _finite(
                        grad.kernel(), (mol.natm, 3), "calculated gradient"
                    )
                    np.save(
                        directory / "gradient-hartree-bohr.npy",
                        gradient,
                        allow_pickle=False,
                    )
                    record["gradient_hartree_bohr"] = gradient.tolist()
                    record["gradient_translation_residual"] = float(
                        np.linalg.norm(gradient.sum(axis=0))
                    )
                if "dipole" in data["properties"]:
                    dipole = _finite(
                        mf.dip_moment(unit="Debye", verbose=0),
                        (3,),
                        "calculated dipole",
                    )
                    np.save(directory / "dipole-debye.npy", dipole, allow_pickle=False)
                    record["dipole_debye"] = dipole.tolist()
                    record["dipole_origin_bohr"] = [0.0, 0.0, 0.0]
                    record["dipole_definition"] = (
                        "total electronic plus nuclear density expectation, "
                        "input Cartesian axes"
                    )
                if "hessian" in data["properties"]:
                    raw = np.asarray(calculator.Hessian().kernel(), dtype=float)
                    hessian = _finite(
                        raw.transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm),
                        (3 * mol.natm, 3 * mol.natm),
                        "calculated Hessian",
                    )
                    np.save(
                        directory / "hessian-hartree-bohr2.npy",
                        hessian,
                        allow_pickle=False,
                    )
                    asymmetry = float(np.max(np.abs(hessian - hessian.T)))
                    if asymmetry > data["settings"]["hessian_symmetry_tolerance"]:
                        raise BackendCalculationError(
                            "Actual Hessian symmetry exceeds its tolerance."
                        )
                    record["hessian_hartree_bohr2"] = hessian.tolist()
                    record["hessian_evidence"] = {
                        "derivative": "analytic",
                        "max_asymmetry_hartree_bohr2": asymmetry,
                        "coordinate_order": "atom-major xyz, unweighted Cartesian",
                        "engine_tensor_order": "atom_i,atom_j,cart_i,cart_j",
                    }
                record["status"] = "complete"
    except (
        BackendCalculationError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        record["errors"].append(
            {"code": "CHECKPOINT_CALCULATION_FAILED", "message": str(exc)}
        )
        if record["energy_hartree"] is not None:
            record["status"] = "partial"
    finally:
        try:
            record["checkpoint_consumption"] = prepared.consumption_evidence()
            _json(
                directory / "checkpoint-consumption.json",
                record["checkpoint_consumption"],
            )
        finally:
            prepared.close()
    return _seal(directory, record)
