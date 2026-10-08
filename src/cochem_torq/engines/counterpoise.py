"""Real closed-shell monomers in a full complex basis.

Ghosts are basis centers with zero nuclear charge, not nuclei removed after an
ordinary molecular calculation. All basis-center derivatives are retained.
D4, when explicitly selected, is evaluated on the physical fragment only; a
ghost basis center is never supplied as a dispersion atom.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .pyscf_backend import (
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

_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


class GhostPySCFBackend:
    """Restricted HF/DFT/MP2 energy and analytic all-center gradient.

    ``request.molecule`` describes every physical atom of the full complex,
    while ``active_atom_ids``, ``fragment_charge`` and ``fragment_multiplicity``
    explicitly identify the physical monomer electronic state. No spins or
    fragment charges are inferred. A ghost calculation is not a molecule
    optimization or a fragment response property.
    """

    def evaluate(
        self,
        request: Any,
        workspace: str | Path,
        *,
        active_atom_ids: list[str],
        fragment_charge: int,
        fragment_multiplicity: int,
    ) -> dict[str, Any]:
        data = _normalize(request)
        molecule = data["molecule"]
        ids = molecule.get("atom_ids")
        if not ids or len(ids) != len(molecule["symbols"]) or len(set(ids)) != len(ids):
            raise BackendInputError("Counterpoise requires explicit unique atom_ids.")
        if (
            not active_atom_ids
            or len(set(active_atom_ids)) != len(active_atom_ids)
            or not set(active_atom_ids) < set(ids)
        ):
            raise BackendInputError(
                "A ghost monomer requires a strict nonempty subset of atom_ids."
            )
        if type(fragment_charge) is not int:
            raise BackendInputError("The fragment charge must be an explicit integer.")
        if type(fragment_multiplicity) is not int or fragment_multiplicity != 1:
            raise BackendInputError(
                "Only explicitly defined closed-shell singlet fragments are "
                "implemented."
            )
        if set(data["properties"]) - {"energy", "gradient"}:
            raise BackendInputError(
                "The ghost adapter exposes energy and all-center gradient only."
            )
        if data["settings"]["check_stability"] is not True:
            raise BackendInputError(
                "Counterpoise acceptance requires actual SCF stability checks."
            )
        active = [
            index for index, atom_id in enumerate(ids) if atom_id in active_atom_ids
        ]
        directory = Path(workspace).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise BackendInputError(
                "Ghost workspace must be empty; stale results cannot be reused."
            )
        _json(
            directory / "request.json",
            {
                **data,
                "active_atom_ids": active_atom_ids,
                "fragment_charge": fragment_charge,
                "fragment_multiplicity": fragment_multiplicity,
            },
        )
        with (
            _environment(data["settings"], directory),
            (directory / "launcher.log").open("w") as log,
        ):
            with redirect_stdout(log), redirect_stderr(log):
                record = self._evaluate(data, directory, active, fragment_charge)
        return _seal(directory, record)

    def _evaluate(
        self, data: dict[str, Any], directory: Path, active: list[int], charge: int
    ) -> dict[str, Any]:
        import pyscf
        from pyscf import dft, gto, mp, scf
        from pyscf.dft import libxc

        molecule, method, settings = data["molecule"], data["method"], data["settings"]
        count = len(molecule["symbols"])
        geometry = np.asarray(molecule["geometry_bohr"])
        ghost_indices = sorted(set(range(count)) - set(active))
        record: dict[str, Any] = {
            "schema_version": "cochem-torq.pyscf-ghost-result.v1",
            "engine": "PySCF",
            "engine_version": pyscf.__version__,
            "libxc_version": libxc.libxc_version(),
            "adapter_source_sha256": _SOURCE_SHA256,
            "engine_installation_sha256": _engine_fingerprint()["digest"],
            "method": method,
            "settings": settings,
            "geometry_bohr": geometry.tolist(),
            "basis_center_atom_ids": molecule["atom_ids"],
            "physical_atom_ids": [molecule["atom_ids"][i] for i in active],
            "ghost_atom_ids": [molecule["atom_ids"][i] for i in ghost_indices],
            "fragment_charge": charge,
            "fragment_multiplicity": 1,
            "status": "failed",
            "energy_hartree": None,
            "gradient_hartree_bohr": None,
            "units": {
                "energy": "hartree",
                "geometry": "bohr",
                "gradient": "hartree/bohr",
            },
            "errors": [],
            "quality_flags": [
                "electronic_model_error_uncalibrated",
                "same_restricted_state_occupation_and_stability_do_not_prove_unique_scf_root",
            ],
        }
        mol = None
        try:
            mol = gto.Mole()
            mol.atom = [
                (symbol if index in active else f"ghost-{symbol}", geometry[index])
                for index, symbol in enumerate(molecule["symbols"])
            ]
            mol.unit = "Bohr"
            mol.basis = method["basis"]
            mol.charge = charge
            mol.spin = 0
            mol.symmetry = False
            mol.verbose = 4
            mol.max_memory = settings["memory_mb"]
            mol.output = str(directory / "pyscf.log")
            mol.build()
            expected_electrons = (
                sum(gto.charge(molecule["symbols"][i]) for i in active) - charge
            )
            if (
                expected_electrons <= 0
                or expected_electrons % 2
                or mol.nelectron != expected_electrons
            ):
                raise BackendInputError(
                    "Fragment charge/spin and physical nuclei do not define a "
                    "closed-shell monomer."
                )
            if any(mol.atom_charge(i) != 0 for i in ghost_indices):
                raise BackendCalculationError(
                    "A requested ghost center has nonzero nuclear charge."
                )
            if method["name"] in ("pbe", "b3lyp"):
                mf = dft.RKS(mol)
                mf.xc = method["name"]
                record["density_functional"] = {
                    "libxc_alias": mf.xc,
                    "resolved_components": [
                        [int(component), float(coefficient)]
                        for component, coefficient in libxc.parse_xc(mf.xc)[1]
                    ],
                    "hybrid_coefficients": list(libxc.parse_xc(mf.xc)[0]),
                    "references": list(libxc.xc_reference(mf.xc)),
                }
                # PySCF 2.14 energy radii adjustment strips the ghost label,
                # while its default full grid-response radii use Z=0. That
                # mismatch is not a derivative of the accepted energy. Use
                # a declared full-complex quadrature partition for both
                # energy and grid response. The electronic Hamiltonian and
                # every AO evaluation still use the physical ghost molecule.
                grid_molecule = gto.M(
                    atom=list(zip(molecule["symbols"], geometry.tolist())),
                    unit="Bohr",
                    basis=method["basis"],
                    charge=molecule["charge"],
                    spin=0,
                    symmetry=False,
                    verbose=0,
                )
                mf.grids = dft.gen_grid.Grids(grid_molecule)
                mf.grids.level = settings["dft_grid_level"]
                mf.small_rho_cutoff = 0.0
                mf.grids.build(with_non0tab=True)
                record["integration_grid"] = {
                    "partition": (
                        "full-complex atom-centered Becke with Treutler element radii"
                    ),
                    "partition_atom_ids": molecule["atom_ids"],
                    "partition_element_symbols": molecule["symbols"],
                    "partition_geometry_bohr": geometry.tolist(),
                    "partition_nuclear_charge_descriptors": (
                        grid_molecule.atom_charges().tolist()
                    ),
                    "physical_hamiltonian_nuclear_charges": mol.atom_charges().tolist(),
                    "level": settings["dft_grid_level"],
                    "small_density_pruning": False,
                    "grid_response": (
                        "analytic derivative of this same "
                        "full-complex quadrature partition"
                    ),
                    "grid_point_count": int(len(mf.grids.weights)),
                }
                np.savez_compressed(
                    directory / "full-complex-integration-grid.npz",
                    coordinates_bohr=mf.grids.coords,
                    weights_bohr3=mf.grids.weights,
                )
                _json(
                    directory / "integration-grid-definition.json",
                    record["integration_grid"],
                )
            else:
                mf = scf.RHF(mol)
            mf.conv_tol = settings["scf_energy_tolerance"]
            mf.conv_tol_grad = settings["scf_gradient_tolerance"]
            mf.max_cycle = settings["scf_max_cycle"]
            mf.chkfile = str(directory / "wavefunction.chk")
            mf.kernel()
            record["scf"] = {
                "converged": bool(mf.converged),
                "energy_hartree": float(mf.e_tot),
                "cycles": int(mf.cycles),
                "electron_count": int(mol.nelectron),
                "occupations": np.asarray(mf.mo_occ).tolist(),
                "spin_squared": float(mf.spin_square()[0]),
            }
            if not mf.converged:
                raise BackendCalculationError("Ghost monomer SCF did not converge.")
            record["stability"] = _stability(mf, True)
            if record["stability"]["status"] != "stable":
                raise BackendCalculationError(
                    "Ghost monomer restricted SCF is unstable; state "
                    "changes require review."
                )
            calculator = mf
            if method["name"] == "mp2":
                calculator = mp.MP2(mf, frozen=None)
                calculator.kernel()
                record["mp2"] = {
                    "correlation_energy_hartree": float(calculator.e_corr),
                    "same_spin_hartree": float(calculator.e_corr_ss),
                    "opposite_spin_hartree": float(calculator.e_corr_os),
                }
            electronic_energy = float(calculator.e_tot)
            if not np.isfinite(electronic_energy):
                raise BackendCalculationError("Ghost electronic energy is nonfinite.")
            energy = electronic_energy
            gradient = None
            if "gradient" in data["properties"]:
                grad = calculator.nuc_grad_method()
                if method["name"] in ("pbe", "b3lyp"):
                    grad.grid_response = True
                gradient = _finite(
                    grad.kernel(), (count, 3), "ghost all-center gradient"
                )
                np.save(
                    directory / "electronic-gradient-hartree-bohr.npy",
                    gradient,
                    allow_pickle=False,
                )
            if method["dispersion"] == "d4":
                physical = {
                    **data,
                    "molecule": {
                        "symbols": [molecule["symbols"][i] for i in active],
                        "geometry_bohr": geometry[active].tolist(),
                        "charge": charge,
                    },
                }
                dispersion = _d4(physical, geometry[active])
                _json(directory / "dftd4-physical-fragment.json", dispersion)
                record["dispersion"] = dispersion
                energy += dispersion["energy_hartree"]
                if gradient is not None:
                    gradient = gradient.copy()
                    gradient[active] += np.asarray(dispersion["gradient_hartree_bohr"])
            elif method["name"] in ("pbe", "b3lyp"):
                record["quality_flags"].append(
                    "bare_dft_validation_profile_not_dispersion_compliant"
                )
            _json(directory / "basis-definition.json", mol._basis)
            _json(directory / "engine-installation.json", _engine_fingerprint())
            np.save(
                directory / "basis-center-geometry-bohr.npy",
                geometry,
                allow_pickle=False,
            )
            np.save(
                directory / "nuclear-charges.npy",
                mol.atom_charges(),
                allow_pickle=False,
            )
            np.save(
                directory / "orbital-energies-hartree.npy",
                mf.mo_energy,
                allow_pickle=False,
            )
            np.save(
                directory / "orbital-occupations.npy", mf.mo_occ, allow_pickle=False
            )
            record["electronic_energy_hartree"] = electronic_energy
            record["energy_hartree"] = float(energy)
            record["basis_function_count"] = int(mol.nao_nr())
            record["basis_definition_sha256"] = sha256(
                (directory / "basis-definition.json").read_bytes()
            ).hexdigest()
            if gradient is not None:
                _finite(gradient, (count, 3), "total ghost all-center gradient")
                np.save(
                    directory / "gradient-hartree-bohr.npy",
                    gradient,
                    allow_pickle=False,
                )
                record["gradient_hartree_bohr"] = gradient.tolist()
                record["gradient_translation_residual"] = float(
                    np.linalg.norm(gradient.sum(axis=0))
                )
                record["gradient_definition"] = (
                    "analytic derivative with respect to every physical and ghost "
                    "basis-center coordinate"
                )
            record["status"] = "complete"
        except (
            BackendCalculationError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            record["errors"].append(
                {"code": "GHOST_COMPONENT_FAILED", "message": str(exc)}
            )
        finally:
            if (
                mol is not None
                and getattr(mol, "stdout", None)
                and not mol.stdout.closed
            ):
                mol.stdout.flush()
                mol.stdout.close()
        return record
