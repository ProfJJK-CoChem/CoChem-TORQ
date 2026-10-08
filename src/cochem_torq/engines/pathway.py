"""Bounded real HF first-order saddle and downhill endpoint candidates.

The workflow does not misname local displaced minimizations as IRC trajectories
or infer chemical connectivity from RMSD. A full verified TS/pathway remains a
separate gate requiring chemical mode and endpoint identity evidence.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from ..domain import Contract, Molecule
from .diagnostics import electronic_diagnostics, stationary_point_diagnostics
from .pyscf_backend import (
    BackendCalculationError,
    BackendInputError,
    PySCFBackend,
    _build,
    _environment,
    _json,
    _normalize,
    _seal,
    _stability,
)

_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


class PathwayRequest(Contract):
    schema_version: Literal["cochem.torq.pathway-request/1"] = (
        "cochem.torq.pathway-request/1"
    )
    molecule: Molecule
    method: dict[str, Any]
    maxsteps: StrictInt = Field(default=80, ge=1, le=150)
    follow_downhill_endpoints: StrictBool = True
    displacement_max_bohr: StrictFloat = Field(default=0.15, gt=0, le=0.5)

    @model_validator(mode="after")
    def bounded_state(self) -> Self:
        data = _normalize(
            {
                "molecule": self.molecule,
                "method": self.method,
                "properties": ["energy", "gradient", "hessian"],
            }
        )
        if data["method"]["name"] != "hf":
            raise ValueError(
                "This actual analytic-Hessian saddle profile currently supports "
                "restricted HF only."
            )
        if len(self.molecule.symbols) < 3 or len(self.molecule.symbols) > 30:
            raise ValueError("The bounded saddle profile requires 3 to 30 atoms.")
        if not self.molecule.atom_ids:
            raise ValueError(
                "Explicit persistent atom IDs are required for endpoint correspondence."
            )
        if data["settings"]["check_stability"] is not True:
            raise ValueError(
                "Actual SCF stability checks are mandatory for this saddle profile."
            )
        return self


class PathwayResult(Contract):
    schema_version: Literal["cochem.torq.pathway-result/1"] = (
        "cochem.torq.pathway-result/1"
    )
    status: Literal["available", "failed"]
    observable: Literal["first_order_saddle_candidate"] = "first_order_saddle_candidate"
    ts_verified: Literal[False] = False
    irc: dict[str, Any]
    method: dict[str, Any]
    settings: dict[str, Any]
    atom_ids: list[str]
    components: dict[str, dict[str, Any]]
    stationary_evidence: dict[str, dict[str, Any]]
    electronic_diagnostics: dict[str, dict[str, Any]]
    downhill_endpoints: dict[str, dict[str, Any]]
    errors: list[str]
    adapter_source_sha256: str
    quality_flags: list[str]
    optimizer: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None

    @model_validator(mode="after")
    def candidate_only(self) -> Self:
        if self.irc.get("status") != "unavailable" or not self.irc.get("reason"):
            raise ValueError("This candidate workflow cannot certify an IRC.")
        if self.status == "available":
            evidence = self.stationary_evidence.get("saddle", {})
            if (
                self.errors
                or evidence.get("classification") != "first_order_saddle_candidate"
            ):
                raise ValueError(
                    "An available saddle candidate requires real accepted "
                    "first-order curvature."
                )
        elif not self.errors:
            raise ValueError("Failed saddle workflow requires its actual error.")
        return self


def _masses(molecule: Molecule) -> list[float]:
    from Libraries.cochem_isotopes import isotope_mass

    numbers = molecule.isotopes or [None] * len(molecule.symbols)
    return [
        isotope_mass(f"{number}{symbol}" if number is not None else symbol)
        for symbol, number in zip(molecule.symbols, numbers)
    ]


def locate_saddle_candidate(
    request: PathwayRequest | dict[str, Any], workspace: str | Path
) -> dict[str, Any]:
    """Run actual geomeTRIC TS optimization and independently verify curvature.

    Failed higher stages retain their authentic earlier component results. No
    electronic-state/method recovery or arbitrary symmetry-breaking is applied.
    """
    if not isinstance(request, PathwayRequest):
        request = PathwayRequest.model_validate(request)
    data = _normalize(
        {
            "molecule": request.molecule,
            "method": request.method,
            "properties": ["energy", "gradient", "hessian"],
        }
    )
    directory = Path(workspace).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise BackendInputError(
            "Pathway workspace must be empty; stale structures or paths cannot"
            " be reused."
        )
    _json(directory / "request.json", request.model_dump(mode="json"))
    result: dict[str, Any] = {
        "schema_version": "cochem.torq.pathway-result/1",
        "status": "failed",
        "observable": "first_order_saddle_candidate",
        "ts_verified": False,
        "irc": {
            "status": "unavailable",
            "reason": (
                "A qualified mass-weighted IRC integrator with step convergence "
                "and mapped endpoint identity is not implemented by this candidate"
                " workflow."
            ),
        },
        "method": data["method"],
        "settings": data["settings"],
        "atom_ids": request.molecule.atom_ids,
        "components": {},
        "stationary_evidence": {},
        "electronic_diagnostics": {},
        "downhill_endpoints": {},
        "errors": [],
        "adapter_source_sha256": _SOURCE_SHA256,
        "quality_flags": [
            "experimental_bounded_hf_saddle_profile",
            "chemical_mode_relevance_requires_review",
            "endpoint_chemical_identity_requires_review",
            "no_automatic_model_or_state_changes",
        ],
    }
    trajectory = []
    backend = PySCFBackend()
    mol = None

    def callback(envs: dict[str, Any]) -> None:
        trajectory.append(
            {
                "evaluation": int(envs["self"].cycle),
                "geometry_bohr": np.asarray(envs["coords"]).tolist(),
                "energy_hartree": float(envs["energy"]),
                "gradient_hartree_bohr": np.asarray(envs["gradients"]).tolist(),
            }
        )

    try:
        initial = backend.evaluate(data, directory / "initial-characterization")
        result["components"]["initial"] = initial
        if initial["status"] != "complete":
            raise BackendCalculationError(
                "Initial real Hessian/stability calculation failed: "
                f"{initial['errors']}"
            )
        with (
            _environment(data["settings"], directory),
            (directory / "geometric-launcher.log").open("w") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            from pyscf.geomopt import geometric_solver

            initial_directory = directory / "optimizer-initial"
            initial_directory.mkdir()
            mol, mf, calculator, _ = _build(data, initial_directory)
            if not mf.converged or _stability(mf, True)["status"] != "stable":
                raise BackendCalculationError(
                    "Optimizer initial SCF state is not converged/stable."
                )
            mf.chkfile = str(directory / "optimizer-wavefunction.chk")
            gradient = calculator.nuc_grad_method()
            # PySCF writes its actual initial analytic Hessian to this explicit
            # solver input; retain it instead of relying on deleted temp files.
            hessian_file = directory / "solver-initial-hessian-hartree-bohr2.txt"
            converged, optimized = geometric_solver.kernel(
                gradient,
                assert_convergence=True,
                callback=callback,
                maxsteps=request.maxsteps,
                transition=True,
                hessian=f"file:{hessian_file}",
                convergence_energy=1e-8,
                convergence_grms=1e-5,
                convergence_gmax=1.5e-5,
                convergence_drms=1e-4,
                convergence_dmax=1.5e-4,
            )
        if not converged:
            raise BackendCalculationError(
                "Actual geomeTRIC saddle optimization did not converge within the "
                "declared step budget."
            )
        final_request = deepcopy(data)
        final_request["molecule"]["geometry_bohr"] = optimized.atom_coords(
            unit="Bohr"
        ).tolist()
        final = backend.evaluate(final_request, directory / "saddle-final-independent")
        result["components"]["saddle"] = final
        diagnostics = stationary_point_diagnostics(final, _masses(request.molecule))
        result["stationary_evidence"]["saddle"] = diagnostics
        result["electronic_diagnostics"]["saddle"] = electronic_diagnostics(final)
        if diagnostics["classification"] != "first_order_saddle_candidate":
            raise BackendCalculationError(
                "Optimized structure is not a stationary stable first-order saddle"
                " candidate of its actual projected Hessian."
            )
        result["optimizer"] = {
            "engine": "geomeTRIC",
            "converged": bool(converged),
            "evaluation_count": len(trajectory),
            "maxsteps": request.maxsteps,
            "target": "transition",
            "final_independent_gradient_checked": True,
        }
        if request.follow_downhill_endpoints:
            mode = np.asarray(diagnostics["imaginary_cartesian_modes"][0]).reshape(
                -1, 3
            )
            scale = request.displacement_max_bohr / float(np.max(np.abs(mode)))
            for sign in [-1, 1]:
                label = "negative" if sign == -1 else "positive"
                displaced = deepcopy(final_request)
                displaced["molecule"]["geometry_bohr"] = (
                    np.asarray(final["geometry_bohr"]) + sign * scale * mode
                ).tolist()
                displaced["optimization"] = {"maxsteps": request.maxsteps}
                endpoint = backend.optimize(
                    displaced, directory / f"downhill-{label}-relaxed"
                )
                result["components"][f"endpoint_{label}"] = endpoint
                endpoint_diagnostics = stationary_point_diagnostics(
                    endpoint, _masses(request.molecule)
                )
                result["stationary_evidence"][f"endpoint_{label}"] = (
                    endpoint_diagnostics
                )
                if (
                    endpoint_diagnostics["classification"] != "minimum"
                    or endpoint.get("optimization", {}).get("converged") is not True
                    or endpoint.get("optimization", {}).get("final_gradient_verified")
                    is not True
                    or endpoint["energy_hartree"] >= final["energy_hartree"]
                ):
                    raise BackendCalculationError(
                        "A displaced actual endpoint did not relax to "
                        "a lower-energy verified minimum."
                    )
                result["downhill_endpoints"][label] = {
                    "geometry_bohr": endpoint["geometry_bohr"],
                    "energy_hartree": endpoint["energy_hartree"],
                    "electronic_barrier_hartree": final["energy_hartree"]
                    - endpoint["energy_hartree"],
                    "component_manifest_sha256": endpoint["manifest_sha256"],
                    "initial_max_cartesian_displacement_bohr": (
                        request.displacement_max_bohr
                    ),
                    "construction": (
                        "imaginary-mode displacement followed by genuine unconstrained "
                        "same-model minimization"
                    ),
                    "irc_trajectory": False,
                    "mapped_chemical_endpoint_identity": "unverified_requires_review",
                }
        result["status"] = "available"
    except (
        BackendInputError,
        BackendCalculationError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        result["errors"].append(str(exc))
    finally:
        if mol is not None and getattr(mol, "stdout", None) and not mol.stdout.closed:
            mol.stdout.flush()
            mol.stdout.close()
        _json(directory / "saddle-optimization-trajectory.json", trajectory)
    typed = PathwayResult.model_validate(result)
    sealed = _seal(directory, typed.model_dump(mode="json"))
    return PathwayResult.model_validate(sealed).model_dump(mode="json")
