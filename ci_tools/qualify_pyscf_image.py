"""Genuine, bounded PySCF installation and native-derivative qualification.

This is a named H2/RHF/STO-3G software/numerical profile. It does not establish
experimental spectroscopy accuracy or qualify other methods and derivatives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, scf


def qualify(
    output: Path,
    image_identity: str,
    source_commit: str,
    source_worktree_status: str = "clean",
) -> dict:
    if pyscf.__version__ != "2.14.0":
        raise RuntimeError(
            "This image profile requires the actually installed PySCF 2.14.0"
        )
    if importlib.metadata.version("geometric") != "1.1.1":
        raise RuntimeError("This image profile requires geomeTRIC 1.1.1")
    lib.num_threads(1)
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    def run(positions: np.ndarray):
        molecule = gto.M(
            atom=list(zip(["H", "H"], positions.tolist())),
            basis="sto-3g",
            unit="Angstrom",
            charge=0,
            spin=0,
            verbose=0,
        )
        electronic = scf.RHF(molecule)
        electronic.conv_tol = 1e-12
        electronic.max_cycle = 100
        electronic.kernel()
        if not electronic.converged or not np.isfinite(electronic.e_tot):
            raise RuntimeError("The real qualification SCF did not converge")
        return electronic

    electronic = run(coordinates)
    energy = float(electronic.e_tot)
    # Fixed canonical H2 geometry; a known minimal-basis HF regression quantity,
    # never represented as an experimental or high-accuracy reference.
    if abs(energy - (-1.1167593073964255)) > 2e-9:
        raise RuntimeError(f"Unexpected H2/RHF/STO-3G reference energy: {energy}")
    gradient = np.asarray(electronic.nuc_grad_method().kernel())
    native_hessian = np.asarray(electronic.Hessian().kernel())
    hessian = native_hessian.transpose(0, 2, 1, 3).reshape(6, 6)
    if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise RuntimeError("The engine produced nonfinite derivatives")
    step_bohr = 1e-4
    step_angstrom = step_bohr * lib.param.BOHR
    fd_gradient = np.zeros((2, 3))
    fd_hessian = np.zeros((6, 6))
    for coordinate in range(6):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus.flat[coordinate] += step_angstrom
        minus.flat[coordinate] -= step_angstrom
        ep, em = run(plus), run(minus)
        fd_gradient.flat[coordinate] = (ep.e_tot - em.e_tot) / (2 * step_bohr)
        fd_hessian[:, coordinate] = (
            ep.nuc_grad_method().kernel().reshape(6)
            - em.nuc_grad_method().kernel().reshape(6)
        ) / (2 * step_bohr)
    gradient_error = float(np.max(np.abs(gradient - fd_gradient)))
    hessian_error = float(np.max(np.abs(hessian - fd_hessian)))
    symmetry_error = float(np.max(np.abs(hessian - hessian.T)))
    translation_error = float(np.max(np.abs(native_hessian.sum(axis=1))))
    if (
        gradient_error > 2e-6
        or hessian_error > 2e-5
        or symmetry_error > 1e-10
        or translation_error > 1e-9
    ):
        raise RuntimeError(
            "Genuine derivative qualification exceeded its "
            "preregistered numerical tolerance"
        )
    output.mkdir(parents=True, exist_ok=False)
    arrays = output / "native-and-finite-difference-derivatives.npz"
    np.savez(
        arrays,
        coordinates_angstrom=coordinates,
        energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        finite_difference_gradient=fd_gradient,
        finite_difference_hessian=fd_hessian,
    )
    report = {
        "schema_version": "cochem.engine-qualification/1",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "pyscf",
        "engine_version": pyscf.__version__,
        "geometric_version": importlib.metadata.version("geometric"),
        "image_identity": image_identity,
        "source_commit": source_commit,
        "source_worktree_status": source_worktree_status,
        "platform": platform.platform(),
        "python": sys.version,
        "qualified_scope": (
            "H2/RHF/STO-3G energy, native gradient and native Hessian on Linux CPU"
        ),
        "experimental_accuracy_established": False,
        "spectroscopy_accuracy_established": False,
        "geometry_angstrom": coordinates.tolist(),
        "charge": 0,
        "multiplicity": 1,
        "energy_hartree": energy,
        "scf_converged": bool(electronic.converged),
        "finite_difference_step_bohr": step_bohr,
        "native_derivative_units": {
            "gradient": "hartree/bohr",
            "hessian": "hartree/bohr^2",
            "displacement_conversion_angstrom_per_bohr": float(lib.param.BOHR),
            "conversion_source": "actual pyscf.lib.param.BOHR",
        },
        "gradient_max_absolute_error": gradient_error,
        "hessian_max_absolute_error": hessian_error,
        "hessian_symmetry_error": symmetry_error,
        "hessian_translation_error": translation_error,
        "derivative_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "dependencies": sorted(
            [
                {"name": d.metadata["Name"], "version": d.version}
                for d in importlib.metadata.distributions()
            ],
            key=lambda d: d["name"].lower(),
        ),
    }
    (output / "qualification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "engine": "pyscf",
                "version": pyscf.__version__,
                "gradient_error": gradient_error,
                "hessian_error": hessian_error,
            }
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-worktree-status", choices=["clean", "modified"], default="clean"
    )
    args = parser.parse_args()
    qualify(
        args.output,
        args.image_identity,
        args.source_commit,
        args.source_worktree_status,
    )
