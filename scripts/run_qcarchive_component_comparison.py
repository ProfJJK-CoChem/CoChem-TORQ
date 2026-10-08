"""Run a bounded, unqualified NH3 B2PLYP computational component comparison.

This research example consumes captured MolSSI source bytes. It does not add a
production method, reference uncertainty, accuracy target, or qualification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

NOTEBOOK_SHA256 = "6358808518e39b6cae4b8375518b784838359b2daa76afc3b3e6d34f9f142427"
STDOUT_SHA256 = "e8e721e3443e7acdc585c9a88595b18dacb2837bfb404ceb0594b57065d8b4e3"
LICENSE_SHA256 = "8d857c958223fcba344ad384d87824bfd1cd5e1bfef804463f14042c48f82257"
SOURCE_URL = (
    "https://github.com/MolSSI/QCArchiveExamples/blob/"
    "60ee49a3dc881cd64d42840a2a5f3bf31f7b049f/cookbook/molecule_records.ipynb"
)
XC = "0.53*HF + 0.47*B88, 0.73*LYP"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    # Serialization must succeed before an immutable evidence file is created.
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with path.open("xb") as output:
        output.write(encoded)


def unique(pattern: str, text: str) -> str:
    found = re.findall(pattern, text, re.MULTILINE)
    if len(found) != 1:
        raise ValueError("Captured source field is missing or ambiguous")
    return str(found[0])


def load_reference(directory: Path) -> dict[str, Any]:
    """Verify original bytes and derive facts from the actual native output."""
    paths = {
        "source-notebook.ipynb": NOTEBOOK_SHA256,
        "source-stdout.txt": STDOUT_SHA256,
        "source-license.txt": LICENSE_SHA256,
    }
    blobs = {}
    for name, expected in paths.items():
        data = (directory / name).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Captured source digest mismatch: {name}")
        blobs[name] = data
    notebook = json.loads(blobs["source-notebook.ipynb"])
    outputs = notebook["cells"][7]["outputs"]
    mirrored = "".join(part for output in outputs for part in output.get("text", []))
    if mirrored.encode() != blobs["source-stdout.txt"]:
        raise ValueError("Native output does not match its original notebook cell")
    record_listing = "".join(
        part
        for output in notebook["cells"][3]["outputs"]
        for part in output.get("data", {}).get("text/plain", [])
    )
    if re.findall(r"ResultRecord\(id='(\d+)'", record_listing)[9] != "1847316":
        raise ValueError("Original notebook record selection changed")
    text = blobs["source-stdout.txt"].decode()
    if "Geometry (in Bohr), charge = 0, multiplicity = 1:" not in text:
        raise ValueError("Captured charge, multiplicity, or geometry units changed")
    geometry = re.findall(
        r"^\s*([NH])\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
        r"(-?\d+\.\d+)\s+(\d+\.\d+)\s*$",
        text,
        re.MULTILINE,
    )
    if [row[0] for row in geometry] != ["N", "H", "H", "H"]:
        raise ValueError("Captured atom order or geometry is incomplete")
    if not re.search(r"PAIRS\s+1\s+5\s+4\s+110\s+110\s+0", text):
        raise ValueError("Captured frozen-core convention is unavailable")
    for pattern in (
        r"Basis Set:\s+AUG-CC-PVTZ\s*$",
        r"0\.4700\s+XC_GGA_X_B88",
        r"0\.5300\s+HF\s*$",
        r"0\.7300\s+XC_GGA_C_LYP",
        r"0\.2700\s+MP2\s*$",
        r"Pruning Scheme\s*=\s*FLAT",
        r"Radial Points\s*=\s*100",
        r"Spherical Points\s*=\s*302",
    ):
        if not re.search(pattern, text, re.MULTILINE):
            raise ValueError("Captured B2PLYP recipe or grid convention is unavailable")
    patterns = {
        "orbital_scf_energy": r"DFT Reference Energy\s*=\s*(-?\d+\.\d+)",
        "mp2_same_spin_energy": r"^\s*Same-Spin Energy\s*=\s*(-?\d+\.\d+)",
        "mp2_opposite_spin_energy": r"^\s*Opposite-Spin Energy\s*=\s*(-?\d+\.\d+)",
        "mp2_correlation_energy": r"^\s*Correlation Energy\s*=\s*(-?\d+\.\d+)",
        "scaled_mp2_correlation": r"Scaled MP2 Correlation\s*=\s*(-?\d+\.\d+)",
        "double_hybrid_total_energy": (
            r"@Final double-hybrid DFT total energy\s*=\s*(-?\d+\.\d+)"
        ),
    }
    return {
        "geometry_bohr_as_printed": [list(row[:4]) for row in geometry],
        "masses_amu_as_printed": [row[4] for row in geometry],
        "source_values_hartree_as_printed": {
            name: unique(pattern, text) for name, pattern in patterns.items()
        },
        "source_stdout_sha256": STDOUT_SHA256,
        "source_notebook_sha256": NOTEBOOK_SHA256,
        "source_url": SOURCE_URL,
        "source_record_id": 1847316,
        "source_molecule_id": 2,
        "source_engine": "Psi4",
        "source_engine_version": None,
        "source_uncertainty": None,
        "benchmark_acceptance": False,
    }


def residuals(values: dict[str, float], reference: dict[str, Any]) -> dict[str, Any]:
    printed = reference["source_values_hartree_as_printed"]
    return {
        name: {
            "calculated_hartree": value,
            "source_psi4_hartree_as_printed": printed[
                "orbital_scf_energy" if name == "hybrid_reference_energy" else name
            ],
            "calculated_minus_source_hartree": value
            - float(
                printed[
                    "orbital_scf_energy" if name == "hybrid_reference_energy" else name
                ]
            ),
        }
        for name, value in values.items()
    }


def run_density_fitted(
    destination: Path, reference: dict[str, Any], memory_mb: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    pyscf = importlib.import_module("pyscf")
    gto = importlib.import_module("pyscf.gto")
    dft = importlib.import_module("pyscf.dft")
    df = importlib.import_module("pyscf.df")
    radi = importlib.import_module("pyscf.dft.radi")
    gen_grid = importlib.import_module("pyscf.dft.gen_grid")
    libxc = importlib.import_module("pyscf.dft.libxc")
    dfmp2 = importlib.import_module("pyscf.mp.dfmp2")
    destination.mkdir()
    geometry = reference["geometry_bohr_as_printed"]
    mol = gto.M(
        atom=[(row[0], [float(v) for v in row[1:4]]) for row in geometry],
        unit="Bohr",
        charge=0,
        spin=0,
        basis="aug-cc-pvtz",
        cart=False,
        max_memory=memory_mb,
        verbose=4,
        output=str(destination / "pyscf-native.log"),
    )
    try:
        mf = dft.RKS(mol).density_fit(auxbasis="aug-cc-pvtz-jkfit")
        mf.xc = XC
        mf.max_memory = memory_mb
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 100
        mf.chkfile = str(destination / "converged-rks.chk")
        mf.grids.atom_grid = (100, 302)
        mf.grids.prune = None
        mf.grids.radi_method = radi.treutler_ahlrichs
        mf.grids.radii_adjust = radi.treutler_atomic_radii_adjust
        mf.grids.becke_scheme = gen_grid.original_becke
        mf.small_rho_cutoff = 0
        iterations = []

        def observe(environment: dict[str, Any]) -> None:
            iterations.append(
                {
                    "cycle": int(environment["cycle"]),
                    "energy_hartree": float(environment["e_tot"]),
                    "orbital_gradient_norm": float(environment["norm_gorb"]),
                    "density_change_norm": float(environment["norm_ddm"]),
                }
            )

        mf.callback = observe
        orbital_energy = float(mf.kernel())
        if not mf.converged or not np.isfinite(orbital_energy):
            raise RuntimeError("Actual SCF failed; no correlated result is substituted")
        if mol.nao_nr() != 115 or mf.with_df.auxmol.nao_nr() != 242:
            raise RuntimeError("Actual primary/JKFIT count differs from the source")
        ri = df.DF(mol, auxbasis="aug-cc-pvtz-ri")
        ri.max_memory = memory_mb
        ri.build()
        if ri.auxmol.nao_nr() != 244:
            raise RuntimeError("Actual RI basis count differs from the source")
        post = dfmp2.DFMP2(
            mf,
            frozen=1,
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
            mo_energy=mf.mo_energy,
        )
        post.with_df = ri
        post.max_memory = memory_mb
        if post.with_df is mf.with_df:
            raise RuntimeError("MP2 must use its separate RI auxiliary object")
        post.kernel(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff, with_t2=False)
        values = {
            "orbital_scf_energy": orbital_energy,
            "mp2_same_spin_energy": float(post.e_corr_ss),
            "mp2_opposite_spin_energy": float(post.e_corr_os),
            "mp2_correlation_energy": float(post.e_corr),
        }
        values["scaled_mp2_correlation"] = 0.27 * values["mp2_correlation_energy"]
        values["double_hybrid_total_energy"] = (
            orbital_energy + values["scaled_mp2_correlation"]
        )
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("Actual electronic energy is nonfinite")
        arrays = {}
        bases = {}
        for label, body in (
            ("orbital", mol),
            ("jkfit", mf.with_df.auxmol),
            ("mp2_ri", ri.auxmol),
        ):
            bases[label] = body._basis
            for field in ("_atm", "_bas", "_env"):
                arrays[label + field] = getattr(body, field)
        arrays.update(
            mo_coeff=mf.mo_coeff,
            mo_energy=mf.mo_energy,
            mo_occ=mf.mo_occ,
            grid_coords_bohr=mf.grids.coords,
            grid_weights_bohr3=mf.grids.weights,
        )
        np.savez_compressed(destination / "actual-basis-orbitals-grid.npz", **arrays)
        write_json(destination / "actual-expanded-bases.json", bases)
        result = {
            "status": "completed unqualified research comparison",
            "versions": {
                "pyscf": pyscf.__version__,
                "libxc": libxc.libxc_version(),
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            },
            "actual_scf_converged": bool(mf.converged),
            "scf_iterations": iterations,
            "actual_counts": {
                "orbital_ao": int(mol.nao_nr()),
                "jkfit_auxiliary": int(mf.with_df.auxmol.nao_nr()),
                "mp2_ri_auxiliary": int(ri.auxmol.nao_nr()),
                "grid_points": len(mf.grids.weights),
                "occupied_orbitals": int(np.count_nonzero(mf.mo_occ)),
                "active_mp2_occupied": int(post.nocc),
                "frozen_occupied_orbitals": 1,
            },
            "mp2_used_explicit_converged_rks_orbital_energies": bool(
                np.array_equal(post.mo_energy, mf.mo_energy)
            ),
            "separate_jkfit_ri_objects": post.with_df is not mf.with_df,
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256_c_order": digest(np.ascontiguousarray(value).tobytes()),
                }
                for name, value in arrays.items()
            },
            "components_and_residuals": residuals(values, reference),
            "comparison_tolerance": None,
            "benchmark_acceptance": False,
            "scientific_profile_qualification": False,
            "curator_attestation": False,
        }
        write_json(destination / "comparison.json", result)
        return result
    finally:
        mol.stdout.close()


def run_torq(
    destination: Path, reference: dict[str, Any], memory_mb: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    checkout_source = Path(__file__).resolve().parents[1] / "src"
    if (checkout_source / "cochem_torq/engines/revdsd.py").is_file():
        sys.path.insert(0, str(checkout_source))
    engine = importlib.import_module("cochem_torq.engines.revdsd")
    source = Path(engine.__file__)
    before = digest(source.read_bytes())
    recipe = engine.ResearchRecipe(
        name="qca-nh3-b2plyp-component-reproduction-research",
        orbital_xc=XC,
        energy_xc=XC,
        opposite_spin_scale=0.27,
        same_spin_scale=0.27,
        frozen_occupied_orbitals=(0,),
        source_evidence=(SOURCE_URL + "#cells-5-and-7", "sha256:" + STDOUT_SHA256),
    )
    evaluator = engine.ExperimentalDoubleHybrid(
        recipe,
        basis="aug-cc-pvtz",
        grid_level=5,
        scf_energy_tolerance=1e-10,
        scf_gradient_tolerance=1e-8,
        max_cycle=100,
        max_memory_mb=memory_mb,
        threads=1,
        check_reference_stability=True,
    )
    geometry = reference["geometry_bohr_as_printed"]
    native = evaluator.evaluate(
        [row[0] for row in geometry],
        np.array([[float(v) for v in row[1:4]] for row in geometry]),
        charge=0,
        multiplicity=1,
        artifact_directory=destination,
    )
    after = digest(source.read_bytes())
    if before != after:
        raise RuntimeError("TORQ engine source changed during native evaluation")
    components = native["components"]
    values = {
        "orbital_scf_energy": float(native["orbital_generating_energy_hartree"]),
        "hybrid_reference_energy": float(components["hybrid_reference_hartree"]),
        "mp2_same_spin_energy": float(components["same_spin_pt2_hartree"]),
        "mp2_opposite_spin_energy": float(components["opposite_spin_pt2_hartree"]),
        "mp2_correlation_energy": float(
            components["same_spin_pt2_hartree"]
            + components["opposite_spin_pt2_hartree"]
        ),
        "scaled_mp2_correlation": float(components["scaled_pt2_hartree"]),
        "double_hybrid_total_energy": float(native["energy_hartree"]),
    }
    result = {
        "status": "completed unqualified existing TORQ energy comparison",
        "recipe": recipe.name,
        "recipe_sha256": recipe.fingerprint,
        "engine_source_sha256_before": before,
        "engine_source_sha256_after": after,
        "engine_source_unchanged": True,
        "native_artifact_directory": str(
            Path(native["artifact_directory"]).relative_to(destination.parent)
        ),
        "actual_scf_converged": native["scf_converged"],
        "orbital_reference_stability": native["reference_stability"],
        "components_and_residuals": residuals(values, reference),
        "convention_differences": [
            "TORQ conventional SCF/PT2 integrals; Psi4 used JKFIT/RI density fitting",
            "TORQ grid_level5/default pruning; Psi4 printed flat 100x302 Treutler",
            "Historical Psi4 version, basis bytes and partition identity absent",
        ],
        "comparison_tolerance": None,
        "benchmark_acceptance": False,
        "scientific_profile_qualification": False,
        "curator_attestation": False,
        "native_gradient_or_hessian_computed": False,
        "exact_revdsd_claimed": False,
    }
    write_json(destination.parent / "torq-comparison.json", result)
    return result


def native_worker(args: argparse.Namespace) -> int:
    destination = args.output_dir / "native"
    destination.mkdir()
    try:
        get_affinity = getattr(os, "sched_getaffinity", None)
        set_affinity = getattr(os, "sched_setaffinity", None)
        if get_affinity is None or set_affinity is None:
            raise RuntimeError("This bounded example requires Linux CPU affinity")
        cpu = min(get_affinity(0))
        set_affinity(0, {cpu})
        if metadata.version("pyscf") != "2.14.0":
            raise RuntimeError("Use the declared calculations dependency PySCF2.14.0")
        reference = load_reference(args.reference_dir)
        memory_mb = min(768, max(128, args.max_rss_mib * 3 // 4))
        lib = importlib.import_module("pyscf.lib")
        lib.num_threads(1)
        write_json(
            destination / "scientific-input.json",
            {
                **reference,
                "method": "B2PLYP",
                "orbital_xc": XC,
                "energy_xc": XC,
                "opposite_spin_scale": 0.27,
                "same_spin_scale": 0.27,
                "frozen_occupied_orbitals": [0],
                "charge": 0,
                "multiplicity": 1,
                "basis": "aug-cc-pvtz",
                "jkfit": "aug-cc-pvtz-jkfit",
                "mp2_ri": "aug-cc-pvtz-ri",
                "standalone_grid": "unpruned100x302Treutler/originalBecke",
                "torq_grid": "level5/default pruning",
                "cpu_affinity": [cpu],
                "threads": 1,
                "pyscf_memory_mb": memory_mb,
                "geometry_optimization": False,
                "native_derivatives": False,
            },
        )
        run_density_fitted(destination / "density-fitted", reference, memory_mb)
        gc.collect()
        # Return freed libc arenas before the independent conventional evaluation.
        ctypes = importlib.import_module("ctypes")
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except AttributeError:
            pass
        run_torq(destination / "torq", reference, memory_mb)
        return 0
    except Exception as error:
        write_json(
            destination / "failure.json",
            {"status": "failed", "type": type(error).__name__, "message": str(error)},
        )
        raise


def supervise(args: argparse.Namespace) -> int:
    """Observe the original owned child, wait it, and retain exact failed evidence."""
    psutil = importlib.import_module("psutil")
    output = args.output_dir.resolve()
    output.mkdir(mode=0o700, parents=True)
    args.output_dir = output
    if shutil.disk_usage(output).free < 256 * 1024**2:
        write_json(
            output / "failure.json",
            {"status": "resource rejected", "reason": "Less than 256MiB free scratch"},
        )
        return 2
    try:
        reference = load_reference(args.reference_dir)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "source rejected",
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        return 2
    write_json(output / "verified-source.json", reference)
    scratch = output / "scratch"
    scratch.mkdir()
    worker_script = Path(__file__).resolve()
    worker_sha = digest(worker_script.read_bytes())
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TMPDIR": str(scratch),
        }
    )
    command = [
        sys.executable,
        "-I",
        "-B",
        str(worker_script),
        "--native-worker",
        "--reference-dir",
        str(args.reference_dir.resolve()),
        "--output-dir",
        str(output),
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--max-rss-mib",
        str(args.max_rss_mib),
    ]
    started = time.monotonic()
    peak = 0
    reason = None
    observed: dict[tuple[int, float], str] = {}
    with (output / "worker.log").open("xb") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        actual = psutil.Process(child.pid)
        created = float(actual.create_time())
        write_json(
            output / "process-start.json",
            {
                "pid": child.pid,
                "create_time": created,
                "worker_script_sha256": worker_sha,
                "wall_budget_seconds": args.max_wall_seconds,
                "sampled_owned_tree_rss_limit_bytes": args.max_rss_mib * 1024**2,
            },
        )
        try:
            while child.poll() is None:
                try:
                    if actual.create_time() != created:
                        raise RuntimeError("Original worker identity changed")
                    rss = int(actual.memory_info().rss)
                    for descendant in actual.children(recursive=True):
                        try:
                            observed[(descendant.pid, descendant.create_time())] = (
                                descendant.name()
                            )
                            rss += int(descendant.memory_info().rss)
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, rss)
                    if rss > args.max_rss_mib * 1024**2:
                        reason = "sampled owned-tree RSS exceeded its budget"
                    elif time.monotonic() - started > args.max_wall_seconds:
                        reason = "wall time exceeded its budget"
                    if reason is not None:
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.1)
        except KeyboardInterrupt:
            reason = "caller interrupted owned calculation"
        except Exception as error:
            reason = f"supervisor failed: {type(error).__name__}: {error}"
        finally:
            if reason is not None and child.poll() is None:
                identity = psutil.Process(child.pid)
                if float(identity.create_time()) != created:
                    raise RuntimeError("Cannot terminate a different worker identity")
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            exit_code = child.wait()
    remaining = []
    for (pid, start), name in observed.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == start and process.is_running():
                remaining.append({"pid": pid, "create_time": start, "name": name})
        except psutil.NoSuchProcess:
            pass
    inventory = [
        {
            "path": str(path.relative_to(output)),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path.read_bytes()),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    receipt = {
        "status": "completed" if exit_code == 0 and reason is None else "failed",
        "worker_pid": child.pid,
        "worker_create_time": created,
        "actual_original_worker_waited": True,
        "exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - started,
        "peak_sampled_owned_tree_rss_bytes": peak,
        "budget_stop_reason": reason,
        "remaining_observed_descendants": remaining,
        "worker_script_unchanged": digest(worker_script.read_bytes()) == worker_sha,
        "native_output_inventory": inventory,
        "benchmark_acceptance": False,
        "scientific_profile_qualification": False,
        "curator_attestation": False,
    }
    if remaining or not receipt["worker_script_unchanged"]:
        receipt["status"] = "failed"
    write_json(output / "execution-receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(output / "execution-receipt.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks/published-values/qcarchive-nh3",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-wall-seconds", type=int, default=300)
    parser.add_argument("--max-rss-mib", type=int, default=1024)
    parser.add_argument("--verify-source-only", action="store_true")
    parser.add_argument("--native-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_wall_seconds <= 300 or not 256 <= args.max_rss_mib <= 1024:
        parser.error("Use wall1..300seconds and sampled RSS256..1024MiB")
    if args.verify_source_only:
        reference = load_reference(args.reference_dir)
        print(
            json.dumps(
                {"status": "original source bytes verified", **reference},
                sort_keys=True,
            )
        )
        return 0
    if args.output_dir is None:
        parser.error("A fresh --output-dir is required for actual calculations")
    if args.native_worker:
        return native_worker(args)
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
