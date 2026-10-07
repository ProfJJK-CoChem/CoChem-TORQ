"""Delta-Learning (Delta-ML) Architecture for CoChem-TORQ.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic physics, dynamic baselines, and exact unit harmonization.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.constants as const
import torch
from mendeleev import element

from Libraries.cochem_torq_inference_errors import BaselineExecutionError
from Libraries.cochem_torq_inference_schemas import DeltaMLConfig

try:
    from cochem_base.exceptions import DispersionIntegrationError
    from cochem_base.schemas import DeltaMLDispersionConfig
except ImportError:
    class DispersionIntegrationError(RuntimeError):
        pass
    DeltaMLDispersionConfig = None

logger = logging.getLogger("CoChem-TORQ.DeltaML")

# Conversion factors via scipy.constants
HARTREE_TO_EV: float = float(const.value("Hartree energy in eV"))  # ~27.211386245981 eV [D]
BOHR_TO_ANGSTROM: float = float(const.value("Bohr radius") * 1e10)  # ~0.529177210544 Angstrom [D]
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: float = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # ~51.422067511 eV/Angstrom [D]
KCAL_PER_MOL_TO_EV: float = float(const.calorie * 1000.0 / (const.N_A * const.eV))  # ~0.0433641 eV [D]


class UnitHarmonizer:
    """Rigorous unit harmonization utilities between quantum chemistry and internal standard. [D]"""

    @staticmethod
    def convert_energy(
        value: float | torch.Tensor,
        from_unit: str,
        to_unit: str = "eV",
    ) -> float | torch.Tensor:
        """Convert scalar energy between standard computational chemistry units. [D]"""
        if from_unit == to_unit:
            return value

        # Convert to eV first
        if from_unit == "eV":
            val_ev = value
        elif from_unit == "Hartree":
            val_ev = value * HARTREE_TO_EV
        elif from_unit == "kcal/mol":
            val_ev = value * KCAL_PER_MOL_TO_EV
        else:
            raise ValueError(f"Unsupported energy unit: {from_unit}")

        # Convert from eV to target unit
        if to_unit == "eV":
            return val_ev
        elif to_unit == "Hartree":
            return val_ev / HARTREE_TO_EV
        elif to_unit == "kcal/mol":
            return val_ev / KCAL_PER_MOL_TO_EV
        else:
            raise ValueError(f"Unsupported target energy unit: {to_unit}")

    @staticmethod
    def convert_forces(
        forces: torch.Tensor,
        from_length_unit: str = "Bohr",
        to_length_unit: str = "Angstrom",
        from_energy_unit: str = "Hartree",
        to_energy_unit: str = "eV",
    ) -> torch.Tensor:
        """Convert force tensors between atomic units and internal standard (eV/Angstrom). [D]"""
        # F = -dE / dR.
        # factor = (E_conversion_factor) / (R_conversion_factor)
        factor = 1.0
        if from_energy_unit == "Hartree" and to_energy_unit == "eV":
            factor *= HARTREE_TO_EV
        elif from_energy_unit == "eV" and to_energy_unit == "Hartree":
            factor /= HARTREE_TO_EV

        if from_length_unit == "Bohr" and to_length_unit == "Angstrom":
            factor /= BOHR_TO_ANGSTROM
        elif from_length_unit == "Angstrom" and to_length_unit == "Bohr":
            factor *= BOHR_TO_ANGSTROM

        return forces * factor


class BaselinePhysicsEngine:
    """Base contract for genuine physical baseline calculation engines. [M]"""

    def calculate(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: Sequence[int],
    ) -> tuple[float, torch.Tensor]:
        """Compute baseline potential energy (eV) and forces (eV/Angstrom). [M]"""
        raise BaselineExecutionError("Base physics engine has no underlying calculator defined", method="NONE")



class GFN2Result(dict):
    """Result dictionary from GFN2-xTB calculations supporting dict access and tuple unpacking [M]."""

    def __init__(
        self,
        *args: Any,
        energy_ev: float,
        forces: torch.Tensor | None = None,
        charge: int = 0,
        uhf: int = 0,
        multiplicity: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (not np.isfinite(energy_ev) or forces is None or forces.ndim != 2 or forces.shape[1] != 3
                or not len(forces) or not torch.isfinite(forces).all()):
            raise ValueError("GFN2Result requires a finite calculated energy and forces.")
        self["energy_ev"] = energy_ev
        self["energy"] = energy_ev
        self["forces"] = forces
        self["forces_ev_angstrom"] = forces
        self["charge"] = charge
        self["uhf"] = uhf
        self["multiplicity"] = multiplicity

    def __iter__(self) -> Any:
        # Support tuple unpacking: energy, forces = engine.calculate(...)
        yield self["energy_ev"]
        yield self["forces"]

    def __getitem__(self, item: Any) -> Any:
        if item == 0:
            return self["energy_ev"]
        if item == 1:
            return self["forces"]
        return super().__getitem__(item)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'GFN2Result' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class GFN2xTBEngine(BaselinePhysicsEngine):
    """Adapter for physical GFN2-xTB semi-empirical calculations [M].

    Prioritizes direct in-memory evaluation via the xtb-python C-API bindings,
    with subprocess fallback to standalone CLI xtb.
    Validates physical electron-spin parity:
    (N_e - 2S) % 2 == 0 and 2S >= 0 and N_e > 0.
    Maps uhf = multiplicity - 1 (unpaired electrons 2S).
    """

    def __init__(self) -> None:
        self.xtb_available = False
        self.has_xtb_python = False
        self._check_environment()

    def _check_environment(self) -> None:
        """Check availability of actual GFN2-xTB implementations only."""
        try:
            from xtb.interface import Calculator, Param  # noqa: F401
            self.xtb_available = True
            self.has_xtb_python = True
            return
        except ImportError:
            self.has_xtb_python = False

        if shutil.which("xtb") is not None:
            self.xtb_available = True
            return

        self.xtb_available = False

    @staticmethod
    def _map_spin_to_uhf(multiplicity: int) -> int:
        """Map spin multiplicity M (2S + 1) to number of unpaired electrons uhf = 2S."""
        if multiplicity < 1:
            raise ValueError(f"Invalid spin multiplicity: {multiplicity}. Multiplicity must be >= 1.")
        return multiplicity - 1

    def validate_electron_parity(
        self,
        atoms_or_z: Any,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> bool:
        """Validate physical electron-spin parity before computation [M].

        Z_tot = sum(Z_i)
        N_e = Z_tot - charge
        2S = multiplicity - 1
        Enforces (N_e - 2S) % 2 == 0, 2S >= 0, and N_e > 0.
        """
        if hasattr(atoms_or_z, "numbers"):
            atomic_numbers = [int(z) for z in atoms_or_z.numbers]
        elif isinstance(atoms_or_z, list | tuple | np.ndarray | torch.Tensor):
            if len(atoms_or_z) > 0 and isinstance(atoms_or_z[0], str):
                atomic_numbers = [int(element(s).atomic_number) for s in atoms_or_z]
            else:
                atomic_numbers = [int(z) for z in atoms_or_z]
        else:
            raise ValueError(f"Cannot extract atomic numbers from: {type(atoms_or_z)}")

        z_tot = sum(atomic_numbers)
        n_e = z_tot - int(charge)
        two_s = self._map_spin_to_uhf(multiplicity)

        if n_e <= 0:
            raise ValueError(
                f"Invalid electron count: N_e={n_e} (Z_tot={z_tot}, charge={charge}). Electron count must be > 0."
            )
        if two_s < 0:
            raise ValueError(f"Invalid unpaired electron count: 2S={two_s} from multiplicity {multiplicity}.")

        if (n_e - two_s) % 2 != 0:
            raise ValueError(
                f"Electron parity violation: system has {n_e} electrons (Z_tot={z_tot}, charge={charge}) "
                f"which cannot support spin multiplicity {multiplicity} (unpaired electrons 2S={two_s}). "
                f"(N_e - 2S) must be an even integer."
            )
        return True

    def calculate(
        self,
        atoms: Any = None,
        charge: int = 0,
        uhf: int | None = None,
        scratch_dir: str | Path | None = None,
        multiplicity: int | None = None,
        **kwargs: Any,
    ) -> GFN2Result:
        """Compute baseline potential energy and forces via in-memory xtb-python or CLI xtb [M].

        Supports backwards-compatible signatures:
        calculate(coordinates, atomic_numbers, ...)
        calculate(atoms, charge=..., uhf=..., scratch_dir=..., multiplicity=...)
        """
        coords_input = None
        z_input = None

        if isinstance(atoms, torch.Tensor) or (isinstance(atoms, np.ndarray) and atoms.ndim == 2 and atoms.shape[1] == 3):
            coords_input = atoms
            if isinstance(charge, list | tuple | np.ndarray | Sequence) and not isinstance(charge, int | float):
                z_input = charge
                charge = int(kwargs.get("charge", 0))
                multiplicity = int(kwargs.get("multiplicity", 1))
            else:
                z_input = kwargs.get("atomic_numbers")
        elif atoms is not None and hasattr(atoms, "positions") and hasattr(atoms, "numbers"):
            coords_input = torch.tensor(atoms.positions, dtype=torch.float64)
            z_input = list(atoms.numbers)
            if hasattr(atoms, "info"):
                if charge == 0 and "charge" in atoms.info:
                    charge = int(atoms.info["charge"])
                if uhf is None and "uhf" in atoms.info:
                    uhf = int(atoms.info["uhf"])
                elif uhf is None and multiplicity is None and "multiplicity" in atoms.info:
                    multiplicity = int(atoms.info["multiplicity"])
        elif "coordinates" in kwargs and "atomic_numbers" in kwargs:
            coords_input = kwargs["coordinates"]
            z_input = kwargs["atomic_numbers"]

        if coords_input is None or z_input is None:
            raise ValueError("calculate() requires atoms or (coordinates, atomic_numbers)")

        if uhf is not None:
            two_s = int(uhf)
            effective_multiplicity = two_s + 1
        elif multiplicity is not None:
            effective_multiplicity = int(multiplicity)
            two_s = self._map_spin_to_uhf(effective_multiplicity)
        else:
            effective_multiplicity = int(kwargs.get("multiplicity", 1))
            two_s = self._map_spin_to_uhf(effective_multiplicity)

        # Validate electron parity before computation [M]
        self.validate_electron_parity(z_input, charge=charge, multiplicity=effective_multiplicity)

        if not self.xtb_available:
            raise BaselineExecutionError(
                "TORQ_BASELINE_UNAVAILABLE: GFN2-xTB executable or xtb-python library not found in runtime environment",
                method="GFN2-xTB",
                diagnostics={"atomic_count": len(z_input)},
            )

        coords_np = coords_input.detach().cpu().numpy() if isinstance(coords_input, torch.Tensor) else np.asarray(coords_input, dtype=np.float64)
        z_np = np.array([int(z) for z in z_input], dtype=np.int32)
        device = coords_input.device if isinstance(coords_input, torch.Tensor) else "cpu"
        dtype = coords_input.dtype if isinstance(coords_input, torch.Tensor) else torch.float64

        if coords_np.shape != (len(z_np), 3) or not np.isfinite(coords_np).all():
            raise ValueError("GFN2-xTB requires finite coordinates with shape (N, 3).")
        failures = []
        if self.has_xtb_python:
            try:
                from xtb.interface import Calculator, Param
                calc = Calculator(Param.GFN2xTB, z_np, coords_np / BOHR_TO_ANGSTROM,
                                  charge=float(charge), uhf=two_s)
                result = calc.singlepoint()
                forces = -np.asarray(result.get_gradient()) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
                return GFN2Result(energy_ev=float(result.get_energy()) * HARTREE_TO_EV,
                                  forces=torch.tensor(forces, dtype=dtype, device=device),
                                  charge=charge, multiplicity=effective_multiplicity, uhf=two_s)
            except Exception as exc:
                failures.append(f"xtb-python: {exc}")
        executable = shutil.which("xtb")
        if executable is not None:
            from ase import Atoms
            from ase.io import write
            # A unique child directory prevents stale files from being parsed and
            # preserves all caller-owned files, including supplied scratch data.
            if scratch_dir is not None:
                Path(scratch_dir).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="torq_xtb_", dir=scratch_dir) as work:
                workdir = Path(work)
                xyz_path = workdir / "mol.xyz"
                write(str(xyz_path), Atoms(numbers=z_np, positions=coords_np), format="xyz")
                try:
                    result = subprocess.run(
                        [executable, str(xyz_path), "--gfn", "2", "--grad", "--chrg", str(charge), "--uhf", str(two_s)],
                        cwd=workdir, capture_output=True, text=True, check=True,
                    )
                    matches = re.findall(r"TOTAL ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)", result.stdout)
                    if not matches:
                        raise ValueError("xTB output contains no TOTAL ENERGY.")
                    energy_hartree = float(matches[-1].replace("D", "E").replace("d", "e"))
                    lines = (workdir / "gradient").read_text().splitlines()
                    # Turbomole $grad: marker, cycle/energy, N coordinates, N gradients.
                    if not lines or lines[0].strip() != "$grad":
                        raise ValueError("Unrecognized xTB gradient file format.")
                    grad_lines = lines[2 + len(z_np):2 + 2 * len(z_np)]
                    gradients = np.asarray([[float(v.replace("D", "E").replace("d", "e")) for v in line.split()] for line in grad_lines])
                    if gradients.shape != coords_np.shape or not np.isfinite(gradients).all():
                        raise ValueError("xTB gradient is missing, malformed, or non-finite.")
                    return GFN2Result(energy_ev=energy_hartree * HARTREE_TO_EV,
                                      forces=torch.tensor(-gradients * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM, dtype=dtype, device=device),
                                      charge=charge, multiplicity=effective_multiplicity, uhf=two_s)
                except Exception as exc:
                    failures.append(f"xtb CLI: {exc}")
        raise BaselineExecutionError(
            "GFN2-xTB calculation unavailable or failed; no substitute Hamiltonian used.",
            method="GFN2-xTB", diagnostics={"failures": failures},
        )


class PM6Engine(BaselinePhysicsEngine):
    """Adapter for semi-empirical PM6 Hamiltonian calculations. [M]"""

    def calculate(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: Sequence[int],
    ) -> tuple[float, torch.Tensor]:
        import os
        import shutil
        import subprocess
        import tempfile

        from ase import Atoms
        from ase.io import write

        if shutil.which("mopac") is None:
            raise BaselineExecutionError(
                "TORQ_BASELINE_UNAVAILABLE: PM6 solver not found in runtime environment",
                method="PM6",
                diagnostics={"atomic_count": len(atomic_numbers)},
            )

        atoms = Atoms(numbers=atomic_numbers, positions=coordinates.detach().cpu().numpy())
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                xyz_path = os.path.join(tmpdir, "mol.xyz")
                write(xyz_path, atoms, format="xyz")

                mop_path = os.path.join(tmpdir, "mol.mop")
                with open(mop_path, "w") as f:
                    f.write("PM6 1SCF GRADIENTS\nTitle\n\n")
                    for i in range(len(atomic_numbers)):
                        pos = atoms.positions[i]
                        f.write(f"{atoms.get_chemical_symbols()[i]} {pos[0]} 1 {pos[1]} 1 {pos[2]} 1\n")

                subprocess.run(["mopac", mop_path], cwd=tmpdir, capture_output=True, text=True, check=True)

                out_path = os.path.join(tmpdir, "mol.out")
                energy_ev = None
                forces_ev_angstrom = []
                reading_grad = False
                with open(out_path) as f:
                    for line in f:
                        if "FINAL HEAT OF FORMATION" in line:
                            kcal = float(line.split()[5])
                            energy_ev = kcal * KCAL_PER_MOL_TO_EV
                        elif "FINAL POINT AND DERIVATIVES" in line:
                            reading_grad = True
                        elif reading_grad and len(line.split()) == 8:
                            parts = line.split()
                            try:
                                fx = -float(parts[5]) * KCAL_PER_MOL_TO_EV
                                fy = -float(parts[6]) * KCAL_PER_MOL_TO_EV
                                fz = -float(parts[7]) * KCAL_PER_MOL_TO_EV
                                forces_ev_angstrom.append([fx, fy, fz])
                            except ValueError:
                                pass

                if energy_ev is None or not np.isfinite(energy_ev):
                    raise ValueError("MOPAC output is missing a finite heat of formation.")
                if len(forces_ev_angstrom) != len(atomic_numbers) or not np.isfinite(forces_ev_angstrom).all():
                    raise ValueError("Failed to parse all forces from MOPAC output.")

                return energy_ev, torch.tensor(forces_ev_angstrom, dtype=coordinates.dtype, device=coordinates.device)

            except Exception as e:
                raise BaselineExecutionError(
                    f"TORQ_BASELINE_EXEC_FAIL: PM6 execution failed: {e}",
                    method="PM6",
                )


class EMTBaselineEngine(BaselinePhysicsEngine):
    """Adapter for Effective Medium Theory (EMT) baseline calculations. [M]"""

    def calculate(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: Sequence[int],
    ) -> tuple[float, torch.Tensor]:
        try:
            import ase  # noqa: F401
            from ase import Atoms
            from ase.calculators.emt import EMT
        except ImportError:
            raise BaselineExecutionError(
                "TORQ_BASELINE_UNAVAILABLE: ASE EMT library not found",
                method="EMT",
            )

        atoms = Atoms(numbers=atomic_numbers, positions=coordinates.detach().cpu().numpy())
        atoms.calc = EMT()

        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        except Exception as e:
            raise BaselineExecutionError(
                f"TORQ_BASELINE_EXEC_FAIL: EMT calculation failed: {e}",
                method="EMT",
            )

        return float(energy), torch.tensor(forces, dtype=coordinates.dtype, device=coordinates.device)


class LennardJonesBaselineEngine(BaselinePhysicsEngine):
    """Authentic physical Lennard-Jones non-bonded baseline engine with Lorentz-Berthelot mixing. [M]"""

    def __init__(self, parameters: dict[int, tuple[float, float]] | None = None) -> None:
        """Require explicitly supplied (sigma in Angstrom, epsilon in eV) parameters.

        The caller must retain the force-field source and atom typing provenance.
        No atomic-radius estimate or default well depth is substituted.
        """
        self.parameters = dict(parameters or {})
        for sigma, epsilon in self.parameters.values():
            if not np.isfinite([sigma, epsilon]).all() or sigma <= 0 or epsilon <= 0:
                raise ValueError("Lennard-Jones parameters must be finite and positive.")

    def _get_params(self, z: int) -> tuple[float, float]:
        if z not in self.parameters:
            raise BaselineExecutionError(
                f"Explicit Lennard-Jones parameters unavailable for atomic number {z}.", method="LennardJones"
            )
        return self.parameters[z]

    def calculate(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: Sequence[int],
    ) -> tuple[float, torch.Tensor]:
        """Evaluate exact physical Lennard-Jones potential and analytical autograd forces. [D]"""
        n_atoms = len(atomic_numbers)
        if n_atoms == 0 or coordinates.shape != (n_atoms, 3) or not torch.isfinite(coordinates).all():
            raise ValueError("A nonempty finite molecular geometry is required.")
        for z in atomic_numbers:
            self._get_params(int(z))
        if n_atoms < 2:
            return 0.0, torch.zeros_like(coordinates)

        coords = coordinates.clone().detach().requires_grad_(True)
        device = coords.device
        dtype = coords.dtype

        # Build pairwise sigma_ij and epsilon_ij tensors
        sigmas = []
        epsilons = []
        for z in atomic_numbers:
            s, e = self._get_params(int(z))
            sigmas.append(s)
            epsilons.append(e)

        sig = torch.tensor(sigmas, dtype=dtype, device=device)
        eps = torch.tensor(epsilons, dtype=dtype, device=device)

        # Lorentz-Berthelot mixing: sigma_ij = (sigma_i + sigma_j)/2, eps_ij = sqrt(eps_i * eps_j)
        sig_ij = 0.5 * (sig.unsqueeze(1) + sig.unsqueeze(0))
        eps_ij = torch.sqrt(eps.unsqueeze(1) * eps.unsqueeze(0))

        # Coordinate differences
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 3]
        dist = torch.norm(diff, dim=-1)  # [N, N]

        # Upper triangular mask (i < j)
        mask = torch.triu(torch.ones((n_atoms, n_atoms), dtype=torch.bool, device=device), diagonal=1)

        # Guard against zero distance
        if torch.any(dist[mask] <= 0):
            raise ValueError("Coincident atoms are invalid for the Lennard-Jones potential.")
        clamped_dist = torch.where(mask, dist, torch.ones_like(dist))
        sr6 = (sig_ij / clamped_dist) ** 6
        sr12 = sr6 ** 2

        lj_pairs = 4.0 * eps_ij * (sr12 - sr6)
        energy = torch.sum(torch.where(mask, lj_pairs, torch.zeros_like(lj_pairs)))

        # Analytical conservative forces via exact autograd: F = -dE / dR
        grads = torch.autograd.grad(energy, coords, create_graph=False)[0]
        forces = -grads

        return float(energy.item()), forces.detach()


class DeltaMLEngine:
    """Delta-Learning engine managing difference mapping and target high-level reconstruction. [M]"""

    def __init__(
        self,
        config: DeltaMLConfig | DeltaMLDispersionConfig | dict[str, Any] | None = None,
        baseline_engine: BaselinePhysicsEngine | None = None,
        dispersion_config: DeltaMLDispersionConfig | None = None,
        use_d3_dispersion: bool | None = None,
    ) -> None:
        self.config = config
        if baseline_engine is not None:
            self.baseline_engine = baseline_engine
        elif isinstance(config, DeltaMLConfig):
            if config.baseline_method == "GFN2-xTB":
                self.baseline_engine = GFN2xTBEngine()
            elif config.baseline_method == "PM6":
                self.baseline_engine = PM6Engine()
            elif config.baseline_method == "LennardJones":
                self.baseline_engine = LennardJonesBaselineEngine()
            elif config.baseline_method == "EMT":
                self.baseline_engine = EMTBaselineEngine()
            else:
                raise ValueError(f"Unknown baseline method: {config.baseline_method}")
        else:
            raise ValueError("An explicitly configured baseline engine is required for Delta-ML.")

        # D3 Dispersion augmentation (REQ-TORQ-DISP-060 [D], [M])
        self.dispersion_config = dispersion_config
        if DeltaMLDispersionConfig is not None and isinstance(config, DeltaMLDispersionConfig):
            self.dispersion_config = config

        if use_d3_dispersion is not None:
            self.use_d3_dispersion = bool(use_d3_dispersion)
        elif self.dispersion_config is not None:
            self.use_d3_dispersion = bool(getattr(self.dispersion_config, "use_d3_dispersion", True))
        elif hasattr(config, "use_d3_dispersion"):
            self.use_d3_dispersion = bool(getattr(config, "use_d3_dispersion", False))
        else:
            self.use_d3_dispersion = False

        self.d3_layer = None
        if self.use_d3_dispersion:
            try:
                from Libraries.cochem_torq_dispersion_d3 import DispersionD3Layer
                from Libraries.cochem_torq_inference_schemas import DispersionD3Config
            except ImportError:
                from cochem_torq_dispersion_d3 import DispersionD3Layer
                from cochem_torq_inference_schemas import DispersionD3Config

            s6 = 1.0
            s8 = 0.0
            if self.dispersion_config is not None:
                s6 = float(getattr(self.dispersion_config, "s6_scale", 1.0))
                s8 = float(getattr(self.dispersion_config, "s8_scale", 0.0))

            d3_cfg = DispersionD3Config(s6=s6, s8=s8)
            self.d3_layer = DispersionD3Layer(config=d3_cfg)

    def compute_baseline(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: Sequence[int],
    ) -> tuple[float, torch.Tensor]:
        """Evaluate baseline potential energy (eV) and forces (eV/Angstrom) with optional D3(BJ) dispersion. [D]"""
        if self.baseline_engine is not None:
            try:
                e_base, f_base = self.baseline_engine.calculate(coordinates, atomic_numbers)
            except Exception as exc:
                raise BaselineExecutionError(
                    f"TORQ_BASELINE_EXEC_FAIL: Baseline calculation failed: {exc}",
                    method=str(type(self.baseline_engine).__name__),
                )
        else:
            raise BaselineExecutionError("No baseline engine is configured.", method="NONE")

        if not isinstance(f_base, torch.Tensor):
            f_base = torch.tensor(f_base, dtype=coordinates.dtype, device=coordinates.device)

        if self.use_d3_dispersion and self.d3_layer is not None:
            try:
                e_disp, f_disp = self.d3_layer.compute_energy_and_forces(coordinates, atomic_numbers)
                e_base = float(e_base) + float(e_disp.item())
                f_base = f_base + f_disp
            except Exception as exc:
                raise DispersionIntegrationError(f"D3 dispersion calculation failed: {exc}")

        if f_base.shape != coordinates.shape or not np.isfinite(e_base) or not torch.isfinite(f_base).all():
            raise BaselineExecutionError("Baseline returned invalid energy or forces.", method=type(self.baseline_engine).__name__)
        return float(e_base), f_base

    @staticmethod
    def compute_delta(
        qm_energy: float,
        qm_forces: torch.Tensor,
        baseline_energy: float,
        baseline_forces: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        r"""Compute physical delta difference targets for ML model training. [D]

        $$E_{\Delta}(\mathbf{R}) = E_{\text{QM}}(\mathbf{R}) - E_{\text{baseline}}(\mathbf{R})$$
        $$\mathbf{F}_{\Delta}(\mathbf{R}) = \mathbf{F}_{\text{QM}}(\mathbf{R}) - \mathbf{F}_{\text{baseline}}(\mathbf{R})$$
        """
        delta_e = float(qm_energy) - float(baseline_energy)
        delta_f = qm_forces - baseline_forces
        return delta_e, delta_f

    @staticmethod
    def reconstruct_target(
        baseline_energy: float,
        baseline_forces: torch.Tensor,
        predicted_delta_energy: float,
        predicted_delta_forces: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        r"""Reconstruct target high-level potential energy surface from predicted delta. [D]

        $$\hat{E}_{\text{target}}(\mathbf{R}) = E_{\text{baseline}}(\mathbf{R}) + \hat{E}_{\Delta}(\mathbf{R})$$
        $$\hat{\mathbf{F}}_{\text{target}}(\mathbf{R}) = \mathbf{F}_{\text{baseline}}(\mathbf{R}) + \hat{\mathbf{F}}_{\Delta}(\mathbf{R})$$
        """
        target_e = float(baseline_energy) + float(predicted_delta_energy)
        target_f = baseline_forces + predicted_delta_forces
        return target_e, target_f

    def forward(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: Sequence[int],
        delta_predictor: Callable[[torch.Tensor, Sequence[int]], tuple[float, torch.Tensor]] | None = None,
    ) -> tuple[float, torch.Tensor]:
        """Execute forward inference pipeline: baseline (with D3) + predicted delta. [M]"""
        if delta_predictor is None:
            raise BaselineExecutionError("A trained delta predictor is required; baseline-only results are available through compute_baseline().", method="Delta-ML")
        e_base, f_base = self.compute_baseline(coordinates, atomic_numbers)
        e_delta, f_delta = delta_predictor(coordinates, atomic_numbers)
        if f_delta.shape != coordinates.shape or not np.isfinite(e_delta) or not torch.isfinite(f_delta).all():
            raise ValueError("Delta predictor returned invalid energy or forces.")
        return self.reconstruct_target(e_base, f_base, e_delta, f_delta)


# DeltaMLCorrector alias for DeltaMLEngine (Suggestion #79)
DeltaMLCorrector = DeltaMLEngine

