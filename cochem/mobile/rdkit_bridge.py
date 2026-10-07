"""SMILES-to-3D Algorithmic Bridge for CoChem Mobile (SRS Chunk 05).

Implements REQ-MOB-040 through REQ-MOB-043:
- REQ-MOB-040: Air-Gap Sanitization & Async Concurrency Isolation
- REQ-MOB-041: Complete Valence Saturation & Explicit Hydrogen Addition
- REQ-MOB-042: Deterministic Conformer Generation & 3-Tier Fallback Ladder
- REQ-MOB-043: MMFF94s/UFF relaxation and XYZ serialization
- Genuine RDKit forcefield evaluations and tabulated Mendeleev isotope masses.

These forcefield calculations do not establish an electronic ground state or
constitute an ab initio geometry optimization.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import Chem
from rdkit.Chem import AllChem

from Libraries.cochem_isotopes import isotope_mass, isotope_record

logger = logging.getLogger(__name__)

MAX_SMILES_LENGTH: int = 2048
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_MAX_ITERS: int = 500


class CoChemDomainError(Exception):
    """Base domain exception for CoChem mobile operations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"details={self.details!r})"
        )


class InvalidSmilesError(CoChemDomainError):
    """Raised for invalid lengths, sanitization, or RDKit chemical parsing.

    Maps to HTTP 422 in the Mobile Presentation Tier (REQ-MOB-040).
    """

    def __init__(
        self,
        message: str,
        smiles: str = "",
        error_details: str | None = None,
        status_code: int = 422,
    ) -> None:
        self.smiles = smiles
        self.error_details = error_details or ""
        self.status_code = status_code
        self.http_status_code = status_code
        details = {
            "smiles": smiles,
            "error_details": self.error_details,
            "status_code": status_code,
        }
        super().__init__(message, details=details)


class ConformerEmbeddingError(CoChemDomainError):
    """Raised when all real distance-geometry attempts fail (REQ-MOB-042)."""

    def __init__(
        self,
        message: str,
        smiles: str = "",
        attempted_tiers: list[str] | None = None,
    ) -> None:
        self.smiles = smiles
        self.attempted_tiers = attempted_tiers or []
        details = {
            "smiles": smiles,
            "attempted_tiers": self.attempted_tiers,
        }
        super().__init__(message, details=details)


class AtomCoordinate3DRecord(BaseModel):
    """Single coordinate with an explicit or most-abundant-isotope mass."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    atom_index: int = Field(
        ge=0, description="0-indexed identifier of atom in conformer"
    )
    symbol: str = Field(
        min_length=1, max_length=3, description="IUPAC chemical element symbol"
    )
    mass_number: int | None = Field(default=None, ge=1)
    x: float = Field(description="Cartesian X coordinate in Angstroms")
    y: float = Field(description="Cartesian Y coordinate in Angstroms")
    z: float = Field(description="Cartesian Z coordinate in Angstroms")

    @property
    def atomic_weight(self) -> float:
        """Tabulated isotope mass in u, with the selection policy made explicit."""
        label = f"{self.mass_number}{self.symbol}" if self.mass_number else self.symbol
        return isotope_mass(label)


class RDKit3DResult(BaseModel):
    """Quantum-ready 3D conformer result representation (REQ-MOB-043)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    canonical_smiles: str = Field(description="Canonical SMILES representation")
    input_smiles: str = Field(description="Original sanitized input SMILES")
    num_atoms: int = Field(
        ge=1, description="Total atom count including explicit hydrogens"
    )
    formal_charge: int = Field(description="Total molecular formal charge Q")
    spin_multiplicity: int = Field(
        ge=1,
        description="Requested multiplicity; ground-state spin remains unverified",
    )
    spin_multiplicity_source: Literal["user_supplied", "closed_shell_valence_default"]
    electronic_state_verified: Literal[False] = False
    energy_kcal_mol: float | None = Field(
        description="Computed forcefield energy in kcal/mol, or null when unavailable"
    )
    force_field_method: Literal["MMFF94s", "UFF", "unavailable"]
    converged: bool = Field(description="True if geometry optimization converged")
    atomic_symbols: list[str] = Field(
        description="Element labels, retaining explicit isotope mass numbers"
    )
    atomic_masses: list[float] = Field(
        description="Mendeleev isotope masses in u, with source/selection records"
    )
    isotope_records: list[dict[str, Any]]
    total_mass_amu: float = Field(gt=0.0, description="Summed molecular mass in amu/Da")
    coordinates_3d: list[list[float]] = Field(
        description="Cartesian Nx3 coordinate matrix in Angstroms"
    )
    xyz_block: str = Field(description="Standard 3-part quantum-ready XYZ string")
    xyz_file_path: str | None = Field(
        default=None, description="Path to serialized .xyz file if written"
    )

    @model_validator(mode="after")
    def validate_observed_force_field_result(self) -> RDKit3DResult:
        if self.energy_kcal_mol is None:
            if self.force_field_method != "unavailable" or self.converged:
                raise ValueError(
                    "Unavailable energy cannot claim a forcefield result or convergence"
                )
        elif self.force_field_method == "unavailable":
            raise ValueError(
                "An energy requires the forcefield that actually evaluated it"
            )
        if any(
            len(values) != self.num_atoms
            for values in (
                self.atomic_symbols,
                self.atomic_masses,
                self.isotope_records,
                self.coordinates_3d,
            )
        ):
            raise ValueError(
                "Every atom requires a coordinate, label and isotope mass record"
            )
        if any(len(row) != 3 for row in self.coordinates_3d):
            raise ValueError("Coordinates must contain three components per atom")
        return self

    def save_xyz(self, file_path: str | Path) -> Path:
        """Serialize quantum-ready XYZ block to disk with strict LF line endings."""
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.xyz_block, encoding="utf-8")
        return p


def validate_and_sanitize_smiles(raw_smiles: str) -> tuple[str, Chem.Mol]:
    """Validate input length, trim whitespace, and parse the molecule with RDKit.

    Parameters
    ----------
    raw_smiles : str
        Raw input SMILES string from thin client.

    Returns
    -------
    Tuple[str, Chem.Mol]
        Cleaned SMILES string and parsed RDKit ROMol object.

    Raises
    ------
    InvalidSmilesError
        If length exceeds boundary (<= 2048), is empty, or fails chemical sanitization.
    """
    if not isinstance(raw_smiles, str):
        raise InvalidSmilesError(
            f"SMILES must be a string, got {type(raw_smiles).__name__}.",
            smiles=str(raw_smiles),
            error_details="Type error: non-string payload",
        )

    clean_smiles = raw_smiles.strip()

    if not clean_smiles:
        raise InvalidSmilesError(
            "SMILES string cannot be empty or whitespace-only.",
            smiles=clean_smiles,
            error_details="Empty input payload",
        )

    if len(clean_smiles) > MAX_SMILES_LENGTH:
        raise InvalidSmilesError(
            f"SMILES length ({len(clean_smiles)}) exceeds maximum boundary "
            f"of {MAX_SMILES_LENGTH} characters.",
            smiles=clean_smiles,
            error_details="Length boundary exceeded (> 2048 chars)",
        )

    try:
        mol = Chem.MolFromSmiles(clean_smiles)
    except Exception as exc:
        raise InvalidSmilesError(
            f"RDKit MolFromSmiles exception on '{clean_smiles}': {exc!s}",
            smiles=clean_smiles,
            error_details=str(exc),
        ) from exc

    if mol is None:
        raise InvalidSmilesError(
            f"Invalid SMILES: RDKit was unable to parse '{clean_smiles}'.",
            smiles=clean_smiles,
            error_details="MolFromSmiles returned None",
        )

    return clean_smiles, mol


def generate_deterministic_3d_coordinates(
    mol: Chem.Mol,
    clean_smiles: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Chem.Mol:
    """Execute ETKDGv3 conformer generation with deterministic 3-tier fallback ladder.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule with explicit hydrogens added.
    clean_smiles : str
        Sanitized SMILES for error traceability.
    random_seed : int, optional
        Deterministic random seed (default 42).

    Returns
    -------
    Chem.Mol
        RDKit molecule with valid 3D conformer embedded.

    Raises
    ------
    ConformerEmbeddingError
        If all embedding tiers fail to embed 3D coordinates.
    """
    num_atoms = mol.GetNumAtoms()

    # Single Atom Guard (N <= 1)
    if num_atoms <= 1:
        if mol.GetNumConformers() == 0:
            conf = Chem.Conformer(num_atoms)
            conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
            mol.AddConformer(conf, assignId=True)
        else:
            conf = mol.GetConformer(0)
            conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        return mol

    attempted_tiers: list[str] = []

    # Default ETKDGv3 Configuration
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.enforceChirality = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True

    attempted_tiers.append(
        f"Default ETKDGv3 (seed={random_seed}, enforceChirality=True, "
        "useExpTorsionAnglePrefs=True)"
    )
    embed_code = AllChem.EmbedMolecule(mol, params)

    # Tier 1 Fallback: Random coordinates initialization
    if embed_code == -1:
        attempted_tiers.append(
            "Tier 1: Random coordinate initialization (useRandomCoords=True)"
        )
        params.useRandomCoords = True
        embed_code = AllChem.EmbedMolecule(mol, params)

    # Tier 2 Fallback: Increased maximum iteration budget (1000)
    if embed_code == -1:
        attempted_tiers.append(
            "Tier 2: Increased iteration budget (maxIterations=1000)"
        )
        params.maxIterations = 1000
        embed_code = AllChem.EmbedMolecule(mol, params)

    # Tier 3: Standard distance geometry without experimental torsion preferences.
    if embed_code == -1:
        attempted_tiers.append(
            "Tier 3: Standard distance geometry "
            "(useExpTorsionAnglePrefs=False, useBasicKnowledge=False)"
        )
        params.useExpTorsionAnglePrefs = False
        params.useBasicKnowledge = False
        embed_code = AllChem.EmbedMolecule(mol, params)

    if embed_code == -1:
        raise ConformerEmbeddingError(
            "Distance geometry embedding failed across all 3 fallback tiers "
            f"for SMILES '{clean_smiles}'.",
            smiles=clean_smiles,
            attempted_tiers=attempted_tiers,
        )

    return mol


def relax_geometry_and_calculate_energy(
    mol: Chem.Mol,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> tuple[float | None, str, bool]:
    """Execute two-tier forcefield relaxation hierarchy (MMFF94s -> UFF).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule with embedded 3D conformer.
    max_iters : int, optional
        Maximum minimization iterations (default 500).

    Returns
    -------
    Tuple[Optional[float], str, bool]
        Evaluated forcefield energy (kcal/mol), its method, and actual minimizer
        status. An unsupported calculation returns (None, "unavailable", False).
        An available but unconverged MMFF calculation retains its own energy and
        status; switching potentials cannot establish convergence of MMFF.
    """
    if not isinstance(max_iters, int) or isinstance(max_iters, bool) or max_iters < 0:
        raise ValueError("max_iters must be a non-negative integer")
    if mol.GetNumAtoms() <= 1:
        # Placing an atom at an origin does not evaluate a physical energy.
        return None, "unavailable", False
    if mol.GetNumConformers() == 0:
        raise ValueError("A forcefield calculation requires an embedded conformer")

    # Tier 1: MMFF94s forcefield
    if AllChem.MMFFHasAllMoleculeParams(mol):
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                if ff is not None:
                    opt_status = ff.Minimize(maxIts=max_iters)
                    energy = float(ff.CalcEnergy())
                    if math.isfinite(energy):
                        return energy, "MMFF94s", opt_status == 0
                    logger.warning(
                        "MMFF94s returned a nonfinite energy; result unavailable"
                    )
        except Exception as exc:
            logger.debug("MMFF94s minimization exception: %s; falling back to UFF", exc)

    # Tier 2: UFF fallback forcefield
    if AllChem.UFFHasAllMoleculeParams(mol):
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is not None:
                opt_status = ff.Minimize(maxIts=max_iters)
                energy = float(ff.CalcEnergy())
                if math.isfinite(energy):
                    return energy, "UFF", opt_status == 0
                logger.warning("UFF returned a nonfinite energy; result unavailable")
        except Exception as exc:
            logger.debug("UFF minimization exception: %s", exc)

    return None, "unavailable", False


def compute_quantum_electronic_states(
    mol: Chem.Mol,
    multiplicity: int | None = None,
) -> tuple[int, int]:
    """Read formal charge and validate a requested electronic-state input.

    SMILES does not determine a molecular ground-state spin. Radical structures
    require an explicit multiplicity; a singlet default for a radical-free
    valence representation is only an input convention. Explicit multiplicities
    are checked against electron count, not asserted to be ground states.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    Tuple[int, int]
        Formal charge (int) and spin multiplicity (int >= 1).
    """
    formal_charge = int(Chem.GetFormalCharge(mol))
    radical_electrons = sum(int(a.GetNumRadicalElectrons()) for a in mol.GetAtoms())
    if multiplicity is None:
        if radical_electrons:
            raise ValueError(
                "Specify multiplicity explicitly for radical SMILES; "
                "radical count does not establish ground-state spin"
            )
        multiplicity = 1
    if (
        not isinstance(multiplicity, int)
        or isinstance(multiplicity, bool)
        or multiplicity < 1
    ):
        raise ValueError("multiplicity must be a positive integer")
    electrons = (
        sum(
            atom.GetAtomicNum() + atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
            for atom in mol.GetAtoms()
        )
        - formal_charge
    )
    if electrons < 0 or multiplicity - 1 > electrons:
        raise ValueError(
            "Multiplicity is incompatible with the molecular electron count"
        )
    if (electrons - (multiplicity - 1)) % 2:
        raise ValueError(
            "Multiplicity is incompatible with molecular electron-count parity"
        )
    return formal_charge, multiplicity


def serialize_standard_xyz(
    mol: Chem.Mol,
    canonical_smiles: str,
    energy_kcal_mol: float | None,
    method: str,
    converged: bool,
    formal_charge: int,
    spin_multiplicity: int,
    spin_multiplicity_source: str = "unspecified",
) -> tuple[str, list[str], list[float], float, list[list[float]]]:
    """Format isotope-preserving XYZ with 8-decimal precision and LF endings.

    Returns
    -------
    Tuple containing:
        - xyz_block (str)
        - atomic_symbols (List[str])
        - atomic_masses (List[float])
        - total_mass_amu (float)
        - coordinates_3d (List[List[float]])
    """
    num_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer(0)
    actual_charge, _ = compute_quantum_electronic_states(
        mol, multiplicity=spin_multiplicity
    )
    if formal_charge != actual_charge:
        raise ValueError(
            "XYZ formal charge must match the actual molecular representation"
        )
    if energy_kcal_mol is None:
        if method != "unavailable" or converged:
            raise ValueError(
                "Unavailable energy cannot claim a forcefield or convergence"
            )
    elif not math.isfinite(energy_kcal_mol) or method not in ("MMFF94s", "UFF"):
        raise ValueError("XYZ energy requires a finite evaluated forcefield result")

    energy_text = (
        f"{energy_kcal_mol:.6f}" if energy_kcal_mol is not None else "unavailable"
    )
    header_comment = (
        f"canonical_smiles={canonical_smiles} | "
        f"energy_kcal_mol={energy_text} | "
        f"method={method} | "
        f"converged={converged} | "
        f"charge={formal_charge} | "
        f"multiplicity={spin_multiplicity} | "
        f"multiplicity_source={spin_multiplicity_source} | "
        "electronic_state_verified=False | "
        "mass_policy=explicit_or_most_abundant_isotope"
    )

    symbols: list[str] = []
    masses: list[float] = []
    coords: list[list[float]] = []
    atom_lines: list[str] = []

    for idx, atom in enumerate(mol.GetAtoms()):
        sym = (
            f"{atom.GetIsotope()}{atom.GetSymbol()}"
            if atom.GetIsotope()
            else atom.GetSymbol()
        )
        symbols.append(sym)
        mass = isotope_mass(sym)
        masses.append(mass)

        pos = conf.GetAtomPosition(idx)
        x = float(pos.x)
        y = float(pos.y)
        z = float(pos.z)
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError("XYZ coordinates must be finite")
        coords.append([x, y, z])
        atom_lines.append(f"{sym:<2}  {x:14.8f}  {y:14.8f}  {z:14.8f}")

    total_mass_amu = float(sum(masses))

    # Standard 3-part XYZ schema with explicit LF endings
    xyz_block = f"{num_atoms}\n{header_comment}\n" + "\n".join(atom_lines) + "\n"

    return xyz_block, symbols, masses, total_mass_amu, coords


def smiles_to_3d(
    smiles: str,
    output_path: str | Path | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_iters: int = DEFAULT_MAX_ITERS,
    multiplicity: int | None = None,
) -> RDKit3DResult:
    """Synchronously convert 2D SMILES into quantum-ready 3D structure.

    Parameters
    ----------
    smiles : str
        2D SMILES string.
    output_path : Optional[Union[str, Path]]
        Optional target path to save the .xyz file.
    random_seed : int, optional
        Random seed for ETKDGv3 (default 42).
    max_iters : int, optional
        Maximum forcefield iterations (default 500).
    multiplicity : int, optional
        Explicit electronic-state input; mandatory for radical SMILES. The
        radical-free singlet default is not a ground-state determination.

    Returns
    -------
    RDKit3DResult
        Validated 3D conformer result.
    """
    # 1. Air-gap validation & SMILES parsing (REQ-MOB-040)
    clean_smiles, mol = validate_and_sanitize_smiles(smiles)

    # 2. Valence saturation & Explicit Hydrogen addition (REQ-MOB-041)
    mol_with_hs = Chem.AddHs(mol)

    # Validate state input before performing embedding/minimization work.
    formal_charge, spin_multiplicity = compute_quantum_electronic_states(
        mol_with_hs,
        multiplicity=multiplicity,
    )

    # 3. Deterministic 3D Conformer Generation with Fallback Ladder (REQ-MOB-042)
    mol_3d = generate_deterministic_3d_coordinates(
        mol_with_hs, clean_smiles, random_seed=random_seed
    )

    # 4. Forcefield Relaxation Hierarchy (REQ-MOB-043)
    energy, method, converged = relax_geometry_and_calculate_energy(
        mol_3d, max_iters=max_iters
    )

    # 6. Canonical SMILES & XYZ Serialization
    canonical_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_3d))

    (
        xyz_block,
        symbols,
        masses,
        total_mass_amu,
        coords,
    ) = serialize_standard_xyz(
        mol=mol_3d,
        canonical_smiles=canonical_smiles,
        energy_kcal_mol=energy,
        method=method,
        converged=converged,
        formal_charge=formal_charge,
        spin_multiplicity=spin_multiplicity,
        spin_multiplicity_source=(
            "user_supplied"
            if multiplicity is not None
            else "closed_shell_valence_default"
        ),
    )

    xyz_file_path_str: str | None = None
    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(xyz_block, encoding="utf-8")
        xyz_file_path_str = str(p)

    return RDKit3DResult(
        canonical_smiles=canonical_smiles,
        input_smiles=clean_smiles,
        num_atoms=mol_3d.GetNumAtoms(),
        formal_charge=formal_charge,
        spin_multiplicity=spin_multiplicity,
        spin_multiplicity_source=(
            "user_supplied"
            if multiplicity is not None
            else "closed_shell_valence_default"
        ),
        energy_kcal_mol=energy,
        force_field_method=method,
        converged=converged,
        atomic_symbols=symbols,
        atomic_masses=masses,
        isotope_records=[isotope_record(symbol) for symbol in symbols],
        total_mass_amu=total_mass_amu,
        coordinates_3d=coords,
        xyz_block=xyz_block,
        xyz_file_path=xyz_file_path_str,
    )


async def smiles_to_3d_async(
    smiles: str,
    output_path: str | Path | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_iters: int = DEFAULT_MAX_ITERS,
    executor: concurrent.futures.Executor | None = None,
    multiplicity: int | None = None,
) -> RDKit3DResult:
    """Asynchronously convert 2D SMILES to 3D on a non-blocking worker thread.

    Guarantees isolation from the main asyncio event loop (REQ-MOB-040).
    """
    loop = asyncio.get_running_loop()
    if executor is not None:
        return await loop.run_in_executor(
            executor,
            smiles_to_3d,
            smiles,
            output_path,
            random_seed,
            max_iters,
            multiplicity,
        )
    return await asyncio.to_thread(
        smiles_to_3d,
        smiles,
        output_path,
        random_seed,
        max_iters,
        multiplicity,
    )


class Smiles3DConformerEngine:
    """Manage concurrent SMILES conversion with a real thread pool."""

    def __init__(
        self, max_workers: int = 4, thread_name_prefix: str = "cochem-smiles-3d"
    ) -> None:
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._closed = False

    def convert(
        self,
        smiles: str,
        output_path: str | Path | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_iters: int = DEFAULT_MAX_ITERS,
        multiplicity: int | None = None,
    ) -> RDKit3DResult:
        """Synchronously execute conversion on the managed thread pool."""
        if self._closed:
            raise RuntimeError("Smiles3DConformerEngine executor is closed.")
        future = self._executor.submit(
            smiles_to_3d,
            smiles,
            output_path,
            random_seed,
            max_iters,
            multiplicity,
        )
        return future.result()

    async def convert_async(
        self,
        smiles: str,
        output_path: str | Path | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_iters: int = DEFAULT_MAX_ITERS,
        multiplicity: int | None = None,
    ) -> RDKit3DResult:
        """Asynchronously execute conversion on the managed thread pool."""
        if self._closed:
            raise RuntimeError("Smiles3DConformerEngine executor is closed.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            smiles_to_3d,
            smiles,
            output_path,
            random_seed,
            max_iters,
            multiplicity,
        )

    async def convert_batch_async(
        self,
        smiles_list: Sequence[str],
        output_dir: str | Path | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        multiplicities: Sequence[int | None] | None = None,
    ) -> list[RDKit3DResult]:
        """Process a batch concurrently, retaining each requested multiplicity."""
        if self._closed:
            raise RuntimeError("Smiles3DConformerEngine executor is closed.")
        if multiplicities is not None and len(multiplicities) != len(smiles_list):
            raise ValueError("Provide one multiplicity entry per SMILES in a batch")

        tasks = []
        for idx, smi in enumerate(smiles_list):
            out_p = None
            if output_dir is not None:
                out_p = Path(output_dir) / f"molecule_{idx:04d}.xyz"
            tasks.append(
                self.convert_async(
                    smi,
                    output_path=out_p,
                    random_seed=random_seed,
                    multiplicity=multiplicities[idx]
                    if multiplicities is not None
                    else None,
                )
            )
        return await asyncio.gather(*tasks)

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the thread pool."""
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> Smiles3DConformerEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown(wait=True)

    async def __aenter__(self) -> Smiles3DConformerEngine:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown(wait=True)
