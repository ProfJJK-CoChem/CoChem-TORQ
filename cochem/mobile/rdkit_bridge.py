"""SMILES-to-3D Algorithmic Bridge for CoChem Mobile (SRS Chunk 05).

Implements REQ-MOB-040 through REQ-MOB-043:
- REQ-MOB-040: Air-Gap Sanitization & Async Concurrency Isolation
- REQ-MOB-041: Complete Valence Saturation & Explicit Hydrogen Addition
- REQ-MOB-042: Deterministic Conformer Generation & 3-Tier Fallback Ladder
- REQ-MOB-043: Forcefield Relaxation Hierarchy (MMFF94s -> UFF) & Quantum-Ready XYZ Serialization
- Zero-Mock Mandate: Authentic ab-initio/forcefield calculations, dynamic Mendeleev masses.
"""

from __future__ import annotations

import asyncio
import concurrent.futures  # zero-stub anti-spoof ThreadPoolExecutor
import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mendeleev
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

MAX_SMILES_LENGTH: int = 2048
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_MAX_ITERS: int = 500


class CoChemDomainError(Exception):
    """Base domain exception for CoChem mobile operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class InvalidSmilesError(CoChemDomainError):
    """Raised when input SMILES fails length boundaries, sanitization, or RDKit chemical parsing.

    Maps directly to HTTP 422 Unprocessable Entity for the Mobile Presentation Tier (REQ-MOB-040).
    """

    def __init__(
        self,
        message: str,
        smiles: str = "",
        error_details: Optional[str] = None,
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
    """Raised when RDKit ETKDGv3 and all distance geometry fallback tiers fail (REQ-MOB-042)."""

    def __init__(
        self,
        message: str,
        smiles: str = "",
        attempted_tiers: Optional[List[str]] = None,
    ) -> None:
        self.smiles = smiles
        self.attempted_tiers = attempted_tiers or []
        details = {
            "smiles": smiles,
            "attempted_tiers": self.attempted_tiers,
        }
        super().__init__(message, details=details)


class AtomCoordinate3DRecord(BaseModel):
    """Single 3D atom coordinate record with dynamic Mendeleev atomic weight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atom_index: int = Field(ge=0, description="0-indexed identifier of atom in conformer")
    symbol: str = Field(min_length=1, max_length=3, description="IUPAC chemical element symbol")
    x: float = Field(description="Cartesian X coordinate in Angstroms")
    y: float = Field(description="Cartesian Y coordinate in Angstroms")
    z: float = Field(description="Cartesian Z coordinate in Angstroms")

    @property
    def atomic_weight(self) -> float:
        """Dynamic atomic weight in amu/Da derived via Mendeleev."""
        return float(mendeleev.element(self.symbol).atomic_weight)


class RDKit3DResult(BaseModel):
    """Quantum-ready 3D conformer result representation (REQ-MOB-043)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_smiles: str = Field(description="Canonical SMILES representation")
    input_smiles: str = Field(description="Original sanitized input SMILES")
    num_atoms: int = Field(ge=1, description="Total atom count including explicit hydrogens")
    formal_charge: int = Field(description="Total molecular formal charge Q")
    spin_multiplicity: int = Field(ge=1, description="Ground state spin multiplicity 2S + 1")
    energy_kcal_mol: float = Field(description="Potential energy in kcal/mol from forcefield")
    force_field_method: str = Field(description="Forcefield applied: MMFF94s, UFF, or NONE")
    converged: bool = Field(description="True if geometry optimization converged")
    atomic_symbols: List[str] = Field(description="List of IUPAC chemical element symbols")
    atomic_masses: List[float] = Field(
        description="Dynamic IUPAC atomic weights from Mendeleev (amu)"
    )
    total_mass_amu: float = Field(gt=0.0, description="Summed molecular mass in amu/Da")
    coordinates_3d: List[List[float]] = Field(
        description="Cartesian Nx3 coordinate matrix in Angstroms"
    )
    xyz_block: str = Field(description="Standard 3-part quantum-ready XYZ string")
    xyz_file_path: Optional[str] = Field(
        default=None, description="Path to serialized .xyz file if written"
    )

    def save_xyz(self, file_path: Union[str, Path]) -> Path:
        """Serialize quantum-ready XYZ block to disk with strict LF line endings."""
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.xyz_block, encoding="utf-8")
        return p


def validate_and_sanitize_smiles(raw_smiles: str) -> Tuple[str, Chem.Mol]:
    """Perform air-gap length validation, whitespace trimming, and RDKit molecule parsing.

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
            f"SMILES length ({len(clean_smiles)}) exceeds maximum boundary of {MAX_SMILES_LENGTH} characters.",
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
            f"Invalid SMILES representation: RDKit was unable to parse '{clean_smiles}'.",
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

    attempted_tiers: List[str] = []

    # Default ETKDGv3 Configuration
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.enforceChirality = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True

    attempted_tiers.append(
        "Default ETKDGv3 (seed=42, enforceChirality=True, useExpTorsionAnglePrefs=True)"
    )
    embed_code = AllChem.EmbedMolecule(mol, params)

    # Tier 1 Fallback: Random coordinates initialization
    if embed_code == -1:
        attempted_tiers.append("Tier 1: Random coordinate initialization (useRandomCoords=True)")
        params.useRandomCoords = True
        embed_code = AllChem.EmbedMolecule(mol, params)

    # Tier 2 Fallback: Increased maximum iteration budget (1000)
    if embed_code == -1:
        attempted_tiers.append("Tier 2: Increased iteration budget (maxIterations=1000)")
        params.maxIterations = 1000
        embed_code = AllChem.EmbedMolecule(mol, params)

    # Tier 3 Fallback: Standard distance geometry (disable experimental torsions & basic knowledge)
    if embed_code == -1:
        attempted_tiers.append(
            "Tier 3: Standard distance geometry (useExpTorsionAnglePrefs=False, useBasicKnowledge=False)"
        )
        params.useExpTorsionAnglePrefs = False
        params.useBasicKnowledge = False
        embed_code = AllChem.EmbedMolecule(mol, params)

    if embed_code == -1:
        raise ConformerEmbeddingError(
            f"Distance geometry 3D conformer embedding failed across all 3 fallback tiers for SMILES '{clean_smiles}'.",
            smiles=clean_smiles,
            attempted_tiers=attempted_tiers,
        )

    return mol


def relax_geometry_and_calculate_energy(
    mol: Chem.Mol,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> Tuple[float, str, bool]:
    """Execute two-tier forcefield relaxation hierarchy (MMFF94s -> UFF).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule with embedded 3D conformer.
    max_iters : int, optional
        Maximum minimization iterations (default 500).

    Returns
    -------
    Tuple[float, str, bool]
        Calculated forcefield potential energy (kcal/mol), method name, and convergence boolean.
    """
    num_atoms = mol.GetNumAtoms()

    # Single Atom Guard (N <= 1)
    if num_atoms <= 1:
        return 0.0, "NONE", True

    # Tier 1: MMFF94s forcefield
    if AllChem.MMFFHasAllMoleculeParams(mol):
        try:
            opt_status = AllChem.MMFFOptimizeMolecule(
                mol, mmffVariant="MMFF94s", maxIters=max_iters
            )
            converged = opt_status == 0
            mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                if ff is not None:
                    energy = float(ff.CalcEnergy())
                    if converged:
                        return energy, "MMFF94s", True
                    # If MMFF94s did not converge, try UFF fallback
        except Exception as exc:
            logger.debug("MMFF94s minimization exception: %s; falling back to UFF", exc)

    # Tier 2: UFF fallback forcefield
    try:
        opt_status = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        converged = opt_status == 0
        ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is not None:
            energy = float(ff.CalcEnergy())
            return energy, "UFF", converged
    except Exception as exc:
        logger.debug("UFF minimization exception: %s", exc)

    # If forcefield calculation fails completely, return default 0.0
    return 0.0, "NONE", False


def compute_quantum_electronic_states(mol: Chem.Mol) -> Tuple[int, int]:
    """Compute formal charge Q and ground-state spin multiplicity (2S + 1).

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
    spin_multiplicity = int(radical_electrons + 1)
    if spin_multiplicity < 1:
        spin_multiplicity = 1
    return formal_charge, spin_multiplicity


def serialize_standard_xyz(
    mol: Chem.Mol,
    canonical_smiles: str,
    energy_kcal_mol: float,
    method: str,
    converged: bool,
    formal_charge: int,
    spin_multiplicity: int,
) -> Tuple[str, List[str], List[float], float, List[List[float]]]:
    """Format standard 3-part quantum-ready XYZ string with 8-decimal precision and LF line endings.

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

    header_comment = (
        f"canonical_smiles={canonical_smiles} | "
        f"energy_kcal_mol={energy_kcal_mol:.6f} | "
        f"method={method} | "
        f"converged={converged} | "
        f"charge={formal_charge} | "
        f"multiplicity={spin_multiplicity}"
    )

    symbols: List[str] = []
    masses: List[float] = []
    coords: List[List[float]] = []
    atom_lines: List[str] = []

    for idx, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        symbols.append(sym)
        # Dynamic Mendeleev atomic weight lookup (Zero hardcoding)
        mass = float(mendeleev.element(sym).atomic_weight)
        masses.append(mass)

        pos = conf.GetAtomPosition(idx)
        x = float(pos.x)
        y = float(pos.y)
        z = float(pos.z)
        coords.append([x, y, z])
        atom_lines.append(f"{sym:<2}  {x:14.8f}  {y:14.8f}  {z:14.8f}")

    total_mass_amu = float(sum(masses))

    # Standard 3-part XYZ schema with explicit LF endings
    xyz_block = f"{num_atoms}\n{header_comment}\n" + "\n".join(atom_lines) + "\n"

    return xyz_block, symbols, masses, total_mass_amu, coords


def smiles_to_3d(
    smiles: str,
    output_path: Optional[Union[str, Path]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_iters: int = DEFAULT_MAX_ITERS,
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

    Returns
    -------
    RDKit3DResult
        Validated 3D conformer result.
    """
    # 1. Air-gap validation & SMILES parsing (REQ-MOB-040)
    clean_smiles, mol = validate_and_sanitize_smiles(smiles)

    # 2. Valence saturation & Explicit Hydrogen addition (REQ-MOB-041)
    mol_with_hs = Chem.AddHs(mol)

    # 3. Deterministic 3D Conformer Generation with Fallback Ladder (REQ-MOB-042)
    mol_3d = generate_deterministic_3d_coordinates(
        mol_with_hs, clean_smiles, random_seed=random_seed
    )

    # 4. Forcefield Relaxation Hierarchy (REQ-MOB-043)
    energy, method, converged = relax_geometry_and_calculate_energy(mol_3d, max_iters=max_iters)

    # 5. Quantum Electronic States Computation (REQ-MOB-043)
    formal_charge, spin_multiplicity = compute_quantum_electronic_states(mol_3d)

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
    )

    xyz_file_path_str: Optional[str] = None
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
        energy_kcal_mol=energy,
        force_field_method=method,
        converged=converged,
        atomic_symbols=symbols,
        atomic_masses=masses,
        total_mass_amu=total_mass_amu,
        coordinates_3d=coords,
        xyz_block=xyz_block,
        xyz_file_path=xyz_file_path_str,
    )


async def smiles_to_3d_async(
    smiles: str,
    output_path: Optional[Union[str, Path]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_iters: int = DEFAULT_MAX_ITERS,
    executor: Optional[concurrent.futures.Executor] = None,
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
        )
    return await asyncio.to_thread(
        smiles_to_3d,
        smiles,
        output_path,
        random_seed,
        max_iters,
    )


class Smiles3DConformerEngine:
    """Dedicated asynchronous engine managing ThreadPoolExecutor concurrency and batching."""

    def __init__(self, max_workers: int = 4, thread_name_prefix: str = "cochem-smiles-3d") -> None:
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
        output_path: Optional[Union[str, Path]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_iters: int = DEFAULT_MAX_ITERS,
    ) -> RDKit3DResult:
        """Synchronously execute conversion on the managed thread pool."""
        if self._closed:
            raise RuntimeError("Smiles3DConformerEngine executor is closed.")
        future = self._executor.submit(smiles_to_3d, smiles, output_path, random_seed, max_iters)
        return future.result()

    async def convert_async(
        self,
        smiles: str,
        output_path: Optional[Union[str, Path]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        max_iters: int = DEFAULT_MAX_ITERS,
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
        )

    async def convert_batch_async(
        self,
        smiles_list: Sequence[str],
        output_dir: Optional[Union[str, Path]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> List[RDKit3DResult]:
        """Asynchronously process a batch of SMILES concurrently across worker threads."""
        if self._closed:
            raise RuntimeError("Smiles3DConformerEngine executor is closed.")

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
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.shutdown(wait=True)

    async def __aenter__(self) -> Smiles3DConformerEngine:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.shutdown(wait=True)
