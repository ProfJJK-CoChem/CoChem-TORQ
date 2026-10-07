"""Asynchronous 3D Conformer Synthesis Engine for CoChem Mobile.

Generates reproducible 3D conformers using RDKit's ETKDGv3 algorithm and MMFF94/UFF
energy minimization strictly adhering to the Zero-Mock mandate.
"""

from __future__ import annotations

import logging
import math
import re

from rdkit import Chem
from rdkit.Chem import AllChem

from cochem.mobile.schemas import (
    AtomCoordinate3D,
    Conformer3DResultSchema,
    SketcherPayloadSchema,
    ValenceValidationResultSchema,
)

logger = logging.getLogger(__name__)


def extract_error_atom_indices(mol: Chem.Mol | None, exception: Exception) -> list[int]:
    """Inspect RDKit sanitize and valence exceptions to extract 0-indexed offending atom indices."""
    indices: list[int] = []
    err_str = str(exception)

    # 1. Match 'atom # X' pattern from AtomValenceException / MolSanitizeException
    for match in re.finditer(r"atom #\s*(\d+)", err_str):
        idx = int(match.group(1))
        if idx not in indices:
            indices.append(idx)

    # 2. Match 'Unkekulized atoms: 0 1 2 3' pattern from KekulizeException
    kek_match = re.search(r"Unkekulized atoms:\s*([\d\s]+)", err_str)
    if kek_match:
        for val in kek_match.group(1).strip().split():
            try:
                idx = int(val)
                if idx not in indices:
                    indices.append(idx)
            except ValueError as _e:
                logger.debug(f"Ignored exception: {_e}")

    # 3. Direct exception method inspection if available
    for attr in ("getAtomIdx", "GetAtomIdx"):
        if hasattr(exception, attr):
            getter = getattr(exception, attr)
            val = getter() if callable(getter) else getter
            if isinstance(val, int) and val not in indices:
                indices.append(val)

    # 4. Fallback explicit valence inspection on molecule atoms
    if not indices and mol is not None:
        standard_max_valences = {
            "H": 1,
            "C": 4,
            "N": 4,
            "O": 2,
            "F": 1,
            "Cl": 1,
            "Br": 1,
            "I": 1,
            "P": 5,
            "S": 6,
        }
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in standard_max_valences:
                try:
                    exp_val = sum(int(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds())
                    if exp_val > standard_max_valences[sym] and atom.GetIdx() not in indices:
                        indices.append(atom.GetIdx())
                except (RuntimeError, ValueError) as val_err:
                    logger.debug("Failed checking explicit valence on atom: %s", val_err)

    return indices


def generate_3d_conformer(
    payload: SketcherPayloadSchema, *, max_minimization_iterations: int = 500,
) -> Conformer3DResultSchema:
    """Synthesize reproducible 3D conformer with ETKDGv3 and MMFF94/UFF minimization.

    Parameters
    ----------
    payload : SketcherPayloadSchema
        The payload containing 2D molfile (V2000), SMILES, and 2D atom coordinates.

    Returns
    -------
    Conformer3DResultSchema
        Resulting 3D conformer coordinates, potential energy, and valence validation state.
    """
    if type(max_minimization_iterations) is not int or max_minimization_iterations < 0:
        raise ValueError("The minimization iteration limit must be a nonnegative integer.")

    # 1. Parse Molecule from V2000 Molfile or fallback to SMILES
    mol: Chem.Mol | None = None
    if payload.molfile_v2000:
        try:
            parsed = Chem.MolFromMolBlock(payload.molfile_v2000, sanitize=False, removeHs=False)
            if parsed is not None and parsed.GetNumAtoms() > 0:
                mol = parsed
        except (RuntimeError, ValueError) as parse_err:
            logger.debug("Molfile parsing exception: %s", parse_err)
            mol = None

    if mol is None and payload.smiles:
        try:
            parsed = Chem.MolFromSmiles(payload.smiles, sanitize=False)
            if parsed is not None and parsed.GetNumAtoms() > 0:
                mol = parsed
        except (RuntimeError, ValueError) as parse_err:
            logger.debug("SMILES parsing exception: %s", parse_err)
            mol = None

    if mol is None:
        return Conformer3DResultSchema(
            success=False,
            smiles=payload.smiles,
            validation=ValenceValidationResultSchema(
                success=False,
                diagnostic_message="Invalid chemical representation: unable to parse Molfile or SMILES.",
                atom_error_indices=[],
            ),
        )

    # 2. Execute Molecule Sanitization and Catch Valence Violations
    try:
        Chem.SanitizeMol(mol)
    except (
        RuntimeError,
        ValueError,
        Chem.rdchem.KekulizeException,
        Chem.rdchem.AtomValenceException,
        Chem.rdchem.MolSanitizeException,
    ) as exc:
        err_indices = extract_error_atom_indices(mol, exc)
        return Conformer3DResultSchema(
            success=False,
            smiles=payload.smiles,
            validation=ValenceValidationResultSchema(
                success=False,
                diagnostic_message=f"Valence/Sanitization error: {exc!s}",
                atom_error_indices=err_indices,
            ),
        )

    if mol.GetNumAtoms() == 0:
        return Conformer3DResultSchema(
            success=False,
            smiles=payload.smiles,
            validation=ValenceValidationResultSchema(
                success=False,
                diagnostic_message="Molecule contains 0 atoms.",
                atom_error_indices=[],
            ),
        )

    # 3. Add Explicit Hydrogens
    mol_with_hs = Chem.AddHs(mol)

    # 4. Configure Distance Geometry with pinned ETKDGv3 parameters
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.enforceChirality = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True

    embed_result = AllChem.EmbedMolecule(mol_with_hs, params)
    if embed_result < 0:
        # Fallback with random coordinates
        params.useRandomCoords = True
        embed_result = AllChem.EmbedMolecule(mol_with_hs, params)
        if embed_result < 0:
            return Conformer3DResultSchema(
                success=False,
                smiles=Chem.MolToSmiles(Chem.RemoveHs(mol_with_hs)),
                validation=ValenceValidationResultSchema(
                    success=False,
                    diagnostic_message="Distance geometry ETKDGv3 3D embedding failed to converge.",
                    atom_error_indices=[],
                ),
            )

    # MMFF94 and UFF are distinct models. Report the selected model and never
    # reinterpret a failed minimization as success using a different potential.
    energy_kcal_mol: float | None = None
    force_field_used: str | None = None
    minimization_converged = False
    diagnostics: list[str] = []
    ff = None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_with_hs):
            props = AllChem.MMFFGetMoleculeProperties(mol_with_hs, mmffVariant="MMFF94")
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol_with_hs, props)
                force_field_used = "MMFF94" if ff is not None else None
            if ff is None:
                diagnostics.append("MMFF94 force-field construction unavailable.")
        else:
            diagnostics.append("MMFF94 parameters unavailable for this molecule.")
        if ff is None:
            if AllChem.UFFHasAllMoleculeParams(mol_with_hs):
                ff = AllChem.UFFGetMoleculeForceField(mol_with_hs)
                force_field_used = "UFF" if ff is not None else None
                if ff is None:
                    diagnostics.append("UFF force-field construction unavailable.")
            else:
                diagnostics.append("UFF parameters unavailable for this molecule.")
        if ff is not None:
            status = int(ff.Minimize(maxIts=max_minimization_iterations))
            if status != 0:
                diagnostics.append(
                    f"{force_field_used} minimization did not converge "
                    f"(RDKit status {status}; iteration limit {max_minimization_iterations})."
                )
            else:
                calculated = float(ff.CalcEnergy())
                if not math.isfinite(calculated):
                    diagnostics.append(f"{force_field_used} energy is nonfinite.")
                else:
                    energy_kcal_mol = calculated
                    minimization_converged = True
    except (RuntimeError, ValueError) as ff_err:
        diagnostics.append(f"Force-field execution failed: {ff_err}")
        logger.debug("Force-field execution exception: %s", ff_err)

    # Retain actual embedded/partially minimized coordinates on failure.
    conf = mol_with_hs.GetConformer(0)
    coords_3d: list[AtomCoordinate3D] = []
    for atom in mol_with_hs.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        if not all(math.isfinite(value) for value in (pos.x, pos.y, pos.z)):
            return Conformer3DResultSchema(
                success=False, smiles=payload.smiles, force_field_used=force_field_used,
                validation=ValenceValidationResultSchema(
                    success=True, diagnostic_message="3D conformer coordinates are nonfinite.",
                    atom_error_indices=[],
                ),
            )
        coords_3d.append(
            AtomCoordinate3D(
                atom_index=idx,
                symbol=atom.GetSymbol(),
                x=float(pos.x),
                y=float(pos.y),
                z=float(pos.z),
            )
        )

    # 7. Generate MDL Molfile V3000 Block and Canonical SMILES
    molfile_v3000 = Chem.MolToMolBlock(mol_with_hs, forceV3000=True)
    canonical_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_with_hs))

    return Conformer3DResultSchema(
        success=minimization_converged,
        smiles=canonical_smiles,
        molfile_v3000=molfile_v3000,
        energy_kcal_mol=energy_kcal_mol,
        force_field_used=force_field_used,
        coordinates_3d=coords_3d,
        validation=ValenceValidationResultSchema(
            success=True,
            diagnostic_message=(
                f"Actual ETKDGv3 geometry minimized with {force_field_used}; "
                "RDKit minimizer converged. This force-field structure is not a quantum equilibrium result."
                if minimization_converged else
                "Actual embedded/partially minimized geometry retained; optimization unavailable or failed. "
                + " ".join(diagnostics)
            ),
            atom_error_indices=[],
        ),
    )
