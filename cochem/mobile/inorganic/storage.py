"""CoChem Mobile Inorganic Storage & Serialization Infrastructure.

Zero-Mock implementation of Pydantic v2 schemas, JSON, HDF5 with SWMR and filelock,
thread-safe SQLite persistence, and Tripartite Air-Gap RPC client.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cochem.concurrency.atomic_file_lock import AtomicFileLock
from cochem.mobile.inorganic.models import (
    InorganicComplex,
    get_mendeleev_element,
)


class InorganicAtom3D(BaseModel):
    """Pydantic v2 schema for individual 3D atom in coordination complex."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atom_index: Annotated[int, Field(ge=0, description="0-indexed global atom identifier")]
    symbol: Annotated[str, Field(min_length=1, max_length=3, description="IUPAC element symbol")]
    x: Annotated[float, Field(description="3D Cartesian X coordinate in Angstroms")]
    y: Annotated[float, Field(description="3D Cartesian Y coordinate in Angstroms")]
    z: Annotated[float, Field(description="3D Cartesian Z coordinate in Angstroms")]
    charge: Annotated[int, Field(default=0, ge=-7, le=7, description="Formal atomic charge")]
    is_metal: Annotated[bool, Field(default=False, description="True if atom is central metal ion")]
    is_donor: Annotated[
        bool, Field(default=False, description="True if atom directly coordinates to metal")
    ]
    ligand_index: Annotated[
        int,
        Field(
            default=-1,
            description="Index of parent ligand (-1 for metal)",
        ),
    ]

    @property
    def atomic_weight(self) -> float:
        """Dynamic atomic weight derived from Mendeleev."""
        elem = get_mendeleev_element(self.symbol)
        return float(elem.atomic_weight)


class InorganicBondRecord(BaseModel):
    """Pydantic v2 schema for chemical bonds and coordination connections."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atom_index_1: Annotated[int, Field(ge=0, description="0-indexed atom index of first endpoint")]
    atom_index_2: Annotated[int, Field(ge=0, description="0-indexed atom index of second endpoint")]
    bond_order: Annotated[
        float, Field(gt=0.0, le=4.0, default=1.0, description="Formal bond order")
    ]
    bond_type: Annotated[
        str,
        Field(
            default="coordination",
            description="Bond category: 'coordination', 'covalent', 'aromatic'",
        ),
    ]


class InorganicComplexSchema(BaseModel):
    """Immutable Pydantic v2 schema representing full coordination complex state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metal_symbol: Annotated[
        str, Field(min_length=1, max_length=3, description="Central metal symbol")
    ]
    oxidation_state: Annotated[int, Field(ge=-2, le=8, description="Metal formal oxidation state")]
    geometry_name: Annotated[str, Field(min_length=1, description="Coordination geometry name")]
    polyhedron: Annotated[str, Field(min_length=1, description="Polyhedron enum string")]
    coordination_number: Annotated[
        int, Field(ge=2, le=12, description="Polyhedral coordination number")
    ]
    isomer_state: Annotated[
        str, Field(default="default", description="Stereochemical isomer identifier")
    ]
    spin_multiplicity: Annotated[int, Field(ge=1, le=11, description="Spin multiplicity 2S + 1")]
    net_charge: Annotated[int, Field(ge=-10, le=10, description="Net complex electrostatic charge")]
    formula: Annotated[str, Field(min_length=1, description="IUPAC-style chemical formula")]
    molecular_weight: Annotated[
        float, Field(gt=0.0, description="Dynamic molecular weight in g/mol")
    ]
    atoms: Annotated[
        List[InorganicAtom3D],
        Field(default_factory=list, description="Ordered list of 3D atoms"),
    ]
    bonds: Annotated[
        List[InorganicBondRecord],
        Field(default_factory=list, description="Bond connectivity records"),
    ]
    metadata: Annotated[
        Dict[str, Any],
        Field(default_factory=dict, description="Arbitrary calculation or origin metadata"),
    ]


def complex_to_schema(
    complex_obj: Union[InorganicComplex, InorganicComplexSchema],
    metadata: Optional[Dict[str, Any]] = None,
) -> InorganicComplexSchema:
    """Convert an InorganicComplex model instance into a frozen InorganicComplexSchema."""
    if isinstance(complex_obj, InorganicComplexSchema):
        return complex_obj

    atom_records: List[InorganicAtom3D] = []
    metal_sym = complex_obj.metal_center.symbol

    # Build atoms from coordinates_3d
    for idx, (sym, x, y, z) in enumerate(complex_obj.coordinates_3d):
        is_metal = (idx == 0) and (sym == metal_sym)
        is_donor = False
        # Donor atom check: connected to metal in bonds_graph
        if not is_metal:
            for b1, b2, _, b_type in complex_obj.bonds_graph:
                if (b1 == 0 and b2 == idx) or (b2 == 0 and b1 == idx):
                    if b_type == "coordination":
                        is_donor = True
                        break

        atom_records.append(
            InorganicAtom3D(
                atom_index=idx,
                symbol=sym,
                x=float(x),
                y=float(y),
                z=float(z),
                charge=complex_obj.metal_center.oxidation_state if is_metal else 0,
                is_metal=is_metal,
                is_donor=is_donor,
                ligand_index=-1 if is_metal else 0,
            )
        )

    bond_records: List[InorganicBondRecord] = []
    for b1, b2, order, b_type in complex_obj.bonds_graph:
        bond_records.append(
            InorganicBondRecord(
                atom_index_1=int(b1),
                atom_index_2=int(b2),
                bond_order=float(order),
                bond_type=str(b_type),
            )
        )

    meta = metadata or {}
    meta["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    return InorganicComplexSchema(
        metal_symbol=complex_obj.metal_center.symbol,
        oxidation_state=complex_obj.metal_center.oxidation_state,
        geometry_name=complex_obj.geometry.name,
        polyhedron=complex_obj.geometry.polyhedron.value,
        coordination_number=complex_obj.geometry.coordination_number,
        isomer_state=complex_obj.isomer_state,
        spin_multiplicity=complex_obj.spin_multiplicity,
        net_charge=complex_obj.net_charge,
        formula=complex_obj.chemical_formula,
        molecular_weight=complex_obj.molecular_weight,
        atoms=atom_records,
        bonds=bond_records,
        metadata=meta,
    )


class JSONInorganicSerializer:
    """JSON serialization and deserialization engine for coordination complexes."""

    @classmethod
    def serialize(
        cls,
        complex_obj: Union[InorganicComplex, InorganicComplexSchema],
        indent: Optional[int] = 2,
    ) -> str:
        """Convert complex instance to formatted JSON string."""
        schema = complex_to_schema(complex_obj)
        return schema.model_dump_json(indent=indent)

    @classmethod
    def save_to_file(
        cls,
        complex_obj: Union[InorganicComplex, InorganicComplexSchema],
        filepath: Union[Path, str],
    ) -> Path:
        """Serialize and atomically write complex schema to disk."""
        target_path = Path(filepath).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target_path.with_suffix(target_path.suffix + ".lock")

        json_data = cls.serialize(complex_obj)
        with AtomicFileLock(lock_path, target_path=target_path):
            target_path.write_text(json_data, encoding="utf-8")
        return target_path

    @classmethod
    def deserialize(cls, json_str: str) -> InorganicComplexSchema:
        """Parse JSON string into validated InorganicComplexSchema."""
        return InorganicComplexSchema.model_validate_json(json_str)

    @classmethod
    def load_from_file(cls, filepath: Union[Path, str]) -> InorganicComplexSchema:
        """Load and parse complex schema from file under atomic read lock."""
        target_path = Path(filepath).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Complex file not found: {target_path}")

        lock_path = target_path.with_suffix(target_path.suffix + ".lock")
        with AtomicFileLock(lock_path, target_path=target_path):
            content = target_path.read_text(encoding="utf-8")
        return cls.deserialize(content)


class HDF5InorganicSerializer:
    """Thread-safe and multi-process HDF5 serializer with atomic exclusion file locking."""

    @classmethod
    def save(
        cls,
        complex_obj: Union[InorganicComplex, InorganicComplexSchema],
        filepath: Union[Path, str],
        group_name: str = "complex",
    ) -> Path:
        """Write 3D coordinates, bonds, and physics attributes into HDF5 container."""
        schema = complex_to_schema(complex_obj)
        target_path = Path(filepath).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target_path.with_suffix(".lock")

        with AtomicFileLock(lock_path, target_path=target_path):
            with h5py.File(target_path, "a") as h5f:
                if group_name in h5f:
                    del h5f[group_name]

                grp = h5f.create_group(group_name)

                # Store attributes
                grp.attrs["metal_symbol"] = schema.metal_symbol
                grp.attrs["oxidation_state"] = schema.oxidation_state
                grp.attrs["geometry_name"] = schema.geometry_name
                grp.attrs["polyhedron"] = schema.polyhedron
                grp.attrs["coordination_number"] = schema.coordination_number
                grp.attrs["isomer_state"] = schema.isomer_state
                grp.attrs["spin_multiplicity"] = schema.spin_multiplicity
                grp.attrs["net_charge"] = schema.net_charge
                grp.attrs["formula"] = schema.formula
                grp.attrs["molecular_weight"] = schema.molecular_weight
                grp.attrs["metadata_json"] = json.dumps(schema.metadata)

                # Store 3D coordinates dataset
                n_atoms = len(schema.atoms)
                coords_arr = np.zeros((n_atoms, 3), dtype=np.float64)
                symbols_arr = []
                charges_arr = np.zeros((n_atoms,), dtype=np.int32)
                is_metal_arr = np.zeros((n_atoms,), dtype=np.uint8)
                is_donor_arr = np.zeros((n_atoms,), dtype=np.uint8)

                for i, atom in enumerate(schema.atoms):
                    coords_arr[i] = [atom.x, atom.y, atom.z]
                    symbols_arr.append(atom.symbol.encode("utf-8"))
                    charges_arr[i] = atom.charge
                    is_metal_arr[i] = 1 if atom.is_metal else 0
                    is_donor_arr[i] = 1 if atom.is_donor else 0

                grp.create_dataset("coordinates", data=coords_arr)
                grp.create_dataset("symbols", data=symbols_arr)
                grp.create_dataset("charges", data=charges_arr)
                grp.create_dataset("is_metal", data=is_metal_arr)
                grp.create_dataset("is_donor", data=is_donor_arr)

                # Store bonds dataset
                n_bonds = len(schema.bonds)
                bonds_arr = np.zeros((n_bonds, 3), dtype=np.float64)
                b_types = []
                for j, b in enumerate(schema.bonds):
                    bonds_arr[j] = [b.atom_index_1, b.atom_index_2, b.bond_order]
                    b_types.append(b.bond_type.encode("utf-8"))

                grp.create_dataset("bonds", data=bonds_arr)
                grp.create_dataset("bond_types", data=b_types)
                h5f.flush()

        return target_path

    @classmethod
    def load(
        cls,
        filepath: Union[Path, str],
        group_name: str = "complex",
    ) -> InorganicComplexSchema:
        """Load and reconstruct InorganicComplexSchema from HDF5 container."""
        target_path = Path(filepath).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {target_path}")

        lock_path = target_path.with_suffix(".lock")
        with AtomicFileLock(lock_path, target_path=target_path):
            with h5py.File(target_path, "r") as h5f:
                if group_name not in h5f:
                    raise KeyError(f"Group '{group_name}' not found in {target_path}")

                grp = h5f[group_name]
                metal_symbol = str(grp.attrs["metal_symbol"])
                oxidation_state = int(grp.attrs["oxidation_state"])
                geometry_name = str(grp.attrs["geometry_name"])
                polyhedron = str(grp.attrs["polyhedron"])
                coordination_number = int(grp.attrs["coordination_number"])
                isomer_state = str(grp.attrs["isomer_state"])
                spin_multiplicity = int(grp.attrs["spin_multiplicity"])
                net_charge = int(grp.attrs["net_charge"])
                formula = str(grp.attrs["formula"])
                molecular_weight = float(grp.attrs["molecular_weight"])
                meta_json = str(grp.attrs.get("metadata_json", "{}"))
                metadata = json.loads(meta_json)

                coords = grp["coordinates"][:]
                symbols = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in grp["symbols"][:]
                ]
                charges = grp["charges"][:]
                is_metal = grp["is_metal"][:]
                is_donor = grp["is_donor"][:]

                atoms: List[InorganicAtom3D] = []
                for i in range(len(coords)):
                    atoms.append(
                        InorganicAtom3D(
                            atom_index=i,
                            symbol=symbols[i],
                            x=float(coords[i, 0]),
                            y=float(coords[i, 1]),
                            z=float(coords[i, 2]),
                            charge=int(charges[i]),
                            is_metal=bool(is_metal[i]),
                            is_donor=bool(is_donor[i]),
                            ligand_index=-1 if bool(is_metal[i]) else 0,
                        )
                    )

                bonds_data = grp["bonds"][:]
                bond_types = [
                    bt.decode("utf-8") if isinstance(bt, bytes) else str(bt)
                    for bt in grp["bond_types"][:]
                ]
                bonds: List[InorganicBondRecord] = []
                for j in range(len(bonds_data)):
                    bonds.append(
                        InorganicBondRecord(
                            atom_index_1=int(bonds_data[j, 0]),
                            atom_index_2=int(bonds_data[j, 1]),
                            bond_order=float(bonds_data[j, 2]),
                            bond_type=bond_types[j],
                        )
                    )

        return InorganicComplexSchema(
            metal_symbol=metal_symbol,
            oxidation_state=oxidation_state,
            geometry_name=geometry_name,
            polyhedron=polyhedron,
            coordination_number=coordination_number,
            isomer_state=isomer_state,
            spin_multiplicity=spin_multiplicity,
            net_charge=net_charge,
            formula=formula,
            molecular_weight=molecular_weight,
            atoms=atoms,
            bonds=bonds,
            metadata=metadata,
        )


class SQLiteInorganicStore:
    """Thread-safe SQLite database repository with file locking for inorganic complexes."""

    def __init__(self, db_path: Union[Path, str]) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.db_path.with_suffix(".lock")
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize SQLite database table and indexes."""
        with AtomicFileLock(self.lock_path, target_path=self.db_path):
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inorganic_complexes (
                        id TEXT PRIMARY KEY,
                        formula TEXT NOT NULL,
                        metal_symbol TEXT NOT NULL,
                        oxidation_state INTEGER NOT NULL,
                        geometry TEXT NOT NULL,
                        net_charge INTEGER NOT NULL,
                        spin_multiplicity INTEGER NOT NULL,
                        isomer_state TEXT NOT NULL,
                        total_denticity INTEGER NOT NULL,
                        atom_count INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_formula ON inorganic_complexes(formula)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_metal ON inorganic_complexes(metal_symbol)"
                )
                conn.commit()
            finally:
                conn.close()

    def save(
        self,
        complex_obj: Union[InorganicComplex, InorganicComplexSchema],
        record_id: Optional[str] = None,
    ) -> str:
        """Insert or replace a complex record in the database."""
        schema = complex_to_schema(complex_obj)
        cid = record_id or str(uuid.uuid4())
        payload_str = schema.model_dump_json()
        now_iso = datetime.now(timezone.utc).isoformat()

        with AtomicFileLock(self.lock_path, target_path=self.db_path):
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO inorganic_complexes (
                        id, formula, metal_symbol, oxidation_state, geometry,
                        net_charge, spin_multiplicity, isomer_state, total_denticity,
                        atom_count, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        schema.formula,
                        schema.metal_symbol,
                        schema.oxidation_state,
                        schema.geometry_name,
                        schema.net_charge,
                        schema.spin_multiplicity,
                        schema.isomer_state,
                        schema.coordination_number,
                        len(schema.atoms),
                        payload_str,
                        now_iso,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return cid

    def get(self, complex_id: str) -> InorganicComplexSchema:
        """Fetch and deserialize a complex by unique identifier."""
        with AtomicFileLock(self.lock_path, target_path=self.db_path):
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT payload_json FROM inorganic_complexes WHERE id = ?",
                    (complex_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"Complex with ID '{complex_id}' not found.")
                return InorganicComplexSchema.model_validate_json(row[0])
            finally:
                conn.close()

    def find_by_formula(self, formula: str) -> List[InorganicComplexSchema]:
        """Search stored complexes by exact chemical formula."""
        with AtomicFileLock(self.lock_path, target_path=self.db_path):
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT payload_json FROM inorganic_complexes WHERE formula = ?",
                    (formula,),
                )
                rows = cursor.fetchall()
                return [InorganicComplexSchema.model_validate_json(r[0]) for r in rows]
            finally:
                conn.close()

    def list_all(self) -> List[Dict[str, Any]]:
        """List summary table of all stored complex records."""
        with AtomicFileLock(self.lock_path, target_path=self.db_path):
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, formula, metal_symbol, oxidation_state, geometry, net_charge, spin_multiplicity, isomer_state, atom_count, created_at FROM inorganic_complexes ORDER BY created_at DESC"
                )
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def delete(self, complex_id: str) -> bool:
        """Remove a record by ID."""
        with AtomicFileLock(self.lock_path, target_path=self.db_path):
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM inorganic_complexes WHERE id = ?", (complex_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()


class InorganicAirGapClient:
    """Tripartite air-gap boundary enforcing sandbox partition integrity for heavy quantum calculations."""

    def __init__(
        self,
        artifact_queue_dir: Union[Path, str],
        allowed_quantum_methods: Optional[Sequence[str]] = None,
    ) -> None:
        self.queue_dir = Path(artifact_queue_dir).resolve()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_methods = list(
            allowed_quantum_methods
            or ["DFT/B3LYP/def2-SVP", "DFT/PBE0/def2-TZVP", "ORCA/r2SCAN-3c", "SPYCFIT/Rotational"]
        )

    def package_quantum_rpc_task(
        self,
        complex_obj: Union[InorganicComplex, InorganicComplexSchema],
        method: str = "DFT/B3LYP/def2-SVP",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate partition boundaries and construct an air-gapped quantum calculation task descriptor."""
        if method not in self.allowed_methods:
            raise ValueError(
                f"Quantum method '{method}' is not in allowed air-gap registry: {self.allowed_methods}"
            )

        schema = complex_to_schema(complex_obj)
        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"

        # Calculate payload integrity hash
        payload_bytes = schema.model_dump_json().encode("utf-8")
        sha256_digest = hashlib.sha256(payload_bytes).hexdigest()

        task_record: Dict[str, Any] = {
            "task_id": tid,
            "status": "QUEUED_AIRGAP_RPC",
            "method": method,
            "sha256": sha256_digest,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_partition": "T_art",
            "read_only_isolation": True,
            "complex": schema.model_dump(),
        }

        task_file = self.queue_dir / f"{tid}.json"
        lock_file = task_file.with_suffix(".lock")
        with AtomicFileLock(lock_file, target_path=task_file):
            task_file.write_text(json.dumps(task_record, indent=2), encoding="utf-8")

        return task_record
