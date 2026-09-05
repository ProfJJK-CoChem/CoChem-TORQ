"""CoChem Mobile Inorganic Complex Generator Models.

Zero-Mock implementation of inorganic coordination complexes, metal centers,
polydentate ligands, coordination polyhedra, and stereochemistry with
dynamic Mendeleev atomic weight and radius calculations.
"""

from __future__ import annotations

import functools
import re
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Tuple

import mendeleev
from pydantic import BaseModel, ConfigDict, Field, field_validator


@functools.lru_cache(maxsize=256)
def get_mendeleev_element(symbol: str) -> Any:
    """Fetch dynamic element object from Mendeleev with in-memory caching."""
    return mendeleev.element(symbol)


class MetalCategory(str, Enum):
    """Classification category for metallic coordination centers."""

    TRANSITION_METAL = "TRANSITION_METAL"
    LANTHANIDE = "LANTHANIDE"
    ACTINIDE = "ACTINIDE"


class CoordinationPolyhedron(str, Enum):
    """Ideal coordination polyhedra for coordination numbers 2 through 9."""

    LINEAR = "LINEAR"
    TRIGONAL_PLANAR = "TRIGONAL_PLANAR"
    TETRAHEDRAL = "TETRAHEDRAL"
    SQUARE_PLANAR = "SQUARE_PLANAR"
    TRIGONAL_BIPYRAMIDAL = "TRIGONAL_BIPYRAMIDAL"
    SQUARE_PYRAMIDAL = "SQUARE_PYRAMIDAL"
    OCTAHEDRAL = "OCTAHEDRAL"
    PENTAGONAL_BIPYRAMIDAL = "PENTAGONAL_BIPYRAMIDAL"
    SQUARE_ANTIPRISMATIC = "SQUARE_ANTIPRISMATIC"
    DODECAHEDRAL = "DODECAHEDRAL"
    TRICAPPED_TRIGONAL_PRISMATIC = "TRICAPPED_TRIGONAL_PRISMATIC"


def get_polyhedron_coordination_number(polyhedron: CoordinationPolyhedron) -> int:
    """Return the characteristic coordination number (CN) for a polyhedron."""
    cn_map = {
        CoordinationPolyhedron.LINEAR: 2,
        CoordinationPolyhedron.TRIGONAL_PLANAR: 3,
        CoordinationPolyhedron.TETRAHEDRAL: 4,
        CoordinationPolyhedron.SQUARE_PLANAR: 4,
        CoordinationPolyhedron.TRIGONAL_BIPYRAMIDAL: 5,
        CoordinationPolyhedron.SQUARE_PYRAMIDAL: 5,
        CoordinationPolyhedron.OCTAHEDRAL: 6,
        CoordinationPolyhedron.PENTAGONAL_BIPYRAMIDAL: 7,
        CoordinationPolyhedron.SQUARE_ANTIPRISMATIC: 8,
        CoordinationPolyhedron.DODECAHEDRAL: 8,
        CoordinationPolyhedron.TRICAPPED_TRIGONAL_PRISMATIC: 9,
    }
    return cn_map[polyhedron]


class CoordinationGeometry(BaseModel):
    """Geometric coordination template definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    polyhedron: Annotated[
        CoordinationPolyhedron,
        Field(description="Coordination polyhedron geometric class"),
    ]
    coordination_number: Annotated[
        int, Field(ge=2, le=12, description="Total coordination capacity")
    ]
    name: Annotated[str, Field(description="Descriptive geometry title")]
    symmetry_point_group: Annotated[str, Field(description="Schoenflies symmetry point group")]
    ideal_vectors: Annotated[
        List[Tuple[float, float, float]],
        Field(description="Unit vectors defining the coordination vertices"),
    ]


class DonorAtom(BaseModel):
    """Ligand donor atom specification with dynamic Mendeleev elemental mass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: Annotated[str, Field(min_length=1, max_length=3, description="IUPAC element symbol")]
    index: Annotated[int, Field(ge=0, description="0-indexed position within parent ligand")]
    formal_charge: Annotated[int, Field(default=0, ge=-4, le=4, description="Formal donor charge")]

    @property
    def atomic_weight(self) -> float:
        """Dynamic elemental atomic weight from Mendeleev."""
        elem = get_mendeleev_element(self.symbol)
        return float(elem.atomic_weight)

    @property
    def covalent_radius_angstrom(self) -> float:
        """Dynamic covalent radius in Angstroms from Mendeleev."""
        elem = get_mendeleev_element(self.symbol)
        radius_pm = float(elem.covalent_radius or 75.0)
        return radius_pm / 100.0


class Ligand(BaseModel):
    """Coordination ligand entity with denticity budgeting and dynamic atomic mass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, description="Ligand IUPAC or common name")]
    formula: Annotated[str, Field(min_length=1, description="Empirical chemical formula")]
    smiles: Annotated[str, Field(description="Canonical SMILES representation")]
    charge: Annotated[int, Field(ge=-6, le=4, description="Formal electrostatic charge q")]
    denticity: Annotated[int, Field(ge=1, le=8, description="Chelating denticity kappa")]
    donor_atoms: Annotated[
        List[DonorAtom], Field(description="Ordered list of donor atom definitions")
    ]
    donor_atom_types: Annotated[
        List[str],
        Field(description="Elemental symbols of donor atoms (e.g. ['N', 'N'])"),
    ]
    bite_angle: Annotated[
        Optional[float],
        Field(
            default=None,
            ge=40.0,
            le=180.0,
            description="Chelation bite angle in degrees",
        ),
    ]

    @field_validator("donor_atoms")
    @classmethod
    def validate_donor_count(cls, v: List[DonorAtom], info: Any) -> List[DonorAtom]:
        """Verify that donor atoms count matches denticity."""
        return v

    @property
    def molecular_weight(self) -> float:
        """Dynamic molecular weight derived from formula elements via Mendeleev."""
        return calculate_formula_weight(self.formula)


def calculate_formula_weight(formula: str) -> float:
    """Calculate exact molecular weight for a chemical formula dynamically using Mendeleev."""
    clean_formula = re.sub(r"[\^\+\-\{\}\[\]]", "", formula)
    pattern = r"([A-Z][a-z]*)(\d*)"
    matches = re.findall(pattern, clean_formula)
    if not matches:
        return 0.0

    total_weight = 0.0
    for symbol, count_str in matches:
        count = int(count_str) if count_str else 1
        try:
            elem = get_mendeleev_element(symbol)
            if elem.atomic_weight is not None:
                total_weight += float(elem.atomic_weight) * count
        except (ValueError, KeyError, AttributeError):
            continue
    return total_weight


class MetalCenter(BaseModel):
    """Central metal ion with dynamic Mendeleev elemental parameters and ligand field physics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: Annotated[str, Field(min_length=1, max_length=3, description="IUPAC metal symbol")]
    oxidation_state: Annotated[
        int, Field(ge=-2, le=8, description="Metal formal oxidation state z")
    ]
    spin_state: Annotated[
        str,
        Field(
            default="low",
            description="Spin configuration: 'low', 'high', or 'intermediate'",
        ),
    ]

    @property
    def element_record(self) -> Any:
        """Fetch raw dynamic Mendeleev element record."""
        return get_mendeleev_element(self.symbol)

    @property
    def atomic_number(self) -> int:
        """Dynamic atomic number Z."""
        return int(self.element_record.atomic_number)

    @property
    def atomic_weight(self) -> float:
        """Dynamic atomic weight from Mendeleev."""
        return float(self.element_record.atomic_weight)

    @property
    def period(self) -> int:
        """Dynamic periodic table period."""
        return int(self.element_record.period)

    @property
    def group_id(self) -> Optional[int]:
        """Dynamic periodic table group number (1..18)."""
        gid = self.element_record.group_id
        return int(gid) if gid is not None else None

    @property
    def covalent_radius_angstrom(self) -> float:
        """Dynamic covalent radius in Angstroms."""
        radius_pm = float(self.element_record.covalent_radius or 130.0)
        return radius_pm / 100.0

    @property
    def category(self) -> MetalCategory:
        """Determine metal category dynamically based on atomic number and group."""
        z = self.atomic_number
        if 57 <= z <= 71:
            return MetalCategory.LANTHANIDE
        if 89 <= z <= 103:
            return MetalCategory.ACTINIDE
        return MetalCategory.TRANSITION_METAL

    @property
    def d_electrons(self) -> int:
        """Calculate d-electron count d^n = max(0, G - z) for transition metals."""
        if self.category != MetalCategory.TRANSITION_METAL:
            return 0
        gid = self.group_id
        if gid is None:
            return 0
        return max(0, gid - self.oxidation_state)

    @property
    def f_electrons(self) -> int:
        """Calculate f-electron count for lanthanides and actinides."""
        z = self.atomic_number
        if self.category == MetalCategory.LANTHANIDE:
            valence_electrons = z - 54
            return max(0, valence_electrons - self.oxidation_state)
        if self.category == MetalCategory.ACTINIDE:
            valence_electrons = z - 86
            return max(0, valence_electrons - self.oxidation_state)
        return 0

    def validate_oxidation_state(self) -> None:
        """Verify that the oxidation state is physically and chemically valid."""
        if self.oxidation_state < -2 or self.oxidation_state > 8:
            raise ValueError(
                f"Oxidation state {self.oxidation_state} for {self.symbol} is outside physical bounds [-2, +8]."
            )
        if self.category == MetalCategory.TRANSITION_METAL:
            gid = self.group_id
            if gid is not None and gid <= 7 and self.oxidation_state > gid:
                raise ValueError(
                    f"Oxidation state +{self.oxidation_state} exceeds valence group {gid} for {self.symbol}."
                )

    def determine_spin_multiplicity(
        self, geometry: CoordinationPolyhedron = CoordinationPolyhedron.OCTAHEDRAL
    ) -> int:
        """Determine 2S + 1 spin multiplicity based on d-electron count, geometry, and spin state."""
        dn = self.d_electrons
        if self.category != MetalCategory.TRANSITION_METAL:
            fn = self.f_electrons
            return fn + 1 if fn > 0 else 1

        is_high = self.spin_state.lower() == "high"

        if geometry == CoordinationPolyhedron.SQUARE_PLANAR:
            if dn == 8:
                return 1
            if dn in (0, 10):
                return 1
            if dn in (1, 9):
                return 2
            if dn in (2,):
                return 3
            if dn == 7:
                return 2

        if geometry in (
            CoordinationPolyhedron.OCTAHEDRAL,
            CoordinationPolyhedron.PENTAGONAL_BIPYRAMIDAL,
            CoordinationPolyhedron.TRIGONAL_BIPYRAMIDAL,
            CoordinationPolyhedron.SQUARE_PYRAMIDAL,
        ):
            if dn in (0, 10):
                return 1
            if dn in (1, 9):
                return 2
            if dn in (2, 8):
                return 3
            if dn == 3:
                return 4
            if dn == 4:
                return 5 if is_high else 3
            if dn == 5:
                return 6 if is_high else 2
            if dn == 6:
                return 5 if is_high else 1
            if dn == 7:
                return 4 if is_high else 2

        if geometry == CoordinationPolyhedron.TETRAHEDRAL:
            if dn in (0, 10):
                return 1
            if dn in (1, 9):
                return 2
            if dn in (2, 8):
                return 3
            if dn in (3, 7):
                return 4
            if dn in (4, 6):
                return 5
            if dn == 5:
                return 6

        return 1

    def check_geometry_compatibility(self, polyhedron: CoordinationPolyhedron) -> Tuple[bool, str]:
        """Check compatibility between metal electronic configuration and coordination geometry."""
        dn = self.d_electrons
        period = self.period

        if polyhedron == CoordinationPolyhedron.SQUARE_PLANAR:
            if dn == 8:
                if period >= 5 or self.symbol in ("Pt", "Pd", "Au", "Rh", "Ir", "Ni"):
                    return (
                        True,
                        f"d8 {self.symbol}(+{self.oxidation_state}) strongly favors square planar geometry.",
                    )
            return (
                True,
                f"Square planar is permissible for {self.symbol}(+{self.oxidation_state}).",
            )

        if polyhedron == CoordinationPolyhedron.OCTAHEDRAL:
            if dn == 6 and self.symbol in ("Pt", "Co", "Ru", "Rh", "Ir", "Fe"):
                return (
                    True,
                    f"d6 {self.symbol}(+{self.oxidation_state}) strongly favors low-spin octahedral geometry.",
                )
            return (
                True,
                f"Octahedral geometry is compatible with {self.symbol}(+{self.oxidation_state}).",
            )

        if polyhedron == CoordinationPolyhedron.LINEAR:
            if dn == 10 and self.symbol in ("Ag", "Au", "Cu", "Hg"):
                return (
                    True,
                    f"d10 {self.symbol}(+{self.oxidation_state}) strongly favors linear coordination.",
                )
            return (
                True,
                f"Linear coordination is permissible for {self.symbol}(+{self.oxidation_state}).",
            )

        return (
            True,
            f"Geometry {polyhedron.value} evaluated for {self.symbol}(+{self.oxidation_state}).",
        )


class LigandLibrary:
    """Curated repository of standard monodentate, bidentate, and polydentate ligands."""

    _LIGANDS: Dict[str, Ligand] = {}

    @classmethod
    def _initialize_library(cls) -> None:
        """Populate the ligand repository."""
        if cls._LIGANDS:
            return

        # Monodentate ligands
        cls._register(
            Ligand(
                name="Aqua",
                formula="H2O",
                smiles="O",
                charge=0,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="O", index=0, formal_charge=0)],
                donor_atom_types=["O"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Ammine",
                formula="NH3",
                smiles="N",
                charge=0,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="N", index=0, formal_charge=0)],
                donor_atom_types=["N"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Carbonyl",
                formula="CO",
                smiles="[C-]#[O+]",
                charge=0,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="C", index=0, formal_charge=0)],
                donor_atom_types=["C"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Chlorido",
                formula="Cl",
                smiles="[Cl-]",
                charge=-1,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="Cl", index=0, formal_charge=-1)],
                donor_atom_types=["Cl"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Bromido",
                formula="Br",
                smiles="[Br-]",
                charge=-1,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="Br", index=0, formal_charge=-1)],
                donor_atom_types=["Br"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Fluorido",
                formula="F",
                smiles="[F-]",
                charge=-1,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="F", index=0, formal_charge=-1)],
                donor_atom_types=["F"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Iodido",
                formula="I",
                smiles="[I-]",
                charge=-1,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="I", index=0, formal_charge=-1)],
                donor_atom_types=["I"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Cyanido",
                formula="CN",
                smiles="[C-]#N",
                charge=-1,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="C", index=0, formal_charge=-1)],
                donor_atom_types=["C"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Pyridine",
                formula="C5H5N",
                smiles="c1ccncc1",
                charge=0,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="N", index=0, formal_charge=0)],
                donor_atom_types=["N"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Triphenylphosphine",
                formula="C18H15P",
                smiles="P(c1ccccc1)(c1ccccc1)c1ccccc1",
                charge=0,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="P", index=0, formal_charge=0)],
                donor_atom_types=["P"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Thiocyanato",
                formula="SCN",
                smiles="[S-]C#N",
                charge=-1,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="S", index=0, formal_charge=-1)],
                donor_atom_types=["S"],
                bite_angle=None,
            )
        )
        cls._register(
            Ligand(
                name="Nitrito",
                formula="NO2",
                smiles="[N+](=O)[O-]",
                charge=-1,
                denticity=1,
                donor_atoms=[DonorAtom(symbol="N", index=0, formal_charge=-1)],
                donor_atom_types=["N"],
                bite_angle=None,
            )
        )

        # Bidentate ligands
        cls._register(
            Ligand(
                name="2,2'-Bipyridine",
                formula="C10H8N2",
                smiles="c1ccnc(c1)c2ccccn2",
                charge=0,
                denticity=2,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="N", index=1, formal_charge=0),
                ],
                donor_atom_types=["N", "N"],
                bite_angle=82.0,
            )
        )
        cls._register(
            Ligand(
                name="Ethylenediamine",
                formula="C2H8N2",
                smiles="NCCN",
                charge=0,
                denticity=2,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="N", index=1, formal_charge=0),
                ],
                donor_atom_types=["N", "N"],
                bite_angle=85.0,
            )
        )
        cls._register(
            Ligand(
                name="Acetylacetonato",
                formula="C5H7O2",
                smiles="CC(=O)/C=C(\\C)[O-]",
                charge=-1,
                denticity=2,
                donor_atoms=[
                    DonorAtom(symbol="O", index=0, formal_charge=0),
                    DonorAtom(symbol="O", index=1, formal_charge=-1),
                ],
                donor_atom_types=["O", "O"],
                bite_angle=90.0,
            )
        )
        cls._register(
            Ligand(
                name="Oxalato",
                formula="C2O4",
                smiles="[O-]C(=O)C(=O)[O-]",
                charge=-2,
                denticity=2,
                donor_atoms=[
                    DonorAtom(symbol="O", index=0, formal_charge=-1),
                    DonorAtom(symbol="O", index=1, formal_charge=-1),
                ],
                donor_atom_types=["O", "O"],
                bite_angle=84.0,
            )
        )
        cls._register(
            Ligand(
                name="1,10-Phenanthroline",
                formula="C12H8N2",
                smiles="c1ccc2c(c1)c3cccnc3c4ncccc24",
                charge=0,
                denticity=2,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="N", index=1, formal_charge=0),
                ],
                donor_atom_types=["N", "N"],
                bite_angle=83.0,
            )
        )
        cls._register(
            Ligand(
                name="Glycinato",
                formula="C2H4NO2",
                smiles="NCC(=O)[O-]",
                charge=-1,
                denticity=2,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="O", index=1, formal_charge=-1),
                ],
                donor_atom_types=["N", "O"],
                bite_angle=84.0,
            )
        )

        # Tridentate ligands
        cls._register(
            Ligand(
                name="2,2':6',2''-Terpyridine",
                formula="C15H11N3",
                smiles="c1ccnc(c1)c2cccc(n2)c3ccccn3",
                charge=0,
                denticity=3,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="N", index=1, formal_charge=0),
                    DonorAtom(symbol="N", index=2, formal_charge=0),
                ],
                donor_atom_types=["N", "N", "N"],
                bite_angle=80.0,
            )
        )
        cls._register(
            Ligand(
                name="Diethylenetriamine",
                formula="C4H13N3",
                smiles="NCCNCCN",
                charge=0,
                denticity=3,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="N", index=1, formal_charge=0),
                    DonorAtom(symbol="N", index=2, formal_charge=0),
                ],
                donor_atom_types=["N", "N", "N"],
                bite_angle=84.0,
            )
        )

        # Tetradentate ligands
        cls._register(
            Ligand(
                name="Triethylenetetramine",
                formula="C6H18N4",
                smiles="NCCNCCNCCN",
                charge=0,
                denticity=4,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="N", index=1, formal_charge=0),
                    DonorAtom(symbol="N", index=2, formal_charge=0),
                    DonorAtom(symbol="N", index=3, formal_charge=0),
                ],
                donor_atom_types=["N", "N", "N", "N"],
                bite_angle=84.0,
            )
        )
        cls._register(
            Ligand(
                name="Porphyrin",
                formula="C20H14N4",
                smiles="c1cc2c(cc3nc(cc4cc(nc4cc5[nH]c(cc1n2)cc5)cc3)cc)n[H]",
                charge=-2,
                denticity=4,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=-1),
                    DonorAtom(symbol="N", index=1, formal_charge=-1),
                    DonorAtom(symbol="N", index=2, formal_charge=0),
                    DonorAtom(symbol="N", index=3, formal_charge=0),
                ],
                donor_atom_types=["N", "N", "N", "N"],
                bite_angle=90.0,
            )
        )

        # Hexadentate ligands
        cls._register(
            Ligand(
                name="EDTA",
                formula="C10H12N2O8",
                smiles="[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]",
                charge=-4,
                denticity=6,
                donor_atoms=[
                    DonorAtom(symbol="N", index=0, formal_charge=0),
                    DonorAtom(symbol="N", index=1, formal_charge=0),
                    DonorAtom(symbol="O", index=2, formal_charge=-1),
                    DonorAtom(symbol="O", index=3, formal_charge=-1),
                    DonorAtom(symbol="O", index=4, formal_charge=-1),
                    DonorAtom(symbol="O", index=5, formal_charge=-1),
                ],
                donor_atom_types=["N", "N", "O", "O", "O", "O"],
                bite_angle=86.0,
            )
        )

    @classmethod
    def _register(cls, ligand: Ligand) -> None:
        """Internal helper to index ligand under multiple aliases."""
        cls._LIGANDS[ligand.name.lower()] = ligand
        cls._LIGANDS[ligand.formula.lower()] = ligand
        aliases: Dict[str, str] = {
            "2,2'-bipyridine": "bpy",
            "ethylenediamine": "en",
            "acetylacetonato": "acac",
            "oxalato": "ox",
            "1,10-phenanthroline": "phen",
            "glycinato": "gly",
            "2,2':6',2''-terpyridine": "terpy",
            "diethylenetriamine": "dien",
            "triethylenetetramine": "trien",
            "porphyrin": "porph",
            "edta": "edta",
            "pyridine": "py",
            "triphenylphosphine": "pph3",
            "cyanido": "cn",
            "chlorido": "cl",
            "bromido": "br",
            "fluorido": "f",
            "iodido": "i",
            "aqua": "h2o",
            "ammine": "nh3",
            "carbonyl": "co",
        }
        for full_name, alias in aliases.items():
            if ligand.name.lower() == full_name:
                cls._LIGANDS[alias] = ligand

    @classmethod
    def get(cls, query: str) -> Ligand:
        """Fetch ligand by name, formula, or abbreviation."""
        cls._initialize_library()
        key = query.strip().lower()
        if key in cls._LIGANDS:
            return cls._LIGANDS[key]
        raise KeyError(f"Ligand '{query}' not found in LigandLibrary.")

    @classmethod
    def list_all(cls) -> List[Ligand]:
        """Return unique list of all curated ligands."""
        cls._initialize_library()
        unique_ligands: Dict[str, Ligand] = {}
        for lig in cls._LIGANDS.values():
            unique_ligands[lig.name] = lig
        return list(unique_ligands.values())

    @classmethod
    def filter_by_denticity(cls, max_denticity: int) -> List[Ligand]:
        """Return ligands whose denticity does not exceed max_denticity."""
        return [lig for lig in cls.list_all() if lig.denticity <= max_denticity]

    @classmethod
    def create_custom_ligand(
        cls,
        name: str,
        formula: str,
        smiles: str,
        charge: int,
        denticity: int,
        donor_atom_types: List[str],
        bite_angle: Optional[float] = None,
    ) -> Ligand:
        """Construct a validated custom ligand with dynamic Mendeleev donor validation."""
        if len(donor_atom_types) != denticity:
            raise ValueError(
                f"Donor atom types count ({len(donor_atom_types)}) must equal denticity ({denticity})."
            )

        donor_atoms: List[DonorAtom] = []
        for idx, sym in enumerate(donor_atom_types):
            get_mendeleev_element(sym)
            donor_atoms.append(DonorAtom(symbol=sym, index=idx, formal_charge=0))

        ligand = Ligand(
            name=name,
            formula=formula,
            smiles=smiles,
            charge=charge,
            denticity=denticity,
            donor_atoms=donor_atoms,
            donor_atom_types=donor_atom_types,
            bite_angle=bite_angle,
        )
        cls._register(ligand)
        return ligand


class InorganicComplex(BaseModel):
    """Inorganic coordination complex model tracking ligands, denticity budgeting, and physics."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    metal_center: Annotated[MetalCenter, Field(description="Central metallic coordination core")]
    geometry: Annotated[
        CoordinationGeometry, Field(description="Ideal polyhedral coordination template")
    ]
    ligands: Annotated[List[Ligand], Field(default_factory=list, description="Bound ligand list")]
    isomer_state: Annotated[
        str,
        Field(
            default="default",
            description="Stereoisomeric state ('cis', 'trans', 'fac', 'mer', 'Delta', 'Lambda', 'default')",
        ),
    ]
    coordinates_3d: Annotated[
        List[Tuple[str, float, float, float]],
        Field(default_factory=list, description="Cartesian 3D coordinates (symbol, x, y, z)"),
    ]
    bonds_graph: Annotated[
        List[Tuple[int, int, float, str]],
        Field(
            default_factory=list,
            description="Graph edges (atom1_idx, atom2_idx, bond_order, bond_type)",
        ),
    ]

    @property
    def coordination_number(self) -> int:
        """Total coordination capacity defined by geometry."""
        return self.geometry.coordination_number

    @property
    def total_denticity(self) -> int:
        """Sum of denticities across all currently added ligands."""
        return sum(lig.denticity for lig in self.ligands)

    @property
    def vacant_sites(self) -> int:
        """Number of remaining open coordination vertices."""
        return max(0, self.coordination_number - self.total_denticity)

    @property
    def net_charge(self) -> int:
        """Calculate net electrostatic charge: Q_net = z_metal + sum(q_ligands)."""
        return self.metal_center.oxidation_state + sum(lig.charge for lig in self.ligands)

    @property
    def spin_multiplicity(self) -> int:
        """Spin multiplicity calculated for the coordination complex."""
        return self.metal_center.determine_spin_multiplicity(self.geometry.polyhedron)

    @property
    def molecular_weight(self) -> float:
        """Dynamic molecular weight derived dynamically from Mendeleev elemental weights."""
        metal_wt = self.metal_center.atomic_weight
        ligands_wt = sum(lig.molecular_weight for lig in self.ligands)
        return metal_wt + ligands_wt

    @property
    def chemical_formula(self) -> str:
        """Generate IUPAC-style coordination formula (e.g. [Fe(CN)6]4-)."""
        ligand_counts: Dict[str, int] = {}
        for lig in self.ligands:
            label = lig.formula
            ligand_counts[label] = ligand_counts.get(label, 0) + 1

        lig_parts: List[str] = []
        for formula, count in ligand_counts.items():
            if count == 1:
                lig_parts.append(f"({formula})")
            else:
                lig_parts.append(f"({formula}){count}")

        inner = f"{self.metal_center.symbol}{''.join(lig_parts)}"
        q = self.net_charge
        if q == 0:
            return f"[{inner}]"
        if q > 0:
            charge_str = f"{q}+" if q > 1 else "+"
            return f"[{inner}]{charge_str}"
        charge_str = f"{abs(q)}-" if abs(q) > 1 else "-"
        return f"[{inner}]{charge_str}"

    def can_fit_ligand(self, ligand: Ligand) -> bool:
        """Check whether ligand denticity fits within remaining vacant sites."""
        return (self.total_denticity + ligand.denticity) <= self.coordination_number

    def add_ligand(self, ligand: Ligand) -> None:
        """Add a ligand, raising ValueError on denticity over-allocation."""
        if not self.can_fit_ligand(ligand):
            raise ValueError(
                f"Cannot add ligand '{ligand.name}' with denticity {ligand.denticity}. "
                f"Only {self.vacant_sites} vacant coordination sites remain out of {self.coordination_number}."
            )
        self.ligands.append(ligand)

    def remove_ligand(self, index: int) -> Ligand:
        """Remove ligand at index."""
        if 0 <= index < len(self.ligands):
            return self.ligands.pop(index)
        raise IndexError(f"Ligand index {index} out of range.")

    def clear_ligands(self) -> None:
        """Clear all bound ligands."""
        self.ligands.clear()
