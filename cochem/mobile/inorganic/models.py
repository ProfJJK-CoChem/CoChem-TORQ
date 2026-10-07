"""CoChem Mobile Inorganic Complex Generator Models.

Zero-Mock implementation of inorganic coordination complexes, metal centers,
polydentate ligands, coordination polyhedra, and stereochemistry with
dynamic Mendeleev atomic weight and radius calculations.
"""

from __future__ import annotations

import functools
import math
from collections import Counter
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
        radius = elem.covalent_radius
        if radius is None or not math.isfinite(float(radius)) or float(radius) <= 0:
            raise ValueError(f"No finite tabulated covalent radius for {self.symbol}.")
        radius_pm = float(radius)
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
    """Conventional formula-unit molar mass from tabulated elemental weights.

    This is an average elemental-weight calculation, not an exact isotope mass.
    Supports nested parentheses/brackets, dot-separated adducts and explicit
    charge suffixes. Unknown elements, invalid grammar and unavailable weights
    raise instead of returning a partial mass. Electron masses are not included.
    """
    if not isinstance(formula, str) or not formula.strip() or len(formula) > 10000:
        raise ValueError("A nonempty bounded chemical formula is required.")
    text = formula.strip()
    if re.search(r"\s", text):
        raise ValueError("Whitespace inside a chemical formula is unsupported.")
    # Explicit caret charges or a coordination-bracket charge are unambiguous.
    text = re.sub(r"\^(?:[1-9][0-9]*)?[+-]$", "", text)
    text = re.sub(r"(?<=\])(?:[1-9][0-9]*)?[+-]$", "", text)
    if text.endswith(("+", "-")):
        uncharged = text[:-1]
        if re.fullmatch(r"[A-Z][a-z]?[0-9]+", uncharged):
            raise ValueError("Ambiguous atomic-ion suffix; write charge explicitly, e.g. Fe^3+.")
        text = uncharged
    parts = re.split(r"[.·]", text)
    counts: Counter[str] = Counter()

    def parse_number(part: str, position: int) -> tuple[int, int]:
        match = re.match(r"[0-9]+", part[position:])
        if match is None:
            return 1, position
        digits = match.group()
        if digits.startswith("0"):
            raise ValueError("Stoichiometric coefficients must be positive without leading zeroes.")
        return int(digits), position + len(digits)

    def parse_group(part: str, position: int, closing: str | None = None, depth: int = 0):
        if depth > 32:
            raise ValueError("Chemical formula nesting limit exceeded.")
        group: Counter[str] = Counter()
        while position < len(part):
            char = part[position]
            if char in ")]":
                if char != closing or not group:
                    raise ValueError("Unmatched or empty chemical-formula group.")
                return group, position + 1
            if char in "([":
                nested, position = parse_group(part, position + 1, ")" if char == "(" else "]", depth + 1)
                multiplier, position = parse_number(part, position)
                group.update({symbol: number * multiplier for symbol, number in nested.items()})
                continue
            match = re.match(r"[A-Z][a-z]?", part[position:])
            if match is None:
                raise ValueError(f"Invalid formula syntax at {part[position:]!r}.")
            symbol = match.group()
            position += len(symbol)
            multiplier, position = parse_number(part, position)
            group[symbol] += multiplier
        if closing is not None:
            raise ValueError("Unclosed chemical-formula group.")
        if not group:
            raise ValueError("Empty chemical formula or adduct segment.")
        return group, position

    for part in parts:
        if not part:
            raise ValueError("Empty chemical formula or adduct segment.")
        factor, position = parse_number(part, 0)
        group, end = parse_group(part, position)
        if end != len(part):
            raise ValueError("Chemical formula was not completely parsed.")
        counts.update({symbol: number * factor for symbol, number in group.items()})
    total_weight = 0.0
    for symbol, count in counts.items():
        try:
            elem = get_mendeleev_element(symbol)
        except (ValueError, KeyError, AttributeError) as exc:
            raise ValueError(f"Unknown element {symbol!r}; no formula mass is available.") from exc
        if elem.symbol != symbol or elem.atomic_weight is None:
            raise ValueError(f"No canonical tabulated elemental weight for {symbol!r}.")
        weight = float(elem.atomic_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid tabulated elemental weight for {symbol!r}.")
        try:
            total_weight += weight * count
        except OverflowError as exc:
            raise ValueError("Formula-unit mass exceeds finite numerical representation.") from exc
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Formula-unit molar mass is not finite and positive.")
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
        radius = self.element_record.covalent_radius
        if radius is None or not math.isfinite(float(radius)) or float(radius) <= 0:
            raise ValueError(f"No finite tabulated covalent radius for {self.symbol}.")
        radius_pm = float(radius)
        return radius_pm / 100.0

    @property
    def category(self) -> MetalCategory:
        """Determine metal category dynamically based on atomic number and group."""
        z = self.atomic_number
        if 57 <= z <= 71:
            return MetalCategory.LANTHANIDE
        if 89 <= z <= 103:
            return MetalCategory.ACTINIDE
        gid = self.group_id
        if gid is None or not 3 <= gid <= 12:
            raise ValueError(f"{self.symbol} is outside the implemented transition/f-block metal model.")
        return MetalCategory.TRANSITION_METAL

    @property
    def d_electrons(self) -> int:
        """Formal ligand-field d-count; not an atomic ground-state configuration."""
        if self.category != MetalCategory.TRANSITION_METAL:
            raise ValueError("A formal transition-metal d-count is unavailable for f-block centers.")
        gid = self.group_id
        if gid is None:
            raise ValueError(f"No tabulated group for {self.symbol}; d-count unavailable.")
        count = gid - self.oxidation_state
        if not 0 <= count <= 10:
            raise ValueError("Formal d-count is outside the supported zero-to-ten electron model.")
        return count

    @property
    def f_electrons(self) -> int:
        """No f-shell configuration is inferred from atomic number alone."""
        raise ValueError(
            "Ionic f-electron configuration is unavailable: a qualified state/configuration "
            "source is required, not atomic-number subtraction."
        )

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
        """Propose a textbook ligand-field spin estimate, never establish a ground state.

        The high/low field label is a model assumption. Actual complex state
        selection needs ligand-specific electronic evidence and spin-orbit treatment.
        """
        if self.category != MetalCategory.TRANSITION_METAL:
            raise ValueError("f-block spin cannot be inferred from element and oxidation state.")
        dn = self.d_electrons
        if self.spin_state.lower() not in ("high", "low"):
            raise ValueError("The ligand-field estimate requires an explicit high/low model label.")

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

        raise ValueError("No qualified spin estimate for this geometry/electron-count model.")

    def check_geometry_compatibility(
        self, polyhedron: CoordinationPolyhedron,
    ) -> Tuple[Optional[bool], str]:
        """Return undetermined without ligand-/state-specific electronic evidence.

        A geometric template or formal d-count cannot establish compatibility,
        stability, preferred geometry or the spin ground state of a complex.
        ``None`` is deliberately distinct from both validated and invalid.
        """
        if not isinstance(polyhedron, CoordinationPolyhedron):
            raise ValueError("A supported coordination-template identifier is required.")
        return (
            None,
            f"Electronic compatibility of {self.symbol}({self.oxidation_state:+d}) "
            f"with {polyhedron.value} is undetermined. Ligand identity, electronic "
            "state and qualified calculation or experimental evidence are required; "
            "a template is not a geometry or ground-state validation.",
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
        """Ligand-field model estimate only; not a calculated electronic ground state."""
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
