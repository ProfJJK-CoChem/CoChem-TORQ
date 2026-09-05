"""CoChem Mobile Inorganic Coordination Complex Module (SRS Chunk 06).

Zero-Mock implementation of touch-optimized Inorganic Complex Generator UI,
polyhedral coordinate templates (CN=2..9), chelate stereochemistry assemblers,
Pydantic v2 schemas, and dynamic Mendeleev atomic weight integration.
"""

from __future__ import annotations

from cochem.mobile.inorganic.engine import (
    ChelateAssembler,
    CoordinateAssembler,
    InorganicAssemblyEngine,
    IsomerResolver,
    PolyhedronTemplateRegistry,
)
from cochem.mobile.inorganic.models import (
    CoordinationGeometry,
    CoordinationPolyhedron,
    DonorAtom,
    InorganicComplex,
    Ligand,
    LigandLibrary,
    MetalCategory,
    MetalCenter,
    calculate_formula_weight,
    get_polyhedron_coordination_number,
)
from cochem.mobile.inorganic.storage import (
    HDF5InorganicSerializer,
    InorganicAirGapClient,
    InorganicAtom3D,
    InorganicBondRecord,
    InorganicComplexSchema,
    JSONInorganicSerializer,
    SQLiteInorganicStore,
    complex_to_schema,
)
from cochem.mobile.inorganic.ui import (
    ComplexSummaryCard,
    InorganicBuilderScreen,
    InorganicBuilderWidget,
    IsomerPickerWidget,
    LigandBudgetWidget,
    LigandSelectorDialog,
)

__all__ = [
    "ChelateAssembler",
    "ComplexSummaryCard",
    "CoordinateAssembler",
    "CoordinationGeometry",
    "CoordinationPolyhedron",
    "DonorAtom",
    "HDF5InorganicSerializer",
    "InorganicAirGapClient",
    "InorganicAssemblyEngine",
    "InorganicAtom3D",
    "InorganicBondRecord",
    "InorganicBuilderScreen",
    "InorganicBuilderWidget",
    "InorganicComplex",
    "InorganicComplexSchema",
    "IsomerPickerWidget",
    "IsomerResolver",
    "JSONInorganicSerializer",
    "Ligand",
    "LigandBudgetWidget",
    "LigandLibrary",
    "LigandSelectorDialog",
    "MetalCategory",
    "MetalCenter",
    "PolyhedronTemplateRegistry",
    "SQLiteInorganicStore",
    "calculate_formula_weight",
    "complex_to_schema",
    "get_polyhedron_coordination_number",
]
