"""Inorganic scientific APIs with notebook UI dependencies loaded on demand.

Models, assembly and storage can be imported with their own dependencies.
Requesting a UI export imports its genuine widget implementation and propagates
ImportError when the optional notebook dependencies are unavailable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    'ChelateAssembler': ('cochem.mobile.inorganic.engine', 'ChelateAssembler'),
    'ComplexSummaryCard': ('cochem.mobile.inorganic.ui', 'ComplexSummaryCard'),
    'CoordinateAssembler': ('cochem.mobile.inorganic.engine', 'CoordinateAssembler'),
    'CoordinationGeometry': ('cochem.mobile.inorganic.models', 'CoordinationGeometry'),
    'CoordinationPolyhedron': ('cochem.mobile.inorganic.models', 'CoordinationPolyhedron'),
    'DonorAtom': ('cochem.mobile.inorganic.models', 'DonorAtom'),
    'HDF5InorganicSerializer': ('cochem.mobile.inorganic.storage', 'HDF5InorganicSerializer'),
    'InorganicAirGapClient': ('cochem.mobile.inorganic.storage', 'InorganicAirGapClient'),
    'InorganicAssemblyEngine': ('cochem.mobile.inorganic.engine', 'InorganicAssemblyEngine'),
    'InorganicAtom3D': ('cochem.mobile.inorganic.storage', 'InorganicAtom3D'),
    'InorganicBondRecord': ('cochem.mobile.inorganic.storage', 'InorganicBondRecord'),
    'InorganicBuilderScreen': ('cochem.mobile.inorganic.ui', 'InorganicBuilderScreen'),
    'InorganicBuilderWidget': ('cochem.mobile.inorganic.ui', 'InorganicBuilderWidget'),
    'InorganicComplex': ('cochem.mobile.inorganic.models', 'InorganicComplex'),
    'InorganicComplexSchema': ('cochem.mobile.inorganic.storage', 'InorganicComplexSchema'),
    'IsomerPickerWidget': ('cochem.mobile.inorganic.ui', 'IsomerPickerWidget'),
    'IsomerResolver': ('cochem.mobile.inorganic.engine', 'IsomerResolver'),
    'JSONInorganicSerializer': ('cochem.mobile.inorganic.storage', 'JSONInorganicSerializer'),
    'Ligand': ('cochem.mobile.inorganic.models', 'Ligand'),
    'LigandBudgetWidget': ('cochem.mobile.inorganic.ui', 'LigandBudgetWidget'),
    'LigandLibrary': ('cochem.mobile.inorganic.models', 'LigandLibrary'),
    'LigandSelectorDialog': ('cochem.mobile.inorganic.ui', 'LigandSelectorDialog'),
    'MetalCategory': ('cochem.mobile.inorganic.models', 'MetalCategory'),
    'MetalCenter': ('cochem.mobile.inorganic.models', 'MetalCenter'),
    'PolyhedronTemplateRegistry': ('cochem.mobile.inorganic.engine', 'PolyhedronTemplateRegistry'),
    'SQLiteInorganicStore': ('cochem.mobile.inorganic.storage', 'SQLiteInorganicStore'),
    'calculate_formula_weight': ('cochem.mobile.inorganic.models', 'calculate_formula_weight'),
    'complex_to_schema': ('cochem.mobile.inorganic.storage', 'complex_to_schema'),
    'get_polyhedron_coordination_number': ('cochem.mobile.inorganic.models', 'get_polyhedron_coordination_number'),
}

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

def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    # Do not replace a missing widget dependency with a placeholder UI.
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
