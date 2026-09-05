"""Domain-specific typed exceptions for CoChem-TOPOS Graph Theory and Topology subsystem."""

from __future__ import annotations


try:
    from cochem_base.exceptions import CoChemError  # type: ignore[import-not-found]
except ImportError:
    class CoChemError(Exception):  # type: ignore[no-redef]
        """Root fallback exception for CoChem errors."""


class TopologyError(CoChemError):
    """Raised when graph topological invariants, contiguity, or conservation laws fail."""


class StericClashError(CoChemError):
    """Raised when geometric steric clashes are detected or physical radii are undefined."""


class IsomorphismMismatchError(CoChemError):
    """Raised when subgraph isomorphism matching fails or violates attribute constraints."""


class ChiralityAssignmentError(CoChemError):
    """Raised when stereocenter assignment encounters degenerate, planar, or collinear geometries."""


class CoChemToposException(TopologyError):
    """Root domain exception for CoChem-TOPOS Graph Theory operations."""


class SolventBuilderError(CoChemToposException):
    """Raised when explicit solvent builder encounters invalid geometry, density, or bounding box."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class TopologicalCanonicalizationError(CoChemToposException):
    """Raised when topological graph canonicalization or isomorphism invariant indexing fails."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)

