"""Typed domain-specific exception hierarchy for CoChem Mobile complex assembly.

Derived strictly from CoChemMOBException, ensuring full traceability and zero-stub compliance.
"""

from __future__ import annotations


class CoChemMOBException(Exception):
    """Base exception for all CoChem Mobile subsystem errors."""


class MendeleevLookupError(CoChemMOBException):
    """Raised when dynamic elemental property retrieval fails or element is unresolvable."""


class CoordinationGeometryMismatchError(CoChemMOBException):
    """Raised when total ligand denticity does not match coordination number."""


class SingularRotationAxisError(CoChemMOBException):
    """Raised when Rodrigues rotation axis vector has zero norm or fails normalization."""


class KabschReflectionError(CoChemMOBException):
    """Raised when improper rotation (det(R) = -1) fails SO(3) sign-correction."""


class StericClashDetectedError(CoChemMOBException):
    """Raised when dihedral sweep and constrained UFF relaxation fail to resolve steric overlap."""


class UnphysicalMonomerSeparationError(CoChemMOBException):
    """Raised when center-of-mass ligand-metal distance violates physical asymptotic bounds."""


class QuantumParityError(CoChemMOBException):
    """Raised when total electron count and spin multiplicity violate quantum parity."""


class SWMRStorageLockError(CoChemMOBException):
    """Raised when cross-platform filelock acquisition fails or times out for HDF5 storage."""
