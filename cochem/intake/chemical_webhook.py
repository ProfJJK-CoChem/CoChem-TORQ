"""Air-Gapped Chemical Webhook Payload Validator & Quantum Invariant Engine.

Validates chemical ingress payloads against Pydantic schema, dynamic element definitions
from Mendeleev, coordinate finiteness, net charge conservation, and quantum spin-parity bounds.
"""

import math
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

from mendeleev import element as mendeleev_element
from pydantic import ConfigDict, Field, model_validator
from pydantic.main import BaseModel

Coordinate3D = Annotated[List[float], Field(min_length=3, max_length=3)]


class ChemicalPayloadSchema(BaseModel):
    """Pydantic v2 chemical webhook payload model enforcing quantum invariants."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_id: str = Field(
        ..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.]+$"
    )
    smiles: Optional[str] = Field(
        default=None, min_length=1, max_length=4096, pattern=r"^[\x20-\x7E]+$"
    )
    inchi: Optional[str] = Field(
        default=None, min_length=1, max_length=4096, pattern=r"^[\x20-\x7E]+$"
    )
    symbols: Optional[List[str]] = Field(
        default=None, min_length=1, max_length=1000
    )
    coordinates_3d: Optional[List[Coordinate3D]] = Field(
        default=None, min_length=1, max_length=1000
    )
    charge: int = Field(default=0, ge=-20, le=20)
    spin_multiplicity: int = Field(default=1, ge=1, le=10)
    energy_unit: Literal["hartree", "kcal/mol", "eV"] = "hartree"

    @model_validator(mode="after")
    def validate_chemical_invariants(self) -> "ChemicalPayloadSchema":
        """Validate chemical representation completeness and quantum spin-parity invariants."""
        has_coords = self.coordinates_3d is not None
        has_symbols = self.symbols is not None

        if has_coords != has_symbols:
            raise ValueError(
                "Both 'coordinates_3d' and 'symbols' must be provided together."
            )

        has_repr = (
            bool(self.smiles)
            or bool(self.inchi)
            or (has_coords and has_symbols)
        )
        if not has_repr:
            raise ValueError(
                "Payload must contain at least one valid chemical representation: "
                "'smiles', 'inchi', or both 'coordinates_3d' and 'symbols'."
            )

        if has_coords and has_symbols:
            assert self.symbols is not None and self.coordinates_3d is not None

            if len(self.symbols) != len(self.coordinates_3d):
                raise ValueError(
                    f"Length mismatch: {len(self.symbols)} symbols vs "
                    f"{len(self.coordinates_3d)} coordinates"
                )

            z_tot = 0
            for sym in self.symbols:
                canon_sym = sym.strip().capitalize()
                try:
                    elem = mendeleev_element(canon_sym)
                    atomic_number = int(elem.atomic_number)
                    if not (1 <= atomic_number <= 118):
                        raise ValueError(
                            f"Atomic number out of range: {atomic_number}"
                        )
                    z_tot += atomic_number
                except Exception as err:
                    raise ValueError(
                        f"Invalid chemical element: '{sym}'"
                    ) from err

            for coord_vec in self.coordinates_3d:
                for coord in coord_vec:
                    if not math.isfinite(coord):
                        raise ValueError(
                            f"Non-finite floating-point coordinate detected: {coord}"
                        )

            n_electrons = z_tot - self.charge
            if n_electrons <= 0:
                raise ValueError(
                    f"Non-positive electron count (N_elec={n_electrons}): "
                    f"total nuclear charge Z={z_tot}, charge={self.charge}"
                )

            if self.spin_multiplicity > n_electrons + 1:
                raise ValueError(
                    f"Physical spin multiplicity bound violated: "
                    f"2S+1={self.spin_multiplicity} exceeds maximum {n_electrons + 1} "
                    f"for system with N_elec={n_electrons} electrons."
                )

            if (n_electrons % 2) == (self.spin_multiplicity % 2):
                expected_parity = "odd" if (n_electrons % 2 == 0) else "even"
                raise ValueError(
                    f"Quantum spin-parity violation: For N_elec={n_electrons}, "
                    f"spin_multiplicity (2S+1) must be {expected_parity}, "
                    f"got {self.spin_multiplicity}."
                )

        return self


def format_validation_error_response(
    exc: Exception, job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Format structured HTTP 422 Unprocessable Entity response for intake rejections.

    Args:
        exc: Caught exception.
        job_id: Optional job identifier string.

    Returns:
        Structured response dictionary matching HTTP 422 standard format.
    """
    return {
        "status": 422,
        "error": "Unprocessable Entity",
        "job_id": job_id or "unknown",
        "detail": str(exc),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
