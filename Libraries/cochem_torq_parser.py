"""CoChem-TORQ Output Parser for Quantum Electronic Structure Calculations.

Compliant with Method Matrix v4 §8B.3 and Anti-Spoofing Directives.
"""

from __future__ import annotations

import re
from typing import Final, List, Optional, Tuple
import numpy as np

# Authoritative regex pattern matching <S^2> with arbitrary spacing and exponent syntax (** or ^)
S2_PATTERN: Final[re.Pattern] = re.compile(
    r"<\s*S\s*(\*\*|\^)\s*2\s*>[^:\r\n]*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE
)



def parse_spin_contamination_s2(output_text: str) -> Optional[float]:
    r"""Parse <S^2> spin expectation value from ORCA output using robust regex.

    Handles whitespace variations, exponential notation, and both power forms (** and ^).
    Uses S2_PATTERN: r"<\s*S\s*(\*\*|\^)\s*2\s*>\s*:\s*([\d\.]+)"

    Parameters
    ----------
    output_text : str
        Standard output or log content from an electronic structure run.

    Returns
    -------
    Optional[float]
        Parsed <S^2> expectation value if discovered, None otherwise.
    """
    match = S2_PATTERN.search(output_text)
    if match:
        try:
            return float(match.group(2))
        except ValueError:
            return None
    return None


def parse_orca_energy_and_gradient(
    output_text: str, n_atoms: Optional[int] = None
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Parse final single point electronic energy and nuclear Cartesian gradient from ORCA output."""
    energy: Optional[float] = None
    e_match = re.search(
        r"(?:FINAL SINGLE POINT ENERGY|TOTAL ENERGY)\s+(-?\d+\.\d+)", output_text
    )
    if e_match:
        try:
            energy = float(e_match.group(1))
        except ValueError:
            pass

    gradient: Optional[np.ndarray] = None
    grad_match = re.search(
        r"CARTESIAN GRADIENT.*?\n\n(.*?)(?=\n\n|\n[A-Z]|\Z)", output_text, re.DOTALL
    )
    if grad_match:
        parsed_grad = []
        for line in grad_match.group(1).strip().splitlines():
            parts = line.split()
            if len(parts) >= 6 and not line.startswith("-"):
                try:
                    parsed_grad.append(
                        [float(parts[3]), float(parts[4]), float(parts[5])]
                    )
                except ValueError:
                    pass
        if parsed_grad:
            if n_atoms is None or len(parsed_grad) == n_atoms:
                gradient = np.array(parsed_grad, dtype=np.float64)

    return energy, gradient


__all__ = [
    "S2_PATTERN",
    "parse_spin_contamination_s2",
    "parse_orca_energy_and_gradient",
]
