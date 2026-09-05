"""CoChem-TORQ: Input Deck Compiler and Sanitizer.

Compliant with Method Matrix v4 §8B.3 and §9A.5 (Absolute Ban on Calc_Hess true).
Automates model Hessian substitution (InHess XTB2 or InHess Lindh).
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

logger = logging.getLogger("TorqCompiler")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Compiler] %(message)s")


class TorqDeckSanitizer:
    """AST-level and token-based parser for ORCA input decks.

    Enforces Method Matrix v4 rules:
    - Absolute ban on Calc_Hess true during geometry optimizations.
    - Automatic preconditioning with InHess XTB2 (or InHess Lindh).
    """

    @classmethod
    def sanitize_deck(cls, deck_content: str, xtb_available: bool = True) -> Tuple[str, bool]:
        """Sanitize an ORCA input deck to strictly enforce Method Matrix rules.

        If '! Opt' or an optimization directive is active, unconditionally flags and
        excises 'Calc_Hess true', substituting model Hessian preconditioning
        ('InHess XTB2' or 'InHess Lindh').

        Parameters
        ----------
        deck_content : str
            Raw ORCA input deck text.
        xtb_available : bool, optional
            Whether xTB is available for InHess XTB2 (defaults to True).

        Returns
        -------
        Tuple[str, bool]
            (Sanitized deck content, Boolean indicating whether Calc_Hess true was excised).
        """
        lines = deck_content.splitlines()
        is_opt = any(
            re.search(r"!\s*.*\bOpt\b", line, re.IGNORECASE)
            for line in lines
            if line.strip().startswith("!")
        ) or any("opt" in line.lower() for line in lines if line.strip().startswith("!"))

        calc_hess_regex = re.compile(r"\bcalc_hess\s*(=|\s+)\s*true\b", re.IGNORECASE)
        excised = False
        new_lines: list[str] = []
        has_inhess = any(re.search(r"\bInHess\b", line, re.IGNORECASE) for line in lines)
        substitute_hess = "InHess XTB2" if xtb_available else "InHess Lindh"

        for line in lines:
            stripped = line.strip()
            if calc_hess_regex.search(stripped):
                excised = True
                warning_msg = (
                    "[METHOD_MATRIX_VIOLATION] Calc_Hess true is strictly prohibited for geometry "
                    f"optimizations (§8B.3). Automatically substituted {substitute_hess}."
                )
                logger.warning(warning_msg)
                try:
                    from cochem_base.telemetry import emit_telemetry_warning
                    emit_telemetry_warning(warning_msg)
                except Exception:
                    pass
                # Omit this line
                continue

            new_lines.append(line)

        if excised:
            if not has_inhess:
                has_geom = False
                final_lines: list[str] = []
                for line in new_lines:
                    final_lines.append(line)
                    if line.strip().lower().startswith("%geom"):
                        final_lines.append(f"  {substitute_hess}")
                        has_geom = True

                if not has_geom:
                    insert_idx = 1 if len(final_lines) > 1 else 0
                    final_lines.insert(insert_idx, f"%geom\n  {substitute_hess}\nend")

                return "\n".join(final_lines) + ("\n" if deck_content.endswith("\n") else ""), True

        return "\n".join(new_lines) + ("\n" if deck_content.endswith("\n") else ""), excised


def sanitize_orca_deck(deck_content: str, xtb_available: bool = True) -> Tuple[str, bool]:
    """Convenience wrapper around TorqDeckSanitizer.sanitize_deck."""
    return TorqDeckSanitizer.sanitize_deck(deck_content, xtb_available=xtb_available)


__all__ = ["TorqDeckSanitizer", "sanitize_orca_deck"]
