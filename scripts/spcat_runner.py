"""Legacy SPCAT entry point, gated pending a qualified deck/consumer implementation.

Use Libraries.cochem_torq_spcat.PickettSPCATRunner for actual externally prepared
native decks. The independently named teaching rigid rotor is available through
the student application; it never substitutes for an unsuccessful SPCAT run.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MethodologyViolationError(ValueError):
    """The requested observable is not supported by supplied evidence."""


class SPCATDeckConfig(BaseModel):
    """Explicit supplied properties; omitted quantities cannot imply zero."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    a_mhz: float = Field(gt=0.0)
    b_mhz: float = Field(gt=0.0)
    c_mhz: float = Field(gt=0.0)
    dj_khz: float
    djk_khz: float
    dk_khz: float
    d1_khz: float
    d2_khz: float
    mu_a: float
    mu_b: float
    mu_c: float
    temperature_k: float = Field(default=298.15, gt=0.0)
    constant_type: Literal["B0", "Be"]
    delta_b_vib_mhz: float | None = None

    @model_validator(mode="after")
    def ordered_constants(self):
        if not self.a_mhz >= self.b_mhz >= self.c_mhz:
            raise ValueError("Principal-axis constants require A >= B >= C.")
        return self


def compute_ray_asymmetry_parameter(a: float, b: float, c: float) -> float | None:
    """Ray kappa; spherical-top degeneracy has no defined Ray parameter."""
    if not all(math.isfinite(value) and value > 0 for value in (a, b, c)):
        raise ValueError("Finite positive rotational constants are required.")
    if not a >= b >= c:
        raise ValueError("Principal-axis constants require A >= B >= C.")
    if a == c:
        return None
    return (2.0 * b - a - c) / (a - c)


class SPCATRunner:
    """Compatibility facade retaining explicit unavailable scientific behavior."""

    def __init__(self, spcat_bin_path: Path | str | None = None) -> None:
        if spcat_bin_path is not None:
            selected = Path(spcat_bin_path)
            if not selected.is_file():
                raise FileNotFoundError(
                    "Explicitly selected SPCAT binary is unavailable."
                )
            self.spcat_bin = selected.resolve()
        else:
            found = shutil.which("spcat") or shutil.which("spcat.exe")
            self.spcat_bin = Path(found).resolve() if found else None

    @classmethod
    def validate_rotor_parameters(
        cls, config: SPCATDeckConfig, allow_unvibrated_be: bool = False
    ) -> None:
        compute_ray_asymmetry_parameter(config.a_mhz, config.b_mhz, config.c_mhz)
        if config.constant_type == "Be" and not allow_unvibrated_be:
            raise MethodologyViolationError(
                "Equilibrium constants cannot be relabeled B0. A supplied scalar "
                "correction does not establish all three axis-specific corrections."
            )

    def generate_parquet_catalog(
        self,
        config: SPCATDeckConfig,
        output_parquet_path: Path | str,
        allow_unvibrated_be: bool = False,
        scratch_dir: Path | str | None = None,
    ) -> Path:
        """Reject the unqualified historical automatic deck/catalog conversion."""
        self.validate_rotor_parameters(config, allow_unvibrated_be)
        raise NotImplementedError(
            "Automatic SPCAT decks/catalogs are not qualified: native reduction, "
            "representation, parameter uncertainties, partition functions, state "
            "assignment and actual consumer round trips are required. Use "
            "PickettSPCATRunner with independently prepared native decks. No "
            "replacement solver or catalog is generated."
        )
