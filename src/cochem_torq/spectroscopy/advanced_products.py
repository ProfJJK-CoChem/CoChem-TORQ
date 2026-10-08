"""Typed workflow products retaining the exact nonresonant Watson model limits."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator
from typing_extensions import Self

from .results import ScientificContext
from .rovibrational_perturbation import (
    WatsonHarmonicDistortion,
    WatsonVibrationRotationResult,
)


class ModelGroundStateConstants(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    scientific_context: ScientificContext
    constants_mhz: tuple[StrictFloat, StrictFloat, StrictFloat]
    observable: Literal["nonresonant_Watson_model_B0"] = "nonresonant_Watson_model_B0"
    watson_result_artifact_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    watson_result: WatsonVibrationRotationResult
    independent_scientific_qualification: Literal[False] = False
    identification_ready: Literal[False] = False

    @model_validator(mode="after")
    def source_and_applicability(self) -> Self:
        if (
            self.watson_result.ground_state_constants_mhz is None
            or self.constants_mhz != self.watson_result.ground_state_constants_mhz
            or self.watson_result_artifact_sha256
            not in self.scientific_context.parent_artifact_sha256
        ):
            raise ValueError(
                "Model B0 requires actual gated Watson results and artifact."
            )
        first = self.scientific_context.model_dump(exclude={"parent_artifact_sha256"})
        second = self.watson_result.scientific_context.model_dump(
            exclude={"parent_artifact_sha256"}
        )
        if first != second:
            raise ValueError(
                "Model B0 changed the original atom/isotope/mode/frame identity."
            )
        return self


class UnreducedHarmonicDistortion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    scientific_context: ScientificContext
    distortion: WatsonHarmonicDistortion
    watson_result_artifact_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    convention: Literal["unreduced_harmonic_Watson_operator"] = (
        "unreduced_harmonic_Watson_operator"
    )
    identification_ready: Literal[False] = False

    @model_validator(mode="after")
    def artifact_binding(self) -> Self:
        if self.watson_result_artifact_sha256 not in (
            self.scientific_context.parent_artifact_sha256
        ):
            raise ValueError("Distortion requires its actual retained Watson artifact.")
        return self
