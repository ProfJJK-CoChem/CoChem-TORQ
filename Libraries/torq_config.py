from pydantic import BaseModel, field_validator, model_validator

class TorqRunParams(BaseModel):
    tier: str
    wall_time_tier: str
    engine: str
    method: str
    basis_set: str
    keywords: list[str]
    anharmonicity: str | None = None
    dispersion: str | None = None
    bsse_correction: str | None = None
    cabs_mappings: dict[str, str] | None = None

    @field_validator('keywords', mode='after')
    @classmethod
    def check_calc_hess(cls, v: list[str]) -> list[str]:
        for kw in v:
            if "calc_hess" in kw.lower() and "true" in kw.lower():
                raise ValueError("Calc_Hess true is strictly prohibited for initial hessians. Use 'InHess XTB2' or 'Lindh' instead.")
        return v

    @model_validator(mode='after')
    def check_bsse_counterpoise(self):
        if self.bsse_correction and self.bsse_correction.lower() in ["counterpoise", "cp"]:
            basis_lower = self.basis_set.lower()
            if "aug" in basis_lower:
                raise ValueError("Counterpoise correction must be restricted to non-augmented triple zeta basis sets. Augmented basis sets are not allowed.")
            if "tz" not in basis_lower and "triple" not in basis_lower:
                raise ValueError("Counterpoise correction must be restricted to non-augmented triple zeta basis sets. The basis set does not appear to be triple zeta.")
        return self
