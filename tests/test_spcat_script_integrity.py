"""Reject unavailable catalog physics before any output or invented deck is written."""

import pytest
from pydantic import ValidationError

from scripts.spcat_runner import (
    MethodologyViolationError,
    SPCATDeckConfig,
    SPCATRunner,
    compute_ray_asymmetry_parameter,
)


def test_spherical_ray_parameter_is_undefined():
    assert compute_ray_asymmetry_parameter(1.0, 1.0, 1.0) is None
    assert compute_ray_asymmetry_parameter(3.0, 1.0, 1.0) == -1.0
    with pytest.raises(ValueError):
        compute_ray_asymmetry_parameter(float("nan"), 1.0, 1.0)


def test_omitted_properties_cannot_be_zero_or_ground_state():
    with pytest.raises(ValidationError):
        SPCATDeckConfig(a_mhz=3.0, b_mhz=2.0, c_mhz=1.0)


def test_unqualified_catalog_writes_nothing(tmp_path):
    # Explicit algebraic model inputs test software gating, not a measured molecule.
    config = SPCATDeckConfig(
        a_mhz=3.0,
        b_mhz=2.0,
        c_mhz=1.0,
        dj_khz=0.0,
        djk_khz=0.0,
        dk_khz=0.0,
        d1_khz=0.0,
        d2_khz=0.0,
        mu_a=1.0,
        mu_b=0.0,
        mu_c=0.0,
        constant_type="Be",
        delta_b_vib_mhz=0.1,
    )
    with pytest.raises(MethodologyViolationError):
        SPCATRunner.validate_rotor_parameters(config)
    output = tmp_path / "catalog.parquet"
    scratch = tmp_path / "scratch"
    with pytest.raises(NotImplementedError, match="not qualified"):
        SPCATRunner().generate_parquet_catalog(
            config, output, allow_unvibrated_be=True, scratch_dir=scratch
        )
    assert not output.exists() and not scratch.exists()
    with pytest.raises(FileNotFoundError):
        SPCATRunner(tmp_path / "missing-native-binary")
