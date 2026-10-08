"""Actual hardware queries and Torch layers with declared mathematical inputs.

No GPU, memory provider, trained scientific model, or inference result is mocked.
The numerical layers below verify execution/aggregation, not QM qualification.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import psutil
import pytest
import torch

from Libraries.cochem_torq_committee_ensemble import (
    CommitteeEnsemble,
    CommitteeEnsembleConfig,
    DeviceMemoryTelemetry,
    EnsembleConsensusError,
    compute_committee_moments,
    get_available_vram_mb,
    get_device_memory_telemetry,
)


def test_real_cpu_telemetry_identifies_system_memory_provider():
    result = get_device_memory_telemetry("cpu")
    assert result.device == "cpu"
    assert result.memory_kind == "system_ram"
    assert result.query_provider == "psutil.virtual_memory"
    assert 0 <= result.available_mb <= psutil.virtual_memory().total / 1024**2


def test_real_cuda_query_or_explicit_observed_unavailability():
    if torch.cuda.is_available():
        result = get_device_memory_telemetry("cuda")
        assert result.memory_kind == "gpu_vram"
        assert result.query_provider == "torch.cuda.mem_get_info"
        assert result.available_mb >= 0
        assert get_available_vram_mb() >= 0
    else:
        with pytest.raises(EnsembleConsensusError, match="no CUDA device"):
            get_available_vram_mb()
        with pytest.raises(EnsembleConsensusError, match="no CUDA device"):
            get_device_memory_telemetry("cuda")


def test_invalid_actual_cuda_index_never_substitutes_system_ram():
    # A device index beyond the actual device count is genuinely unavailable.
    invalid_index = torch.cuda.device_count() + 100
    with pytest.raises(EnsembleConsensusError, match="unavailable"):
        get_device_memory_telemetry(f"cuda:{invalid_index}")


def test_unsupported_memory_profile_is_explicit():
    with pytest.raises(EnsembleConsensusError, match="unsupported"):
        get_device_memory_telemetry("meta")


@pytest.mark.parametrize(
    "value", [None, -1.0, float("nan"), float("inf"), True, 1 + 2j]
)
def test_invalid_declared_capacity_is_rejected(value):
    with pytest.raises(EnsembleConsensusError, match="finite and nonnegative"):
        DeviceMemoryTelemetry("cpu", "system_ram", "psutil.virtual_memory", value)


def test_gpu_record_cannot_describe_system_ram():
    with pytest.raises(EnsembleConsensusError, match="mismatched"):
        DeviceMemoryTelemetry("cuda", "system_ram", "psutil.virtual_memory", 0.0)


def mathematical_models():
    models = [torch.nn.Linear(3, 1, bias=False).double() for _ in range(2)]
    with torch.no_grad():
        models[0].weight.fill_(1.0)
        models[1].weight.fill_(2.0)
    return models


def test_actual_cpu_layers_retain_inference_and_label_cpu_capacity():
    inputs = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
    ensemble = CommitteeEnsemble(mathematical_models())
    result = ensemble(inputs)
    assert torch.allclose(result.mean, torch.full((2, 1), 4.5, dtype=torch.float64))
    assert torch.allclose(result.variance, torch.full((2, 1), 4.5, dtype=torch.float64))
    gradient = torch.autograd.grad(result.mean.sum(), inputs)[0]
    assert torch.allclose(gradient, torch.full_like(inputs, 1.5))
    assert ensemble.last_memory_telemetry.memory_kind == "system_ram"
    assert ensemble.last_memory_telemetry.device == "cpu"
    serial = CommitteeEnsemble(
        mathematical_models(), CommitteeEnsembleConfig(vectorized=False)
    )
    assert torch.equal(serial(inputs).mean, result.mean)
    assert np.isfinite(ensemble.last_memory_telemetry.available_mb)


def test_cpu_layers_cannot_claim_cuda_stream_execution():
    ensemble = CommitteeEnsemble(
        mathematical_models(), CommitteeEnsembleConfig(concurrency_mode="cuda_streams")
    )
    with pytest.raises(EnsembleConsensusError, match="CUDA execution profile"):
        ensemble(torch.ones((1, 3), dtype=torch.float64))
    assert ensemble.last_memory_telemetry is None


def test_actual_meta_tensors_cannot_use_cpu_memory_query():
    models = [torch.nn.Linear(3, 1, bias=False, device="meta") for _ in range(2)]
    ensemble = CommitteeEnsemble(models)
    with pytest.raises(EnsembleConsensusError, match="unsupported"):
        ensemble(torch.ones((1, 3), device="meta"))
    assert ensemble.last_memory_telemetry is None


def test_actual_mixed_devices_cannot_share_a_capacity_profile():
    models = [torch.nn.Linear(3, 1, device="cpu"), torch.nn.Linear(3, 1, device="meta")]
    ensemble = CommitteeEnsemble(models)
    with pytest.raises(EnsembleConsensusError, match="Mixed device profiles"):
        ensemble(torch.ones((1, 3)))
    assert ensemble.last_memory_telemetry is None


def test_declared_mathematical_sample_statistics_and_immutable_labels():
    result = compute_committee_moments(
        torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64),
        torch.tensor(
            [[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]],
            dtype=torch.float64,
        ),
    )
    assert result.mean.item() == 3.0
    assert result.variance.item() == 4.0
    assert result.std.item() == 2.0
    assert result.per_atom_force_variance.item() == pytest.approx(4.0 / 3.0)
    assert result.max_force_std == 2.0
    assert result.uncertainty_kind == "ensemble_disagreement"
    assert result.calibration_status == "not_established"
    with pytest.raises(FrozenInstanceError):
        result.calibration_status = "established"


@pytest.mark.parametrize(
    "energies",
    [
        torch.tensor([1.0, float("nan")], dtype=torch.float64),
        torch.tensor([1.0, float("inf")], dtype=torch.float64),
        torch.tensor([1.0, -float("inf")], dtype=torch.float64),
        torch.tensor([1 + 0j, 2 + 1j], dtype=torch.complex128),
    ],
)
def test_invalid_actual_energy_tensors_cannot_produce_physical_variance(energies):
    with pytest.raises(EnsembleConsensusError, match="finite real"):
        compute_committee_moments(energies)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 2 + 1j])
def test_invalid_actual_force_tensors_cannot_produce_physical_variance(value):
    dtype = torch.complex128 if isinstance(value, complex) else torch.float64
    forces = torch.tensor([[[0, 0, 0]], [[value, 0, 0]]], dtype=dtype)
    with pytest.raises(EnsembleConsensusError, match="finite real"):
        compute_committee_moments(torch.tensor([1.0, 2.0], dtype=torch.float64), forces)


@pytest.mark.parametrize("values", [[1e308, 1e308], [-1e308, 1e308]])
def test_real_energy_arithmetic_overflow_is_an_explicit_failure(values):
    energies = torch.tensor(values, dtype=torch.float64)
    assert torch.isfinite(energies).all()
    with pytest.raises(EnsembleConsensusError, match="Computed energy"):
        compute_committee_moments(energies)


def test_real_force_arithmetic_overflow_is_an_explicit_failure():
    forces = torch.tensor([[[-1e308, 0, 0]], [[1e308, 0, 0]]], dtype=torch.float64)
    assert torch.isfinite(forces).all()
    with pytest.raises(EnsembleConsensusError, match="Computed.*force dispersion"):
        compute_committee_moments(torch.tensor([1.0, 2.0], dtype=torch.float64), forces)


def test_sample_statistics_preserve_actual_torch_autograd():
    energies = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    result = compute_committee_moments(energies)
    assert torch.equal(
        torch.autograd.grad(result.variance, energies)[0],
        torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64),
    )
