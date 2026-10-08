"""Actual calculator availability, declared EMT computation, and statistical mathematics.

EMT here is requested explicitly as a computational example. It is not MACE,
AIMNet2, a quantum surrogate qualification, or a student accuracy profile. The
ONNX identity graph tests runtime serialization only, never molecular physics.
"""

import json
import logging
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_torq_mace import (
    FLOAT32_NOISE_FLOOR_EH,
    FLOAT32_NOISE_FLOOR_EV,
    FLOAT32_NOISE_FLOOR_KCAL_MOL,
    TorqMACETriage,
    compute_pes_derivatives,
    evaluate_physical_potential,
    generate_adaptive_grid,
    get_covalent_radius,
    interpolate_coordinates,
    onnx_cpu_fallback,
)

logger = logging.getLogger(__name__)


def generate_identity_onnx_model_bytes() -> bytes:
    """
    Constructs a minimal valid ONNX ModelProto byte buffer directly
    via protobuf encoding for verifying ONNX CPU execution provider sessions.
    """

    def varint(n: int) -> bytes:
        res = bytearray()
        while n >= 0x80:
            res.append((n & 0x7F) | 0x80)
            n >>= 7
        res.append(n & 0x7F)
        return bytes(res)

    def field_bytes(num: int, b: bytes) -> bytes:
        return varint((num << 3) | 2) + varint(len(b)) + b

    def field_str(num: int, s: str) -> bytes:
        return field_bytes(num, s.encode("utf-8"))

    def field_varint(num: int, v: int) -> bytes:
        return varint((num << 3) | 0) + varint(v)

    def make_vi(name: str, elem_type: int = 1, dims: list[int] = [1, 3]) -> bytes:
        shape = field_bytes(
            1, b"".join(field_bytes(1, field_varint(1, d)) for d in dims)
        )
        tensor_type = field_varint(1, elem_type) + shape
        t = field_bytes(1, tensor_type)
        return field_str(1, name) + field_bytes(2, t)

    node = (
        field_str(1, "X")
        + field_str(2, "Y")
        + field_str(3, "identity_node")
        + field_str(4, "Identity")
    )
    graph = (
        field_bytes(1, node)
        + field_str(2, "identity_graph")
        + field_bytes(11, make_vi("X"))
        + field_bytes(12, make_vi("Y"))
    )
    opset = field_str(1, "") + field_varint(2, 14)
    model = (
        field_varint(1, 8)
        + field_str(2, "cochem_onnx")
        + field_bytes(7, graph)
        + field_bytes(8, opset)
    )
    return model


def test_mace_triage_init(tmp_path: Path) -> None:
    grid_file = tmp_path / "torq_grid.json"
    grid_data = {
        "symbols": ["H", "H"],
        "grid_points": [
            {
                "dihedral_angles": [0],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            },
            {
                "dihedral_angles": [30],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]],
            },
        ],
    }
    grid_file.write_text(json.dumps(grid_data), encoding="utf-8")

    if find_spec("mace") is not None:
        pytest.skip(
            "A pinned, independently verified MACE weight artifact is required before this legacy positive path can be qualified."
        )
    with pytest.raises(RuntimeError, match="Requested calculator.*unavailable"):
        TorqMACETriage(grid_filepath=str(grid_file), model_name="MACE-OFF24m")


def test_aimnet2_triage_init(tmp_path: Path) -> None:
    grid_file = tmp_path / "torq_grid.json"
    grid_data = {
        "symbols": ["H", "H"],
        "grid_points": [
            {
                "dihedral_angles": [0],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            }
        ],
    }
    grid_file.write_text(json.dumps(grid_data), encoding="utf-8")

    if find_spec("aimnet2calc") is not None:
        pytest.skip(
            "A pinned, independently verified AIMNet2 weight artifact is required before this legacy positive path can be qualified."
        )
    with pytest.raises(RuntimeError, match="Requested calculator.*unavailable"):
        TorqMACETriage(grid_filepath=str(grid_file), model_name="AIMNet2")


def _actual_emt_triage(tmp_path: Path) -> TorqMACETriage:
    """Select the genuine ASE empirical EMT calculator, retaining its own identity."""
    pytest.importorskip("ase")
    source = tmp_path / "declared-emt-inputs.json"
    source.write_text(
        json.dumps(
            {
                "symbols": ["H", "H"],
                "grid_points": [
                    {
                        "dihedral_angles": [index * 30.0],
                        "coordinates": [[0, 0, 0], [0, 0, distance]],
                    }
                    for index, distance in enumerate((0.74, 1.2, 0.8, 1.0))
                ],
            }
        )
    )
    return TorqMACETriage(grid_filepath=str(source), model_name="EMT")


def test_extract_topographic_extrema(tmp_path: Path) -> None:
    triage = _actual_emt_triage(tmp_path)
    observed = triage.evaluate_grid()
    suggestions = triage.extract_topographic_extrema()
    assert len(suggestions) == len(observed)
    assert all(
        item["advisory_only"] and not item["eligible_for_pruning"]
        for item in suggestions
    )
    assert [item["raw_energy_ev"] for item in suggestions] == [
        item["raw_energy_ev"] for item in observed
    ]


def test_compute_pes_derivatives_uniform() -> None:
    angles = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    energies = 5.0 * (1.0 - np.cos(np.radians(angles)))

    gradients, curvatures = compute_pes_derivatives(angles, energies)
    assert len(gradients) == len(angles)
    assert len(curvatures) == len(angles)

    # At 0 deg: dE/dθ ≈ 0, d²E/dθ² > 0 (minimum)
    assert abs(gradients[0]) < 1e-2
    assert curvatures[0] > 0.0

    # At 90 deg: dE/dθ > 0 (maximum slope)
    assert gradients[3] > 0.0


def test_compute_pes_derivatives_edge_cases() -> None:
    # Missing or insufficient samples cannot establish a curvature or zero force.
    for angles, energies in (([], []), ([45.0], [2.5]), ([0.0, 10.0], [0.0, 2.0])):
        with pytest.raises(ValueError, match="At least three"):
            compute_pes_derivatives(angles, energies)


def test_interpolate_coordinates_linear() -> None:
    c1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    c2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)

    interp_mid = interpolate_coordinates(c1, c2, fraction=0.5)
    assert np.allclose(interp_mid[1], [0.0, 0.0, 1.5])


def test_interpolate_coordinates_rodrigues_rotation() -> None:
    # 4 atoms: H - C - C - H defining a dihedral
    c1 = np.array(
        [
            [-1.0, 1.0, 0.0],  # H1 (atom 0)
            [0.0, 0.0, 0.0],  # C1 (atom 1, pivot)
            [1.5, 0.0, 0.0],  # C2 (atom 2, axis along x)
            [2.5, 1.0, 0.0],  # H2 (atom 3, rotates)
        ],
        dtype=np.float64,
    )
    c2 = c1.copy()

    # Rotate by 90 degrees around C1-C2 bond axis
    interp_rot = interpolate_coordinates(
        c1,
        c2,
        fraction=1.0,
        dihedral_indices=(0, 1, 2, 3),
        delta_angle_deg=90.0,
        moving_atom_indices=[3],
    )

    # Atom 3 (H2) rotated 90 deg around X-axis: y=0, z=1
    assert abs(interp_rot[3, 0] - 2.5) < 1e-4
    assert abs(interp_rot[3, 1] - 0.0) < 1e-4
    assert abs(interp_rot[3, 2] - 1.0) < 1e-4


def test_evaluate_physical_potential() -> None:
    symbols = ["H", "H"]
    coordinates = [[0, 0, 0], [0, 0, 0.74]]
    with pytest.raises(RuntimeError, match="No potential calculator selected"):
        evaluate_physical_potential(symbols, coordinates)
    pytest.importorskip("pyscf")
    from pyscf import lib

    previous = lib.num_threads()
    try:
        lib.num_threads(1)
        energy, forces, converged = evaluate_physical_potential(
            symbols, coordinates, use_pyscf=True
        )
    finally:
        lib.num_threads(previous)
    assert np.isfinite(energy) and np.isfinite(forces).all()
    assert forces.shape == (2, 3)
    assert converged is True
    assert (
        np.linalg.norm(forces[0]) > 0
    )  # The declared input is not assumed stationary.


def test_generate_adaptive_grid_standalone() -> None:
    symbols = ["H", "H"]
    grid_points = [
        {
            "dihedral_angles": [0.0],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        },
        {
            "dihedral_angles": [60.0],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.20]],
        },
        {
            "dihedral_angles": [120.0],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]],
        },
    ]

    with pytest.raises(RuntimeError, match="No potential calculator selected"):
        generate_adaptive_grid(
            grid_points=grid_points,
            symbols=symbols,
            slope_threshold=0.01,
            max_angular_step=45.0,
            min_angular_step=5.0,
            refinement_subdivisions=2,
        )


def test_onnx_cpu_fallback_configuration() -> None:
    pytest.importorskip("onnxruntime")
    config = onnx_cpu_fallback()
    assert isinstance(config, dict)
    assert config["provider"] == "CPUExecutionProvider"
    assert "CPUExecutionProvider" in config["providers"]
    assert config["intra_op_num_threads"] >= 1
    assert config["inter_op_num_threads"] >= 1
    assert config["physical_cores"] >= 1
    assert config["session_options"] is not None


def test_onnx_cpu_fallback_execution_with_bytes() -> None:
    pytest.importorskip("onnxruntime")
    model_bytes = generate_identity_onnx_model_bytes()
    session = onnx_cpu_fallback(model_path=model_bytes)
    assert session is not None
    assert "CPUExecutionProvider" in session.get_providers()

    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    output = session.run(None, {"X": input_data})
    assert len(output) == 1
    assert np.allclose(output[0], input_data)


def test_torq_mace_triage_adaptive_workflow(tmp_path: Path) -> None:
    """No implicit learned potential or fallback is selected for a new workflow."""
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"symbols": ["H", "H"], "grid_points": []}))
    with pytest.raises(RuntimeError, match="ONNX.*not implemented"):
        TorqMACETriage(grid_filepath=str(source), model_name="unqualified.onnx")


def test_torq_mace_triage_onnx_method(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    triage = _actual_emt_triage(tmp_path)
    configuration = triage.onnx_cpu_fallback()
    assert configuration["provider"] == "CPUExecutionProvider"
    assert triage.onnx_config == configuration
    assert (
        triage.model_name == "EMT"
    )  # Configuring a runtime does not select a different potential.


def test_mendeleev_covalent_radius() -> None:
    r_c = get_covalent_radius("C")
    r_h = get_covalent_radius("H")
    r_o = get_covalent_radius("O")
    assert 0.70 <= r_c <= 0.80
    assert 0.30 <= r_h <= 0.35
    assert 0.60 <= r_o <= 0.70


def test_float32_noise_floor_constants() -> None:
    assert FLOAT32_NOISE_FLOOR_EH == 4.0e-6
    assert FLOAT32_NOISE_FLOOR_EV > 0.0
    assert FLOAT32_NOISE_FLOOR_KCAL_MOL > 0.0


def test_guard_g4_rank_inversion_audit(tmp_path: Path) -> None:
    """Check actual SciPy arithmetic on observed values, not an independent benchmark."""
    triage = _actual_emt_triage(tmp_path)
    values = triage.evaluate_grid()
    energies = [item["relative_energy_kcal_mol"] for item in values]
    audit = triage.audit_rank_inversion(energies, spearman_threshold=0.9)
    assert audit["spearman_rho"] == pytest.approx(1.0)
    assert audit["cull_eligible"] is False
    assert audit["advisory_only"] is True
    assert audit["eligible_for_pruning"] is False
    assert audit["g4_status"] == "DIAGNOSTIC_PASSED_ADVISORY_ONLY"
    assert audit["max_rank_displacement"] == 0
    assert len(triage.triage_results) == len(values)


def test_rank_statistics_reject_undefined_or_missing_evidence(tmp_path: Path) -> None:
    triage = _actual_emt_triage(tmp_path)
    triage.evaluate_grid()
    with pytest.raises(ValueError, match="constant energy series"):
        triage.audit_rank_inversion([1.0] * 4)
    with pytest.raises(ValueError, match="finite observed energies"):
        triage.audit_rank_inversion([0.0, 1.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="Mismatch"):
        triage.audit_rank_inversion([0.0, 1.0])
    triage.grid_points = triage.grid_points[:2]
    observed = triage.evaluate_grid()
    with pytest.raises(ValueError, match="at least three"):
        triage.audit_rank_inversion(
            [item["relative_energy_kcal_mol"] for item in observed]
        )


@pytest.mark.parametrize("symbol", ["H", "O", "Fe", "Og"])
def test_mace_and_quench_share_the_actual_named_radius_definition(symbol):
    from mendeleev import element

    from Libraries.cochem_torq_mace import get_covalent_radius_record
    from Libraries.cochem_torq_quench import get_covalent_radius_record as quench_record

    actual = element(symbol)
    record = get_covalent_radius_record(symbol)
    assert record == quench_record(symbol)
    assert record["property"] == "covalent_radius_pyykko"
    assert get_covalent_radius(symbol) == float(actual.covalent_radius_pyykko) / 100.0
    assert len(record["source"]["database_sha256"]) == 64


def test_mace_does_not_guess_a_radius_for_unknown_elements():
    with pytest.raises(ValueError):
        get_covalent_radius("UnknownElement")
