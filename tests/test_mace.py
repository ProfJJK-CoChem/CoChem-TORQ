import json
import logging
from pathlib import Path

import numpy as np

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


def generate_minimal_onnx_model_bytes() -> bytes:
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

    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="MACE-OFF24m")
    assert triage.symbols == ["H", "H"]
    assert len(triage.grid_points) == 2
    assert triage.batch_size in (16, 512)
    assert triage.model_name == "MACE-OFF24m"
    assert triage.scf_tolerance_guard == 1e-5
    assert triage.device in ["cpu", "cuda"]


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

    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="AIMNet2")
    assert triage.model_name == "AIMNet2"
    assert triage.scf_tolerance_guard == 1e-5


def test_extract_topographic_extrema() -> None:
    triage = TorqMACETriage.__new__(TorqMACETriage)
    triage.triage_results = [
        {
            "dihedral_angles": [0],
            "status": "converged",
            "relative_energy_kcal_mol": 0.0,
        },
        {
            "dihedral_angles": [30],
            "status": "converged",
            "relative_energy_kcal_mol": 5.2,
        },
        {
            "dihedral_angles": [60],
            "status": "converged",
            "relative_energy_kcal_mol": 1.1,
        },
    ]
    extrema = triage.extract_topographic_extrema()
    assert len(extrema) >= 1
    assert any(p["relative_energy_kcal_mol"] == 0.0 for p in extrema)


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
    # Empty inputs
    g_empty, c_empty = compute_pes_derivatives([], [])
    assert len(g_empty) == 0
    assert len(c_empty) == 0

    # Single point
    g_one, c_one = compute_pes_derivatives([45.0], [2.5])
    assert len(g_one) == 1
    assert g_one[0] == 0.0

    # Two points
    g_two, c_two = compute_pes_derivatives([0.0, 10.0], [0.0, 2.0])
    assert len(g_two) == 2
    assert abs(g_two[0] - 0.2) < 1e-5


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
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    energy_ev, forces, converged = evaluate_physical_potential(symbols, coords)
    assert isinstance(energy_ev, float)
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 3)
    assert converged is True


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

    refined_grid = generate_adaptive_grid(
        grid_points=grid_points,
        symbols=symbols,
        slope_threshold=0.01,
        max_angular_step=45.0,
        min_angular_step=5.0,
        refinement_subdivisions=2,
    )

    assert len(refined_grid) > len(grid_points)
    angles = [float(p["dihedral_angles"][0]) for p in refined_grid]
    assert angles == sorted(angles)
    assert all("gradient_kcal_mol_deg" in p for p in refined_grid)
    assert all("curvature_kcal_mol_deg2" in p for p in refined_grid)
    assert all("is_ts_candidate" in p for p in refined_grid)


def test_onnx_cpu_fallback_configuration() -> None:
    config = onnx_cpu_fallback()
    assert isinstance(config, dict)
    assert config["provider"] == "CPUExecutionProvider"
    assert "CPUExecutionProvider" in config["providers"]
    assert config["intra_op_num_threads"] >= 1
    assert config["inter_op_num_threads"] >= 1
    assert config["physical_cores"] >= 1
    assert config["session_options"] is not None


def test_onnx_cpu_fallback_execution_with_bytes() -> None:
    model_bytes = generate_minimal_onnx_model_bytes()
    session = onnx_cpu_fallback(model_path=model_bytes)
    assert session is not None
    assert "CPUExecutionProvider" in session.get_providers()

    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    output = session.run(None, {"X": input_data})
    assert len(output) == 1
    assert np.allclose(output[0], input_data)


def test_torq_mace_triage_adaptive_workflow(tmp_path: Path) -> None:
    grid_file = tmp_path / "torsion_scan.json"
    grid_data = {
        "symbols": ["C", "C", "H", "H"],
        "grid_points": [
            {
                "dihedral_angles": [0.0],
                "coordinates": [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [-0.5, 0.9, 0.0],
                    [2.0, 0.9, 0.0],
                ],
            },
            {
                "dihedral_angles": [90.0],
                "coordinates": [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [-0.5, 0.9, 0.0],
                    [2.0, 0.0, 0.9],
                ],
            },
            {
                "dihedral_angles": [180.0],
                "coordinates": [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [-0.5, 0.9, 0.0],
                    [2.0, -0.9, 0.0],
                ],
            },
        ],
    }
    grid_file.write_text(json.dumps(grid_data), encoding="utf-8")

    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="MACE-OFF24m")
    initial_results = triage.evaluate_grid()
    assert len(initial_results) == 3

    # Generate adaptive grid
    refined_results = triage.generate_adaptive_grid(
        slope_threshold=0.01,
        max_angular_step=60.0,
        min_angular_step=10.0,
        refinement_subdivisions=2,
    )
    assert len(refined_results) >= 4

    # Full triage execution with adaptive refinement
    summary = triage.run_triage(refine_adaptive=True)
    assert summary["num_grid_points"] >= 4
    assert "extrema" in summary
    assert isinstance(summary["extrema"], list)


def test_torq_mace_triage_onnx_method() -> None:
    triage = TorqMACETriage.__new__(TorqMACETriage)
    config = triage.onnx_cpu_fallback()
    assert isinstance(config, dict)
    assert config["provider"] == "CPUExecutionProvider"
    assert triage.onnx_config == config


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


def test_guard_g4_rank_inversion_audit() -> None:
    triage = TorqMACETriage.__new__(TorqMACETriage)
    triage.triage_results = [
        {"dihedral_angles": [0], "relative_energy_kcal_mol": 0.0},
        {"dihedral_angles": [30], "relative_energy_kcal_mol": 1.5},
        {"dihedral_angles": [60], "relative_energy_kcal_mol": 4.2},
        {"dihedral_angles": [90], "relative_energy_kcal_mol": 8.0},
    ]

    # Highly correlated reference energies
    ref_energies = [0.0, 1.4, 4.3, 8.1]
    audit = triage.audit_rank_inversion(ref_energies, spearman_threshold=0.9)
    assert audit["spearman_rho"] >= 0.95
    assert audit["cull_eligible"] is True
    assert audit["g4_status"] == "PASSED"
    assert audit["max_rank_displacement"] == 0
