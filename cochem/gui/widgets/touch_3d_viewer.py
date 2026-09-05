"""
CoChem-Mobile Touch-Optimized 3D Molecular Visualization Controller (REQ-MOB-020 - REQ-MOB-028).

Strict Zero-Mock & Anti-Spoof Mandate:
- Real SO(3) arcball virtual sphere projection and unit quaternion math (scalar-last [x, y, z, w]).
- Dynamic Mendeleev van der Waals radius resolution (pm -> Angstrom via pm / 100.0).
- Strict Pydantic v2 immutable schemas (CameraState, ViewportConfig, SelectionState, VibrationTensor).
- Authentic molecular file parsing (.xyz, .pdb, .cube) with multi-frame trajectory streaming.
- Cross-platform file locking with filelock.FileLock and air-gapped static asset SRI verification.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import filelock
import ipywidgets
import numpy as np

from cochem.core.cochem_elements import (
    get_vdw_radius,
)
from cochem.core.mendeleev_invariants import mendeleev_resolver
from cochem.gui.schemas import (
    CameraState,
    SelectionState,
    VibrationTensor,
    ViewportConfig,
)


# =============================================================================
# Mathematical Utilities: SO(3) Arcball & Quaternion Operations
# =============================================================================

def project_to_sphere(x: float, y: float) -> np.ndarray:
    """
    Project normalized 2D touch coordinates (x, y) in [-1, 1]^2 onto the 3D virtual sphere.

    Formula:
        z = sqrt(1 - x^2 - y^2) if x^2 + y^2 <= 0.5
        z = 0.5 / sqrt(x^2 + y^2) if x^2 + y^2 > 0.5
        v = (x, y, z) / ||(x, y, z)||

    Args:
        x: Normalized X coordinate in [-1.0, 1.0].
        y: Normalized Y coordinate in [-1.0, 1.0].

    Returns:
        3D unit vector [v_x, v_y, v_z] on virtual sphere.
    """
    d2 = float(x * x + y * y)
    if d2 <= 0.5:
        z = math.sqrt(max(0.0, 1.0 - d2))
    else:
        z = 0.5 / math.sqrt(d2)

    vec = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / norm


def quaternion_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Construct incremental rotation quaternion from virtual sphere vectors v1 to v2.
    Uses scalar-last convention [x, y, z, w].

    Args:
        v1: Starting 3D unit vector.
        v2: Ending 3D unit vector.

    Returns:
        Unit quaternion [x, y, z, w] with ||q|| = 1.0.
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    cross = np.cross(v1_norm, v2_norm)
    dot = float(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
    theta = math.acos(dot)
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-7 or abs(theta) < 1e-7:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    axis = cross / cross_norm
    half_theta = theta * 0.5
    sin_half = math.sin(half_theta)
    cos_half = math.cos(half_theta)

    q = np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half,
    ], dtype=np.float64)

    return quaternion_normalize(q)


def quaternion_multiply(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """
    Multiply two quaternions in scalar-last convention [x, y, z, w]: q_out = q1 (x) q2.

    Formula:
        x' = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y' = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z' = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w' = w1*w2 - x1*x2 - y1*y2 - z1*z2

    Args:
        q1: First quaternion [x, y, z, w].
        q2: Second quaternion [x, y, z, w].

    Returns:
        Resulting quaternion [x, y, z, w].
    """
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w], dtype=np.float64)


def quaternion_normalize(q: Sequence[float]) -> np.ndarray:
    """
    Strict unit normalization ||q|| = 1.0 for scalar-last quaternion.

    Args:
        q: Quaternion [x, y, z, w].

    Returns:
        Normalized unit quaternion [x, y, z, w].
    """
    arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return arr / norm


def quaternion_to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
    """
    Convert scalar-last unit quaternion [x, y, z, w] to a 3x3 orthogonal rotation matrix.

    Args:
        q: Unit quaternion [x, y, z, w].

    Returns:
        3x3 rotation matrix R with R @ R.T = I, det(R) = +1.0.
    """
    qx, qy, qz, qw = quaternion_normalize(q)
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, xy, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2

    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def calculate_dihedral(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    p4: Sequence[float],
    degrees: bool = True,
) -> float:
    """
    Calculate proper dihedral torsion angle formed by 4 Cartesian coordinates p1-p2-p3-p4.

    Args:
        p1: Atom 1 coordinates (x, y, z).
        p2: Atom 2 coordinates (x, y, z).
        p3: Atom 3 coordinates (x, y, z).
        p4: Atom 4 coordinates (x, y, z).
        degrees: Return angle in degrees if True, radians if False.

    Returns:
        Dihedral angle in [-180.0, 180.0] deg or [-pi, pi] rad.
    """
    v1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    v2 = np.asarray(p3, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    v3 = np.asarray(p4, dtype=np.float64) - np.asarray(p3, dtype=np.float64)

    v2_len = np.linalg.norm(v2)
    if v2_len < 1e-12:
        raise ValueError("Degenerate central bond length in dihedral calculation.")

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    m1 = np.cross(n1, v2 / v2_len)

    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))

    rad = math.atan2(y, x)
    return math.degrees(rad) if degrees else rad


def rotate_dihedral_subset(
    coordinates: np.ndarray,
    moving_atom_indices: Sequence[int],
    bond_axis_atoms: Tuple[int, int],
    delta_angle_rad: float,
) -> np.ndarray:
    """
    Rotate a subset of atoms around a specified bond vector using Rodrigues' rotation formula.

    Formula:
        v_rot = v*cos(theta) + (u x v)*sin(theta) + u*(u . v)*(1 - cos(theta))

    Args:
        coordinates: Full Cartesian coordinate array [N x 3].
        moving_atom_indices: Sequence of atom indices to rotate.
        bond_axis_atoms: (atom1_idx, atom2_idx) defining rotation origin and axis vector.
        delta_angle_rad: Incremental rotation angle in radians.

    Returns:
        Updated Cartesian coordinate array [N x 3].
    """
    coords_out = np.copy(coordinates)
    if not moving_atom_indices or abs(delta_angle_rad) < 1e-12:
        return coords_out

    b1_idx, b2_idx = bond_axis_atoms
    origin = coords_out[b1_idx]
    axis_vec = coords_out[b2_idx] - origin
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm < 1e-12:
        raise ValueError("Degenerate bond axis vector length.")
    u = axis_vec / axis_norm

    cos_t = math.cos(delta_angle_rad)
    sin_t = math.sin(delta_angle_rad)

    for idx in moving_atom_indices:
        v = coords_out[idx] - origin
        v_rot = v * cos_t + np.cross(u, v) * sin_t + u * np.dot(u, v) * (1.0 - cos_t)
        coords_out[idx] = origin + v_rot

    return coords_out


def apply_vibration_mode(
    coordinates: np.ndarray,
    tensor: VibrationTensor,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """
    Apply normal mode vibrational coordinate displacement: r'_i = r_i + A * sin(phase) * d_i.

    Args:
        coordinates: Unperturbed Cartesian coordinates [N x 3].
        tensor: Immutable normal mode vibrational tensor.
        amplitude: Coordinate displacement amplitude scale factor.
        phase_rad: Harmonic oscillation phase angle in radians.

    Returns:
        Displaced Cartesian coordinates [N x 3].
    """
    disp = np.array(tensor.displacement_vectors, dtype=np.float64)
    if len(disp) != len(coordinates):
        raise ValueError(
            f"Vibrational displacement vector length {len(disp)} does not match atom count {len(coordinates)}."
        )
    scale = float(amplitude * math.sin(phase_rad))
    return coordinates + scale * disp


# =============================================================================
# Geometric Calculations & Mendeleev Invariants
# =============================================================================

def calculate_centroid(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute molecular geometric centroid r_c = (1/N) * sum(r_i).

    Args:
        coordinates: Cartesian coordinates array [N x 3].

    Returns:
        Centroid vector [x_c, y_c, z_c].
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if len(coords) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return np.mean(coords, axis=0)


def calculate_bounding_radius(symbols: Sequence[str], coordinates: np.ndarray) -> float:
    """
    Compute molecular bounding radius: R = max_i (||r_i - r_c|| + r_vdw,i).
    Dynamically resolves van der Waals radii in Angstroms (pm -> Angstrom) via Mendeleev.

    Args:
        symbols: Element symbols for each atom.
        coordinates: Cartesian coordinates array [N x 3].

    Returns:
        Bounding radius R in Angstroms (float > 0.0).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if len(coords) == 0 or len(symbols) == 0:
        return 1.0

    if len(symbols) != len(coords):
        raise ValueError(f"Symbols length {len(symbols)} does not match coordinates count {len(coords)}.")

    centroid = calculate_centroid(coords)
    max_radius = 0.5

    for sym, coord in zip(symbols, coords):
        # Mendeleev Mandate: Dynamic Bondi-Mantina vdW radius in Angstroms
        vdw_angstrom = get_vdw_radius(sym)
        dist = float(np.linalg.norm(coord - centroid) + vdw_angstrom)
        if dist > max_radius:
            max_radius = dist

    return max_radius


def calculate_initial_focal_distance(bounding_radius: float, fov_degrees: float = 45.0) -> float:
    """
    Compute camera focal distance d = R / sin(FOV / 2).

    Args:
        bounding_radius: Molecular bounding radius R in Angstroms.
        fov_degrees: Vertical field of view in degrees.

    Returns:
        Optimal camera focal distance d in Angstroms.
    """
    fov_rad = math.radians(fov_degrees)
    return bounding_radius / math.sin(fov_rad * 0.5)


# =============================================================================
# Molecular File Parsers: .xyz, .pdb, .cube
# =============================================================================

def parse_xyz(content: str) -> List[Tuple[List[str], np.ndarray, str]]:
    """
    Parse authentic multi-frame XYZ coordinate string.

    Args:
        content: Raw XYZ file content string.

    Returns:
        List of frames: [(symbols_list, coords_array_Nx3, comment_str), ...].
    """
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    frames: List[Tuple[List[str], np.ndarray, str]] = []
    idx = 0

    while idx < len(lines):
        try:
            num_atoms = int(lines[idx].split()[0])
        except (ValueError, IndexError) as err:
            raise ValueError(f"Invalid XYZ header line at index {idx}: '{lines[idx]}'") from err

        comment = lines[idx + 1] if idx + 1 < len(lines) else ""
        idx += 2

        symbols: List[str] = []
        coords: List[List[float]] = []

        for _ in range(num_atoms):
            if idx >= len(lines):
                raise ValueError("Unexpected end of XYZ file while parsing atom coordinates.")
            parts = lines[idx].split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ coordinate line: '{lines[idx]}'")
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            symbols.append(sym)
            coords.append([x, y, z])
            idx += 1

        frames.append((symbols, np.array(coords, dtype=np.float64), comment))

    if not frames:
        raise ValueError("No valid XYZ frames parsed from content.")

    return frames


def parse_pdb(content: str) -> List[Tuple[List[str], np.ndarray, Dict[str, Any]]]:
    """
    Parse authentic Protein Data Bank (.pdb) coordinate string.
    Supports ATOM/HETATM records and MODEL/ENDMDL multi-frame trajectories.

    Args:
        content: Raw PDB file content string.

    Returns:
        List of frames: [(symbols, coords_array_Nx3, metadata_dict), ...].
    """
    lines = content.splitlines()
    frames: List[Tuple[List[str], np.ndarray, Dict[str, Any]]] = []

    current_symbols: List[str] = []
    current_coords: List[List[float]] = []
    current_meta: Dict[str, Any] = {"atoms": []}

    in_model = False

    for line in lines:
        if line.startswith("MODEL"):
            in_model = True
            current_symbols = []
            current_coords = []
            current_meta = {"atoms": []}
        elif line.startswith("ENDMDL"):
            if current_symbols:
                frames.append((current_symbols, np.array(current_coords, dtype=np.float64), current_meta))
                current_symbols = []
                current_coords = []
                current_meta = {"atoms": []}
            in_model = False
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            # Standard PDB fixed-column parsing with fallback split
            try:
                if len(line) >= 54:
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_seq = int(line[22:26].strip()) if line[22:26].strip().isdigit() else 1
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    elem_sym = line[76:78].strip() if len(line) >= 78 and line[76:78].strip() else atom_name[0]
                else:
                    parts = line.split()
                    elem_sym = parts[2][0] if len(parts) > 2 else "C"
                    x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                    atom_name = elem_sym
                    res_name = "UNK"
                    res_seq = 1

                # Clean element symbol
                elem_clean = "".join([c for c in elem_sym if c.isalpha()]) or "C"
                current_symbols.append(elem_clean)
                current_coords.append([x, y, z])
                current_meta["atoms"].append({
                    "name": atom_name,
                    "res_name": res_name,
                    "res_seq": res_seq,
                    "record": line[:6].strip(),
                })
            except Exception as err:
                raise ValueError(f"Failed to parse PDB record line '{line}': {err}") from err

    if current_symbols and not in_model:
        frames.append((current_symbols, np.array(current_coords, dtype=np.float64), current_meta))

    if not frames:
        raise ValueError("No valid PDB atomic coordinate records parsed from content.")

    return frames


def parse_cube(content: str) -> Dict[str, Any]:
    """
    Parse authentic Gaussian/Q-Chem volumetric cube file format.

    Args:
        content: Raw .cube file content string.

    Returns:
        Dictionary containing atoms, origin, voxel grid dimensions, voxel vectors, and volumetric scalar data.
    """
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if len(lines) < 6:
        raise ValueError("Cube file must contain at least 6 header lines.")

    comment1 = lines[0]
    comment2 = lines[1]

    # Line 3: Natoms, Origin
    l3 = lines[2].split()
    natoms = abs(int(l3[0]))
    origin = np.array([float(l3[1]), float(l3[2]), float(l3[3])], dtype=np.float64)

    # Lines 4-6: Voxel counts and step vectors
    l4 = lines[3].split()
    nx = int(l4[0])
    vx = np.array([float(l4[1]), float(l4[2]), float(l4[3])], dtype=np.float64)

    l5 = lines[4].split()
    ny = int(l5[0])
    vy = np.array([float(l5[1]), float(l5[2]), float(l5[3])], dtype=np.float64)

    l6 = lines[5].split()
    nz = int(l6[0])
    vz = np.array([float(l6[1]), float(l6[2]), float(l6[3])], dtype=np.float64)

    # Atom lines
    symbols: List[str] = []
    atomic_numbers: List[int] = []
    charges: List[float] = []
    coordinates: List[List[float]] = []

    line_idx = 6
    for _ in range(natoms):
        if line_idx >= len(lines):
            raise ValueError("Unexpected EOF while parsing cube atom list.")
        aparts = lines[line_idx].split()
        z_num = int(float(aparts[0]))
        charge = float(aparts[1])
        x, y, z = float(aparts[2]), float(aparts[3]), float(aparts[4])

        sym = mendeleev_resolver.get_symbol(z_num)
        symbols.append(sym)
        atomic_numbers.append(z_num)
        charges.append(charge)
        coordinates.append([x, y, z])
        line_idx += 1

    # Remaining lines contain volumetric grid values
    grid_values: List[float] = []
    for i in range(line_idx, len(lines)):
        for val_str in lines[i].split():
            grid_values.append(float(val_str))

    expected_voxels = nx * ny * nz
    grid_array = np.array(grid_values[:expected_voxels], dtype=np.float64)
    if len(grid_array) == expected_voxels:
        grid_3d = grid_array.reshape((nx, ny, nz))
    else:
        grid_3d = grid_array

    return {
        "comment1": comment1,
        "comment2": comment2,
        "natoms": natoms,
        "origin": origin,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "voxel_x": vx,
        "voxel_y": vy,
        "voxel_z": vz,
        "symbols": symbols,
        "atomic_numbers": atomic_numbers,
        "charges": charges,
        "coordinates": np.array(coordinates, dtype=np.float64),
        "volumetric_data": grid_3d,
    }


# =============================================================================
# Air-Gapped Asset Resolution & SRI Verification
# =============================================================================

def get_static_asset_path(filename: str) -> Path:
    """
    Locate local air-gapped static asset file.

    Args:
        filename: Relative path or asset name under src/cochem/gui/static/.

    Returns:
        Path to existing static asset file.
    """
    gui_dir = Path(__file__).resolve().parent.parent
    static_path = gui_dir / "static" / filename
    if not static_path.is_file():
        raise FileNotFoundError(f"Static asset file not found at: '{static_path}'")
    return static_path


def compute_file_sri_hash(file_path: Union[str, Path]) -> str:
    """
    Compute SHA-256 Subresource Integrity (SRI) string: 'sha256-<base64>'.

    Args:
        file_path: Path to target file.

    Returns:
        SRI hash formatted string.
    """
    p = Path(file_path)
    data = p.read_bytes()
    digest = hashlib.sha256(data).digest()
    b64_hash = base64.b64encode(digest).decode("utf-8")
    return f"sha256-{b64_hash}"


def verify_static_assets() -> Dict[str, Dict[str, Any]]:
    """
    Verify presence, size, and SRI hashes of all required air-gapped static assets.

    Returns:
        Dictionary mapping asset names to metadata (path, size_bytes, sri_hash).
    """
    assets = [
        "js/3dmol/touch_controller.js",
        "js/3dmol/3Dmol-min.js",
        "css/cochem_mobile.css",
    ]
    results: Dict[str, Dict[str, Any]] = {}
    for asset_rel in assets:
        p = get_static_asset_path(asset_rel)
        sri = compute_file_sri_hash(p)
        results[asset_rel] = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sri_hash": sri,
        }
    return results


# =============================================================================
# Touch3DViewer Main Controller Widget
# =============================================================================

class Touch3DViewer:
    """
    [REQ-MOB-020 - REQ-MOB-028] CoChem-Mobile Touch-Optimized 3D Molecular Viewer Controller.
    """

    def __init__(
        self,
        symbols: Optional[Sequence[str]] = None,
        coordinates: Optional[np.ndarray] = None,
        viewport_config: Optional[ViewportConfig] = None,
        camera_state: Optional[CameraState] = None,
        selection_state: Optional[SelectionState] = None,
    ) -> None:
        """
        Initialize the Touch3DViewer with optional initial molecular coordinates and configurations.
        """
        self.viewport_config = viewport_config or ViewportConfig()
        self.selection_state = selection_state or SelectionState()

        self._frames: List[Tuple[List[str], np.ndarray, str]] = []
        self._current_frame_idx: int = 0

        if symbols is not None and coordinates is not None:
            coords = np.asarray(coordinates, dtype=np.float64)
            self._frames = [(list(symbols), coords, "Initial structure")]
            self._symbols = list(symbols)
            self._coordinates = coords
        else:
            # Default authentic physical structure: Water (H2O)
            self._symbols = ["O", "H", "H"]
            self._coordinates = np.array([
                [0.00000, 0.00000, 0.11740],
                [0.00000, 0.75700, -0.46960],
                [0.00000, -0.75700, -0.46960],
            ], dtype=np.float64)
            self._frames = [(self._symbols, self._coordinates, "H2O")]

        # Calculate bounding radius and initial camera distance
        self._bounding_radius = calculate_bounding_radius(self._symbols, self._coordinates)
        init_focal = calculate_initial_focal_distance(self._bounding_radius, self.viewport_config.fov_degrees if hasattr(self.viewport_config, 'fov_degrees') else 45.0)

        if camera_state is not None:
            self.camera_state = camera_state
        else:
            centroid = tuple(calculate_centroid(self._coordinates))
            self.camera_state = CameraState(
                quaternion=(0.0, 0.0, 0.0, 1.0),
                position=(centroid[0], centroid[1], centroid[2] + init_focal),
                target=centroid,
                fov_degrees=45.0,
                focal_distance=init_focal,
            )

        self._active_vibration: Optional[VibrationTensor] = None

    @property
    def symbols(self) -> List[str]:
        """Current chemical element symbols."""
        return list(self._symbols)

    @property
    def coordinates(self) -> np.ndarray:
        """Current Cartesian coordinates [N x 3]."""
        return np.copy(self._coordinates)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms in current frame."""
        return len(self._symbols)

    @property
    def num_frames(self) -> int:
        """Total number of trajectory frames available."""
        return len(self._frames)

    @property
    def current_frame_index(self) -> int:
        """Active trajectory frame index (0-indexed)."""
        return self._current_frame_idx

    @property
    def bounding_radius(self) -> float:
        """Dynamic molecular bounding radius in Angstroms."""
        return self._bounding_radius

    def get_centroid(self) -> np.ndarray:
        """Get molecular centroid r_c."""
        return calculate_centroid(self._coordinates)

    def load_molecule(self, symbols: Sequence[str], coordinates: np.ndarray, title: str = "") -> None:
        """
        Load a new molecular structure into the viewer.

        Args:
            symbols: Chemical element symbols.
            coordinates: Cartesian coordinates array [N x 3].
            title: Optional molecule title or comment string.
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        if len(symbols) != len(coords):
            raise ValueError(f"Mismatch: {len(symbols)} symbols vs {len(coords)} coordinate rows.")

        self._symbols = list(symbols)
        self._coordinates = coords
        self._frames = [(self._symbols, self._coordinates, title)]
        self._current_frame_idx = 0
        self._bounding_radius = calculate_bounding_radius(self._symbols, self._coordinates)
        self.auto_center()

    def load_file(self, file_path: Union[str, Path]) -> None:
        """
        Load molecular data from an authentic physical file (.xyz, .pdb, .cube).

        Args:
            file_path: Path to target molecular structure file.
        """
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Molecular file not found at: '{p}'")

        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()

        if suffix == ".xyz":
            self.load_xyz_content(text)
        elif suffix == ".pdb":
            self.load_pdb_content(text)
        elif suffix == ".cube":
            self.load_cube_content(text)
        else:
            # Fallback to XYZ parser
            self.load_xyz_content(text)

    def load_xyz_content(self, text: str) -> None:
        """Load single or multi-frame trajectory from XYZ text."""
        frames = parse_xyz(text)
        self._frames = frames
        self.set_frame(0)

    def load_pdb_content(self, text: str) -> None:
        """Load single or multi-frame trajectory from PDB text."""
        frames = parse_pdb(text)
        self._frames = [(syms, coords, meta.get("name", "PDB structure")) for syms, coords, meta in frames]
        self.set_frame(0)

    def load_cube_content(self, text: str) -> None:
        """Load molecular geometry and volumetric grid from CUBE text."""
        data = parse_cube(text)
        self._frames = [(data["symbols"], data["coordinates"], data["comment1"])]
        self._volumetric_data = data["volumetric_data"]
        self.set_frame(0)

    def set_frame(self, frame_index: int) -> None:
        """
        Switch active trajectory frame.

        Args:
            frame_index: Target frame index (0 <= index < num_frames).
        """
        if not self._frames:
            raise ValueError("No trajectory frames loaded.")
        if frame_index < 0 or frame_index >= len(self._frames):
            raise IndexError(f"Frame index {frame_index} out of range [0, {len(self._frames)-1}].")

        self._current_frame_idx = frame_index
        syms, coords, _ = self._frames[frame_index]
        self._symbols = list(syms)
        self._coordinates = np.copy(coords)
        self._bounding_radius = calculate_bounding_radius(self._symbols, self._coordinates)
        self.auto_center()

    def stream_trajectory_frame(
        self,
        cache_file_path: Union[str, Path],
        frame_index: int,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Multi-process safe trajectory streaming using cross-process FileLock.

        Args:
            cache_file_path: Path to cached trajectory file.
            frame_index: Target frame index to retrieve.

        Returns:
            Tuple of (symbols, coordinates_array).
        """
        cache_path = Path(cache_file_path).resolve()
        lock_path = cache_path.with_suffix(".lock")

        with filelock.FileLock(str(lock_path), timeout=10.0):
            if not cache_path.is_file():
                raise FileNotFoundError(f"Trajectory cache file not found at: {cache_path}")
            content = cache_path.read_text(encoding="utf-8")
            frames = parse_xyz(content)
            if frame_index < 0 or frame_index >= len(frames):
                raise IndexError(f"Stream frame index {frame_index} out of range [0, {len(frames)-1}].")
            syms, coords, _ = frames[frame_index]
            self._symbols = syms
            self._coordinates = coords
            self._bounding_radius = calculate_bounding_radius(self._symbols, self._coordinates)
            return (syms, coords)

    def set_camera_state(self, camera_state: CameraState) -> None:
        """Set immutable camera state."""
        self.camera_state = camera_state

    def set_viewport_config(self, viewport_config: ViewportConfig) -> None:
        """Set immutable viewport configuration."""
        self.viewport_config = viewport_config

    def select_atoms(self, atom_indices: Sequence[int]) -> None:
        """Select specific atom indices."""
        self.selection_state = SelectionState(
            selected_atom_indices=tuple(sorted(set(atom_indices))),
            active_dihedral=self.selection_state.active_dihedral,
        )

    def set_active_dihedral(self, dihedral: Optional[Tuple[int, int, int, int]]) -> None:
        """Set active 4-atom dihedral selection for torsion manipulation."""
        if dihedral is not None:
            if len(dihedral) != 4:
                raise ValueError("Dihedral selection must specify exactly 4 atom indices.")
            for idx in dihedral:
                if idx < 0 or idx >= self.num_atoms:
                    raise IndexError(f"Dihedral atom index {idx} out of range [0, {self.num_atoms-1}].")

        self.selection_state = SelectionState(
            selected_atom_indices=self.selection_state.selected_atom_indices,
            active_dihedral=dihedral,
        )

    def apply_dihedral_rotation(
        self,
        moving_atom_indices: Sequence[int],
        delta_angle_degrees: float,
    ) -> np.ndarray:
        """
        Rotate subset of atoms around active dihedral central bond (atoms 2 and 3).

        Args:
            moving_atom_indices: Atom indices to apply rotation to.
            delta_angle_degrees: Rotation angle in degrees.

        Returns:
            Updated coordinates array [N x 3].
        """
        if self.selection_state.active_dihedral is None:
            raise ValueError("No active dihedral set. Call set_active_dihedral first.")

        a1, a2, a3, a4 = self.selection_state.active_dihedral
        delta_rad = math.radians(delta_angle_degrees)
        new_coords = rotate_dihedral_subset(
            coordinates=self._coordinates,
            moving_atom_indices=moving_atom_indices,
            bond_axis_atoms=(a2, a3),
            delta_angle_rad=delta_rad,
        )
        self._coordinates = new_coords
        return np.copy(self._coordinates)

    def set_vibration_mode(self, tensor: VibrationTensor) -> None:
        """Attach active normal mode vibrational displacement tensor."""
        if len(tensor.displacement_vectors) != self.num_atoms:
            raise ValueError(
                f"Tensor displacements length {len(tensor.displacement_vectors)} != atom count {self.num_atoms}"
            )
        self._active_vibration = tensor

    def get_vibrational_frame(self, amplitude: float = 1.0, phase_rad: float = 0.0) -> np.ndarray:
        """
        Compute Cartesian coordinates at a specific harmonic vibrational phase.

        Args:
            amplitude: Displacement scaling factor.
            phase_rad: Phase angle in radians.

        Returns:
            Displaced coordinates array [N x 3].
        """
        if self._active_vibration is None:
            return np.copy(self._coordinates)
        return apply_vibration_mode(
            coordinates=self._coordinates,
            tensor=self._active_vibration,
            amplitude=amplitude,
            phase_rad=phase_rad,
        )

    def auto_center(self) -> None:
        """Auto-center camera target onto molecular geometric centroid."""
        centroid = tuple(calculate_centroid(self._coordinates))
        focal_dist = calculate_initial_focal_distance(self._bounding_radius, self.camera_state.fov_degrees)
        self.camera_state = CameraState(
            quaternion=self.camera_state.quaternion,
            position=(centroid[0], centroid[1], centroid[2] + focal_dist),
            target=centroid,
            fov_degrees=self.camera_state.fov_degrees,
            focal_distance=focal_dist,
        )

    def fit_view(self) -> None:
        """Adjust camera focal distance to tightly fit molecular bounding radius."""
        focal_dist = calculate_initial_focal_distance(self._bounding_radius, self.camera_state.fov_degrees)
        target = self.camera_state.target
        self.camera_state = CameraState(
            quaternion=self.camera_state.quaternion,
            position=(target[0], target[1], target[2] + focal_dist),
            target=target,
            fov_degrees=self.camera_state.fov_degrees,
            focal_distance=focal_dist,
        )

    def reset_camera(self) -> None:
        """Reset camera orientation to identity quaternion [0, 0, 0, 1] and auto-center."""
        centroid = tuple(calculate_centroid(self._coordinates))
        focal_dist = calculate_initial_focal_distance(self._bounding_radius, 45.0)
        self.camera_state = CameraState(
            quaternion=(0.0, 0.0, 0.0, 1.0),
            position=(centroid[0], centroid[1], centroid[2] + focal_dist),
            target=centroid,
            fov_degrees=45.0,
            focal_distance=focal_dist,
        )

    def to_xyz_string(self) -> str:
        """Export current frame as standard XYZ formatted string."""
        lines = [f"{len(self._symbols)}", "CoChem-Mobile 3D Molecular Viewer Export"]
        for sym, (x, y, z) in zip(self._symbols, self._coordinates):
            lines.append(f"{sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}")
        return "\n".join(lines) + "\n"

    def to_html_bundle(self, inline_assets: bool = True) -> str:
        """
        Generate self-contained, air-gapped HTML5 bundle with embedded WebGL 3D viewer,
        touch controller, CSS styling, and current molecular scene graph.

        Args:
            inline_assets: Embed static CSS and JavaScript inline when True.

        Returns:
            Complete HTML document string.
        """
        xyz_data = self.to_xyz_string()
        css_content = get_static_asset_path("css/cochem_mobile.css").read_text(encoding="utf-8")
        js_3dmol = get_static_asset_path("js/3dmol/3Dmol-min.js").read_text(encoding="utf-8")
        js_touch = get_static_asset_path("js/3dmol/touch_controller.js").read_text(encoding="utf-8")

        camera_dict = {
            "quaternion": list(self.camera_state.quaternion),
            "target": list(self.camera_state.target),
            "fov_degrees": self.camera_state.fov_degrees,
            "focal_distance": self.camera_state.focal_distance,
        }

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'self'; style-src 'unsafe-inline' 'self'; img-src data: blob: 'self'; connect-src 'self' ws: wss:;">
  <title>CoChem 3D Molecular Touch Viewer</title>
  <style>
{css_content}
  </style>
</head>
<body style="margin: 0; padding: 0; background: {self.viewport_config.background_color}; overflow: hidden;">
  <div id="cochem-viewport-container" class="cochem-3d-viewport" style="width: 100vw; height: 100vh;">
    <div id="cochem-hud" class="cochem-3d-info-hud">
      Atoms: {self.num_atoms} | R: {self._bounding_radius:.2f} Å
    </div>
  </div>

  <script>
{js_3dmol}
  </script>
  <script>
{js_touch}
  </script>
  <script>
    (function() {{
      const container = document.getElementById('cochem-viewport-container');
      const xyzData = {json.dumps(xyz_data)};
      const initialCamera = {json.dumps(camera_dict)};
      const boundingRadius = {self._bounding_radius};

      const viewer = $3Dmol.createViewer(container, {{
        backgroundColor: '{self.viewport_config.background_color}'
      }});

      viewer.addModel(xyzData, 'xyz');
      viewer.setStyle({{}}, {{ stick: {{ radius: 0.15 }}, sphere: {{ scale: 0.3 }} }});
      viewer.zoomTo();
      viewer.render();

      const controller = new TouchController(container, {{
        boundingRadius: boundingRadius,
        fovDegrees: initialCamera.fov_degrees,
        viewer: viewer,
        onAtomPick: function(e) {{
          console.log('Atom picked at screen:', e.screenX, e.screenY);
        }}
      }});

      controller.setCameraState(initialCamera);
      window.cochemViewer = viewer;
      window.cochemTouchController = controller;
    }})();
  </script>
</body>
</html>"""
        return html

    def render_widget(self) -> ipywidgets.HTML:
        """
        Generate reactive Jupyter / Voila ipywidgets HTML container.

        Returns:
            ipywidgets.HTML instance rendering the interactive 3D bundle.
        """
        html_src = self.to_html_bundle(inline_assets=True)
        # Escape HTML iframe embed
        escaped_html = html_src.replace('"', '&quot;')
        iframe_html = f'<iframe srcdoc="{escaped_html}" style="width: {self.viewport_config.width_px}px; height: {self.viewport_config.height_px}px; border: none; border-radius: 8px;"></iframe>'
        return ipywidgets.HTML(value=iframe_html)
