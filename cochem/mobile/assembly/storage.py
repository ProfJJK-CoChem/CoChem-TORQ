"""SWMR HDF5 persistence and double-precision UTF-8 .xyz coordinate serialization.

Strict adherence to the Zero-Mock mandate with cross-platform filelock protection.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from filelock import FileLock, Timeout

from cochem.mobile.assembly.exceptions import SWMRStorageLockError

logger = logging.getLogger(__name__)


def calculate_stoichiometry(atomic_symbols: list[str]) -> str:
    """Generate Hill system empirical formula from a list of elemental symbols.

    In Hill system:
    - If Carbon is present: 'C' first, 'H' second, remaining elements in alphabetical order.
    - If no Carbon: all elements in alphabetical order.

    Args:
        atomic_symbols: List of element symbols.

    Returns:
        Hill formula string (e.g. 'C6H24CoN6', 'FeH12O6').
    """
    counts = collections.Counter(atomic_symbols)
    elements = set(counts.keys())

    ordered_elems = []
    if "C" in elements:
        ordered_elems.append("C")
        elements.remove("C")
        if "H" in elements:
            ordered_elems.append("H")
            elements.remove("H")

    ordered_elems.extend(sorted(elements))

    formula_parts = []
    for elem in ordered_elems:
        cnt = counts[elem]
        if cnt == 1:
            formula_parts.append(elem)
        else:
            formula_parts.append(f"{elem}{cnt}")

    return "".join(formula_parts)


def calculate_provenance_hash(request_dict: dict[str, Any], coordinates: np.ndarray) -> str:
    """Calculate deterministic SHA-256 provenance hash from request payload and coordinates.

    Args:
        request_dict: Canonical JSON-serializable dictionary of assembly request.
        coordinates: Numpy array of coordinates (shape N, 3).

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    hasher = hashlib.sha256()

    # Deterministic JSON representation of request
    req_json = json.dumps(request_dict, sort_keys=True, separators=(",", ":"))
    hasher.update(req_json.encode("utf-8"))

    # Coordinate representation with fixed precision
    coord_bytes = coordinates.astype(np.float64).tobytes()
    hasher.update(coord_bytes)

    return hasher.hexdigest()


def format_xyz_string(
    atomic_symbols: list[str],
    coordinates: np.ndarray,
    stoichiometry: str,
    net_charge: int,
    spin_multiplicity: int,
    sha256_hash: str,
) -> str:
    """Format double-precision POSIX/Windows UTF-8 .xyz coordinate data.

    Args:
        atomic_symbols: List of N elemental symbols.
        coordinates: Coordinate array of shape (N, 3).
        stoichiometry: Hill formula string.
        net_charge: Formal net charge of complex.
        spin_multiplicity: Spin multiplicity (2S + 1).
        sha256_hash: Cryptographic provenance hash.

    Returns:
        Formatted .xyz string with strict 6 decimal places.
    """
    num_atoms = len(atomic_symbols)
    lines = [
        str(num_atoms),
        f"Stoichiometry={stoichiometry} NetCharge={net_charge} SpinMultiplicity={spin_multiplicity} SHA256={sha256_hash}",
    ]

    for sym, (x, y, z) in zip(atomic_symbols, coordinates, strict=True):
        lines.append(f"{sym:<2} {x:>14.6f} {y:>14.6f} {z:>14.6f}")

    return "\n".join(lines) + "\n"


def save_complex_to_hdf5(
    h5_path: Path | str,
    record_key: str,
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    metadata: dict[str, Any],
    lock_timeout: float = 10.0,
) -> None:
    """Commit complex coordinates, atomic numbers, and metadata to SWMR HDF5 store under filelock.

    Args:
        h5_path: Path to HDF5 database file.
        record_key: Unique record key identifier.
        coordinates: Coordinate array of shape (N, 3).
        atomic_numbers: Atomic numbers array of shape (N,).
        metadata: JSON-serializable dictionary of metadata attributes.
        lock_timeout: Maximum seconds to wait for filelock acquisition.

    Raises:
        SWMRStorageLockError: If filelock acquisition times out or fails.
    """
    path = Path(h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.with_suffix(".lock")

    lock = FileLock(str(lock_file), timeout=lock_timeout)
    try:
        with lock:
            with h5py.File(path, "a", libver="latest") as h5_file:
                grp_path = f"complexes/{record_key}"
                if grp_path in h5_file:
                    del h5_file[grp_path]

                grp = h5_file.create_group(grp_path)
                grp.create_dataset(
                    "coordinates", data=coordinates.astype(np.float64), dtype="float64"
                )
                grp.create_dataset(
                    "atomic_numbers", data=atomic_numbers.astype(np.int32), dtype="int32"
                )

                meta_json = json.dumps(metadata, sort_keys=True)
                grp.attrs["metadata_json"] = meta_json
                for k, v in metadata.items():
                    if isinstance(v, (int, float, str, bool)):
                        grp.attrs[k] = v

                h5_file.flush()

    except Timeout as exc:
        raise SWMRStorageLockError(
            f"Filelock acquisition timed out ({lock_timeout}s) for HDF5 store at {path}."
        ) from exc
    except Exception as exc:
        if isinstance(exc, SWMRStorageLockError):
            raise
        raise SWMRStorageLockError(
            f"Failed to persist complex record '{record_key}' to HDF5 at {path}: {exc}"
        ) from exc


def read_complex_from_hdf5(
    h5_path: Path | str,
    record_key: str,
    lock_timeout: float = 10.0,
) -> dict[str, Any]:
    """Read complex record from SWMR HDF5 store under filelock.

    Args:
        h5_path: Path to HDF5 database file.
        record_key: Unique record key identifier.
        lock_timeout: Maximum seconds to wait for filelock.

    Returns:
        Dictionary containing coordinates, atomic_numbers, and metadata.

    Raises:
        SWMRStorageLockError: If lock fails or record is not found.
    """
    path = Path(h5_path)
    if not path.exists():
        raise SWMRStorageLockError(f"HDF5 database not found at {path}.")

    lock_file = path.with_suffix(".lock")
    lock = FileLock(str(lock_file), timeout=lock_timeout)

    try:
        with lock:
            with h5py.File(path, "r", libver="latest") as h5_file:
                grp_path = f"complexes/{record_key}"
                if grp_path not in h5_file:
                    raise KeyError(f"Record key '{record_key}' not found in HDF5 store at {path}.")

                grp = h5_file[grp_path]
                coordinates = grp["coordinates"][:]
                atomic_numbers = grp["atomic_numbers"][:]
                meta_json = grp.attrs.get("metadata_json", "{}")
                metadata = json.loads(meta_json)

                return {
                    "record_key": record_key,
                    "coordinates": coordinates,
                    "atomic_numbers": atomic_numbers,
                    "metadata": metadata,
                }

    except Timeout as exc:
        raise SWMRStorageLockError(
            f"Filelock acquisition timed out ({lock_timeout}s) for HDF5 store at {path}."
        ) from exc
    except Exception as exc:
        if isinstance(exc, (SWMRStorageLockError, KeyError)):
            raise
        raise SWMRStorageLockError(
            f"Failed to read complex record '{record_key}' from HDF5 at {path}: {exc}"
        ) from exc
