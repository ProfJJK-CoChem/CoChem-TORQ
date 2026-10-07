"""Explicit isotope resolution using the installed Mendeleev database.

An unqualified element selects its most abundant naturally occurring isotope.
Elements without a tabulated natural abundance require an explicit mass number.
Tabulated isotope masses have measurement uncertainty; they are not exact SI constants.
"""

from __future__ import annotations

import re
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path


@lru_cache(maxsize=512)
def _resolve(symbol: str) -> tuple:
    from mendeleev import element

    text = {"D": "2H", "T": "3H"}.get(symbol.strip(), symbol.strip())
    match = re.fullmatch(r"(?:(\d+)([A-Za-z]{1,2})|([A-Za-z]{1,2})(\d*)?)", text)
    if match is None:
        raise ValueError(f"Invalid element/isotope: {symbol!r}")
    number = match.group(1) or match.group(4)
    elem = element((match.group(2) or match.group(3)).capitalize())
    candidates = [iso for iso in elem.isotopes if iso.mass is not None]
    if number:
        candidates = [iso for iso in candidates if iso.mass_number == int(number)]
        if not candidates:
            raise ValueError(f"No tabulated mass for requested isotope {symbol!r}")
        selected = candidates[0]
        policy = "explicit_mass_number"
    else:
        candidates = [
            iso for iso in candidates if iso.abundance is not None and iso.abundance > 0
        ]
        if not candidates:
            raise ValueError(
                f"Specify an isotope for {symbol!r}; no natural-abundance default exists"
            )
        selected = max(candidates, key=lambda iso: iso.abundance)
        policy = "most_abundant_naturally_occurring_isotope"
    return (
        elem.symbol,
        int(selected.mass_number),
        float(selected.mass),
        float(selected.mass_uncertainty)
        if selected.mass_uncertainty is not None
        else None,
        policy,
    )


@lru_cache(maxsize=1)
def _database_source() -> tuple[str, str]:
    import mendeleev

    database = Path(mendeleev.__file__).parent / "elements.db"
    if not database.is_file():
        raise RuntimeError("The actual Mendeleev isotope database is missing.")
    hasher = sha256()
    with database.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return version("mendeleev"), hasher.hexdigest()


def isotope_record(symbol: str) -> dict:
    element, mass_number, mass_u, uncertainty, policy = _resolve(symbol)
    distribution_version, database_digest = _database_source()
    return {
        "element": element,
        "mass_number": mass_number,
        "label": f"{mass_number}{element}",
        "mass_u": mass_u,
        "mass_uncertainty_u": uncertainty,
        "selection_policy": policy,
        "source": {
            "database": "Mendeleev elements.db",
            "distribution_version": distribution_version,
            "database_sha256": database_digest,
            "tabulated_mass_is_exact": False,
        },
    }


@lru_cache(maxsize=512)
def isotope_mass(symbol: str) -> float:
    return _resolve(symbol)[2]
