"""Topology properties use actual database definitions without substitution.

The tests query genuine Mendeleev records. Geometry cases exercise only the
explicitly labelled radius heuristic, never electronic or validated graph data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from mendeleev import element
from mendeleev.fetch import fetch_table
from pydantic import ValidationError

from Libraries.cochem_torq_topology import (
    TorqTopology,
    get_atomic_mass,
    get_atomic_number,
    get_isotopic_mass,
    get_pyykko_radius,
    get_vdw_radius,
)
from Libraries.torq_config import TorqRunParams


def _tabulated_isotope_mass(symbol: str, number: int) -> float:
    rows = [item for item in element(symbol).isotopes if item.mass_number == number]
    assert len(rows) == 1 and rows[0].mass is not None
    return float(rows[0].mass)


@pytest.mark.parametrize(
    ("label", "symbol", "number"),
    [
        ("13C", "C", 13),
        ("C13", "C", 13),
        ("C-13", "C", 13),
        ("C:13", "C", 13),
        ("C_13", "C", 13),
        ("13-C", "C", 13),
        ("D", "H", 2),
        ("d", "H", 2),
        ("H-2", "H", 2),
        ("T", "H", 3),
        ("18O", "O", 18),
    ],
)
def test_explicit_notation_preserves_the_actual_tabulated_isotope(
    label: str, symbol: str, number: int
) -> None:
    expected = _tabulated_isotope_mass(symbol, number)
    assert get_atomic_mass(label) == expected
    assert get_isotopic_mass(label) == expected
    assert get_isotopic_mass(label, number) == expected
    assert expected != number


@pytest.mark.parametrize("symbol", ["H", "C", "O", "Au", "Og"])
def test_bare_atomic_mass_preserves_the_named_database_atomic_weight(
    symbol: str,
) -> None:
    observed = element(symbol).atomic_weight
    assert observed is not None and np.isfinite(observed) and observed > 0
    assert get_atomic_mass(symbol) == float(observed)


@pytest.mark.parametrize("symbol", ["H", "C", "O"])
def test_bare_isotope_default_is_actual_most_abundant_natural_isotope(
    symbol: str,
) -> None:
    candidates = [
        item
        for item in element(symbol).isotopes
        if item.mass is not None and item.abundance is not None and item.abundance > 0
    ]
    selected = max(candidates, key=lambda item: item.abundance)
    assert get_isotopic_mass(symbol) == float(selected.mass)
    assert get_isotopic_mass(symbol) != get_atomic_mass(symbol)


@pytest.mark.parametrize("label", ["999H", "H999", "H-999", "C-999", "0H"])
def test_missing_explicit_isotope_never_becomes_an_atomic_weight(label: str) -> None:
    with pytest.raises(ValueError):
        get_atomic_mass(label)
    with pytest.raises(ValueError):
        get_isotopic_mass(label)


@pytest.mark.parametrize("symbol", ["H", "C", "Og"])
def test_missing_separate_mass_argument_is_rejected(symbol: str) -> None:
    with pytest.raises(ValueError, match="No tabulated mass"):
        get_isotopic_mass(symbol, 999)


@pytest.mark.parametrize(
    ("label", "argument"), [("13C", 12), ("D", 3), ("T", 2), ("H:2", 3)]
)
def test_conflicting_explicit_isotope_declarations_are_rejected(
    label: str, argument: int
) -> None:
    with pytest.raises(ValueError, match="conflict"):
        get_isotopic_mass(label, argument)


@pytest.mark.parametrize(
    "argument", [True, False, 1.0, 13.0, np.int64(13), "13", 0, -1]
)
def test_invalid_mass_argument_cannot_bypass_validation_through_the_cache(
    argument,
) -> None:
    # Prime legitimate keys: bool/float equality must not select cached integers.
    assert get_isotopic_mass("H", 1) == _tabulated_isotope_mass("H", 1)
    assert get_isotopic_mass("C", 13) == _tabulated_isotope_mass("C", 13)
    with pytest.raises(ValueError, match="positive integer"):
        get_isotopic_mass("H", argument)
    with pytest.raises(ValueError, match="positive integer"):
        get_isotopic_mass("C", argument)


@pytest.mark.parametrize("symbol", ["Tc", "Og"])
def test_no_natural_abundance_default_requires_an_explicit_isotope(symbol: str) -> None:
    rows = element(symbol).isotopes
    assert not any(item.abundance is not None and item.abundance > 0 for item in rows)
    with pytest.raises(ValueError, match="Specify an isotope"):
        get_isotopic_mass(symbol)
    selected = next(item for item in rows if item.mass is not None)
    assert get_isotopic_mass(symbol, int(selected.mass_number)) == float(selected.mass)


@pytest.mark.parametrize("label", [1, "", "H:", "C-13junk", "1.5H", "UnknownElement"])
def test_malformed_element_identity_is_not_coerced(label) -> None:
    with pytest.raises(ValueError):
        get_atomic_mass(label)
    with pytest.raises(ValueError):
        get_isotopic_mass(label)


@pytest.mark.parametrize("label", ["999C", "0C", "999H"])
def test_element_properties_and_graphs_cannot_admit_nonexistent_isotope_tags(
    label: str,
) -> None:
    for getter in (get_atomic_number, get_pyykko_radius, get_vdw_radius):
        with pytest.raises(ValueError):
            getter(label)
    with pytest.raises(ValueError):
        TorqTopology([label], [[0.0, 0.0, 0.0]])


@pytest.mark.parametrize("symbol", ["H", "C", "O", "Au", "Ts", "Og"])
def test_pyykko_radius_uses_the_exact_named_database_definition(symbol: str) -> None:
    value = element(symbol).covalent_radius_pyykko
    assert value is not None and np.isfinite(value) and value > 0
    assert get_pyykko_radius(symbol) == float(value) / 100.0


def test_available_and_missing_radius_definitions_use_actual_database_rows() -> None:
    table = fetch_table("elements")
    assert len(table) > 0 and table.symbol.is_unique
    missing_vdw = table.loc[table.vdw_radius.isna(), "symbol"].tolist()
    assert missing_vdw, "The genuine database must supply missingness evidence."
    for symbol in missing_vdw:
        with pytest.raises(ValueError, match="vdw_radius"):
            get_vdw_radius(symbol)
    # The current genuine table has Pyykko values for all118 elements. If an
    # actual database edition omits one, missingness must remain explicit.
    for symbol in table.loc[table.covalent_radius_pyykko.isna(), "symbol"]:
        with pytest.raises(ValueError, match="covalent_radius_pyykko"):
            get_pyykko_radius(symbol)


@pytest.mark.parametrize("symbol", ["H", "C", "O", "Au"])
def test_default_vdw_radius_retains_its_definition(symbol: str) -> None:
    value = element(symbol).vdw_radius
    assert value is not None and np.isfinite(value) and value > 0
    assert get_vdw_radius(symbol) == float(value) / 100.0


def test_radius_graph_and_bond_orders_are_explicit_heuristic_candidates() -> None:
    graph = TorqTopology(["C", "H"], [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]).graph
    assert graph.graph["evidence_class"] == "geometric_radius_heuristic"
    assert graph.graph["independently_verified"] is False
    assert graph.graph["coordinate_unit"] == "angstrom"
    assert graph.graph["radius_definition"] == "Mendeleev covalent_radius_pyykko"
    assert graph.graph["bond_order_status"] == "heuristic_candidate"
    assert graph.graph["tolerance_multiplier"] == 1.15
    assert graph.has_edge(0, 1)


@pytest.mark.parametrize(
    "coordinates",
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]],
        np.array([[0.0, 0.0, 0.0], [1.0 + 1j, 0.0, 0.0]]),
        [[0.0, 0.0, 0.0]],
    ],
)
def test_invalid_geometry_is_not_silently_repaired_for_graph_construction(
    coordinates,
) -> None:
    with pytest.raises(ValueError):
        TorqTopology(["C", "H"], coordinates)


@pytest.mark.parametrize(
    "tier", ["T1-10s", "T2-1m", "T3-1h", "T4-1mo", "low", "medium", "high", "ultra"]
)
def test_documentary_tiers_cannot_become_executable_engine_inputs(
    tmp_path, tier: str
) -> None:
    topology = TorqTopology(["C", "H"], [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
    path = tmp_path / "cascade-plan.json"
    plan = topology.generate_cascade_parameters(tier=tier, output_path=path)
    assert json.loads(path.read_text()) == plan
    assert plan["schema_version"] == "cochem.torq.legacy-cascade-planning/1"
    assert plan["status"] == "planning_only"
    assert plan["dispatch_authorized"] is False
    assert plan["scientific_qualification"] == "unqualified"
    assert plan["engine_input"] is None
    assert plan["keywords"] == []
    for name in (
        "engine",
        "method",
        "basis_set",
        "dispersion",
        "anharmonicity",
        "bsse_correction",
        "cabs_mappings",
    ):
        assert plan[name] is None
    # Genuine consumer validation, rather than a declaration-only boolean check.
    with pytest.raises(ValidationError):
        TorqRunParams.model_validate(plan)


def test_f12_planning_retains_missing_auxiliary_definitions_instead_of_filenames(
    tmp_path,
) -> None:
    topology = TorqTopology(["C", "H"], [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
    plan = topology.generate_cascade_parameters(
        tier="T3", output_path=tmp_path / "f12-plan.json"
    )
    assert plan["proposal"]["requested_method"] == "CCSD(T)-F12"
    assert plan["proposal"]["requested_basis"] == "cc-pVTZ-F12"
    assert plan["auxiliary_basis"]["status"] == "unavailable"
    assert plan["auxiliary_basis"]["values"] is None
    assert plan["cabs_mappings"] is None
    assert "cc-pVTZ-F12-OptRI" not in json.dumps(plan)
    assert "cc-pVTZ-F12-JKFIT" not in json.dumps(plan)
    assert "cc-pVTZ-F12-MP2FIT" not in json.dumps(plan)


def test_known_modern_method_does_not_implicitly_validate_a_legacy_recipe(
    tmp_path,
) -> None:
    topology = TorqTopology(["C", "H"], [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
    plan = topology.generate_cascade_parameters(
        tier="T1",
        method="hf",
        basis_set="sto-3g",
        output_path=tmp_path / "hf-plan.json",
    )
    assert plan["proposal"]["requested_method"] == "hf"
    assert plan["proposal"]["requested_basis"] == "sto-3g"
    assert plan["engine_input"] is None
    assert plan["dispatch_authorized"] is False
    assert plan["auxiliary_basis"]["status"] == "not_requested"


@pytest.mark.parametrize(
    "arguments",
    [
        {"tier": "T5"},
        {"tier": ""},
        {"tier": True},
        {"method": ""},
        {"basis_set": ""},
        {"method": 1},
    ],
)
def test_unknown_or_malformed_planning_request_preserves_existing_file(
    tmp_path, arguments
) -> None:
    topology = TorqTopology(["C", "H"], [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
    path = tmp_path / "existing-plan.json"
    original = b'{"existing_user_document":true}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError):
        topology.generate_cascade_parameters(output_path=path, **arguments)
    assert path.read_bytes() == original
