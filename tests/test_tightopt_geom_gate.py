"""Fail-closed Method Matrix §4.4 `%geom` / `! TightOpt` gate for CoChem-TORQ.

Enforces decision `sys-1126-a3d4` and CoChem Method Matrix §4.3–§4.4 / §9B.4:
1. Any ORCA QM optimization input deck emitting `! ... TightOpt` MUST include
   the full §4.4 `%geom` convergence block (`TolE <= 1e-7`, `TolRMSG <= 3e-6`,
   `TolMaxG <= 1e-5`, `TolRMSD <= 5e-5`, `TolMaxD <= 1e-4`), `TightSCF` or
   `VeryTightSCF`, and `DEFGRID3` (for DFT/Hessian decks).
2. The ONLY permitted exception for `! TightOpt` without §4.4 `%geom` is
   MLFF/ExtOpt float32 pre-screening (`ExtOpt` paired with `%scf TolE 1e-5 end`
   per Method Matrix §4.4 L435 & §9B.4 L2598, explicitly pre-optimization only).
3. Canonical `src/cochem_torq/` must not emit bare unverified `! TightOpt` decks.
"""

from __future__ import annotations

import re
from pathlib import Path

from Libraries.cochem_torq_goat import (
    GoatConfig,
    GoatExtOptDriver,
    GoatMode,
    generate_orca_goat_input,
)
from scripts.oet_client import OETClient

REPO_ROOT = Path(__file__).resolve().parent.parent

SECTION_44_LIMITS: dict[str, float] = {
    "tole": 1.0e-7,
    "tolrmsg": 3.0e-6,
    "tolmaxg": 1.0e-5,
    "tolrmsd": 5.0e-5,
    "tolmaxd": 1.0e-4,
}


def validate_orca_tightopt_deck(
    deck_text: str,
    *,
    require_dft_grid: bool = True,
) -> list[str]:
    """Return a list of compliance violations; empty list means compliant."""
    violations: list[str] = []
    bang_lines = [
        line.strip() for line in deck_text.splitlines() if line.strip().startswith("!")
    ]
    combined_bang = " ".join(bang_lines)
    if not re.search(r"\btightopt\b", combined_bang, re.IGNORECASE):
        return violations

    # MLFF/ExtOpt float32 pre-screening exception (Method Matrix §4.4 & §9B.4)
    if re.search(r"\bextopt\b", combined_bang, re.IGNORECASE):
        scf_match = re.search(
            r"%scf\b(.*?)\bend\b", deck_text, re.IGNORECASE | re.DOTALL
        )
        if not scf_match or not re.search(
            r"\btole\s+([0-9.eEdD+-]+)", scf_match.group(1), re.IGNORECASE
        ):
            violations.append(
                "ExtOpt TightOpt pre-screen deck missing mandatory "
                "`%scf TolE ... end` noise-floor guard."
            )
        return violations

    # QM optimization deck: require TightSCF/VeryTightSCF, DEFGRID3, and §4.4 %geom
    if not re.search(r"\b(?:very)?tightscf\b", combined_bang, re.IGNORECASE):
        violations.append(
            "QM TightOpt deck missing mandatory `TightSCF` or `VeryTightSCF`."
        )

    if require_dft_grid and not re.search(
        r"\bdefgrid3\b", combined_bang, re.IGNORECASE
    ):
        violations.append(
            "DFT/Hessian TightOpt deck missing mandatory `DEFGRID3` integration grid."
        )

    geom_match = re.search(r"%geom\b(.*?)\bend\b", deck_text, re.IGNORECASE | re.DOTALL)
    if not geom_match:
        violations.append(
            "Bare `! TightOpt` deck missing Method Matrix §4.4 "
            "`%geom` convergence block."
        )
        return violations

    geom_body = geom_match.group(1)
    for key, max_allowed in SECTION_44_LIMITS.items():
        tok = re.search(
            rf"\b{key}\s+([0-9]+(?:\.[0-9]+)?(?:[eEdD][+-]?[0-9]+)?)",
            geom_body,
            re.IGNORECASE,
        )
        if not tok:
            violations.append(
                f"Section 4.4 `%geom` missing required `{key}` (<= {max_allowed:.1e})."
            )
            continue
        val = float(tok.group(1).replace("D", "e").replace("d", "e"))
        if val > max_allowed * (1.0 + 1e-12):
            violations.append(
                f"Section 4.4 `%geom` `{key}={val:.2e}` exceeds `{max_allowed:.1e}`."
            )
    return violations


def test_bare_tightopt_deck_fails_closed() -> None:
    bare_deck = """! wB97M-V def2-QZVPP def2/J RIJCOSX TightOpt TightSCF DEFGRID3
%pal nprocs 4 end
* xyz 0 1
  O 0.0 0.0 0.0
  H 0.0 0.757 0.586
  H 0.0 -0.757 0.586
*
"""
    violations = validate_orca_tightopt_deck(bare_deck, require_dft_grid=True)
    assert violations, "Bare ! TightOpt deck without %geom must be rejected."
    assert any("%geom" in v for v in violations)


def test_loose_geom_tolerances_fail_closed() -> None:
    loose_deck = """! wB97M-V def2-QZVPP def2/J RIJCOSX TightOpt TightSCF DEFGRID3
%geom
  TolE 1e-6
  TolRMSG 3e-5
  TolMaxG 1e-4
  TolRMSD 6e-4
  TolMaxD 1e-3
end
* xyz 0 1
  O 0.0 0.0 0.0
*
"""
    violations = validate_orca_tightopt_deck(loose_deck, require_dft_grid=True)
    assert len(violations) == 5


def test_section44_compliant_deck_passes() -> None:
    compliant_deck = """! wB97M-V def2-QZVPP def2/J RIJCOSX TightOpt TightSCF DEFGRID3
%geom
  InHess XTB2
  TolE     1e-7
  TolRMSG  3e-6
  TolMaxG  1e-5
  TolRMSD  5e-5
  TolMaxD  1e-4
end
* xyz 0 1
  O 0.0 0.0 0.0
*
"""
    assert validate_orca_tightopt_deck(compliant_deck, require_dft_grid=True) == []


def test_torq_goat_mlff_extopt_emitter_satisfies_prescreen_contract() -> None:
    cfg = GoatConfig(
        mode=GoatMode.GOAT_EXPLORE,
        driver=GoatExtOptDriver.AIMNET2,
        threads=4,
        tight_opt=True,
        scf_tol_e=1e-5,
    )
    deck = generate_orca_goat_input(cfg, xyz_filename="input.xyz")
    assert "! GOAT-EXPLORE ExtOpt TightOpt" in deck
    assert validate_orca_tightopt_deck(deck, require_dft_grid=False) == []


def test_torq_oet_client_extopt_emitter_satisfies_prescreen_contract() -> None:
    client = OETClient(host="localhost", port=8888)
    deck = client.format_orca_extopt_input(
        xyz_filename="input.xyz", pal=4, tight_opt=True, scf_tole=1e-5
    )
    assert "! GOAT-EXPLORE ExtOpt TightOpt" in deck
    assert validate_orca_tightopt_deck(deck, require_dft_grid=False) == []


def test_canonical_src_cochem_torq_has_no_unverified_bare_tightopt() -> None:
    src_root = REPO_ROOT / "src" / "cochem_torq"
    offenders: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), 1):
            if re.search(r"!\s*[^\n]*\btightopt\b", line, re.IGNORECASE):
                offenders.append(
                    f"{py_file.relative_to(REPO_ROOT)}:{idx}: {line.strip()}"
                )
    assert offenders == [], (
        f"Unexpected bare ! TightOpt emission in src/cochem_torq: {offenders}"
    )
