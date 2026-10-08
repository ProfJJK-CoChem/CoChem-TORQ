"""Consumer contract tests against retained genuine xTB producer output.

The fixture is a BASE-approved TOPOS 0.1.0 installed-wheel water optimization,
xTB 6.7.1, E=-5.070544054679 Eh. These tests do not re-run that engine or TORQ.
The separate installed-wheel acceptance runs the actual consumer round trip.
"""

import hashlib
import json
from pathlib import Path

import pytest

from Libraries import cochem_torq_topos_handoff as receiver

FIXTURE = Path(__file__).parent / "fixtures" / "topos_native_xtb_reviewed_water.json"
HISTORICAL_FIXTURE = FIXTURE.with_name(
    "topos_native_xtb_reviewed_water_legacy_pre_abcluster.json"
)


def test_current_fixture_retains_exact_authentic_installed_acceptance_bytes():
    from topos.review import verify_torq_handoff

    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == (
        "d76512c2c41cc796842db7ae0d9458078f74f8bd158f01bf6216e1075f40795c"
    )
    handoff = verify_torq_handoff(FIXTURE)
    assert handoff["run_id"] == "run_719d0bbddef742aeba87068878d989cb"
    assert handoff["handoff_sha256"] == (
        "e242be884e7a1dd31b620a497a0701944e9229a1121bae1c7106d683f5cc36a5"
    )
    assert handoff["record"]["request"]["abcluster_options"] is None
    assert handoff["members"][0]["candidate"]["energy_hartree"] == -5.070544054679


def test_historical_fixture_is_preserved_and_rejected_by_current_record_profile(
    tmp_path,
):
    assert hashlib.sha256(HISTORICAL_FIXTURE.read_bytes()).hexdigest() == (
        "6953ebda7161a359b6b7a9196948f3b77449105147798e913f63af54ee0f9075"
    )
    historical = json.loads(HISTORICAL_FIXTURE.read_text())
    assert "abcluster_options" not in historical["record"]["request"]
    with pytest.raises(ValueError, match="Reviewed ensemble/record identity mismatch"):
        receiver.consume_topos_handoff(HISTORICAL_FIXTURE, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_consumer_preserves_exact_members_and_acknowledges_only_import(tmp_path):
    source = json.loads(FIXTURE.read_text())
    imported = receiver.consume_topos_handoff(FIXTURE, tmp_path / "import")
    assert imported["ensemble"]["members"] == source["members"]
    assert imported["ensemble"]["computation_performed"] is False
    assert imported["ensemble"]["status"] == "imported-awaiting-calculation"
    assert imported["receipt"]["handoff_sha256"] == source["handoff_sha256"]
    assert imported == receiver.load_imported_ensemble(tmp_path / "import")
    assert imported == receiver.consume_topos_handoff(FIXTURE, tmp_path / "import")


@pytest.mark.parametrize("field", ["schema_version", "geometry", "ensemble_sha256"])
def test_consumer_rejects_altered_source_before_acknowledgment(tmp_path, field):
    source = json.loads(FIXTURE.read_text())
    if field == "geometry":
        source["members"][0]["molecule"]["coordinates"][0][0] += 1.0
    else:
        source[field] = "invalid"
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        receiver.consume_topos_handoff(path, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_consumer_detects_modified_durable_member(tmp_path):
    target = tmp_path / "import"
    receiver.consume_topos_handoff(FIXTURE, target)
    (target / "ensemble.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        receiver.load_imported_ensemble(target)
    with pytest.raises(ValueError, match="checksum"):
        receiver.consume_topos_handoff(FIXTURE, target)


def test_interrupted_commit_has_no_acknowledged_import(tmp_path, monkeypatch):
    def interrupted(*args):
        raise OSError("controlled rename interruption")

    monkeypatch.setattr(receiver.os, "rename", interrupted)
    with pytest.raises(OSError, match="interruption"):
        receiver.consume_topos_handoff(FIXTURE, tmp_path / "import")
    assert not (tmp_path / "import").exists()
    assert not list(tmp_path.glob(".import.pending-*"))


def test_symlinked_handoff_is_refused(tmp_path):
    source = tmp_path / "link.json"
    source.symlink_to(FIXTURE)
    with pytest.raises(ValueError, match="symlink"):
        receiver.consume_topos_handoff(source, tmp_path / "import")
