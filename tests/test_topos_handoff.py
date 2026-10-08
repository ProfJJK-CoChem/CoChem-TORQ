"""Consumer contract tests against retained genuine xTB producer output.

The fixture is a BASE-approved TOPOS 0.1.0 installed-wheel water optimization,
xTB 6.7.1, E=-5.070544054679 Eh. These tests do not re-run that engine or TORQ.
The separate installed-wheel acceptance runs the actual consumer round trip.
"""

import errno
import hashlib
import json
import os
import resource
import signal
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path

import pytest

from ci_tools.setup_reviewed_topos import _execution_environment, verify_profile
from Libraries import cochem_torq_topos_handoff as receiver

FIXTURE = Path(__file__).parent / "fixtures" / "topos_native_xtb_reviewed_water.json"
HISTORICAL_FIXTURE = FIXTURE.with_name(
    "topos_native_xtb_reviewed_water_legacy_pre_abcluster.json"
)


def _isolated_reviewed_topos(function):
    """Run every original assertion using the exact separate producer wheel."""

    @wraps(function)
    def execute(*args, **kwargs):
        repository = Path(__file__).resolve().parents[1]
        python = verify_profile(repository)
        if os.environ.get("COCHEM_REVIEWED_TOPOS_CHILD") == "1":
            assert Path(os.sys.executable) == python
            return function(*args, **kwargs)
        node = os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]
        environment = _execution_environment()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(repository / "src"), str(repository)]
        )
        environment["COCHEM_REVIEWED_TOPOS_CHILD"] = "1"
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        environment.pop("PYTEST_ADDOPTS", None)
        with tempfile.TemporaryDirectory(prefix="cochem_exec_reviewed_topos_") as work:
            report = Path(work) / "results.xml"
            completed = subprocess.run(
                [
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                    f"--junitxml={report}",
                    node,
                ],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            cases = ET.parse(report).getroot().findall(".//testcase")
            assert len(cases) == 1
            assert not any(
                case.find(kind) is not None
                for case in cases
                for kind in ("skipped", "failure", "error")
            )

    return execute


@_isolated_reviewed_topos
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


@_isolated_reviewed_topos
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


@_isolated_reviewed_topos
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
@_isolated_reviewed_topos
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


@_isolated_reviewed_topos
def test_consumer_detects_modified_durable_member(tmp_path):
    target = tmp_path / "import"
    receiver.consume_topos_handoff(FIXTURE, target)
    (target / "ensemble.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        receiver.load_imported_ensemble(target)
    with pytest.raises(ValueError, match="checksum"):
        receiver.consume_topos_handoff(FIXTURE, target)


@_isolated_reviewed_topos
def test_interrupted_commit_has_no_acknowledged_import(tmp_path):
    from topos.review import verify_torq_handoff

    # Populate genuine element-database caches before limiting transaction writes;
    # SQLite may otherwise require a temporary sort file during first validation.
    assert verify_torq_handoff(FIXTURE)["run_id"] == (
        "run_719d0bbddef742aeba87068878d989cb"
    )
    # A real kernel file-size limit interrupts the actual staged write. This
    # single-test subprocess restores its process limits and signal disposition.
    previous_limits = resource.getrlimit(resource.RLIMIT_FSIZE)
    previous_signal = signal.getsignal(signal.SIGXFSZ)
    try:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (128, previous_limits[1]))
        with pytest.raises(OSError) as interrupted:
            receiver.consume_topos_handoff(FIXTURE, tmp_path / "import")
        assert interrupted.value.errno == errno.EFBIG
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, previous_limits)
        signal.signal(signal.SIGXFSZ, previous_signal)
    assert not (tmp_path / "import").exists()
    assert not list(tmp_path.glob(".import.pending-*"))


@_isolated_reviewed_topos
def test_symlinked_handoff_is_refused(tmp_path):
    source = tmp_path / "link.json"
    source.symlink_to(FIXTURE)
    with pytest.raises(ValueError, match="symlink"):
        receiver.consume_topos_handoff(source, tmp_path / "import")
