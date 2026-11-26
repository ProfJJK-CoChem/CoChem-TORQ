# Changelog

All notable changes to CoChem-TORQ governance, zero-mock enforcement, and verification gates are documented in this file.

## [Unreleased] - 2026-11-26

### Governance & Anti-Spoof Hardening (`own-1126-84f4`, `swarm-1126-9fc4`)

- **Transparent Disclosure — Stdlib Concurrency Reclassification (Amnesty Burn to `[]`):**
  When burning `.anti_spoof_amnesty.json` from its prior 7-file concurrency whitelist
  (`Libraries/cochem_path_manager.py`, `Libraries/cochem_torq_goat.py`,
  `Libraries/cochem_torq_ipc.py`, `Libraries/cochem_torq_telemetry.py`,
  `tests/test_cochem_torq_init.py`, `tests/test_ipc.py`, `tests/test_torq_telemetry.py`)
  to `[]` in commit `5e975d0`, Python standard-library concurrency modules
  (`threading`, `multiprocessing`, `concurrent.futures`) were reclassified out of
  `BANNED_CONCURRENCY_MODULES` in `ci_tools/anti_spoof_linter.py` because standard-library
  concurrency is not test-double mocking. External distributed schedulers
  (`parsl`, `dask`, `ray`, `mpi4py`, `celery`) remain banned unless explicitly amnestied,
  and CI strictly enforces `.anti_spoof_amnesty.json == []`. Also fixed a residual
  `from concurrent import futures` false-positive branch in `ci_tools/anti_spoof_linter.py`.
- **Widened Indirect-Mock & Runtime-Injection Detection:**
  Extended both `ci_tools/anti_spoof_linter.py` and `tests/test_zero_mock_and_no_sim_gate.py`
  to catch indirect mock aliases (`from unittest import mock as ...`, `import unittest as ut; ut.mock`),
  dynamic/aliased `importlib.import_module("unittest.mock" | "mock" | "pytest_mock" | ...)`
  and `__import__` calls, `sys.modules` mock injection (`sys.modules[...] = ...`,
  `update`, `setdefault`, `__setitem__`, `pop`, `del`), `pytest-mock` / `mocker` / `MockerFixture`,
  HTTP/network mock libraries (`responses`, `vcr`, `vcrpy`, `requests_mock`, `mockito`,
  `flexmock`, `respx`, `httpretty`), and module-level or engine-module `setattr(...)` interception.
- **Seeded-Canary & Concurrency Determinism Verification:**
  Added seeded-canary self-tests verifying all indirect mock bypasses are caught, plus a
  serial-vs-parallel concurrency determinism test (`workers=1` vs `workers=2` SHA-256
  hash-identical outputs on authentic `AsymmetricTopDiagonalizer` workloads with verified
  exception propagation and zero swallowed worker exceptions).
- **Mockup-Leak Media Firewall (`swarm-1126-9fc4`):**
  Added a pure-stdlib PNG (`tEXt`/`zTXt`/`iTXt`) and PDF (XMP + FlateDecode stream) firewall
  in `tests/test_zero_mock_and_no_sim_gate.py` rejecting `cochem:mockup=true` metadata and
  `NOT SCIENTIFIC DATA` watermark strings in tracked images/PDFs, backed by seeded-canary tests.
