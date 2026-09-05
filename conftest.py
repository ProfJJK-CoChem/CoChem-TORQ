import os, sys
def pytest_configure(config):
    if "cochem_exec_" not in os.getcwd() and os.environ.get("COCHEM_DISABLE_SANDBOX_CHECK") != "1":
        raise RuntimeError("Tests must be run in the CoChem sandbox environment.")

import sys
from pathlib import Path

torq_root: Path = Path(__file__).resolve().parent
if str(torq_root) not in sys.path:
    sys.path.insert(0, str(torq_root))

