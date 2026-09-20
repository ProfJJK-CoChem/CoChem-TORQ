"""CoChem-TORQ Package."""

import sys
from pathlib import Path

# Provide access to the local Libraries module if running from source tree.
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from Libraries.cochem_torq_init import bootstrap_environment  # noqa: E402

# IMPLEMENTATION TASK 1.1: Implement SHA-256 hash validation for fit_provenance.json in TORQ initialization
bootstrap_environment(validate_provenance=True)
