import sys
from pathlib import Path

torq_root: Path = Path(__file__).resolve().parent
if str(torq_root) not in sys.path:
    sys.path.insert(0, str(torq_root))
