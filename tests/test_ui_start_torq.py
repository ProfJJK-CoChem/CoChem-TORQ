"""Execute the actual lightweight Codespaces launcher without a QM engine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "UI" / "Start_TORQ.ipynb"


def test_student_notebook_is_valid_and_clear_about_execution_roles() -> None:
    import nbformat

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "Codespaces" in source and "GitHub Actions" in source
    assert "launch_student_app" in source
    assert all(
        not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code"
    )


def test_actual_notebook_code_executes_headless_without_calculation(
    tmp_path: Path,
) -> None:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code"
    )
    script = tmp_path / "launch_actual_notebook.py"
    script.write_text(
        code
        + "\nassert app.student_session.request is None\n"
        "assert app.student_session.submission is None\napp.close()\n",
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]
        ),
        MPLBACKEND="Agg",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
