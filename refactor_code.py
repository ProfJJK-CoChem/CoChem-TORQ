import os
import re
from pathlib import Path

lib_dir = Path(r"D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries")

def refactor_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. Add ARTIFACTS_DIR if missing
    if "ARTIFACTS_DIR =" not in content:
        import_pathlib = "from pathlib import Path"
        if import_pathlib not in content:
            content = content.replace("import os", "import os\nfrom pathlib import Path")
        
        # insert ARTIFACTS_DIR after imports
        content = re.sub(
            r"(import logging\n(logger = [^\n]+\n)?)",
            r"\1\nARTIFACTS_DIR = os.environ.get('COCHEM_ARTIFACTS_DIR', str(Path.home() / 'cochem_artifacts'))\n",
            content,
            count=1
        )
        # if not found, put it near the top
        if "ARTIFACTS_DIR =" not in content:
            content = re.sub(r"(from pathlib import Path\n)", r"\1\nARTIFACTS_DIR = os.environ.get('COCHEM_ARTIFACTS_DIR', str(Path.home() / 'cochem_artifacts'))\n", content, count=1)

    # 2. Fix output_dir = "." and os.path.join
    content = re.sub(r'output_dir:\s*str\s*=\s*"\."', 'output_dir: str | None = None', content)
    
    # Replace os.makedirs(output_dir...) with Path logic
    path_logic = """    if not output_dir or output_dir == ".":
            output_dir = ARTIFACTS_DIR
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)"""
        
    content = re.sub(
        r'([ \t]+)os\.makedirs\(output_dir, exist_ok=True\)',
        lambda m: path_logic.replace("    ", m.group(1)),
        content
    )

    # Replace os.path.join(output_dir, ...) with str(out_path / ...)
    content = re.sub(r'os\.path\.join\(output_dir,\s*(f"[^"]+")\)', r'str(out_path / \1)', content)

    # 3. Dynamic grid tightening for ORCA
    if "cochem_torq_orca.py" in filepath.name:
        grid_logic = """
        # Enforce Method Matrix: dynamic grid tightening
        if "opt" in method_line.lower() or "opt" in final_extra.lower():
            if "defgrid1" in method_line.lower() or "defgrid1" in final_extra.lower():
                if "defgrid3" not in method_line.lower() and "defgrid3" not in final_extra.lower():
                    final_extra += "\\n! defgrid3\\n%geom AutoGrid true end\\n"
        
        method_parts = []"""
        content = content.replace("method_parts = []", grid_logic)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Refactored {filepath.name}")

for p in lib_dir.rglob("*.py"):
    refactor_file(p)
