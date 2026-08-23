import re
from pathlib import Path

lib_dir = Path(r"D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries")

for p in lib_dir.rglob("*.py"):
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix indentation
    bad_indent = """        if not output_dir or output_dir == ".":
                        output_dir = ARTIFACTS_DIR
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)"""
    good_indent = """        if not output_dir or output_dir == ".":
            output_dir = ARTIFACTS_DIR
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)"""
    
    content = content.replace(bad_indent, good_indent)

    # Fix method_line in cochem_torq_orca.py
    if "cochem_torq_orca.py" in p.name:
        content = content.replace(
            'if "opt" in method_line.lower() or "opt" in final_extra.lower():',
            'if "opt" in (method or "").lower() or "opt" in final_extra.lower():'
        )
        content = content.replace(
            'if "defgrid1" in method_line.lower() or "defgrid1" in final_extra.lower():',
            'if "defgrid1" in (method or "").lower() or "defgrid1" in final_extra.lower():'
        )
        content = content.replace(
            'if "defgrid3" not in method_line.lower() and "defgrid3" not in final_extra.lower():',
            'if "defgrid3" not in (method or "").lower() and "defgrid3" not in final_extra.lower():'
        )

    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
