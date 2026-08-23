import os
import re

directories_to_scan = ['.', './Libraries', './tests', './HPC_Launchers', './UI']
patterns = {
    'mock': r'(?i)mock',
    'stub': r'(?i)stub',
    'dummy': r'(?i)dummy',
    'placeholder': r'(?i)placeholder',
    'NotImplementedError': r'NotImplementedError',
    'pass': r'\bpass\b',
    'monkeypatch': r'monkeypatch',
    'base64': r'base64',
    'mendeleev': r'mendeleev',
    'atomic_mass': r'(?i)mass\s*=\s*[\d\.]+',
}

results = {k: [] for k in patterns}

for root, dirs, files in os.walk('.'):
    # Skip git and other non-source dirs
    if any(d in root for d in ['.git', '__pycache__', '.pytest_cache', 'cochem_setup', '.old_plan_docs', '.mypy_cache', '.ruff_cache']):
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        for k, v in patterns.items():
                            if re.search(v, line):
                                results[k].append(f"{filepath}:{i+1}:{line.strip()}")
            except Exception as e:
                pass

with open('audit_results.txt', 'w', encoding='utf-8') as f:
    for k, v in results.items():
        f.write(f"--- {k} ---\n")
        for match in v:
            f.write(f"{match}\n")
        f.write(f"Total: {len(v)}\n\n")
