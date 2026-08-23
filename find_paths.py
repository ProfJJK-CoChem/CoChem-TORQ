import os
import re

PATTERNS = [
    r'["\'][a-zA-Z]:\\[^"\']*["\']',
    r'["\']/tmp/[^"\']*["\']',
    r'["\']/usr/[^"\']*["\']',
    r'["\']/home/[^"\']*["\']',
]

for root, _, files in os.walk('Libraries'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    for p in PATTERNS:
                        if re.search(p, line):
                            print(f"HARDCODED PATH found in {path}:{i+1} -> {line.strip()}")
