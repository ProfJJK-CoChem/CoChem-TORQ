import os
import re

banned_patterns = {
    'semantic_spoofing': re.compile(r'(np\.eye|np\.zeros|np\.ones)')
}

def search_files(directory):
    for root, _, files in os.walk(directory):
        if '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py') and file.startswith('test_'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            for cat, pat in banned_patterns.items():
                                if pat.search(line):
                                    print(f"[{cat}] {filepath}:{i+1} : {line.strip()}")
                except Exception as e:
                    pass

search_files('D:/__CoChem/GitHub-Repo/CoChem-TORQ/tests')
