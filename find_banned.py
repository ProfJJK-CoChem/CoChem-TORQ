import os
import re

BANNED_WORDS = ["mock", "stub", "dummy", "placeholder", "fake", "sample", "# TODO"]

def search_banned_words(directory):
    for root, _, files in os.walk(directory):
        if '.git' in root or '__pycache__' in root or '.pytest_cache' in root or '.ruff_cache' in root:
            continue
        for file in files:
            if not file.endswith('.py'):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for word in BANNED_WORDS:
                        if re.search(r'\b' + re.escape(word) + r'\b', content, re.IGNORECASE) or word == "# TODO" and "# TODO" in content:
                            print(f"BANNED WORD '{word}' found in {path}")
            except Exception as e:
                pass

search_banned_words('.')
