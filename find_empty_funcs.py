import ast
import os

for root, _, files in os.walk('Libraries'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = node.body
                    # check if body is just pass or docstring + pass
                    is_empty = False
                    if len(body) == 1 and isinstance(body[0], ast.Pass):
                        is_empty = True
                    elif len(body) == 2 and isinstance(body[0], ast.Expr) and isinstance(body[1], ast.Pass):
                        is_empty = True
                    if is_empty:
                        print(f"Empty function found: {filepath} -> {node.name}")
