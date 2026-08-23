import ast
import os
import sys

def check_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = node.body
                # Check for just pass or docstring + pass
                is_empty = False
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    is_empty = True
                elif len(body) == 2 and isinstance(body[0], ast.Expr) and isinstance(body[1], ast.Pass):
                    is_empty = True
                elif len(body) == 1 and isinstance(body[0], ast.Raise) and hasattr(body[0].exc, 'id') and body[0].exc.id == 'NotImplementedError':
                    is_empty = True
                
                if is_empty:
                    print(f'{filepath}:{node.lineno} Empty or stub {type(node).__name__}: {node.name}')
    except Exception as e:
        pass

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            check_file(os.path.join(root, file))
