import ast
import os
import sys

def check_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source, filename=filepath)
    violations = []
    
    for node in ast.walk(tree):
        # 1. Detect np.linspace or np.meshgrid or np.zeros
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'np' and node.func.attr in ['linspace', 'meshgrid', 'zeros', 'ones', 'eye', 'random']:
                    violations.append(f"Line {node.lineno}: Banned numpy function np.{node.func.attr} used for synthetic data.")
        
        # 2. Detect large string literal assignments (mocking file outputs)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        # If a string literal assigned to a variable is > 5 lines, it's a mocked file
                        if node.value.value.count('\n') > 5:
                            violations.append(f"Line {node.lineno}: Large inline string assignment '{target.id}' (Mocked file).")
                            
    return violations

def check_orca_fixtures(target_dir):
    violations = []
    fixtures_dir = os.path.join(target_dir, "fixtures")
    if not os.path.exists(fixtures_dir):
        return violations
    for f in os.listdir(fixtures_dir):
        if f.endswith(".out"):
            path = os.path.join(fixtures_dir, f)
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                if "O   R   C   A" not in content or "ORCA TERMINATED NORMALLY" not in content:
                    violations.append(f"{f} is missing standard ORCA markers.")
    return violations

def check_wavefunction_variance(target_dir):
    sys.path.insert(0, target_dir)
    violations = []
    try:
        from test_tensor_extractor import generate_synthetic_wavefunction
        import numpy as np
        wf = generate_synthetic_wavefunction(1.0)
        variance = np.var(wf)
        if variance <= 0:
            violations.append("Wavefunction variance is not > 0.")
    except Exception as e:
        violations.append(f"Failed to check wavefunction variance: {e}")
    finally:
        if target_dir in sys.path:
            sys.path.remove(target_dir)
    return violations

def main():
    target_dir = r"D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests"
    total_violations = 0
    
    # 1. AST Checks
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                violations = check_file(path)
                if violations:
                    print(f"Violations in {path}:")
                    for v in violations:
                        print(f"  - {v}")
                    total_violations += len(violations)
                    
    # 2. Check ORCA fixtures
    orca_violations = check_orca_fixtures(target_dir)
    if orca_violations:
        print("ORCA Fixture Violations:")
        for v in orca_violations:
            print(f"  - {v}")
        total_violations += len(orca_violations)
        
    # 3. Check wavefunction variance
    var_violations = check_wavefunction_variance(target_dir)
    if var_violations:
        print("Wavefunction Variance Violations:")
        for v in var_violations:
            print(f"  - {v}")
        total_violations += len(var_violations)
    
    if total_violations > 0:
        sys.exit(1)
    else:
        print("Validation Passed: No inline data mocking detected. Files and test data are physically valid.")
        sys.exit(0)

if __name__ == '__main__':
    main()
