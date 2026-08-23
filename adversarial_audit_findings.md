# Adversarial Audit Findings

During the adversarial audit of the CoChem-TORQ codebase, the following violations of the Zero-Mock and Anti-Spoofing directives were discovered. Specifically, there is rampant **semantic spoofing** occurring in the test suite where synthetic data structures (`np.eye`, `np.zeros`, `np.ones`) are used to fake physical tensors.

## Violations Found

1. **`tests/test_engine.py`**:
   - `mo_mat = np.eye(n_basis)` (Line 539)
   - `fock_mat = np.diag(np.linspace(-2.0, 1.0, n_basis))` (Line 540)
   *Context*: Faking molecular orbital and Fock matrices for `ORCAStepResult`.

2. **`tests/test_jax_builder.py`**:
   - `v_pot = np.zeros(n_pts)` (Lines 115, 150)
   - `vpt2_x_matrix = np.zeros((n_modes, n_modes))` (Line 221)
   *Context*: Faking potential energy surfaces and anharmonicity matrices.

3. **`tests/test_tensor_extractor.py`**:
   - `"wavefunction": np.ones((50, 50)).tolist()` (Line 709)
   *Context*: Faking a Sinc-DVR wavefunction payload with an unphysical matrix of ones.

4. **`tests/test_torq_telemetry.py`**:
   - `pes_grid = np.ones((30, 30)) * 250.0` (Line 375)
   - `pes = np.ones((20, 20))` (Line 517)
   *Context*: Faking a 2D Potential Energy Surface grid.

## Additional Checks Performed (Passed)
- No `unittest.mock`, `MagicMock`, `patch`, or `pytest.monkeypatch` found in the codebase.
- No string obfuscation (`base64`) used to hide mocks.
- `NotImplementedError` and dead-end `pass` blocks are non-existent in logic functions.
- The Mendeleev Library Mandate is respected (`mendeleev` is imported, and no hardcoded masses like `1.008` exist).

## Required Action
The Agent Council (`cochem-audit` and `cochem-improve`) must review these findings, confirm the semantic spoofing violations, determine the remedy (e.g., rewriting the tests to use PySCF to generate real physical tensors, or removing the synthetic tests entirely), and document the lesson learned to `d:\__CoChem\.docs\lessons.md`.
