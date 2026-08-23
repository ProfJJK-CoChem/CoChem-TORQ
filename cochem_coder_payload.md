Cycle 3: Implement code for prompt at D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task10_jax_builder.md strictly adhering to Zero-Mock mandate. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ. Generate unit tests first. IMPORTANT: You MUST update/create `pytest.ini` to restrict `testpaths` to ONLY the tests you are writing for this prompt, otherwise the global 1500+ test suite will run and crash your context. 
Test Failures from previous run:
Output: ============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: D:\__CoChem\GitHub-Repo\CoChem-TORQ
configfile: pytest.ini
plugins: anyio-4.10.0, typeguard-4.6.0
collected 19 items / 1 error

=================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_mpqc.py _____________________
C:\Users\ansac\anaconda3\Lib\importlib\__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
C:\Users\ansac\anaconda3\Lib\site-packages\_pytest\assertion\rewrite.py:186: in exec_module
    exec(co, module.__dict__)
tests\test_mpqc.py:10: in <module>
    from Libraries.cochem_torq_neb import run_ts_optimization, _run_irc_validation, compute_kabsch_rmsd
Libraries\cochem_torq_neb.py:11: in <module>
    ARTIFACTS_DIR = os.environ.get('COCHEM_ARTIFACTS_DIR', str(Path.home() / 'cochem_artifacts'))
                    ^^
E   NameError: name 'os' is not defined. Did you forget to import 'os'?
=========================== short test summary info ===========================
ERROR tests/test_mpqc.py - NameError: name 'os' is not defined. Did you forge...
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 1.66s ===============================

Error: 