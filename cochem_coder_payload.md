Cycle 10: Implement code for prompt at D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task8_watchdog.md strictly adhering to Zero-Mock mandate. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ. Generate unit tests first. IMPORTANT: You MUST update/create `pytest.ini` to restrict `testpaths` to ONLY the tests you are writing for this prompt, otherwise the global 1500+ test suite will run and crash your context. 
Test Failures from previous run:
Output: ============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: D:\__CoChem\GitHub-Repo\CoChem-TORQ
configfile: pytest.ini
plugins: anyio-4.10.0, typeguard-4.6.0, zarr-3.3.0
collected 189 items

tests\test_adaptive_error.py ................                            [  8%]
tests\test_alignment.py .........................                        [ 21%]
tests\test_engine.py ............................F.......                [ 40%]
tests\test_h5_healer.py .......FF.                                       [ 46%]
tests\test_jax_builder.py ..........                                     [ 51%]
tests\test_tensor_extractor.py ..................                        [ 60%]
tests\test_torq_export.py ..FF......F.....F...                           [ 71%]
tests\test_torq_telemetry.py ...............                             [ 79%]
tests\test_watchdog.py .......................................           [100%]

================================== FAILURES ===================================
__ TestWavefunctionPropagationAndOPI.test_opi_persistent_threading_generator __

self = <test_engine.TestWavefunctionPropagationAndOPI object at 0x000001FCF0C4EAD0>
ethanediol_geometry = (['C', 'C', 'O', 'O', 'H', 'H', ...], array([[-0.732,  0.385,  0.   ],
       [ 0.732, -0.385,  0.   ],
       [-1.432..., -1.44 , -0.22 ],
       [ 1.15 , -0.21 ,  0.99 ],
       [-1.3  , -1.31 ,  0.77 ],
       [ 1.3  ,  1.31 , -0.77 ]]))
tmp_path = WindowsPath('C:/Users/ansac/AppData/Local/Temp/pytest-of-ansac/pytest-9231/test_opi_persistent_threading_0')

    def test_opi_persistent_threading_generator(
        self, ethanediol_geometry: Tuple[List[str], np.ndarray], tmp_path: Path
    ) -> None:
        syms, coords = ethanediol_geometry
        ctx = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
    
        payload = DispatchPayload(
            symbols=syms,
            coordinates=coords,
            charge=0,
            multiplicity=1,
            method="r2SCAN-3c",
        )
    
        # Generate 4-step trajectory
        traj = [coords + (i * 0.005) for i in range(4)]
        generator = opi_persistent_threading(payload, context=ctx, trajectory=traj)
    
>       results: List[ORCAStepResult] = list(generator)
                                        ^^^^^^^^^^^^^^^

tests\test_engine.py:547: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

input_payload = DispatchPayload(symbols=['C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H'], coordinates=array([[-0.732,  0.385,  0.   ...one, counterpoise=False, initial_hessian='XTB2', moinp_path=None, use_moread=False, grid_level='defgrid3', metadata={})
context = ExecutionContext(tier=<EnvironmentTier.LOCAL_WINDOWS: 'LOCAL_WINDOWS'>, custom_scratch_dir=WindowsPath('C:/Users/ansac..., max_memory_mb=65291, num_cores=24, gpu_available=False, vram_mb=0, session_id='bda9cfdd-9d26-44b3-9ef5-d7bb929a721d')
n_steps = 3
trajectory = [array([[-0.732,  0.385,  0.   ],
       [ 0.732, -0.385,  0.   ],
       [-1.432, -0.385,  0.98 ],
       [ 1.432,  0..., -1.425, -0.205],
       [ 1.165, -0.195,  1.005],
       [-1.285, -1.295,  0.785],
       [ 1.315,  1.325, -0.755]])]

    def opi_persistent_threading(
        input_payload: DispatchPayload,
        context: Optional[ExecutionContext] = None,
        n_steps: int = 3,
        trajectory: Optional[List[np.ndarray]] = None
    ) -> Generator[ORCAStepResult, None, None]:
        """
        Interfaces with the ORCA execution engine, yielding ORCAStepResult instances
        across optimization or PES sweep steps with dynamic wavefunction propagation.
        Handles Windows / MPI execution cleanly to prevent exit code 126.
        """
        if context is None:
            context = ExecutionContext()
    
        current_coords = np.copy(input_payload.coordinates)
        steps_to_run = trajectory if trajectory is not None else [current_coords for _ in range(n_steps)]
    
        scratch_dir = context.get_scratch_dir("opi_thread")
        orca_bin = os.environ.get("ORCA_PATH", "orca")
    
        # Determine safe core allocation (avoid MPI error 126 on Windows when MPI is unconfigured)
        safe_n_procs = context.num_cores if is_openmpi_supported() else 1
    
        last_gbw_path: Optional[Path] = None
    
        for idx, step_coords in enumerate(steps_to_run):
            step_payload = input_payload.model_copy(deep=True)
            step_payload.coordinates = step_coords
    
            # Dynamically propagate previous step's wavefunction seed via MOREAD
            if idx > 0 and last_gbw_path and last_gbw_path.exists():
                step_payload.use_moread = True
                step_payload.moinp_path = str(last_gbw_path)
    
            # Append EnGrad if not already present
            if "engrad" not in step_payload.extra_options.lower() and "engrad" not in step_payload.method.lower():
                step_payload.extra_options = f"! EnGrad\n{step_payload.extra_options}".strip()
    
            job_base = scratch_dir / f"opi_step_{idx:04d}_{context.session_id[:8]}"
            inp_path = job_base.with_suffix(".inp")
            out_path = job_base.with_suffix(".out")
            gbw_path = job_base.with_suffix(".gbw")
            engrad_path = job_base.with_suffix(".engrad")
    
            inp_content = step_payload.to_orca_input(
                n_procs=safe_n_procs,
                max_core_mb=max(1000, context.max_memory_mb // max(1, safe_n_procs))
            )
            inp_path.write_text(inp_content, encoding="utf-8")
    
            logger.info(f"[OPI Thread] Executing ORCA step {idx} (n_procs={safe_n_procs}) at {inp_path}")
            try:
                stdout, stderr, ret_code = execute_subprocess_safe(
                    cmd=[orca_bin, str(inp_path)],
                    cwd=scratch_dir,
                    timeout=3600.0
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(stdout)
            except Exception as e:
                logger.error(f"[OPI Thread] ORCA execution failed at step {idx}: {e}")
                raise RuntimeError(f"ORCA execution failed at step {idx}: {e}")
    
            # Parse energy, gradient, convergence
            energy, grad, converged = _parse_orca_engrad_or_output(engrad_path, stdout, len(step_coords))
    
            # Spin observables
            s_ideal, s_obs, s_dev = None, None, None
            if input_payload.multiplicity > 1:
                s2_match = re.search(r"Expectation value of <S\*\*2>\s+:\s+([\d\.]+)", stdout)
                s2_ideal_match = re.search(r"Ideal value s\*\(s\+1\)\s+for\s+S=\S+\s+:\s+([\d\.]+)", stdout)
                if s2_match and s2_ideal_match:
                    s_obs = float(s2_match.group(1))
                    s_ideal, s_obs, s_dev = validate_spin_contamination(input_payload.multiplicity, s_obs)
    
            # Read GBW binary bytes
            gbw_data = None
            if gbw_path.exists():
                gbw_data = gbw_path.read_bytes()
                last_gbw_path = gbw_path
    
            # Generate / extract physical in-memory MO and Fock tensors for OPI threading
            if not gbw_data:
                raise ValueError("Missing physical MO tensor data. Cannot extract MO and Fock tensors without valid GBW data or explicit text output.")
            # [SPOOFING RISK DETECTED]
            # Currently, we lack the parser to extract physical MO/Fock tensors directly from the binary GBW file.
            # Instead of mocking with np.eye / np.diag, we raise an explicit physical error.
>           raise ValueError("Missing physical MO tensor data: unable to parse tensors from GBW file.")
E           ValueError: Missing physical MO tensor data: unable to parse tensors from GBW file.

Libraries\cochem_torq_engine.py:995: ValueError
______________ TestZombieDetection.test_detect_active_local_pid _______________

self = <test_h5_healer.TestZombieDetection object at 0x000001FCF0C4FB10>
tmp_path = WindowsPath('C:/Users/ansac/AppData/Local/Temp/pytest-of-ansac/pytest-9231/test_detect_active_local_pid0')

    def test_detect_active_local_pid(self, tmp_path):
        db_path = tmp_path / "quantum.h5"
        lock_path = get_lock_path(db_path)
    
        # Use our own PID which is guaranteed to be alive
        meta = LockMetadata(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            slurm_job_id=None,
            created_at=time.time(),
            session_id=str(uuid.uuid4())
        )
        lock_path.write_text(meta.model_dump_json(), encoding="utf-8")
    
        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is False
>       assert detected is None
E       AssertionError: assert {'created_at': 1787527533.9221723, 'extra': {}, 'hostname': 'JJDesktop', 'pid': 39952, ...} is None

tests\test_h5_healer.py:162: AssertionError
_______________ TestZombieDetection.test_detect_dead_local_pid ________________

self = <test_h5_healer.TestZombieDetection object at 0x000001FCF0CA0D60>
tmp_path = WindowsPath('C:/Users/ansac/AppData/Local/Temp/pytest-of-ansac/pytest-9231/test_detect_dead_local_pid0')

    def test_detect_dead_local_pid(self, tmp_path):
        db_path = tmp_path / "quantum.h5"
        lock_path = get_lock_path(db_path)
    
        # Create a real process and forcefully terminate it to simulate a crash
        proc = subprocess.Popen([sys.executable, "-c", "from rdkit import Chem; from rdkit.Chem import AllChem; m=Chem.AddHs(Chem.MolFromSmiles('C'*50)); AllChem.EmbedMolecule(m); [AllChem.MMFFOptimizeMolecule(m, maxIters=1000) for _ in range(500)]"])
        dead_pid = proc.pid
        proc.kill()
        proc.wait()
    
        meta = LockMetadata(
            pid=dead_pid,
            hostname=socket.gethostname(),
            slurm_job_id=None,
            created_at=time.time(),
            session_id=str(uuid.uuid4())
        )
        lock_path.write_text(meta.model_dump_json(), encoding="utf-8")
    
        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is True
        assert detected is not None
>       assert "dead" in detected["zombie_reason"].lower()
E       AssertionError: assert 'dead' in 'pid 33196 does not exist in local process table'
E        +  where 'pid 33196 does not exist in local process table' = <built-in method lower of str object at 0x000001FC803A1290>()
E        +    where <built-in method lower of str object at 0x000001FC803A1290> = 'PID 33196 does not exist in local process table'.lower

tests\test_h5_healer.py:186: AssertionError
____________________ test_kraitchman_zpve_defect_clamping _____________________

    def test_kraitchman_zpve_defect_clamping() -> None:
        """Validates that negative radicands (R_g < 0) are clamped to 0.0000."""
        i_a, i_b, i_c = 10.0, 25.0, 30.0
        parent_moments = (i_a, i_b, i_c)
        sub_moments = (i_a + 1.5, i_b + 0.1, i_c + 0.1)
    
        with pytest.warns(
            KraitchmanZPVEWarning,
            match="ZPVE defect produced imaginary substitution coordinate",
        ):
            result = calculate_kraitchman_coords(
                parent_moments=parent_moments,
                substituted_moments=sub_moments,
>               parent_mass=float(element("C").atomic_weight) * 5,
                                  ^^^^^^^
                delta_m=1.003355,
            )
E           NameError: name 'element' is not defined

tests\test_torq_export.py:181: NameError

During handling of the above exception, another exception occurred:

    def test_kraitchman_zpve_defect_clamping() -> None:
        """Validates that negative radicands (R_g < 0) are clamped to 0.0000."""
        i_a, i_b, i_c = 10.0, 25.0, 30.0
        parent_moments = (i_a, i_b, i_c)
        sub_moments = (i_a + 1.5, i_b + 0.1, i_c + 0.1)
    
>       with pytest.warns(
            KraitchmanZPVEWarning,
            match="ZPVE defect produced imaginary substitution coordinate",
        ):
E       Failed: DID NOT WARN. No warnings of type (<class 'Libraries.cochem_torq_export.KraitchmanZPVEWarning'>,) were emitted.
E        Emitted warnings: [].

tests\test_torq_export.py:174: Failed
__________________ test_kraitchman_piecewise_costain_bounds ___________________

    def test_kraitchman_piecewise_costain_bounds() -> None:
        """Validates Piecewise Costain Bounds for large and small coordinates."""
        res = calculate_kraitchman_coords(
            parent_moments=(10.0, 20.0, 25.0),
            substituted_moments=(10.2, 20.4, 25.3),
>           parent_mass=float(element("V").atomic_weight),
                              ^^^^^^^
            delta_m=1.00335,
        )
E       NameError: name 'element' is not defined

tests\test_torq_export.py:197: NameError
________________ test_kraitchman_dictionary_and_planar_inputs _________________

    def test_kraitchman_dictionary_and_planar_inputs() -> None:
        """Validates calculate_kraitchman_coords() with dictionary input structures."""
        parent_dict = {"a": 12.5, "b": 24.0, "c": 36.5}
        sub_dict = {"a": 12.8, "b": 24.4, "c": 36.9}
    
        res = calculate_kraitchman_coords(
            parent_moments=parent_dict,
            substituted_moments=sub_dict,
>           parent_mass=float(element("Se").atomic_weight),
                              ^^^^^^^
            delta_m=1.003355,
        )
E       NameError: name 'element' is not defined

tests\test_torq_export.py:540: NameError
________________ test_kraitchman_exact_zero_denominator_guard _________________

    def test_kraitchman_exact_zero_denominator_guard() -> None:
        """Validates Kraitchman coordinates when moments are identical (Ia == Ib)."""
        parent_moments = (20.0, 20.0, 40.0)
        sub_moments = (20.5, 20.5, 40.8)
    
        with pytest.warns(KraitchmanSingularityWarning):
            res = calculate_kraitchman_coords(
                parent_moments=parent_moments,
                substituted_moments=sub_moments,
>               parent_mass=float(element("V").atomic_weight),
                                  ^^^^^^^
                delta_m=1.0,
                singularity_threshold=1e-4,
            )
E           NameError: name 'element' is not defined

tests\test_torq_export.py:688: NameError

During handling of the above exception, another exception occurred:

    def test_kraitchman_exact_zero_denominator_guard() -> None:
        """Validates Kraitchman coordinates when moments are identical (Ia == Ib)."""
        parent_moments = (20.0, 20.0, 40.0)
        sub_moments = (20.5, 20.5, 40.8)
    
>       with pytest.warns(KraitchmanSingularityWarning):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT WARN. No warnings of type (<class 'Libraries.cochem_torq_export.KraitchmanSingularityWarning'>,) were emitted.
E        Emitted warnings: [].

tests\test_torq_export.py:684: Failed
============================== warnings summary ===============================
tests/test_torq_export.py::test_kraitchman_singularity_guard_damping
  d:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_torq_export.py:154: KraitchmanZPVEWarning: ZPVE defect produced imaginary substitution coordinate for axis a (R_a = -6.310941e+00 < 0). Clamping to 0.0000.
    result = calculate_kraitchman_coords(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_engine.py::TestWavefunctionPropagationAndOPI::test_opi_persistent_threading_generator
FAILED tests/test_h5_healer.py::TestZombieDetection::test_detect_active_local_pid
FAILED tests/test_h5_healer.py::TestZombieDetection::test_detect_dead_local_pid
FAILED tests/test_torq_export.py::test_kraitchman_zpve_defect_clamping - Fail...
FAILED tests/test_torq_export.py::test_kraitchman_piecewise_costain_bounds - ...
FAILED tests/test_torq_export.py::test_kraitchman_dictionary_and_planar_inputs
FAILED tests/test_torq_export.py::test_kraitchman_exact_zero_denominator_guard
============ 7 failed, 182 passed, 1 warning in 117.60s (0:01:57) =============

Error: 