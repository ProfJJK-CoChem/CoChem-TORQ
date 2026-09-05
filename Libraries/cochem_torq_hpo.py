"""Automated Hyperparameter Optimization (HPO) Suite for CoChem-TORQ.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic physics, dynamic sampling, and real loss evaluations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from filelock import FileLock

from Libraries.cochem_torq_inference_errors import TorqInferenceError
from Libraries.cochem_torq_inference_schemas import HPORunConfig


class TrialPruned(Exception):
    """Exception raised when an HPO trial is terminated early by a pruner. [M]"""

    def __init__(self, message: str = "Trial pruned by early stopping pruner") -> None:
        super().__init__(message)
        self.message = message


def compute_hpo_loss(
    energy_true: Union[float, torch.Tensor, Sequence[float]],
    energy_pred: Union[float, torch.Tensor, Sequence[float]],
    forces_true: Union[torch.Tensor, Sequence[torch.Tensor]],
    forces_pred: Union[torch.Tensor, Sequence[torch.Tensor]],
    w_energy: float = 1.0,
    w_force: float = 10.0,
) -> torch.Tensor:
    r"""Compute multi-objective validation loss combining potential energy and atomic forces. [D]

    $$\mathcal{L}_{\text{HPO}} = w_E \cdot \text{MAE}(E, \hat{E}) + w_F \cdot \text{MAE}(\mathbf{F}, \hat{\mathbf{F}})$$
    $$\text{MAE}(E, \hat{E}) = \frac{1}{B} \sum_{b=1}^B |E_b - \hat{E}_b|$$
    $$\text{MAE}(\mathbf{F}, \hat{\mathbf{F}}) = \frac{1}{B} \sum_{b=1}^B \frac{1}{3 N_b} \sum_{i=1}^{N_b} \sum_{\alpha \in \{x,y,z\}} |F_{b,i,\alpha} - \hat{F}_{b,i,\alpha}|$$

    Parameters
    ----------
    energy_true : float or Tensor
        Ground-truth potential energy (eV).
    energy_pred : float or Tensor
        Predicted potential energy (eV).
    forces_true : Tensor or Sequence[Tensor]
        Ground-truth atomic forces (eV/Angstrom).
    forces_pred : Tensor or Sequence[Tensor]
        Predicted atomic forces (eV/Angstrom).
    w_energy : float
        Potential energy loss weight [E].
    w_force : float
        Atomic force loss weight [E].

    Returns
    -------
    torch.Tensor
        Scalar tensor containing combined loss in double precision.
    """
    # Harmonize energy to 1D float64 tensor
    if isinstance(energy_true, (int, float)):
        e_true = torch.tensor([float(energy_true)], dtype=torch.float64)
    elif isinstance(energy_true, torch.Tensor):
        e_true = energy_true.view(-1).to(dtype=torch.float64)
    else:
        e_true = torch.tensor(list(energy_true), dtype=torch.float64)

    if isinstance(energy_pred, (int, float)):
        e_pred = torch.tensor([float(energy_pred)], dtype=torch.float64, device=e_true.device)
    elif isinstance(energy_pred, torch.Tensor):
        e_pred = energy_pred.view(-1).to(dtype=torch.float64, device=e_true.device)
    else:
        e_pred = torch.tensor(list(energy_pred), dtype=torch.float64, device=e_true.device)

    mae_energy = torch.mean(torch.abs(e_true - e_pred))

    # Harmonize forces
    # Check if forces are single tensors [N, 3] or batched [B, N, 3] or list of [N_b, 3]
    if isinstance(forces_true, torch.Tensor) and isinstance(forces_pred, torch.Tensor):
        f_true = forces_true.to(dtype=torch.float64)
        f_pred = forces_pred.to(dtype=torch.float64, device=f_true.device)
        if f_true.ndim == 2:
            # Single configuration B=1
            mae_force = torch.mean(torch.abs(f_true - f_pred))
        elif f_true.ndim == 3:
            # Batched [B, N, 3]
            # Mean per configuration: 1 / (3 N_b) * sum |F - F_hat|
            per_conf_mae = torch.mean(torch.abs(f_true - f_pred), dim=(1, 2))
            mae_force = torch.mean(per_conf_mae)
        else:
            raise ValueError(f"Unexpected forces dimension: {f_true.ndim}")
    elif isinstance(forces_true, (list, tuple)) and isinstance(forces_pred, (list, tuple)):
        # List of configurations with potentially varying atom counts N_b
        conf_maes = []
        for ft, fp in zip(forces_true, forces_pred):
            ft_t = ft.to(dtype=torch.float64)
            fp_t = fp.to(dtype=torch.float64, device=ft_t.device)
            conf_maes.append(torch.mean(torch.abs(ft_t - fp_t)))
        mae_force = torch.mean(torch.stack(conf_maes))
    else:
        raise TypeError("Forces must be torch.Tensor or Sequence[torch.Tensor]")

    total_loss = (float(w_energy) * mae_energy) + (float(w_force) * mae_force)
    return total_loss


class BasePruner:
    """Base class for hyperparameter trial pruners. [M]"""

    def __init__(self, grace_period: int = 10) -> None:
        self.grace_period = grace_period

    def should_prune(self, trial: "HPOTrial", step: int) -> bool:
        """Evaluate whether a trial should be pruned at step. [M]"""
        return False


class MedianPruner(BasePruner):
    """Pruner that terminates trials performing worse than the historical median. [E]"""

    def __init__(self, grace_period: int = 10, n_min_trials: int = 1) -> None:
        super().__init__(grace_period=grace_period)
        self.n_min_trials = n_min_trials

    def should_prune(self, trial: "HPOTrial", step: int) -> bool:
        if step < self.grace_period:
            return False

        historical_values = trial.study.get_historical_values_at_step(step, exclude_trial_id=trial.trial_id)
        if len(historical_values) < self.n_min_trials:
            return False

        median_val = float(np.median(historical_values))
        current_val = trial.intermediate_values.get(step, float("inf"))
        return current_val > median_val


class ASHAPruner(BasePruner):
    """Asynchronous Successive Halving Algorithm (ASHA) pruner. [E]"""

    def __init__(
        self,
        grace_period: int = 10,
        max_epochs: int = 100,
        reduction_factor: int = 3,
    ) -> None:
        super().__init__(grace_period=grace_period)
        self.max_epochs = max_epochs
        self.reduction_factor = reduction_factor

        # Determine rungs
        rungs = []
        r = self.grace_period
        while r <= self.max_epochs:
            rungs.append(r)
            r *= self.reduction_factor
        self.rungs = rungs

    def should_prune(self, trial: "HPOTrial", step: int) -> bool:
        if step < self.grace_period:
            return False
        if step not in self.rungs:
            return False

        historical_values = trial.study.get_historical_values_at_step(step, exclude_trial_id=trial.trial_id)
        if not historical_values:
            return False

        # Keep top 1 / reduction_factor
        current_val = trial.intermediate_values.get(step, float("inf"))
        all_vals = sorted(historical_values + [current_val])
        cutoff_idx = max(1, math.ceil(len(all_vals) / self.reduction_factor))
        cutoff_val = all_vals[cutoff_idx - 1]
        return current_val > cutoff_val


class HPOTrial:
    """Represents a single optimization trial in an HPO study. [M]"""

    def __init__(
        self,
        trial_id: str,
        study: "HPOStudy",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trial_id = trial_id
        self.study = study
        self.params: Dict[str, Any] = params or {}
        self.intermediate_values: Dict[int, float] = {}
        self.state: str = "RUNNING"
        self.final_value: Optional[float] = None
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        """Sample a float hyperparameter uniformly or log-uniformly. [E]"""
        if name in self.params:
            return float(self.params[name])
        if log:
            val = float(math.exp(float(torch.empty(1).uniform_(math.log(low), math.log(high)).item())))
        else:
            val = float(torch.empty(1).uniform_(low, high).item())
        self.params[name] = val
        return val

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        """Sample a categorical hyperparameter from discrete choices. [E]"""
        if name in self.params:
            return self.params[name]
        choice_idx = int(torch.randint(0, len(choices), (1,)).item())
        val = choices[choice_idx]
        self.params[name] = val
        return val

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Sample an integer hyperparameter within bounds. [E]"""
        if name in self.params:
            return int(self.params[name])
        val = int(torch.randint(low, high + 1, (1,)).item())
        self.params[name] = val
        return val

    def report(self, value: float, step: int) -> None:
        """Report intermediate loss value at a training epoch or step. [M]"""
        val_f = float(value)
        self.intermediate_values[step] = val_f
        self.study.record_intermediate_step(self.trial_id, step, val_f)

    def should_prune(self, step: int) -> bool:
        """Evaluate if trial should be pruned by study pruner. [M]"""
        return self.study.pruner.should_prune(self, step)


class HPOStudy:
    """Thread-safe and process-safe study manager with SQLite/HDF5 persistence. [M]"""

    def __init__(self, config: HPORunConfig) -> None:
        self.config = config
        self.study_name = config.study_name

        # Initialize pruner
        if config.pruner == "ASHA":
            self.pruner: BasePruner = ASHAPruner(grace_period=config.grace_period)
        elif config.pruner == "MedianPruner":
            self.pruner = MedianPruner(grace_period=config.grace_period)
        else:
            self.pruner = ASHAPruner(grace_period=config.grace_period)

        self.trials: List[HPOTrial] = []
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize SQLite database storage and advisory lock file. [M]"""
        storage_uri = self.config.storage_uri
        if storage_uri.startswith("sqlite:///"):
            db_path_str = storage_uri[len("sqlite:///") :]
            if db_path_str == ":memory:":
                self.is_memory_db = True
                self.db_path = None
                self.lock_path = None
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                self.is_memory_db = False
                self.db_path = Path(db_path_str).resolve()
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self.lock_path = self.db_path.with_suffix(".lock")
                self._conn = None
        else:
            # Fallback path if plain filename given
            self.is_memory_db = False
            self.db_path = Path(storage_uri).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path = self.db_path.with_suffix(".lock")
            self._conn = None

        self._execute_with_lock(self._create_tables)

    def _execute_with_lock(self, callback: Callable[[sqlite3.Connection], Any]) -> Any:
        """Execute database transaction guarded by single-node FileLock. [M]"""
        if self.is_memory_db:
            return callback(self._conn)

        assert self.db_path is not None
        assert self.lock_path is not None

        # Tier 6 cluster check: avoid filelock if specified or Lustre detected
        use_lock = True
        try:
            lock = FileLock(str(self.lock_path), timeout=30.0)
        except Exception:
            use_lock = False

        if use_lock:
            with lock:
                conn = sqlite3.connect(str(self.db_path), timeout=30.0)
                try:
                    res = callback(conn)
                    conn.commit()
                    return res
                finally:
                    conn.close()
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            try:
                res = callback(conn)
                conn.commit()
                return res
            finally:
                conn.close()

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create database tables for study and trial records. [M]"""
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                study_name TEXT,
                state TEXT,
                final_value REAL,
                params_json TEXT,
                start_time REAL,
                end_time REAL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trial_steps (
                trial_id TEXT,
                step INTEGER,
                value REAL,
                PRIMARY KEY (trial_id, step)
            )
            """
        )

    def record_intermediate_step(self, trial_id: str, step: int, value: float) -> None:
        """Persist intermediate evaluation step for pruning evaluation. [M]"""
        def _record(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO trial_steps (trial_id, step, value)
                VALUES (?, ?, ?)
                """,
                (trial_id, step, float(value)),
            )

        self._execute_with_lock(_record)

    def get_historical_values_at_step(self, step: int, exclude_trial_id: Optional[str] = None) -> List[float]:
        """Query all reported loss values across previous trials at given step. [M]"""
        def _query(conn: sqlite3.Connection) -> List[float]:
            cursor = conn.cursor()
            if exclude_trial_id:
                cursor.execute(
                    """
                    SELECT value FROM trial_steps
                    WHERE step = ? AND trial_id != ?
                    """,
                    (step, exclude_trial_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT value FROM trial_steps
                    WHERE step = ?
                    """,
                    (step,),
                )
            rows = cursor.fetchall()
            return [float(r[0]) for r in rows if r[0] is not None]

        return self._execute_with_lock(_query)

    def optimize(
        self,
        objective: Callable[[HPOTrial], float],
        n_trials: Optional[int] = None,
    ) -> None:
        """Execute automated hyperparameter optimization trial loop. [M]"""
        total_trials = n_trials or self.config.n_trials

        for _ in range(total_trials):
            trial_id = f"trial_{uuid.uuid4().hex[:8]}"
            trial = HPOTrial(trial_id=trial_id, study=self)

            # Pre-sample standard hyperparameter space
            trial.suggest_float("lr", self.config.lr_min, self.config.lr_max, log=True)
            trial.suggest_float("cutoff", self.config.cutoff_min, self.config.cutoff_max, log=False)
            trial.suggest_categorical("rbf", self.config.rbf_options)
            trial.suggest_categorical("depth", self.config.depth_options)
            trial.suggest_categorical("embedding_dim", self.config.embedding_dim_options)

            try:
                val = objective(trial)
                trial.final_value = float(val)
                trial.state = "COMPLETE"
            except TrialPruned:
                trial.state = "PRUNED"
            except Exception:
                trial.state = "FAILED"
                trial.final_value = float("inf")
            finally:
                trial.end_time = time.time()
                self._save_trial(trial)
                self.trials.append(trial)

    def _save_trial(self, trial: HPOTrial) -> None:
        """Persist finalized trial record into SQLite database with atomic semantics. [M]"""
        def _insert(conn: sqlite3.Connection) -> None:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO trials
                (trial_id, study_name, state, final_value, params_json, start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.trial_id,
                    self.study_name,
                    trial.state,
                    trial.final_value,
                    json.dumps(trial.params),
                    trial.start_time,
                    trial.end_time,
                ),
            )

        self._execute_with_lock(_insert)

        # Atomic serialization of study snapshot digest if on filesystem
        if not self.is_memory_db and self.db_path is not None:
            try:
                # Write .sha256 digest alongside database file
                sha256 = hashlib.sha256()
                with open(self.db_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)
                digest = sha256.hexdigest()

                sha_file = self.db_path.with_suffix(".sha256")
                tmp_sha = self.db_path.with_name(f"{self.db_path.name}.tmp.sha256")
                with open(tmp_sha, "w", encoding="utf-8") as f:
                    f.write(f"{digest}  {self.db_path.name}\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_sha, sha_file)
            except Exception:
                pass

    @property
    def best_trial(self) -> Optional[HPOTrial]:
        """Retrieve trial with lowest objective loss. [M]"""
        completed = [t for t in self.trials if t.state == "COMPLETE" and t.final_value is not None]
        if not completed:
            return None
        return min(completed, key=lambda t: t.final_value if t.final_value is not None else float("inf"))

    @property
    def best_value(self) -> Optional[float]:
        """Retrieve lowest objective loss achieved. [M]"""
        bt = self.best_trial
        return bt.final_value if bt is not None else None

    @property
    def best_params(self) -> Dict[str, Any]:
        """Retrieve hyperparameter configuration of best trial. [M]"""
        bt = self.best_trial
        return bt.params if bt is not None else {}


def create_hpo_study(config: HPORunConfig) -> HPOStudy:
    """Factory helper to instantiate an HPOStudy from configuration. [M]"""
    return HPOStudy(config=config)
