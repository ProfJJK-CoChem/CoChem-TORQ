"""Replica Exchange Molecular Dynamics (REMD) Engine with BAOAB Langevin Thermostat (Chunk 21).

Method Matrix v4 Provenance Tags:
- [M] Mandated: Strict double-precision enforcement, rolling divergence guard (acceptance < 5% or |Delta T| > 50 K).
- [D] Derived: BAOAB Langevin thermostat splitting, geometric temperature scheduling,
               Metropolis swap probability, explicit velocity rescaling to prevent thermal inversion.
- [E] Empirical: Alternating odd-even swap schedule calibrated for rapid phase-space mixing.

Strict Zero-Mock Mandate v3: Completely authentic physics and mathematical rigor.
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any, Callable, Deque, List, Optional, Tuple, Union
import torch

from Libraries.cochem_torq_md_errors import ReplicaExchangeDivergenceError
from Libraries.cochem_torq_md_schemas import ExchangeLog, REMDConfig
from Libraries.cochem_torq_symplectic import (
    BOLTZMANN_CONSTANT,
    KAPPA_ACC,
    compute_conservative_forces,
    compute_dimensional_acceleration,
    compute_instantaneous_temperature,
    compute_kinetic_energy,
    remove_center_of_mass_momentum,
)


def compute_geometric_temperature_schedule(
    n_replicas: int,
    t_min_k: float,
    t_max_k: float,
) -> Tuple[List[float], List[float]]:
    """Compute geometric temperature schedule and corresponding thermodynamic betas [D].

    T_k = T_min * (T_max / T_min) ** (k / (M - 1))
    beta_k = 1.0 / (k_B * T_k)

    Parameters
    ----------
    n_replicas : int
        Total number of replicas M (>= 2).
    t_min_k : float
        Minimum setpoint temperature in Kelvin.
    t_max_k : float
        Maximum setpoint temperature in Kelvin.

    Returns
    -------
    Tuple[List[float], List[float]]
        List of temperatures in Kelvin and list of thermodynamic betas in 1/eV.
    """
    if n_replicas < 2:
        raise ValueError(f"n_replicas ({n_replicas}) must be at least 2.")
    if t_max_k <= t_min_k:
        raise ValueError(
            f"t_max_k ({t_max_k}) must be strictly greater than t_min_k ({t_min_k})."
        )

    ratio = t_max_k / t_min_k
    temperatures: List[float] = []
    betas: List[float] = []

    for k in range(n_replicas):
        exponent = k / (n_replicas - 1)
        t_k = round(t_min_k * (ratio**exponent), 4)
        beta_k = 1.0 / (BOLTZMANN_CONSTANT * t_k)
        temperatures.append(t_k)
        betas.append(beta_k)


    return temperatures, betas


def evaluate_metropolis_swap(
    beta_i: float,
    beta_j: float,
    energy_i_ev: float,
    energy_j_ev: float,
) -> Tuple[float, bool]:
    """Evaluate Metropolis swap acceptance between adjacent replicas [D].

    Delta_ij = (beta_i - beta_j) * (U_i - U_j)
    P_swap = min(1.0, exp(Delta_ij))

    Parameters
    ----------
    beta_i : float
        Thermodynamic beta of replica i in 1/eV.
    beta_j : float
        Thermodynamic beta of replica j in 1/eV.
    energy_i_ev : float
        Potential energy of configuration in replica i in eV.
    energy_j_ev : float
        Potential energy of configuration in replica j in eV.

    Returns
    -------
    Tuple[float, bool]
        Acceptance probability P_swap in [0, 1] and boolean acceptance decision.
    """
    delta_ij = (beta_i - beta_j) * (energy_i_ev - energy_j_ev)

    if delta_ij >= 0.0:
        p_swap = 1.0
    else:
        p_swap = float(math.exp(delta_ij))

    xi = float(torch.rand(1).item())
    accepted = xi < p_swap
    return p_swap, accepted


def rescale_velocities_on_swap(
    v_i: torch.Tensor,
    v_j: torch.Tensor,
    t_i: float,
    t_j: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicitly rescale velocities upon replica configuration exchange [D].

    v_i^new = v_j * sqrt(T_i / T_j)
    v_j^new = v_i * sqrt(T_j / T_i)

    Parameters
    ----------
    v_i : torch.Tensor
        Velocities of replica i prior to exchange.
    v_j : torch.Tensor
        Velocities of replica j prior to exchange.
    t_i : float
        Setpoint temperature of replica i in Kelvin.
    t_j : float
        Setpoint temperature of replica j in Kelvin.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rescaled velocities (v_i_new, v_j_new).
    """
    scale_i = math.sqrt(t_i / t_j)
    scale_j = math.sqrt(t_j / t_i)

    v_i_new = v_j * scale_i
    v_j_new = v_i * scale_j
    return v_i_new, v_j_new


def baoab_langevin_step(
    coords: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    masses: torch.Tensor,
    t_target_k: float,
    friction_ps: float = 1.0,
    timestep_fs: float = 0.5,
    remove_com: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Execute a single canonical BAOAB Langevin thermostat integration step [D].

    B: v = v + 0.5 * dt * a
    A: r = r + 0.5 * dt * v
    O: v = c1 * v + c2 * sqrt(k_B * T / m) * eta
    A: r = r + 0.5 * dt * v
    B: F = -grad E, a = kappa_acc * (F / m), v = v + 0.5 * dt * a

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of shape (N_atoms, 3).
    velocities : torch.Tensor
        Velocities of shape (N_atoms, 3).
    accelerations : torch.Tensor
        Accelerations of shape (N_atoms, 3).
    potential_fn : Callable[[torch.Tensor], torch.Tensor]
        Callable returning scalar potential energy in eV.
    masses : torch.Tensor
        Atomic masses of shape (N_atoms, 1) or (N_atoms,) in u.
    t_target_k : float
        Bath setpoint temperature in Kelvin.
    friction_ps : float, optional
        Friction coefficient in ps^-1. Defaults to 1.0.
    timestep_fs : float, optional
        Integration timestep in fs. Defaults to 0.5.
    remove_com : bool, optional
        Whether to project center-of-mass momentum. Defaults to True.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
        Updated (coords, velocities, forces, accelerations, potential_energy_ev).
    """
    m = masses.view(-1, 1)
    dt = float(timestep_fs)
    gamma = float(friction_ps) * 1e-3  # ps^-1 to fs^-1

    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(KAPPA_ACC * (1.0 - c1**2))

    # 1. B step: Half-step velocity update
    v = velocities + 0.5 * dt * accelerations

    # 2. A step: Half-step coordinate update
    r = coords + 0.5 * dt * v

    # 3. O step: Stochastic thermal bath coupling
    eta = torch.randn_like(v)
    sigma_v = c2 * torch.sqrt((BOLTZMANN_CONSTANT * t_target_k) / m)
    v = c1 * v + sigma_v * eta

    if remove_com:
        v = remove_center_of_mass_momentum(v, m)

    # 4. A step: Half-step coordinate update
    r = r + 0.5 * dt * v

    # 5. B step: Analytical force & full-step acceleration update
    f, e_pot = compute_conservative_forces(
        potential_fn, r, return_energy=True, create_graph=False
    )
    a = compute_dimensional_acceleration(f, m)
    v = v + 0.5 * dt * a

    return r, v, f, a, float(e_pot)


class ReplicaState:
    """Individual replica state in REMD ensemble [M]."""

    def __init__(
        self,
        replica_id: int,
        temperature_k: float,
        beta: float,
        coords: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
        accelerations: torch.Tensor,
        potential_energy_ev: float,
    ) -> None:
        self.replica_id = replica_id
        self.temperature_k = temperature_k
        self.beta = beta
        self.coords = coords
        self.velocities = velocities
        self.forces = forces
        self.accelerations = accelerations
        self.potential_energy_ev = potential_energy_ev


class ReplicaExchangeEngine:
    """Replica Exchange Molecular Dynamics Engine with Alternating Odd-Even Trials [M]."""

    def __init__(
        self,
        config: REMDConfig,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        masses: torch.Tensor,
    ) -> None:
        """Initialize REMD Engine.

        Parameters
        ----------
        config : REMDConfig
            Configuration specifying replica count, temperatures, and swap intervals.
        potential_fn : Callable[[torch.Tensor], torch.Tensor]
            Potential energy function.
        masses : torch.Tensor
            Atomic masses tensor of shape (N_atoms, 1) or (N_atoms,) in u.
        """
        self.config = config
        self.potential_fn = potential_fn
        self.masses = masses.view(-1, 1).to(dtype=torch.float64)
        self.n_atoms = int(self.masses.shape[0])

        self.temperatures, self.betas = compute_geometric_temperature_schedule(
            n_replicas=config.n_replicas,
            t_min_k=config.t_min_k,
            t_max_k=config.t_max_k,
        )

        self.replicas: List[ReplicaState] = []
        self.rolling_attempts: Deque[ExchangeLog] = deque(maxlen=50)
        self.exchange_history: List[ExchangeLog] = []

    def initialize_replicas(
        self,
        initial_coords: torch.Tensor,
        initial_velocities: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize all replicas at their scheduled setpoint temperatures [M].

        Parameters
        ----------
        initial_coords : torch.Tensor
            Base Cartesian coordinates of shape (N_atoms, 3) and dtype torch.float64.
        initial_velocities : Optional[torch.Tensor], optional
            Starting velocities. If None, Maxwell-Boltzmann velocities are sampled.
        """
        self.replicas.clear()
        base_coords = initial_coords.clone().detach().to(dtype=torch.float64)

        for k in range(self.config.n_replicas):
            t_k = self.temperatures[k]
            beta_k = self.betas[k]
            coords_k = base_coords.clone()

            if initial_velocities is not None:
                vel_k = initial_velocities.clone().detach().to(dtype=torch.float64)
            else:
                # Sample Maxwell-Boltzmann velocity distribution at T_k
                std_k = torch.sqrt(
                    KAPPA_ACC * BOLTZMANN_CONSTANT * t_k / self.masses
                )
                vel_k = torch.randn_like(coords_k) * std_k
                vel_k = remove_center_of_mass_momentum(vel_k, self.masses)

            f_k, e_pot_k = compute_conservative_forces(
                self.potential_fn, coords_k, return_energy=True, create_graph=False
            )
            a_k = compute_dimensional_acceleration(f_k, self.masses)

            state = ReplicaState(
                replica_id=k,
                temperature_k=t_k,
                beta=beta_k,
                coords=coords_k,
                velocities=vel_k,
                forces=f_k,
                accelerations=a_k,
                potential_energy_ev=float(e_pot_k),
            )
            self.replicas.append(state)

    def step_replicas(
        self,
        step_idx: int,
        timestep_fs: float = 0.5,
    ) -> None:
        """Step all replicas forward by one BAOAB Langevin integration step [D].

        Parameters
        ----------
        step_idx : int
            Current simulation step index.
        timestep_fs : float, optional
            Integration timestep in fs. Defaults to 0.5.
        """
        for rep in self.replicas:
            r, v, f, a, e_pot = baoab_langevin_step(
                coords=rep.coords,
                velocities=rep.velocities,
                accelerations=rep.accelerations,
                potential_fn=self.potential_fn,
                masses=self.masses,
                t_target_k=rep.temperature_k,
                friction_ps=self.config.friction_ps,
                timestep_fs=timestep_fs,
                remove_com=True,
            )
            rep.coords = r
            rep.velocities = v
            rep.forces = f
            rep.accelerations = a
            rep.potential_energy_ev = e_pot

    def attempt_exchanges(
        self,
        step_idx: int,
        cycle_idx: int,
    ) -> List[ExchangeLog]:
        """Execute alternating odd-even swap trial across adjacent replicas [D].

        Parameters
        ----------
        step_idx : int
            Current simulation step index.
        cycle_idx : int
            Swap attempt counter (even cycles swap 0-1, 2-3...; odd cycles swap 1-2, 3-4...).

        Returns
        -------
        List[ExchangeLog]
            Audit records for all attempted swaps in this trial cycle.
        """
        logs: List[ExchangeLog] = []
        is_odd_cycle = cycle_idx % 2 == 1
        start_idx = 1 if is_odd_cycle else 0

        for i in range(start_idx, self.config.n_replicas - 1, 2):
            j = i + 1
            rep_i = self.replicas[i]
            rep_j = self.replicas[j]

            p_swap, accepted = evaluate_metropolis_swap(
                beta_i=rep_i.beta,
                beta_j=rep_j.beta,
                energy_i_ev=rep_i.potential_energy_ev,
                energy_j_ev=rep_j.potential_energy_ev,
            )

            if accepted:
                # 1. Swap configuration coordinates
                r_temp = rep_i.coords.clone()
                rep_i.coords = rep_j.coords.clone()
                rep_j.coords = r_temp

                # 2. Rescale velocities to preserve canonical distributions
                v_i_new, v_j_new = rescale_velocities_on_swap(
                    rep_i.velocities,
                    rep_j.velocities,
                    rep_i.temperature_k,
                    rep_j.temperature_k,
                )
                rep_i.velocities = v_i_new
                rep_j.velocities = v_j_new

                # 3. Re-evaluate analytical forces and accelerations
                f_i, e_i = compute_conservative_forces(
                    self.potential_fn, rep_i.coords, return_energy=True, create_graph=False
                )
                f_j, e_j = compute_conservative_forces(
                    self.potential_fn, rep_j.coords, return_energy=True, create_graph=False
                )
                rep_i.forces = f_i
                rep_i.accelerations = compute_dimensional_acceleration(f_i, self.masses)
                rep_i.potential_energy_ev = float(e_i)

                rep_j.forces = f_j
                rep_j.accelerations = compute_dimensional_acceleration(f_j, self.masses)
                rep_j.potential_energy_ev = float(e_j)

            log_entry = ExchangeLog(
                attempt_step=step_idx,
                replica_i=i,
                replica_j=j,
                temp_i_k=rep_i.temperature_k,
                temp_j_k=rep_j.temperature_k,
                energy_i_ev=rep_i.potential_energy_ev,
                energy_j_ev=rep_j.potential_energy_ev,
                p_swap=p_swap,
                accepted=accepted,
            )
            logs.append(log_entry)
            self.rolling_attempts.append(log_entry)
            self.exchange_history.append(log_entry)

        # Check rolling convergence / divergence guards [M]
        self._check_divergence_guards()

        return logs

    def _check_divergence_guards(self) -> None:
        """Enforce rolling acceptance >= 5% and temperature deviation <= 50 K [M]."""
        if len(self.rolling_attempts) >= 50:
            accepted_count = sum(1 for entry in self.rolling_attempts if entry.accepted)
            acceptance_rate = accepted_count / len(self.rolling_attempts)

            if acceptance_rate < 0.05:
                raise ReplicaExchangeDivergenceError(
                    reason=(
                        f"Rolling swap acceptance rate ({acceptance_rate:.4f}) dropped below "
                        f"5% threshold over last {len(self.rolling_attempts)} trials."
                    ),
                    diagnostics={"acceptance_rate": acceptance_rate},
                )

        # Check instantaneous kinetic temperature deviation across all replicas
        for rep in self.replicas:
            e_kin = compute_kinetic_energy(rep.velocities, self.masses)
            t_inst = compute_instantaneous_temperature(
                e_kin, self.n_atoms, remove_com=True
            )
            deviation = abs(t_inst - rep.temperature_k)
            if deviation > 50.0 and len(self.rolling_attempts) >= 50:
                raise ReplicaExchangeDivergenceError(
                    reason=(
                        f"Replica {rep.replica_id} instantaneous temperature ({t_inst:.2f} K) "
                        f"deviated by > 50 K from setpoint ({rep.temperature_k:.2f} K)."
                    ),
                    diagnostics={
                        "replica_id": rep.replica_id,
                        "t_inst": t_inst,
                        "t_setpoint": rep.temperature_k,
                        "deviation": deviation,
                    },
                )
