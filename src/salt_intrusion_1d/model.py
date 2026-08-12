"""Numerical model for one-dimensional salt intrusion.

The equation

    C_t + u(t) C_x = D(t) C_xx

is discretized with backward Euler in time, first-order upwinding for
advection, and centered differences for dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, gmres

BoundaryCondition = Literal["neumann", "danckwerts"]
DispersionMode = Literal["constant", "velocity_dependent"]
InitialCondition = Literal["zero", "exponential"]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parameters defining one numerical experiment.

    ``right_boundary="danckwerts"`` switches the boundary at ``x=L``:

    * for ``u >= 0``, homogeneous Neumann is used (advective outflow);
    * for ``u < 0``, a Danckwerts inflow condition is used.

    The left boundary is a prescribed marine reservoir for every velocity sign.
    """

    length_m: float = 50_000.0
    n_cells: int = 500
    final_time_s: float = 60.0 * 12.4 * 3600.0
    dt_s: float = 60.0

    river_velocity_m_s: float = 10.0 / 1050.0
    tidal_velocity_amplitude_m_s: float = 0.35
    tidal_period_s: float = 12.4 * 3600.0

    dispersion_mode: DispersionMode = "velocity_dependent"
    base_dispersion_m2_s: float = 50.0
    dispersion_kappa: float = 0.5
    channel_width_m: float = 300.0
    cross_section_area_m2: float = 1050.0

    sea_salinity_mean_psu: float = 12.0
    sea_salinity_amplitude_psu: float = 4.0
    upstream_salinity_psu: float = 0.0
    right_boundary: BoundaryCondition = "danckwerts"

    initial_condition: InitialCondition = "zero"
    initial_decay_length_m: float = 1_000.0
    intrusion_threshold_psu: float = 0.5
    store_every_steps: int = 30

    def __post_init__(self) -> None:
        positive = {
            "length_m": self.length_m,
            "n_cells": self.n_cells,
            "final_time_s": self.final_time_s,
            "dt_s": self.dt_s,
            "tidal_period_s": self.tidal_period_s,
            "base_dispersion_m2_s": self.base_dispersion_m2_s,
            "channel_width_m": self.channel_width_m,
            "cross_section_area_m2": self.cross_section_area_m2,
            "store_every_steps": self.store_every_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.n_cells < 2:
            raise ValueError("n_cells must be at least 2.")
        if self.dispersion_kappa < 0:
            raise ValueError("dispersion_kappa cannot be negative.")
        if self.river_velocity_m_s < 0:
            raise ValueError("river_velocity_m_s cannot be negative.")
        if self.sea_salinity_mean_psu < abs(self.sea_salinity_amplitude_psu):
            raise ValueError("The prescribed marine salinity cannot be negative.")
        if self.upstream_salinity_psu < 0:
            raise ValueError("upstream_salinity_psu cannot be negative.")
        if self.intrusion_threshold_psu < 0:
            raise ValueError("intrusion_threshold_psu cannot be negative.")
        if self.right_boundary not in ("neumann", "danckwerts"):
            raise ValueError("right_boundary must be 'neumann' or 'danckwerts'.")
        if self.dispersion_mode not in ("constant", "velocity_dependent"):
            raise ValueError(
                "dispersion_mode must be 'constant' or 'velocity_dependent'."
            )
        if self.initial_condition not in ("zero", "exponential"):
            raise ValueError("initial_condition must be 'zero' or 'exponential'.")
        if self.initial_condition == "exponential" and self.initial_decay_length_m <= 0:
            raise ValueError("initial_decay_length_m must be positive.")

        steps = self.final_time_s / self.dt_s
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1.0e-10):
            raise ValueError("final_time_s must be an integer multiple of dt_s.")

    @property
    def dx_m(self) -> float:
        return self.length_m / self.n_cells

    @property
    def n_steps(self) -> int:
        return int(round(self.final_time_s / self.dt_s))


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Numerical solution and diagnostics from one experiment."""

    config: SimulationConfig
    x_m: NDArray[np.float64]
    times_s: NDArray[np.float64]
    intrusion_length_m: NDArray[np.float64]
    total_salt_psu_m: NDArray[np.float64]
    internal_salt_content_psu_m3: NDArray[np.float64]
    trapezoidal_salt_content_psu_m3: NDArray[np.float64]
    left_numerical_flux_psu_m_s: NDArray[np.float64]
    right_numerical_flux_psu_m_s: NDArray[np.float64]
    left_physical_flux_psu_m_s: NDArray[np.float64]
    right_physical_flux_psu_m_s: NDArray[np.float64]
    discrete_balance_residual_psu_m3_s: NDArray[np.float64]
    physical_balance_residual_psu_m3_s: NDArray[np.float64]
    right_boundary_salinity_psu: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    dispersion_m2_s: NDArray[np.float64]
    stored_times_s: NDArray[np.float64]
    stored_profiles_psu: NDArray[np.float64]

    def last_cycle_mask(self) -> NDArray[np.bool_]:
        start = max(0.0, self.times_s[-1] - self.config.tidal_period_s)
        return self.times_s >= start

    def mean_intrusion_last_cycle_m(self) -> float:
        mask = self.last_cycle_mask()
        time = self.times_s[mask]
        values = self.intrusion_length_m[mask]
        if time.size == 1:
            return float(values[0])
        return float(np.trapezoid(values, time) / (time[-1] - time[0]))

    def max_intrusion_last_cycle_m(self) -> float:
        return float(np.max(self.intrusion_length_m[self.last_cycle_mask()]))

    @property
    def final_profile_psu(self) -> NDArray[np.float64]:
        return self.stored_profiles_psu[-1].copy()


@dataclass(frozen=True, slots=True)
class PeriodicityCriteria:
    """Stopping criteria for a cycle-by-cycle periodicity search.

    Convergence is accepted only when every tolerance is satisfied for
    ``consecutive_cycles`` successive tidal cycles.
    """

    profile_linf_tolerance_psu: float = 1.0e-3
    relative_salt_tolerance: float = 1.0e-4
    mean_intrusion_tolerance_m: float = 10.0
    min_cycles: int = 20
    consecutive_cycles: int = 3
    max_cycles: int = 400

    def __post_init__(self) -> None:
        tolerances = {
            "profile_linf_tolerance_psu": self.profile_linf_tolerance_psu,
            "relative_salt_tolerance": self.relative_salt_tolerance,
            "mean_intrusion_tolerance_m": self.mean_intrusion_tolerance_m,
        }
        for name, value in tolerances.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.min_cycles < 2:
            raise ValueError("min_cycles must be at least 2.")
        if self.consecutive_cycles < 1:
            raise ValueError("consecutive_cycles must be at least 1.")
        if self.max_cycles < self.min_cycles:
            raise ValueError("max_cycles must be at least min_cycles.")


@dataclass(frozen=True, slots=True)
class CycleDiagnostic:
    """Diagnostics evaluated at equal tidal phases in consecutive cycles."""

    cycle: int
    mean_intrusion_m: float
    max_intrusion_m: float
    end_intrusion_m: float
    end_total_salt_psu_m: float
    end_right_boundary_salinity_psu: float
    profile_change_linf_psu: float
    total_salt_relative_change: float
    mean_intrusion_change_m: float
    within_tolerances: bool
    convergence_streak: int


@dataclass(frozen=True, slots=True)
class PeriodicSimulationResult:
    """Result of an automatic search for a tidally periodic solution."""

    config: SimulationConfig
    criteria: PeriodicityCriteria
    last_cycle: SimulationResult
    diagnostics: tuple[CycleDiagnostic, ...]
    converged: bool

    @property
    def cycles_completed(self) -> int:
        return len(self.diagnostics)


@dataclass(frozen=True, slots=True)
class FixedPointPeriodicResult:
    """Tidally periodic solution obtained from the one-cycle fixed point."""

    config: SimulationConfig
    last_cycle: SimulationResult
    converged: bool
    residual_linf_psu: float
    linear_solver_info: int
    linear_iterations: int
    cycle_map_evaluations: int

    @property
    def cycles_completed(self) -> int:
        """Compatibility value for diagnostics based on cycle marching."""

        return 0


def angular_frequency(config: SimulationConfig) -> float:
    return 2.0 * np.pi / config.tidal_period_s


def velocity(time_s: ArrayLike, config: SimulationConfig) -> NDArray[np.float64]:
    """Return ``u(t) = -u_river + U_tide sin(omega t)``."""

    time = np.asarray(time_s, dtype=float)
    return (
        -config.river_velocity_m_s
        + config.tidal_velocity_amplitude_m_s
        * np.sin(angular_frequency(config) * time)
    )


def dispersion(
    time_s: ArrayLike,
    config: SimulationConfig,
) -> NDArray[np.float64]:
    """Return the constant or velocity-dependent dispersion coefficient."""

    time = np.asarray(time_s, dtype=float)
    if config.dispersion_mode == "constant":
        return np.full_like(time, config.base_dispersion_m2_s, dtype=float)
    if config.dispersion_mode == "velocity_dependent":
        return (
            config.base_dispersion_m2_s
            + config.dispersion_kappa
            * np.abs(velocity(time, config))
            * config.channel_width_m
        )
    raise ValueError(f"Unknown dispersion mode: {config.dispersion_mode!r}")


def sea_salinity(
    time_s: ArrayLike,
    config: SimulationConfig,
) -> NDArray[np.float64]:
    """Prescribed salinity of the marine reservoir at ``x=0``."""

    time = np.asarray(time_s, dtype=float)
    return (
        config.sea_salinity_mean_psu
        + config.sea_salinity_amplitude_psu
        * np.sin(angular_frequency(config) * time)
    )


def right_boundary_relation(
    velocity_m_s: float,
    dispersion_m2_s: float,
    dx_m: float,
    upstream_salinity_psu: float,
    boundary: BoundaryCondition,
) -> tuple[float, float]:
    """Return ``theta, eta`` such that ``C_N = theta*C_(N-1) + eta``.

    Homogeneous Neumann gives ``theta=1`` and ``eta=0``.  With Danckwerts
    inflow at ``x=L`` and ``u<0``,

        u C_N - D (C_N-C_(N-1))/dx = u C_upstream.

    Therefore

        theta = D/(D-u*dx),
        eta   = -u*dx*C_upstream/(D-u*dx).
    """

    if dispersion_m2_s <= 0 or dx_m <= 0:
        raise ValueError("dispersion_m2_s and dx_m must be positive.")
    if boundary == "neumann" or velocity_m_s >= 0:
        return 1.0, 0.0
    if boundary != "danckwerts":
        raise ValueError(f"Unknown right boundary: {boundary!r}")

    denominator = dispersion_m2_s - velocity_m_s * dx_m
    if denominator <= 0:
        raise FloatingPointError("Non-positive Danckwerts denominator.")
    theta = dispersion_m2_s / denominator
    eta = (
        -velocity_m_s
        * dx_m
        * upstream_salinity_psu
        / denominator
    )
    return float(theta), float(eta)


def intrusion_length(
    profile_psu: ArrayLike,
    x_m: ArrayLike,
    threshold_psu: float,
) -> float:
    """Locate the furthest threshold crossing using linear interpolation."""

    profile = np.asarray(profile_psu, dtype=float)
    x = np.asarray(x_m, dtype=float)
    if profile.ndim != 1 or x.ndim != 1 or profile.shape != x.shape:
        raise ValueError("profile_psu and x_m must be one-dimensional and aligned.")
    if threshold_psu < 0:
        raise ValueError("threshold_psu cannot be negative.")

    indices = np.flatnonzero(profile >= threshold_psu)
    if indices.size == 0:
        return 0.0
    last = int(indices[-1])
    if last == profile.size - 1:
        return float(x[-1])

    left_value = profile[last]
    right_value = profile[last + 1]
    if np.isclose(left_value, right_value):
        return float(x[last])
    fraction = (left_value - threshold_psu) / (left_value - right_value)
    return float(x[last] + np.clip(fraction, 0.0, 1.0) * (x[last + 1] - x[last]))


def numerical_flux(
    left_salinity_psu: float,
    right_salinity_psu: float,
    velocity_m_s: float,
    dispersion_m2_s: float,
    dx_m: float,
) -> float:
    """Return the conservative upwind-diffusive face flux.

    Positive flux is oriented toward increasing ``x``.
    """

    advective = (
        max(velocity_m_s, 0.0) * left_salinity_psu
        + min(velocity_m_s, 0.0) * right_salinity_psu
    )
    diffusive = -dispersion_m2_s * (
        right_salinity_psu - left_salinity_psu
    ) / dx_m
    return float(advective + diffusive)


def boundary_fluxes(
    profile_psu: ArrayLike,
    velocity_m_s: float,
    dispersion_m2_s: float,
    dx_m: float,
) -> tuple[float, float, float, float]:
    """Return numerical face fluxes and physical boundary fluxes.

    The tuple is ``(Jhat_left, Jhat_right, J_left, J_right)``.  The physical
    boundary fluxes use first-order one-sided derivatives, consistently with
    the boundary relations enforced by the solver.
    """

    profile = np.asarray(profile_psu, dtype=float)
    if profile.ndim != 1 or profile.size < 3:
        raise ValueError("profile_psu must contain at least three nodes.")
    left_numerical = numerical_flux(
        profile[0], profile[1], velocity_m_s, dispersion_m2_s, dx_m
    )
    right_numerical = numerical_flux(
        profile[-2], profile[-1], velocity_m_s, dispersion_m2_s, dx_m
    )
    left_physical = (
        velocity_m_s * profile[0]
        - dispersion_m2_s * (profile[1] - profile[0]) / dx_m
    )
    right_physical = (
        velocity_m_s * profile[-1]
        - dispersion_m2_s * (profile[-1] - profile[-2]) / dx_m
    )
    return (
        float(left_numerical),
        float(right_numerical),
        float(left_physical),
        float(right_physical),
    )


def _initial_profile(
    x_m: NDArray[np.float64],
    config: SimulationConfig,
    initial_profile_psu: ArrayLike | None,
    *,
    allow_negative: bool = False,
) -> NDArray[np.float64]:
    if initial_profile_psu is not None:
        profile = np.asarray(initial_profile_psu, dtype=float).copy()
        if profile.shape != x_m.shape:
            raise ValueError("initial_profile_psu must have n_cells + 1 entries.")
        if not allow_negative and np.any(profile < 0):
            raise ValueError("initial_profile_psu cannot contain negative values.")
    elif config.initial_condition == "zero":
        profile = np.zeros_like(x_m)
    elif config.initial_condition == "exponential":
        profile = (
            float(sea_salinity(0.0, config))
            * np.exp(-x_m / config.initial_decay_length_m)
        )
    else:
        raise ValueError(f"Unknown initial condition: {config.initial_condition!r}")

    # Dirichlet data are enforced at the initial boundary node.  For the zero
    # option, the resulting corner incompatibility is explicit and intentional.
    profile[0] = float(sea_salinity(0.0, config))
    return profile


def simulate(
    config: SimulationConfig,
    initial_profile_psu: ArrayLike | None = None,
    *,
    _allow_negative_initial_profile: bool = False,
) -> SimulationResult:
    """Run the implicit finite-difference solver."""

    n_cells = config.n_cells
    n_internal = n_cells - 1
    dx = config.dx_m
    dt = config.dt_s

    x = np.linspace(0.0, config.length_m, n_cells + 1)
    times = np.arange(config.n_steps + 1, dtype=float) * dt
    velocities = np.asarray(velocity(times, config), dtype=float)
    dispersions = np.asarray(dispersion(times, config), dtype=float)

    regular_save_indices = np.arange(
        0,
        config.n_steps + 1,
        config.store_every_steps,
    )
    steps_per_tide = config.tidal_period_s / dt
    if np.isclose(steps_per_tide, round(steps_per_tide), rtol=0.0, atol=1.0e-10):
        cycle_save_indices = np.arange(
            0,
            config.n_steps + 1,
            int(round(steps_per_tide)),
        )
    else:
        cycle_save_indices = np.array([], dtype=int)
    save_indices = np.unique(
        np.concatenate(
            (
                regular_save_indices,
                cycle_save_indices,
                np.array([config.n_steps]),
            )
        )
    )
    stored_profiles = np.empty((save_indices.size, n_cells + 1), dtype=float)
    stored_times = times[save_indices]
    next_save = 0

    intrusion = np.empty(config.n_steps + 1, dtype=float)
    total_salt = np.empty(config.n_steps + 1, dtype=float)
    internal_content = np.empty(config.n_steps + 1, dtype=float)
    trapezoidal_content = np.empty(config.n_steps + 1, dtype=float)
    left_numerical_flux = np.empty(config.n_steps + 1, dtype=float)
    right_numerical_flux = np.empty(config.n_steps + 1, dtype=float)
    left_physical_flux = np.empty(config.n_steps + 1, dtype=float)
    right_physical_flux = np.empty(config.n_steps + 1, dtype=float)
    discrete_residual = np.full(config.n_steps + 1, np.nan, dtype=float)
    physical_residual = np.full(config.n_steps + 1, np.nan, dtype=float)
    right_salinity = np.empty(config.n_steps + 1, dtype=float)

    current = _initial_profile(
        x,
        config,
        initial_profile_psu,
        allow_negative=_allow_negative_initial_profile,
    )
    stored_profiles[next_save] = current
    next_save += 1
    intrusion[0] = intrusion_length(current, x, config.intrusion_threshold_psu)
    total_salt[0] = np.trapezoid(current, x)
    internal_content[0] = config.cross_section_area_m2 * dx * np.sum(current[1:-1])
    trapezoidal_content[0] = config.cross_section_area_m2 * total_salt[0]
    (
        left_numerical_flux[0],
        right_numerical_flux[0],
        left_physical_flux[0],
        right_physical_flux[0],
    ) = boundary_fluxes(current, velocities[0], dispersions[0], dx)
    right_salinity[0] = current[-1]

    for step in range(config.n_steps):
        new_index = step + 1
        u = float(velocities[new_index])
        diffusivity = float(dispersions[new_index])
        left_value = float(sea_salinity(times[new_index], config))

        r = diffusivity * dt / dx**2
        alpha = u * dt / dx
        if u >= 0:
            lower = -r - alpha
            diagonal = 1.0 + 2.0 * r + alpha
            upper = -r
        else:
            lower = -r
            diagonal = 1.0 + 2.0 * r - alpha
            upper = -r + alpha

        rhs = current[1:n_cells].copy()
        rhs[0] -= lower * left_value

        theta, eta = right_boundary_relation(
            velocity_m_s=u,
            dispersion_m2_s=diffusivity,
            dx_m=dx,
            upstream_salinity_psu=config.upstream_salinity_psu,
            boundary=config.right_boundary,
        )

        main = np.full(n_internal, diagonal, dtype=float)
        main[-1] += upper * theta
        rhs[-1] -= upper * eta

        banded = np.zeros((3, n_internal), dtype=float)
        banded[1] = main
        if n_internal > 1:
            banded[0, 1:] = upper
            banded[2, :-1] = lower

        internal = solve_banded(
            (1, 1),
            banded,
            rhs,
            overwrite_ab=True,
            overwrite_b=True,
            check_finite=False,
        )

        updated = np.empty_like(current)
        updated[0] = left_value
        updated[1:n_cells] = internal
        updated[n_cells] = theta * internal[-1] + eta
        current = updated

        intrusion[new_index] = intrusion_length(
            current,
            x,
            config.intrusion_threshold_psu,
        )
        total_salt[new_index] = np.trapezoid(current, x)
        internal_content[new_index] = (
            config.cross_section_area_m2 * dx * np.sum(current[1:-1])
        )
        trapezoidal_content[new_index] = (
            config.cross_section_area_m2 * total_salt[new_index]
        )
        (
            left_numerical_flux[new_index],
            right_numerical_flux[new_index],
            left_physical_flux[new_index],
            right_physical_flux[new_index],
        ) = boundary_fluxes(current, u, diffusivity, dx)
        discrete_residual[new_index] = (
            (internal_content[new_index] - internal_content[step]) / dt
            - config.cross_section_area_m2
            * (left_numerical_flux[new_index] - right_numerical_flux[new_index])
        )
        physical_residual[new_index] = (
            (trapezoidal_content[new_index] - trapezoidal_content[step]) / dt
            - config.cross_section_area_m2
            * (left_physical_flux[new_index] - right_physical_flux[new_index])
        )
        right_salinity[new_index] = current[-1]

        if next_save < save_indices.size and new_index == save_indices[next_save]:
            stored_profiles[next_save] = current
            next_save += 1

    return SimulationResult(
        config=config,
        x_m=x,
        times_s=times,
        intrusion_length_m=intrusion,
        total_salt_psu_m=total_salt,
        internal_salt_content_psu_m3=internal_content,
        trapezoidal_salt_content_psu_m3=trapezoidal_content,
        left_numerical_flux_psu_m_s=left_numerical_flux,
        right_numerical_flux_psu_m_s=right_numerical_flux,
        left_physical_flux_psu_m_s=left_physical_flux,
        right_physical_flux_psu_m_s=right_physical_flux,
        discrete_balance_residual_psu_m3_s=discrete_residual,
        physical_balance_residual_psu_m3_s=physical_residual,
        right_boundary_salinity_psu=right_salinity,
        velocity_m_s=velocities,
        dispersion_m2_s=dispersions,
        stored_times_s=stored_times,
        stored_profiles_psu=stored_profiles,
    )


def solve_periodic_fixed_point(
    config: SimulationConfig,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    residual_tolerance_psu: float = 1.0e-8,
    restart: int = 120,
    maxiter: int = 250,
) -> FixedPointPeriodicResult:
    """Solve the exact one-tide fixed-point problem with matrix-free GMRES.

    For the linear, periodically forced model, one complete tidal cycle defines
    an affine map ``F(C)=A C+b``.  The periodic initial profile satisfies
    ``(I-A)C=b``.  Solving that system directly avoids mistaking a very slowly
    decaying transient for a periodic state on long domains.
    """

    positive = {
        "rtol": rtol,
        "atol": atol,
        "residual_tolerance_psu": residual_tolerance_psu,
        "restart": restart,
        "maxiter": maxiter,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    steps_per_cycle = config.tidal_period_s / config.dt_s
    if not np.isclose(
        steps_per_cycle,
        round(steps_per_cycle),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "tidal_period_s must be an integer multiple of dt_s "
            "for a periodic fixed-point solve."
        )

    cycle_steps = int(round(steps_per_cycle))
    map_config = replace(
        config,
        final_time_s=config.tidal_period_s,
        store_every_steps=cycle_steps,
    )
    left_value = float(sea_salinity(0.0, map_config))
    n_unknowns = map_config.n_cells
    cycle_map_evaluations = 0

    def cycle_map(interior_profile: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal cycle_map_evaluations
        cycle_map_evaluations += 1
        full_profile = np.empty(n_unknowns + 1, dtype=float)
        full_profile[0] = left_value
        full_profile[1:] = interior_profile
        result = simulate(
            map_config,
            full_profile,
            _allow_negative_initial_profile=True,
        )
        return result.final_profile_psu[1:]

    zero_response = cycle_map(np.zeros(n_unknowns, dtype=float))

    def fixed_point_operator(
        interior_profile: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        homogeneous_response = cycle_map(interior_profile) - zero_response
        return interior_profile - homogeneous_response

    operator = LinearOperator(
        (n_unknowns, n_unknowns),
        matvec=fixed_point_operator,
        dtype=float,
    )
    linear_iterations = 0

    def count_iteration(_: float) -> None:
        nonlocal linear_iterations
        linear_iterations += 1

    periodic_interior, solver_info = gmres(
        operator,
        zero_response,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        callback=count_iteration,
        callback_type="pr_norm",
    )

    periodic_initial = np.empty(n_unknowns + 1, dtype=float)
    periodic_initial[0] = left_value
    periodic_initial[1:] = periodic_interior
    final_cycle_config = replace(map_config, store_every_steps=1)
    final_cycle = simulate(final_cycle_config, periodic_initial)
    residual = float(
        np.max(np.abs(final_cycle.final_profile_psu - periodic_initial))
    )
    converged = solver_info == 0 and residual <= residual_tolerance_psu

    return FixedPointPeriodicResult(
        config=config,
        last_cycle=final_cycle,
        converged=converged,
        residual_linf_psu=residual,
        linear_solver_info=int(solver_info),
        linear_iterations=linear_iterations,
        cycle_map_evaluations=cycle_map_evaluations,
    )


def simulate_until_periodic(
    config: SimulationConfig,
    criteria: PeriodicityCriteria | None = None,
    initial_profile_psu: ArrayLike | None = None,
) -> PeriodicSimulationResult:
    """Advance complete tidal cycles until a periodicity criterion is met.

    Profiles are compared at the same tidal phase, namely at the end of each
    complete cycle.  The forcing functions in this model are periodic with
    ``config.tidal_period_s``; consequently, every one-cycle solve may use the
    same local time interval ``[0, tidal_period_s]``.

    The returned ``last_cycle`` contains the time series and profiles of the
    final cycle only.  The complete transient is represented compactly by the
    cycle diagnostics.
    """

    stopping = criteria or PeriodicityCriteria()
    steps_per_cycle = config.tidal_period_s / config.dt_s
    if not np.isclose(
        steps_per_cycle,
        round(steps_per_cycle),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "tidal_period_s must be an integer multiple of dt_s "
            "for periodicity detection."
        )

    cycle_config = replace(
        config,
        final_time_s=config.tidal_period_s,
    )
    previous_profile: NDArray[np.float64] | None = None
    previous_salt: float | None = None
    previous_mean_intrusion: float | None = None
    current_initial = initial_profile_psu
    diagnostics: list[CycleDiagnostic] = []
    convergence_streak = 0
    converged = False
    cycle_result: SimulationResult | None = None

    for cycle in range(1, stopping.max_cycles + 1):
        cycle_result = simulate(cycle_config, current_initial)
        final_profile = cycle_result.final_profile_psu
        end_salt = float(cycle_result.total_salt_psu_m[-1])
        mean_intrusion = cycle_result.mean_intrusion_last_cycle_m()

        if previous_profile is None:
            profile_change = np.nan
            salt_change = np.nan
            mean_intrusion_change = np.nan
            within_tolerances = False
        else:
            profile_change = float(
                np.max(np.abs(final_profile - previous_profile))
            )
            salt_denominator = max(
                abs(float(previous_salt)),
                np.finfo(float).eps,
            )
            salt_change = abs(end_salt - float(previous_salt)) / salt_denominator
            mean_intrusion_change = abs(
                mean_intrusion - float(previous_mean_intrusion)
            )
            within_tolerances = (
                cycle >= stopping.min_cycles
                and profile_change <= stopping.profile_linf_tolerance_psu
                and salt_change <= stopping.relative_salt_tolerance
                and mean_intrusion_change <= stopping.mean_intrusion_tolerance_m
            )

        convergence_streak = (
            convergence_streak + 1 if within_tolerances else 0
        )
        diagnostics.append(
            CycleDiagnostic(
                cycle=cycle,
                mean_intrusion_m=mean_intrusion,
                max_intrusion_m=cycle_result.max_intrusion_last_cycle_m(),
                end_intrusion_m=float(cycle_result.intrusion_length_m[-1]),
                end_total_salt_psu_m=end_salt,
                end_right_boundary_salinity_psu=float(
                    cycle_result.right_boundary_salinity_psu[-1]
                ),
                profile_change_linf_psu=profile_change,
                total_salt_relative_change=salt_change,
                mean_intrusion_change_m=mean_intrusion_change,
                within_tolerances=within_tolerances,
                convergence_streak=convergence_streak,
            )
        )

        if convergence_streak >= stopping.consecutive_cycles:
            converged = True
            break

        previous_profile = final_profile
        previous_salt = end_salt
        previous_mean_intrusion = mean_intrusion
        current_initial = final_profile

    if cycle_result is None:
        raise RuntimeError("No tidal cycle was simulated.")

    return PeriodicSimulationResult(
        config=config,
        criteria=stopping,
        last_cycle=cycle_result,
        diagnostics=tuple(diagnostics),
        converged=converged,
    )
