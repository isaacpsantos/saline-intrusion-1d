"""One-dimensional advection-dispersion model for salt intrusion."""

from .model import (
    BoundaryCondition,
    CycleDiagnostic,
    DispersionMode,
    FixedPointPeriodicResult,
    InitialCondition,
    PeriodicSimulationResult,
    PeriodicityCriteria,
    SimulationConfig,
    SimulationResult,
    boundary_fluxes,
    dispersion,
    intrusion_length,
    numerical_flux,
    right_boundary_relation,
    sea_salinity,
    simulate,
    simulate_until_periodic,
    solve_periodic_fixed_point,
    velocity,
)

__all__ = [
    "BoundaryCondition",
    "CycleDiagnostic",
    "DispersionMode",
    "FixedPointPeriodicResult",
    "InitialCondition",
    "PeriodicSimulationResult",
    "PeriodicityCriteria",
    "SimulationConfig",
    "SimulationResult",
    "boundary_fluxes",
    "dispersion",
    "intrusion_length",
    "numerical_flux",
    "right_boundary_relation",
    "sea_salinity",
    "simulate",
    "simulate_until_periodic",
    "solve_periodic_fixed_point",
    "velocity",
]

__version__ = "1.0.1"
