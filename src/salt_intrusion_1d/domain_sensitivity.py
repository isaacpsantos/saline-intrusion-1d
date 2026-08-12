"""Sensitivity of the periodic solution to the computational-domain length."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from .experiment import MOUTH_OFFSET_KM, synthetic_config
from .model import (
    FixedPointPeriodicResult,
    PeriodicSimulationResult,
    PeriodicityCriteria,
    simulate_until_periodic,
    solve_periodic_fixed_point,
)

CAPTURE_FROM_MOUTH_KM = 41.0
CAPTURE_DOMAIN_M = (CAPTURE_FROM_MOUTH_KM - MOUTH_OFFSET_KM) * 1_000.0
PeriodicDomainResult = PeriodicSimulationResult | FixedPointPeriodicResult


@dataclass(frozen=True, slots=True)
class DomainMetrics:
    """Diagnostics for one converged domain-length experiment."""

    length_km: float
    cycles_completed: int
    converged: bool
    mean_intrusion_from_mouth_km: float
    max_intrusion_from_mouth_km: float
    distance_front_to_boundary_km: float
    capture_mean_salinity_psu: float
    capture_max_salinity_psu: float
    capture_time_above_threshold_h: float
    capture_fraction_above_threshold: float
    periodic_solver: str
    cycle_map_evaluations: int
    periodic_residual_linf_psu: float


def run_domain_sensitivity(
    lengths_km: Iterable[float] = (50.0, 60.0, 70.0),
    discharge_m3_s: float = 2.0,
    dx_m: float = 100.0,
    dt_s: float = 60.0,
    criteria: PeriodicityCriteria | None = None,
    solver: Literal["cycle_iteration", "fixed_point"] = "cycle_iteration",
) -> dict[float, PeriodicDomainResult]:
    """Run each domain independently until a periodic regime is reached.

    The spatial step is held fixed, so changing ``L`` changes the number of
    cells rather than the numerical resolution. Full profiles are stored at
    every time step in the final cycle so that salinity can be evaluated at the
    water-intake location.
    """

    if dx_m <= 0:
        raise ValueError("dx_m must be positive.")
    if solver not in ("cycle_iteration", "fixed_point"):
        raise ValueError("solver must be 'cycle_iteration' or 'fixed_point'.")
    results: dict[float, PeriodicDomainResult] = {}
    for raw_length in lengths_km:
        length_km = float(raw_length)
        if length_km <= 0:
            raise ValueError("Every domain length must be positive.")
        n_cells_float = length_km * 1_000.0 / dx_m
        if not np.isclose(
            n_cells_float,
            round(n_cells_float),
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("Each domain length must be divisible by dx_m.")
        if length_km * 1_000.0 < CAPTURE_DOMAIN_M:
            raise ValueError("Every domain must contain the capture point.")

        config = synthetic_config(
            discharge_m3_s=discharge_m3_s,
            boundary="danckwerts",
            cycles=1,
            n_cells=int(round(n_cells_float)),
            dt_s=dt_s,
            store_every_steps=1,
            length_m=length_km * 1_000.0,
        )
        if solver == "fixed_point":
            results[length_km] = solve_periodic_fixed_point(config)
        else:
            results[length_km] = simulate_until_periodic(
                config,
                criteria=criteria,
            )
    return results


def salinity_at_location(
    result: PeriodicDomainResult,
    location_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate the final-cycle salinity at a fixed spatial location."""

    cycle = result.last_cycle
    if location_m < cycle.x_m[0] or location_m > cycle.x_m[-1]:
        raise ValueError("location_m lies outside the computational domain.")
    values = np.array(
        [
            np.interp(location_m, cycle.x_m, profile)
            for profile in cycle.stored_profiles_psu
        ],
        dtype=float,
    )
    return cycle.stored_times_s.copy(), values


def duration_above_threshold(
    times_s: ArrayLike,
    values: ArrayLike,
    threshold: float,
) -> float:
    """Return time above a threshold using linear crossing interpolation."""

    times = np.asarray(times_s, dtype=float)
    data = np.asarray(values, dtype=float)
    if times.ndim != 1 or data.ndim != 1 or times.shape != data.shape:
        raise ValueError("times_s and values must be one-dimensional and aligned.")
    if times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must contain at least two increasing values.")

    duration = 0.0
    for left_time, right_time, left_value, right_value in zip(
        times[:-1],
        times[1:],
        data[:-1],
        data[1:],
        strict=True,
    ):
        interval = right_time - left_time
        left_above = left_value >= threshold
        right_above = right_value >= threshold
        if left_above and right_above:
            duration += interval
        elif left_above != right_above:
            if np.isclose(left_value, right_value):
                fraction = 0.5
            else:
                fraction = (threshold - left_value) / (right_value - left_value)
            fraction = float(np.clip(fraction, 0.0, 1.0))
            duration += interval * (fraction if right_above else 1.0 - fraction)
    return float(duration)


def domain_metrics(
    result: PeriodicDomainResult,
    capture_domain_m: float = CAPTURE_DOMAIN_M,
) -> DomainMetrics:
    """Calculate the scientific diagnostics for one domain."""

    cycle = result.last_cycle
    times_s, capture = salinity_at_location(result, capture_domain_m)
    threshold = cycle.config.intrusion_threshold_psu
    duration_s = duration_above_threshold(times_s, capture, threshold)
    cycle_duration_s = times_s[-1] - times_s[0]
    mean_intrusion_km = (
        cycle.mean_intrusion_last_cycle_m() / 1_000.0 + MOUTH_OFFSET_KM
    )
    max_intrusion_km = (
        cycle.max_intrusion_last_cycle_m() / 1_000.0 + MOUTH_OFFSET_KM
    )
    boundary_from_mouth_km = (
        cycle.config.length_m / 1_000.0 + MOUTH_OFFSET_KM
    )
    return DomainMetrics(
        length_km=cycle.config.length_m / 1_000.0,
        cycles_completed=result.cycles_completed,
        converged=result.converged,
        mean_intrusion_from_mouth_km=mean_intrusion_km,
        max_intrusion_from_mouth_km=max_intrusion_km,
        distance_front_to_boundary_km=boundary_from_mouth_km - max_intrusion_km,
        capture_mean_salinity_psu=float(
            np.trapezoid(capture, times_s) / cycle_duration_s
        ),
        capture_max_salinity_psu=float(np.max(capture)),
        capture_time_above_threshold_h=duration_s / 3_600.0,
        capture_fraction_above_threshold=float(duration_s / cycle_duration_s),
        periodic_solver=(
            "fixed_point"
            if isinstance(result, FixedPointPeriodicResult)
            else "cycle_iteration"
        ),
        cycle_map_evaluations=(
            result.cycle_map_evaluations
            if isinstance(result, FixedPointPeriodicResult)
            else result.cycles_completed
        ),
        periodic_residual_linf_psu=(
            result.residual_linf_psu
            if isinstance(result, FixedPointPeriodicResult)
            else float(
                np.max(
                    np.abs(
                        result.last_cycle.final_profile_psu
                        - result.last_cycle.stored_profiles_psu[0]
                    )
                )
            )
        ),
    )


def common_domain_metrics(
    results: dict[float, PeriodicDomainResult],
) -> dict[str, float]:
    """Compare exactly two domains on their common interval."""

    if len(results) != 2:
        raise ValueError("Exactly two domain-length results are required.")
    ordered = sorted(results.items())
    (short_length, short_result), (long_length, long_result) = ordered
    short_cycle = short_result.last_cycle
    long_cycle = long_result.last_cycle
    if not np.array_equal(short_cycle.stored_times_s, long_cycle.stored_times_s):
        raise ValueError("The final-cycle storage times must be identical.")

    long_on_short = np.empty_like(short_cycle.stored_profiles_psu)
    for index, profile in enumerate(long_cycle.stored_profiles_psu):
        long_on_short[index] = np.interp(
            short_cycle.x_m,
            long_cycle.x_m,
            profile,
        )
    profile_difference = np.abs(
        short_cycle.stored_profiles_psu - long_on_short
    )

    times_short, capture_short = salinity_at_location(
        short_result,
        CAPTURE_DOMAIN_M,
    )
    times_long, capture_long = salinity_at_location(
        long_result,
        CAPTURE_DOMAIN_M,
    )
    if not np.array_equal(times_short, times_long):
        raise ValueError("The capture time grids must be identical.")

    return {
        "short_domain_km": short_length,
        "long_domain_km": long_length,
        "common_profile_linf_psu": float(np.max(profile_difference)),
        "capture_salinity_linf_psu": float(
            np.max(np.abs(capture_short - capture_long))
        ),
        "capture_mean_salinity_difference_psu": float(
            abs(
                float(np.trapezoid(capture_short, times_short))
                - float(np.trapezoid(capture_long, times_long))
            )
            / (times_short[-1] - times_short[0])
        ),
        "mean_intrusion_difference_m": abs(
            short_cycle.mean_intrusion_last_cycle_m()
            - long_cycle.mean_intrusion_last_cycle_m()
        ),
        "max_intrusion_difference_m": abs(
            short_cycle.max_intrusion_last_cycle_m()
            - long_cycle.max_intrusion_last_cycle_m()
        ),
    }


def adjacent_domain_metrics(
    results: dict[float, PeriodicDomainResult],
) -> list[dict[str, float]]:
    """Compare every pair of consecutive domain lengths."""

    ordered = sorted(results.items())
    if len(ordered) < 2:
        raise ValueError("At least two domain-length results are required.")
    return [
        common_domain_metrics(dict(ordered[index : index + 2]))
        for index in range(len(ordered) - 1)
    ]


def write_domain_sensitivity_summary(
    results: dict[float, PeriodicDomainResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write one per-domain table and one direct-comparison table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "domain_sensitivity_summary.csv"
    rows = [domain_metrics(result) for _, result in sorted(results.items())]
    fieldnames = list(DomainMetrics.__dataclass_fields__)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: getattr(row, name) for name in fieldnames}
            )

    comparison_path = output_dir / "domain_sensitivity_comparison.csv"
    comparisons = adjacent_domain_metrics(results)
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    return summary_path, comparison_path


def write_capture_timeseries(
    results: dict[float, PeriodicDomainResult],
    output_dir: Path,
) -> Path:
    """Write the final-cycle intake salinity for all domains."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "capture_salinity_timeseries.csv"
    ordered = sorted(results.items())
    series = [
        (length, *salinity_at_location(result, CAPTURE_DOMAIN_M))
        for length, result in ordered
    ]
    reference_time = series[0][1]
    for _, times, _ in series[1:]:
        if not np.array_equal(reference_time, times):
            raise ValueError("The capture time grids must be identical.")
    fieldnames = ["time_h"] + [
        f"capture_salinity_L{length:g}_psu" for length, _, _ in series
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        for index, time_s in enumerate(reference_time):
            writer.writerow(
                [time_s / 3_600.0]
                + [values[index] for _, _, values in series]
            )
    return path


def plot_domain_sensitivity(
    results: dict[float, PeriodicDomainResult],
    output_path: Path | None = None,
) -> Figure:
    """Plot intrusion, intake salinity and final profiles for all domains."""

    if len(results) < 2:
        raise ValueError("At least two domain-length results are required.")
    ordered = sorted(results.items())
    color_map = plt.get_cmap("viridis")
    colors = color_map(np.linspace(0.05, 0.95, len(ordered)))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))

    for color, (length, result) in zip(colors, ordered, strict=True):
        cycle = result.last_cycle
        label = rf"$L={length:g}\,\mathrm{{km}}$"
        axes[0, 0].plot(
            cycle.times_s / 3_600.0,
            cycle.intrusion_length_m / 1_000.0 + MOUTH_OFFSET_KM,
            color=color,
            linewidth=2.0,
            label=label,
        )
        capture_times, capture = salinity_at_location(
            result,
            CAPTURE_DOMAIN_M,
        )
        axes[0, 1].plot(
            capture_times / 3_600.0,
            capture,
            color=color,
            linewidth=2.0,
            label=label,
        )
        axes[1, 0].plot(
            cycle.x_m / 1_000.0 + MOUTH_OFFSET_KM,
            cycle.final_profile_psu,
            color=color,
            linewidth=2.0,
            label=label,
        )

    threshold = ordered[0][1].config.intrusion_threshold_psu
    axes[0, 0].axhline(
        CAPTURE_FROM_MOUTH_KM,
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
        label="Captação",
    )
    axes[0, 0].set_xlabel("Tempo no último ciclo de maré (h)")
    axes[0, 0].set_ylabel("Distância da frente à foz (km)")
    axes[0, 0].set_title("Frente de intrusão")

    axes[0, 1].axhline(
        threshold,
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
        label=r"$C_{\mathrm{lim}}$",
    )
    axes[0, 1].set_xlabel("Tempo no último ciclo de maré (h)")
    axes[0, 1].set_ylabel("Salinidade na captação (PSU)")
    axes[0, 1].set_title("Captação a 41 km da foz")

    axes[1, 0].axvline(
        CAPTURE_FROM_MOUTH_KM,
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
    )
    axes[1, 0].axhline(
        threshold,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1, 0].set_xlabel("Distância à foz (km)")
    axes[1, 0].set_ylabel("Salinidade (PSU)")
    axes[1, 0].set_title("Perfil ao final do ciclo")

    for color, ((short_length, short_result), (long_length, long_result)) in zip(
        colors[1:],
        zip(ordered[:-1], ordered[1:], strict=True),
        strict=True,
    ):
        short_cycle = short_result.last_cycle
        long_cycle = long_result.last_cycle
        long_on_short = np.array(
            [
                np.interp(short_cycle.x_m, long_cycle.x_m, profile)
                for profile in long_cycle.stored_profiles_psu
            ]
        )
        maximum_difference_by_x = np.max(
            np.abs(short_cycle.stored_profiles_psu - long_on_short),
            axis=0,
        )
        axes[1, 1].plot(
            short_cycle.x_m / 1_000.0 + MOUTH_OFFSET_KM,
            maximum_difference_by_x,
            color=color,
            linewidth=2.0,
            label=f"$L={short_length:g}$ vs. $L={long_length:g}$ km",
        )
    axes[1, 1].axvline(
        CAPTURE_FROM_MOUTH_KM,
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
    )
    axes[1, 1].set_xlabel("Distância à foz (km)")
    axes[1, 1].set_ylabel("Diferença máxima no ciclo (PSU)")
    axes[1, 1].set_title("Diferenças entre domínios consecutivos")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def plot_domain_convergence(
    results: dict[float, PeriodicDomainResult],
    output_path: Path | None = None,
) -> Figure:
    """Plot convergence of the intake and intrusion metrics with domain length."""

    if len(results) < 3:
        raise ValueError("At least three domain-length results are required.")
    ordered = sorted(results.items())
    lengths = np.array([length for length, _ in ordered], dtype=float)
    metrics = [domain_metrics(result) for _, result in ordered]
    comparisons = adjacent_domain_metrics(results)

    capture_mean = np.array(
        [metric.capture_mean_salinity_psu for metric in metrics],
        dtype=float,
    )
    intrusion_mean = np.array(
        [metric.mean_intrusion_from_mouth_km for metric in metrics],
        dtype=float,
    )
    intrusion_max = np.array(
        [metric.max_intrusion_from_mouth_km for metric in metrics],
        dtype=float,
    )
    long_lengths = np.array(
        [item["long_domain_km"] for item in comparisons],
        dtype=float,
    )
    capture_differences = np.array(
        [item["capture_mean_salinity_difference_psu"] for item in comparisons],
        dtype=float,
    )
    intrusion_differences = np.array(
        [item["mean_intrusion_difference_m"] / 1_000.0 for item in comparisons],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    axes[0, 0].plot(lengths, capture_mean, "o-", linewidth=2.0)
    axes[0, 0].set_xlabel(r"$L$ (km)")
    axes[0, 0].set_ylabel("Salinidade média (PSU)")
    axes[0, 0].set_title("Captação a 41 km da foz")

    axes[0, 1].plot(
        lengths,
        intrusion_mean,
        "o-",
        linewidth=2.0,
        label="Média",
    )
    axes[0, 1].plot(
        lengths,
        intrusion_max,
        "s--",
        linewidth=2.0,
        label="Máxima",
    )
    axes[0, 1].set_xlabel(r"$L$ (km)")
    axes[0, 1].set_ylabel("Distância da foz (km)")
    axes[0, 1].set_title("Frente de intrusão")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].semilogy(
        long_lengths,
        capture_differences,
        "o-",
        linewidth=2.0,
    )
    axes[1, 0].axhline(0.05, color="tab:red", linestyle=":", label="0,05 PSU")
    axes[1, 0].set_xlabel(r"Maior $L$ do par (km)")
    axes[1, 0].set_ylabel("Diferença entre médias (PSU)")
    axes[1, 0].set_title("Convergência na captação")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].semilogy(
        long_lengths,
        intrusion_differences,
        "o-",
        linewidth=2.0,
    )
    axes[1, 1].axhline(0.1, color="tab:red", linestyle=":", label="0,1 km")
    axes[1, 1].set_xlabel(r"Maior $L$ do par (km)")
    axes[1, 1].set_ylabel("Diferença na média (km)")
    axes[1, 1].set_title("Convergência da frente")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig
