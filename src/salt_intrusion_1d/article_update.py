"""Finite-horizon parameter study used to revise the teaching article.

The article asks what happens during a synthetic 31-day low-flow episode.
Consequently, this module evaluates domain independence after 60 tidal cycles
from the prescribed fresh-water initial condition.  It does not replace that
question by the infinite-time periodic fixed-point problem.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np

from .domain_sensitivity import (
    CAPTURE_DOMAIN_M,
    CAPTURE_FROM_MOUTH_KM,
    duration_above_threshold,
)
from .experiment import MOUTH_OFFSET_KM, synthetic_config
from .model import SimulationConfig, SimulationResult, simulate

ARTICLE_BASE_DISPERSION_M2_S = 30.0
ARTICLE_DISPERSION_KAPPA = 0.25
ARTICLE_CYCLES = 60
ARTICLE_DX_M = 12.5
ARTICLE_DT_S = 7.5

# Resolution used in the v0.7 parameter screening.  It remains explicit in
# ``main`` so that the historical screening can still be reproduced, while
# calls to ``article_config`` default to the verified v0.9 article mesh.
SCREENING_DX_M = 100.0
SCREENING_DT_S = 60.0

CAPTURE_MEAN_TOLERANCE_PSU = 0.05
INTRUSION_TOLERANCE_KM = 0.1
MINIMUM_FRONT_MARGIN_KM = 5.0

DispersionChoice = Literal["constant", "velocity_dependent"]


@dataclass(frozen=True, slots=True)
class TransientMetrics:
    """Last-cycle diagnostics for one finite-horizon simulation."""

    discharge_m3_s: float
    length_km: float
    cycles: int
    base_dispersion_m2_s: float
    dispersion_kappa: float
    dispersion_mode: str
    mean_intrusion_from_mouth_km: float
    max_intrusion_from_mouth_km: float
    distance_front_to_boundary_km: float
    capture_mean_salinity_psu: float
    capture_min_salinity_psu: float
    capture_max_salinity_psu: float
    capture_time_above_threshold_h: float
    capture_fraction_above_threshold: float


@dataclass(frozen=True, slots=True)
class TransientDomainComparison:
    """Differences between two finite domains at the same final horizon."""

    discharge_m3_s: float
    short_domain_km: float
    long_domain_km: float
    capture_mean_difference_psu: float
    capture_linf_difference_psu: float
    mean_intrusion_difference_km: float
    max_intrusion_difference_km: float
    common_profile_linf_psu: float


def article_config(
    discharge_m3_s: float,
    length_km: float = 50.0,
    *,
    cycles: int = ARTICLE_CYCLES,
    dx_m: float = ARTICLE_DX_M,
    dt_s: float = ARTICLE_DT_S,
    base_dispersion_m2_s: float = ARTICLE_BASE_DISPERSION_M2_S,
    dispersion_kappa: float = ARTICLE_DISPERSION_KAPPA,
    dispersion_mode: DispersionChoice = "velocity_dependent",
    store_every_steps: int = 30,
) -> SimulationConfig:
    """Return the revised finite-horizon configuration for the article."""

    if length_km <= 0 or dx_m <= 0:
        raise ValueError("length_km and dx_m must be positive.")
    n_cells_float = length_km * 1_000.0 / dx_m
    if not np.isclose(n_cells_float, round(n_cells_float)):
        raise ValueError("length_km must be divisible by dx_m.")
    config = synthetic_config(
        discharge_m3_s=discharge_m3_s,
        boundary="danckwerts",
        cycles=cycles,
        n_cells=int(round(n_cells_float)),
        dt_s=dt_s,
        store_every_steps=store_every_steps,
        length_m=length_km * 1_000.0,
    )
    return replace(
        config,
        dispersion_mode=dispersion_mode,
        base_dispersion_m2_s=base_dispersion_m2_s,
        dispersion_kappa=dispersion_kappa,
    )


def run_article_case(
    discharge_m3_s: float,
    length_km: float = 50.0,
    **config_kwargs: Any,
) -> SimulationResult:
    """Simulate one revised article scenario."""

    return simulate(
        article_config(
            discharge_m3_s,
            length_km,
            **config_kwargs,
        )
    )


def _capture_series(
    result: SimulationResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return intake salinity during the final complete tidal cycle."""

    start = result.times_s[-1] - result.config.tidal_period_s
    mask = result.stored_times_s >= start
    times = result.stored_times_s[mask]
    values = np.array(
        [
            np.interp(CAPTURE_DOMAIN_M, result.x_m, profile)
            for profile in result.stored_profiles_psu[mask]
        ],
        dtype=float,
    )
    return times - start, values


def transient_metrics(result: SimulationResult) -> TransientMetrics:
    """Calculate last-cycle metrics for a finite-horizon result."""

    config = result.config
    times, capture = _capture_series(result)
    duration = duration_above_threshold(
        times,
        capture,
        config.intrusion_threshold_psu,
    )
    cycle_duration = times[-1] - times[0]
    mean_intrusion = (
        result.mean_intrusion_last_cycle_m() / 1_000.0 + MOUTH_OFFSET_KM
    )
    max_intrusion = (
        result.max_intrusion_last_cycle_m() / 1_000.0 + MOUTH_OFFSET_KM
    )
    boundary_from_mouth = config.length_m / 1_000.0 + MOUTH_OFFSET_KM
    return TransientMetrics(
        discharge_m3_s=round(config.river_velocity_m_s * 1_050.0, 12),
        length_km=config.length_m / 1_000.0,
        cycles=int(round(config.final_time_s / config.tidal_period_s)),
        base_dispersion_m2_s=config.base_dispersion_m2_s,
        dispersion_kappa=config.dispersion_kappa,
        dispersion_mode=config.dispersion_mode,
        mean_intrusion_from_mouth_km=mean_intrusion,
        max_intrusion_from_mouth_km=max_intrusion,
        distance_front_to_boundary_km=boundary_from_mouth - max_intrusion,
        capture_mean_salinity_psu=float(
            np.trapezoid(capture, times) / cycle_duration
        ),
        capture_min_salinity_psu=float(np.min(capture)),
        capture_max_salinity_psu=float(np.max(capture)),
        capture_time_above_threshold_h=duration / 3_600.0,
        capture_fraction_above_threshold=duration / cycle_duration,
    )


def compare_transient_domains(
    results: dict[float, SimulationResult],
) -> TransientDomainComparison:
    """Compare two domain lengths after the same finite simulation horizon."""

    if len(results) != 2:
        raise ValueError("Exactly two domain results are required.")
    (short_length, short), (long_length, long) = sorted(results.items())
    if not np.array_equal(short.times_s, long.times_s):
        raise ValueError("The time grids must be identical.")

    short_times, short_capture = _capture_series(short)
    long_times, long_capture = _capture_series(long)
    if not np.array_equal(short_times, long_times):
        raise ValueError("The stored final-cycle times must be identical.")

    start = short.times_s[-1] - short.config.tidal_period_s
    short_mask = short.stored_times_s >= start
    long_mask = long.stored_times_s >= start
    long_on_short = np.array(
        [
            np.interp(short.x_m, long.x_m, profile)
            for profile in long.stored_profiles_psu[long_mask]
        ]
    )
    profile_difference = np.abs(
        short.stored_profiles_psu[short_mask] - long_on_short
    )
    short_metrics = transient_metrics(short)
    long_metrics = transient_metrics(long)
    return TransientDomainComparison(
        discharge_m3_s=short_metrics.discharge_m3_s,
        short_domain_km=short_length,
        long_domain_km=long_length,
        capture_mean_difference_psu=abs(
            short_metrics.capture_mean_salinity_psu
            - long_metrics.capture_mean_salinity_psu
        ),
        capture_linf_difference_psu=float(
            np.max(np.abs(short_capture - long_capture))
        ),
        mean_intrusion_difference_km=abs(
            short_metrics.mean_intrusion_from_mouth_km
            - long_metrics.mean_intrusion_from_mouth_km
        ),
        max_intrusion_difference_km=abs(
            short_metrics.max_intrusion_from_mouth_km
            - long_metrics.max_intrusion_from_mouth_km
        ),
        common_profile_linf_psu=float(np.max(profile_difference)),
    )


def configuration_is_acceptable(
    metrics: TransientMetrics,
    comparison: TransientDomainComparison,
) -> bool:
    """Apply the stated domain-independence and front-margin criteria."""

    return (
        metrics.distance_front_to_boundary_km >= MINIMUM_FRONT_MARGIN_KM
        and comparison.capture_mean_difference_psu
        <= CAPTURE_MEAN_TOLERANCE_PSU
        and comparison.mean_intrusion_difference_km
        <= INTRUSION_TOLERANCE_KM
        and comparison.max_intrusion_difference_km
        <= INTRUSION_TOLERANCE_KM
    )


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    data = list(rows)
    if not data:
        raise ValueError("At least one row is required.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return path


def write_cycle_landmarks(
    results: dict[float, SimulationResult],
    path: Path,
    cycles: tuple[int, ...] = (3, 10, 20, 30, 40, 50, 60),
) -> Path:
    """Write the article's selected cycle-by-cycle intrusion landmarks."""

    rows: list[dict[str, object]] = []
    for discharge, result in sorted(results.items(), reverse=True):
        steps = int(round(result.config.tidal_period_s / result.config.dt_s))
        for cycle in cycles:
            start = (cycle - 1) * steps
            end = cycle * steps
            time = result.times_s[start : end + 1]
            intrusion = result.intrusion_length_m[start : end + 1]
            rows.append(
                {
                    "discharge_m3_s": discharge,
                    "cycle": cycle,
                    "time_days": cycle * result.config.tidal_period_s / 86_400.0,
                    "mean_intrusion_from_mouth_km": (
                        np.trapezoid(intrusion, time)
                        / result.config.tidal_period_s
                        / 1_000.0
                        + MOUTH_OFFSET_KM
                    ),
                    "max_intrusion_from_mouth_km": (
                        np.max(intrusion) / 1_000.0 + MOUTH_OFFSET_KM
                    ),
                }
            )
    return _write_rows(path, rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def plot_selected_scenarios(
    results: dict[float, SimulationResult],
    path: Path,
) -> Path:
    """Plot the two discharge scenarios used in the revised article."""

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    colors = {10.0: "tab:blue", 2.0: "tab:orange"}
    for discharge in (10.0, 2.0):
        result = results[discharge]
        label = rf"$Q={discharge:g}\,\mathrm{{m^3/s}}$"
        axes[0].plot(
            result.x_m / 1_000.0 + MOUTH_OFFSET_KM,
            result.final_profile_psu,
            linewidth=2.2,
            color=colors[discharge],
            label=label,
        )
        mask = result.last_cycle_mask()
        start = result.times_s[mask][0]
        axes[1].plot(
            (result.times_s[mask] - start) / 3_600.0,
            result.intrusion_length_m[mask] / 1_000.0 + MOUTH_OFFSET_KM,
            linewidth=2.2,
            color=colors[discharge],
            label=label,
        )

    axes[0].axhline(0.5, color="0.35", linestyle="--", linewidth=1.2)
    axes[0].axvline(
        CAPTURE_FROM_MOUTH_KM,
        color="tab:red",
        linestyle=":",
        linewidth=1.3,
        label="Captação",
    )
    axes[0].set(
        xlabel="Distância da foz (km)",
        ylabel="Salinidade (PSU)",
        title="Perfil ao final de 60 ciclos",
        xlim=(MOUTH_OFFSET_KM, MOUTH_OFFSET_KM + 50.0),
    )
    axes[1].axhline(
        CAPTURE_FROM_MOUTH_KM,
        color="tab:red",
        linestyle=":",
        linewidth=1.3,
        label="Captação",
    )
    axes[1].set(
        xlabel="Tempo no último ciclo (h)",
        ylabel="Distância da frente à foz (km)",
        title="Frente de intrusão no ciclo 60",
    )
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_selected_domain_check(
    results: dict[float, SimulationResult],
    path: Path,
) -> Path:
    """Plot the 50--60 km domain check for the critical discharge."""

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for length, result in sorted(results.items()):
        label = rf"$L={length:g}\,\mathrm{{km}}$"
        mask = result.last_cycle_mask()
        start = result.times_s[mask][0]
        axes[0].plot(
            (result.times_s[mask] - start) / 3_600.0,
            result.intrusion_length_m[mask] / 1_000.0 + MOUTH_OFFSET_KM,
            linewidth=2.1,
            label=label,
        )
        capture_time, capture = _capture_series(result)
        axes[1].plot(
            capture_time / 3_600.0,
            capture,
            linewidth=2.1,
            label=label,
        )
    axes[0].set(
        xlabel="Tempo no último ciclo (h)",
        ylabel="Distância da frente à foz (km)",
        title=r"Frente salina: $Q=2\,\mathrm{m^3/s}$",
    )
    axes[1].axhline(
        0.5,
        color="tab:red",
        linestyle=":",
        linewidth=1.3,
        label=r"$C_{\mathrm{lim}}$",
    )
    axes[1].set(
        xlabel="Tempo no último ciclo (h)",
        ylabel="Salinidade na captação (PSU)",
        title="Captação a 41 km da foz",
    )
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dispersion_comparison(
    results: dict[str, SimulationResult],
    path: Path,
) -> Path:
    """Compare constant and velocity-dependent dispersion for Q=10 m3/s."""

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    labels = {
        "constant": r"$D(t)=D_0$",
        "velocity_dependent": r"$D(t)=D_0+\kappa |u(t)|B$",
    }
    styles = {"constant": "--", "velocity_dependent": "-"}
    for mode in ("constant", "velocity_dependent"):
        result = results[mode]
        axes[0].plot(
            result.x_m / 1_000.0 + MOUTH_OFFSET_KM,
            result.final_profile_psu,
            styles[mode],
            linewidth=2.2,
            label=labels[mode],
        )
        mask = result.last_cycle_mask()
        start = result.times_s[mask][0]
        axes[1].plot(
            (result.times_s[mask] - start) / 3_600.0,
            result.intrusion_length_m[mask] / 1_000.0 + MOUTH_OFFSET_KM,
            styles[mode],
            linewidth=2.2,
            label=labels[mode],
        )
    axes[0].axhline(0.5, color="0.35", linestyle=":", linewidth=1.2)
    axes[0].set(
        xlabel="Distância da foz (km)",
        ylabel="Salinidade (PSU)",
        title=r"Perfil final: $Q=10\,\mathrm{m^3/s}$",
        xlim=(MOUTH_OFFSET_KM, MOUTH_OFFSET_KM + 50.0),
    )
    axes[1].set(
        xlabel="Tempo no último ciclo (h)",
        ylabel="Distância da frente à foz (km)",
        title="Efeito da dispersão no ciclo 60",
    )
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the finite-horizon parameter update for the article."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_article_update_v0.7"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    screening_rows: list[dict[str, object]] = []
    for base_dispersion in (5.0, 10.0, 20.0, 30.0):
        for kappa in (0.1, 0.2, 0.3):
            pair = {
                length: run_article_case(
                    2.0,
                    length,
                    dx_m=SCREENING_DX_M,
                    dt_s=SCREENING_DT_S,
                    base_dispersion_m2_s=base_dispersion,
                    dispersion_kappa=kappa,
                )
                for length in (50.0, 60.0)
            }
            metrics = transient_metrics(pair[50.0])
            comparison = compare_transient_domains(pair)
            screening_rows.append(
                {
                    "base_dispersion_m2_s": base_dispersion,
                    "dispersion_kappa": kappa,
                    "mean_intrusion_L50_from_mouth_km": (
                        metrics.mean_intrusion_from_mouth_km
                    ),
                    "max_intrusion_L50_from_mouth_km": (
                        metrics.max_intrusion_from_mouth_km
                    ),
                    "front_margin_L50_km": (
                        metrics.distance_front_to_boundary_km
                    ),
                    "capture_mean_L50_psu": (
                        metrics.capture_mean_salinity_psu
                    ),
                    "capture_mean_difference_psu": (
                        comparison.capture_mean_difference_psu
                    ),
                    "mean_intrusion_difference_km": (
                        comparison.mean_intrusion_difference_km
                    ),
                    "max_intrusion_difference_km": (
                        comparison.max_intrusion_difference_km
                    ),
                    "accepted": configuration_is_acceptable(
                        metrics,
                        comparison,
                    ),
                }
            )
    _write_rows(args.output / "parameter_screening.csv", screening_rows)

    selected_by_discharge = {
        discharge: run_article_case(
            discharge,
            dx_m=SCREENING_DX_M,
            dt_s=SCREENING_DT_S,
        )
        for discharge in (10.0, 2.0)
    }
    selected_domains = {
        discharge: {
            length: (
                selected_by_discharge[discharge]
                if length == 50.0
                else run_article_case(
                    discharge,
                    length,
                    dx_m=SCREENING_DX_M,
                    dt_s=SCREENING_DT_S,
                )
            )
            for length in (50.0, 60.0)
        }
        for discharge in (10.0, 2.0)
    }
    selected_rows: list[dict[str, object]] = []
    for discharge, pair in sorted(selected_domains.items(), reverse=True):
        comparison = compare_transient_domains(pair)
        for length, result in sorted(pair.items()):
            row = asdict(transient_metrics(result))
            row["domain_check_capture_mean_difference_psu"] = (
                comparison.capture_mean_difference_psu
            )
            row["domain_check_mean_intrusion_difference_km"] = (
                comparison.mean_intrusion_difference_km
            )
            row["domain_check_max_intrusion_difference_km"] = (
                comparison.max_intrusion_difference_km
            )
            row["domain_check_accepted"] = configuration_is_acceptable(
                transient_metrics(pair[50.0]),
                comparison,
            )
            selected_rows.append(row)
    _write_rows(args.output / "selected_domain_summary.csv", selected_rows)
    write_cycle_landmarks(
        selected_by_discharge,
        args.output / "selected_cycle_landmarks.csv",
    )
    plot_selected_scenarios(
        selected_by_discharge,
        args.output / "selected_scenarios.png",
    )
    plot_selected_domain_check(
        selected_domains[2.0],
        args.output / "selected_domain_check_Q2.png",
    )

    dispersion_results = {
        "velocity_dependent": selected_by_discharge[10.0],
        "constant": run_article_case(
            10.0,
            dx_m=SCREENING_DX_M,
            dt_s=SCREENING_DT_S,
            dispersion_mode="constant",
            dispersion_kappa=0.0,
        ),
    }
    _write_rows(
        args.output / "dispersion_comparison_Q10.csv",
        [
            asdict(transient_metrics(dispersion_results[mode]))
            for mode in ("constant", "velocity_dependent")
        ],
    )
    plot_dispersion_comparison(
        dispersion_results,
        args.output / "dispersion_comparison_Q10.png",
    )

    refinement_rows: list[dict[str, object]] = []
    for discharge in (10.0, 2.0):
        for dx_m, dt_s in ((200.0, 120.0), (100.0, 60.0), (50.0, 30.0)):
            result = (
                selected_by_discharge[discharge]
                if (dx_m, dt_s) == (100.0, 60.0)
                else run_article_case(
                    discharge,
                    dx_m=dx_m,
                    dt_s=dt_s,
                    store_every_steps=max(1, int(round(1_800.0 / dt_s))),
                )
            )
            metrics = transient_metrics(result)
            refinement_rows.append(
                {
                    "discharge_m3_s": discharge,
                    "dx_m": dx_m,
                    "dt_s": dt_s,
                    "mean_intrusion_from_mouth_km": (
                        metrics.mean_intrusion_from_mouth_km
                    ),
                    "max_intrusion_from_mouth_km": (
                        metrics.max_intrusion_from_mouth_km
                    ),
                }
            )
    _write_rows(
        args.output / "selected_refinement.csv",
        refinement_rows,
    )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
