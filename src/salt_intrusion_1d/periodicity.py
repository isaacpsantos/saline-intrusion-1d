"""Automatic search for a tidally periodic synthetic solution."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .experiment import MOUTH_OFFSET_KM, synthetic_config
from .model import (
    PeriodicSimulationResult,
    PeriodicityCriteria,
    SimulationConfig,
    simulate_until_periodic,
)


@dataclass(frozen=True, slots=True)
class BoundaryPeakDiagnostic:
    """Peak of ``C(L,t)`` and its relation to a velocity reversal."""

    dt_s: float
    peak_time_s: float
    peak_salinity_psu: float
    positive_to_negative_crossing_s: float

    @property
    def peak_minus_crossing_s(self) -> float:
        """Signed time difference: peak time minus reversal time."""

        return self.peak_time_s - self.positive_to_negative_crossing_s


def run_periodicity_experiment(
    discharge_m3_s: float = 2.0,
    n_cells: int = 500,
    dt_s: float = 60.0,
    store_every_steps: int = 30,
    criteria: PeriodicityCriteria | None = None,
) -> PeriodicSimulationResult:
    """Run the Danckwerts case until convergence or the cycle limit."""

    config = synthetic_config(
        discharge_m3_s=discharge_m3_s,
        boundary="danckwerts",
        cycles=1,
        n_cells=n_cells,
        dt_s=dt_s,
        store_every_steps=store_every_steps,
    )
    return simulate_until_periodic(config, criteria=criteria)


def velocity_sign_changes_in_cycle(
    config: SimulationConfig,
) -> tuple[tuple[float, str], ...]:
    """Return exact velocity sign changes during one synthetic tidal cycle.

    The prescribed velocity is

    ``u(t) = -u_river + U_tide sin(2*pi*t/T)``.

    Each returned pair contains the time in seconds and the direction of the
    change: ``"negative_to_positive"`` or ``"positive_to_negative"``.
    """

    amplitude = config.tidal_velocity_amplitude_m_s
    if amplitude <= 0:
        return ()
    ratio = config.river_velocity_m_s / amplitude
    if ratio < 0 or ratio >= 1:
        return ()

    phase = float(np.arcsin(ratio))
    factor = config.tidal_period_s / (2.0 * np.pi)
    return (
        (phase * factor, "negative_to_positive"),
        ((np.pi - phase) * factor, "positive_to_negative"),
    )


def boundary_peak_diagnostic(
    result: PeriodicSimulationResult,
) -> BoundaryPeakDiagnostic:
    """Locate the sampled maximum of ``C(L,t)`` in the final cycle."""

    cycle = result.last_cycle
    changes = velocity_sign_changes_in_cycle(cycle.config)
    crossing = next(
        (
            time_s
            for time_s, direction in changes
            if direction == "positive_to_negative"
        ),
        None,
    )
    if crossing is None:
        raise ValueError(
            "The configuration has no positive-to-negative velocity crossing."
        )
    peak_index = int(np.argmax(cycle.right_boundary_salinity_psu))
    return BoundaryPeakDiagnostic(
        dt_s=cycle.config.dt_s,
        peak_time_s=float(cycle.times_s[peak_index]),
        peak_salinity_psu=float(
            cycle.right_boundary_salinity_psu[peak_index]
        ),
        positive_to_negative_crossing_s=crossing,
    )


def run_temporal_refinement(
    coarse_result: PeriodicSimulationResult,
    refinement_factor: int = 2,
) -> dict[float, PeriodicSimulationResult]:
    """Repeat a periodic experiment with a smaller time step.

    The already-computed coarse result is reused. The refined simulation uses
    the same physical and spatial parameters, periodicity criteria and an
    equivalent interval between stored full profiles.
    """

    if refinement_factor < 2:
        raise ValueError("refinement_factor must be at least 2.")
    coarse_dt = coarse_result.config.dt_s
    refined_dt = coarse_dt / refinement_factor
    steps_per_cycle = coarse_result.config.tidal_period_s / refined_dt
    if not np.isclose(
        steps_per_cycle,
        round(steps_per_cycle),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "The refined time step must divide the tidal period exactly."
        )

    storage_interval_s = (
        coarse_result.config.store_every_steps * coarse_dt
    )
    refined_store_every = max(1, int(round(storage_interval_s / refined_dt)))
    refined_config = replace(
        coarse_result.config,
        final_time_s=coarse_result.config.tidal_period_s,
        dt_s=refined_dt,
        store_every_steps=refined_store_every,
    )
    refined = simulate_until_periodic(
        refined_config,
        criteria=coarse_result.criteria,
    )
    return {coarse_dt: coarse_result, refined_dt: refined}


def temporal_refinement_metrics(
    results: dict[float, PeriodicSimulationResult],
) -> dict[str, float]:
    """Compare the two final-cycle solutions on the refined time grid."""

    if len(results) != 2:
        raise ValueError("Exactly two time-step results are required.")
    ordered = sorted(results.items(), reverse=True)
    (coarse_dt, coarse), (refined_dt, refined) = ordered
    coarse_cycle = coarse.last_cycle
    refined_cycle = refined.last_cycle

    coarse_salinity_on_refined = np.interp(
        refined_cycle.times_s,
        coarse_cycle.times_s,
        coarse_cycle.right_boundary_salinity_psu,
    )
    coarse_intrusion_on_refined = np.interp(
        refined_cycle.times_s,
        coarse_cycle.times_s,
        coarse_cycle.intrusion_length_m,
    )
    coarse_peak = boundary_peak_diagnostic(coarse)
    refined_peak = boundary_peak_diagnostic(refined)

    return {
        "coarse_dt_s": coarse_dt,
        "refined_dt_s": refined_dt,
        "coarse_cycles": float(coarse.cycles_completed),
        "refined_cycles": float(refined.cycles_completed),
        "coarse_peak_time_h": coarse_peak.peak_time_s / 3_600.0,
        "refined_peak_time_h": refined_peak.peak_time_s / 3_600.0,
        "velocity_reversal_time_h": (
            refined_peak.positive_to_negative_crossing_s / 3_600.0
        ),
        "coarse_peak_minus_reversal_s": (
            coarse_peak.peak_minus_crossing_s
        ),
        "refined_peak_minus_reversal_s": (
            refined_peak.peak_minus_crossing_s
        ),
        "coarse_peak_salinity_psu": coarse_peak.peak_salinity_psu,
        "refined_peak_salinity_psu": refined_peak.peak_salinity_psu,
        "peak_salinity_change_psu": abs(
            refined_peak.peak_salinity_psu - coarse_peak.peak_salinity_psu
        ),
        "boundary_salinity_linf_change_psu": float(
            np.max(
                np.abs(
                    refined_cycle.right_boundary_salinity_psu
                    - coarse_salinity_on_refined
                )
            )
        ),
        "intrusion_linf_change_m": float(
            np.max(
                np.abs(
                    refined_cycle.intrusion_length_m
                    - coarse_intrusion_on_refined
                )
            )
        ),
        "mean_intrusion_change_m": abs(
            refined_cycle.mean_intrusion_last_cycle_m()
            - coarse_cycle.mean_intrusion_last_cycle_m()
        ),
        "max_intrusion_change_m": abs(
            refined_cycle.max_intrusion_last_cycle_m()
            - coarse_cycle.max_intrusion_last_cycle_m()
        ),
    }


def plot_boundary_temporal_refinement(
    results: dict[float, PeriodicSimulationResult],
    output_path: Path | None = None,
) -> Figure:
    """Plot ``C(L,t)`` and ``u(t)`` for the time-step comparison."""

    if len(results) != 2:
        raise ValueError("Exactly two time-step results are required.")
    ordered = sorted(results.items(), reverse=True)
    finest = ordered[-1][1].last_cycle
    changes = velocity_sign_changes_in_cycle(finest.config)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    colors = ("tab:blue", "tab:orange")
    for color, (dt_s, result) in zip(colors, ordered, strict=True):
        cycle = result.last_cycle
        axes[0].plot(
            cycle.times_s / 3_600.0,
            cycle.right_boundary_salinity_psu,
            color=color,
            linewidth=2.0,
            label=rf"$\Delta t={dt_s:g}\,\mathrm{{s}}$",
        )
        peak = boundary_peak_diagnostic(result)
        axes[0].plot(
            peak.peak_time_s / 3_600.0,
            peak.peak_salinity_psu,
            marker="o",
            color=color,
            markersize=5,
        )

    for index, (time_s, direction) in enumerate(changes):
        direction_label = (
            r"$v<0\rightarrow v>0$"
            if direction == "negative_to_positive"
            else r"$v>0\rightarrow v<0$"
        )
        for ax in axes:
            ax.axvline(
                time_s / 3_600.0,
                color="0.35",
                linestyle="--",
                linewidth=1.2,
            )
        axes[1].annotate(
            direction_label,
            xy=(time_s / 3_600.0, 0.0),
            xytext=(8 if index == 0 else -8, 8),
            textcoords="offset points",
            ha="left" if index == 0 else "right",
            fontsize=9,
        )

    axes[0].axhline(
        finest.config.intrusion_threshold_psu,
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
        label=r"$C_{\mathrm{lim}}$",
    )
    axes[0].set_ylabel(r"$C(L,t)$ (PSU)")
    axes[0].set_title(
        r"Salinidade em $x=L$ e refinamento temporal"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        finest.times_s / 3_600.0,
        finest.velocity_m_s,
        color="tab:green",
        linewidth=1.8,
    )
    axes[1].axhline(0.0, color="0.25", linewidth=1.0)
    axes[1].set_xlabel("Tempo no último ciclo de maré (h)")
    axes[1].set_ylabel(r"$v(t)$ (m/s)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig


def write_temporal_refinement_summary(
    results: dict[float, PeriodicSimulationResult],
    output_dir: Path,
) -> Path:
    """Write the time-step comparison metrics to a one-row CSV file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "temporal_refinement_summary.csv"
    metrics = temporal_refinement_metrics(results)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return path


def write_periodicity_diagnostics(
    result: PeriodicSimulationResult,
    output_dir: Path,
) -> Path:
    """Write one row of diagnostics per simulated tidal cycle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "periodicity_diagnostics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "cycle",
                "mean_intrusion_domain_km",
                "mean_intrusion_from_mouth_km",
                "max_intrusion_domain_km",
                "max_intrusion_from_mouth_km",
                "end_intrusion_domain_km",
                "end_total_salt_psu_m",
                "end_right_boundary_salinity_psu",
                "profile_change_linf_psu",
                "total_salt_relative_change",
                "mean_intrusion_change_m",
                "within_tolerances",
                "convergence_streak",
            ],
        )
        writer.writeheader()
        for item in result.diagnostics:
            mean_domain_km = item.mean_intrusion_m / 1_000.0
            max_domain_km = item.max_intrusion_m / 1_000.0
            writer.writerow(
                {
                    "cycle": item.cycle,
                    "mean_intrusion_domain_km": mean_domain_km,
                    "mean_intrusion_from_mouth_km": (
                        mean_domain_km + MOUTH_OFFSET_KM
                    ),
                    "max_intrusion_domain_km": max_domain_km,
                    "max_intrusion_from_mouth_km": (
                        max_domain_km + MOUTH_OFFSET_KM
                    ),
                    "end_intrusion_domain_km": (
                        item.end_intrusion_m / 1_000.0
                    ),
                    "end_total_salt_psu_m": item.end_total_salt_psu_m,
                    "end_right_boundary_salinity_psu": (
                        item.end_right_boundary_salinity_psu
                    ),
                    "profile_change_linf_psu": item.profile_change_linf_psu,
                    "total_salt_relative_change": (
                        item.total_salt_relative_change
                    ),
                    "mean_intrusion_change_m": item.mean_intrusion_change_m,
                    "within_tolerances": item.within_tolerances,
                    "convergence_streak": item.convergence_streak,
                }
            )
    return path


def write_periodic_last_cycle_series(
    result: PeriodicSimulationResult,
    output_dir: Path,
) -> Path:
    """Write the complete time series of the final simulated tidal cycle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "periodic_last_cycle_timeseries.csv"
    cycle = result.last_cycle
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_in_cycle_h",
                "velocity_m_s",
                "dispersion_m2_s",
                "intrusion_domain_km",
                "intrusion_from_mouth_km",
                "right_boundary_salinity_psu",
                "total_salt_psu_m",
            ]
        )
        for index, time_s in enumerate(cycle.times_s):
            intrusion_domain_km = (
                cycle.intrusion_length_m[index] / 1_000.0
            )
            writer.writerow(
                [
                    time_s / 3_600.0,
                    cycle.velocity_m_s[index],
                    cycle.dispersion_m2_s[index],
                    intrusion_domain_km,
                    intrusion_domain_km + MOUTH_OFFSET_KM,
                    cycle.right_boundary_salinity_psu[index],
                    cycle.total_salt_psu_m[index],
                ]
            )
    return path


def plot_periodicity_diagnostics(
    result: PeriodicSimulationResult,
    output_dir: Path,
) -> list[Path]:
    """Save convergence histories and the final tidal cycle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    diagnostics = result.diagnostics
    cycles = np.array([item.cycle for item in diagnostics])
    mean_intrusion = np.array(
        [item.mean_intrusion_m for item in diagnostics]
    ) / 1_000.0 + MOUTH_OFFSET_KM

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(cycles, mean_intrusion, linewidth=2.0)
    ax.set(
        xlabel="Ciclo de maré",
        ylabel="Intrusão média a partir da foz (km)",
        title="Evolução até o regime periódico",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "periodicity_mean_intrusion.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    quantities = (
        (
            np.array(
                [item.profile_change_linf_psu for item in diagnostics]
            ),
            result.criteria.profile_linf_tolerance_psu,
            r"$\|C_k-C_{k-1}\|_\infty$ (PSU)",
        ),
        (
            np.array(
                [item.total_salt_relative_change for item in diagnostics]
            ),
            result.criteria.relative_salt_tolerance,
            "Variação relativa da massa de sal",
        ),
        (
            np.array(
                [item.mean_intrusion_change_m for item in diagnostics]
            ),
            result.criteria.mean_intrusion_tolerance_m,
            r"$|\overline{L}_{s,k}-\overline{L}_{s,k-1}|$ (m)",
        ),
    )
    for ax, (values, tolerance, ylabel) in zip(axes, quantities, strict=True):
        ax.semilogy(cycles, values, linewidth=1.8)
        ax.axhline(tolerance, color="tab:red", linestyle="--", linewidth=1.3)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Ciclo de maré")
    axes[0].set_title("Diagnósticos de periodicidade")
    fig.tight_layout()
    path = output_dir / "periodicity_criteria.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    cycle = result.last_cycle
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        cycle.times_s / 3_600.0,
        cycle.intrusion_length_m / 1_000.0 + MOUTH_OFFSET_KM,
        linewidth=2.0,
    )
    ax.set(
        xlabel="Tempo no último ciclo de maré (h)",
        ylabel="Distância da frente salina à foz (km)",
        title=f"Último ciclo simulado (ciclo {result.cycles_completed})",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "periodicity_last_cycle.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        cycle.times_s / 3_600.0,
        cycle.right_boundary_salinity_psu,
        linewidth=2.0,
    )
    ax.axhline(
        cycle.config.intrusion_threshold_psu,
        color="tab:red",
        linestyle="--",
        linewidth=1.3,
        label=r"$C_{\mathrm{lim}}$",
    )
    ax.set(
        xlabel="Tempo no último ciclo de maré (h)",
        ylabel=r"$C(L,t)$ (PSU)",
        title=rf"Salinidade na fronteira $x=L$ (ciclo "
        f"{result.cycles_completed})",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "periodicity_right_boundary.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search automatically for a tidally periodic solution."
    )
    parser.add_argument("--discharge", type=float, default=2.0)
    parser.add_argument("--cells", type=int, default=500)
    parser.add_argument("--dt", type=float, default=60.0)
    parser.add_argument("--max-cycles", type=int, default=400)
    parser.add_argument("--min-cycles", type=int, default=20)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--profile-tol", type=float, default=1.0e-3)
    parser.add_argument("--salt-tol", type=float, default=1.0e-4)
    parser.add_argument("--intrusion-tol-m", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_periodicity"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    criteria = PeriodicityCriteria(
        profile_linf_tolerance_psu=args.profile_tol,
        relative_salt_tolerance=args.salt_tol,
        mean_intrusion_tolerance_m=args.intrusion_tol_m,
        min_cycles=args.min_cycles,
        consecutive_cycles=args.consecutive,
        max_cycles=args.max_cycles,
    )
    result = run_periodicity_experiment(
        discharge_m3_s=args.discharge,
        n_cells=args.cells,
        dt_s=args.dt,
        criteria=criteria,
    )
    table = write_periodicity_diagnostics(result, args.output)
    last_cycle_series = write_periodic_last_cycle_series(result, args.output)
    figures = plot_periodicity_diagnostics(result, args.output)
    status = "converged" if result.converged else "not converged"
    last = result.diagnostics[-1]
    print(
        f"Periodicity search: {status} after {result.cycles_completed} cycles."
    )
    print(
        f"Last mean intrusion: "
        f"{last.mean_intrusion_m / 1_000.0 + MOUTH_OFFSET_KM:.3f} km "
        f"from the mouth."
    )
    print(f"Diagnostics: {table}")
    print(f"Last-cycle series: {last_cycle_series}")
    print(f"Figures: {len(figures)} files in {args.output}")


if __name__ == "__main__":
    main()
