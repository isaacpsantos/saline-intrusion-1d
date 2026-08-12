"""Reproducible comparison of right-boundary conditions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from .model import SimulationConfig, SimulationResult, simulate

MOUTH_OFFSET_KM = 10.0
CHANNEL_AREA_M2 = 300.0 * 3.5


def synthetic_config(
    discharge_m3_s: float,
    boundary: str,
    cycles: int = 60,
    n_cells: int = 500,
    dt_s: float = 60.0,
    store_every_steps: int = 30,
    length_m: float = 50_000.0,
) -> SimulationConfig:
    """Return the synthetic RSM-inspired configuration used in the comparison."""

    tidal_period_s = 12.4 * 3600.0
    return SimulationConfig(
        length_m=length_m,
        n_cells=n_cells,
        final_time_s=cycles * tidal_period_s,
        dt_s=dt_s,
        river_velocity_m_s=discharge_m3_s / CHANNEL_AREA_M2,
        tidal_velocity_amplitude_m_s=0.35,
        tidal_period_s=tidal_period_s,
        dispersion_mode="velocity_dependent",
        base_dispersion_m2_s=50.0,
        dispersion_kappa=0.5,
        channel_width_m=300.0,
        sea_salinity_mean_psu=12.0,
        sea_salinity_amplitude_psu=4.0,
        upstream_salinity_psu=0.0,
        right_boundary=boundary,  # type: ignore[arg-type]
        initial_condition="zero",
        intrusion_threshold_psu=0.5,
        store_every_steps=store_every_steps,
    )


def run_comparison(
    cycles: int = 60,
    n_cells: int = 500,
    dt_s: float = 60.0,
    store_every_steps: int = 30,
) -> dict[tuple[float, str], SimulationResult]:
    """Run both boundaries for the two synthetic discharge scenarios."""

    results: dict[tuple[float, str], SimulationResult] = {}
    for discharge in (10.0, 2.0):
        base = synthetic_config(
            discharge_m3_s=discharge,
            boundary="neumann",
            cycles=cycles,
            n_cells=n_cells,
            dt_s=dt_s,
            store_every_steps=store_every_steps,
        )
        for boundary in ("neumann", "danckwerts"):
            config = replace(base, right_boundary=boundary)
            results[(discharge, boundary)] = simulate(config)
    return results


def write_summary(
    results: dict[tuple[float, str], SimulationResult],
    output_dir: Path,
) -> Path:
    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "discharge_m3_s",
                "right_boundary",
                "mean_intrusion_last_cycle_domain_km",
                "max_intrusion_last_cycle_domain_km",
                "mean_intrusion_last_cycle_from_mouth_km",
                "max_intrusion_last_cycle_from_mouth_km",
                "final_total_salt_psu_m",
                "final_right_boundary_salinity_psu",
            ],
        )
        writer.writeheader()
        for (discharge, boundary), result in sorted(results.items()):
            mean_domain = result.mean_intrusion_last_cycle_m() / 1_000.0
            max_domain = result.max_intrusion_last_cycle_m() / 1_000.0
            writer.writerow(
                {
                    "discharge_m3_s": discharge,
                    "right_boundary": boundary,
                    "mean_intrusion_last_cycle_domain_km": mean_domain,
                    "max_intrusion_last_cycle_domain_km": max_domain,
                    "mean_intrusion_last_cycle_from_mouth_km": (
                        mean_domain + MOUTH_OFFSET_KM
                    ),
                    "max_intrusion_last_cycle_from_mouth_km": (
                        max_domain + MOUTH_OFFSET_KM
                    ),
                    "final_total_salt_psu_m": result.total_salt_psu_m[-1],
                    "final_right_boundary_salinity_psu": (
                        result.right_boundary_salinity_psu[-1]
                    ),
                }
            )
    return path


def write_last_cycle_series(
    results: dict[tuple[float, str], SimulationResult],
    output_dir: Path,
) -> Path:
    path = output_dir / "last_cycle_timeseries.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "discharge_m3_s",
                "right_boundary",
                "time_h",
                "velocity_m_s",
                "dispersion_m2_s",
                "intrusion_domain_km",
                "intrusion_from_mouth_km",
                "right_boundary_salinity_psu",
                "total_salt_psu_m",
            ]
        )
        for (discharge, boundary), result in sorted(results.items()):
            indices = np.flatnonzero(result.last_cycle_mask())
            for index in indices:
                intrusion_domain = result.intrusion_length_m[index] / 1_000.0
                writer.writerow(
                    [
                        discharge,
                        boundary,
                        result.times_s[index] / 3_600.0,
                        result.velocity_m_s[index],
                        result.dispersion_m2_s[index],
                        intrusion_domain,
                        intrusion_domain + MOUTH_OFFSET_KM,
                        result.right_boundary_salinity_psu[index],
                        result.total_salt_psu_m[index],
                    ]
                )
    return path


def write_cycle_summary(
    results: dict[tuple[float, str], SimulationResult],
    output_dir: Path,
) -> Path:
    """Write cycle-by-cycle metrics, including a periodicity diagnostic."""

    path = output_dir / "cycle_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "discharge_m3_s",
                "right_boundary",
                "cycle",
                "mean_intrusion_domain_km",
                "max_intrusion_domain_km",
                "end_intrusion_domain_km",
                "end_total_salt_psu_m",
                "end_right_boundary_salinity_psu",
                "cycle_end_profile_change_linf_psu",
                "cycle_end_total_salt_relative_change",
            ],
        )
        writer.writeheader()

        for (discharge, boundary), result in sorted(results.items()):
            config = result.config
            steps_per_cycle_float = config.tidal_period_s / config.dt_s
            if not np.isclose(
                steps_per_cycle_float,
                round(steps_per_cycle_float),
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise ValueError(
                    "The tidal period must be an integer multiple of dt_s "
                    "for cycle summaries."
                )
            steps_per_cycle = int(round(steps_per_cycle_float))
            n_complete_cycles = config.n_steps // steps_per_cycle
            previous_profile: np.ndarray | None = None
            previous_salt: float | None = None

            for cycle in range(1, n_complete_cycles + 1):
                start = (cycle - 1) * steps_per_cycle
                end = cycle * steps_per_cycle
                cycle_slice = slice(start, end + 1)
                cycle_times = result.times_s[cycle_slice]
                cycle_intrusion = result.intrusion_length_m[cycle_slice]
                mean_intrusion = np.trapezoid(
                    cycle_intrusion,
                    cycle_times,
                ) / config.tidal_period_s

                stored_index = np.flatnonzero(
                    np.isclose(
                        result.stored_times_s,
                        result.times_s[end],
                        rtol=0.0,
                        atol=1.0e-10,
                    )
                )
                if stored_index.size != 1:
                    raise RuntimeError("Cycle-end profile was not stored uniquely.")
                profile = result.stored_profiles_psu[int(stored_index[0])]
                end_salt = float(result.total_salt_psu_m[end])

                if previous_profile is None:
                    profile_change = np.nan
                    salt_change = np.nan
                else:
                    profile_change = float(np.max(np.abs(profile - previous_profile)))
                    denominator = max(abs(previous_salt), np.finfo(float).eps)
                    salt_change = abs(end_salt - previous_salt) / denominator

                writer.writerow(
                    {
                        "discharge_m3_s": discharge,
                        "right_boundary": boundary,
                        "cycle": cycle,
                        "mean_intrusion_domain_km": mean_intrusion / 1_000.0,
                        "max_intrusion_domain_km": (
                            np.max(cycle_intrusion) / 1_000.0
                        ),
                        "end_intrusion_domain_km": (
                            result.intrusion_length_m[end] / 1_000.0
                        ),
                        "end_total_salt_psu_m": end_salt,
                        "end_right_boundary_salinity_psu": (
                            result.right_boundary_salinity_psu[end]
                        ),
                        "cycle_end_profile_change_linf_psu": profile_change,
                        "cycle_end_total_salt_relative_change": salt_change,
                    }
                )
                previous_profile = profile
                previous_salt = end_salt
    return path


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


def plot_results(
    results: dict[tuple[float, str], SimulationResult],
    output_dir: Path,
) -> list[Path]:
    _style()
    paths: list[Path] = []
    labels = {"neumann": "Neumann homogênea", "danckwerts": "Danckwerts"}
    styles = {"neumann": "-", "danckwerts": "--"}

    for discharge in (10.0, 2.0):
        reference = results[(discharge, "neumann")]
        last_cycle_start_h = (
            reference.times_s[-1] - reference.config.tidal_period_s
        ) / 3_600.0

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for boundary in ("neumann", "danckwerts"):
            result = results[(discharge, boundary)]
            mask = result.last_cycle_mask()
            ax.plot(
                result.times_s[mask] / 3_600.0 - last_cycle_start_h,
                result.intrusion_length_m[mask] / 1_000.0 + MOUTH_OFFSET_KM,
                styles[boundary],
                linewidth=2.2,
                label=labels[boundary],
            )
        ax.set(
            xlabel="Tempo no último ciclo de maré (h)",
            ylabel="Distância da frente salina à foz (km)",
            title=rf"Comprimento de intrusão: $Q={discharge:g}\,\mathrm{{m^3/s}}$",
        )
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"intrusion_Q{discharge:g}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for boundary in ("neumann", "danckwerts"):
            result = results[(discharge, boundary)]
            ax.plot(
                result.x_m / 1_000.0 + MOUTH_OFFSET_KM,
                result.final_profile_psu,
                styles[boundary],
                linewidth=2.2,
                label=labels[boundary],
            )
        ax.axhline(
            reference.config.intrusion_threshold_psu,
            color="0.35",
            linestyle=":",
            linewidth=1.3,
            label=r"$C_{\mathrm{lim}}=0{,}5$ PSU",
        )
        ax.set(
            xlabel="Distância da foz (km)",
            ylabel="Salinidade (PSU)",
            title=rf"Perfil final: $Q={discharge:g}\,\mathrm{{m^3/s}}$",
            xlim=(MOUTH_OFFSET_KM, MOUTH_OFFSET_KM + 50.0),
        )
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"final_profile_Q{discharge:g}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for boundary in ("neumann", "danckwerts"):
            result = results[(discharge, boundary)]
            mask = result.last_cycle_mask()
            ax.plot(
                result.times_s[mask] / 3_600.0 - last_cycle_start_h,
                result.right_boundary_salinity_psu[mask],
                styles[boundary],
                linewidth=2.2,
                label=labels[boundary],
            )
        ax.set(
            xlabel="Tempo no último ciclo de maré (h)",
            ylabel=r"$C(L,t)$ (PSU)",
            title=rf"Salinidade em $x=L$: $Q={discharge:g}\,\mathrm{{m^3/s}}$",
        )
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"right_boundary_Q{discharge:g}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Neumann and Danckwerts conditions at x=L."
    )
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--cells", type=int, default=500)
    parser.add_argument("--dt", type=float, default=60.0, help="Time step in seconds.")
    parser.add_argument(
        "--store-every",
        type=int,
        default=30,
        help="Store one full profile every this many time steps.",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cycles <= 0:
        raise SystemExit("--cycles must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    results = run_comparison(
        cycles=args.cycles,
        n_cells=args.cells,
        dt_s=args.dt,
        store_every_steps=args.store_every,
    )
    summary = write_summary(results, args.output)
    series = write_last_cycle_series(results, args.output)
    cycle_summary = write_cycle_summary(results, args.output)
    figures = plot_results(results, args.output)

    print(f"Summary: {summary}")
    print(f"Last-cycle series: {series}")
    print(f"Cycle summary: {cycle_summary}")
    for (discharge, boundary), result in sorted(results.items()):
        print(
            f"Q={discharge:>4.1f} m3/s | {boundary:11s} | "
            f"mean={result.mean_intrusion_last_cycle_m()/1e3:7.3f} km "
            f"(domain) | max={result.max_intrusion_last_cycle_m()/1e3:7.3f} km"
        )
    print(f"Figures: {len(figures)} files in {args.output}")


if __name__ == "__main__":
    main()
