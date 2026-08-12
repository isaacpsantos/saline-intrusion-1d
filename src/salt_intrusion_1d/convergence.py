"""Separated spatial and temporal refinement for the teaching article.

The finite-horizon experiment is integrated for 60 tidal cycles.  Spatial
refinement keeps ``dt`` fixed, while temporal refinement keeps ``dx`` fixed.
The study reports both self-convergence differences between consecutive
discretizations and errors relative to deliberately finer numerical reference
solutions.  No analytical solution is assumed for the estuarine case.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .article_update import (
    _capture_series,
    compare_transient_domains,
    plot_selected_scenarios,
    run_article_case,
    transient_metrics,
)
from .experiment import MOUTH_OFFSET_KM
from .model import SimulationResult

CRITICAL_DISCHARGE_M3_S = 2.0
SPATIAL_DX_M = (400.0, 200.0, 100.0, 50.0, 25.0)
SPATIAL_FIXED_DT_S = 15.0
TEMPORAL_DT_S = (240.0, 120.0, 60.0, 30.0, 15.0)
TEMPORAL_FIXED_DX_M = 25.0
VERIFICATION_FINE_DX_M = 25.0
VERIFICATION_FINE_DT_S = 15.0
PRODUCTION_DX_M = 12.5
PRODUCTION_DT_S = 7.5
REFERENCE_DX_M = 6.25
REFERENCE_DT_S = 3.75


def observed_order(coarse_error: float, fine_error: float, ratio: float = 2.0) -> float:
    """Return the self-convergence order from two consecutive errors."""

    if coarse_error <= 0 or fine_error <= 0:
        raise ValueError("Errors must be positive.")
    if ratio <= 1:
        raise ValueError("The refinement ratio must exceed one.")
    return float(np.log(coarse_error / fine_error) / np.log(ratio))


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    data = list(rows)
    if not data:
        raise ValueError("At least one row is required.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return path


def _last_cycle_intrusion(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    mask = result.last_cycle_mask()
    time = result.times_s[mask]
    return time - time[0], result.intrusion_length_m[mask]


def consecutive_error(
    coarse: SimulationResult,
    fine: SimulationResult,
) -> dict[str, float]:
    """Compare a discretization with the next finer result."""

    fine_profile_on_coarse = np.interp(
        coarse.x_m,
        fine.x_m,
        fine.final_profile_psu,
    )
    profile_difference = coarse.final_profile_psu - fine_profile_on_coarse

    coarse_time, coarse_intrusion = _last_cycle_intrusion(coarse)
    fine_time, fine_intrusion = _last_cycle_intrusion(fine)
    fine_intrusion_on_coarse = np.interp(
        coarse_time,
        fine_time,
        fine_intrusion,
    )
    intrusion_difference = coarse_intrusion - fine_intrusion_on_coarse

    coarse_capture_time, coarse_capture = _capture_series(coarse)
    fine_capture_time, fine_capture = _capture_series(fine)
    fine_capture_on_coarse = np.interp(
        coarse_capture_time,
        fine_capture_time,
        fine_capture,
    )
    capture_difference = coarse_capture - fine_capture_on_coarse

    coarse_metrics = transient_metrics(coarse)
    fine_metrics = transient_metrics(fine)
    return {
        "final_profile_rms_error_psu": float(
            np.sqrt(np.mean(profile_difference**2))
        ),
        "final_profile_linf_error_psu": float(
            np.max(np.abs(profile_difference))
        ),
        "last_cycle_intrusion_rms_error_km": float(
            np.sqrt(np.mean(intrusion_difference**2)) / 1_000.0
        ),
        "last_cycle_intrusion_linf_error_km": float(
            np.max(np.abs(intrusion_difference)) / 1_000.0
        ),
        "last_cycle_capture_rms_error_psu": float(
            np.sqrt(np.mean(capture_difference**2))
        ),
        "last_cycle_capture_linf_error_psu": float(
            np.max(np.abs(capture_difference))
        ),
        "mean_intrusion_difference_km": abs(
            coarse_metrics.mean_intrusion_from_mouth_km
            - fine_metrics.mean_intrusion_from_mouth_km
        ),
        "max_intrusion_difference_km": abs(
            coarse_metrics.max_intrusion_from_mouth_km
            - fine_metrics.max_intrusion_from_mouth_km
        ),
        "capture_mean_difference_psu": abs(
            coarse_metrics.capture_mean_salinity_psu
            - fine_metrics.capture_mean_salinity_psu
        ),
        "capture_max_difference_psu": abs(
            coarse_metrics.capture_max_salinity_psu
            - fine_metrics.capture_max_salinity_psu
        ),
    }


def reference_error(
    approximation: SimulationResult,
    reference: SimulationResult,
) -> dict[str, float]:
    """Compare one approximation with a finer numerical reference solution.

    The reference profile and last-cycle time series are interpolated onto the
    approximation grids.  The returned quantities are errors relative to a
    numerical reference, not errors relative to an exact solution.
    """

    if not np.isclose(
        approximation.config.length_m,
        reference.config.length_m,
    ):
        raise ValueError("Approximation and reference must use the same domain.")
    if not np.isclose(
        approximation.config.final_time_s,
        reference.config.final_time_s,
    ):
        raise ValueError(
            "Approximation and reference must use the same final time."
        )

    errors = consecutive_error(approximation, reference)
    return {
        key.replace("_error_", "_reference_error_").replace(
            "_difference_",
            "_reference_error_",
        ): value
        for key, value in errors.items()
    }


REFERENCE_ERROR_COLUMNS = tuple(
    column.replace("_error_", "_reference_error_").replace(
        "_difference_",
        "_reference_error_",
    )
    for column in (
        "final_profile_rms_error_psu",
        "final_profile_linf_error_psu",
        "last_cycle_intrusion_rms_error_km",
        "last_cycle_intrusion_linf_error_km",
        "last_cycle_capture_rms_error_psu",
        "last_cycle_capture_linf_error_psu",
        "mean_intrusion_difference_km",
        "max_intrusion_difference_km",
        "capture_mean_difference_psu",
        "capture_max_difference_psu",
    )
)


def add_reference_errors(
    rows: list[dict[str, object]],
    results: list[SimulationResult],
    reference: SimulationResult,
) -> list[dict[str, object]]:
    """Add reference-solution errors and their apparent pairwise orders."""

    if len(rows) != len(results):
        raise ValueError("Rows and results must have the same length.")
    reference_errors = [
        reference_error(result, reference)
        for result in results
    ]
    for row, errors in zip(rows, reference_errors):
        row.update(errors)
    for column in REFERENCE_ERROR_COLUMNS:
        order_column = column.replace(
            "_reference_error_",
            "_reference_observed_order_",
        )
        for index, row in enumerate(rows):
            if index < len(rows) - 1:
                row[order_column] = observed_order(
                    reference_errors[index][column],
                    reference_errors[index + 1][column],
                )
            else:
                row[order_column] = ""
    return rows


ERROR_COLUMNS = (
    "final_profile_rms_error_psu",
    "final_profile_linf_error_psu",
    "last_cycle_intrusion_rms_error_km",
    "last_cycle_intrusion_linf_error_km",
    "last_cycle_capture_rms_error_psu",
    "last_cycle_capture_linf_error_psu",
    "mean_intrusion_difference_km",
    "max_intrusion_difference_km",
    "capture_mean_difference_psu",
    "capture_max_difference_psu",
)


def refinement_rows(
    results: list[SimulationResult],
    *,
    varied_parameter: str,
) -> list[dict[str, object]]:
    """Build metrics, consecutive errors and observed orders."""

    if varied_parameter not in {"dx_m", "dt_s"}:
        raise ValueError("varied_parameter must be 'dx_m' or 'dt_s'.")
    if len(results) < 3:
        raise ValueError("At least three refinement levels are required.")

    rows: list[dict[str, object]] = []
    for result in results:
        metrics = transient_metrics(result)
        rows.append(
            {
                varied_parameter: getattr(result.config, varied_parameter),
                "dx_m": result.config.dx_m,
                "dt_s": result.config.dt_s,
                "mean_intrusion_from_mouth_km": (
                    metrics.mean_intrusion_from_mouth_km
                ),
                "max_intrusion_from_mouth_km": (
                    metrics.max_intrusion_from_mouth_km
                ),
                "capture_mean_salinity_psu": (
                    metrics.capture_mean_salinity_psu
                ),
                "capture_max_salinity_psu": (
                    metrics.capture_max_salinity_psu
                ),
                "capture_fraction_above_threshold": (
                    metrics.capture_fraction_above_threshold
                ),
            }
        )

    errors = [
        consecutive_error(coarse, fine)
        for coarse, fine in zip(results[:-1], results[1:])
    ]
    for index, error in enumerate(errors):
        rows[index].update(error)
    for column in ERROR_COLUMNS:
        rows[-1][column] = ""
        order_column = column.replace("error", "observed_order").replace(
            "difference",
            "observed_order",
        )
        for index in range(len(rows)):
            if index < len(errors) - 1:
                rows[index][order_column] = observed_order(
                    errors[index][column],
                    errors[index + 1][column],
                )
            else:
                rows[index][order_column] = ""
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def plot_convergence_errors(
    spatial_rows: list[dict[str, object]],
    temporal_rows: list[dict[str, object]],
    path: Path,
) -> Path:
    """Plot self-convergence errors and first-order reference slopes."""

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    panels = (
        (
            axes[0],
            spatial_rows[:-1],
            "dx_m",
            r"$\Delta x$ (m)",
            "Refinamento espacial",
        ),
        (
            axes[1],
            temporal_rows[:-1],
            "dt_s",
            r"$\Delta t$ (s)",
            "Refinamento temporal",
        ),
    )
    for axis, rows, parameter, xlabel, title in panels:
        resolution = np.array([float(row[parameter]) for row in rows])
        profile = np.array(
            [float(row["final_profile_rms_error_psu"]) for row in rows]
        )
        capture = np.array(
            [float(row["last_cycle_capture_rms_error_psu"]) for row in rows]
        )
        axis.loglog(resolution, profile, "o-", linewidth=1.8, label="Perfil final")
        axis.loglog(
            resolution,
            capture,
            "s-",
            linewidth=1.8,
            label="Captação no último ciclo",
        )
        reference = profile[-1] * resolution / resolution[-1]
        axis.loglog(
            resolution,
            reference,
            ":",
            color="0.25",
            linewidth=1.4,
            label="inclinação 1",
        )
        axis.set(xlabel=xlabel, ylabel="Erro RMS (PSU)", title=title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_reference_errors(
    spatial_rows: list[dict[str, object]],
    temporal_rows: list[dict[str, object]],
    joint_rows: list[dict[str, object]],
    path: Path,
) -> Path:
    """Plot errors measured against the three numerical references."""

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    panels = (
        (
            axes[0],
            spatial_rows,
            "dx_m",
            r"$\Delta x$ (m)",
            "Referência espacial",
        ),
        (
            axes[1],
            temporal_rows,
            "dt_s",
            r"$\Delta t$ (s)",
            "Referência temporal",
        ),
        (
            axes[2],
            joint_rows,
            "dx_m",
            r"$\Delta x$ (m)",
            "Referência conjunta",
        ),
    )
    for axis, rows, parameter, xlabel, title in panels:
        resolution = np.array([float(row[parameter]) for row in rows])
        profile = np.array(
            [
                float(row["final_profile_rms_reference_error_psu"])
                for row in rows
            ]
        )
        capture = np.array(
            [
                float(row["last_cycle_capture_rms_reference_error_psu"])
                for row in rows
            ]
        )
        axis.loglog(resolution, profile, "o-", linewidth=1.8, label="Perfil final")
        axis.loglog(
            resolution,
            capture,
            "s-",
            linewidth=1.8,
            label="Captação no último ciclo",
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(resolution)
        axis.set_xticklabels([f"{value:g}" for value in resolution])
        axis.minorticks_off()
        axis.set(xlabel=xlabel, ylabel="Erro RMS (PSU)", title=title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_refinement_metrics(
    spatial_rows: list[dict[str, object]],
    temporal_rows: list[dict[str, object]],
    path: Path,
) -> Path:
    """Plot the key physical metrics along each refinement sequence."""

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    panels = (
        (axes[0, 0], spatial_rows, "dx_m", r"$\Delta x$ (m)", "Espacial"),
        (axes[0, 1], temporal_rows, "dt_s", r"$\Delta t$ (s)", "Temporal"),
    )
    for axis, rows, parameter, xlabel, title in panels:
        resolution = np.array([float(row[parameter]) for row in rows])
        axis.semilogx(
            resolution,
            [float(row["mean_intrusion_from_mouth_km"]) for row in rows],
            "o-",
            label="média",
        )
        axis.semilogx(
            resolution,
            [float(row["max_intrusion_from_mouth_km"]) for row in rows],
            "s-",
            label="máxima",
        )
        axis.invert_xaxis()
        axis.set(
            xlabel=xlabel,
            ylabel="Distância da foz (km)",
            title=f"Intrusão — refinamento {title.lower()}",
        )
        axis.legend()

    panels = (
        (axes[1, 0], spatial_rows, "dx_m", r"$\Delta x$ (m)", "espacial"),
        (axes[1, 1], temporal_rows, "dt_s", r"$\Delta t$ (s)", "temporal"),
    )
    for axis, rows, parameter, xlabel, title in panels:
        resolution = np.array([float(row[parameter]) for row in rows])
        axis.semilogx(
            resolution,
            [float(row["capture_mean_salinity_psu"]) for row in rows],
            "o-",
            label="média",
        )
        axis.semilogx(
            resolution,
            [float(row["capture_max_salinity_psu"]) for row in rows],
            "s-",
            label="máxima",
        )
        axis.axhline(0.5, color="tab:red", linestyle=":", linewidth=1.2)
        axis.invert_xaxis()
        axis.set(
            xlabel=xlabel,
            ylabel="Salinidade (PSU)",
            title=f"Captação — refinamento {title}",
        )
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_latex_tables(
    spatial_rows: list[dict[str, object]],
    temporal_rows: list[dict[str, object]],
    path: Path,
) -> Path:
    """Write compact manuscript-ready LaTeX tables."""

    def table(rows: list[dict[str, object]], parameter: str, symbol: str) -> str:
        lines = [
            r"\begin{tabular}{rrrrrr}",
            r"\hline",
            (
                f"${symbol}$ & $\\overline{{L}}_s$ & $L_s^{{\\max}}$ & "
                r"$\overline{C}_{\mathrm{cap}}$ & $E_{\mathrm{RMS}}$ & $p$ \\"
            ),
            r"\hline",
        ]
        order_key = "final_profile_rms_observed_order_psu"
        for row in rows:
            error = row["final_profile_rms_error_psu"]
            order = row[order_key]
            error_text = "--" if error == "" else f"{float(error):.3e}"
            order_text = "--" if order == "" else f"{float(order):.2f}"
            lines.append(
                f"{float(row[parameter]):.0f} & "
                f"{float(row['mean_intrusion_from_mouth_km']):.3f} & "
                f"{float(row['max_intrusion_from_mouth_km']):.3f} & "
                f"{float(row['capture_mean_salinity_psu']):.3f} & "
                f"{error_text} & {order_text} \\\\"
            )
        lines.extend((r"\hline", r"\end{tabular}"))
        return "\n".join(lines)

    content = (
        "% Generated by salt_intrusion_1d.convergence\n"
        "% Errors compare each row with the next finer discretization.\n\n"
        + table(spatial_rows, "dx_m", r"\Delta x\,(\mathrm{m})")
        + "\n\n"
        + table(temporal_rows, "dt_s", r"\Delta t\,(\mathrm{s})")
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def write_reference_latex_tables(
    spatial_rows: list[dict[str, object]],
    temporal_rows: list[dict[str, object]],
    joint_rows: list[dict[str, object]],
    path: Path,
) -> Path:
    """Write compact LaTeX tables for reference-solution errors."""

    def table(
        rows: list[dict[str, object]],
        parameter: str,
        symbol: str,
    ) -> str:
        lines = [
            r"\begin{tabular}{rrrrrr}",
            r"\hline",
            (
                f"${symbol}$ & $\\overline{{L}}_s$ & "
                r"$\overline{C}_{\mathrm{cap}}$ & "
                r"$E_{\mathrm{RMS}}^{\mathrm{ref}}$ & "
                r"$E_{\infty}^{\mathrm{ref}}$ & "
                r"$E_{\overline{L}}^{\mathrm{ref}}$ \\"
            ),
            r"\hline",
        ]
        for row in rows:
            lines.append(
                f"{float(row[parameter]):.2f} & "
                f"{float(row['mean_intrusion_from_mouth_km']):.3f} & "
                f"{float(row['capture_mean_salinity_psu']):.3f} & "
                f"{float(row['final_profile_rms_reference_error_psu']):.3e} & "
                f"{float(row['final_profile_linf_reference_error_psu']):.3e} & "
                f"{float(row['mean_intrusion_reference_error_km']):.3e} \\\\"
            )
        lines.extend((r"\hline", r"\end{tabular}"))
        return "\n".join(lines)

    content = (
        "% Generated by salt_intrusion_1d.convergence\n"
        "% Errors are relative to finer numerical reference solutions.\n\n"
        + table(spatial_rows, "dx_m", r"\Delta x\,(\mathrm{m})")
        + "\n\n"
        + table(temporal_rows, "dt_s", r"\Delta t\,(\mathrm{s})")
        + "\n\n"
        + table(joint_rows, "dx_m", r"\Delta x\,(\mathrm{m})")
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run separated spatial and temporal convergence studies."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_convergence_v0.9"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[float, float, float, float], SimulationResult] = {}

    def run(
        discharge: float,
        dx_m: float,
        dt_s: float,
        length_km: float = 50.0,
    ) -> SimulationResult:
        key = (discharge, dx_m, dt_s, length_km)
        if key not in cache:
            cache[key] = run_article_case(
                discharge,
                length_km,
                dx_m=dx_m,
                dt_s=dt_s,
                store_every_steps=max(1, int(round(1_800.0 / dt_s))),
            )
        return cache[key]

    spatial_results = [
        run(CRITICAL_DISCHARGE_M3_S, dx_m, SPATIAL_FIXED_DT_S)
        for dx_m in SPATIAL_DX_M
    ]
    temporal_results = [
        run(CRITICAL_DISCHARGE_M3_S, TEMPORAL_FIXED_DX_M, dt_s)
        for dt_s in TEMPORAL_DT_S
    ]
    spatial_reference = run(
        CRITICAL_DISCHARGE_M3_S,
        REFERENCE_DX_M,
        SPATIAL_FIXED_DT_S,
    )
    temporal_reference = run(
        CRITICAL_DISCHARGE_M3_S,
        TEMPORAL_FIXED_DX_M,
        REFERENCE_DT_S,
    )
    spatial = refinement_rows(spatial_results, varied_parameter="dx_m")
    temporal = refinement_rows(temporal_results, varied_parameter="dt_s")
    add_reference_errors(spatial, spatial_results, spatial_reference)
    add_reference_errors(temporal, temporal_results, temporal_reference)
    _write_rows(args.output / "spatial_refinement.csv", spatial)
    _write_rows(args.output / "temporal_refinement.csv", temporal)
    plot_convergence_errors(
        spatial,
        temporal,
        args.output / "convergence_errors.png",
    )
    plot_refinement_metrics(
        spatial,
        temporal,
        args.output / "refinement_metrics.png",
    )
    write_latex_tables(
        spatial,
        temporal,
        args.output / "convergence_tables.tex",
    )

    joint_results = [
        run(
            CRITICAL_DISCHARGE_M3_S,
            VERIFICATION_FINE_DX_M,
            VERIFICATION_FINE_DT_S,
        ),
        run(
            CRITICAL_DISCHARGE_M3_S,
            PRODUCTION_DX_M,
            PRODUCTION_DT_S,
        ),
    ]
    joint_reference = run(
        CRITICAL_DISCHARGE_M3_S,
        REFERENCE_DX_M,
        REFERENCE_DT_S,
    )
    joint_rows = []
    for result in joint_results:
        row = asdict(transient_metrics(result))
        row.update({"dx_m": result.config.dx_m, "dt_s": result.config.dt_s})
        joint_rows.append(row)
    add_reference_errors(joint_rows, joint_results, joint_reference)
    reference_row = asdict(transient_metrics(joint_reference))
    reference_row.update(
        {
            "dx_m": joint_reference.config.dx_m,
            "dt_s": joint_reference.config.dt_s,
            "reference_role": "joint_reference",
        }
    )
    for column in REFERENCE_ERROR_COLUMNS:
        reference_row[column] = 0.0
        reference_row[
            column.replace(
                "_reference_error_",
                "_reference_observed_order_",
            )
        ] = ""
    for row, role in zip(
        joint_rows,
        ("verification_fine", "adopted_article_mesh"),
    ):
        row["reference_role"] = role
    _write_rows(
        args.output / "joint_reference_errors.csv",
        [*joint_rows, reference_row],
    )
    plot_reference_errors(
        spatial,
        temporal,
        joint_rows,
        args.output / "reference_errors.png",
    )
    write_reference_latex_tables(
        spatial,
        temporal,
        joint_rows,
        args.output / "reference_tables.tex",
    )

    reference_rows = []
    for role, result in (
        ("spatial_reference", spatial_reference),
        ("temporal_reference", temporal_reference),
        ("joint_reference", joint_reference),
    ):
        row = asdict(transient_metrics(result))
        row.update(
            {
                "reference_role": role,
                "dx_m": result.config.dx_m,
                "dt_s": result.config.dt_s,
            }
        )
        reference_rows.append(row)
    _write_rows(args.output / "reference_mesh_summary.csv", reference_rows)

    production_results = {
        discharge: run(
            discharge,
            PRODUCTION_DX_M,
            PRODUCTION_DT_S,
        )
        for discharge in (10.0, 2.0)
    }
    production_rows = [
        asdict(transient_metrics(production_results[discharge]))
        for discharge in (10.0, 2.0)
    ]
    _write_rows(args.output / "production_mesh_summary.csv", production_rows)
    plot_selected_scenarios(
        production_results,
        args.output / "production_scenarios.png",
    )

    domain_fine = run(
        CRITICAL_DISCHARGE_M3_S,
        PRODUCTION_DX_M,
        PRODUCTION_DT_S,
        60.0,
    )
    domain_rows = [
        asdict(transient_metrics(production_results[2.0])),
        asdict(transient_metrics(domain_fine)),
    ]
    _write_rows(args.output / "production_domain_check_Q2.csv", domain_rows)
    domain_comparison = compare_transient_domains(
        {
            50.0: production_results[2.0],
            60.0: domain_fine,
        }
    )
    _write_rows(
        args.output / "production_domain_comparison_Q2.csv",
        [asdict(domain_comparison)],
    )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
