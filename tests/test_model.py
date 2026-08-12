from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from salt_intrusion_1d import (
    PeriodicityCriteria,
    SimulationConfig,
    boundary_fluxes,
    intrusion_length,
    right_boundary_relation,
    simulate,
    simulate_until_periodic,
    solve_periodic_fixed_point,
)
from salt_intrusion_1d.periodicity import (
    boundary_peak_diagnostic,
    temporal_refinement_metrics,
    velocity_sign_changes_in_cycle,
)
from salt_intrusion_1d.domain_sensitivity import (
    adjacent_domain_metrics,
    duration_above_threshold,
)
from salt_intrusion_1d.article_update import (
    ARTICLE_BASE_DISPERSION_M2_S,
    ARTICLE_DISPERSION_KAPPA,
    TransientDomainComparison,
    TransientMetrics,
    article_config,
    configuration_is_acceptable,
)
from salt_intrusion_1d.convergence import observed_order, reference_error


class BoundaryTests(unittest.TestCase):
    def test_boundary_fluxes_match_danckwerts_condition(self) -> None:
        profile = np.array([12.0, 8.0, 5.0, 4.0])
        velocity_value = -0.2
        diffusivity = 50.0
        dx = 100.0
        theta, eta = right_boundary_relation(
            velocity_value, diffusivity, dx, 2.0, "danckwerts"
        )
        profile[-1] = theta * profile[-2] + eta
        _, _, _, right_flux = boundary_fluxes(
            profile, velocity_value, diffusivity, dx
        )
        self.assertAlmostEqual(right_flux, velocity_value * 2.0)

    def test_danckwerts_relation_is_weighted_average(self) -> None:
        theta, eta = right_boundary_relation(
            velocity_m_s=-0.2,
            dispersion_m2_s=50.0,
            dx_m=100.0,
            upstream_salinity_psu=2.0,
            boundary="danckwerts",
        )
        self.assertAlmostEqual(theta, 50.0 / 70.0)
        self.assertAlmostEqual(eta, 20.0 * 2.0 / 70.0)
        self.assertAlmostEqual(theta + eta / 2.0, 1.0)

    def test_neumann_used_for_nonnegative_velocity(self) -> None:
        theta, eta = right_boundary_relation(
            velocity_m_s=0.2,
            dispersion_m2_s=50.0,
            dx_m=100.0,
            upstream_salinity_psu=2.0,
            boundary="danckwerts",
        )
        self.assertEqual((theta, eta), (1.0, 0.0))

    def test_danckwerts_flux_identity(self) -> None:
        velocity = -0.2
        diffusivity = 50.0
        dx = 100.0
        upstream = 2.0
        interior = 5.0
        theta, eta = right_boundary_relation(
            velocity,
            diffusivity,
            dx,
            upstream,
            "danckwerts",
        )
        boundary = theta * interior + eta
        numerical_flux = velocity * boundary - diffusivity * (
            boundary - interior
        ) / dx
        self.assertAlmostEqual(numerical_flux, velocity * upstream)


class SolverTests(unittest.TestCase):
    @staticmethod
    def base_config() -> SimulationConfig:
        return SimulationConfig(
            length_m=1_000.0,
            n_cells=40,
            final_time_s=200.0,
            dt_s=10.0,
            river_velocity_m_s=0.2,
            tidal_velocity_amplitude_m_s=0.0,
            tidal_period_s=1_000.0,
            dispersion_mode="constant",
            base_dispersion_m2_s=10.0,
            sea_salinity_mean_psu=5.0,
            sea_salinity_amplitude_psu=0.0,
            upstream_salinity_psu=5.0,
            right_boundary="danckwerts",
            intrusion_threshold_psu=0.5,
            store_every_steps=3,
        )

    def test_constant_solution_is_preserved_by_danckwerts(self) -> None:
        config = self.base_config()
        initial = np.full(config.n_cells + 1, 5.0)
        result = simulate(config, initial)
        np.testing.assert_allclose(result.final_profile_psu, 5.0, atol=1.0e-12)

    def test_constant_solution_is_preserved_by_neumann(self) -> None:
        config = replace(self.base_config(), right_boundary="neumann")
        initial = np.full(config.n_cells + 1, 5.0)
        result = simulate(config, initial)
        np.testing.assert_allclose(result.final_profile_psu, 5.0, atol=1.0e-12)

    def test_nonnegative_bounded_solution_with_velocity_reversal(self) -> None:
        config = replace(
            self.base_config(),
            final_time_s=1_000.0,
            river_velocity_m_s=0.02,
            tidal_velocity_amplitude_m_s=0.2,
            upstream_salinity_psu=0.0,
            sea_salinity_mean_psu=12.0,
            sea_salinity_amplitude_psu=4.0,
        )
        result = simulate(config)
        self.assertGreaterEqual(float(result.stored_profiles_psu.min()), -1.0e-12)
        self.assertLessEqual(float(result.stored_profiles_psu.max()), 16.0 + 1.0e-12)
        self.assertTrue(np.any(result.velocity_m_s < 0))
        self.assertTrue(np.any(result.velocity_m_s > 0))

    def test_final_profile_is_always_stored(self) -> None:
        config = replace(self.base_config(), store_every_steps=7)
        result = simulate(config)
        self.assertEqual(result.stored_times_s[-1], result.times_s[-1])
        self.assertEqual(
            result.stored_profiles_psu.shape[1],
            config.n_cells + 1,
        )

    def test_cycle_end_profiles_are_stored(self) -> None:
        config = replace(
            self.base_config(),
            final_time_s=2_000.0,
            store_every_steps=777,
        )
        result = simulate(config)
        np.testing.assert_array_equal(
            result.stored_times_s,
            np.array([0.0, 1_000.0, 2_000.0]),
        )

    def test_discrete_salt_balance_is_satisfied_to_roundoff(self) -> None:
        config = replace(
            self.base_config(),
            final_time_s=1_000.0,
            river_velocity_m_s=0.02,
            tidal_velocity_amplitude_m_s=0.2,
            upstream_salinity_psu=0.0,
        )
        result = simulate(config)
        residual = result.discrete_balance_residual_psu_m3_s[1:]
        scale = config.cross_section_area_m2 * (
            np.abs(result.left_numerical_flux_psu_m_s[1:])
            + np.abs(result.right_numerical_flux_psu_m_s[1:])
        )
        relative = np.abs(residual) / np.maximum(scale, 1.0)
        self.assertLess(float(np.max(relative)), 1.0e-11)

    def test_trapezoidal_content_includes_cross_section_area(self) -> None:
        config = self.base_config()
        result = simulate(config)
        np.testing.assert_allclose(
            result.trapezoidal_salt_content_psu_m3,
            config.cross_section_area_m2 * result.total_salt_psu_m,
        )


class MetricTests(unittest.TestCase):
    def test_intrusion_length_interpolates_threshold(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        concentration = np.array([2.0, 1.0, 0.0])
        value = intrusion_length(concentration, x, threshold_psu=0.5)
        self.assertAlmostEqual(value, 1.5)

    def test_duration_above_threshold_interpolates_crossings(self) -> None:
        times = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([0.0, 2.0, 2.0, 0.0])
        duration = duration_above_threshold(times, values, threshold=1.0)
        self.assertAlmostEqual(duration, 2.0)

    def test_duration_above_threshold_validates_time_grid(self) -> None:
        with self.assertRaises(ValueError):
            duration_above_threshold([0.0, 0.0], [0.0, 1.0], 0.5)

    def test_adjacent_domain_metrics_requires_two_results(self) -> None:
        with self.assertRaises(ValueError):
            adjacent_domain_metrics({})


class ArticleUpdateTests(unittest.TestCase):
    @staticmethod
    def metrics(front_margin_km: float = 10.0) -> TransientMetrics:
        return TransientMetrics(
            discharge_m3_s=2.0,
            length_km=50.0,
            cycles=60,
            base_dispersion_m2_s=ARTICLE_BASE_DISPERSION_M2_S,
            dispersion_kappa=ARTICLE_DISPERSION_KAPPA,
            dispersion_mode="velocity_dependent",
            mean_intrusion_from_mouth_km=44.85,
            max_intrusion_from_mouth_km=47.34,
            distance_front_to_boundary_km=front_margin_km,
            capture_mean_salinity_psu=0.87,
            capture_min_salinity_psu=0.60,
            capture_max_salinity_psu=1.18,
            capture_time_above_threshold_h=12.4,
            capture_fraction_above_threshold=1.0,
        )

    @staticmethod
    def comparison(
        intrusion_difference_km: float = 0.064,
    ) -> TransientDomainComparison:
        return TransientDomainComparison(
            discharge_m3_s=2.0,
            short_domain_km=50.0,
            long_domain_km=60.0,
            capture_mean_difference_psu=0.0023,
            capture_linf_difference_psu=0.004,
            mean_intrusion_difference_km=intrusion_difference_km,
            max_intrusion_difference_km=intrusion_difference_km,
            common_profile_linf_psu=0.01,
        )

    def test_article_config_uses_selected_parameters(self) -> None:
        config = article_config(2.0)
        self.assertEqual(
            config.base_dispersion_m2_s,
            ARTICLE_BASE_DISPERSION_M2_S,
        )
        self.assertEqual(config.dispersion_kappa, ARTICLE_DISPERSION_KAPPA)
        self.assertEqual(config.n_cells, 4_000)
        self.assertEqual(config.dt_s, 7.5)
        self.assertEqual(config.final_time_s, 60.0 * config.tidal_period_s)

    def test_selected_configuration_satisfies_acceptance_logic(self) -> None:
        self.assertTrue(
            configuration_is_acceptable(
                self.metrics(),
                self.comparison(),
            )
        )

    def test_acceptance_rejects_material_domain_dependence(self) -> None:
        self.assertFalse(
            configuration_is_acceptable(
                self.metrics(),
                self.comparison(intrusion_difference_km=0.11),
            )
        )


class ConvergenceTests(unittest.TestCase):
    def test_observed_order_recovers_first_order(self) -> None:
        self.assertAlmostEqual(observed_order(0.2, 0.1), 1.0)

    def test_observed_order_recovers_second_order(self) -> None:
        self.assertAlmostEqual(observed_order(0.04, 0.01), 2.0)

    def test_observed_order_rejects_invalid_errors(self) -> None:
        with self.assertRaises(ValueError):
            observed_order(0.0, 0.1)

    def test_reference_error_vanishes_for_identical_results(self) -> None:
        result = simulate(SolverTests.base_config())
        errors = reference_error(result, result)
        for value in errors.values():
            self.assertAlmostEqual(value, 0.0)

    def test_reference_error_requires_same_domain(self) -> None:
        coarse = simulate(SolverTests.base_config())
        fine_config = replace(
            SolverTests.base_config(),
            length_m=2_000.0,
            n_cells=80,
        )
        fine = simulate(fine_config)
        with self.assertRaises(ValueError):
            reference_error(coarse, fine)


class PeriodicityTests(unittest.TestCase):
    def test_direct_fixed_point_recovers_constant_periodic_solution(self) -> None:
        config = SolverTests.base_config()
        result = solve_periodic_fixed_point(
            config,
            rtol=1.0e-11,
            atol=1.0e-13,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.linear_solver_info, 0)
        self.assertLess(result.residual_linf_psu, 1.0e-9)
        np.testing.assert_allclose(
            result.last_cycle.stored_profiles_psu,
            5.0,
            atol=1.0e-9,
        )

    def test_physical_simulation_rejects_negative_initial_salinity(self) -> None:
        config = SolverTests.base_config()
        initial = np.full(config.n_cells + 1, 5.0)
        initial[-1] = -1.0
        with self.assertRaises(ValueError):
            simulate(config, initial)

    def test_constant_solution_reaches_automatic_stopping_criterion(self) -> None:
        config = SolverTests.base_config()
        initial = np.full(config.n_cells + 1, 5.0)
        criteria = PeriodicityCriteria(
            profile_linf_tolerance_psu=1.0e-10,
            relative_salt_tolerance=1.0e-10,
            mean_intrusion_tolerance_m=1.0e-10,
            min_cycles=2,
            consecutive_cycles=2,
            max_cycles=10,
        )
        result = simulate_until_periodic(config, criteria, initial)
        self.assertTrue(result.converged)
        self.assertEqual(result.cycles_completed, 3)
        self.assertEqual(result.diagnostics[-1].convergence_streak, 2)

    def test_maximum_cycle_limit_is_reported(self) -> None:
        config = SolverTests.base_config()
        criteria = PeriodicityCriteria(
            profile_linf_tolerance_psu=1.0e-30,
            relative_salt_tolerance=1.0e-30,
            mean_intrusion_tolerance_m=1.0e-30,
            min_cycles=2,
            consecutive_cycles=2,
            max_cycles=2,
        )
        result = simulate_until_periodic(config, criteria)
        self.assertFalse(result.converged)
        self.assertEqual(result.cycles_completed, 2)

    def test_velocity_sign_changes_are_found_analytically(self) -> None:
        config = replace(
            SolverTests.base_config(),
            river_velocity_m_s=0.1,
            tidal_velocity_amplitude_m_s=0.2,
        )
        changes = velocity_sign_changes_in_cycle(config)
        self.assertEqual(
            tuple(direction for _, direction in changes),
            ("negative_to_positive", "positive_to_negative"),
        )
        self.assertAlmostEqual(changes[0][0], config.tidal_period_s / 12.0)
        self.assertAlmostEqual(changes[1][0], 5.0 * config.tidal_period_s / 12.0)

    def test_peak_diagnostic_uses_positive_to_negative_crossing(self) -> None:
        config = replace(
            SolverTests.base_config(),
            final_time_s=1_000.0,
            river_velocity_m_s=0.02,
            tidal_velocity_amplitude_m_s=0.2,
        )
        criteria = PeriodicityCriteria(
            min_cycles=2,
            consecutive_cycles=1,
            max_cycles=2,
        )
        result = simulate_until_periodic(config, criteria)
        diagnostic = boundary_peak_diagnostic(result)
        changes = velocity_sign_changes_in_cycle(config)
        self.assertEqual(
            diagnostic.positive_to_negative_crossing_s,
            changes[1][0],
        )

    def test_temporal_metrics_require_two_results(self) -> None:
        with self.assertRaises(ValueError):
            temporal_refinement_metrics({})


if __name__ == "__main__":
    unittest.main()
