# Tests for pystrata logging infrastructure

import logging

import numpy as np
import pytest

import pystrata


class TestLoggingInfrastructure:
    """Test the enable_logging / disable_logging functions."""

    def test_enable_logging_default(self, caplog):
        """Test that enable_logging enables INFO level logging."""
        pystrata.enable_logging()

        try:
            logger = logging.getLogger("pystrata")
            assert logger.level == logging.INFO
        finally:
            pystrata.disable_logging()

    def test_enable_logging_debug(self, caplog):
        """Test that enable_logging with DEBUG level works."""
        pystrata.enable_logging("DEBUG")

        try:
            logger = logging.getLogger("pystrata")
            assert logger.level == logging.DEBUG
        finally:
            pystrata.disable_logging()

    def test_enable_logging_string_level(self, caplog):
        """Test that string log levels work."""
        pystrata.enable_logging("WARNING")

        try:
            logger = logging.getLogger("pystrata")
            assert logger.level == logging.WARNING
        finally:
            pystrata.disable_logging()

    def test_disable_logging(self, caplog):
        """Test that disable_logging removes handler and sets WARNING level."""
        pystrata.enable_logging()
        pystrata.disable_logging()

        logger = logging.getLogger("pystrata")
        assert logger.level == logging.WARNING

    def test_logging_silent_by_default(self, caplog):
        """Test that pystrata is silent by default (no output)."""
        # Get a fresh state
        pystrata.disable_logging()

        # No messages should be captured at INFO level since logging is disabled
        with caplog.at_level(logging.INFO, logger="pystrata"):
            pystrata.propagation.LinearElasticCalculator()
            # Would need a motion to run, but just instantiating shouldn't log

        # Check no pystrata logs were captured
        pystrata_records = [r for r in caplog.records if r.name.startswith("pystrata")]
        assert len(pystrata_records) == 0


class TestEQLLogging:
    """Test that EQL calculator logs iteration info."""

    @pytest.fixture
    def simple_profile(self):
        """Create a simple profile for testing."""
        return pystrata.site.Profile(
            [
                pystrata.site.Layer(
                    pystrata.site.DarendeliSoilType(
                        18.0, plas_index=30, ocr=1, stress_mean=50
                    ),
                    10,
                    200,
                ),
                pystrata.site.Layer(
                    pystrata.site.SoilType("Rock", 22.0, damping=0.01),
                    0,
                    800,
                ),
            ]
        )

    @pytest.fixture
    def simple_motion(self):
        """Create a simple motion for testing."""
        # Create a simple synthetic motion
        times = np.linspace(0, 5, 500)
        accels = 0.1 * np.sin(2 * np.pi * 1.0 * times) * np.exp(-times / 2)
        return pystrata.motion.TimeSeriesMotion(
            filename="test",
            description="synthetic motion",
            time_step=times[1] - times[0],
            accels=accels,
        )

    def test_eql_logs_convergence(self, caplog, simple_profile, simple_motion):
        """Test that EQL calculator logs convergence info."""
        pystrata.enable_logging("INFO")

        try:
            with caplog.at_level(logging.INFO, logger="pystrata"):
                calc = pystrata.propagation.EquivalentLinearCalculator()
                loc = simple_profile.location("outcrop", index=-1)
                calc(simple_motion, simple_profile, loc)

            # Check that convergence message was logged
            pystrata_records = [
                r for r in caplog.records if r.name.startswith("pystrata")
            ]
            convergence_msgs = [
                r
                for r in pystrata_records
                if "converged" in r.message.lower() or "EQL" in r.message
            ]
            assert len(convergence_msgs) > 0, "Expected EQL convergence log message"

        finally:
            pystrata.disable_logging()


class TestTimeDomainLogging:
    """Test that time-domain propagation logs info."""

    def test_propagate_time_domain_logs(self, caplog):
        """Test that propagate_time_domain logs completion."""
        from pystrata.time_integration import propagate_time_domain

        pystrata.enable_logging("INFO")

        try:
            with caplog.at_level(logging.INFO, logger="pystrata"):
                times = np.linspace(0, 1, 1000)
                input_accel = np.zeros_like(times)
                input_accel[100:110] = 0.1

                propagate_time_domain(
                    times=times,
                    input_accel=input_accel,
                    thicknesses=np.array([10.0, 20.0]),
                    densities=np.array([1800.0, 2000.0]),
                    shear_mods=np.array([50e6, 100e6]),
                    damping_ratios=np.array([0.02, 0.01]),
                    boundary="rigid",
                )

            # Check that completion message was logged
            pystrata_records = [
                r for r in caplog.records if "propagate_time_domain" in r.message
            ]
            assert len(pystrata_records) > 0, (
                "Expected propagate_time_domain log message"
            )

        finally:
            pystrata.disable_logging()


class TestCurveFittingLogging:
    """Test that curve fitting logs info."""

    def test_fit_profile_logs(self, caplog):
        """Test that fit_profile logs completion."""
        pystrata.enable_logging("INFO")

        try:
            profile = pystrata.site.Profile(
                [
                    pystrata.site.Layer(
                        pystrata.site.DarendeliSoilType(
                            18.0, plas_index=30, ocr=1, stress_mean=50
                        ),
                        10,
                        200,
                    ),
                    pystrata.site.Layer(
                        pystrata.site.SoilType("Rock", 22.0, damping=0.01),
                        0,
                        800,
                    ),
                ]
            )

            with caplog.at_level(logging.INFO, logger="pystrata"):
                from pystrata.curve_fitting import fit_profile

                fit_profile(profile, model="mkz", seed=42)

            # Check that completion message was logged
            pystrata_records = [r for r in caplog.records if "fit_profile" in r.message]
            assert len(pystrata_records) > 0, "Expected fit_profile log message"

        finally:
            pystrata.disable_logging()
