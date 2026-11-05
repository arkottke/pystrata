"""Tests of the site attenuation (kappa) of the motions and its correction."""

import copy

import numpy as np
import pygmm
import pytest
from numpy.testing import assert_allclose
from pygmm.fourier_spectrum import SourceTheoryModel

from pystrata import dam, motion, output, propagation, runner, site

FREQS = np.geomspace(0.2, 40.0, 40)
MIN_FREQ, MAX_FREQ = 10.0, 30.0


def _darendeli(**kw) -> site.SoilType:
    return site.SoilType.from_curves(pygmm.DarendeliSoilType(**kw).curves())


@pytest.fixture(scope="module")
def profile():
    return site.Profile(
        [
            site.Layer(
                _darendeli(unit_wt=18.0, plas_index=0, ocr=1, stress_mean=200),
                thickness,
                shear_vel,
            )
            for thickness, shear_vel in [(10, 300), (10, 400), (20, 600)]
        ]
        + [site.Layer(site.SoilType("Rock", 24.0, None, 0.01), 0, 1200)]
    )


@pytest.fixture(scope="module")
def rvt_motion():
    return motion.RvtMotion.from_fas(SourceTheoryModel(6.5, 20, "wna"))


@pytest.fixture(scope="module")
def ts_motion():
    return motion.TimeSeriesMotion.load_at2_file("examples/data/NIS090.AT2")


OUT_IN = output.OutputLocation("outcrop", index=-1)
OUT_OUT = output.OutputLocation("outcrop", index=0)


def _locs(profile):
    return profile.location("outcrop", index=-1), profile.location("outcrop", index=0)


def _values(out):
    """Values of an output that has been called once."""
    values = np.asarray(out.values)
    return values if values.ndim == 1 else values[:, 0]


def _surface_kappa(calc, motion_, profile, **kwds):
    """Attenuation of the motion at the surface from the output."""
    loc_in, _ = _locs(profile)
    out = output.FourierAmplitudeSpectrumOutput(
        motion_.freqs, OUT_OUT, kwds.get("ko_bandwidth")
    )
    calc(motion_, profile, loc_in)
    out(calc)
    return motion.fit_attenuation(motion_.freqs, _values(out), MIN_FREQ, MAX_FREQ)[0]


# fit_attenuation and calc_attenuation


def test_fit_attenuation_recovers_kappa():
    freqs = np.linspace(0.1, 50, 500)
    kappa, r_sqr, fit_freqs, fitted = motion.fit_attenuation(
        freqs, 3.0 * np.exp(-np.pi * 0.04 * freqs), MIN_FREQ, MAX_FREQ
    )
    assert kappa == pytest.approx(0.04)
    assert r_sqr == pytest.approx(1.0)
    assert fit_freqs.min() >= MIN_FREQ and fit_freqs.max() <= MAX_FREQ
    assert_allclose(fitted, 3.0 * np.exp(-np.pi * 0.04 * fit_freqs))


def test_fit_attenuation_uses_the_magnitude_of_complex_amplitudes():
    freqs = np.linspace(0.1, 50, 500)
    amps = 3.0 * np.exp(-np.pi * 0.04 * freqs) * np.exp(1j * freqs)
    assert motion.fit_attenuation(freqs, amps, 10, 30)[0] == pytest.approx(0.04)


def test_fit_attenuation_needs_amplitudes():
    with pytest.raises(ValueError, match="Fewer than two"):
        motion.fit_attenuation([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 10, 30)


def test_time_series_attenuation_matches_the_rvt_api(ts_motion):
    kappa, r_sqr, freqs, fitted = ts_motion.calc_attenuation(MIN_FREQ, MAX_FREQ)
    assert np.isfinite(kappa) and 0 < r_sqr <= 1
    assert freqs.shape == fitted.shape
    assert (
        kappa
        == motion.fit_attenuation(
            ts_motion.freqs, ts_motion.fourier_amps, MIN_FREQ, MAX_FREQ
        )[0]
    )


def test_rvt_attenuation_agrees_with_the_fit(rvt_motion):
    assert rvt_motion.calc_attenuation(MIN_FREQ, MAX_FREQ)[0] == pytest.approx(
        motion.fit_attenuation(
            rvt_motion.freqs, rvt_motion.fourier_amps, MIN_FREQ, MAX_FREQ
        )[0]
    )


# KappaCorrectedCalculator


@pytest.mark.parametrize("target", [0.01, 0.03, 0.06])
@pytest.mark.parametrize("motion_name", ["rvt_motion", "ts_motion"])
def test_corrected_motion_has_the_target_kappa(profile, motion_name, target, request):
    motion_ = request.getfixturevalue(motion_name)
    calc = propagation.KappaCorrectedCalculator(
        propagation.LinearElasticCalculator(), target, MIN_FREQ, MAX_FREQ
    )
    assert _surface_kappa(calc, motion_, profile) == pytest.approx(target, abs=1e-8)


def test_correction_lowers_the_amplitudes_when_the_target_is_larger(
    profile, rvt_motion
):
    """A larger target kappa attenuates the high frequencies more, not less."""
    plain = propagation.LinearElasticCalculator()
    kappa = _surface_kappa(plain, rvt_motion, profile)
    tf = plain.calc_accel_tf(*_locs(profile))

    for target, sign in [(kappa + 0.02, -1), (kappa - 0.02, +1)]:
        calc = propagation.KappaCorrectedCalculator(
            propagation.LinearElasticCalculator(), target, MIN_FREQ, MAX_FREQ
        )
        calc(rvt_motion, profile, _locs(profile)[0])
        ratio = np.abs(calc.calc_accel_tf(*_locs(profile))) / np.abs(tf)
        assert np.sign(np.log(ratio[-1] / ratio[0])) == sign


def test_target_equal_to_the_fit_changes_nothing(profile, rvt_motion):
    plain = propagation.LinearElasticCalculator()
    kappa = _surface_kappa(plain, rvt_motion, profile)
    calc = propagation.KappaCorrectedCalculator(
        propagation.LinearElasticCalculator(), kappa, MIN_FREQ, MAX_FREQ
    )
    calc(rvt_motion, profile, _locs(profile)[0])
    assert_allclose(
        calc.calc_accel_tf(*_locs(profile)), plain.calc_accel_tf(*_locs(profile))
    )


def test_initial_and_site_kappa(profile, rvt_motion):
    calc = propagation.KappaCorrectedCalculator(
        propagation.LinearElasticCalculator(), 0.03, MIN_FREQ, MAX_FREQ
    )
    assert calc.initial_kappa is None
    loc_in, loc_out = _locs(profile)
    calc(rvt_motion, profile, loc_in)
    assert calc.initial_kappa == pytest.approx(
        rvt_motion.calc_attenuation(MIN_FREQ, MAX_FREQ)[0]
    )
    # The site changes the attenuation of the input motion
    assert calc.calc_kappa(loc_out) != pytest.approx(calc.initial_kappa, abs=1e-4)
    # The input motion is not changed
    assert calc.calc_kappa(loc_in) == pytest.approx(calc.initial_kappa)
    assert_allclose(calc.calc_accel_tf(loc_in, loc_in), 1.0)


def test_transfer_function_between_locations_is_the_ratio(profile, rvt_motion):
    calc = propagation.KappaCorrectedCalculator(
        propagation.LinearElasticCalculator(), 0.02, MIN_FREQ, MAX_FREQ
    )
    loc_in, loc_out = _locs(profile)
    loc_mid = profile.location("within", depth=15.0)
    calc(rvt_motion, profile, loc_in)
    assert_allclose(
        calc.calc_accel_tf(loc_mid, loc_out),
        calc.calc_accel_tf(loc_in, loc_out) / calc.calc_accel_tf(loc_in, loc_mid),
    )


def test_wrapped_attributes_and_strain_are_not_changed(profile, rvt_motion):
    plain = propagation.EquivalentLinearCalculator()
    calc = propagation.KappaCorrectedCalculator(
        propagation.EquivalentLinearCalculator(), 0.01, MIN_FREQ, MAX_FREQ
    )
    loc_in, loc_out = _locs(profile)
    plain(rvt_motion, profile, loc_in)
    strains = [layer.strain_max for layer in profile[:-1]]
    calc(rvt_motion, profile, loc_in)

    assert calc.motion is rvt_motion
    assert calc.profile is profile
    assert calc.loc_input is loc_in
    assert_allclose([layer.strain_max for layer in profile[:-1]], strains)
    assert_allclose(
        calc.calc_strain_tf(loc_in, profile.location("within", depth=15.0)),
        plain.calc_strain_tf(loc_in, profile.location("within", depth=15.0)),
    )


def test_every_output_is_corrected(profile, rvt_motion):
    """The correction needs no output of its own."""
    loc_in, _ = _locs(profile)
    outputs = [
        output.FourierAmplitudeSpectrumOutput(FREQS, OUT_OUT),
        output.ResponseSpectrumOutput(FREQS, OUT_OUT, 0.05),
        output.AccelTransferFunctionOutput(FREQS, OUT_IN, OUT_OUT),
        output.ResponseSpectrumRatioOutput(FREQS, OUT_IN, OUT_OUT, 0.05),
    ]

    def run(calc):
        calc(rvt_motion, profile, loc_in)
        results = copy.deepcopy(outputs)
        for out in results:
            out(calc)
        return [_values(out) for out in results]

    plain = run(propagation.LinearElasticCalculator())
    corrected = run(
        propagation.KappaCorrectedCalculator(
            propagation.LinearElasticCalculator(), 0.005, MIN_FREQ, MAX_FREQ
        )
    )
    for before, after in zip(plain, corrected, strict=True):
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after)


def test_time_series_output_is_corrected(profile, ts_motion):
    loc_in, _ = _locs(profile)

    def run(calc):
        out = output.AccelerationTSOutput(OUT_OUT)
        calc(ts_motion, profile, loc_in)
        out(calc)
        return _values(out)

    plain = run(propagation.LinearElasticCalculator())
    corrected = run(
        propagation.KappaCorrectedCalculator(
            propagation.LinearElasticCalculator(), 0.001, MIN_FREQ, MAX_FREQ
        )
    )
    assert plain.shape == corrected.shape
    assert not np.allclose(plain, corrected)
    # A smaller attenuation has more high-frequency energy
    assert np.abs(np.diff(corrected)).sum() > np.abs(np.diff(plain)).sum()


def test_smoothing_option(profile, rvt_motion):
    calc = propagation.KappaCorrectedCalculator(
        propagation.LinearElasticCalculator(),
        0.03,
        MIN_FREQ,
        MAX_FREQ,
        ko_bandwidth=30,
    )
    assert calc.ko_bandwidth == 30
    calc(rvt_motion, profile, _locs(profile)[0])
    assert np.isfinite(calc.initial_kappa)
    assert np.all(np.isfinite(calc.calc_accel_tf(*_locs(profile))))


def test_frequency_range_must_increase():
    with pytest.raises(ValueError, match="increasing"):
        propagation.KappaCorrectedCalculator(
            propagation.LinearElasticCalculator(), 0.03, 30.0, 10.0
        )


def test_copy_and_ensemble(profile, rvt_motion):
    """The calculator is copied for each task of an ensemble."""
    template = propagation.KappaCorrectedCalculator(
        propagation.LinearElasticCalculator(), 0.02, MIN_FREQ, MAX_FREQ
    )
    copied = copy.deepcopy(template)
    assert copied.kappa_target == 0.02 and copied.calc is not template.calc

    loc_in, _ = _locs(profile)
    outputs = output.OutputCollection(
        [output.ResponseSpectrumOutput(FREQS, OUT_OUT, 0.05)]
    )
    results = runner.run_ensemble(profile, rvt_motion, template, outputs, count=1)

    direct = copy.deepcopy(template)
    direct(rvt_motion, profile, loc_in)
    expected = output.ResponseSpectrumOutput(FREQS, OUT_OUT, 0.05)
    expected(direct)
    assert_allclose(_values(results[0]), _values(expected))


def test_dam_calculator_is_corrected(rvt_motion):
    soil = site.SoilType("fill", 21.0, None, 0.05)
    rock = site.SoilType("rock", 25.0, None, 0.0)
    d = dam.Dam.from_inhomogeneity(soil, 50.0, 300.0, n_layers=10)
    prof = dam.DamProfile(d, site.Profile([site.Layer(rock, 1.0, 1500.0)]))
    loc_in = prof.location("outcrop", index=len(prof) - 1)
    crest = prof.location("within", depth=0.0)

    plain = dam.DamLinearElasticCalculator()
    plain(rvt_motion, prof, loc_in)
    calc = propagation.KappaCorrectedCalculator(
        dam.DamLinearElasticCalculator(), 0.04, MIN_FREQ, MAX_FREQ
    )
    calc(rvt_motion, prof, loc_in)

    tf = calc.calc_accel_tf(loc_in, crest)
    amps = tf * rvt_motion.fourier_amps
    assert motion.fit_attenuation(rvt_motion.freqs, amps, MIN_FREQ, MAX_FREQ)[
        0
    ] == pytest.approx(0.04, abs=1e-8)
    assert not np.allclose(tf, plain.calc_accel_tf(loc_in, crest))
