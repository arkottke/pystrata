import numpy as np
import pytest

from pystrata import dam, site
from pystrata import motion as _motion
from pystrata.dam import (
    RectangularCanyon,
    canyon_freq_ratio,
    canyon_rect_natural_freqs,
    canyon_rect_rigid_base_tf,
    wedge_natural_freqs,
    wedge_rigid_base_tf,
    wedge_truncated_rigid_base_tf,
)

from . import FPATH_DATA

H, VS = 50.0, 300.0


def test_invalid_length():
    with pytest.raises(ValueError):
        RectangularCanyon(0)


def test_long_canyon_recovers_2d():
    freqs = np.array([0.5, 1.0, 2.0])
    length = 5000.0
    tf = canyon_rect_rigid_base_tf(freqs, length / 2, H, VS, length, 0.05, n_modes=2000)
    ref = wedge_rigid_base_tf(freqs, H, VS, damping=0.05)
    np.testing.assert_allclose(tf, ref, rtol=2e-3)


def test_long_truncated_canyon_recovers_2d():
    freqs = np.array([0.5, 1.0, 2.0])
    length, r0 = 5000.0, 10.0
    tf = canyon_rect_rigid_base_tf(
        freqs, length / 2, H, VS, length, 0.05, crest_offset=r0, n_modes=2000
    )
    ref = wedge_truncated_rigid_base_tf(freqs, H, VS, r0, damping=0.05)
    np.testing.assert_allclose(tf, ref, rtol=2e-3)


@pytest.mark.parametrize("r0", [0.0, 10.0])
def test_abutment_and_zero_freq(r0):
    tf = canyon_rect_rigid_base_tf([0.0, 3.0], 0.0, H, VS, 200.0, 0.05, crest_offset=r0)
    np.testing.assert_allclose(tf, 1.0, atol=1e-12)
    tf0 = canyon_rect_rigid_base_tf([0.0], 100.0, H, VS, 200.0, 0.05, crest_offset=r0)
    np.testing.assert_allclose(tf0, 1.0)


def test_symmetric():
    f = np.linspace(0.2, 4, 30)
    a = canyon_rect_rigid_base_tf(f, 60.0, H, VS, 200.0, 0.05)
    b = canyon_rect_rigid_base_tf(f, 140.0, H, VS, 200.0, 0.05)
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_peak_at_natural_frequency():
    length = 150.0
    f_n = canyon_rect_natural_freqs(H, VS, length, n=1)[0]
    freqs = np.linspace(0.5 * f_n, 1.5 * f_n, 3001)
    amp = np.abs(canyon_rect_rigid_base_tf(freqs, length / 2, H, VS, length, 0.005))
    assert freqs[np.argmax(amp)] == pytest.approx(f_n, rel=2e-3)


def test_freq_ratio():
    length = 100.0
    expected = np.sqrt(1 + (np.pi * H / (2.4048 * length)) ** 2)
    assert canyon_freq_ratio(H, VS, length) == pytest.approx(expected, rel=1e-4)
    assert canyon_freq_ratio(H, VS, 1e6) == pytest.approx(1.0, abs=1e-6)
    f2d = wedge_natural_freqs(H, VS, 0.0, 1)[0]
    f3d = canyon_rect_natural_freqs(H, VS, length, n=1)[0]
    assert f3d > f2d


# Layered dam in a canyon on a site


SOIL = site.SoilType("dam", 20.0, None, 0.02)
ROCK = site.SoilType("rock", 25.0, None, 0.0)
FREQS = np.linspace(0.2, 6, 40)
FREQS_LONG = np.array([0.5, 2.3, 4.5, 6.0])


def rock_site(vel=1500.0):
    return site.Profile([site.Layer(ROCK, 1.0, vel)])


def make_profile(length, n_layers=10, vel_site=1500.0, crest_width=0.0, **kwds):
    canyon = None if length is None else RectangularCanyon(length, **kwds)
    d = dam.Dam.from_inhomogeneity(
        SOIL,
        H,
        VS,
        n_layers=n_layers,
        slope=2.5,
        crest_width=crest_width,
        canyon=canyon,
    )
    return dam.DamProfile(d, rock_site(vel_site))


@pytest.mark.parametrize("crest_width", [0.0, 8.0])
def test_long_canyon_matches_the_2d_dam(crest_width):
    # The waves from the abutments are damped out at the center of a long canyon
    p3 = make_profile(12000.0, crest_width=crest_width, n_modes=1200)
    p2 = make_profile(None, crest_width=crest_width)
    np.testing.assert_allclose(
        p3.transfer_function(FREQS_LONG),
        p2.transfer_function(FREQS_LONG),
        rtol=1e-4,
    )


def test_rigid_base_matches_the_closed_form():
    length = 160.0
    # A very stiff half space is a rigid base
    p = make_profile(length, n_layers=1, vel_site=2e5)
    tf = p.transfer_function(FREQS, x=0.3 * length)
    ref = canyon_rect_rigid_base_tf(
        FREQS, 0.3 * length, H, p.dam.shear_vels[0], length, damping=0.02
    )
    # The dam is a single layer, but its damping follows the complex modulus
    np.testing.assert_allclose(np.abs(tf), np.abs(ref), rtol=3e-2)


def test_layer_splitting_invariance():
    length = 200.0
    one = site.Layer(SOIL, H, 280.0)
    d1 = dam.Dam([one], slope=2.5, canyon=RectangularCanyon(length))
    d5 = dam.Dam(
        [site.Layer(SOIL, H / 5, 280.0) for _ in range(5)],
        slope=2.5,
        canyon=RectangularCanyon(length),
    )
    t1 = dam.DamProfile(d1, rock_site()).transfer_function(FREQS)
    t5 = dam.DamProfile(d5, rock_site()).transfer_function(FREQS)
    np.testing.assert_allclose(t1, t5, rtol=1e-6)


def test_abutment_moves_with_the_outcrop():
    p = make_profile(200.0)
    for depth in [0.0, 25.0, 49.0]:
        tf = p.transfer_function(FREQS, depth_out=depth, x=0.0)
        np.testing.assert_allclose(tf, 1.0, atol=1e-8)


def test_symmetry():
    p = make_profile(200.0)
    np.testing.assert_allclose(
        p.transfer_function(FREQS, x=60.0),
        p.transfer_function(FREQS, x=140.0),
        atol=1e-8,
    )


def test_canyon_stiffens_the_dam():
    freqs = np.linspace(0.3, 8, 1500)

    def first_peak(length):
        amp = np.abs(make_profile(length).transfer_function(freqs))
        peaks = np.nonzero((amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:]))[0] + 1
        return freqs[peaks[0]]

    f_2d, f_200, f_100 = first_peak(None), first_peak(200.0), first_peak(100.0)
    assert f_2d < f_200 < f_100


def test_zero_frequency_is_rigid_body():
    p = make_profile(200.0)
    np.testing.assert_allclose(p.transfer_function([0.0]), 1.0)


def test_canyon_is_carried():
    p = make_profile(200.0)
    assert p.canyon == RectangularCanyon(200.0)
    assert p.copy().canyon == p.canyon
    assert p.dam.canyon == p.canyon
    assert p.dam.scaled(1.2).canyon == p.canyon
    assert make_profile(None).canyon is None


def test_position_outside_the_canyon():
    with pytest.raises(ValueError, match="within the canyon"):
        make_profile(200.0).transfer_function(FREQS, x=250.0)


def test_sliding_mass_transfer_function_in_a_canyon():
    surface = dam.toe_circle(make_profile(None).dam)
    t2 = make_profile(None).sliding_mass_transfer_function(FREQS, surface)
    t3 = make_profile(150.0).sliding_mass_transfer_function(FREQS, surface, n_x=5)
    assert np.all(np.isfinite(t3))
    # The abutments restrain the mass: lower response at the 2D resonance
    assert np.abs(t3).max() < np.abs(t2).max()


# Equivalent-linear


@pytest.fixture(scope="module")
def motion():
    return _motion.TimeSeriesMotion.load_at2_file(FPATH_DATA / "NIS090.AT2", scale=0.3)


def nonlinear_profile(canyon):
    strains = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    mr = site.ModulusReductionCurve("", strains, [1, 0.97, 0.7, 0.25, 0.05])
    dc = site.DampingCurve("", strains, [0.01, 0.015, 0.05, 0.12, 0.2])
    fill = site.SoilType("fill", 20.0, mr, dc)
    d = dam.Dam.from_inhomogeneity(
        fill, 60.0, 250.0, m=2 / 3, n_layers=12, slope=2.5, canyon=canyon
    )
    return dam.DamProfile(d, rock_site())


def test_eql_in_a_canyon(motion):
    results = {}
    for name, canyon in [("2d", None), ("3d", RectangularCanyon(400.0, n_modes=5))]:
        p = nonlinear_profile(canyon)
        calc = dam.DamEquivalentLinearCalculator()
        loc = p.location("outcrop", index=len(p) - 1)
        calc(motion, p, loc)
        assert all(layer.strain_max > 0 for layer in p[:-1])
        crest = p.location("within", depth=0)
        results[name] = motion.calc_peak(calc.calc_accel_tf(loc, crest))
    assert results["3d"] > motion.pga
    assert results["3d"] != pytest.approx(results["2d"], rel=1e-3)


def test_longitudinal_strain_vanishes_without_a_canyon(motion):
    p = nonlinear_profile(None)
    calc = dam.DamEquivalentLinearCalculator()
    loc = p.location("outcrop", index=len(p) - 1)
    calc(motion, p, loc)
    lout = p.location("within", depth=20.0)
    assert not np.any(calc.calc_strain_xy_tf(loc, lout))


def test_longitudinal_strain_peaks_at_the_abutment(motion):
    p = nonlinear_profile(RectangularCanyon(400.0, n_modes=5))
    loc = p.location("outcrop", index=len(p) - 1)
    lout = p.location("within", depth=30.0)
    peaks = []
    for x in [1.0, 100.0, 200.0]:
        calc = dam.DamLinearElasticCalculator()
        calc.canyon_x = x
        calc(motion, p, loc)
        peaks.append(np.abs(calc.calc_strain_xy_tf(loc, lout)).max())
    assert peaks[0] > peaks[1] > peaks[2]
