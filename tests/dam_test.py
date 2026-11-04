import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import jv

import pystrata
from pystrata import dam, site
from pystrata.dam import analytical

from . import FPATH_DATA

SOIL = site.SoilType("dam", 20.0, None, 0.02)
ROCK = site.SoilType("rock", 25.0, None, 0.0)
FREQS = np.linspace(0.1, 12, 60)


def rock_site(vel=1500.0):
    return site.Profile([site.Layer(ROCK, 1.0, vel)])


def lin_calc():
    return dam.DamLinearElasticCalculator()


# Analytical (Table 2)


def test_table2_inhomogeneous():
    f = analytical.wedge_natural_freqs(100.0, 300.0, m=2 / 3, n=2)
    # T1 = 2.57 H / V, T1 / T2 = 2, omega_1 = (7 / 9) pi V / H
    assert_allclose(1 / f[0] * 300 / 100, 2.571, atol=2e-3)
    assert_allclose(f[1] / f[0], 2.0)
    assert_allclose(2 * np.pi * f[0], 7 / 9 * np.pi * 300 / 100)


def test_table2_homogeneous():
    f = analytical.wedge_natural_freqs(100.0, 300.0, m=0.0, n=2)
    assert_allclose(1 / f[0] * 300 / 100, 2.61, atol=5e-3)
    assert_allclose(f[1] / f[0], 2.30, atol=5e-3)


def test_rigid_base_homogeneous_is_bessel():
    tf = analytical.wedge_rigid_base_tf(FREQS, 100.0, 300.0)
    assert_allclose(tf, 1 / jv(0, 2 * np.pi * FREQS * 100.0 / 300.0))


def test_mode_shape_normalized():
    f1 = analytical.wedge_natural_freqs(100.0, 300.0, 2 / 3, 1)[0]
    u = analytical.wedge_mode_shape([0.0, 100.0], f1, 100.0, 300.0, 2 / 3)
    assert_allclose(u, [1.0, 0.0], atol=1e-12)


def test_order_range():
    with pytest.raises(ValueError):
        analytical.wedge_order(2.0)


# Dam model


def test_dam_properties():
    d = dam.Dam.from_inhomogeneity(SOIL, 100.0, 300.0, m=2 / 3, n_layers=200)
    assert_allclose(d.height, 100.0)
    assert_allclose(d.shear_vel_avg, 300.0, rtol=5e-3)
    assert all(isinstance(layer, dam.WedgeLayer) for layer in d.layers)
    # G ~ z^(2/3) -> Vs ~ z^(1/3)
    assert_allclose(d.shear_vels[-1] / d.shear_vels[99], (199.5 / 99.5) ** (1 / 3))


def test_dam_plain_layers_converted():
    d = dam.Dam([site.Layer(SOIL, 50.0, 200.0)])
    assert isinstance(d.layers[0], dam.WedgeLayer)


def test_dam_scaled():
    d = dam.Dam.from_inhomogeneity(SOIL, 100.0, 300.0, 0.0, 4).scaled(2.0)
    assert_allclose(d.shear_vels, 600.0)


def test_fixed_base_freqs_match_analytical():
    d = dam.Dam.from_inhomogeneity(SOIL, 100.0, 300.0, m=2 / 3, n_layers=60)
    freqs = d.fixed_base_freqs(3)
    expected = analytical.wedge_natural_freqs(100.0, d.shear_vel_avg, 2 / 3, 3)
    assert_allclose(freqs, expected, rtol=5e-3)


def test_fixed_base_freq_homogeneous_exact():
    d = dam.Dam([site.Layer(SOIL, 100.0, 300.0)])
    assert_allclose(
        d.fixed_base_freq(), analytical.wedge_natural_freqs(100.0, 300.0, 0, 1)[0]
    )


def test_tune_fixed_base():
    d = dam.Dam.from_inhomogeneity(SOIL, 80.0, 250.0, m=2 / 3, n_layers=20)
    tuned = d.tune(3.0)
    assert_allclose(tuned.fixed_base_freq(), 3.0, rtol=1e-6)
    # Velocity ratios are preserved
    ratio = tuned.shear_vels / d.shear_vels
    assert_allclose(ratio, ratio[0])


def test_tune_coupled():
    d = dam.Dam.from_inhomogeneity(SOIL, 80.0, 250.0, m=2 / 3, n_layers=20)
    ste = rock_site(1500.0)
    tuned = d.tune(1.5, site=ste)
    assert_allclose(dam.DamProfile(tuned, ste).peak_freq(1.5), 1.5, rtol=1e-3)
    # The foundation shifts the peak away from the fixed-base frequency
    assert not np.isclose(tuned.shear_vel_avg, d.tune(1.5).shear_vel_avg, rtol=1e-3)


def test_tune_invalid():
    d = dam.Dam([site.Layer(SOIL, 50.0, 200.0)])
    with pytest.raises(ValueError):
        d.tune(-1.0)


# Profile


def test_dam_profile_layout():
    d = dam.Dam.from_inhomogeneity(SOIL, 100.0, 300.0, 0.0, 5)
    ste = site.Profile([site.Layer(SOIL, 20.0, 400.0), site.Layer(ROCK, 1.0, 1500.0)])
    p = dam.DamProfile(d, ste)
    assert len(p) == 7
    assert p.n_dam_layers == 5
    assert_allclose(p[5].depth, 100.0)
    assert_allclose(p[5].stress_vert(0), 20.0 * 100.0)
    assert_allclose(p.wt_depth, 100.0 + ste.wt_depth)
    # Layers are copies
    assert p[0] is not d.layers[0]


def test_dam_profile_copy():
    d = dam.Dam.from_inhomogeneity(SOIL, 100.0, 300.0, 0.0, 5)
    p = dam.DamProfile(d, rock_site())
    q = p.copy()
    assert isinstance(q, dam.DamProfile)
    assert len(q) == len(p)
    assert all(
        isinstance(a, dam.WedgeLayer) == isinstance(b, dam.WedgeLayer)
        for a, b in zip(p, q, strict=True)
    )


# Calculators


def test_elastic_rock_homogeneous_matches_table2():
    d = dam.Dam([site.Layer(SOIL, 100.0, 300.0)])
    p = dam.DamProfile(d, rock_site())
    tf = p.transfer_function(FREQS)
    imped = 20.0 * 300.0 / (25.0 * 1500.0)
    expected = analytical.wedge_elastic_base_tf(FREQS, 100.0, 300.0, 0.02, imped)
    assert_allclose(tf, expected, rtol=1e-10)


def test_rigid_base_matches_table2():
    """A very stiff, undamped rock approaches the rigid-base solution."""
    d = dam.Dam([site.Layer(site.SoilType("", 20.0, None, 0.0), 100.0, 300.0)])
    p = dam.DamProfile(d, rock_site(3e8))
    f = np.array([0.5, 1.0, 2.0, 3.0])
    tf = p.transfer_function(f)
    assert_allclose(tf, analytical.wedge_rigid_base_tf(f, 100.0, 300.0), rtol=1e-4)


def test_layer_splitting_invariance():
    """Splitting a homogeneous dam into layers does not change the response."""
    one = dam.Dam([site.Layer(SOIL, 100.0, 300.0)])
    many = dam.Dam([site.Layer(SOIL, 12.5, 300.0) for _ in range(8)])
    tf1 = dam.DamProfile(one, rock_site()).transfer_function(FREQS)
    tf2 = dam.DamProfile(many, rock_site()).transfer_function(FREQS)
    assert_allclose(tf1, tf2, rtol=1e-8)


def test_layered_inhomogeneous_first_peak():
    d = dam.Dam.from_inhomogeneity(
        site.SoilType("", 20.0, None, 0.005), 100.0, 300.0, m=2 / 3, n_layers=50
    )
    p = dam.DamProfile(d, rock_site(3e6))
    expected = analytical.wedge_natural_freqs(100.0, d.shear_vel_avg, 2 / 3, 1)[0]
    assert_allclose(p.peak_freq(f_guess=expected), expected, rtol=1e-2)


def test_vanishing_dam_matches_plain_site():
    """A dam of negligible height leaves the site response unchanged."""
    layers = [site.Layer(SOIL, 10.0, 250.0), site.Layer(ROCK, 1.0, 1500.0)]
    d = dam.Dam([site.Layer(SOIL, 1e-6, 300.0)])
    p = dam.DamProfile(d, site.Profile([layer.copy() for layer in layers]))
    calc = lin_calc()
    freqs = 2 * np.pi * FREQS
    calc._calc_waves(freqs, p)
    tf_dam = calc.calc_accel_tf(
        p.location("outcrop", index=len(p) - 1), p.location("within", index=1)
    )

    plain = site.Profile([layer.copy() for layer in layers])
    calc2 = pystrata.propagation.LinearElasticCalculator()
    calc2._calc_waves(freqs, plain)
    tf_plain = calc2.calc_accel_tf(
        plain.location("outcrop", index=1), plain.location("within", index=0)
    )
    assert_allclose(tf_dam, tf_plain, rtol=1e-6)


def test_zero_frequency_is_rigid_body():
    d = dam.Dam([site.Layer(SOIL, 50.0, 300.0)])
    p = dam.DamProfile(d, rock_site())
    tf = p.transfer_function([0.0, 1.0])
    # A rigid body moves as one: the crest follows the outcrop motion
    assert_allclose(tf[0], 1.0)
    assert np.isfinite(tf).all()


def test_outcrop_in_dam_raises():
    d = dam.Dam([site.Layer(SOIL, 50.0, 300.0)])
    p = dam.DamProfile(d, rock_site())
    calc = lin_calc()
    calc._calc_waves(2 * np.pi * FREQS, p)
    with pytest.raises(ValueError):
        calc.wave_at_location(p.location("outcrop", depth=10.0))


def test_oblique_incidence_not_supported():
    d = dam.Dam([site.Layer(SOIL, 50.0, 300.0)])
    p = dam.DamProfile(d, rock_site())
    calc = dam.DamLinearElasticCalculator(incidence_angle=10.0)
    with pytest.raises(NotImplementedError):
        calc._calc_waves(2 * np.pi * FREQS, p)


def test_plain_profile_rejected():
    calc = lin_calc()
    with pytest.raises(ValueError):
        calc._calc_waves(2 * np.pi * FREQS, rock_site())


# Equivalent-linear


@pytest.fixture(scope="module")
def motion():
    return pystrata.motion.TimeSeriesMotion.load_at2_file(
        FPATH_DATA / "NIS090.AT2", scale=0.3
    )


def nonlinear_dam_profile():
    strains = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    mr = site.ModulusReductionCurve("", strains, [1, 0.97, 0.7, 0.25, 0.05])
    dc = site.DampingCurve("", strains, [0.01, 0.015, 0.05, 0.12, 0.2])
    fill = site.SoilType("fill", 20.0, mr, dc)
    d = dam.Dam.from_inhomogeneity(fill, 60.0, 250.0, m=2 / 3, n_layers=12)
    return dam.DamProfile(d, rock_site())


def test_eql_dam(motion):
    p = nonlinear_dam_profile()
    calc = dam.DamEquivalentLinearCalculator()
    loc = p.location("outcrop", index=len(p) - 1)
    calc(motion, p, loc)

    # Strains were computed in every dam layer and softened the dam
    assert all(layer.strain_max > 0 for layer in p[:-1])
    assert all(layer.shear_mod_reduc < 1 for layer in p[:12])

    crest = p.location("within", depth=0)
    crest_accel = motion.calc_peak(calc.calc_accel_tf(loc, crest))
    assert crest_accel > motion.pga  # amplified beyond the outcrop motion


def test_eql_outputs(motion):
    p = nonlinear_dam_profile()
    out = pystrata.output.OutputCollection(
        [
            pystrata.output.MaxStrainProfile(),
            pystrata.output.MaxAccelProfile(),
            pystrata.output.ResponseSpectrumOutput(
                np.logspace(-1, 1, 20),
                pystrata.output.OutputLocation("within", depth=0),
                0.05,
            ),
        ]
    )
    calc = dam.DamEquivalentLinearCalculator()
    calc(motion, p, p.location("outcrop", index=len(p) - 1))
    out(calc)
    assert all(len(o.values) for o in out)


def test_fdm_dam(motion):
    p = nonlinear_dam_profile()
    calc = dam.DamFrequencyDependentEqlCalculator(max_iterations=10)
    calc(motion, p, p.location("outcrop", index=len(p) - 1))
    assert all(np.all(np.asarray(layer.strain_max) > 0) for layer in p[:-1])


def test_tune_coupled_soft_foundation():
    """The peak nearest the target is used, not a lower foundation mode."""
    alluvium = site.SoilType("alluvium", 19.0, None, 0.03)
    ste = site.Profile(
        [site.Layer(alluvium, 30.0, 220.0), site.Layer(ROCK, 1.0, 1500.0)]
    )
    d = dam.Dam.from_inhomogeneity(SOIL, 100.0, 300.0, m=2 / 3, n_layers=20)
    tuned = d.tune(1.5, site=ste)
    assert_allclose(dam.DamProfile(tuned, ste).peak_freq(1.5), 1.5, rtol=1e-3)


def test_tune_divergence_raises():
    """A target that is not a resonance of the system cannot be reached."""
    d = dam.Dam.from_inhomogeneity(SOIL, 80.0, 250.0, m=2 / 3, n_layers=20)
    with pytest.raises((RuntimeError, ValueError)):
        d.tune(3.0, site=rock_site(600.0))


@pytest.mark.parametrize("damping", [0.02, 0.2])
def test_many_layers_match_one_layer_at_high_frequency(damping):
    """Splitting a homogeneous dam into layers must not change the response.

    A J0/Y0 formulation loses all precision across the interfaces of a damped dam above
    a few Hz and returns a zero transfer function; the single layer needs no interface
    and has no such problem.
    """
    soil = site.SoilType("dam", 20.0, None, damping)
    freqs = np.array([1.0, 10.0, 20.0, 50.0, 80.0])
    height, vel = 200.0, 200.0

    def tf(n_layers):
        layers = [site.Layer(soil, height / n_layers, vel) for _ in range(n_layers)]
        p = dam.DamProfile(dam.Dam(layers), rock_site())
        return p.transfer_function(freqs)

    one, many = tf(1), tf(30)
    assert np.all(np.abs(one) > 0)
    assert_allclose(many, one, rtol=1e-6)


def _layered_dam_profile():
    layers = [site.Layer(SOIL, 25.0, 200.0 + 40 * i) for i in range(4)]
    return dam.DamProfile(dam.Dam(layers), rock_site())


def test_undamped_transfer_function_peaks_higher():
    p = _layered_dam_profile()
    freqs = np.linspace(0.5, 8, 400)
    damped = np.abs(p.transfer_function(freqs, damped=True))
    undamped = np.abs(p.transfer_function(freqs, damped=False))
    assert undamped.max() > damped.max()
    # The profile itself keeps its damping
    assert p[0].damping_min == SOIL.damping_min


def test_dam_follows_the_layers_after_the_velocities_change():
    from pystrata import variation

    p = _layered_dam_profile()
    varied = next(
        variation.iter_varied_profiles(
            p,
            1,
            seed=3,
            var_velocity=variation.ToroVelocityVariation.generic_model("USGS C"),
        )
    )
    assert isinstance(varied, dam.DamProfile)
    assert varied.n_dam_layers == 4
    assert not np.allclose(varied.dam.shear_vels, p.dam.shear_vels)
    assert_allclose(
        [layer.initial_shear_vel for layer in varied.dam.layers],
        [layer.initial_shear_vel for layer in varied.layers[:4]],
    )


def test_resampling_variations_reject_dam_profiles():
    from pystrata import variation

    p = _layered_dam_profile()
    with pytest.raises(NotImplementedError, match="dam profiles"):
        variation.ToroThicknessVariation()(p)


@pytest.mark.parametrize("model", ["seed", "kramer", "dormieux"])
def test_analytical_complex_velocity_follows_the_modulus_model(monkeypatch, model):
    monkeypatch.setattr(site, "COMP_MODULUS_MODEL", model)
    layer = site.Layer(site.SoilType("S", 20.0, None, 0.05), 10.0, 250.0)
    assert_allclose(analytical.comp_shear_vel(250.0, 0.05), layer.comp_shear_vel)
