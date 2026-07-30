# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# Copyright (C) Albert Kottke, 2013-2022
import contextlib
import json
import string

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pytest

import pystrata
import pystrata.propagation as _prop

from . import FPATH_DATA


@contextlib.contextmanager
def _use_python_dispatch():
    """Temporarily force pure-Python dispatch (no numba)."""
    saved = (
        _prop._calc_waves_dispatch,
        _prop._calc_waves_general_dispatch,
        _prop._wave_at_location_dispatch,
        _prop._calc_strain_tf_dispatch,
    )
    _prop._calc_waves_dispatch = _prop._calc_waves_python
    _prop._calc_waves_general_dispatch = _prop._calc_waves_general_python
    _prop._wave_at_location_dispatch = _prop._wave_at_location_python
    _prop._calc_strain_tf_dispatch = _prop._calc_strain_tf_python
    try:
        yield
    finally:
        (
            _prop._calc_waves_dispatch,
            _prop._calc_waves_general_dispatch,
            _prop._wave_at_location_dispatch,
            _prop._calc_strain_tf_dispatch,
        ) = saved


def read_cluster(ws, cols, names, row_start, row_end):
    d = dict()
    for c, name in zip(cols, names):
        range_str = f"{c}{row_start}:{c}{row_end}"
        d[name] = [row[0].value for row in ws[range_str]]
    return d


def read_deepsoil_results(name):
    wb = openpyxl.load_workbook(str(FPATH_DATA / (name + ".xlsx")))
    ws = wb["Layer1"]
    data = list(ws.values)
    names = ",".join(string.ascii_uppercase[: len(data[0])])
    records = np.rec.fromrecords(data, names=names)

    def extract_cols(records, cols, first, last, names):
        return {
            name: records[col][first:last].astype(float)
            for col, name in zip(cols, names)
        }

    d = dict()
    # Read the time series
    d["time_series"] = extract_cols(
        records, "ABCDE", 1, 11800, ["time", "accel", "strain", "stress", "arias_int"]
    )
    # Read the response spectrum
    d["resp_spec"] = extract_cols(records, "GH", 1, 114, ["period", "psa"])
    # Read the Fourier amplitude
    d["fourier_spec"] = extract_cols(
        records, "JKL", 1, 16384, ["freq", "ampl", "ratio"]
    )

    return d


def load_ts():
    fpath = FPATH_DATA / "ChiChi.txt"
    with fpath.open() as fp:
        parts = next(fp).split()
        time_step = float(parts[1])
        accels = [float(line.split()[1]) for line in fp]

    return pystrata.motion.TimeSeriesMotion(
        fpath.name, "ChiChi.txt from DeepSoil v6.1", time_step, accels
    )


class DeepSoilComparison:
    rtol = 0.005
    atol = 0.001

    ref_name = NotImplemented
    calc = NotImplemented
    profile = NotImplemented

    # Override in subclasses to force pure-Python dispatch
    _use_python = False

    @classmethod
    def setup_class(cls):
        cls.ref = read_deepsoil_results(cls.ref_name)
        ctx = _use_python_dispatch() if cls._use_python else contextlib.nullcontext()
        with ctx:
            cls._run_calc()

    @classmethod
    def _run_calc(cls):
        # Perform the calculation
        cls.calc(load_ts(), cls.profile, cls.profile.location("outcrop", index=-1))
        cls.outputs = pystrata.output.OutputCollection(
            [
                pystrata.output.AccelerationTSOutput(
                    pystrata.output.OutputLocation("outcrop", depth=0)
                ),
                pystrata.output.AriasIntensityTSOutput(
                    pystrata.output.OutputLocation("outcrop", depth=0)
                ),
                pystrata.output.StrainTSOutput(
                    pystrata.output.OutputLocation("within", depth=10), in_percent=True
                ),
                pystrata.output.StressTSOutput(
                    pystrata.output.OutputLocation("within", depth=10), normalized=True
                ),
                pystrata.output.ResponseSpectrumOutput(
                    [
                        100.0000,
                        93.9744,
                        88.3119,
                        82.9910,
                        77.9903,
                        73.2912,
                        68.8753,
                        64.7249,
                        60.8250,
                        57.1602,
                        53.7158,
                        50.4793,
                        47.4377,
                        44.5794,
                        41.8932,
                        39.3690,
                        36.9969,
                        34.7676,
                        32.6727,
                        30.7040,
                        28.8540,
                        27.1154,
                        25.4816,
                        23.9462,
                        22.5034,
                        21.1474,
                        19.8732,
                        18.6757,
                        17.5504,
                        16.4930,
                        15.4992,
                        14.5653,
                        13.6877,
                        12.8629,
                        12.0879,
                        11.3595,
                        10.6751,
                        10.0319,
                        9.4274,
                        8.8594,
                        8.3256,
                        7.8239,
                        7.3525,
                        6.9094,
                        6.4931,
                        6.1019,
                        5.7342,
                        5.3887,
                        5.0640,
                        4.7589,
                        4.4721,
                        4.2027,
                        3.9494,
                        3.7115,
                        3.4878,
                        3.2777,
                        3.0802,
                        2.8946,
                        2.7202,
                        2.5563,
                        2.4022,
                        2.2575,
                        2.1215,
                        1.9936,
                        1.8735,
                        1.7606,
                        1.6545,
                        1.5549,
                        1.4612,
                        1.3731,
                        1.2904,
                        1.2126,
                        1.1396,
                        1.0709,
                        1.0064,
                        0.9457,
                        0.8888,
                        0.8352,
                        0.7849,
                        0.7376,
                        0.6931,
                        0.6514,
                        0.6121,
                        0.5752,
                        0.5406,
                        0.5080,
                        0.4774,
                        0.4486,
                        0.4216,
                        0.3962,
                        0.3723,
                        0.3499,
                        0.3288,
                        0.3090,
                        0.2904,
                        0.2729,
                        0.2564,
                        0.2410,
                        0.2265,
                        0.2128,
                        0.2000,
                        0.1879,
                        0.1766,
                        0.1660,
                        0.1560,
                        0.1466,
                        0.1378,
                        0.1295,
                        0.1217,
                        0.1143,
                        0.1074,
                        0.1010,
                        0.1000,
                    ],
                    pystrata.output.OutputLocation("outcrop", depth=0),
                    0.05,
                ),
            ]
        )
        cls.outputs(cls.calc)

    def test_times(self):
        ref = self.ref["time_series"]["time"]
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[0].refs[:n], ref, rtol=self.rtol, atol=self.atol
        )

    def test_accels(self):
        ref = self.ref["time_series"]["time"]
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[0].refs[:n], ref, rtol=self.rtol, atol=self.atol
        )

    def test_arias_ints(self):
        ref = self.ref["time_series"]["arias_int"]
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[1].values[:n], ref, rtol=self.rtol, atol=self.atol
        )

    def test_strains(self):
        ref = self.ref["time_series"]["strain"]
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[2].values[:n], ref, rtol=self.rtol, atol=self.atol
        )

    def test_stresses(self):
        ref = self.ref["time_series"]["stress"]
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[3].values[:n], ref, rtol=self.rtol, atol=self.atol
        )

    def test_periods(self):
        np.testing.assert_allclose(
            self.outputs[4].periods,
            self.ref["resp_spec"]["period"],
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_osc_freqs(self):
        np.testing.assert_allclose(
            1 / self.outputs[4].refs,
            self.ref["resp_spec"]["period"],
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_spec_accels(self):
        np.testing.assert_allclose(
            self.outputs[4].values,
            self.ref["resp_spec"]["psa"],
            rtol=self.rtol,
            atol=self.atol,
        )


class TestExample02LE(DeepSoilComparison):
    # Test the linear elastic wave propagation
    ref_name = "ds-example-2a-le"
    calc = pystrata.propagation.LinearElasticCalculator()
    profile = pystrata.site.Profile(
        [
            pystrata.site.Layer(
                pystrata.site.SoilType("Soil", 20.0, mod_reduc=None, damping=0), 20, 500
            ),
            pystrata.site.Layer(
                pystrata.site.SoilType("Rock", 25.0, mod_reduc=None, damping=0.02),
                0,
                760,
            ),
        ]
    )


class TestExample02EL(DeepSoilComparison):
    # Run the linear elastic test with the EL calculator.
    ref_name = "ds-example-2a-le"
    calc = pystrata.propagation.EquivalentLinearCalculator()
    profile = pystrata.site.Profile(
        [
            pystrata.site.Layer(
                pystrata.site.SoilType("Soil", 20.0, mod_reduc=None, damping=0), 20, 500
            ),
            pystrata.site.Layer(
                pystrata.site.SoilType("Rock", 25.0, mod_reduc=None, damping=0.02),
                0,
                760,
            ),
        ]
    )


class TestExample04EL(DeepSoilComparison):
    ref_name = "ds-example-4-eql"
    calc = pystrata.propagation.EquivalentLinearCalculator()
    profile = pystrata.site.Profile(
        [
            pystrata.site.Layer(
                pystrata.site.SoilType("Soil", 20.0, mod_reduc=None, damping=0.05),
                20,
                500,
            ),
            pystrata.site.Layer(
                pystrata.site.SoilType("Rock", 25.0, mod_reduc=None, damping=0.02),
                0,
                760,
            ),
        ]
    )


class TestExample02LEPython(TestExample02LE):
    """Test linear elastic wave propagation with pure-Python dispatch."""

    _use_python = True


class TestExample02ELPython(TestExample02EL):
    """Test EQL (linear elastic input) with pure-Python dispatch."""

    _use_python = True


class TestExample04ELPython(TestExample04EL):
    """Test EQL with pure-Python dispatch."""

    _use_python = True


class QWLComparison:
    rtol = 0.01
    atol = 0.01

    index = NotImplementedError

    @classmethod
    def setup_class(cls):
        fpath = FPATH_DATA / "qwl_tests.json"
        data = json.load(fpath.open())[cls.index]

        thickness = np.diff(data["site"]["depth"])

        profile = pystrata.site.Profile()
        for i, thick in enumerate(thickness):
            if "damping" in data["site"]:
                damping = data["site"]["damping"][i]
            else:
                damping = None

            profile.append(
                pystrata.site.Layer(
                    pystrata.site.SoilType(
                        f"{i}",
                        data["site"]["density"][i] * pystrata.motion.GRAVITY,
                        damping=damping,
                    ),
                    thick * 1000,
                    data["site"]["velocity"][i] * 1000,
                )
            )

        profile.update_layers()

        if "site_atten" in data["site"]:
            site_atten = data["site"]["site_atten"]
        else:
            site_atten = profile.site_attenuation()

        cls.motion = pystrata.motion.Motion(data["freqs"])
        cls.calc = pystrata.propagation.QuarterWaveLenCalculator(site_atten=site_atten)
        cls.calc(cls.motion, profile, profile.location("outcrop", index=-1))
        cls.data = data

    def test_crustal_amp(self):
        ref = self.data["crustal_amp"]
        np.testing.assert_allclose(
            self.calc.crustal_amp, ref, rtol=self.rtol, atol=self.atol
        )

    def test_site_term(self):
        ref = self.data["site_term"]
        np.testing.assert_allclose(
            self.calc.site_term, ref, rtol=self.rtol, atol=self.atol
        )


class TestQwl0(QWLComparison):
    index = 0


# Not sure why this test fails. I suspect there is a typo in one of the
# equations
@pytest.mark.xfail
class TestQwl1(QWLComparison):
    index = 1


# class TestQwl2(QWLComparison):
#     rtol = 0.05
#     index = 2


def test_quarter_wavelength_fit():
    fpath = FPATH_DATA / "qwl_tests.json"
    data = json.load(fpath.open())[0]
    thickness = np.diff(data["site"]["depth"])

    profile = pystrata.site.Profile()
    for i, (thick, vel, density) in enumerate(
        zip(thickness, data["site"]["velocity"], data["site"]["density"])
    ):
        profile.append(
            pystrata.site.Layer(
                pystrata.site.SoilType(f"{i}", density * pystrata.motion.GRAVITY),
                thick * 1000,
                vel * 1000,
            )
        )

    profile.update_layers()

    motion = pystrata.motion.Motion(data["freqs"])
    calc = pystrata.propagation.QuarterWaveLenCalculator(
        site_atten=data["site"]["site_atten"]
    )
    calc(motion, profile, profile.location("outcrop", index=-1))

    calc.fit("crustal_amp", data["crustal_amp"])

    np.testing.assert_allclose(
        profile.initial_shear_vel, calc.profile.initial_shear_vel, rtol=0.2
    )


def test_linear_elastic_nrattle():
    """Test against nrattle."""
    ctl = pystrata.tools.read_nrattle_ctl(FPATH_DATA / "nrattle.ctl")
    results = np.rec.fromrecords(
        np.loadtxt(
            FPATH_DATA / "test_nrattle_02mar12.nrattle_amps4plot.out",
            skiprows=19,
            usecols=(0, 2),
        ),
        names="freq,amp",
    )

    # Create pystrata objects
    profile = pystrata.tools.profile_from_nrattle_ctl(ctl)
    motion = pystrata.motion.Motion(freqs=results.freq)
    calc = pystrata.propagation.LinearElasticCalculator()

    loc_in = profile.location("outcrop", index=(ctl["hs_layer"] - 1))
    loc_out = profile.location("outcrop", depth=ctl["out_depth"])

    calc(
        motion,
        profile,
        # Need to convert to zero-based indices
        profile.location("outcrop", index=(ctl["hs_layer"] - 1)),
    )
    trans_func = np.abs(calc.calc_accel_tf(loc_in, loc_out))
    # Data is provided to 4 decimal places
    rounded = np.round(trans_func, 4)

    if 0:
        # Make a plot for debugging
        fig = plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(nrows=3)

        ax = fig.add_subplot(spec[:2])
        ax.plot(results.freq, results.amp, label="nRattle")
        ax.plot(motion.freqs, trans_func, label="pystrata", ls="--")

        ax.legend()
        ax.set(xscale="linear", ylabel="|Transfer Func.|")

        ax = fig.add_subplot(spec[2])
        ax.plot(results.freq, (rounded - results.amp) / results.amp)
        ax.set_label("Rel. Diff.")
        ax.set(xlabel="Frequency (Hz)", xscale="linear", ylabel="Rel. Diff.")

        fig.savefig("test_propagation-test_linear_elastic_nrattle.png", dpi=200)

    np.testing.assert_allclose(
        results.amp,
        rounded,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# General (inhomogeneous / obliquely incident) SII wave propagation
# ---------------------------------------------------------------------------


def _general_profile(soil_damping=0.03, rock_damping=0.01):
    """Two-layer soil-over-rock profile used by the general-wave tests."""
    profile = pystrata.site.Profile(
        [
            pystrata.site.Layer(
                pystrata.site.SoilType("Soil", 18.0, None, soil_damping), 30, 300
            ),
            pystrata.site.Layer(
                pystrata.site.SoilType("Rock", 22.0, None, rock_damping), 0, 900
            ),
        ]
    )
    profile.update_layers()
    return profile


def _general_motion():
    freqs = np.logspace(-1, 2, 1024)
    return pystrata.motion.Motion(freqs=freqs)


def _run_accel_tf(profile, motion, **kwds):
    calc = pystrata.propagation.LinearElasticCalculator(**kwds)
    loc_in = profile.location("outcrop", index=len(profile) - 1)
    loc_out = profile.location("outcrop", index=0)
    calc(motion, profile, loc_in)
    return calc, np.abs(calc.calc_accel_tf(loc_in, loc_out))


def test_general_default_unchanged():
    """`incidence_angle=0` must use the untouched path and be bit-identical."""
    profile = _general_profile()
    motion = _general_motion()
    _, tf_default = _run_accel_tf(profile, motion)
    _, tf_zero = _run_accel_tf(_general_profile(), motion, incidence_angle=0.0)
    assert np.array_equal(tf_default, tf_zero)


def test_general_reduces_to_homogeneous():
    """The general kernel at theta->0 reproduces the normal-incidence result."""
    profile = _general_profile()
    motion = _general_motion()
    _, tf_default = _run_accel_tf(profile, motion)
    # A tiny angle forces the general path but is physically negligible.
    _, tf_general = _run_accel_tf(_general_profile(), motion, incidence_angle=1e-7)
    np.testing.assert_allclose(tf_general, tf_default, rtol=1e-9)


@pytest.mark.parametrize("theta", [0, 20, 40, 60])
def test_general_vertical_wave_number(theta):
    """Layer vertical wave numbers follow d_beta = sqrt(k_S**2 - k**2) exactly.

    With elastic (undamped) layers the wave numbers are real, so they can be
    compared to the closed-form Snell's-law expression to machine precision. The
    horizontal slowness is fixed by the half space (Vs = 900 m/s).
    """
    profile = _general_profile(soil_damping=0.0, rock_damping=0.0)
    motion = pystrata.motion.Motion(freqs=np.linspace(0.1, 10, 64))
    calc, _ = _run_accel_tf(profile, motion, incidence_angle=theta)

    ang_freqs = motion.angular_freqs
    sin_t = np.sin(np.radians(theta))
    # Soil layer (Vs = 300) with horizontal slowness set by the rock (Vs = 900).
    expected_soil = (ang_freqs / 300.0) * np.sqrt(1 - (300 / 900) ** 2 * sin_t**2)
    np.testing.assert_allclose(calc._wave_nums[0].real, expected_soil, rtol=1e-10)
    # Half space: d_beta = (omega / Vs) * cos(theta).
    expected_rock = (ang_freqs / 900.0) * np.cos(np.radians(theta))
    np.testing.assert_allclose(calc._wave_nums[1].real, expected_rock, rtol=1e-10)


@pytest.mark.parametrize("theta", [15, 30, 45])
def test_general_resonance_shift(theta):
    """Fundamental resonance shifts with incidence angle per Snell's law.

    For an elastic soil layer (Vs, thickness H) over a stiffer half space, the
    vertical wave number in the soil is k_S * sqrt(1 - (Vs/Vs_hs)**2 sin^2 theta),
    so the fundamental frequency is Vs/(4H) / sqrt(1 - (Vs/Vs_hs)**2 sin^2 theta).
    """
    profile = _general_profile(soil_damping=0.0, rock_damping=0.0)
    # Fine linear grid around the ~2.5 Hz fundamental for a sharp peak location.
    motion = pystrata.motion.Motion(freqs=np.linspace(2.0, 3.2, 4000))
    _, tf = _run_accel_tf(profile, motion, incidence_angle=theta)

    f_peak = motion.freqs[np.argmax(tf)]
    vs, h, vs_hs = 300.0, 30.0, 900.0
    factor = np.sqrt(1 - (vs / vs_hs) ** 2 * np.sin(np.radians(theta)) ** 2)
    f_expected = vs / (4 * h) / factor
    np.testing.assert_allclose(f_peak, f_expected, rtol=5e-3)


def test_general_python_numba_parity():
    """Pure-Python and numba general kernels agree for an oblique case."""
    motion = _general_motion()
    _, tf_numba = _run_accel_tf(
        _general_profile(), motion, incidence_angle=30, inhomogeneity=10
    )
    with _use_python_dispatch():
        _, tf_python = _run_accel_tf(
            _general_profile(), motion, incidence_angle=30, inhomogeneity=10
        )
    np.testing.assert_allclose(tf_python, tf_numba, rtol=1e-10)


def test_general_inhomogeneity_normal_incidence():
    """A normally incident inhomogeneous wave (theta=0, gamma!=0) is admissible.

    Even at normal incidence a nonzero degree of inhomogeneity yields a finite, physical
    response that differs from the homogeneous case.
    """
    profile = _general_profile()
    motion = _general_motion()
    _, tf_homog = _run_accel_tf(_general_profile(), motion)
    _, tf_inhomog = _run_accel_tf(profile, motion, incidence_angle=0, inhomogeneity=20)
    assert np.all(np.isfinite(tf_inhomog))
    assert np.any(np.abs(tf_inhomog - tf_homog) > 1e-6)


def test_general_zero_frequency_and_grazing_finite():
    """Zero frequency and near-grazing incidence stay finite (no NaN/Inf)."""
    profile = _general_profile()
    # Include f = 0 and a steep angle; verify the transfer function is finite.
    motion = pystrata.motion.Motion(freqs=np.linspace(0.0, 50.0, 512))
    _, tf = _run_accel_tf(profile, motion, incidence_angle=80, inhomogeneity=30)
    assert np.all(np.isfinite(tf))


def test_general_equivalent_linear_forwards_kwargs():
    """EQL and FDM forward the incidence/inhomogeneity parameters."""
    eql = pystrata.propagation.EquivalentLinearCalculator(
        incidence_angle=25, inhomogeneity=10
    )
    assert eql.incidence_angle == pytest.approx(25)
    assert eql.inhomogeneity == pytest.approx(10)

    fdm = pystrata.propagation.FrequencyDependentEqlCalculator(
        incidence_angle=25, inhomogeneity=10
    )
    assert fdm.incidence_angle == pytest.approx(25)
    assert fdm.inhomogeneity == pytest.approx(10)


#
# def compare_ts_results(calc, name):
#     ref_soil, ref_inp = (name)
#
#     # Compare the time series
#     # Only compare the number of values in the DeepSoil results,
#     # which doesn't include the zero padding added by the FFT.
#     n = len(ref_soil['time_series']['accel'])
#     loc_surface = pystrata.output.OutputLocation('outcrop', index=0)
#     loc_midheight = pystrata.output.OutputLocation(
#         'within', depth=(calc.profile[0].thickness / 2))
#     for key, output in [
#         ('accel',),
#         ('arias_int',
#          pystrata.output.AriasIntensityTSOutput(loc_surface)),
#         ('strain', pystrata.output.StrainTSOutput(loc_midheight)),
#         ('stress', pystrata.output.StressTSOutput(loc_midheight, damped=False)),
#     ]:
#         output(calc)
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots()
#         ax.plot(
#             ref_soil['time_series']['time'],
#             ref_soil['time_series'][key], 'b-'
#         )
#         ax.plot(output.refs, output.values, 'r--')
#         ax.set_xlim(10, 40)
#         fig.tight_layout()
#         fig.savefig('test')
#
#
#
# def test_linear(ts):
#     calc =
#     calc(ts, profile, profile.location('outcrop', index=-1))
#     compare_ts_results(calc, '')
#
#
# def test_equiv_linear():
#     pass
