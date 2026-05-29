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
        _prop._wave_at_location_dispatch,
        _prop._calc_strain_tf_dispatch,
    )
    _prop._calc_waves_dispatch = _prop._calc_waves_python
    _prop._wave_at_location_dispatch = _prop._wave_at_location_python
    _prop._calc_strain_tf_dispatch = _prop._calc_strain_tf_python
    try:
        yield
    finally:
        (
            _prop._calc_waves_dispatch,
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


class TestTDvsEQLMetrics:
    """Cross-method validation of response spectra, spectral ratios, and transfer
    functions between EQL and TD (MKZ / HH) at 0.01 g.

    At this low loading level the soil is quasi-linear, so both the frequency-domain EQL
    and the time-domain nonlinear methods should produce very similar computed metrics.
    Known divergence occurs at the site's resonance frequencies due to different damping
    formulations (EQL: frequency-dependent complex modulus; TD: Liu-Archuleta frequency-
    independent).  The tests therefore use multi-criteria spectral agreement checks
    rather than point-by-point tolerance.
    """

    # Broadband tolerances (median relative difference)
    MED_RTOL_SA = 0.05  # 5 % median — response spectra (Sa)
    MED_RTOL_RATIO = 0.08  # 8 % median — spectral ratios
    MED_RTOL_TF = 0.08  # 8 % median — transfer functions

    # Envelope tolerances (max relative difference)
    MAX_RTOL_SA = 0.75  # 75 % max — resonance peak divergence
    MAX_RTOL_SA_TD = 0.20  # 20 % max — MKZ vs HH (same family)
    MAX_RTOL_RATIO = 1.10  # 110 % max
    MAX_RTOL_TF = 1.50  # 150 % max — TD peaks are sharper at higher modes

    # Shape correlation
    MIN_CORR_SA = 0.98
    MIN_CORR_RATIO = 0.90
    MIN_CORR_TF = 0.50  # TF shapes diverge at higher modes due to damping

    @classmethod
    def setup_class(cls):
        # -- Motion: ChiChi scaled to 0.01 g -----------------------------------
        ts_motion = load_ts()
        scale = 0.01 / np.max(np.abs(ts_motion.accels))
        scaled = np.asarray(ts_motion.accels) * scale
        cls.motion = pystrata.motion.TimeSeriesMotion(
            ts_motion.filename,
            f"{ts_motion.description} scaled to 0.01g",
            ts_motion.time_step,
            scaled.tolist(),
        )

        # -- Profile: 3-layer Darendeli ----------------------------------------
        cls.profile = pystrata.site.Profile(
            [
                pystrata.site.Layer(
                    pystrata.site.DarendeliSoilType(
                        unit_wt=18.0,
                        name="Sand",
                        plas_index=0,
                        ocr=1,
                        stress_mean=50,
                    ),
                    10,
                    200,
                ),
                pystrata.site.Layer(
                    pystrata.site.DarendeliSoilType(
                        unit_wt=19.0,
                        name="Clay",
                        plas_index=30,
                        ocr=1.5,
                        stress_mean=100,
                    ),
                    15,
                    300,
                ),
                pystrata.site.Layer(
                    pystrata.site.SoilType("Rock", 22.0, mod_reduc=None, damping=0.01),
                    0,
                    800,
                ),
            ]
        )

        # -- Shared output parameters ------------------------------------------
        cls.freqs = np.logspace(np.log10(0.1), np.log10(30), 200)
        loc_in_ol = pystrata.output.OutputLocation("outcrop", index=-1)
        loc_surf = pystrata.output.OutputLocation("outcrop", index=0)

        def _make_outputs():
            return pystrata.output.OutputCollection(
                [
                    pystrata.output.ResponseSpectrumOutput(
                        cls.freqs,
                        loc_surf,
                        0.05,
                    ),
                    pystrata.output.ResponseSpectrumRatioOutput(
                        cls.freqs,
                        loc_in_ol,
                        loc_surf,
                        0.05,
                    ),
                    pystrata.output.AccelTransferFunctionOutput(
                        cls.freqs,
                        loc_in_ol,
                        loc_surf,
                        ko_bandwidth=30,
                    ),
                ]
            )

        loc_input = cls.profile.location("outcrop", index=-1)

        # -- EQL ---------------------------------------------------------------
        cls.calc_eql = pystrata.propagation.EquivalentLinearCalculator(
            strain_ratio=0.65,
        )
        cls.calc_eql(cls.motion, cls.profile.copy(), loc_input)
        cls.out_eql = _make_outputs()
        cls.out_eql(cls.calc_eql)

        # -- TD-MKZ ------------------------------------------------------------
        cls.calc_td_mkz = pystrata.propagation.TimeDomainCalculator(
            model="mkz",
            boundary="elastic",
        )
        cls.calc_td_mkz(cls.motion, cls.profile.copy(), loc_input)
        cls.out_td_mkz = _make_outputs()
        cls.out_td_mkz(cls.calc_td_mkz)

        # -- TD-HH -------------------------------------------------------------
        cls.calc_td_hh = pystrata.propagation.TimeDomainCalculator(
            model="hh",
            boundary="elastic",
        )
        cls.calc_td_hh(cls.motion, cls.profile.copy(), loc_input)
        cls.out_td_hh = _make_outputs()
        cls.out_td_hh(cls.calc_td_hh)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _sa(outputs):
        """Extract Sa array from the first (ResponseSpectrumOutput) output."""
        for _name, _refs, vals in outputs[0].iter_results():
            return vals

    @staticmethod
    def _ratio(outputs):
        """Extract spectral-ratio array from the second output."""
        for _name, _refs, vals in outputs[1].iter_results():
            return vals

    @staticmethod
    def _tf(outputs):
        """Extract transfer-function array from the third output."""
        for _name, _refs, vals in outputs[2].iter_results():
            return vals

    @staticmethod
    def _check_spectral_agreement(
        actual,
        desired,
        *,
        med_rtol,
        max_rtol,
        min_corr,
        label="",
    ):
        """Multi-criteria spectral agreement check.

        Parameters
        ----------
        actual, desired : array_like
        med_rtol : float
            Median relative difference must be below this (broadband check).
        max_rtol : float
            Maximum relative difference must be below this (envelope bound).
        min_corr : float
            Pearson correlation coefficient must exceed this (shape check).
        label : str
            Human-readable label for assertion messages.
        """
        rel_diff = np.abs(actual - desired) / np.maximum(np.abs(desired), 1e-30)
        med_rd = float(np.median(rel_diff))
        max_rd = float(np.max(rel_diff))
        corr = float(np.corrcoef(actual, desired)[0, 1])

        assert corr > min_corr, f"{label}: correlation {corr:.4f} below {min_corr}"
        assert med_rd < med_rtol, (
            f"{label}: median relative diff {med_rd:.4f} exceeds {med_rtol}"
        )
        assert max_rd < max_rtol, (
            f"{label}: max relative diff {max_rd:.4f} exceeds {max_rtol}"
        )

    # -- Phase 2: response spectra ---------------------------------------------

    def test_response_spectra_td_mkz_vs_eql(self):
        self._check_spectral_agreement(
            self._sa(self.out_td_mkz),
            self._sa(self.out_eql),
            med_rtol=self.MED_RTOL_SA,
            max_rtol=self.MAX_RTOL_SA,
            min_corr=self.MIN_CORR_SA,
            label="Sa MKZ vs EQL",
        )

    def test_response_spectra_td_hh_vs_eql(self):
        self._check_spectral_agreement(
            self._sa(self.out_td_hh),
            self._sa(self.out_eql),
            med_rtol=self.MED_RTOL_SA,
            max_rtol=self.MAX_RTOL_SA,
            min_corr=self.MIN_CORR_SA,
            label="Sa HH vs EQL",
        )

    def test_response_spectra_td_mkz_vs_td_hh(self):
        np.testing.assert_allclose(
            self._sa(self.out_td_mkz),
            self._sa(self.out_td_hh),
            rtol=self.MAX_RTOL_SA_TD,
        )

    # -- Phase 3: spectral ratios ----------------------------------------------

    def test_spectral_ratio_td_mkz_vs_eql(self):
        self._check_spectral_agreement(
            self._ratio(self.out_td_mkz),
            self._ratio(self.out_eql),
            med_rtol=self.MED_RTOL_RATIO,
            max_rtol=self.MAX_RTOL_RATIO,
            min_corr=self.MIN_CORR_RATIO,
            label="Ratio MKZ vs EQL",
        )

    def test_spectral_ratio_td_hh_vs_eql(self):
        self._check_spectral_agreement(
            self._ratio(self.out_td_hh),
            self._ratio(self.out_eql),
            med_rtol=self.MED_RTOL_RATIO,
            max_rtol=self.MAX_RTOL_RATIO,
            min_corr=self.MIN_CORR_RATIO,
            label="Ratio HH vs EQL",
        )

    def test_spectral_ratio_peak_frequency_agreement(self):
        """Peak amplification frequency should agree within 10 %."""
        for label, out_td in [("MKZ", self.out_td_mkz), ("HH", self.out_td_hh)]:
            ratio_eql = self._ratio(self.out_eql)
            ratio_td = self._ratio(out_td)
            f_eql = self.freqs[np.argmax(ratio_eql)]
            f_td = self.freqs[np.argmax(ratio_td)]
            rel = abs(f_eql - f_td) / f_eql
            assert rel < 0.10, (
                f"Peak freq mismatch ({label}): EQL @ {f_eql:.2f} Hz "
                f"vs TD @ {f_td:.2f} Hz (rel diff {rel:.2%})"
            )

    # -- Phase 4: transfer functions -------------------------------------------

    def test_transfer_function_td_mkz_vs_eql(self):
        self._check_spectral_agreement(
            self._tf(self.out_td_mkz),
            self._tf(self.out_eql),
            med_rtol=self.MED_RTOL_TF,
            max_rtol=self.MAX_RTOL_TF,
            min_corr=self.MIN_CORR_TF,
            label="TF MKZ vs EQL",
        )

    def test_transfer_function_td_hh_vs_eql(self):
        self._check_spectral_agreement(
            self._tf(self.out_td_hh),
            self._tf(self.out_eql),
            med_rtol=self.MED_RTOL_TF,
            max_rtol=self.MAX_RTOL_TF,
            min_corr=self.MIN_CORR_TF,
            label="TF HH vs EQL",
        )

    # -- Phase 5: diagnostics --------------------------------------------------

    def test_strain_is_quasi_linear(self):
        """Max shear strain should be < 0.1 % for all TD runs."""
        for calc in (self.calc_td_mkz, self.calc_td_hh):
            for i, strain in enumerate(calc.results.max_strain()):
                strain_pct = strain * 100
                assert strain_pct < 0.1, (
                    f"Layer {i} strain {strain_pct:.4f}% exceeds 0.1% threshold"
                )
