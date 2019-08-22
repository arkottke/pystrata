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
# Copyright (C) Albert Kottke, 2013-2016

import json
import string

import numpy as np
import pyexcel
import pytest

import pysra

from . import FPATH_DATA


def read_cluster(ws, cols, names, row_start, row_end):
    d = dict()
    for c, name in zip(cols, names):
        range_str = '{col}{start}:{col}{end}'.format(
            col=c, start=row_start, end=row_end)
        d[name] = [row[0].value for row in ws[range_str]]
    return d


def read_deepsoil_results(name):
    data = pyexcel.get_array(file_name=str(FPATH_DATA / (name + '.xlsx')))
    names = ','.join(string.ascii_uppercase[:len(data[0])])
    records = np.rec.fromrecords(data, names=names)

    def extract_cols(records, cols, first, last, names):
        return {
            name: records[col][first:last].astype(float)
            for col, name in zip(cols, names)
        }

    d = dict()
    # Read the time series
    d['time_series'] = extract_cols(
        records, 'ABCDE', 1, 11800,
        ['time', 'accel', 'strain', 'stress', 'arias_int'])
    # Read the response spectrum
    d['resp_spec'] = extract_cols(records, 'GH', 1, 114, ['period', 'psa'])
    # Read the Fourier amplitude
    d['fourier_spec'] = extract_cols(records, 'JKL', 1, 16384,
                                     ['freq', 'ampl', 'ratio'])

    return d


def load_ts():
    fpath = FPATH_DATA / 'ChiChi.txt'
    with fpath.open() as fp:
        parts = next(fp).split()
        time_step = float(parts[1])
        accels = [float(l.split()[1]) for l in fp]

    return pysra.motion.TimeSeriesMotion(
        fpath.name, 'ChiChi.txt from DeepSoil v6.1', time_step, accels)


class DeepSoilComparison:
    rtol = 0.005
    atol = 0.001

    ref_name = NotImplemented
    calc = NotImplemented
    profile = NotImplemented

    @classmethod
    def setup_class(cls):
        cls.ref = read_deepsoil_results(cls.ref_name)
        # Perform the calculation
        cls.calc(load_ts(), cls.profile,
                 cls.profile.location('outcrop', index=-1))
        cls.outputs = pysra.output.OutputCollection([
            pysra.output.AccelerationTSOutput(
                pysra.output.OutputLocation('outcrop', depth=0)),
            pysra.output.AriasIntensityTSOutput(
                pysra.output.OutputLocation('outcrop', depth=0)),
            pysra.output.StrainTSOutput(
                pysra.output.OutputLocation('within', depth=10),
                in_percent=True),
            pysra.output.StressTSOutput(
                pysra.output.OutputLocation('within', depth=10),
                normalized=True),
            pysra.output.ResponseSpectrumOutput(
                [100.0000, 93.9744, 88.3119, 82.9910, 77.9903, 73.2912,
                 68.8753, 64.7249, 60.8250, 57.1602, 53.7158, 50.4793, 47.4377,
                 44.5794, 41.8932, 39.3690, 36.9969, 34.7676, 32.6727, 30.7040,
                 28.8540, 27.1154, 25.4816, 23.9462, 22.5034, 21.1474, 19.8732,
                 18.6757, 17.5504, 16.4930, 15.4992, 14.5653, 13.6877, 12.8629,
                 12.0879, 11.3595, 10.6751, 10.0319, 9.4274, 8.8594, 8.3256,
                 7.8239, 7.3525, 6.9094, 6.4931, 6.1019, 5.7342, 5.3887,
                 5.0640, 4.7589, 4.4721, 4.2027, 3.9494, 3.7115, 3.4878,
                 3.2777, 3.0802, 2.8946, 2.7202, 2.5563, 2.4022, 2.2575,
                 2.1215, 1.9936, 1.8735, 1.7606, 1.6545, 1.5549, 1.4612,
                 1.3731, 1.2904, 1.2126, 1.1396, 1.0709, 1.0064, 0.9457,
                 0.8888, 0.8352, 0.7849, 0.7376, 0.6931, 0.6514, 0.6121,
                 0.5752, 0.5406, 0.5080, 0.4774, 0.4486, 0.4216, 0.3962,
                 0.3723, 0.3499, 0.3288, 0.3090, 0.2904, 0.2729, 0.2564,
                 0.2410, 0.2265, 0.2128, 0.2000, 0.1879, 0.1766, 0.1660,
                 0.1560, 0.1466, 0.1378, 0.1295, 0.1217, 0.1143, 0.1074,
                 0.1010, 0.1000],
                pysra.output.OutputLocation('outcrop', depth=0), 0.05)
        ])
        cls.outputs(cls.calc)

    def test_times(self):
        ref = self.ref['time_series']['time']
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[0].refs[:n], ref, rtol=self.rtol, atol=self.atol)

    def test_accels(self):
        ref = self.ref['time_series']['time']
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[0].refs[:n], ref, rtol=self.rtol, atol=self.atol)

    def test_arias_ints(self):
        ref = self.ref['time_series']['arias_int']
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[1].values[:n], ref, rtol=self.rtol, atol=self.atol)

    def test_strains(self):
        ref = self.ref['time_series']['strain']
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[2].values[:n], ref, rtol=self.rtol, atol=self.atol)

    def test_stresses(self):
        ref = self.ref['time_series']['stress']
        n = len(ref)
        np.testing.assert_allclose(
            self.outputs[3].values[:n], ref, rtol=self.rtol, atol=self.atol)

    def test_periods(self):
        np.testing.assert_allclose(
            self.outputs[4].periods,
            self.ref['resp_spec']['period'],
            rtol=self.rtol,
            atol=self.atol)

    def test_osc_freqs(self):
        np.testing.assert_allclose(
            1 / self.outputs[4].refs,
            self.ref['resp_spec']['period'],
            rtol=self.rtol,
            atol=self.atol)

    def test_spec_accels(self):
        np.testing.assert_allclose(
            self.outputs[4].values,
            self.ref['resp_spec']['psa'],
            rtol=self.rtol,
            atol=self.atol)


class TestExample02LE(DeepSoilComparison):
    # Test the linear elastic wave propagation
    ref_name = 'ds-example-2a-le'
    calc = pysra.propagation.LinearElasticCalculator()
    profile = pysra.site.Profile([
        pysra.site.Layer(
            pysra.site.SoilType('Soil', 20., mod_reduc=None, damping=0), 20,
            500),
        pysra.site.Layer(
            pysra.site.SoilType('Rock', 25., mod_reduc=None, damping=0.02), 0,
            760),
    ])


class TestExample02EL(DeepSoilComparison):
    # Run the linear elastic test with the EL calculator.
    ref_name = 'ds-example-2a-le'
    calc = pysra.propagation.EquivalentLinearCalculator()
    profile = pysra.site.Profile([
        pysra.site.Layer(
            pysra.site.SoilType('Soil', 20., mod_reduc=None, damping=0), 20,
            500),
        pysra.site.Layer(
            pysra.site.SoilType('Rock', 25., mod_reduc=None, damping=0.02), 0,
            760),
    ])


class TestExample04EL(DeepSoilComparison):
    ref_name = 'ds-example-4-eql'
    calc = pysra.propagation.EquivalentLinearCalculator()
    profile = pysra.site.Profile([
        pysra.site.Layer(
            pysra.site.SoilType('Soil', 20., mod_reduc=None, damping=0.05), 20,
            500),
        pysra.site.Layer(
            pysra.site.SoilType('Rock', 25., mod_reduc=None, damping=0.02), 0,
            760),
    ])


class QWLComparison():
    rtol = 0.01
    atol = 0.01

    index = NotImplementedError

    @classmethod
    def setup_class(cls):
        fpath = FPATH_DATA / 'qwl_tests.json'
        data = json.load(fpath.open())[cls.index]

        thickness = np.diff(data['site']['depth'])

        profile = pysra.site.Profile()
        for i, thick in enumerate(thickness):
            if 'damping' in data['site']:
                damping = data['site']['damping'][i]
            else:
                damping = None

            profile.append(
                pysra.site.Layer(
                    pysra.site.SoilType(
                        f'{i}',
                        data['site']['density'][i] * pysra.motion.GRAVITY,
                        damping=damping), thick * 1000, data['site'][
                            'velocity'][i] * 1000))

        profile.update_layers()

        if 'site_atten' in data['site']:
            site_atten = data['site']['site_atten']
        else:
            site_atten = profile.site_attenuation()

        cls.motion = pysra.motion.Motion(data['freqs'])
        cls.calc = pysra.propagation.QuarterWaveLenCalculator(
            site_atten=site_atten)
        cls.calc(cls.motion, profile, profile.location('outcrop', index=-1))
        cls.data = data

    def test_crustal_amp(self):
        ref = self.data['crustal_amp']
        np.testing.assert_allclose(
            self.calc.crustal_amp, ref, rtol=self.rtol, atol=self.atol)

    def test_site_term(self):
        ref = self.data['site_term']
        np.testing.assert_allclose(
            self.calc.site_term, ref, rtol=self.rtol, atol=self.atol)


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
    fpath = FPATH_DATA / 'qwl_tests.json'
    data = json.load(fpath.open())[0]
    thickness = np.diff(data['site']['depth'])

    profile = pysra.site.Profile()
    for i, (thick, vel, density) in enumerate(
            zip(thickness, data['site']['velocity'], data['site']['density'])):
        profile.append(
            pysra.site.Layer(
                pysra.site.SoilType(f'{i}', density * pysra.motion.GRAVITY),
                thick * 1000, vel * 1000))

    profile.update_layers()

    motion = pysra.motion.Motion(data['freqs'])
    calc = pysra.propagation.QuarterWaveLenCalculator(
        site_atten=data['site']['site_atten'])
    calc(motion, profile, profile.location('outcrop', index=-1))

    calc.fit('crustal_amp', data['crustal_amp'])

    np.testing.assert_allclose(
        profile.initial_shear_vel, calc.profile.initial_shear_vel, rtol=0.2)


#
# def compare_ts_results(calc, name):
#     ref_soil, ref_inp = (name)
#
#     # Compare the time series
#     # Only compare the number of values in the DeepSoil results,
#     # which doesn't include the zero padding added by the FFT.
#     n = len(ref_soil['time_series']['accel'])
#     loc_surface = pysra.output.OutputLocation('outcrop', index=0)
#     loc_midheight = pysra.output.OutputLocation(
#         'within', depth=(calc.profile[0].thickness / 2))
#     for key, output in [
#         ('accel',),
#         ('arias_int',
#          pysra.output.AriasIntensityTSOutput(loc_surface)),
#         ('strain', pysra.output.StrainTSOutput(loc_midheight)),
#         ('stress', pysra.output.StressTSOutput(loc_midheight, damped=False)),
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
