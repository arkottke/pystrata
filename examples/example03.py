#!/usr/bin/env python
# encoding: utf-8

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
"""Compute transfer functions for within and outcrop conditions."""

import matplotlib.pyplot as plt
import numpy as np

import pysra

motion = pysra.motion.SourceTheoryRvtMotion(6.5, 20, 'wna')
motion.calc_fourier_amps()
profile = pysra.site.Profile([
    pysra.site.Layer(
        pysra.site.DarendeliSoilType(
            'Soil', 18., plas_index=0, ocr=1, mean_stress=0.50), 30, 400),
    pysra.site.Layer(pysra.site.SoilType('Rock', 24., None, 0.01), 0, 1200),
])

osc_freqs = np.logspace(-1, 2, 181)
outputs = pysra.output.OutputCollection(
    pysra.output.AccelTransferFunctionOutput(
        motion.freqs,
        # Input (outcrop). Here the wave_field is specified as a string,
        # but the enum value is looked up during the initialization of the
        # OutputLocation object. The location is specified by the index of
        # the layer within the profile 0 for top, and -1 for the last.
        pysra.output.OutputLocation('outcrop', index=-1),
        # Surface (outcrop)
        pysra.output.OutputLocation('outcrop', index=0), ),
    # Input (outcrop).
    pysra.output.ResponseSpectrumOutput(
        osc_freqs,
        # The wave_field may also be directly specified and the location may
        #  be specified by the depth within the profile.
        pysra.output.OutputLocation(
            pysra.motion.WaveField.outcrop, depth=profile[-1].depth),
        0.05),
    # Surface (outcrop).
    pysra.output.ResponseSpectrumOutput(
        osc_freqs, pysra.output.OutputLocation('outcrop', index=0), 0.05), )

# Compute the response
calc = pysra.propagation.EquivalentLinearCalculation(strain_ratio=0.65)
calc(motion, profile, profile.location('outcrop', index=-1))

# Compute the outputs
outputs(calc)

# Transfer function
accel_tf = outputs[0]
fig, ax = plt.subplots()
ax.plot(accel_tf.freqs, accel_tf.values.real, '-', color='C0', label='Real')
ax.plot(
    accel_tf.freqs, accel_tf.values.imag, '-', color='C1', label='Imaginary')
ax.plot(
    accel_tf.freqs, abs(accel_tf.values), '-', color='C2', label='Absolute')
ax.set_xlabel('Frequency (Hz)')
ax.set_xscale('log')
ax.set_ylabel('Accel. Transfer Function')

ax.grid()
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(__file__.replace('.py', '-tf.png'), dpi=150)

# Response spectrum
ars_input = outputs[1]
ars_surface = outputs[2]
fig, ax = plt.subplots()
for name, ars, color in zip(['Input', 'Surface'], [ars_input, ars_surface],
                            ['C0', 'C1']):
    ax.plot(ars.freqs, ars.values, '-', color=color, label=name)
ax.set_xlabel('Frequency (Hz)')
ax.set_xscale('log')
ax.set_xlim(0.1, 100)
ax.set_ylabel('5%-Damped, Spectral Accel. (g)')
ax.set_yscale('log')

ax.grid()
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(__file__.replace('.py', '-ars.png'), dpi=150)
