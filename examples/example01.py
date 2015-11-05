"""Compute transfer functions for within and outcrop conditions."""

import collections
import sys

import matplotlib.pyplot as plt
import numpy as np

import pysra

motion = pysra.motion.Motion(np.logspace(-1, 2, 301))

profiles = [
    # Initial
    pysra.site.Profile([
        pysra.site.Layer(
            pysra.site.SoilType(
                'Soil', 18., None, 0.05
            ),
            30, 400
        ),
        pysra.site.Layer(
            pysra.site.SoilType(
                'Rock', 24., None, 0.01
            ),
            0, 1200
        ),
    ]),
    # Reduced properties
    pysra.site.Profile([
        pysra.site.Layer(
            pysra.site.SoilType(
                'Soil', 18., None, 0.08
            ),
            30, 300
        ),
        pysra.site.Layer(
            pysra.site.SoilType(
                'Rock', 24., None, 0.01
            ),
            0, 1200
        ),
    ])
]

wave_fields = ['outcrop', 'within']

calc = pysra.propagation.LinearElasticCalculator()

rsrs = collections.OrderedDict()
for wave_field in wave_fields:
    trans_funcs = []
    for p in profiles:
        surface = p.location('outcrop', index=0)
        bedrock = p.location(wave_field, index=-1)

        calc.calc_waves(motion, p.data, bedrock)
        trans_funcs.append(calc.calc_accel_tf(bedrock, surface))

    rsrs[wave_field] = np.abs(trans_funcs[1]) / np.abs(trans_funcs[0])

fig, ax = plt.subplots()

for (label, rsr), color in zip(rsrs.items(), ['red', 'blue']):
    ax.plot(motion.freqs, rsr, '-', color=color, label=label)

ax.set_xlabel('Frequency (Hz)')
ax.set_xscale('log')
ax.set_xlim(0.1, 30)

ax.set_ylabel('$RSR_{NL-L}$')
ax.set_yscale('log')

ax.grid()

ax.legend(loc='upper left', title='Bedrock Wave Field')

fig.tight_layout()

fig.savefig('example01')
