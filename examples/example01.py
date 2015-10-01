"""Compute transfer functions for within and outcrop conditions."""

import sys

import collections
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('../../pyrvt')
import pysra

__author__ = 'albert'

GRAVITY = 9.81

motion = pysra.base.motion.Motion(np.logspace(-1, 2, 301))

profiles = [
    # Initial
    pysra.base.site.Profile([
        pysra.base.site.Layer(
            pysra.base.site.SoilType(
                'Soil', 18., GRAVITY, None, 0.05
            ),
            30, 400
        ),
        pysra.base.site.Layer(
            pysra.base.site.SoilType(
                'Rock', 24., GRAVITY, None, 0.01
            ),
            0, 1200
        ),
    ]),
    # Reduced properties
    pysra.base.site.Profile([
        pysra.base.site.Layer(
            pysra.base.site.SoilType(
                'Soil', 18., GRAVITY, None, 0.08
            ),
            30, 300
        ),
        pysra.base.site.Layer(
            pysra.base.site.SoilType(
                'Rock', 24., GRAVITY, None, 0.01
            ),
            0, 1200
        ),
    ])
]

wave_fields = ['outcrop', 'within']

calc = pysra.base.propagation.LinearElasticCalculator()

rsrs = collections.OrderedDict()
for wave_field in wave_fields:
    trans_funcs = []
    for p in profiles:
        calc.calc_waves(motion, p.data)
        surface = p.location(0, 'outcrop')
        bedrock = p.location(p[0].thickness, wave_field)
        trans_funcs.append(
            calc.calc_accel_tf(bedrock, surface)
        )

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
