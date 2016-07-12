"""Compute transfer functions for within and outcrop conditions."""

import collections
import sys

import matplotlib.pyplot as plt
import numpy as np

import pysra

motion = pysra.motion.SourceTheoryRvtMotion(6.5, 20, 'wna')

pysra.site.DarendeliNonlinearProperty(
    0, 1, 0.50, param='mod_reduc')

profile = pysra.site.Profile([
    pysra.site.Layer(
        pysra.site.SoilType(
            'Soil', 18.,
            pysra.site.DarendeliNonlinearProperty(
                0, 1, 0.50, param='mod_reduc'),
            pysra.site.DarendeliNonlinearProperty(
                0, 1, 0.50, param='damping')
        ),
        30, 400
    ),
    pysra.site.Layer(
        pysra.site.SoilType(
            'Rock', 24., None, 0.01
        ),
        0, 1200
    ),
])

loc_surface = profile.location('outcrop', index=0)
loc_bedrock = profile.location('outcrop', index=-1)

calc = pysra.propagation.EquivalentLinearCalculation(strain_ratio=0.65)
calc.calc_waves(motion, profile.data, loc_bedrock)

osc_damping = 0.05
osc_freqs = np.logspace(-1, 2, 181)
ars_surface = motion.compute_osc_accels(osc_freqs, osc_damping)
trans_funcs.append(calc.calc_accel_tf(bedrock, surface))

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
