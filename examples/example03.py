"""Compute transfer functions for within and outcrop conditions."""

import matplotlib.pyplot as plt
import numpy as np

import pysra

motion = pysra.motion.SourceTheoryRvtMotion(6.5, 20, 'wna')
motion.calc_fourier_amps()
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
calc(motion, profile.data, loc_bedrock)

osc_damping = 0.05
osc_freqs = np.logspace(-1, 2, 181)
ars_input = motion.calc_osc_accels(osc_freqs, osc_damping)
ars_surface = motion.calc_osc_accels(
    osc_freqs, osc_damping, calc.calc_accel_tf(loc_bedrock, loc_surface))

fig, ax = plt.subplots()

for name, value, color in zip(['Input', 'Surface'],
                              [ars_input, ars_surface],
                              ['blue', 'red']):
    ax.plot(osc_freqs, value, '-', color=color, label=name)

ax.set_xlabel('Frequency (Hz)')
ax.set_xscale('log')
ax.set_xlim(0.1, 100)

ax.set_ylabel('5%-Damped, Spectral Accel. (g)')
ax.set_yscale('log')

ax.grid()
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(__file__.replace('.py', '.png'), dpi=150)
