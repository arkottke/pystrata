"""Randomize nonlinear properties."""

import matplotlib.pyplot as plt
import numpy as np

import pysra

soil_type = pysra.site.DarendeliSoilType(
    'Soil', 18., plas_index=0, ocr=1, mean_stress=0.5)
n = 30
correlation = 0

fig, axarr = plt.subplots(2, 2, sharex=True, sharey='row',
                          subplot_kw={'xscale': 'log'})

for i, (variation, name) in enumerate(zip(
        [pysra.variation.DarendeliVariation(correlation),
         pysra.variation.SpidVariation(correlation)],
        ['Darendeli (2001)', 'EPRI SPID (2014)'])):
    realizations = [variation(soil_type) for _ in range(n)]
    for j, prop in enumerate(['mod_reduc', 'damping']):
        axarr[j, i].plot(
            getattr(soil_type, prop).strains,
            np.transpose([getattr(r, prop).values for r in realizations]),
            'b-', linewidth=0.5, alpha=0.8
        )
        if j == 0:
            axarr[j, i].set_title(name)

axarr[0, 0].set_ylabel('$G/G_{max}$')
axarr[1, 0].set_ylabel('$D$ (%)')
plt.setp(axarr[1, :], xlabel='Strain, $\gamma$ (%)')

fig.tight_layout()
fig.savefig(__file__.replace('.py', '.png'), dpi=150)
