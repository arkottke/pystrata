"""Use Geopsy output format to define the velocity profile"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np

import pysra


def iter_geopsy_profiles(fname):
    """Read a Geopsy formatted text file created by gpdcreport."""
    with open(fname) as fp:
        next(fp)
        while True:
            try:
                line = next(fp)
            except StopIteration:
                break

            m = re.search(r'Layered model (\d+): value=([0-9.]+)', line)
            id, score = m.groups()
            count = int(next(fp))
            d = {
                'id': id,
                'score': score,
                'layers': [],
            }
            cols = ['thickness', 'vel_comp', 'vel_shear', 'density']
            for _ in range(count):
                values = [float(p) for p in next(fp).split()]
                d['layers'].append(dict(zip(cols, values)))

            yield d


motion = pysra.motion.SourceTheoryRvtMotion(6.5, 20, 'wna')
motion.calc_fourier_amps()

calc = pysra.propagation.EquivalentLinearCalculation(strain_ratio=0.65)

site_amp = pysra.output.ResponseSpectrumRatioOutput(
    np.logspace(-1, 2, 181),
    pysra.output.OutputLocation('outcrop', index=-1),
    pysra.output.OutputLocation('outcrop', index=0),
    0.05
)

fname_profiles = os.path.join(
    os.path.dirname(__file__), 'data', 'best100_GM_linux.txt')
site_ampls = []
for geopsy_profile in iter_geopsy_profiles(fname_profiles):
    profile = pysra.site.Profile([
        pysra.site.Layer(
            pysra.site.SoilType(
                'soil-%d' % i, l['density'] / pysra.GRAVITY, damping=0.05),
            l['thickness'], l['vel_shear']
        ) for i, l in enumerate(geopsy_profile['layers'])
    ])
    # Use 1% damping for the half-space
    profile[-1].soil_type.damping = 0.01
    # Compute the waves from the last layer
    calc(motion, profile, profile.location('outcrop', index=-1))
    # Compute the site amplification
    site_amp(calc)

fig, ax = plt.subplots()
ax.plot(site_amp.freqs, site_amp.values, 'b-', alpha=0.6)

ax.set_xlabel('Frequency (Hz)')
ax.set_xscale('log')
ax.set_xlim(0.1, 100)

ax.set_ylabel('5%-Damped, Spectral Amplification (Surface/Input)')
ax.set_yscale('log')

ax.grid()
fig.tight_layout()
fig.savefig(__file__.replace('.py', '.png'), dpi=150)
