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

osc_damping = 0.05
osc_freqs = np.logspace(-1, 2, 181)

ars_input = motion.calc_osc_accels(osc_freqs, osc_damping)

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
    # Find the locations of the surface and bedrock
    loc_surface = profile.location('outcrop', index=0)
    loc_bedrock = profile.location('outcrop', index=-1)
    # Compute the waves
    calc(motion, profile.data, loc_bedrock)
    # Compute the site amplification
    ars_surface = motion.calc_osc_accels(
        osc_freqs, osc_damping, calc.calc_accel_tf(loc_bedrock, loc_surface))

    site_ampls.append(ars_surface / ars_input)

fig, ax = plt.subplots()
ax.plot(osc_freqs, np.transpose(site_ampls), 'b-', alpha=0.6)

ax.set_xlabel('Frequency (Hz)')
ax.set_xscale('log')
ax.set_xlim(0.1, 100)

ax.set_ylabel('5%-Damped, Spectral Amplification (Surface/Input)')
ax.set_yscale('log')

ax.grid()
fig.tight_layout()
fig.savefig(__file__.replace('.py', '.png'), dpi=150)