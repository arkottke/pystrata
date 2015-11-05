"""Compute transfer functions for within and outcrop conditions."""

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('..')
sys.path.append('../../pyrvt')
import pysra

count = 10

motion = pysra.motion.Motion(np.logspace(-1, 2, 301))

profile = pysra.site.Profile([
        pysra.site.Layer(
            pysra.site.SoilType(
                'Soil-1', 18., None, 0.05
            ),
            30, 400
        ),
        pysra.site.Layer(
            pysra.site.SoilType(
                'Soil-2', 19., None, 0.05
            ),
            20, 600
        ),
        pysra.site.Layer(
            pysra.site.SoilType(
                'Rock', 24., None, 0.01
            ),
            0, 1200
        ),
    ])

profile.update_depths()

toro_thickness = pysra.variation.ToroThicknessVariation()
toro_velocity = pysra.variation.ToroVelocityVariation.generic_model('USGS B')

# Create realizations of the profile with varied thickness
varied_thick = [toro_thickness(profile) for _ in range(count)]

# For each realization of varied thickness, vary the shear-wave velocity
varied_vel = [toro_velocity(rt) for rt in varied_thick]

fig, ax = plt.subplots()

for p in varied_vel:
    ax.plot(
        [l.initial_shear_vel for l in p], [l.depth for l in p],
        drawstyle='steps-pre')

ax.set_xlabel('$V_s$ (m/s)')
ax.set_xscale('log')

ax.set_ylabel('Depth (m)')
ax.set_ylim(55, 0)

ax.grid()

fig.tight_layout()

fig.savefig('example02')
