"""Compute transfer functions for within and outcrop conditions."""

import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('../../pyrvt')
import pysra

__author__ = 'albert'

count = 10
GRAVITY = 9.81

motion = pysra.base.motion.Motion(np.logspace(-1, 2, 301))

profile = pysra.base.site.Profile([
        pysra.base.site.Layer(
            pysra.base.site.SoilType(
                'Soil-1', 18., GRAVITY, None, 0.05
            ),
            30, 400
        ),
        pysra.base.site.Layer(
            pysra.base.site.SoilType(
                'Soil-2', 19., GRAVITY, None, 0.05
            ),
            20, 600
        ),
        pysra.base.site.Layer(
            pysra.base.site.SoilType(
                'Rock', 24., GRAVITY, None, 0.01
            ),
            0, 1200
        ),
    ])

profile.update_depths()

toro_thickness = pysra.base.variation.ToroThicknessVariation()
toro_velocity = pysra.base.variation.ToroVelocityVariation.generic_model('USGS B')

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
ax.set_ylim(35, 0)

ax.grid()

fig.tight_layout()

fig.savefig('example02')
