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

profile.update_layers()

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
fig.savefig(__file__.replace('.py', '.png'), dpi=150)
