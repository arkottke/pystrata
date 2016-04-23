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
# Copyright (C) Albert Kottke, 2013-2015

import os

import numpy as np

import pysra

soil = pysra.site.SoilType('soil', 17, None, 0.05)
rock = pysra.site.SoilType('rock', 23, None, 0.01)
layers = [
    pysra.site.Layer(soil, 40, 400),
    pysra.site.Layer(rock, 0, 1500),
]
profile = pysra.site.Profile(layers)

motion = pysra.motion.SourceTheoryRvtMotion(6, 20, 'wna')
motion.calc_fourier_amps()

loc_in = profile.location('outcrop', index=-1)
loc_out = profile.location('outcrop', index=0)

calculator = pysra.propagation.LinearElasticCalculator()
calculator(motion, profile, loc_in)

trans_func = calculator.calc_accel_tf(loc_in, loc_out)

if os.environ.get('TRAVIS', False) is False:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(xscale='log'))
    ax.plot(motion.freqs, np.abs(trans_func))

    fig.tight_layout()
    print(__file__)
    fig.savefig('test_analysis.png')

assert np.all(np.isfinite(trans_func))
