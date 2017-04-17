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
"""Use Dask for parallel calculations."""

from dask import delayed

import matplotlib.pyplot as plt

<<<<<<< HEAD
=======
import pysra

>>>>>>> 4d71de0126fd2979338192782a55642ad41b2c46
profile = pysra.site.Profile([
    pysra.site.Layer(
        pysra.site.SoilType('Soil-1', 18.,
                            pysra.site.DarendeliNonlinearProperty(
                                0, 1, 0.50, param='mod_reduc'),
                            pysra.site.DarendeliNonlinearProperty(
                                0, 1, 0.50, param='damping')), 30, 400),
    pysra.site.Layer(
        pysra.site.SoilType('Soil-2', 19.,
                            pysra.site.DarendeliNonlinearProperty(
                                0, 1, 2., param='mod_reduc'),
                            pysra.site.DarendeliNonlinearProperty(
                                0, 1, 2., param='damping')), 20, 600),
    pysra.site.Layer(pysra.site.SoilType('Rock', 24., None, 0.01), 0, 1200),
])
profile.update_layers()

var_thickness = pysra.variation.ToroThicknessVariation()
var_velocity = pysra.variation.ToroVelocityVariation.generic_model('USGS C')
var_nlcurves = pysra.variation.SpidVariation(
    -0.5, std_mod_reduc=0.15, std_damping=0.30)

varied = delayed(
    pysra.variation.iter_varied_profiles(
        profile,
        1,
        var_thickness=var_thickness,
        var_velocity=var_velocity,
        var_nlcurves=var_nlcurves))

fig, axarr = plt.subplots(
    2, 2, sharex=True, sharey='row', subplot_kw={'xscale': 'log'})

for i, (variation, name) in enumerate(
        zip([
            pysra.variation.DarendeliVariation(correlation),
            pysra.variation.SpidVariation(correlation)
        ], ['Darendeli (2001)', 'EPRI SPID (2014)'])):
    realizations = [variation(soil_type) for _ in range(n)]
    for j, prop in enumerate(['mod_reduc', 'damping']):
        axarr[j, i].plot(
            getattr(soil_type, prop).strains,
            np.transpose([getattr(r, prop).values for r in realizations]),
            'b-',
            linewidth=0.5,
            alpha=0.8)
        if j == 0:
            axarr[j, i].set_title(name)
