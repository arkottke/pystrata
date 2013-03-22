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
# Copyright (C) Albert Kottke, 2013

import numpy as np

from scipy.interpolate import interp1d


class NonlinearProperty(object):
    """Docstring for NonlinearProperty """

    def __init__(self, name='', strains=[], values=[]):
        """Class for nonlinear property with a method for log-linear
        interpolation.

        Parameters
        ----------
        name : str, optional
            used for identification
        strains : float iterable
            strains for each of the values
        values : float iterable
            value of the property corresponding to each strain
        """
        self.name = name
        self._strains = strains
        self._values = values

        self._update()

    def __call__(self, strain):
        """Return the nonlinear property at a specific strain.

        If the strain is within the range of the provided strains, then
        log-linear interpolation is calculate the value at the requested
        strain.  If the strain falls outside the provided range then the value
        corresponding to the smallest or largest value is returned.

        Parameters
        ----------
        strain : float

        Returns
        -------
        The nonlinear property at the requested strain.
        """

        if strain < self.strains[0]:
            value = self.values[0]
        elif strain > self.strains[-1]:
            value = self.values[-1]
        else:
            value = self._interpolater(np.log(strain))

        return value

    @property
    def strains(self):
        """Strains"""
        return self._strains

    @strains.setter
    def strains(self, strains):
        self._strains = strains
        self._update()

    @property
    def values(self):
        """Values"""
        return self._values

    @values.setter
    def values(self, values):
        self._values = values
        self._update()

    def _update(self):
        """Initialize the 1D interpolation."""
        if self.strains and self.values:
            self._interpolater = interp1d(np.log(self.strains), self.values)


class SoilType(object):
    """Docstring for SoilType """

    def __init__(self, name='', unit_wt=0., gravity=0., mod_reduc=None,
                 damping=None):
        """Soil Type

        Parameters:
        -----------

        name : str, optional
            used for identification
        unit_wt : float
            unit weight of the material in [kN/m3] or [lbf/ft3]
        gravity : float
            gravity in [m/s2] or [ft/s2]
        mod_reduc : NonlinearProperty or None
            shear-modulus reduction. If None, then no reduction is applied
        damping : NonlinearProperty or float
            damping ratio
        """

        self.name = name
        self.unit_wt = unit_wt
        self.gravity = gravity
        self.mod_reduc = mod_reduc
        self.damping = damping


class IterativeValue(object):
    def __init__(self, value):
        self._value = value
        self._previous = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._previous = self._value
        self._value = value

    @property
    def previous(self):
        return self._previous

    def relative_error(self):
        """The relative error, in percent, between the two iterations."""
        err = None
        if self.previous:
            err = 100. * (self._previous - self._value) / self._value

        return err


class Layer(object):
    """Docstring for Layer """

    def __init__(self, soil_type, thickness, velocity):
        """@todo: to be defined1 """

        self.soil_type = soil_type
        self.thickness = thickness
        self.velocity = velocity

        self._shear_mod = IterativeValue(None)
        self._damping = IterativeValue(None)
        self._strain = IterativeValue(None)

    @property
    def damping(self):
        return self._damping

    def max_shear_mod(self):
        return (self.soil_type.unit_wt * self.velocity ** 2
                / self.soil_type.gravity)

    @property
    def shear_mod(self):
        return self._shear_mod

    @property
    def strain(self):
        return self._strain

    @strain.setter
    def strain(self, strain):
        self._strain.value = strain

        # Update the shear modulus
        max_shear_mod = self.max_shear_mod()
        try:
            self._shear_mod.value = (max_shear_mod *
                                     self.soil_type.mod_reduc(strain))
        except TypeError:
            self._shear_mod.value = max_shear_mod

        # Update the damping
        try:
            self._damping.value = self.soil_type.damping(strain)
        except TypeError:
            self._damping.value = self.soil_type.damping


class Profile(object):
    """Docstring for Profile """

    def __init__(self):
        """@todo: to be defined1 """

        self.layers = []

    def auto_discretize(self):
        pass
