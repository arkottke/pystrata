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

from typing import Iterable

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

    @property
    def density(self):
        return self.unit_wt / self.gravity

# TODO for nonlinear site response this class wouldn't be used. Better way to do this? Maybe have the calculator create it?
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

    def __init__(self, soil_type: SoilType, thickness: float, shear_vel: float):
        """@todo: to be defined1 """
        self._profile = None

        self._soil_type = soil_type
        self._thickness = thickness
        self._initial_shear_vel = shear_vel

        self._shear_mod = IterativeValue(None)
        self._damping = IterativeValue(None)
        self._strain = IterativeValue(None)

        self._depth = 0

    @property
    def depth(self):
        return self._depth

    @property
    def depth_mid(self):
        return self._depth + self._thickness / 2

    @property
    def depth_base(self):
        return self._depth + self._thickness

    @classmethod
    def duplicate(cls, other):
        return cls(other.soil_type, other.thickness, other.shear_vel)

    @property
    def density(self):
        return self.soil_type.density

    @property
    def damping(self):
        return self._damping

    @property
    def initial_shear_mod(self):
        return self.density * self.initial_shear_vel ** 2

    @property
    def initial_shear_vel(self):
        return self._initial_shear_vel

    @property
    def comp_shear_mod(self):
        '''Complex shear modulus from Kramer (1996).'''
        return self.shear_mod * (1 - self.damping.value ** 2 +
                                 2j * self.damping.value)

    @property
    def comp_shear_vel(self):
        return np.sqrt(self.comp_shear_mod / self.density)

    @property
    def shear_mod(self):
        return self._shear_mod.value

    @property
    def shear_vel(self):
        return np.sqrt(self.shear_mod / self.density)

    @property
    def strain(self):
        return self._strain

    @property
    def soil_type(self):
        return self._soil_type

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness
        self._profile.update_depths(self, self._profile.index(self) + 1)

    @strain.setter
    def strain(self, strain):
        self._strain.value = strain

        # Update the shear modulus and damping
        try:
            mod_reduc = self.soil_type.mod_reduc(strain)
        except TypeError:
            mod_reduc = 1.
        self._shear_mod.value = self.initial_shear_mod * mod_reduc

        try:
            self._damping.value = self.soil_type.damping(strain)
        except TypeError:
            # No iteration provided by damping
            self._damping.value = self.soil_type.damping


class Location(object):
    def __init__(self, layer: Layer, index: int, depth_within: float,
                 wave_field: str=''):
        self._layer = layer
        self._index = index
        self._depth_within = depth_within
        self._wave_field = wave_field

    @property
    def layer(self) -> Layer:
        return self._layer

    @property
    def index(self) -> int:
        return self._index

    @property
    def depth_within(self) -> float:
        return self._depth_within

    @property
    def wave_field(self) -> str:
        return self._wave_field

    @wave_field.setter
    def wave_field(self, wave_field: str):
        assert wave_field in ['within', 'outcrop', 'incoming_only']
        self._wave_field = wave_field

class Profile(object):
    """Docstring for Profile """

    def __init__(self, layers: Iterable(Layer)=[]):
        """@todo: to be defined1 """
        self._layers = []
        for l in layers:
            self.append_layer(l)

    def index(self, layer: Layer) -> int:
        return self._layers.index(layer)

    def update_depths(self, start_layer: int=0):
        if start_layer < 1:
            depth = 0
        else:
            depth = self._layers[start_layer - 1].depth_base

        for l in self._layers[start_layer:]:
            l._depth = depth
            if l != self._layers[-1]:
                depth = l.depth_base

    @property
    def layers(self):
        return self._layers

    def append_layer(self, layer: Layer):
        self.insert_layer(len(self._layers), layer)

    def insert_layer(self, index: int, layer: Layer):
        layer._profile = self
        self._layers.insert(index, layer)
        self.update_depths(index)

    def auto_discretize(self):
        raise NotImplementedError

    def location(self, depth: float, wave_field: str):
        for i, l in enumerate(self._layers[:-1]):
            if l.depth <= depth < l.depth_base:
                break
        else:
            # Bedrock
            i = len(self._layers) - 1
            l = self._layers[-1]

        return Location(l, i, depth - l.depth, wave_field)
