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

import collections

from typing import Iterable, Optional, Union

import numpy as np

from scipy.interpolate import interp1d


class NonlinearProperty(object):
    """Docstring for NonlinearProperty """

    def __init__(self, name='', strains=None, values=None):
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
        self._strains = strains or list()
        self._values = values or list()

        self._update()

    def __call__(self, strain: float):
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

    def __init__(self, name: str = '', unit_wt: float = 0., gravity: float = 0.,
                 mod_reduc: Optional[NonlinearProperty] = None,
                 damping: Union[NonlinearProperty, float] = None):
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
            shear-modulus reduction curves. If None, linear behavior with no reduction is used
        damping : NonlinearProperty or float
            damping ratio. If float, then linear behavior with constant damping is used.
        """

        self.name = name
        self.unit_wt = unit_wt
        self.gravity = gravity
        self.mod_reduc = mod_reduc
        self.damping = damping

    @property
    def density(self) -> float:
        return self.unit_wt / self.gravity

    @property
    def damping_min(self) -> float:
        """Return the small-strain damping."""
        try:
            return self.damping.values[0]
        except AttributeError:
            return self.damping


# TODO for nonlinear site response this class wouldn't be used. Better way to do this? Maybe have the calculator create it?
class IterativeValue(object):
    def __init__(self, value: Optional[float]):
        self._value = value
        self._previous = None

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float):
        self._previous = self._value
        self._value = value

    @property
    def previous(self) -> float:
        return self._previous

    @property
    def relative_error(self) -> float:
        """The relative error, in percent, between the two iterations.
        """
        if self.previous:
            err = 100. * (self.previous - self.value) / self.value
        else:
            err = None

        return err


class Layer(object):
    """Docstring for Layer """

    def __init__(self, soil_type: SoilType, thickness: float, shear_vel: float):
        """@todo: to be defined1 """
        self._profile = None

        self._soil_type = soil_type
        self._thickness = thickness
        self._initial_shear_vel = shear_vel

        self._shear_mod = IterativeValue(self.initial_shear_mod)
        self._damping = IterativeValue(None)
        self._strain = IterativeValue(None)

        self._depth = 0

    @property
    def depth(self) -> float:
        return self._depth

    @property
    def depth_mid(self) -> float:
        return self._depth + self._thickness / 2

    @property
    def depth_base(self) -> float:
        return self._depth + self._thickness

    @classmethod
    def duplicate(cls, other: 'Layer') -> 'Layer':
        return cls(other.soil_type, other.thickness, other.shear_vel)

    @property
    def density(self) -> float:
        return self.soil_type.density

    @property
    def damping(self) -> IterativeValue:
        return self._damping

    @property
    def initial_shear_mod(self) -> float:
        return self.density * self.initial_shear_vel ** 2

    @property
    def initial_shear_vel(self) -> float:
        return self._initial_shear_vel

    @property
    def comp_shear_mod(self) -> complex:
        '''Complex shear modulus from Kramer (1996).'''
        return self.shear_mod.value * (1 - self.damping.value ** 2 +
                                       2j * self.damping.value)

    @property
    def comp_shear_vel(self) -> complex:
        return np.sqrt(self.comp_shear_mod / self.density)

    @property
    def shear_mod(self) -> IterativeValue:
        return self._shear_mod

    @property
    def shear_vel(self) -> float:
        return np.sqrt(self.shear_mod.value / self.density)

    @property
    def strain(self) -> Union[IterativeValue, float]:
        # FIXME simplify so that a float is always returned?
        return self._strain

    @property
    def soil_type(self) -> SoilType:
        return self._soil_type

    @property
    def thickness(self) -> float:
        return self._thickness

    @thickness.setter
    def thickness(self, thickness: float):
        self._thickness = thickness
        self._profile.update_depths(self, self._profile.index(self) + 1)

    @strain.setter
    def strain(self, strain: float):
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

    @property
    def incr_site_atten(self) -> float:
        return ((2 * self.soil_type.damping_min * self._thickness) /
                self.initial_shear_vel)


class Location(object):
    def __init__(self, layer: Layer, index: int, depth_within: float, wave_field: str):
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

    def __repr__(self):
        return '<Location(layer_index={_index}, depth_within={_depth_within}, wave_field={_wave_field})>'.format(
            **self.__dict__
        )


class Profile(collections.UserList):
    """Docstring for Profile """

    def __init__(self, layers: Iterable(Layer) = None):
        collections.UserList.__init__(self, layers)

    def update_depths(self, start_layer: int = 0):
        if start_layer < 1:
            depth = 0
        else:
            depth = self[start_layer - 1].depth_base

        for l in self[start_layer:]:
            l._depth = depth
            if l != self[-1]:
                depth = l.depth_base

    def auto_discretize(self) -> Iterable(Layer):
        raise NotImplementedError

    def location(self, depth: float, wave_field: str) -> Location:
        for i, l in enumerate(self[:-1]):
            if l.depth <= depth < l.depth_base:
                break
        else:
            # Bedrock
            i = len(self) - 1
            l = self[-1]

        return Location(l, i, depth - l.depth, wave_field)

    def calc_site_attenuation(self) -> float:
        return sum(l.incr_site_atten for l in self._layers)
