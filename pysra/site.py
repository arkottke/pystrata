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

import collections

import numpy as np

from scipy.interpolate import interp1d

from .motion import WaveField, GRAVITY


class NonlinearProperty(object):
    """Class for nonlinear property with a method for log-linear interpolation.

    Parameters
    ----------
    name: str, optional
        used for identification
    strains: :class:`numpy.ndarray`, optional
        strains for each of the values [decimal].
    values: :class:`numpy.ndarray`, optional
        value of the property corresponding to each strain. Damping should be
        specified in decimal, e.g., 0.05 for 5%.
    param: str, optional
        type of parameter. Possible values are:

            mod_reduc
                Shear-modulus reduction curve

            damping
                Damping ratio curve
    """

    PARAMS = ['mod_reduc', 'damping']

    def __init__(self, name='', strains=None, values=None, param=None):
        self.name = name
        self._strains = np.asarray(strains).astype(float)
        self._values = np.asarray(values).astype(float)

        self._interpolater = None

        self._param = None
        self.param = param

        self._update()

    def __call__(self, strain):
        """Return the nonlinear property at a specific strain.

        If the strain is within the range of the provided strains, then the
        value is interpolated in log-space is calculate the value at the
        requested strain.  If the strain falls outside the provided range
        then the value corresponding to the smallest or largest value is
        returned.

        The interpolation is performed using either a cubic-spline, if enough
        points are provided, or using linear interpolation.

        Parameters
        ----------
        strain: float
            Shear strain of interest [decimal].

        Returns
        -------
        float
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
        """Strains [decimal]."""
        return self._strains

    @strains.setter
    def strains(self, strains):
        self._strains = np.asarray(strains).astype(float)
        self._update()

    @property
    def values(self):
        """Values of either shear-modulus reduction or damping ratio."""
        return self._values

    @values.setter
    def values(self, values):
        self._values = np.asarray(values).astype(float)
        self._update()

    @property
    def param(self):
        """Nonlinear parameter name."""
        return self._param

    @param.setter
    def param(self, value):
        if value:
            assert value in self.PARAMS
        self._param = value

    def _update(self):
        """Initialize the 1D interpolation."""

        if self.strains.size and self.strains.size == self.values.size:
            x = np.log(self.strains)
            y = self.values

            if x.size < 4:
                self._interpolater = interp1d(x, y, 'linear')
            else:
                self._interpolater = interp1d(x, y, 'cubic')


class DarendeliNonlinearProperty(NonlinearProperty):
    def __init__(self, plas_index, ocr, mean_stress, freq=1, num_cycles=10,
                 strains=np.logspace(-4, 0.5, num=20), param='mod_reduc'):
        strains = np.asarray(strains)

        name = "Darendeli (PI={:.0f}, OCR={:.1f}, s'_v={:.1f} atm)".format(
            plas_index, ocr, mean_stress)
        # Compute the reference strain based on the PI, OCR, and mean stress
        strain_ref = ((0.0352 + 0.0010 * plas_index * ocr ** 0.3246) *
                      mean_stress ** 0.3483)
        curvature = 0.9190
        mod_reduc = 1 / (1 + (strains / strain_ref) ** curvature)

        if param == 'damping':
            # Empirical minimum damping
            damping_min = ((0.8005 + 0.0129 * plas_index * ocr ** -0.1069) *
                           mean_stress ** -0.2889 *
                           (1 + 0.2919 * np.log(freq)))
            # Masing damping based on shear-modulus reduction
            damping_masing_a1 = (
                (100./np.pi) *
                (4 * (strains - strain_ref *
                      np.log((strains + strain_ref) / strain_ref)) /
                 (strains ** 2 / (strains + strain_ref)) - 2.)
            )
            # Correction between perfect hyperbolic strain model and modified
            # model.
            c1 = -1.1143 * curvature ** 2 + 1.8618 * curvature + 0.2523
            c2 = 0.0805 * curvature ** 2 - 0.0710 * curvature - 0.0095
            c3 = -0.0005 * curvature ** 2 + 0.0002 * curvature + 0.0003
            damping_masing = (c1 * damping_masing_a1 +
                              c2 * damping_masing_a1 ** 2 +
                              c3 * damping_masing_a1 ** 3)
            # Masing correction factor
            masing_corr = 0.6329 - 0.00566 * np.log(num_cycles)
            # Compute the damping in percent
            damping = (damping_min +
                       damping_masing * masing_corr * mod_reduc ** 0.1)
            # Convert to decimal values
            values = damping / 100.
        else:
            values = mod_reduc

        super(DarendeliNonlinearProperty, self).__init__(
                name, strains, values, param
        )


class SoilType(object):
    """Soiltype that combines nonlinear behavior and material properties.

    Parameters
    ----------
    name: str, optional
        used for identification
    unit_wt  float
        unit weight of the material in [kN/m³]
    mod_reduc: :class:`NonlinearProperty` or None
        shear-modulus reduction curves. If None, linear behavior with no
        reduction is used
    damping: :class:`NonlinearProperty` or float
        damping ratio. If float, then linear behavior with constant damping
        is used.
    """

    def __init__(self, name='', unit_wt=0., mod_reduc=None, damping=None):
        self.name = name
        self._unit_wt = unit_wt
        self.mod_reduc = mod_reduc
        self.damping = damping

    @property
    def unit_wt(self):
        return self._unit_wt

    @property
    def density(self):
        """Density of the soil in kg/m³."""
        return self.unit_wt / GRAVITY

    @property
    def damping_min(self):
        """Return the small-strain damping."""
        try:
            return self.damping.values[0]
        except AttributeError:
            return self.damping

    @property
    def is_nonlinear(self):
        """If nonlinear properties are specified."""
        return any(isinstance(p, NonlinearProperty)
                   for p in [self.mod_reduc, self.damping])

    def __eq__(self, other):
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ['name', 'unit_wt', 'mod_reduc', 'damping'])


# TODO: for nonlinear site response this class wouldn't be used. Better way
# to do this? Maybe have the calculator create it?
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

    @property
    def relative_error(self):
        """The relative error, in percent, between the two iterations.
        """
        if self.previous:
            err = 100. * (self.previous - self.value) / self.value
        else:
            err = 0
        return err


class Layer(object):
    """Docstring for Layer """

    def __init__(self, soil_type, thickness, shear_vel):
        """@todo: to be defined1 """
        self._profile = None

        self._soil_type = soil_type
        self._thickness = thickness
        self._initial_shear_vel = shear_vel

        self._shear_mod = IterativeValue(self.initial_shear_mod)
        self._damping = IterativeValue(self.soil_type.damping_min)
        self._strain = IterativeValue(None)

        self._depth = 0
        self._vert_stress = 0

    @property
    def depth(self):
        """Depth to the top of the layer [m]."""
        return self._depth

    @property
    def depth_mid(self):
        """Depth to the middle of the layer [m]."""
        return self._depth + self._thickness / 2

    @property
    def depth_base(self):
        """Depth to the base of the layer [m]."""
        return self._depth + self._thickness

    @classmethod
    def duplicate(cls, other):
        """Create a copy of the layer."""
        return cls(other.soil_type, other.thickness, other.shear_vel)

    @property
    def density(self):
        """Density of soil in [kg/m³]."""
        return self.soil_type.density

    @property
    def damping(self):
        """Strain-compatible damping."""
        return self._damping

    @property
    def initial_shear_mod(self):
        """Initial (small-strain) shear modulus from Kramer (1996) [kN/m²]."""
        return self.density * self.initial_shear_vel ** 2

    @property
    def initial_shear_vel(self):
        """Initial (small-strain) shear-wave velocity [m/s]."""
        return self._initial_shear_vel

    @property
    def comp_shear_mod(self):
        """Strain-compatible complex shear modulus [kN/m²].

        Frequency independent formulation."""
        return self.shear_mod.value * (1 + 2j * self.damping.value)

    @property
    def comp_shear_vel(self):
        """Strain-compatible complex shear-wave velocity [m/s]."""
        return np.sqrt(self.comp_shear_mod / self.density)

    @property
    def max_error(self):
        return max(
            self.shear_mod.relative_error,
            self.damping.relative_error,
        )

    @property
    def shear_mod(self):
        """Strain-compatible shear modulus [kN//m²]."""
        return self._shear_mod

    @property
    def shear_vel(self):
        """Strain-compatible shear-wave velocity [m/s]."""
        return np.sqrt(self.shear_mod.value / self.density)

    @property
    def strain(self):
        # FIXME simplify so that a float is always returned?
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
        self._profile.update_layers(self, self._profile.index(self) + 1)

    @property
    def unit_wt(self):
        return self.soil_type.unit_wt

    def vert_stress(self, depth_within=0, effective=False):
        """Vertical stress from the top of the layer [kN//m²]."""
        assert depth_within <= self.thickness
        vert_stress = self._vert_stress + depth_within * self.unit_wt
        if effective:
            pore_pressure = self._profile.pore_pressure(
                self.depth + depth_within)
            vert_stress -= pore_pressure
        return vert_stress

    @strain.setter
    def strain(self, strain):
        if self.soil_type.is_nonlinear:
            self._strain.value = strain
        else:
            self._strain = strain

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
    def incr_site_atten(self):
        return ((2 * self.soil_type.damping_min * self._thickness) /
                self.initial_shear_vel)


class Location(object):
    """Location within a profile"""
    def __init__(self, index, layer, wave_field, depth_within=0):
        self._index = index
        self._layer = layer
        self._depth_within = depth_within

        if not isinstance(wave_field, WaveField):
            wave_field = WaveField[wave_field]
        self._wave_field = wave_field

    @property
    def depth_within(self):
        return self._depth_within

    @property
    def layer(self):
        return self._layer

    @property
    def index(self):
        return self._index

    @property
    def wave_field(self):
        return self._wave_field

    def vert_stress(self, effective=False):
        return self._layer.vert_stress(self.depth_within, effective=effective)

    def __repr__(self):
        return (
            '<Location(layer_index={_index}, depth_within={_depth_within} '
            'wave_field={_wave_field})>'.
            format(**self.__dict__)
        )


class Profile(collections.UserList):
    """Docstring for Profile """

    def __init__(self, layers=None, wt_depth=0):
        collections.UserList.__init__(self, layers)
        self.wt_depth = wt_depth
        if layers:
            self.update_layers()

    def update_layers(self, start_layer=0):
        if start_layer < 1:
            depth = 0
            vert_stress = 0
        else:
            ref_layer = self[start_layer - 1]
            depth = ref_layer.depth_base
            vert_stress = ref_layer.vert_stress(
                ref_layer.thickness, effective=False)

        for l in self[start_layer:]:
            l._profile = self
            l._depth = depth
            l._vert_stress = vert_stress
            if l != self[-1]:
                # Use the layer to compute the values at the base of the
                # layer, and apply them at the top of the next layer
                depth = l.depth_base
                vert_stress = l.vert_stress(l.thickness, effective=False)

    def iter_soil_types(self):
        yielded = set()
        for l in self:
            if l.soil_type in yielded:
                continue
            else:
                yielded.add(l)
                yield l.soil_type

    def auto_discretize(self):
        raise NotImplementedError

    def pore_pressure(self, depth):
        """Pore pressure at a given depth in [kN//m²].

        Parameters
        ----------
        depth

        Returns
        -------

        """
        return GRAVITY * max(depth - self.wt_depth, 0)

    def calc_site_attenuation(self):
        return sum(l.incr_site_atten for l in self)

    def location(self, wave_field, depth=None, index=None):
        """Create a Location for a specific depth.

        Parameters
        ----------
        wave_field: str
            Wave field. See :class:`Location` for possible values.
        depth: float, optional
            Depth corresponding to the :class`Location` of interest. If
            provided, then index is ignored.
        index: int, optional
            Index corresponding to layer of interest in :class:`Profile`. If
             provided, then depth is ignored and location is provided a top
             of layer.

        Returns
        -------
        Location
            Corresponding :class:`Location` object.
        """
        if not isinstance(wave_field, WaveField):
            wave_field = WaveField[wave_field]

        if index is None and depth is not None:
            for i, l in enumerate(self[:-1]):
                if l.depth <= depth < l.depth_base:
                    depth_within = depth - l.depth
                    break
            else:
                # Bedrock
                i = len(self) - 1
                l = self[-1]
                depth_within = 0
        elif index is not None and depth is None:
            l = self[index]
            i = self.index(l)
            depth_within = 0
        else:
            raise NotImplementedError

        return Location(i, l, wave_field, depth_within)
