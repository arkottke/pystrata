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

import numpy as np

from .site import Location
from .motion import WaveField, GRAVITY


class LinearElasticCalculator(object):
    def __init__(self):
        self._waves_a = np.array([])
        self._waves_b = np.array([])
        self._wave_nums = np.array([])

        self._loc_input = None
        self._motion = None
        self._profile = None

    @property
    def motion(self):
        return self._motion

    @property
    def profile(self):
        return self._profile

    @property
    def loc_input(self):
        return self._loc_input

    def __call__(self, motion, profile, loc_input):
        """Perform the wave propagation.

        Parameters
        ----------
        motion: :class:`~.base.motion.Motion`
            Input motion.

        profile: :class:`~.base.site.Profile`
            Site profile.

        loc_input: :class:`~.base.site.Location`
            Location of the input motion.
        """
        self._motion = motion
        self._profile = profile
        self._loc_input = loc_input

        # Set intial properties
        for l in profile:
            if l.strain is None:
                l.strain = 0.

        self._calc_waves(motion.angular_freqs, profile)

    def _calc_waves(self, angular_freqs, profile):
        """Compute the wave numbers and amplitudes (up- and down-going).

        Parameters
        ----------
        angular_freqs: :class:`numpy.ndarray`
            Angular frequency at which the waves are computed.

        profile: :class:`~.base.site.Profile`
            Site profile.
        """

        # Compute the complex wave numbers of the system
        wave_nums = np.empty((len(profile), len(angular_freqs)), np.complex)
        for i, l in enumerate(profile):
            wave_nums[i, :] = angular_freqs / l.comp_shear_vel

        # Compute the waves. In the top surface layer, the up-going and
        # down-going waves have an amplitude of 1 as they are completely
        # reflected at the surface.
        waves_a = np.ones_like(wave_nums, np.complex)
        waves_b = np.ones_like(wave_nums, np.complex)
        for i, l in enumerate(profile[:-1]):
            # Complex impedance
            cimped = ((wave_nums[i] * l.comp_shear_mod) /
                      (wave_nums[i + 1] * profile[i + 1].comp_shear_mod))

            # Complex term to simplify equations -- uses full layer height
            cterm = 1j * wave_nums[i, :] * l.thickness

            waves_a[i + 1, :] = (
                0.5 * waves_a[i] * (1 + cimped) * np.exp(cterm) +
                0.5 * waves_b[i] * (1 - cimped) * np.exp(-cterm)
            )
            waves_b[i + 1, :] = (
                0.5 * waves_a[i] * (1 - cimped) * np.exp(cterm) +
                0.5 * waves_b[i] * (1 + cimped) * np.exp(-cterm)
            )

        # fixme: Better way to handle this?
        # Set wave amplitudes to 1 at frequencies near 0
        mask = np.isclose(angular_freqs, 0)
        waves_a[-1, mask] = 1.
        waves_b[-1, mask] = 1.

        self._waves_a = waves_a
        self._waves_b = waves_b
        self._wave_nums = wave_nums

    def wave_at_location(self, l):
        """Compute the wave field at specific location.

        Parameters
        ----------
        l : site.Location
            :class:`site.Location` of the input

        Returns
        -------
        `np.ndarray`
            Amplitude and phase of waves
        """
        cterm = 1j * self._wave_nums[l.index] * l.depth_within

        if l.wave_field == WaveField.within:
            return (self._waves_a[l.index] * np.exp(cterm) +
                    self._waves_b[l.index] * np.exp(-cterm))
        elif l.wave_field == WaveField.outcrop:
            return 2 * self._waves_a[l.index] * np.exp(cterm)
        elif l.wave_field == WaveField.incoming_only:
            return self._waves_a[l.index] * np.exp(cterm)
        else:
            raise NotImplementedError

    def calc_accel_tf(self, lin, lout):
        """Compute the acceleration transfer function.

        Parameters
        ----------
        lin : :class:`~site.Location`
            Location of input
        lout : :class:`~site.Location`
            Location of output. Note that this would typically be midheight
            of the layer.

        """
        tf = self.wave_at_location(lout) / self.wave_at_location(lin)
        return tf

    def calc_stress_tf(self, lin, lout, damped):
        """Compute the stress transfer function.

        Parameters
        ----------
        lin : :class:`~site.Location`
            Location of input
        lout : :class:`~site.Location`
            Location of output. Note that this would typically be midheight
            of the layer.

        """
        tf = self.calc_strain_tf(lin, lout)
        if damped:
            # Scale by complex shear modulus to include the influence of
            # damping
            tf *= lout.layer.comp_shear_mod
        else:
            tf *= lout.layer.shear_mod.value

        return tf

    def calc_strain_tf(self, lin, lout):
        """Compute the strain transfer function from `lout` to
        `location_in`.

        The strain transfer function from the acceleration at layer `n`
        (outcrop) to the mid-height of layer `m` (within) is defined as

        Parameters
        ----------
        lin : :class:`~site.Location`
            Location of input
        lout : :class:`~site.Location`
            Location of output. Note that this would typically be midheight
            of the layer.

        Returns
        -------
        strain_tf : :class:`numpy.ndarray`
            Transfer function to be applied to an acceleration FAS.
        """
        # FIXME: Correct discussion for using acceleration FAS
        # Strain(angFreq, z=h_m/2)
        # ------------------------ =
        #    accel_n(angFreq)
        #
        #          i k*_m [ A_m exp(i k*_m h_m / 2) - B_m exp(-i k*_m h_m / 2)]
        #          ------------------------------------------------------------
        #                         -angFreq^2 (2 * A_n)
        #
        assert lout.wave_field == WaveField.within
        # The numerator cannot be computed using wave_at_location() because
        # it is A - B.
        cterm = 1j * self._wave_nums[lout.index, :] * lout.depth_within
        numer = (1j * self._wave_nums[lout.index, :] *
                 (self._waves_a[lout.index, :] * np.exp(cterm) -
                  self._waves_b[lout.index, :] * np.exp(-cterm)))
        denom = -self.motion.angular_freqs ** 2 * self.wave_at_location(lin)
        # Scale into units from gravity
        tf = GRAVITY * numer / denom
        # Set frequencies close to zero to zero
        mask = np.isclose(self.motion.angular_freqs, 0)
        tf[mask] = 0

        return tf


class EquivalentLinearCalculation(LinearElasticCalculator):
    """Equivalent-linear site response calculator.
    """
    def __init__(self, strain_ratio=0.65, tolerance=0.01, max_iterations=15):
        """Initialize the class.

        Parameters
        ----------
        strain_ratio: float, default=0.65
            Ratio between the maximum strain and effective strain used to
            compute strain compatible properties.

        tolerance: float, default=0.01
            Tolerance in the iterative properties, which would cause the
            iterative process to terminate.

        max_iterations: int, default=15
            Maximum number of iterations to perform.
        """
        super().__init__()
        self._strain_ratio = strain_ratio
        self._tolerance = tolerance
        self._max_iterations = max_iterations

    def __call__(self, motion, profile, loc_input):
        """Perform the wave propagation.

        Parameters
        ----------
        motion: :class:`~.base.motion.Motion`
            Input motion.

        profile: :class:`~.base.site.Profile`
            Site profile.

        loc_input: :class:`~.base.site.Location`
            Location of the input motion.
        """
        self._motion = motion
        self._profile = profile
        self._loc_input = loc_input

        # Estimate the strain based on the PGV and shear-wave velocity
        for l in profile:
            l.strain = motion.pgv / l.initial_shear_vel

        iteration = 0
        while iteration < self.max_iterations:
            self._calc_waves(motion.angular_freqs, profile)

            for i, l in enumerate(profile[:-1]):
                l.strain = (
                    self.strain_ratio * motion.calc_peak(
                        self.calc_strain_tf(
                            loc_input,
                            Location(i, l, 'within', l.thickness / 2))
                    )
                )
            # Maximum error (damping and shear modulus) over all layers
            max_error = max(l.max_error for l in profile)
            if max_error < self.tolerance:
                break
            iteration += 1

    @property
    def strain_ratio(self):
        return self._strain_ratio

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def max_iterations(self):
        return self._max_iterations

    @classmethod
    def calc_strain_ratio(cls, mag):
        """Compute the effective strain ratio using Idriss and Sun (1991).

        Parameters
        ----------
        mag: float
            Magnitude of the input motion.

        Returns
        -------
        float
            Effective strain ratio
        """
        return (mag - 1) / 10
