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

import copy

import numpy as np
from scipy.stats import truncnorm, norm

from . import site

# Limit of number of standard deviation for number generation
STD_LIM = 2
# Need to scale the standard deviation to achieve sample standard deviation
# based on the truncation. Given truncation of 2 standard deviations,
# the input standard deviation must be increased to 1.136847 to maintain a
# unit standard deviation for the random samples. This is based on equation
# on Wikipedia.
#
# https://en.wikipedia.org/wiki/Truncated_normal_distribution#Moments
#
# If STD_LIM is changed, then this should be adjusted.
STD_SCALE = 1 / np.sqrt((
    1 +
    (-STD_LIM * norm.pdf(-STD_LIM) -
     STD_LIM * norm.pdf(STD_LIM)) /
    (norm.cdf(STD_LIM) - norm.cdf(-STD_LIM)) -
    ((norm.pdf(-STD_LIM) - norm.pdf(STD_LIM)) /
     (norm.cdf(STD_LIM) - norm.cdf(-STD_LIM)) ** 2)))


def randnorm(size=1):
    """Random number generator that follows a truncated normal distribution.

    This is the defalut random number generator used by the program. It
    generates normally distributed values ranging from -2 to +2 with unit
    standard deviation.

    The state of the random number generator is controlled by the
    ``np.random.RandomState`` instance.

    Parameters
    ----------
    size : int
        Number of random values to compute

    Returns
    -------
    rvs : ndarray or scalar
        Random variates of given `size`.
    """
    return truncnorm.rvs(-STD_LIM, STD_LIM, scale=STD_SCALE, size=size)


class ToroThicknessVariation(object):
    """ Toro (1995) [T95]_ thickness variation model.

    The recommended values are provided as defaults to this model.

    .. rubric:: References

    .. [T95] Toro, G. R. (1995). Probabilistic models of site velocity
        profiles for generic and site-specific ground-motion amplification
        studies. Brookhaven National Laboratory Technical Report: 779574.

    Parameters
    ----------
    c_1: float, optional
        :math:`c_1` model parameter.
    c_2: float, optional
        :math:`c_2` model parameter.
    c_3: float, optional
        :math:`c_3` model parameter.

    """

    def __init__(self, c_1=10.86, c_2=-0.89, c_3=1.98):
        self._c_1 = c_1
        self._c_2 = c_2
        self._c_3 = c_3

    @property
    def c_3(self):
        return self._c_3

    @property
    def c_2(self):
        return self._c_2

    @property
    def c_1(self):
        return self._c_1

    def iter_thickness(self, depth_total):
        """Iterate over the varied thicknesses.

        The layering is generated using a non-homogenous Poisson process. The
        following routine is used to generate the layering. The rate
        function, :math:`\lambda(t)`, is integrated from 0 to t to generate
        cumulative rate function, :math:`\Lambda(t)`. This function is then
        inverted producing :math:`\Lambda^{-1}(t)`. Random variables
        are produced using the a exponential random variation with
        :math:`\mu = 1` and converted to the nonhomogenous variables using
        the inverted function.

        Parameters
        ----------
        depth_total: float
            Total depth generated. Last thickness is truncated to achieve
            this depth.

        Yields
        ------
        float
            Varied thickness.

        """
        total = 0
        depth_prev = 0

        while depth_prev < depth_total:
            # Add a random exponential increment
            total += np.random.exponential(1.0)

            # Convert between x and depth using the inverse of \Lambda(t)
            depth = np.power(
                (self.c_2 * total) / self.c_3 + total / self.c_3 +
                np.power(self.c_1, self.c_2 + 1),
                1 / (self.c_2 + 1)) - self.c_1

            thickness = depth - depth_prev

            if depth > depth_total:
                thickness = (depth_total - depth_prev)
                depth = depth_prev + thickness

            depth_mid = (depth_prev + depth) / 2
            yield thickness, depth_mid

            depth_prev = depth

    def __call__(self, profile):
        """Calculated a varied thickness profile.

        Parameters
        ----------
        profile : site.Profile
            Profile to be varied. Not modified in place.

        Returns
        -------
        site.Profile
            Varied site profile.

        """

        profile_varied = site.Profile()
        for (thickness, depth_mid) in \
                self.iter_thickness(profile[-2].depth_base):
            # Locate the proper layer and add it to the model
            for l in profile:
                if l.depth < depth_mid <= l.depth_base:
                    profile_varied.append(
                        site.Layer(l.soil_type, thickness, l.initial_shear_vel)
                    )
                    break
            else:
                raise LookupError

        # Add the half-space
        l = profile[-1]
        profile_varied.append(
            site.Layer(l.soil_type, 0, l.initial_shear_vel)
        )

        profile_varied.update_layers()

        return profile_varied


class ToroVelocityVariation(object):
    """ Toro (1995) [T95] velocity variation model.

    Default values can be selected with :meth:`.generic_model`.

    Parameters
    ----------
    ln_std: float, optional
        :math:`\sigma_{ln}` model parameter.
    rho_0: float, optional
        :math:`ρ_0` model parameter.
    delta: float, optional
        :math:`\Delta` model parameter.
    rho_200: float, optional
        :math:`ρ_200` model parameter.
    h_0: float, optional
        :math:`h_0` model parameter.
    b: float, optional
        :math:`b` model parameter.
    """

    PARAMS = {
        'Geomatrix AB': {
            'ln_std': 0.46,
            'rho_0': 0.96,
            'delta': 13.1,
            'rho_200': 0.96,
            'h_0': 0.0,
            'b': 0.095,
        },
        'Geomatrix CD': {
            'ln_std': 0.38,
            'rho_0': 0.99,
            'delta': 8.0,
            'rho_200': 1.00,
            'h_0': 0.0,
            'b': 0.160,
        },
        'USGS AB': {
            'ln_std': 0.35,
            'rho_0': 0.95,
            'delta': 4.2,
            'rho_200': 1.00,
            'h_0': 0.0,
            'b': 0.138,
        },
        'USGS CD': {
            'ln_std': 0.36,
            'rho_0': 0.99,
            'delta': 3.9,
            'rho_200': 1.00,
            'h_0': 0.0,
            'b': 0.293,
        },
        'USGS A': {
            'ln_std': 0.36,
            'rho_0': 0.95,
            'delta': 3.4,
            'rho_200': 0.42,
            'h_0': 0.0,
            'b': 0.063,
        },
        'USGS B': {
            'ln_std': 0.27,
            'rho_0': 0.97,
            'delta': 3.8,
            'rho_200': 1.00,
            'h_0': 0.0,
            'b': 0.293,
        },
        'USGS C': {
            'ln_std': 0.31,
            'rho_0': 0.99,
            'delta': 3.9,
            'rho_200': 0.98,
            'h_0': 0.0,
            'b': 0.344,
        },
        'USGS D': {
            'ln_std': 0.37,
            'rho_0': 0.00,
            'delta': 5.0,
            'rho_200': 0.50,
            'h_0': 0.0,
            'b': 0.744,
        },
    }

    def __init__(self, ln_std, rho_0, delta, rho_200, h_0, b):
        """Initialize the model."""
        self._ln_std = ln_std
        self._rho_0 = rho_0
        self._delta = delta
        self._rho_200 = rho_200
        self._h_0 = h_0
        self._b = b

    def iter_correlated_variables(self, profile):
        """Iteratively provide corrlated variables.

        Parameters
        ----------
        profile : site.Profile
            Input site profile

        Yields
        ------
        float
            Correlated random variable of velocity ranging from
            :math:`-\infty` to :math:`+\infty`.
        """
        var_prev = randnorm()
        yield var_prev

        for i in range(len(profile) - 1):
            h = (profile[i + 1].depth_mid + profile[i].depth_mid) / 2
            t = profile[i + 1].depth_mid - profile[i].depth_mid

            if h <= 200.:
                corr_d = (
                    self.rho_200 * np.power(
                        (h + self.rho_0) / (200 + self.rho_0),
                        self.b)
                )
            else:
                corr_d = self.rho_200

            corr_t = self.rho_0 * np.exp(-t / self.delta)

            # Correlation coefficient
            corr = (1 - corr_d) * corr_t + corr_d

            # Correlated random variable
            var_cur = corr * var_prev + randnorm() * np.sqrt(1 - corr ** 2)
            yield var_cur
            var_prev = var_cur

        # Bedrock is perfectly correlated with layer above it
        yield var_prev

    def __call__(self, profile):
        """Calculate a varied shear-wave velocity profile.

        Parameters
        ----------
        profile : site.Profile
            Profile to be varied. Not modified in place.

        Returns
        -------
        site.Profile
            Varied site profile.
        """

        profile_varied = site.Profile()
        for l, corr_var in zip(profile,
                               self.iter_correlated_variables(profile)):
            # FIXME: add layer specific ln_std
            ln_std = self.ln_std
            shear_vel_varied = l.initial_shear_vel * np.exp(ln_std * corr_var)

            profile_varied.append(
                site.Layer(
                    l.soil_type,
                    l.thickness,
                    shear_vel_varied,
                )
            )

        profile_varied.update_layers()

        return profile_varied

    @property
    def ln_std(self):
        return self._ln_std

    @property
    def rho_0(self):
        return self._rho_0

    @property
    def delta(self):
        return self._delta

    @property
    def rho_200(self):
        return self._rho_200

    @property
    def h_0(self):
        return self._h_0

    @property
    def b(self):
        return self._b

    @classmethod
    def site_classes(cls):
        return cls.PARAMS.keys()

    @classmethod
    def generic_model(cls, site_class, **kwds):
        """Use generic model parameters based on site class.

        Parameters
        ----------
        site_class: str
            Site classification. Possible options are:
             * Geomatrix AB
             * Geomatrix CD
             * USGS AB
             * USGS CD
             * USGS A
             * USGS B
             * USGS C
             * USGS D

            See the report for definitions of the Geomatrix site
            classication. USGS site classification is based on :math:`V_{s30}`:

            =========== =====================
            Site Class  :math:`V_{s30}` (m/s)
            =========== =====================
            A           >750 m/s
            B           360 to 750 m/s
            C           180 to 360 m/s
            D           <180 m/s
            =========== =====================

        Returns
        -------
        ToroVelocityVariation
            Initialized :class:`ToroVelocityVariation` with generic parameters.
        """
        p = dict(cls.PARAMS[site_class])
        p.update(kwds)
        return cls(**p)


class SoilTypeVariation(object):
    def __init__(self, correlation,
                 limits_mod_reduc=[0, 1], limits_damping=[0, 0.15]):
        self._correlation = correlation
        self._limits_mod_reduc = list(limits_mod_reduc)
        self._limits_damping = list(limits_damping)

    def __call__(self, soil_type):
        def get_values(nlp):
            try:
                return nlp.values
            except AttributeError:
                return np.asarray(nlp).astype(float)

        mod_reduc = get_values(soil_type.mod_reduc)
        damping = get_values(soil_type.damping)

        # Create correlated random variables. Generating truncated
        # correlated random variables is challenging. Instead, we just loop
        # until it works.
        #
        # todo: More elegant solution?
        while True:
            randvar = np.random.multivariate_normal(
                [0, 0],
                [[STD_SCALE ** 2, self.correlation * STD_SCALE ** 2],
                 [self.correlation * STD_SCALE ** 2, STD_SCALE ** 2]])
            if np.all(abs(randvar) < STD_LIM):
                break

        varied_mod_reduc, varied_damping = self._get_varied(
            randvar, mod_reduc, damping)

        # Clip the values to the specified min/max
        varied_mod_reduc = np.clip(
            varied_mod_reduc, self.limits_mod_reduc[0],
            self.limits_mod_reduc[1]
        )
        varied_damping = np.clip(
            varied_damping, self.limits_damping[0], self.limits_damping[1]
        )

        # Set the values
        realization = copy.deepcopy(soil_type)
        for attr_name, values in zip(['mod_reduc', 'damping'],
                                     [varied_mod_reduc, varied_damping]):
            try:
                getattr(realization, attr_name).values = values
            except AttributeError:
                setattr(realization, attr_name, values)
        return realization

    def _get_varied(self, randvar, mod_reduc, damping):
        raise NotImplementedError

    @property
    def correlation(self):
        return self._correlation

    @property
    def limits_damping(self):
        return self._limits_damping

    @property
    def limits_mod_reduc(self):
        return self._limits_mod_reduc


class DarendeliVariation(SoilTypeVariation):
    def _get_varied(self, randvar, mod_reduc, damping):
        mod_reduc_means = mod_reduc
        mod_reduc_stds = self.calc_std_mod_reduc(mod_reduc_means)
        varied_mod_reduc = mod_reduc_means + randvar[0] * mod_reduc_stds

        damping_means = damping
        damping_stds = self.calc_std_damping(damping_means)
        varied_damping = damping_means + randvar[1] * damping_stds

        return varied_mod_reduc, varied_damping

    @staticmethod
    def calc_std_mod_reduc(mod_reduc):
        """Calculate the standard deviation as a function of G/G_max.

        Equation 7.29 from Darendeli (2001).

        Parameters
        ----------
        mod_reduc : array_like
            Modulus reduction values.

        Returns
        -------
        std : :class:`numpy.ndarray`
            Standard deviation.
        """
        mod_reduc = np.asarray(mod_reduc).astype(float)
        std = (np.exp(-4.23) +
               np.sqrt(0.25 / np.exp(3.62) -
                       (mod_reduc - 0.5) ** 2 / np.exp(3.62)))
        return std

    @staticmethod
    def calc_std_damping(damping):
        """Calculate the standard deviation as a function of damping in decimal.

        Equation 7.30 from Darendeli (2001).

        Parameters
        ----------
        damping : array_like
            Material damping values in decimal.

        Returns
        -------
        std : :class:`numpy.ndarray`
            Standard deviation.
        """
        damping = np.asarray(damping).astype(float)
        std = (np.exp(-5) + np.exp(-0.25) * np.sqrt(100 * damping)) / 100.
        return std


class SpidVariation(SoilTypeVariation):
    """Variation defined by the EPRI SPID (2013) and documented in
    PNNL (2014)."""

    def __init__(self, correlation,
                 limits_mod_reduc=[0, 1], limits_damping=[0, 0.15],
                 std_mod_reduc=0.15, std_damping=0.30):
        super().__init__(
            correlation, limits_mod_reduc, limits_damping)
        self._std_mod_reduc = std_mod_reduc
        self._std_damping = std_damping

    def _get_varied(self, randvar, mod_reduc, damping):
        # Vary the G/Gmax in transformed space.
        # Equation 9.43 of PNNL (2014)
        f_mean = (mod_reduc / (1 - mod_reduc))
        # Instead of constraining the standard deviation at a specific
        # strain, then standard deviation is constrained at G/Gmax of 0.5.
        # This is modified from Equation 9.44 of PNNL (2014).
        f_std = self.std_mod_reduc * (1 / (1 - 0.5))
        f_real = np.exp(randvar[0] * f_std) * f_mean
        # Equation 9.45 of PNNL (2014)
        varied_mod_reduc = f_real / (1 + f_real)

        # Simple log distribution
        varied_damping = \
            np.exp(randvar[1] * self.std_damping) * damping

        return varied_mod_reduc, varied_damping

    @property
    def std_damping(self):
        return self._std_damping

    @property
    def std_mod_reduc(self):
        return self._std_mod_reduc


def iter_varied_profiles(profile, count, var_thickness=None,
                         var_velocity=None,
                         var_soiltypes=None):
    for i in range(count):
        if var_thickness is None:
            varied = copy.deepcopy(profile)
        else:
            varied = var_thickness(profile)

        if var_velocity is not None:
            var_velocity(varied)

        if var_soiltypes is not None:
            for st in varied.iter_soil_types():
                st_varied = var_soiltypes(st)
                # Copy over the varied properties
                for attr in ['mod_reduc', 'damping']:
                    if getattr(st, attr) is not None:
                        getattr(st, attr).values[:] = \
                            getattr(st_varied, attr).values
