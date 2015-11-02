__author__ = 'akottke'

from typing import Iterable, Iterator

import numpy as np

from . import site


class ToroThicknessVariation(object):
    """Toro (1995) [1]_ thickness variation model.


       .. [1] Toro, G. R. (1995). Probabilistic models of site velocity
       profiles for generic and site-specific ground-motion amplification
       studies. Brookhaven National Laboratory Technical Report: 779574.
    """

    def __init__(self,
                 c_1: float = 10.86,
                 c_2: float = -0.89,
                 c_3: float = 1.98,
                 ):
        """Initialize the model.

        The model parameters proposed in Toro (1995) are provided by default.

        Parameters
        ----------
        c_1 : float, optional
            `c_1` model parameter.
        c_2 : float, optional
            `c_2` model parameter.
        c_3 : float, optional
            `c_3` model parameter.
        """

        self._c_3 = c_3
        self._c_1 = c_1
        self._c_2 = c_2

    @property
    def c_3(self) -> float:
        return self._c_3

    @property
    def c_2(self) -> float:
        return self._c_2

    @property
    def c_1(self) -> float:
        return self._c_1

    def iter_thickness(self, depth_total: float) -> Iterator:
        """Iterate over the varied thicknesses.

        The layering is generated using a non-homogenous Poisson process. The
        following routine is used to generate the layering. The rate
        function, :math:`\lambda(t)`, is integrated from 0 to t to generate
        cumulative rate function, :math:`\Lambda(t)`. This function is then
        inverted producing :math:`\Lambda^-1(t)`. Random variables
        are produced using the a
        exponential random variation with :math:`\mu = 1` and converted to the
        nonhomogenous variables using the inverted function. Random variable
        that is a sum of exponential random variables.

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
        sum = 0
        depth_prev = 0

        while depth_prev < depth_total:
            # Add a random c_2ial increment
            sum += np.random.exponential(1.0)

            # Convert between x and depth using the inverse of \Lambda(t)
            depth = np.power(
                (self.c_2 * sum) / self.c_3 + sum / self.c_3 +
                np.power(self.c_1, self.c_2 + 1),
                1 / (self.c_2 + 1)) - self.c_1

            thickness = depth - depth_prev

            if depth > depth_total:
                thickness = (depth_total - depth_prev)
                depth = depth_prev + thickness

            depth_mid = (depth_prev + depth) / 2
            yield thickness, depth_mid

            depth_prev = depth

    def __call__(self, profile) -> site.Profile:
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

        for thickness, depth_mid in self.iter_thickness(profile[-2].depth_base):
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

        profile_varied.update_depths()

        return profile_varied


class ToroVelocityVariation(object):
    """Toro (1995) velocity variation model.

    Citation:
       Toro, G. R. (1995). Probabilistic models of site velocity profiles for
       generic and site-specific ground-motion amplification studies.
       Brookhaven National Laboratory Technical Rep, 779574.

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
        """Initialize the model.

        The model parameters proposed in Toro (1995) are provided by default.

        Parameters
        ----------
        ln_std : float, optional
            :math:`\sigma_\ln` model parameter.
        rho_0 : float, optional
            :math:`\rho_0` model parameter.
        delta : float, optional
            :math:`\delta` model parameter.
        rho_200 : float, optional
            :math:`\rho_200` model parameter.
        h_0 : float, optional
            :math:`h_0` model parameter.
        b : float, optional
            :math:`b` model parameter.
        """
        self._ln_std = ln_std
        self._rho_0 = rho_0
        self._delta = delta
        self._rho_200 = rho_200
        self._h_0 = h_0
        self._b = b

    def iter_correlated_variables(self, profile: site.Profile) -> Iterator:
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
        var_prev = np.random.standard_normal()
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
            var_cur = (corr * var_prev +
                       np.random.standard_normal() * np.sqrt(1 - corr ** 2))

            yield var_cur
            var_prev = var_cur

        # Bedrock is perfectly correlated with layer above it
        yield var_prev

    def __call__(self, profile: site.Profile) -> site.Profile:
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

        profile_varied.update_depths()

        return profile_varied

    @property
    def ln_std(self) -> float:
        return self._ln_std

    @property
    def rho_0(self) -> float:
        return self._rho_0

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def rho_200(self) -> float:
        return self._rho_200

    @property
    def h_0(self) -> float:
        return self._h_0

    @property
    def b(self) -> float:
        return self._b

    @classmethod
    def site_classes(cls) -> Iterable[str]:
        return cls.PARAMS.keys()

    @classmethod
    def generic_model(cls, site_class: str, **kwds) -> 'ToroVelocityVariation':
        """Use generic model parameters based on site class.

        Parameters
        ----------
        site_class : str
            Site classification. Possible options are:
                Geomatrix AB
                Geomatrix CD
                USGS AB
                USGS CD
                USGS A
                USGS B
                USGS C
                USGS D
            See the report for definitions of the Geomatrix site
            classication. USGS site classification is based on :math:`V_{s30}`:
                A: >750 m/s
                B: 360 to 750 m/s
                C: 180 to 360 m/s
                D: <180 m/s
        Returns
        -------
        ToroVelocityVariation
        """
        p = dict(cls.PARAMS[site_class])
        p.update(kwds)
        return cls(**p)
