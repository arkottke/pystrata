# The MIT License (MIT)
#
# Copyright (c) 2016-2018 Albert Kottke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import copy
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.sparse import diags

from . import site

# Used to define the random state. A specific state can be set with:
# random_state.set_seed(42)
random_state = np.random.RandomState()


class TruncatedNorm:
    """Truncated normal random number generator.

    Parameters
    ----------
    limit : float
        Standard normal limits to impose
    """

    def __init__(self, limit):
        self.limit = limit

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, value):
        self._limit = value

        # Need to scale the standard deviation to achieve sample standard
        # deviation based on the truncation. Given truncation of 2 standard
        # deviations, the input standard deviation must be increased to
        # 1.136847 to maintain a unit standard deviation for the random
        # samples.
        self._scale = 1 / np.sqrt(stats.truncnorm.stats(-value, value, moments="v"))

    @property
    def scale(self):
        return self._scale

    def __call__(self, size=1):
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
        return stats.truncnorm.rvs(
            -self.limit, self.limit, scale=self._scale, size=size
        )

    def correlated(self, correl):
        # Acceptance proportion
        accept = np.diff(stats.norm.cdf([-self.limit, self.limit]))[0]
        # The expected number of tries required
        expected = np.ceil(1 / accept).astype(int)

        while True:
            # Compute the multivariate normal with a unit variance and
            # specified standard deviation. Use twice the expected since
            # this calculation is fast and we don't want to loop.
            randvar = np.random.multivariate_normal(
                [0, 0], [[1, correl], [correl, 1]], size=(2 * expected)
            )
            valid = np.all(np.abs(randvar) < self.limit, axis=1)
            if np.any(valid):
                # Return the first valid value
                return randvar[valid][0]


# Random number generator used for all random number. Limited to +/- 2,
# and the standard deviation is scaled to maintain the standard deviation
# FIXME
randnorm = TruncatedNorm(2)


class ToroThicknessVariation(object):
    """Toro (1995) [T95]_ thickness variation model.

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
            depth = (
                np.power(
                    (self.c_2 * total) / self.c_3
                    + total / self.c_3
                    + np.power(self.c_1, self.c_2 + 1),
                    1 / (self.c_2 + 1),
                )
                - self.c_1
            )

            thickness = depth - depth_prev

            if depth > depth_total:
                thickness = depth_total - depth_prev
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
        layers = []
        for (thick, depth_mid) in self.iter_thickness(profile[-2].depth_base):
            # Locate the proper layer and add it to the model
            for l in profile:
                if l.depth < depth_mid <= l.depth_base:
                    layers.append(site.Layer(l.soil_type, thick, l.initial_shear_vel))
                    break
            else:
                raise LookupError

        # Add the half-space
        hsl = profile[-1]
        layers.append(site.Layer(hsl.soil_type, 0, hsl.initial_shear_vel))

        varied = site.Profile(layers, profile.wt_depth)
        return varied


class HalfSpaceDepthVariation(object):
    def __init__(self, dist: stats.rv_continuous):
        self._dist = dist

    def __call__(self, profile: site.Profile) -> site.Profile:
        # Update the distribution with the central value of the profile
        varied_depth = self._dist.rvs()

        # Find the layer
        index, depth_within = profile.lookup_depth(varied_depth)

        print(varied_depth, profile[-1].depth, index, depth_within)

        half_space = profile[-1]

        if index < (len(profile) - 1):
            # Variation is within the layers
            layers = [site.Layer.copy_of(layer) for layer in profile[: (index + 1)]]
            # Reduce the thickness of the layer above the half-space
            layers[-1]._thickness = depth_within
        else:
            # Variation extends past the depth of the model
            orig_thick = profile[-2].thickness
            total_thick = orig_thick + depth_within
            count = np.ceil(total_thick // orig_thick).astype(int)

            thick = total_thick / count

            print(count, thick)

            # Don't copy half-space
            layers = [site.Layer.copy_of(layer) for layer in profile[:-1]]
            # Don't call the setter function as it needs a profile defined
            layers[-1]._thickness = thick
            parent = layers[-1]
            for _ in range(count - 1):
                layers.append(site.Layer.copy_of(parent))

        layers.append(half_space)

        return site.Profile(layers, profile.wt_depth)


class LayerThicknessVariation(object):
    def __init__(
        self,
        models: Union[List[stats.rv_continuous], Dict[int, stats.rv_continuous]],
        discretize_kwds: Optional[Dict[str, float]] = None,
    ) -> None:

        self._models = models
        self._discretize_kwds = discretize_kwds


class VelocityVariation(object):
    """Abstract model for varying the velocity."""

    def __init__(self, vary_bedrock=False):
        self._vary_bedrock = vary_bedrock

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

        mean = np.log(profile.initial_shear_vel)
        covar = self._calc_covar_matrix(profile)

        ln_vel_rand = np.random.multivariate_normal(mean, covar, check_valid="ignore")

        # Limits based on the number of standard deviations
        offset = randnorm.limit * np.sqrt(np.diag(covar))
        ln_vel = np.clip(ln_vel_rand, mean - offset, mean + offset)
        vel = np.exp(ln_vel)

        varied = site.Profile.copy_of(profile)
        # Update the velocities
        end = None if self.vary_bedrock else -1
        for i, v in enumerate(vel[:end]):
            varied[i].initial_shear_vel = v

        return varied

    def _calc_covar_matrix(self, profile):
        """Calculate the covariance matrix.

        Parameters
        ----------
        profile : site.Profile
            Input site profile

        Yields
        ------
        covar : `class`:numpy.array
            Covariance matrix
        """
        corr = self._calc_corr(profile)
        std = self._calc_ln_std(profile)
        # Modify the standard deviation by the truncated norm scale
        std *= randnorm.scale

        var = std**2
        covar = corr * std[:-1] * std[1:]

        # Main diagonal is the variance
        mat = diags([covar, var, covar], [-1, 0, 1]).toarray()

        return mat

    @abstractmethod
    def _calc_corr(self, profile):
        """Compute the adjacent-layer correlations.

        Parameters
        ----------
        profile : site.Profile
            Input site profile

        Yields
        ------
        np.array
            Correlation matrix
        """
        raise NotImplementedError

    @abstractmethod
    def _calc_ln_std(self, profile):
        """Compute the standard deviation for each layer.

        Parameters
        ----------
        profile : site.Profile
            Input site profile

        Yields
        ------
        np.array
            Standard deviation of the shear-wave velocity
        """
        raise NotImplementedError

    @property
    def vary_bedrock(self):
        return self._vary_bedrock


class ToroVelocityVariation(VelocityVariation):
    """Toro (1995) [T95] velocity variation model.

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
        "Geomatrix AB": {
            "ln_std": 0.46,
            "rho_0": 0.96,
            "delta": 13.1,
            "rho_200": 0.96,
            "h_0": 0.0,
            "b": 0.095,
        },
        "Geomatrix CD": {
            "ln_std": 0.38,
            "rho_0": 0.99,
            "delta": 8.0,
            "rho_200": 1.00,
            "h_0": 0.0,
            "b": 0.160,
        },
        "USGS AB": {
            "ln_std": 0.35,
            "rho_0": 0.95,
            "delta": 4.2,
            "rho_200": 1.00,
            "h_0": 0.0,
            "b": 0.138,
        },
        "USGS CD": {
            "ln_std": 0.36,
            "rho_0": 0.99,
            "delta": 3.9,
            "rho_200": 1.00,
            "h_0": 0.0,
            "b": 0.293,
        },
        "USGS A": {
            "ln_std": 0.36,
            "rho_0": 0.95,
            "delta": 3.4,
            "rho_200": 0.42,
            "h_0": 0.0,
            "b": 0.063,
        },
        "USGS B": {
            "ln_std": 0.27,
            "rho_0": 0.97,
            "delta": 3.8,
            "rho_200": 1.00,
            "h_0": 0.0,
            "b": 0.293,
        },
        "USGS C": {
            "ln_std": 0.31,
            "rho_0": 0.99,
            "delta": 3.9,
            "rho_200": 0.98,
            "h_0": 0.0,
            "b": 0.344,
        },
        "USGS D": {
            "ln_std": 0.37,
            "rho_0": 0.00,
            "delta": 5.0,
            "rho_200": 0.50,
            "h_0": 0.0,
            "b": 0.744,
        },
    }

    def __init__(
        self,
        ln_std: float,
        rho_0: float,
        delta: float,
        rho_200: float,
        h_0: float,
        b: float,
        vary_bedrock: bool = False,
    ):
        """Initialize the model."""
        super().__init__(vary_bedrock=vary_bedrock)

        self._ln_std = ln_std
        self._rho_0 = rho_0
        self._delta = delta
        self._rho_200 = rho_200
        self._h_0 = h_0
        self._b = b

    def _calc_corr(self, profile: site.Profile) -> np.ndarray:
        """Compute the adjacent-layer correlations

        Parameters
        ----------
        profile : :class:`site.Profile`
            Input site profile

        Yields
        ------
        corr : :class:`numpy.array`
            Adjacent-layer correlations
        """

        # Toro defines the depth as the average midpoint depths of layers i and i-1.
        depths_mid = np.array(profile.depth_mid)
        depth = np.mean(np.c_[depths_mid[:-1], depths_mid[1:]], axis=1)

        # t variable from Toro; defined as the difference of the midpoint depths
        thick = np.diff(depth)
        # Remove the depth associated with the final layer. We will set that it is
        # perfectly correlated later.
        depth = depth[:-1]

        # Depth dependent correlation
        corr_depth = self.rho_200 * np.power(
            (depth + self.h_0) / (200 + self.h_0), self.b
        )
        corr_depth[depth > 200] = self.rho_200

        # Thickness dependent correlation
        corr_thick = self.rho_0 * np.exp(-thick / self.delta)

        # Final correlation
        # Correlation coefficient
        corr = (1 - corr_depth) * corr_thick + corr_depth

        # Bedrock is perfectly correlated with layer above it
        corr = np.r_[corr, 1]

        return corr

    def _calc_ln_std(self, profile):
        ln_std = self.ln_std * np.ones(len(profile))
        return ln_std

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


class DepthDependToroVelVariation(ToroVelocityVariation):
    """Toro (1995) [T95] velocity variation model modified for a depth
    dependent standard deviation that can be overridden by the soil_type name.

    Default values can be selected with :meth:`.generic_model`.

    Parameters
    ----------
    depth: array_like, optional
        Depths defining the standard deviation model. Default is [0, 15]
        following the SPID model.
    ln_std: array_like, optional
        :math:`\sigma_{ln}` model parameter. Default is [0.25, 0.15]
        following the SPID model.
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
    ln_std_map: dict[str, float], optional
        Mapping between the soil_type and the defined ln_std. Default is *None*.
    """

    def __init__(
        self,
        depth: npt.ArrayLike,
        ln_std: npt.ArrayLike,
        rho_0: float,
        delta: float,
        rho_200: float,
        h_0: float,
        b: float,
        ln_std_map=Optional[Dict[str, float]],
        vary_bedrock=False,
    ):
        """Initialize the model."""
        super().__init__(
            ln_std, rho_0, delta, rho_200, h_0, b, vary_bedrock=vary_bedrock
        )
        self.depth = depth
        self.ln_std_map = ln_std_map or dict()

    def _calc_ln_std(self, profile):
        # Depth based values
        ln_std = np.interp(
            profile.depth_mid,
            self.depth,
            self.ln_std,
            left=self.ln_std[0],
            right=self.ln_std[-1],
        )

        # Update based on soil_type name
        for name, value in self.ln_std_map.items():
            for i, l in enumerate(profile):
                if name in l.soil_type.name:
                    ln_std[i] = value

        return ln_std

    @classmethod
    def generic_model(cls, site_class, /, *, ln_std_map=None, **kwds):
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

        ln_std_map: dict[str, float], optional
            Mapping between the soil_type and the defined ln_std. Default is *None*.

        Returns
        -------
        DepthAndSoilTypeDependToroVelVariation
            Initialized :class:`DepthDependToroVelVariation` with generic parameters.
        """
        p = dict(cls.PARAMS[site_class])
        p.update(kwds)

        if "depth" not in kwds:
            p["depth"] = [0, 15]
            p["ln_std"] = [0.25, 0.15]

        p["ln_std_map"] = ln_std_map
        return cls(**p)


class SoilTypeVariation(object):
    def __init__(
        self,
        correlation,
        limits_mod_reduc=[0.05, 1],
        limits_damping=[0, 0.15],
        vary_bedrock=False,
    ):
        self._vary_bedrock = vary_bedrock
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

        # A pair of correlated random variables
        randvar = randnorm.correlated(self.correlation)

        varied_mod_reduc, varied_damping = self._get_varied(randvar, mod_reduc, damping)

        # Clip the values to the specified min/max
        varied_mod_reduc = np.clip(
            varied_mod_reduc, self.limits_mod_reduc[0], self.limits_mod_reduc[1]
        )
        varied_damping = np.clip(
            varied_damping, self.limits_damping[0], self.limits_damping[1]
        )

        # Set the values
        realization = copy.deepcopy(soil_type)
        for attr_name, values in zip(
            ["mod_reduc", "damping"], [varied_mod_reduc, varied_damping]
        ):
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

    @property
    def vary_bedrock(self):
        return self._vary_bedrock


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
        std = np.exp(-4.23) + np.sqrt(
            0.25 / np.exp(3.62) - (mod_reduc - 0.5) ** 2 / np.exp(3.62)
        )
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
        std = (np.exp(-5) + np.exp(-0.25) * np.sqrt(100 * damping)) / 100.0
        return std


class SpidVariation(SoilTypeVariation):
    """Variation defined by the EPRI SPID (2013) and documented in
    PNNL (2014)."""

    def __init__(
        self,
        correlation,
        limits_mod_reduc=[0, 1],
        limits_damping=[0, 0.15],
        std_mod_reduc=0.15,
        std_damping=0.30,
    ):
        super().__init__(correlation, limits_mod_reduc, limits_damping)
        self._std_mod_reduc = std_mod_reduc
        self._std_damping = std_damping

    def _get_varied(self, randvar, mod_reduc, damping):
        # Vary the G/Gmax in transformed space.
        # Equation 9.43 of PNNL (2014)
        # Here epsilon is added so that the denomiator doesn't go to zero.
        f_mean = mod_reduc / (1 - mod_reduc + np.finfo(float).eps)
        # Instead of constraining the standard deviation at a specific
        # strain, then standard deviation is constrained at G/Gmax of 0.5.
        # This is modified from Equation 9.44 of PNNL (2014).
        f_std = self.std_mod_reduc * (1 / (1 - 0.5))
        f_real = np.exp(randvar[0] * f_std) * f_mean
        # Equation 9.45 of PNNL (2014)
        varied_mod_reduc = f_real / (1 + f_real)

        # Simple log distribution
        varied_damping = np.exp(randvar[1] * self.std_damping) * damping

        return varied_mod_reduc, varied_damping

    @property
    def std_damping(self):
        return self._std_damping

    @property
    def std_mod_reduc(self):
        return self._std_mod_reduc


def iter_varied_profiles(
    profile,
    count,
    var_thickness: ToroThicknessVariation = None,
    var_velocity: VelocityVariation = None,
    var_soiltypes: SoilTypeVariation = None,
):
    for _ in range(count):
        # Copy the profile to form the realization
        p = site.Profile.copy_of(profile)

        if var_thickness:
            p = var_thickness(p)

        if var_velocity:
            p = var_velocity(p)

        if var_soiltypes:
            # Map of varied soil types
            varied = {str(st): var_soiltypes(st) for st in p.iter_soil_types()}
            # Create new layers
            end = None if var_soiltypes.vary_bedrock else -1
            layers = [
                site.Layer(varied[str(l.soil_type)], l.thickness, l.initial_shear_vel)
                for l in p[:end]
            ]
            # Add the unrandomized bedrock
            if not var_soiltypes.vary_bedrock:
                layers.append(p[-1])

            # Create a new profile
            p = site.Profile(layers, p.wt_depth)

        yield p
