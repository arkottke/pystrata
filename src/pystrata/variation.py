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
from collections.abc import Callable, Generator

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.sparse import diags

from . import site
from .units import convert_units

# Default generator used when a random number generator is not provided. Prefer
# passing ``rng=`` explicitly -- a shared global stream cannot be made
# reproducible across processes, because forked workers inherit identical state
# and would draw identical realizations.
_default_rng = np.random.default_rng()


def _as_generator(rng) -> np.random.Generator:
    """Coerce *rng* into a :class:`numpy.random.Generator`.

    Parameters
    ----------
    rng : None, int, or numpy.random.Generator
        ``None`` selects the module-level default generator, so calls that do
        not specify a generator share a single stream. An integer is used as a
        seed.

    Returns
    -------
    numpy.random.Generator
    """
    if rng is None:
        return _default_rng
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


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

    def __call__(self, size=1, rng=None):
        """Random number generator that follows a truncated normal distribution.

        This is the default random number generator used by the program. It
        generates normally distributed values ranging from -2 to +2 with unit
        standard deviation.

        Parameters
        ----------
        size : int
            Number of random values to compute
        rng : None, int, or numpy.random.Generator, optional
            Generator used to draw the variates. Defaults to the module-level
            generator.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.
        """
        return stats.truncnorm.rvs(
            -self.limit,
            self.limit,
            scale=self._scale,
            size=size,
            random_state=_as_generator(rng),
        )

    def correlated(self, correl, rng=None):
        # Acceptance proportion
        accept = np.diff(stats.norm.cdf([-self.limit, self.limit]))[0]
        # The expected number of tries required
        expected = np.ceil(1 / accept).astype(int)

        generator = _as_generator(rng)
        while True:
            # Compute the multivariate normal with a unit variance and
            # specified standard deviation. Use twice the expected since
            # this calculation is fast and we don't want to loop.
            randvar = generator.multivariate_normal(
                [0, 0], [[1, correl], [correl, 1]], size=(2 * expected)
            )
            valid = np.all(np.abs(randvar) < self.limit, axis=1)
            if np.any(valid):
                # Return the first valid value
                return randvar[valid][0]

    def correlated_at_percentile(self, correl, percentile):
        """Return a deterministic correlated pair for a given percentile.

        The primary variate ``z1`` is the inverse CDF of the *scaled*
        truncated normal at ``percentile``.  The secondary variate ``z2``
        is drawn from the conditional distribution
        :math:`z_2 | z_1 = \\rho z_1 + \\sqrt{1 - \\rho^2}\\, z_1`,
        also clipped to ``[-limit, +limit]``.

        Both variates are expressed in the same units as the samples
        returned by :meth:`correlated` (unit standard deviation, truncated
        at ``±limit``).

        Parameters
        ----------
        correl : float
            Correlation coefficient between the two variates.
        percentile : float
            Quantile in ``(0, 1)`` to evaluate.

        Returns
        -------
        pair : ndarray, shape (2,)
            Deterministic correlated variate pair.
        """
        # Map the percentile through the scaled truncated-normal CDF so that
        # the resulting z1 sits at exactly that percentile of the marginal
        # distribution used by the random sampler.
        # stats.truncnorm with scale == self._scale and limits
        # [-limit/scale, +limit/scale] (un-scaled) matches the marginals.
        a, b = -self._limit / self._scale, self._limit / self._scale
        z1 = stats.truncnorm.ppf(percentile, a, b, scale=self._scale)
        # Conditional mean of z2 given z1 under bivariate normal
        z2 = correl * z1 + np.sqrt(max(1.0 - correl**2, 0.0)) * z1
        # Clip both to the truncation window
        z1 = np.clip(z1, -self._limit, self._limit)
        z2 = np.clip(z2, -self._limit, self._limit)
        return np.array([z1, z2])


# Random number generator used for all random number. Limited to +/- 2,
# and the standard deviation is scaled to maintain the standard deviation
# FIXME
randnorm = TruncatedNorm(2)


class ToroThicknessVariation:
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

    def iter_thickness(self, depth_total, rng=None):
        r"""Iterate over the varied thicknesses.

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
        rng : None, int, or numpy.random.Generator, optional
            Generator used to draw the layer increments. Defaults to the
            module-level generator.

        Yields
        ------
        float
            Varied thickness.
        """
        generator = _as_generator(rng)
        total = 0
        depth_prev = 0

        while depth_prev < depth_total:
            # Add a random exponential increment
            total += generator.exponential(1.0)

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

    def __call__(self, profile, rng=None):
        """Calculated a varied thickness profile.

        Parameters
        ----------
        profile : site.Profile
            Profile to be varied. Not modified in place.
        rng : None, int, or numpy.random.Generator, optional
            Generator used to draw the layering. Defaults to the module-level
            generator.

        Returns
        -------
        site.Profile
            Varied site profile.
        """
        layers = []
        for thick, depth_mid in self.iter_thickness(profile[-2].depth_base, rng=rng):
            # Locate the proper layer and add it to the model
            for layer in profile:
                if layer.depth < depth_mid <= layer.depth_base:
                    layers.append(
                        site.Layer(
                            layer.soil_type,
                            thick,
                            layer.initial_shear_vel,
                            layer.damping_min,
                            layer.poissons_ratio,
                        )
                    )
                    break
            else:
                raise LookupError

        # Add the half-space
        hsl = profile[-1]
        layers.append(
            site.Layer(
                hsl.soil_type,
                0,
                hsl.initial_shear_vel,
                poissons_ratio=hsl.poissons_ratio,
            )
        )

        varied = site.Profile(layers, profile.wt_depth)
        return varied


class HalfSpaceDepthVariation:
    """Vary the depth of the half-space.

    The total depth of each realization is drawn directly from *dist*: the
    profile is truncated when the draw is shallower than the seed profile, and
    the deepest soil layer is replicated when it is deeper.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous
        Frozen distribution of the half-space depth [m].
    """

    def __init__(self, dist: stats.rv_continuous):
        self._dist = dist

    @property
    def dist(self) -> stats.rv_continuous:
        """Distribution of the half-space depth."""
        return self._dist

    def depth_limit(self, quantile: float = 0.999) -> float:
        """Half-space depth at the specified quantile.

        Because the total depth is drawn directly from :attr:`dist`, this is
        the corresponding quantile of the realized profile depth. Note that an
        unbounded distribution (e.g. ``norm``) has no finite maximum, so this
        is a quantile rather than a hard limit.

        Parameters
        ----------
        quantile : float, optional
            Quantile in ``(0, 1)``.

        Returns
        -------
        float
            Depth of the half-space [m].
        """
        return float(self._dist.ppf(quantile))

    def __call__(self, profile: site.Profile, rng=None) -> site.Profile:
        """Calculate a profile with a varied half-space depth.

        Parameters
        ----------
        profile : site.Profile
            Profile to be varied. Not modified in place.
        rng : None, int, or numpy.random.Generator, optional
            Generator used to draw the depth. Defaults to the module-level
            generator.

        Returns
        -------
        site.Profile
            Varied site profile.
        """
        # Update the distribution with the central value of the profile
        varied_depth = self._dist.rvs(random_state=_as_generator(rng))

        if varied_depth <= 0:
            raise ValueError(
                f"Sampled a half-space depth of {varied_depth:.3f} m. The "
                "distribution must be limited to positive depths."
            )

        # Find the layer
        index, depth_within = profile.lookup_depth(varied_depth)
        half_space = profile[-1]

        if index < (len(profile) - 1):
            # Variation is within the layers
            layers = [layer.copy() for layer in profile[: (index + 1)]]
            # Reduce the thickness of the layer above the half-space
            layers[-1]._thickness = depth_within
        else:
            # Variation extends past the depth of the model. Sub-divide the
            # added thickness so that no layer is thicker than the original.
            orig_thick = profile[-2].thickness
            total_thick = orig_thick + depth_within
            count = max(int(np.ceil(total_thick / orig_thick)), 1)

            thick = total_thick / count

            # Don't copy half-space
            layers = [layer.copy() for layer in profile[:-1]]
            # Don't call the setter function as it needs a profile defined
            layers[-1]._thickness = thick
            parent = layers[-1]
            for _ in range(count - 1):
                layers.append(parent.copy())

        layers.append(half_space)

        return site.Profile(layers, profile.wt_depth)


class LayerThicknessVariation:
    def __init__(
        self,
        models: list[stats.rv_continuous] | dict[int, stats.rv_continuous],
        discretize_kwds: dict[str, float] | None = None,
    ) -> None:
        self._models = models
        self._discretize_kwds = discretize_kwds


class VelocityVariation:
    """Abstract model for varying the velocity."""

    def __init__(self, vary_bedrock=False):
        self._vary_bedrock = vary_bedrock

    def __call__(self, profile: site.Profile, rng=None) -> site.Profile:
        """Calculate a varied shear-wave velocity profile.

        Parameters
        ----------
        profile : site.Profile
            Profile to be varied. Not modified in place.
        rng : None, int, or numpy.random.Generator, optional
            Generator used to draw the velocities. Defaults to the module-level
            generator.

        Returns
        -------
        site.Profile
            Varied site profile.
        """

        mean = np.log(profile.initial_shear_vel)
        covar = self._calc_covar_matrix(profile)

        ln_vel_rand = _as_generator(rng).multivariate_normal(
            mean, covar, check_valid="ignore"
        )

        # Limits based on the number of standard deviations
        offset = randnorm.limit * np.sqrt(np.diag(covar))
        ln_vel = np.clip(ln_vel_rand, mean - offset, mean + offset)
        vel = np.exp(ln_vel)

        varied = profile.copy()
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
    r"""Toro (1995) [T95] velocity variation model.

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
    vary_bedrock: bool, optional
        If the velocity of the bedrock (half-space) should be varied.
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
        """Compute the adjacent-layer correlations.

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
        # Here the thickness is limited to 100 m to prevent underflow on the
        # exponent. For this thickness, the correlation will be minor
        thick = np.minimum(np.diff(depth), 100)

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

    @staticmethod
    def usgs_site_class(vs30):
        return "USGS " + np.array(list("DCBA"))[np.searchsorted([180, 360, 750], vs30)]


class DepthDependToroVelVariation(ToroVelocityVariation):
    r"""Toro (1995) [T95] velocity variation model modified for a depth dependent
    standard deviation that can be overridden by the soil_type name.

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
    vary_bedrock: bool, optional
        If the velocity of the bedrock (half-space) should be varied.
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
        vary_bedrock: bool = False,
        ln_std_map: dict[str, float] | None = None,
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
            for i, layer in enumerate(profile):
                if name in layer.soil_type.name:
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


class SoilTypeVariation:
    """Base class for soil-type (modulus-reduction and damping) variation.

    Parameters
    ----------
    correlation : float
        Correlation coefficient between the modulus-reduction and damping
        random variables.
    limits_mod_reduc : list[float], optional
        ``[min, max]`` clipping bounds for modulus reduction.
    limits_damping : list[float], optional
        ``[min, max]`` clipping bounds for damping.
    vary_bedrock : bool, optional
        Whether to include the half-space in the variation.
    sample_mode : {'random', 'fixed_percentiles'}, optional
        How samples are drawn when :meth:`iter_varied_profiles` iterates:

        * ``'random'`` *(default)* — each call draws an independent
          random realisation from the truncated-normal distribution.
        * ``'fixed_percentiles'`` — each call uses a pre-specified
          percentile supplied via ``sample_index`` so that the same
          index always produces an identical realisation.  The percentile
          is selected as ``percentiles[sample_index % len(percentiles)]``,
          so the list cycles when *count* is a multiple of its length.
          The caller must pass the ``sample_index`` keyword argument to
          :meth:`__call__`, and :func:`iter_varied_profiles` does this
          automatically.
    percentiles : list[float] | None, optional
        Ordered sequence of quantiles in ``(0, 1)`` used in
        ``'fixed_percentiles'`` mode.  The sequence cycles: iteration *i*
        draws ``percentiles[i % len(percentiles)]``.  Required (and only
        used) when *sample_mode* is ``'fixed_percentiles'``.
    """

    def __init__(
        self,
        correlation,
        limits_mod_reduc=[0.05, 1],
        limits_damping=[0, 0.15],
        vary_bedrock=False,
        sample_mode="random",
        percentiles=None,
    ):
        _valid_modes = {"random", "fixed_percentiles"}
        if sample_mode not in _valid_modes:
            raise ValueError(
                f"sample_mode must be one of {_valid_modes!r}, got {sample_mode!r}"
            )
        if sample_mode == "fixed_percentiles":
            if percentiles is None or len(percentiles) == 0:
                raise ValueError(
                    "percentiles must be a non-empty sequence when "
                    "sample_mode='fixed_percentiles'"
                )
            percentiles = list(percentiles)
            if any(not (0.0 < p < 1.0) for p in percentiles):
                raise ValueError(
                    "All percentile values must be strictly between 0 and 1"
                )
        self._vary_bedrock = vary_bedrock
        self._correlation = correlation
        self._limits_mod_reduc = list(limits_mod_reduc)
        self._limits_damping = list(limits_damping)
        self._sample_mode = sample_mode
        self._percentiles = percentiles

    def __call__(self, soil_type, sample_index=None, rng=None):
        """Return a single varied realisation of *soil_type*.

        Parameters
        ----------
        soil_type : site.SoilType
            The nominal (seed) soil type to vary.
        sample_index : int | None, optional
            Index into :attr:`percentiles` used when
            ``sample_mode='fixed_percentiles'``.  Ignored in
            ``'random'`` mode.  Must be provided (and within range) in
            ``'fixed_percentiles'`` mode.
        rng : None, int, or numpy.random.Generator, optional
            Generator used to draw the variates. Ignored in
            ``'fixed_percentiles'`` mode, which is deterministic. Defaults to
            the module-level generator.
        """

        def get_values(nlp):
            try:
                return nlp.values
            except AttributeError:
                return np.asarray(nlp).astype(float)

        mod_reduc = get_values(soil_type.mod_reduc)
        damping = get_values(soil_type.damping)

        # A pair of correlated random variables
        if self._sample_mode == "fixed_percentiles":
            if sample_index is None:
                raise ValueError(
                    "sample_index must be provided when sample_mode='fixed_percentiles'"
                )
            percentile = self._percentiles[sample_index]
            randvar = randnorm.correlated_at_percentile(self.correlation, percentile)
        else:
            randvar = randnorm.correlated(self.correlation, rng=rng)

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

    def vary_profile(
        self, profile: site.Profile, sample_index: int | None = None, rng=None
    ):
        """Return a profile with varied soil types.

        Parameters
        ----------
        profile : site.Profile
            Input profile to vary.
        sample_index : int | None, optional
            Index into :attr:`percentiles` for ``sample_mode='fixed_percentiles'``.
            Ignored in ``'random'`` mode.
        rng : None, int, or numpy.random.Generator, optional
            Generator used to draw the variates. Defaults to the module-level
            generator.
        """
        # Map of varied soil types
        varied = {
            str(st): self(st, sample_index=sample_index, rng=rng)
            for st in profile.iter_soil_types()
        }

        # Create new layers
        end = None if self.vary_bedrock else -1
        layers = [
            site.Layer(
                varied[str(layer.soil_type)],
                layer.thickness,
                layer.initial_shear_vel,
                layer.damping_min,
                layer.poissons_ratio,
            )
            for layer in profile[:end]
        ]

        # Add the unrandomized bedrock
        if not self.vary_bedrock:
            layers.append(profile[-1])

        return site.Profile(layers, profile.wt_depth)

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

    @property
    def sample_mode(self):
        """Sampling mode: ``'random'`` or ``'fixed_percentiles'``."""
        return self._sample_mode

    @property
    def percentiles(self):
        """Percentile list used in ``'fixed_percentiles'`` mode, or *None*."""
        return self._percentiles


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
    """Variation defined by the EPRI SPID (2013) and documented in PNNL (2014).

    EPRI SPID (2013): https://www.nrc.gov/docs/ML1233/ML12333A170.pdf
    """

    def __init__(
        self,
        correlation,
        limits_mod_reduc=[0, 1],
        limits_damping=[0, 0.15],
        std_mod_reduc=0.15,
        std_damping=0.30,
        sample_mode="random",
        percentiles=None,
    ):
        super().__init__(
            correlation,
            limits_mod_reduc,
            limits_damping,
            sample_mode=sample_mode,
            percentiles=percentiles,
        )
        self._std_mod_reduc = std_mod_reduc
        self._std_damping = std_damping

    def _get_varied(self, randvar, mod_reduc, damping):
        # Vary the G/Gmax in transformed space.

        # PNNL (2014) Hanford Site Wide Hazard Study is available here:
        # https://www.hanford.gov/files.cfm/00_Front_Matter.pdf
        # https://www.hanford.gov/files.cfm/9.0_Ground_Motion_Characterization.pdf

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


class DispersionCheck:
    """Accept/reject a profile based on its surface-wave dispersion curve.

    The check computes a z-score at each frequency assuming a log-normal
    distribution:

    .. math::

        z_i = \\frac{\\ln(V_i / V_{\\text{target},i})}{\\sigma_{\\ln,i}}

    A profile is accepted when ``max(|z|) <= max_z_score``.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    target : array_like
        Target median dispersion velocity [m/s].
    ln_std : array_like
        Log-standard deviation of dispersion velocity at each frequency.
    max_z_score : float
        Maximum allowable absolute z-score.
    wave : str, optional
        Wave type passed to :meth:`~pystrata.site.Profile.calc_dispersion`.
    mode : int, optional
        Mode number (0 = fundamental).
    dc_type : str, optional
        ``"phase"`` or ``"group"``.
    """

    @convert_units(freqs="hertz", target="meter / second")
    def __init__(
        self,
        freqs,
        target,
        ln_std,
        max_z_score,
        wave="rayleigh",
        mode=0,
        dc_type="phase",
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.ln_std = np.asarray(ln_std, dtype=float)
        self.max_z_score = max_z_score
        self.wave = wave
        self.mode = mode
        self.dc_type = dc_type

    def __call__(self, profile: site.Profile) -> bool:
        try:
            velocity = profile.calc_dispersion(
                self.freqs, wave=self.wave, mode=self.mode, dc_type=self.dc_type
            )
        except Exception:
            # Reject profiles where dispersion computation fails
            return False
        z = np.log(velocity / self.target) / self.ln_std
        return float(np.max(np.abs(z))) <= self.max_z_score


def varied_profile(
    profile: site.Profile,
    index: int,
    seed: int | None = None,
    var_depth: HalfSpaceDepthVariation | None = None,
    var_thickness: ToroThicknessVariation | None = None,
    var_velocity: VelocityVariation | None = None,
    var_soiltypes: SoilTypeVariation | None = None,
) -> site.Profile:
    """Generate a single realization of a varied profile.

    Realization *index* is fully determined by *seed*, so it can be generated
    without producing the realizations before it. This is what allows an
    ensemble to be split across processes.

    Parameters
    ----------
    profile : site.Profile
        Seed profile. Not modified in place.
    index : int
        Zero-based realization index.
    seed : int or None
        Base seed. When ``None`` the module-level generator is used and the
        result is not reproducible.
    var_depth, var_thickness, var_velocity, var_soiltypes
        Variation models, applied in that order.

    Returns
    -------
    site.Profile
        Varied profile.

    See Also
    --------
    iter_varied_profiles : Generate a sequence of realizations.
    """
    rng = (
        None
        if seed is None
        else np.random.default_rng(np.random.SeedSequence([seed, index]))
    )

    sample_index = None
    if var_soiltypes and var_soiltypes.sample_mode == "fixed_percentiles":
        sample_index = index % len(var_soiltypes.percentiles)

    return _vary(
        profile,
        rng,
        var_depth,
        var_thickness,
        var_velocity,
        var_soiltypes,
        sample_index,
    )


def _vary(
    profile, rng, var_depth, var_thickness, var_velocity, var_soiltypes, sample_index
):
    """Apply the variation models to a copy of *profile*."""
    varied = profile.copy()

    if var_depth:
        varied = var_depth(varied, rng=rng)

    if var_thickness:
        varied = var_thickness(varied, rng=rng)

    if var_velocity:
        varied = var_velocity(varied, rng=rng)

    if var_soiltypes:
        varied = var_soiltypes.vary_profile(varied, sample_index=sample_index, rng=rng)

    return varied


def iter_varied_profiles(
    profile: site.Profile,
    count: int,
    var_thickness: ToroThicknessVariation | None = None,
    var_velocity: VelocityVariation | None = None,
    var_soiltypes: SoilTypeVariation | None = None,
    check: None | Callable = None,
    max_attempts: int | None = None,
    var_depth: HalfSpaceDepthVariation | None = None,
    seed: int | None = None,
) -> Generator[site.Profile]:
    """Iterate over simulated profiles.

    Parameters
    ----------
    profile : site.Profile
        seed profile
    count : int
        number of interations
    var_thickness : ToroThicknessVariation | None
        model for the thickness variation
    var_velocity : VelocityVariation | None
        model for the velocity variation
    var_soiltypes : SoilTypeVariation | None
        model for the soil type variation.  When ``sample_mode='fixed_percentiles'``
        *count* must be divisible by the number of configured percentiles; the
        percentile list cycles so that each percentile is used
        ``count // len(percentiles)`` times in order.
    check : callable or None
        Optional callable that receives a :class:`~pystrata.site.Profile` and
        returns ``True`` if it should be yielded.  When the check returns
        ``False`` the profile is discarded and a new one is generated.
    max_attempts : int or None
        Maximum number of profile generations to attempt.  Defaults to
        ``count * 100`` when *check* is provided.  Ignored when *check* is
        ``None``.
    var_depth : HalfSpaceDepthVariation | None
        model for the half-space depth variation.  Applied before the
        thickness variation, since it changes the total depth of the profile.
    seed : int or None
        When provided, each realization draws from a generator derived from
        ``SeedSequence([seed, attempt])``.  A given realization is then
        reproducible independent of how many realizations are requested or the
        order in which they are consumed, which is what allows an ensemble to
        be distributed across processes.  When ``None``, the module-level
        generator is used and the sequence is not reproducible.

    Returns
    -------
    site.Profile
        varied profile
    """
    if var_soiltypes and var_soiltypes.sample_mode == "fixed_percentiles":
        n_pct = len(var_soiltypes.percentiles)
        if count % n_pct != 0:
            raise ValueError(
                f"count ({count}) must be divisible by the number of percentiles "
                f"({n_pct}) when sample_mode='fixed_percentiles'"
            )

    if check is not None and max_attempts is None:
        max_attempts = count * 100

    yielded = 0
    attempts = 0
    i = 0
    while yielded < count:
        if check is not None and attempts >= max_attempts:
            raise RuntimeError(
                f"Exceeded {max_attempts} attempts to generate {count} profiles "
                f"({yielded} accepted so far). Consider relaxing the check criteria."
            )

        # Derive a generator for this attempt. Seeding on the attempt counter
        # -- rather than on the number accepted -- keeps rejected realizations
        # from shifting the stream of those that follow.
        rng = (
            None
            if seed is None
            else np.random.default_rng(np.random.SeedSequence([seed, attempts]))
        )

        # Determine the sample_index to forward (None in random mode)
        sample_index = None
        if var_soiltypes and var_soiltypes.sample_mode == "fixed_percentiles":
            sample_index = i % len(var_soiltypes.percentiles)

        _profile = _vary(
            profile,
            rng,
            var_depth,
            var_thickness,
            var_velocity,
            var_soiltypes,
            sample_index,
        )

        i += 1
        attempts += 1

        if check is not None and not check(_profile):
            continue

        yielded += 1
        yield _profile
