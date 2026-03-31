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

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
import numpy as np
import numpy.typing as npt
import pykooh
from scipy.optimize import minimize

from .motion import Motion, WaveField
from .site import Layer, Location, Profile
from .units import GRAVITY


class AbstractCalculator:
    def __init__(self):
        self._loc_input: None | Location = None
        self._motion: None | Motion = None
        self._profile: None | Profile = None

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
        self._motion = motion
        self._profile = profile
        self._loc_input = loc_input

        if reset_layers:
            # Set initial properties
            for layer in profile:
                layer.reset()
                if layer.strain is None:
                    layer.strain = 0.0

    @property
    def motion(self):
        return self._motion

    @property
    def profile(self):
        return self._profile

    @property
    def loc_input(self):
        return self._loc_input


@numba.jit(nopython=True)
def _my_trapz_impl(thickness, prop, depth_max):
    total = 0
    depth = 0

    for t, p in zip(thickness, prop):
        depth += t
        if depth_max < depth:
            # Partial layer
            total += (t - (depth - depth_max)) * p
            break
        total += t * p
    else:
        # Final infinite layer
        total += (depth_max - depth) * p

    return total / depth_max


def _my_trapz_python(thickness, prop, depth_max):
    total = 0
    depth = 0

    for t, p in zip(thickness, prop):
        depth += t
        if depth_max < depth:
            # Partial layer
            total += (t - (depth - depth_max)) * p
            break
        total += t * p
    else:
        # Final infinite layer
        total += (depth_max - depth) * p

    return total / depth_max


my_trapz = _my_trapz_impl if HAS_NUMBA else _my_trapz_python


def _calc_waves_python(angular_freqs, comp_shear_vels, comp_shear_mods, thicknesses):
    """Pure-Python/numpy implementation of wave propagation.

    Parameters
    ----------
    angular_freqs : np.ndarray, shape (n_freqs,)
    comp_shear_vels : np.ndarray, shape (n_layers, n_freqs)
    comp_shear_mods : np.ndarray, shape (n_layers, n_freqs)
    thicknesses : np.ndarray, shape (n_layers,)
    """
    n_layers = comp_shear_vels.shape[0]
    n_freqs = len(angular_freqs)

    wave_nums = np.empty((n_layers, n_freqs), dtype=complex)
    for i in range(n_layers):
        wave_nums[i, :] = angular_freqs / comp_shear_vels[i, :]

    waves_a = np.ones_like(wave_nums, dtype=complex)
    waves_b = np.ones_like(wave_nums, dtype=complex)
    for i in range(n_layers - 1):
        with np.errstate(invalid="ignore"):
            cimped = (wave_nums[i] * comp_shear_mods[i, :]) / (
                wave_nums[i + 1] * comp_shear_mods[i + 1, :]
            )

        cterm = 1j * wave_nums[i, :] * thicknesses[i]

        waves_a[i + 1, :] = 0.5 * waves_a[i] * (1 + cimped) * np.exp(
            cterm
        ) + 0.5 * waves_b[i] * (1 - cimped) * np.exp(-cterm)
        waves_b[i + 1, :] = 0.5 * waves_a[i] * (1 - cimped) * np.exp(
            cterm
        ) + 0.5 * waves_b[i] * (1 + cimped) * np.exp(-cterm)

        mask = ~np.isfinite(cimped)
        waves_a[i + 1, mask] = 1.0
        waves_b[i + 1, mask] = 1.0

    mask = np.isclose(angular_freqs, 0)
    waves_a[-1, mask] = 1.0
    waves_b[-1, mask] = 1.0

    return waves_a, waves_b, wave_nums


def _wave_at_location_python(
    wave_nums_row, depth_within, waves_a_row, waves_b_row, wave_field_code
):
    """Pure-Python/numpy implementation of wave_at_location."""
    cterm = 1j * wave_nums_row * depth_within
    exp_pos = np.exp(cterm)
    if wave_field_code == 1:  # WaveField.within
        return waves_a_row * exp_pos + waves_b_row * np.exp(-cterm)
    elif wave_field_code == 0:  # WaveField.outcrop
        return 2.0 * waves_a_row * exp_pos
    else:  # WaveField.incoming_only
        return waves_a_row * exp_pos


def _calc_strain_tf_python(
    wave_nums_out,
    waves_a_out,
    waves_b_out,
    depth_within_out,
    ang_freqs,
    wave_nums_in,
    waves_a_in,
    waves_b_in,
    depth_within_in,
    wave_field_code_in,
    gravity,
):
    """Pure-Python/numpy implementation of calc_strain_tf."""
    cterm_out = 1j * wave_nums_out * depth_within_out
    numer = (
        1j
        * wave_nums_out
        * (waves_a_out * np.exp(cterm_out) - waves_b_out * np.exp(-cterm_out))
    )

    wave_in = _wave_at_location_python(
        wave_nums_in, depth_within_in, waves_a_in, waves_b_in, wave_field_code_in
    )
    denom = -(ang_freqs**2) * wave_in

    mask = ~np.isclose(ang_freqs, 0)
    tf = np.zeros_like(mask, dtype=complex)
    tf[mask] = gravity * numer[mask] / denom[mask]
    return tf


if HAS_NUMBA:

    @numba.njit(cache=True)
    def _calc_waves_numba(angular_freqs, comp_shear_vels, comp_shear_mods, thicknesses):
        """Numba-accelerated wave propagation computation.

        Parameters
        ----------
        angular_freqs : np.ndarray, shape (n_freqs,)
        comp_shear_vels : np.ndarray, shape (n_layers, n_freqs)
        comp_shear_mods : np.ndarray, shape (n_layers, n_freqs)
        thicknesses : np.ndarray, shape (n_layers,)
        """
        n_layers = comp_shear_vels.shape[0]
        n_freqs = len(angular_freqs)

        wave_nums = np.empty((n_layers, n_freqs), dtype=np.complex128)
        for i in range(n_layers):
            for j in range(n_freqs):
                wave_nums[i, j] = angular_freqs[j] / comp_shear_vels[i, j]

        waves_a = np.ones((n_layers, n_freqs), dtype=np.complex128)
        waves_b = np.ones((n_layers, n_freqs), dtype=np.complex128)
        for i in range(n_layers - 1):
            for j in range(n_freqs):
                denom = wave_nums[i + 1, j] * comp_shear_mods[i + 1, j]
                if denom == 0:
                    waves_a[i + 1, j] = 1.0
                    waves_b[i + 1, j] = 1.0
                    continue

                cimped = (wave_nums[i, j] * comp_shear_mods[i, j]) / denom
                cterm = 1j * wave_nums[i, j] * thicknesses[i]
                exp_pos = np.exp(cterm)
                exp_neg = np.exp(-cterm)

                if np.isfinite(cimped.real) and np.isfinite(cimped.imag):
                    waves_a[i + 1, j] = (
                        0.5 * waves_a[i, j] * (1 + cimped) * exp_pos
                        + 0.5 * waves_b[i, j] * (1 - cimped) * exp_neg
                    )
                    waves_b[i + 1, j] = (
                        0.5 * waves_a[i, j] * (1 - cimped) * exp_pos
                        + 0.5 * waves_b[i, j] * (1 + cimped) * exp_neg
                    )
                else:
                    waves_a[i + 1, j] = 1.0
                    waves_b[i + 1, j] = 1.0

        # Set wave amplitudes to 1 at frequencies near 0
        for j in range(n_freqs):
            if abs(angular_freqs[j]) < 1e-8:
                waves_a[n_layers - 1, j] = 1.0
                waves_b[n_layers - 1, j] = 1.0

        return waves_a, waves_b, wave_nums

    @numba.njit(cache=True)
    def _wave_at_location_numba(
        wave_nums_row, depth_within, waves_a_row, waves_b_row, wave_field_code
    ):
        """Numba-accelerated wave_at_location computation."""
        n = len(wave_nums_row)
        result = np.empty(n, dtype=np.complex128)
        for j in range(n):
            cterm = 1j * wave_nums_row[j] * depth_within
            exp_pos = np.exp(cterm)
            if wave_field_code == 1:  # WaveField.within
                result[j] = waves_a_row[j] * exp_pos + waves_b_row[j] * np.exp(-cterm)
            elif wave_field_code == 0:  # WaveField.outcrop
                result[j] = 2.0 * waves_a_row[j] * exp_pos
            else:  # WaveField.incoming_only
                result[j] = waves_a_row[j] * exp_pos
        return result

    @numba.njit(cache=True)
    def _calc_strain_tf_numba(
        wave_nums_out,
        waves_a_out,
        waves_b_out,
        depth_within_out,
        ang_freqs,
        wave_nums_in,
        waves_a_in,
        waves_b_in,
        depth_within_in,
        wave_field_code_in,
        gravity,
    ):
        """Numba-accelerated strain transfer function computation."""
        n = len(ang_freqs)
        tf = np.zeros(n, dtype=np.complex128)
        for j in range(n):
            if abs(ang_freqs[j]) < 1e-8:
                continue
            # Numerator: strain at output location (A - B form)
            cterm_out = 1j * wave_nums_out[j] * depth_within_out
            exp_pos_out = np.exp(cterm_out)
            exp_neg_out = np.exp(-cterm_out)
            numer = (
                1j
                * wave_nums_out[j]
                * (waves_a_out[j] * exp_pos_out - waves_b_out[j] * exp_neg_out)
            )
            # Denominator: wave at input location
            cterm_in = 1j * wave_nums_in[j] * depth_within_in
            exp_pos_in = np.exp(cterm_in)
            if wave_field_code_in == 1:  # within
                wave_in = waves_a_in[j] * exp_pos_in + waves_b_in[j] * np.exp(-cterm_in)
            elif wave_field_code_in == 0:  # outcrop
                wave_in = 2.0 * waves_a_in[j] * exp_pos_in
            else:  # incoming_only
                wave_in = waves_a_in[j] * exp_pos_in

            denom = -(ang_freqs[j] ** 2) * wave_in
            tf[j] = gravity * numer / denom
        return tf

    _calc_waves_dispatch = _calc_waves_numba
    _wave_at_location_dispatch = _wave_at_location_numba
    _calc_strain_tf_dispatch = _calc_strain_tf_numba
else:
    _calc_waves_dispatch = _calc_waves_python
    _wave_at_location_dispatch = _wave_at_location_python
    _calc_strain_tf_dispatch = _calc_strain_tf_python


class QuarterWaveLenCalculator(AbstractCalculator):
    """Compute quarter-wave length site amplification.

    No consideration for nolninearity is made by this calculator.
    """

    name = "QWL"

    def __init__(self, site_atten=None, method="standard"):
        super().__init__()
        self._site_atten = site_atten
        self._method = method

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
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
        super().__call__(motion, profile, loc_input, reset_layers=reset_layers, **kwds)

        self._crustal_amp, self._site_term = self._calc_amp(
            profile.density, profile.thickness, profile.slowness
        )

    @staticmethod
    def correction_ba23(x):
        a = 0.560
        b = -1.301
        s = 1.398
        d = 4.000
        e = 6.000
        g = 2.000
        h = 0.760
        p = 3.000
        q = 0.333

        fact = (x - b) / s

        eta = (a * fact**d) / ((1 - fact**e) ** g + h * fact**p) ** q

        return eta

    @property
    def method(self) -> str:
        return self._method

    @property
    def crustal_amp(self) -> np.ndarray:
        return self._crustal_amp

    @property
    def site_term(self) -> np.ndarray:
        return self._site_term

    @property
    def site_atten(self) -> float | None:
        return self._site_atten

    def _calc_amp(
        self, density: npt.ArrayLike, thickness: npt.ArrayLike, slowness: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = self.motion.freqs
        # 1/4 wavelength depth -- estimated for mean slowness
        qwl_depth = 1 / (4 * np.mean(slowness) * freqs)

        def qwl_average(param):
            return np.array([my_trapz(thickness, param, qd) for qd in qwl_depth])

        for _ in range(50):
            qwl_slowness = qwl_average(slowness)
            prev_qwl_depth = qwl_depth

            # Compute the mean between the previous depths and the newly
            # computed depths. If the new value is just taken, then this
            # algorithm can osccilate between two solutions.
            qwl_depth = np.mean(
                np.c_[prev_qwl_depth, 1 / (4 * qwl_slowness * freqs)], axis=1
            )
            if np.allclose(prev_qwl_depth, qwl_depth, rtol=0.01):
                break
        else:
            raise RuntimeError("QWL calcuation did not converge.")

        qwl_density = qwl_average(density)

        if self.method == "standard":
            eta = 0.5
        elif self.method == "ba23":
            total_depth = np.sum(thickness[:-1])
            total_slow = my_trapz(thickness, slowness, total_depth)
            freq_bot = 1.0 / (4 * total_depth * total_slow)
            eta = self.correction_ba23(np.log10(freqs / freq_bot))
        else:
            raise NotImplementedError

        crustal_amp = (
            (density[-1] / slowness[-1]) / (qwl_density / qwl_slowness)
        ) ** eta

        site_term = np.array(crustal_amp)
        if self.site_atten:
            site_term *= np.exp(-np.pi * self.site_atten * freqs)

        return crustal_amp, site_term

    def fit(
        self,
        target_type,
        target,
        adjust_thickness=False,
        adjust_site_atten=False,
        adjust_source_vel=False,
    ):
        """Fit to a target crustal amplification or site term.

        The fitting process adjusts the velocity, site attenuation, and layer
        thickness (if enabled) to fit a target values. The frequency range is
        specified by the input motion.

        Parameters
        ----------
        target_type: str
            Options are 'crustal_amp' to only fit to the crustal amplification,
             or 'site_term' to fit both the velocity and the site attenuation
             parameter.
        target: `array_like`
            Target values.
        adjust_thickness: bool (optional)
            If the thickness of the layers is adjusted as well, default: False.
        adjust_site_atten: bool (optional)
            If the site attenuation is adjusted as well, default: False.
        adjust_source_vel: bool (optional)
            If the source velocity should be adjusted, default: False.
        Returns
        -------
        profile: `pyrsa.site.Profile`
            profile optimized to fit a target amplification.
        """
        density = self.profile.density

        nl = len(density)

        # Slowness bounds
        slowness = self.profile.slowness
        thickness = self.profile.thickness
        site_atten = self._site_atten

        # Slowness
        initial = slowness
        bounds = 1 / np.tile((4000, 100), (nl, 1))
        if not adjust_source_vel:
            bounds[-1] = (initial[-1], initial[-1])

        # Thickness bounds
        if adjust_thickness:
            bounds = np.r_[bounds, [[t / 2, 2 * t] for t in thickness]]
            initial = np.r_[initial, thickness]

        # Site attenuation bounds
        if adjust_site_atten:
            bounds = np.r_[bounds, [[0.0001, 0.200]]]
            initial = np.r_[initial, self.site_atten]

        def calc_rmse(this, that):
            return np.mean(((this - that) / that) ** 2)

        def err(x):
            _slowness = x[0:nl]
            if adjust_thickness:
                _thickness = x[nl : (2 * nl)]
            else:
                _thickness = thickness
            if adjust_site_atten:
                self._site_atten = x[-1]

            crustal_amp, site_term = self._calc_amp(density, _thickness, _slowness)

            calc = crustal_amp if target_type == "crustal_amp" else site_term

            err = 10 * calc_rmse(target, calc)
            # Prefer the original values so add the difference to the error
            err += calc_rmse(slowness, _slowness)
            if adjust_thickness:
                err += calc_rmse(thickness, _thickness)
            if adjust_site_atten:
                err += calc_rmse(self._site_atten, site_atten)
            return err

        res = minimize(err, initial, method="L-BFGS-B", bounds=bounds)

        slowness = res.x[0:nl]
        if adjust_thickness:
            thickness = res.x[nl : (2 * nl)]

        profile = Profile(
            [
                Layer(layer.soil_type, thick, 1 / slow)
                for layer, thick, slow in zip(self.profile, thickness, slowness)
            ],
            self.profile.wt_depth,
        )
        # Update the calculated amplificaiton
        self(self.motion, profile, self.loc_input)


class LinearElasticCalculator(AbstractCalculator):
    """Class for performing linear elastic site response."""

    name = "LE"

    def __init__(self):
        super().__init__()

        self._waves_a = np.array([])
        self._waves_b = np.array([])
        self._wave_nums = np.array([])

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
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
        super().__call__(motion, profile, loc_input, reset_layers=reset_layers, **kwds)

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
        n_layers = len(profile)
        n_freqs = len(angular_freqs)

        # Extract layer properties into 2D arrays (n_layers × n_freqs).
        # This handles both scalar (LE/EQL) and vector (FDM) layer properties
        # via numpy broadcasting.
        comp_shear_vels = np.empty((n_layers, n_freqs), dtype=complex)
        comp_shear_mods = np.empty((n_layers, n_freqs), dtype=complex)
        for i, layer in enumerate(profile):
            comp_shear_vels[i, :] = layer.comp_shear_vel
            comp_shear_mods[i, :] = layer.comp_shear_mod
        thicknesses = profile.thickness

        self._waves_a, self._waves_b, self._wave_nums = _calc_waves_dispatch(
            angular_freqs, comp_shear_vels, comp_shear_mods, thicknesses
        )

    def wave_at_location(self, loc: Location) -> np.ndarray:
        """Compute the wave field at specific location.

        Parameters
        ----------
        loc : site.Location
            :class:`site.Location` of the input

        Returns
        -------
        `np.ndarray`
            Amplitude and phase of waves
        """
        return _wave_at_location_dispatch(
            self._wave_nums[loc.index],
            loc.depth_within,
            self._waves_a[loc.index],
            self._waves_b[loc.index],
            loc.wave_field.value,
        )

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
            tf *= lout.layer.shear_mod

        return tf

    def calc_strain_tf(self, lin, lout):
        """Compute the strain transfer function from `lout` to `location_in`.

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

        ang_freqs = self.motion.angular_freqs
        return _calc_strain_tf_dispatch(
            self._wave_nums[lout.index, :],
            self._waves_a[lout.index, :],
            self._waves_b[lout.index, :],
            lout.depth_within,
            ang_freqs,
            self._wave_nums[lin.index, :],
            self._waves_a[lin.index, :],
            self._waves_b[lin.index, :],
            lin.depth_within,
            lin.wave_field.value,
            GRAVITY,
        )


class EquivalentLinearCalculator(LinearElasticCalculator):
    """Class for performing equivalent-linear elastic site response."""

    name = "EQL"

    def __init__(
        self, strain_ratio=0.65, tolerance=0.01, max_iterations=15, strain_limit=0.05
    ):
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

        strain_limit: float, default=0.05
            Limit of strain in calculations. If this strain is exceed, the
            iterative calculation is ended.
        """
        super().__init__()
        self._strain_ratio = strain_ratio
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._strain_limit = strain_limit

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
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
        super().__call__(motion, profile, loc_input, reset_layers=reset_layers, **kwds)

        if reset_layers:
            # Use the previously established layer strains
            self._estimate_strains()

        iteration = 0
        # The iteration at which strains were last limited
        limited_iter = -2
        limited_strains = False

        while iteration < self.max_iterations:
            limited_strains = False
            self._calc_waves(motion.angular_freqs, profile)

            for index, layer in enumerate(profile[:-1]):
                loc_layer = Location(index, layer, "within", layer.thickness / 2)

                # Compute the representative strain(s) within the layer. FDM
                #  will provide a vector of strains.
                strain = self._calc_strain(loc_input, loc_layer, motion)
                if self._strain_limit and np.any(strain > self._strain_limit):
                    limited_strains = True
                    strain = np.minimum(strain, self._strain_limit)
                layer.strain = strain

            # Maximum error (damping and shear modulus) over all layers
            max_error = max(profile.max_error)
            if max_error < self.tolerance:
                break

            # Break, if the strains were limited the last two iterations.
            if limited_strains:
                if limited_iter == (iteration - 1):
                    raise RuntimeError("Strain limit exceeded.")
                else:
                    limited_iter = iteration

            iteration += 1

        # Compute the maximum strain within the profile.
        for index, layer in enumerate(profile[:-1]):
            loc_layer = Location(index, layer, "within", layer.thickness / 2)
            layer.strain_max = self._calc_strain_max(loc_input, loc_layer, motion)

    def _estimate_strains(self):
        """Compute an estimate of the strains."""
        # Estimate the strain based on the PGV and shear-wave velocity
        for layer in self._profile:
            layer.reset()
            # PGV in units of cm/sec
            layer.strain = (self._motion.pgv / 100) / layer.initial_shear_vel

    @property
    def strain_ratio(self):
        return self._strain_ratio

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def strain_limit(self):
        return self._strain_limit

    @classmethod
    def calc_strain_ratio(cls, mag):
        """Compute the effective strain ratio using Idriss and Sun (1992).

        Parameters
        ----------
        mag: float
            Magnitude of the input motion.

        Returns
        -------
        strain_ratio : float
            Effective strain ratio

        References
        ----------
        .. [1] Idriss, I. M., & Sun, J. I. (1992). SHAKE91: A computer program
            for conducting equivalent linear seismic response analyses of
            horizontally layered soil deposits. Center for Geotechnical
            Modeling, Department of Civil and Environmental Engineering,
            University of California, Davis, CA.
        """
        return (mag - 1) / 10

    def _calc_strain(self, loc_input, loc_layer, motion, *args):
        """Compute the strain used for iterations of material properties."""
        strain_max = self._calc_strain_max(loc_input, loc_layer, motion, *args)
        return self.strain_ratio * strain_max

    def _calc_strain_max(self, loc_input, loc_layer, motion, *args):
        """Compute the effective strain at the center of a layer."""
        return motion.calc_peak(self.calc_strain_tf(loc_input, loc_layer))


class FrequencyDependentEqlCalculator(EquivalentLinearCalculator):
    """Class for performing equivalent-linear elastic site response with frequency-
    dependent modulii and damping.

    Parameters
    ----------
    method: str
        method for computing the strain spectrum:
         - ka02: use the Kausel & Assimaki (2002) defined shape for a  smooth spectrum
           for the strain.
         - zr15: use Zalachoris & Rathje (2015) approach of the strain
         spectrum
         - ko:##: use Konno-Omachi with a bandwith of ## to compute the smooth
         spectrum. The strain is then computed as a running maximum from high
         to low frequencies. A value of 20 or 30 is recommended based on
         limited studies.
    strain_ratio: float, default=1.00
        ratio between the maximum strain and effective strain used to compute
        strain compatible properties. There is not clear guidance the use of
        the effective strain ratio. For the `ka02` the recommended value is
        0.65 -- or consistent with an EQL approach. For `zr15` and `ko:##`, there is no
        clear guidance but a value of 1.0 might make sense.
    tolerance: float, default=0.01
        tolerance in the iterative properties, which would cause the iterative
        process to terminate.
    max_iterations: int, default=15
        maximum number of iterations to perform.

    strain_limit: float, default=0.05
        Limit of strain in calculations. If this strain is exceed, the
        iterative calculation is ended.

    References
    ----------
    .. [1] Kausel, E., & Assimaki, D. (2002). Seismic simulation of inelastic
        soils via frequency-dependent moduli and damping. Journal of
        Engineering Mechanics, 128(1), 34-47.
    """

    def __init__(
        self,
        method: str = "ka02",
        strain_ratio: float = 0.65,
        tolerance: float = 0.01,
        max_iterations: int = 15,
        strain_limit: float = 0.05,
    ):
        """Initialize the class."""
        super().__init__(strain_ratio, tolerance, max_iterations, strain_limit)

        self._method = method
        self._smoother = None

    @property
    def name(self):
        return f"FDM-{self.method}"

    @property
    def method(self):
        return self._method

    def _estimate_strains(self):
        """Estimate the strains by running an EQL site response.

        This step was recommended in Section 8.3.1 of Zalachoris (2014).
        """
        eql = EquivalentLinearCalculator()
        eql(self._motion, self._profile, self._loc_input)

    def _calc_strain(self, loc_input, loc_layer, motion, *args):
        freqs = np.array(motion.freqs)
        strain_tf = self.calc_strain_tf(loc_input, loc_layer)
        strain_fas = np.abs(strain_tf * motion.fourier_amps)
        # Maximum strain in the time domain modified by the effective strain
        # ratio
        strain_eff = self.strain_ratio * motion.calc_peak(strain_tf)

        if self._method == "ka02":
            # Equation (8)
            freq_avg = np.trapezoid(freqs * strain_fas, x=freqs) / np.trapezoid(
                strain_fas, x=freqs
            )

            # Find the average strain at frequencies less than the average
            # frequency
            # Equation (8)
            mask = freqs < freq_avg
            strain_avg = np.trapezoid(strain_fas[mask], x=freqs[mask]) / freq_avg

            # Normalize the frequency and strain by the average values
            freqs /= freq_avg
            strain_fas /= strain_avg

            # Fit the smoothed model at frequencies greater than the average
            A = np.c_[-freqs[~mask], -np.log(freqs[~mask])]
            a, b = np.linalg.lstsq(A, np.log(strain_fas[~mask]), rcond=None)[0]
            # This is a modification of the published method that ensures a
            # smooth transition in the strain. Make sure the frequencies are zero.
            shape = np.minimum(
                1,
                np.exp(-a * freqs)
                / np.maximum(np.finfo(float).eps, np.power(freqs, b)),
            )
            strains = strain_eff * shape
        elif self._method.startswith("ko:"):
            if self._smoother is None or not self._smoother.freqs_match(motion.freqs):
                bandwidth = float(self._method[3:])
                self._smoother = pykooh.CachedSmoother(
                    motion.freqs, motion.freqs, bandwidth=bandwidth, normalize=True
                )

            # Konno-Omachi smoothing
            strain_fas_sm = self._smoother(strain_fas)
            strains = strain_eff * strain_fas_sm / np.max(strain_fas_sm)

            strains[::-1] = np.maximum.accumulate(strains[::-1])
        else:
            strains = strain_eff * strain_fas / np.max(strain_fas)

        return strains
