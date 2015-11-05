from typing import List

import numpy as np

from .site import Profile, Location
from .motion import Motion


class LinearElasticCalculator(object):
    def __init__(self):
        self._waves_a = np.array([])
        self._waves_b = np.array([])
        self._wave_nums = np.array([])

        self._input_location = None

    def calc_waves(self,
                   motion: Motion,
                   profile: Profile,
                   input_location: Location):
        """Perform the wave propagation.

        Parameters
        ----------
        motion: :class:`~.base.motion.Motion`
            Input motion.

        profile: :class:`~.base.site.Profile`
            Site profile.

        input_location: :class:`~.base.site.Location`
            Location of the input motion.
        """
        # Set intial properties
        for l in profile:
            if l.strain is None:
                l.strain = 0.

        # Compute the complex wave numbers of the system
        wave_nums = np.empty((len(profile), len(motion.freqs)), np.complex)
        for i, l in enumerate(profile):
            wave_nums[i, :] = motion.angular_freqs / l.comp_shear_vel

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

        self._waves_a = waves_a
        self._waves_b = waves_b
        self._wave_nums = wave_nums

    def wave_at_location(self, l: Location) -> np.ndarray:
        """Compute the wave field at specific location.

        Parameters
        ----------
        l : Location

        Returns
        -------
        `np.ndarray`
            Amplitude and phase of waves
        """
        cterm = 1j * self._wave_nums[l.index] * l.depth_within

        if l.wave_field == 'within':
            return (self._waves_a[l.index] * np.exp(cterm) +
                    self._waves_b[l.index] * np.exp(-cterm))
        elif l.wave_field == 'outcrop':
            return 2 * self._waves_a[l.index] * np.exp(cterm)
        elif l.wave_field == 'incoming_only':
            return self._waves_a[l.index] * np.exp(cterm)
        else:
            raise NotImplementedError

    def calc_accel_tf(self, location_in: Location, location_out: Location) -> \
            np.ndarray:
        """Compute the acceleration transfer function."""
        return (self.wave_at_location(location_out) /
                self.wave_at_location(location_in))

    def calc_stress_tf(self, location_in: Location, location_out: Location) -> \
            np.ndarray:
        """Compute the stress transfer function."""
        trans_func = self.calc_strain_tf(location_in, location_out)
        trans_func *= location_out.layer.comp_shear_mod
        return trans_func

    def calc_strain_tf(self, location_in: Location, location_out: Location) \
            -> np.ndarray:
        # FIXME: Correct discussion for using acceleration FAS
        # The strain transfer function from the acceleration at layer n (outcrop)
        # to the mid-height of layer m (within) is defined as:
        # Strain(angFreq, z=h_m/2)   i k*_m [ A_m exp(i k*_m h_m / 2) - B_m exp(-i k*_m h_m / 2)]
        # ------------------------ = ------------------------------------------------------------
        #    accel_n(angFreq)                       -angFreq^2 (2 * A_n)
        # The problem with this formula is that at low frequencies the division is
        # prone to errors -- in particular when angFreq = 0.
        # To solve this problem, strain is computed from the velocity FAS.  The associated
        # transfer function to compute the strain is then defined as:
        # Strain(angFreq, z=h_m/2)   -i [ A_m exp(i k*_m h_m / 2) - B_m exp(-i k*_m h_m / 2)]
        # ------------------------ = ------------------------------------------------------------
        #      vel_n(angFreq)                       v*_s (2 * A_n)
        assert location_out.wave_field == 'within'
        # The numerator cannot be computed using wave_at_location() because
        # it is A - B.
        cterm = (1j * self._wave_nums[location_out.index, :] *
                 location_out.depth_within)
        numer = (-1j * (self._waves_a[location_out.index, :] * np.exp(cterm) -
                        self._waves_b[location_out.index, :] * np.exp(-cterm)))
        denom = self.wave_at_location(location_in)

        # Strain is inversely proportional to the complex shear-wave velocity
        layer_out = location_out.layer
        ratio = layer_out.soil_type.gravity / layer_out.comp_shear_vel
        trans_func = ratio * (numer / denom)

        return trans_func


class EquivalentLinearCalculation(LinearElasticCalculator):
    """Equivalent-linear site response calculator.

    Parameters
    ----------
    input_location: :class:`pysra.base.site.Location`
        Location of the input motion -- including motion type.

    strain_ratio: float, default=0.65
        Ratio between the maximum strain and effective strain used to
        compute strain compatible properties.

    tolerance: float, default=0.01
        Tolerance in the iterative properties, which would cause the
        iterative process to terminate.

    max_iterations: int, default=15
        Maximum number of iterations to perform.
    """
    def __init__(self,
                 input_location: Location,
                 strain_ratio: float=0.65,
                 tolerance: float=0.01,
                 max_iterations: int=15):
        super().__init__(input_location)
        self._strain_ratio = strain_ratio
        self._tolerance = tolerance
        self._max_iterations = max_iterations

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
    def calc_strain_ratio(cls, mag: float) -> float:
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

    def calc_waves(self,
                   motion: Motion,
                   profile: Profile,
                   input_location: Location):
        """Perform the wave propagation.

        Parameters
        ----------
        motion: :class:`~.base.motion.Motion`
            Input motion.

        profile: :class:`~.base.site.Profile`
            Site profile.

        input_location: :class:`~.base.site.Location`
            Location of the input motion.
        """

        # Estimate the strain based on the PGV and shear-wave velocity
        for l in profile:
            l.strain = motion.pgv / l.initial_shear_vel

        iteration = 0
        while iteration < self.max_iterations:
            LinearElasticCalculator.calc_waves(self, motion, profile)

            for i, l in enumerate(profile[:-1]):
                l.strain = (
                    self.strain_ratio * motion.compute_peak(
                        self.calc_strain_tf(input_location,
                                            Location(i, l, 'within'))
                    )
                )

            max_error = max(
                max(l.shear_mod.relative_error, l.damping.relative_error)
                for l in profile)

            if max_error < self.tolerance:
                break

            iteration += 1