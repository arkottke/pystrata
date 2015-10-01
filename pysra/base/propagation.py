from typing import List

import numpy as np

from .site import Layer, Location
from .motion import Motion


class LinearElasticCalculator(object):
    def __init__(self):
        self._waves_a = np.array([])
        self._waves_b = np.array([])
        self._wave_nums = np.array([])

    def calc_waves(self, motion: Motion, layers: List[Layer]):
        # Set intial properties
        for l in layers:
            l.strain = 0.

        # Compute the complex wave numbers of the system
        wave_nums = np.empty((len(layers), len(motion.freqs)), np.complex)
        for i, l in enumerate(layers):
            wave_nums[i, :] = motion.angular_freqs / l.comp_shear_vel

        # Compute the waves. In the top surface layer, the up-going and
        # down-going waves have an amplitude of 1 as they are completely
        # reflected at the surface.
        waves_a = np.ones_like(wave_nums, np.complex)
        waves_b = np.ones_like(wave_nums, np.complex)
        for i, l in enumerate(layers[:-1]):
            # Complex impedance
            cimped = ((wave_nums[i] * l.comp_shear_mod) /
                      (wave_nums[i + 1] * layers[i + 1].comp_shear_mod))

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
        `np.array`
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
