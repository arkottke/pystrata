# -*- coding: utf-8 -*-
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
from __future__ import annotations

import collections
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.constants
import tomli
from scipy.interpolate import interp1d

from .motion import GRAVITY
from .motion import WaveField

COMP_MODULUS_MODEL = "dormieux"

KPA_TO_ATM = scipy.constants.kilo / scipy.constants.atm

PUBLISHED_CURVES = None


def _load_published_curves():
    """Load published nonlinear curves."""
    global PUBLISHED_CURVES

    fpath = Path(__file__).parent / "data" / "published_curves.toml"
    with fpath.open("rb") as fp:
        models = tomli.load(fp)["models"]

    # Count to make sure there aren't repeated names
    counts = collections.Counter([m["name"] for m in models])
    if max(counts.values()) > 1:
        names = ", ".join([k for k, v in counts.items() if v > 1])
        warnings.warn(f"Repeated names in {fpath}: " + names)

    PUBLISHED_CURVES = {m["name"]: m for m in models}


def known_published_curves() -> List[dict]:
    """List of published curves in the database."""
    if not PUBLISHED_CURVES:
        _load_published_curves()
    return list(PUBLISHED_CURVES.keys())


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
                Damping ratio curve [decimal]
    """

    PARAMS = ["mod_reduc", "damping"]

    def __init__(self, name="", strains=None, values=None, param=None):
        self.name = name
        self._strains = np.asarray(strains).astype(float)
        self._values = np.asarray(values).astype(float)

        self._interpolater = None

        self._param = None
        self.param = param

        self._update()

    @classmethod
    def from_published(cls, name, param):
        assert param in cls.PARAMS
        if not PUBLISHED_CURVES:
            _load_published_curves()

        selected = PUBLISHED_CURVES[name][param]

        return cls(
            name, strains=selected["strains"], values=selected["values"], param=param
        )

    def __call__(self, strains):
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
        strains: float or array_like
            Shear strain of interest [decimal].

        Returns
        -------
        float or array_like
            The nonlinear property at the requested strain(s).
        """
        ln_strains = np.log(np.maximum(1e-9, strains))

        if self.strains.shape == self.values.shape:
            # 1D interpolate
            values = self._interpolater(ln_strains)
        else:
            ln_strains = np.atleast_1d(ln_strains)
            values = np.array([i(ln_strains[0]) for i in self._interpolater])
        return values

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
        """Initialize the interpolation."""

        if not self.strains.size:
            self._interpolater = None
            return

        x = np.log(self.strains)
        y = self.values

        if self.strains.shape == self.values.shape:
            # 1D interpolate
            self._interpolater = interp1d(
                x, y, "linear", bounds_error=False, fill_value=(y[0], y[-1])
            )
        elif self.values.ndim == 2 and self.strains.shape[0] == self.values.shape[0]:
            self._interpolater = [
                interp1d(
                    x,
                    y[:, i],
                    "linear",
                    bounds_error=False,
                    fill_value=(y[0, i], y[-1, i]),
                )
                for i in range(y.shape[1])
            ]
        else:
            self._interpolater = None


class SoilType(object):
    """Soiltype that combines nonlinear behavior and material properties.

    Parameters
    ----------
    name: str, optional
        used for identification
    unit_wt:  float
        unit weight of the material in [kN/m³]
    mod_reduc: :class:`NonlinearProperty` or None
        shear-modulus reduction curves. If None, linear behavior with no
        reduction is used
    damping: :class:`NonlinearProperty` or float
        damping ratio. [decimal] If float, then linear behavior with constant
        damping is used.
    """

    def __init__(self, name="", unit_wt=0.0, mod_reduc=None, damping=None):
        self.name = name
        self._unit_wt = unit_wt
        self.mod_reduc = mod_reduc
        self.damping = damping

    @classmethod
    def from_published(cls, name, unit_wt, *args):
        if not PUBLISHED_CURVES:
            _load_published_curves()

        model_mr = args[0]
        model_d = args[1] if len(args) > 1 else args[0]

        return cls(
            name,
            unit_wt=unit_wt,
            mod_reduc=NonlinearProperty.from_published(model_mr, "mod_reduc"),
            damping=NonlinearProperty.from_published(model_d, "damping"),
        )

    def copy(self):
        return SoilType(self.name, self.unit_wt, self.mod_reduc, self.damping)

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
    def quality(self):
        return 1 / (2 * self.damping_min)

    @property
    def unit_wt(self):
        return self._unit_wt

    @property
    def is_nonlinear(self):
        """If nonlinear properties are specified."""
        return any(
            isinstance(p, NonlinearProperty) for p in [self.mod_reduc, self.damping]
        )

    def __eq__(self, other):
        # return all(
        #     getattr(self, attr) == getattr(other, attr)
        #     for attr in ['name', 'unit_wt', 'mod_reduc', 'damping'])
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)

    def __hash__(self):
        return hash(self.__dict__.values())


class ModifiedHyperbolicSoilType(SoilType):
    def __init__(self, name, unit_wt, damping_min, strains=None):
        """

        Parameters
        ----------
        name: str, optional
        used for identification
        unit_wt:  float
            unit weight of the material in [kN/m³]
        damping_min: float
            Minimum damping at low strains [decimal]
        strains: `array_like`, default: np.logspace(-6, -1.5, num=20)
            shear strains levels [decimal]
        """
        super().__init__(name, unit_wt)

        if strains is None:
            strains = np.logspace(-6, -1.5, num=20)  # in decimal
        else:
            strains = np.asarray(strains)

        # Modified hyperbolic shear modulus reduction
        mod_reduc = 1 / (1 + (strains / self.strain_ref) ** self.curvature)
        self.mod_reduc = NonlinearProperty(name, strains, mod_reduc, "mod_reduc")

        # Masing damping based on shear -modulus reduction [%]
        strains_percent = strains * 100
        strain_ref_percent = self.strain_ref * 100
        damping_masing_a1 = (100.0 / np.pi) * (
            4
            * (
                strains_percent
                - strain_ref_percent
                * np.log((strains_percent + strain_ref_percent) / strain_ref_percent)
            )
            / (strains_percent**2 / (strains_percent + strain_ref_percent))
            - 2.0
        )
        # Correction between perfect hyperbolic strain model and modified
        # model [%].
        c1 = -1.1143 * self.curvature**2 + 1.8618 * self.curvature + 0.2523
        c2 = 0.0805 * self.curvature**2 - 0.0710 * self.curvature - 0.0095
        c3 = -0.0005 * self.curvature**2 + 0.0002 * self.curvature + 0.0003
        damping_masing = (
            c1 * damping_masing_a1
            + c2 * damping_masing_a1**2
            + c3 * damping_masing_a1**3
        )

        # Compute the damping correction in percent
        d_correction = self.masing_scaling * damping_masing * mod_reduc**0.1

        # Prevent the damping from reducing as it can at large strains
        damping = np.maximum.accumulate(d_correction / 100.0)

        # Add the minimum damping component
        if isinstance(damping_min, np.ndarray):
            # Broadcast
            damping = damping_min + damping[:, np.newaxis]
        else:
            damping += damping_min

        # Convert to decimal values
        self.damping = NonlinearProperty(name, strains, damping, "damping")

    @property
    def masing_scaling(self):
        """Scaling of the Masing damping component."""
        return NotImplementedError

    @property
    def curvature(self):
        """Curvature of the model."""
        return NotImplementedError

    @property
    def strain_ref(self):
        """Reference strain."""
        return NotImplementedError


@dataclass
class TwoParamModifiedHyperbolicCoeffs:
    a: float
    b1: float
    b2: float
    g1: float
    g2: float
    d0: float
    d1: float
    d: float
    c: float
    gd1: float
    gd2: float


class TwoParamModifiedHyperbolicSoilType(SoilType):
    def __init__(
        self,
        name: str = "",
        unit_wt: float = 0,
        stress_mean: float = 101.3,
        strains: Optional[npt.ArrayLike] = None,
        coeffs: Optional[TwoParamModifiedHyperbolicCoeffs] = None,
    ):
        """

        Parameters
        ----------
        name: str, optional
        used for identification
        unit_wt:  float
            unit weight of the material in [kN/m³]
        stress_mean: float, default=101.3
            mean effective stress [kN/m²]
        damping_min: float
            Minimum damping at low strains [decimal]
        strains: `array_like`, default: np.logspace(-6, -1.5, num=20)
            shear strains levels [decimal]
        """

        if coeffs is None:
            C = TwoParamModifiedHyperbolicCoeffs(
                1.04, 0.438, -0.007, 0.011, 0.318, 1.47, -0.2, 13.125, 1.187, 0.11, 0.23
            )
        else:
            C = coeffs

        if strains is None:
            strains = np.logspace(-6, -1.5, num=20)  # in decimal
        else:
            strains = np.asarray(strains)

        self._stress_mean = stress_mean
        stress_mean_atm = stress_mean * KPA_TO_ATM

        # Convert from percent to decimal strain
        strain_mr = C.g1 * stress_mean_atm**C.g2 / 100
        b = C.b1 + C.b2 * stress_mean_atm
        values_mr = 1 / ((1 + (strains / strain_mr) ** C.a)) ** b
        mod_reduc = NonlinearProperty(name, strains, values_mr)

        d_min = C.d0 * stress_mean_atm**C.d1
        # Convert from percent to decimal strain
        strain_dr = (C.gd1 * stress_mean_atm**C.gd2) / 100
        term = (strains / strain_dr) ** C.c
        # Convert to damping in decimal
        values_d = (C.d * term + d_min) / (term + 1) / 100
        damping = NonlinearProperty(name, strains, values_d)

        super().__init__(name, unit_wt, mod_reduc, damping)


class DarendeliSoilType(ModifiedHyperbolicSoilType):
    """
    Darendeli (2001) model for fine grained soils.

    Parameters
    ----------
    unit_wt:  float
        unit weight of the material [kN/m³]
    plas_index: float, default=0
        plasticity index [percent]
    ocr: float, default=1
        over-consolidation ratio
    stress_mean: float, default=101.3
        mean effective stress [kN/m²]
    freq: float, default=1
        excitation frequency [Hz]
    num_cycles: float, default=10
        number of cycles of loading
    strains: `array_like`, default: np.logspace(-6, -1.5, num=20)
        shear strains levels [decimal]
    """

    def __init__(
        self,
        unit_wt=0.0,
        plas_index=0,
        ocr=1,
        stress_mean=101.3,
        freq=1,
        num_cycles=10,
        damping_min=None,
        strains=None,
    ):

        self._plas_index = plas_index
        self._ocr = ocr
        self._stress_mean = stress_mean
        self._freq = freq
        self._num_cycles = num_cycles

        if damping_min is None:
            damping_min = self._calc_damping_min()

        name = self._create_name()

        super().__init__(name, unit_wt, damping_min, strains)

    def _calc_damping_min(self):
        """minimum damping [decimal]"""
        return (
            (0.8005 + 0.0129 * self._plas_index * self._ocr**-0.1069)
            * (self._stress_mean * KPA_TO_ATM) ** -0.2889
            * (1 + 0.2919 * np.log(self._freq))
        ) / 100

    @property
    def masing_scaling(self):
        # Masing correction factor
        return 0.6329 - 0.00566 * np.log(self._num_cycles)

    @property
    def strain_ref(self):
        """reference strain [decimal]"""
        return (
            (0.0352 + 0.0010 * self._plas_index * self._ocr**0.3246)
            * (self._stress_mean * KPA_TO_ATM) ** 0.3483
        ) / 100

    @property
    def curvature(self):
        return 0.9190

    def _create_name(self):
        fmt = "Darendeli (PI={:.0f}, OCR={:.1f}, σₘ'={:.1f} kN/m²)"
        return fmt.format(self._plas_index, self._ocr, self._stress_mean)


class MenqSoilType(ModifiedHyperbolicSoilType):
    """
    Menq SoilType for gravelly soils.

    Parameters
    ----------
    unit_wt:  float
        unit weight of the material [kN/m³]
    uniformity_coeff: float, default=10
        uniformity coeffecient (Cᵤ)
    diam_mean: float, default=5
        mean diameter (D₅₀) [mm]
    stress_mean: float, default=101.3
        mean effective stress [kN/m²]
    num_cycles: float, default=10
        number of cycles of loading
    strains: `array_like`, default: np.logspace(-4, 0.5, num=20)
        shear strains levels [decimal]
    """

    def __init__(
        self,
        unit_wt=0.0,
        uniformity_coeff=10,
        diam_mean=5,
        stress_mean=101.3,
        num_cycles=10,
        damping_min=None,
        strains=None,
    ):

        self._uniformity_coeff = uniformity_coeff
        self._diam_mean = diam_mean
        self._stress_mean = stress_mean
        self._num_cycles = num_cycles

        if damping_min is None:
            damping_min = self._calc_damping_min()

        name = self._create_name()

        super().__init__(name, unit_wt, damping_min, strains)

    def _calc_damping_min(self):
        return (
            0.55
            * self._uniformity_coeff**0.1
            * self._diam_mean**-0.3
            * (self._stress_mean * KPA_TO_ATM) ** -0.08
        ) / 100

    @property
    def masing_scaling(self):
        # Masing correction factor
        return 0.6329 - 0.00566 * np.log(self._num_cycles)

    @property
    def strain_ref(self):
        return (
            0.12
            * self._uniformity_coeff**-0.6
            * (self._stress_mean * KPA_TO_ATM)
            ** (0.5 * self._uniformity_coeff**-0.15)
        ) / 100

    @property
    def curvature(self):
        return 0.86 + 0.1 * np.log10(self._stress_mean * KPA_TO_ATM)

    def _create_name(self):
        fmt = "Menq (Cᵤ={:.1f}, D₅₀={:.1f} mm, σₘ'={:.1f} kN/m²)"
        return fmt.format(self._uniformity_coeff, self._diam_mean, self._stress_mean)


class FixedValues:
    """Utility class to store fixed values"""

    def __init__(self, **kwds):
        self._params = kwds

    def __getattr__(self, name):
        return self._params[name]


class KishidaSoilType(SoilType):
    """Empirical nonlinear model for highly organic soils.

    Parameters
    ----------
    name: str, optional
        used for identification
    unit_wt:  float or None, default=None
        unit weight of the material [kN/m³]. If *None*, then unit weight is
        computed by the empirical model.
    stress_vert: float
        vertical effective stress [kN/m²]
    organic_content: float
        organic_content [percent]
    lab_consol_ratio: float, default=1
        laboratory consolidation ratio. This parameter is included for
        completeness, but the default value of 1 should be used for field
        applications.
    strains: `array_like` or None
        shear strains levels. If *None*, a default of `np.logspace(-6, 0.5,
        num=20)` will be used. The first strain should be small such that the
        shear modulus reduction is equal to 1. [decimal]
    """

    def __init__(
        self,
        name="",
        unit_wt=None,
        stress_vert=101.3,
        organic_content=10,
        lab_consol_ratio=1,
        strains=None,
    ):
        super().__init__(name, unit_wt)

        self._stress_vert = float(stress_vert)
        self._organic_content = float(organic_content)
        self._lab_consol_ratio = float(lab_consol_ratio)

        if strains is None:
            strains = np.logspace(-6, -1.5, num=20)
        else:
            strains = np.asarray(strains)

        strains_percent = strains * 100

        # Mean values of the predictors defined in the paper
        x_1_mean = -2.5
        x_2_mean = 4.0
        x_3_mean = 0.5
        # Predictor variables
        x_3 = 2.0 / (1 + np.exp(self._organic_content / 23))
        strain_ref = self._calc_strain_ref(x_3, x_3_mean)
        strain_ref_percent = strain_ref * 100
        x_1 = np.log(strains_percent + strain_ref_percent)
        x_2 = np.log(self._stress_vert)

        if unit_wt is None:
            self._unit_wt = self._calc_unit_wt(x_2, x_3)
        else:
            self._unit_wt = float(unit_wt)

        # Convert to 1D arrays for matrix math support
        ones = np.ones_like(strains)
        x_2 = x_2 * ones
        x_3 = x_3 * ones

        mod_reducs = self._calc_mod_reduc(
            strains_percent,
            strain_ref_percent,
            x_1,
            x_1_mean,
            x_2,
            x_2_mean,
            x_3,
            x_3_mean,
        )
        dampings = self._calc_damping(mod_reducs, x_2, x_2_mean, x_3, x_3_mean)

        name = self._create_name()
        self.mod_reduc = NonlinearProperty(name, strains, mod_reducs, "mod_reduc")
        self.damping = NonlinearProperty(name, strains, dampings, "damping")

    @staticmethod
    def _calc_strain_ref(x_3, x_3_mean):
        """Compute the reference strain using Equation (6)."""
        b_9 = -1.41
        b_10 = -0.950
        return np.exp(b_9 + b_10 * (x_3 - x_3_mean)) / 100

    def _calc_mod_reduc(
        self, strains, strain_ref, x_1, x_1_mean, x_2, x_2_mean, x_3, x_3_mean
    ):
        """Compute the shear modulus reduction using Equation (1)."""

        ones = np.ones_like(strains)
        # Predictor
        x_4 = np.log(self._lab_consol_ratio) * ones
        x = np.c_[
            ones,
            x_1,
            x_2,
            x_3,
            x_4,
            (x_1 - x_1_mean) * (x_2 - x_2_mean),
            (x_1 - x_1_mean) * (x_3 - x_3_mean),
            (x_2 - x_2_mean) * (x_3 - x_3_mean),
            (x_1 - x_1_mean) * (x_2 - x_2_mean) * (x_3 - x_3_mean),
        ]
        # Coefficients
        denom = np.log(
            1 / strain_ref + strains / strain_ref
        )  # TODO: is this percent or decimal?
        b = np.c_[
            5.11 * ones,
            -0.729 * ones,
            (1 - 0.37 * x_3_mean * (1 + ((np.log(strain_ref) - x_1_mean) / denom))),
            -0.693 * ones,
            0.8 - 0.4 * x_3,
            0.37 * x_3_mean / denom,
            0.0 * ones,
            -0.37 * (1 + (np.log(strain_ref) - x_1_mean) / denom),
            0.37 / denom,
        ]
        ln_shear_mod = (b * x).sum(axis=1)
        shear_mod = np.exp(ln_shear_mod)
        mod_reduc = shear_mod / shear_mod[0]
        return mod_reduc

    @staticmethod
    def _calc_damping(mod_reducs, x_2, x_2_mean, x_3, x_3_mean):
        """Compute the damping ratio using Equation (16)."""
        # Mean values of the predictors
        x_1_mean = -1.0
        x_1 = np.log(np.log(1 / mod_reducs) + 0.103)

        ones = np.ones_like(mod_reducs)
        x = np.c_[
            ones,
            x_1,
            x_2,
            x_3,
            (x_1 - x_1_mean) * (x_2 - x_2_mean),
            (x_2 - x_2_mean) * (x_3 - x_3_mean),
        ]
        c = np.c_[2.86, 0.571, -0.103, -0.141, 0.0419, -0.240]

        ln_damping = (c * x).sum(axis=1)
        return np.exp(ln_damping) / 100.0

    @staticmethod
    def _calc_unit_wt(x_1, x_2):
        x = np.r_[1, x_1, x_2]
        d = np.r_[-0.112, 0.038, 0.360]

        ln_density = d.T @ x
        unit_wt = np.exp(ln_density) * scipy.constants.g
        return unit_wt

    def _create_name(self):
        return "Kishida (σᵥ'={:.1f} kN/m², OC={:.0f} %)".format(
            self._stress_vert, self._organic_content
        )


# TODO: for nonlinear site response this class wouldn't be used. Better way
# to do this? Maybe have the calculator create it?
class IterativeValue(object):
    def __init__(self, value):
        self._value = value
        self._previous = 1e-9

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
        """The relative error, in percent, between the two iterations."""
        if self.previous is not None:
            # FIXME
            # Use the maximum strain value -- this is important for error
            #  calculation with frequency dependent properties
            # prev = np.max(self.previous)
            # value = np.max(self.value)
            try:
                err = 100.0 * np.max((self.previous - self.value) / self.value)
            except ZeroDivisionError:
                err = np.inf
        else:
            err = 0
        return err

    def reset(self):
        self._previous = None


class Layer(object):
    """Docstring for Layer"""

    def __init__(self, soil_type, thickness, shear_vel):
        """@todo: to be defined!"""
        self._profile = None

        self._soil_type = soil_type
        self._thickness = thickness
        self.initial_shear_vel = shear_vel

        self._depth = 0
        self._stress_vert = 0

    def __repr__(self):
        index = self._profile.index(self)
        shear_vel = self._initial_shear_vel
        thickness = self._thickness
        st_name = self.soil_type.name
        return (
            f"<Layer(index={index}, "
            f"shear_vel={shear_vel:0.1f} m/s, "
            f"thickness={thickness:0.1f} m, "
            f"soil_type={st_name})>"
        )

    def __eq__(self, other):
        attrs = ["_soil_type", "_thickness", "initial_shear_vel"]
        return (type(self) == type(other)) and all(
            [getattr(self, a) == getattr(other, a) for a in attrs]
        )

    def __hash__(self):
        return hash(self.__dict__.values())

    @classmethod
    def copy_of(cls, other):
        """Return a copy of the Layer instance with previously defined SoilType."""
        return cls(other.soil_type, other.thickness, other.shear_vel)

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

    @property
    def density(self):
        """Density of soil in [kg/m³]."""
        return self.soil_type.density

    @property
    def damping(self):
        """Strain-compatible damping."""
        try:
            value = self._damping.value
        except AttributeError:
            value = self._damping
        return value

    @property
    def initial_shear_mod(self):
        """Initial (small-strain) shear modulus [kN/m²]."""
        return self.density * self.initial_shear_vel**2

    @property
    def initial_shear_vel(self):
        """Initial (small-strain) shear-wave velocity [m/s]."""
        return self._initial_shear_vel

    @initial_shear_vel.setter
    def initial_shear_vel(self, value: float):
        """Set initial (small-strain) shear-wave velocity [m/s]."""

        self._initial_shear_vel = value
        # Reset the iterated values
        self.reset()

    @property
    def comp_shear_mod(self):
        """Strain-compatible complex shear modulus [kN/m²]."""
        damping = self.damping
        if COMP_MODULUS_MODEL == "seed":
            # Frequency independent model (Seed et al., 1970)
            # Correct dissipated energy
            # Incorrect shear modulus: G * \sqrt{1 + 4 \beta^2 }
            comp_factor = 1 + 2j * damping
        elif COMP_MODULUS_MODEL == "kramer":
            # Simplifed shear modulus (Kramer, 1996)
            # Correct dissipated energy
            # Incorrect shear modulus: G * \sqrt{1 + 2 \beta^2 + \beta^4 }
            comp_factor = 1 - damping**2 + 2j * damping
        elif COMP_MODULUS_MODEL == "dormieux":
            # Dormieux and Canou (1990)
            # Correct dissipated energy
            # Correct shear modulus:
            comp_factor = np.sqrt(1 - 4 * damping**2) + 2j * damping
        else:
            raise NotImplementedError
        comp_shear_mod = self.shear_mod * comp_factor
        return comp_shear_mod

    @property
    def comp_shear_vel(self):
        """Strain-compatible complex shear-wave velocity [m/s]."""
        return np.sqrt(self.comp_shear_mod / self.density)

    @property
    def max_error(self):
        return max(self._shear_mod.relative_error, self._damping.relative_error)

    def reset(self):
        self._shear_mod = IterativeValue(self.initial_shear_mod)
        self._damping = IterativeValue(self.soil_type.damping_min)
        # Use a small initial value
        self._strain = IterativeValue(1e-6)

        self.strain_max = None

    @property
    def shear_mod(self):
        """Strain-compatible shear modulus [kN//m²]."""
        try:
            value = self._shear_mod.value
        except AttributeError:
            value = self._shear_mod
        return value

    @property
    def shear_mod_reduc(self):
        return self.shear_mod / self.initial_shear_mod

    @property
    def shear_vel(self):
        """Strain-compatible shear-wave velocity [m/s]."""
        return np.sqrt(self.shear_mod / self.density)

    @property
    def stress_shear_eff(self):
        """Effective shear stress at layer midpoint"""
        return self.shear_mod * self.strain

    @property
    def stress_shear_max(self):
        """Maximum shear stress at layer midpoint"""
        return self.shear_mod * self.strain_max

    @property
    def strain(self):
        try:
            value = self._strain.value
        except AttributeError:
            value = self._strain

        return value

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
            mod_reduc = 1.0

        self._shear_mod.value = self.initial_shear_mod * mod_reduc

        try:
            self._damping.value = self.soil_type.damping(strain)
        except TypeError:
            # No iteration provided by damping
            self._damping.value = self.soil_type.damping

    @property
    def soil_type(self):
        return self._soil_type

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness
        self._profile.update_layers(self._profile.index(self) + 1)

    @property
    def travel_time(self):
        """Travel time through the layer."""
        return self.thickness / self.shear_vel

    @property
    def unit_wt(self):
        return self.soil_type.unit_wt

    def stress_vert(self, depth_within=0, effective=False):
        """Vertical stress from the top of the layer [kN//m²]."""
        assert depth_within <= self.thickness
        stress_vert = self._stress_vert + depth_within * self.unit_wt
        if effective:
            pore_pressure = self._profile.pore_pressure(self.depth + depth_within)
            stress_vert -= pore_pressure
        return stress_vert

    def stress_mean(self, depth_within=0, effective=False, k0=0.5):
        """Mean effective stress from the top of the layer [kN//m²]."""
        stress_vert = self.stress_vert(depth_within, effective)
        return (2 * k0 * stress_vert + stress_vert) / 3.0

    @property
    def incr_site_atten(self):
        return (
            2 * self.soil_type.damping_min * self._thickness
        ) / self.initial_shear_vel


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

    def stress_vert(self, effective=False):
        return self._layer.stress_vert(self.depth_within, effective=effective)

    def __repr__(self):
        return (
            "<Location(layer_index={_index}, depth_within={_depth_within} "
            "wave_field={_wave_field})>".format(**self.__dict__)
        )


class Profile(collections.abc.Container):
    """Soil profile with an infinite halfspace at the base."""

    def __init__(self, layers=None, wt_depth=0):
        super().__init__()
        self.layers = layers or []
        self.wt_depth = wt_depth
        if layers:
            self.update_layers()

    @classmethod
    def from_dataframe(cls, df, wt_depth=0):
        """Create a profile based on a table with columns:
        - thickness (m)
        - vel_shear (m)
        - unit_wt (kN/m³)
        - damping (dec)
        """

        layers = []
        for _, row in df.iterrows():
            layers.append(
                Layer(
                    SoilType(
                        name=row.get("name", ""),
                        unit_wt=row["unit_wt"],
                        mod_reduc=None,
                        damping=row["damping"],
                    ),
                    row["thickness"],
                    row["vel_shear"],
                )
            )
        return cls(layers, wt_depth)

    def __iter__(self):
        return iter(self.layers)

    def __contains__(self, value):
        return value in self.layers

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.layers[key]

    @classmethod
    def copy_of(cls, other):
        """Return a copy of the profile with new Layer instances."""
        return cls([Layer.copy_of(layer) for layer in other], other.wt_depth)

    def index(self, layer):
        return self.layers.index(layer)

    def append(self, layer):
        last = len(self.layers)
        self.layers.append(layer)
        self.update_layers(last)

    def insert(self, index, layer):
        self.layers.insert(index, layer)
        self.update_layers(index)

    def reset_layers(self):
        """Set initial properties from the soil types."""
        for layer in self:
            layer.reset()

    def update_layers(self, start_layer=0):
        if start_layer < 1:
            depth = 0
            stress_vert = 0
        else:
            ref_layer = self[start_layer - 1]
            depth = ref_layer.depth_base
            stress_vert = ref_layer.stress_vert(ref_layer.thickness, effective=False)

        for layer in self[start_layer:]:
            layer._profile = self
            layer._depth = depth
            layer._stress_vert = stress_vert
            if layer != self[-1]:
                # Use the layer to compute the values at the base of the
                # layer, and apply them at the top of the next layer
                depth = layer.depth_base
                stress_vert = layer.stress_vert(layer.thickness, effective=False)

    def iter_soil_types(self):
        yielded = set()
        for layer in self:
            if layer.soil_type in yielded:
                continue
            else:
                yielded.add(layer)
                yield layer.soil_type

    def auto_discretize(
        self,
        max_freq: float = 50.0,
        wave_frac: float = 0.2,
        nonlinear_only: bool = True,
    ) -> Profile:
        """Subdivide the layers to capture strain variation.

        Parameters
        ----------
        max_freq: float
            Maximum frequency of interest [Hz].
        wave_frac: float
            Fraction of wavelength required. Typically 1/3 to 1/5.

        max_thick: float *optional*
            If provided, layers are limited to be at most that thick. This is applied to
            all layers regardless of nonlinearity.

        Returns
        -------
        profile: Profile
            A new profile with modified layer thicknesses
        """
        layers = []
        for l in self[:-1]:
            if not nonlinear_only or l.soil_type.is_nonlinear:
                opt_thickness = l.shear_vel / max_freq * wave_frac
                count = max(np.ceil(l.thickness / opt_thickness).astype(int), 1)
                thickness = l.thickness / count
                for _ in range(count):
                    layers.append(Layer(l.soil_type, thickness, l.shear_vel))
            else:
                layers.append(l)
        # Add the halfspace
        layers.append(self[-1])

        return Profile(layers, wt_depth=self.wt_depth)

    def pore_pressure(self, depth):
        """Pore pressure at a given depth in [kN//m²].

        Parameters
        ----------
        depth

        Returns
        -------
        pore_pressure
        """
        return GRAVITY * max(depth - self.wt_depth, 0)

    def site_attenuation(self):
        return sum(layer.incr_site_atten for layer in self)

    def lookup_depth(self, depth: float) -> Tuple[int, float]:
        """Look up the layer and the depth within the layer for a specific depth.

        Parameters
        ----------
        depth: float
            Depth corresponding to the location of interest.

        Returns
        -------
        index: int
            Layer index

        depth_within: float
            Depth from the top of the layer to achieve the specific depth.
        """

        # Make sure all of the depths to updated
        self.update_layers()

        for i, layer in enumerate(self[:-1]):
            if layer.depth <= depth < layer.depth_base:
                depth_within = depth - layer.depth
                break
        else:
            # Bedrock
            i = len(self) - 1
            depth_within = depth - self[-1].depth

        return i, depth_within

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
            i, depth_within = self.lookup_depth(depth)
            layer = self[i]
        elif index is not None and depth is None:
            layer = self[index]
            i = self.index(layer)
            depth_within = 0
        else:
            raise NotImplementedError

        return Location(i, layer, wave_field, depth_within)

    def time_average_vel(self, depth):
        """Calculate the time-average velocity.

        Parameters
        ----------
        depth: float
            Depth over which the average velocity is computed.

        Returns
        -------
        avg_vel: float
            Time averaged velocity.
        """
        depths = self.depth
        # Final layer is infinite and is treated separately
        travel_times = np.r_[0, self.travel_time[:-1]]
        # If needed, add the final layer to the required depth
        if depths[-1] < depth:
            depths = np.r_[depths, depth]
            travel_times = np.r_[
                travel_times, (depth - self[-1].depth) / self[-1].shear_vel
            ]

        total_travel_times = np.cumsum(travel_times)
        # Interpolate the travel time to the depth of interest
        avg_shear_vel = depth / np.interp(depth, depths, total_travel_times)
        return avg_shear_vel

    def simplified_rayliegh_vel(self):
        """Simplified Rayliegh velocity of the site.

        This follows the simplifications proposed by Urzua et al. (2017)

        Returns
        -------
        rayleigh_vel : float
            Equivalent shear-wave velocity.
        """
        # FIXME: What if last layer has no thickness?
        thicks = self.thickness
        depths_mid = self.depth_mid
        shear_vels = self.initial_shear_vel

        mode_incr = depths_mid * thicks / shear_vels**2
        # Mode shape is computed as the sumation from the base of
        # the profile. Need to append a 0 for the roll performed in the next
        # step
        shape = np.r_[np.cumsum(mode_incr[::-1])[::-1], 0]

        # Roll is used to offset the mode_shape so that the sum
        # can be calculated for two adjacent layers
        freq_fund = np.sqrt(
            4
            * np.sum(thicks * depths_mid**2 / shear_vels**2)
            / np.sum(
                thicks * np.sum(np.c_[shape, np.roll(shape, -1)], axis=1)[:-1] ** 2
            )
        )
        period_fun = 2 * np.pi / freq_fund
        rayleigh_vel = 4 * thicks.sum() / period_fun
        return rayleigh_vel

    def plot(self, prop, ax=None, plot_kwds=None, axis_kwds=None):
        # Defaults
        xlabels = {
            "density": "Density (kN/m³)",
            "max_error": "Max. Error (%)",
            "travel_time": "Travel time (sec)",
            "slowness": "Slowness (1/s)",
            "initial_shear_vel": "Initial $V_s$ (m/s)",
            "shear_vel": "$V_s$ (m/s)",
            "strain": "Strain (dec)",
            "damping": "Damping (dec)",
        }
        _axis_kwds = {
            "ylabel": "Depth (m)",
            "ylim": (1.1 * self.depth[-1], 0),
            "xlabel": xlabels[prop],
            "xlim": (0, None),
        }

        plot_kwds = plot_kwds or dict()
        axis_kwds = {**_axis_kwds, **(axis_kwds or dict())}

        if ax is None:
            fig, ax = plt.subplots()

        ax.step(getattr(self, prop), self.depth, where="pre", **plot_kwds)
        ax.set(**axis_kwds)

        return ax

    @property
    def damping(self):
        return self._get_values("damping")

    @property
    def density(self):
        return self._get_values("density")

    @property
    def depth(self):
        return self._get_values("depth")

    @property
    def depth_mid(self):
        return self._get_values("depth_mid")

    @property
    def thickness(self):
        return self._get_values("thickness")

    @property
    def max_error(self):
        return self._get_values("max_error")

    @property
    def travel_time(self):
        return self._get_values("travel_time")

    @property
    def slowness(self):
        return 1 / self.initial_shear_vel

    @property
    def initial_shear_vel(self):
        return self._get_values("initial_shear_vel")

    @property
    def shear_vel(self):
        return self._get_values("shear_vel")

    @property
    def strain(self):
        return self._get_values("strain")

    @property
    def unit_wt(self):
        return self._get_values("unit_wt")

    def _get_values(self, attr):
        return np.array([getattr(layer, attr) for layer in self])
