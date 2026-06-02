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
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import tomli
from scipy.interpolate import interp1d

from .motion import WaveField
from .units import GRAVITY, KPA_TO_ATM, convert_kwds_units, convert_units

logger = logging.getLogger(__name__)

COMP_MODULUS_MODEL = "dormieux"

PUBLISHED_CURVES = dict()


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


def known_published_curves() -> list[dict]:
    """List of published curves in the database."""
    if not PUBLISHED_CURVES:
        _load_published_curves()
    return list(PUBLISHED_CURVES.keys())


class NonlinearCurve(ABC):
    """Abstract base class for nonlinear curve with log-linear interpolation.

    Parameters
    ----------
    name: str, optional
        used for identification
    strains: :class:`numpy.ndarray`, optional
        strains for each of the values [decimal].
    values: :class:`numpy.ndarray`, optional
        value of the property corresponding to each strain. Damping should be
        specified in decimal, e.g., 0.05 for 5%.
    limits: tuple, optional
        (min, max) limits for clipping interpolated values
    """

    PARAMS = ["mod_reduc", "damping"]

    @convert_units(strains="dimensionless")
    def __init__(self, name="", strains=None, values=None, limits=None):
        self.name = name
        self._strains = np.asarray(strains).astype(float)
        self._values = np.asarray(values).astype(float)

        self._interpolater = None

        if limits is None:
            limits = self._default_limits
        self._limits = limits

        self._update()

    @property
    @abstractmethod
    def param(self) -> str:
        """Nonlinear parameter name ('mod_reduc' or 'damping')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _default_limits(self) -> tuple[float, float]:
        """Default (min, max) limits for this property type."""
        raise NotImplementedError

    @classmethod
    def from_published(cls, name: str, param: str):
        """Create a NonlinearCurve from published curves.

        Parameters
        ----------
        name : str
            Name of the published curve model
        param : str
            Type of parameter: 'mod_reduc' or 'damping'

        Returns
        -------
        ModulusReductionCurve or DampingCurve
            The appropriate subclass instance
        """
        assert param in cls.PARAMS, f"param must be one of {cls.PARAMS}"
        if not PUBLISHED_CURVES:
            _load_published_curves()

        selected = PUBLISHED_CURVES[name][param]

        if param == "mod_reduc":
            return ModulusReductionCurve(
                name, strains=selected["strains"], values=selected["values"]
            )
        else:
            return DampingCurve(
                name, strains=selected["strains"], values=selected["values"]
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

        return np.clip(values, *self._limits)

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


class ModulusReductionCurve(NonlinearCurve):
    """Shear-modulus reduction curve.

    Parameters
    ----------
    name: str, optional
        used for identification
    strains: :class:`numpy.ndarray`, optional
        strains for each of the values [decimal].
    values: :class:`numpy.ndarray`, optional
        shear-modulus reduction values (G/Gmax) corresponding to each strain.
    limits: tuple, optional
        (min, max) limits for clipping interpolated values.
        Default: (0.001, 1)
    """

    @property
    def param(self) -> str:
        return "mod_reduc"

    @property
    def _default_limits(self) -> tuple[float, float]:
        return (0.001, 1)


class DampingCurve(NonlinearCurve):
    """Damping ratio curve.

    Parameters
    ----------
    name: str, optional
        used for identification
    strains: :class:`numpy.ndarray`, optional
        strains for each of the values [decimal].
    values: :class:`numpy.ndarray`, optional
        damping ratio values [decimal] corresponding to each strain.
    limits: tuple, optional
        (min, max) limits for clipping interpolated values.
        Default: (0, 0.49)
    """

    @property
    def param(self) -> str:
        return "damping"

    @property
    def _default_limits(self) -> tuple[float, float]:
        return (0, 0.49)


class SoilType:
    """Soiltype that combines nonlinear behavior and material properties.

    Parameters
    ----------
    name: str, optional
        used for identification
    unit_wt:  float
        unit weight of the material in [kN/m³]
    mod_reduc: :class:`NonlinearCurve` or None
        shear-modulus reduction curves. If None, linear behavior with no
        reduction is used
    damping: :class:`NonlinearCurve` or float
        damping ratio. [decimal] If float, then linear behavior with constant
        damping is used.
    """

    @convert_units(unit_wt="kilonewton / meter ** 3", damping="dimensionless")
    def __init__(
        self,
        name: str = "",
        unit_wt: float = 0.0,
        mod_reduc: None | NonlinearCurve = None,
        damping: float | NonlinearCurve = 0.0,
    ) -> None:
        self.name = name
        self._unit_wt = unit_wt
        self.mod_reduc = mod_reduc
        self.damping = damping

    @classmethod
    def from_published(
        cls,
        name: str = "",
        unit_wt: float = 0.0,
        model: str = "",
        model_damping: str | None = None,
    ) -> SoilType:
        if not PUBLISHED_CURVES:
            _load_published_curves()

        if model_damping is None:
            model_damping = model

        return cls(
            name,
            unit_wt=unit_wt,
            mod_reduc=NonlinearCurve.from_published(model, "mod_reduc"),
            damping=NonlinearCurve.from_published(model_damping, "damping"),
        )

    def copy(self) -> SoilType:
        return SoilType(self.name, self.unit_wt, self.mod_reduc, self.damping)

    @classmethod
    def from_curves(cls, curves, name: str = "", unit_wt: float | None = None) -> "SoilType":
        """Create from any object with .strains, .mod_reduc, .damping, .damping_min.

        Duck-typed: accepts ``pygmm.contracts.NonlinearSoilCurves`` or any
        object with the same attributes.
        """
        if unit_wt is None:
            unit_wt = getattr(curves, "unit_wt", None) or 0.0
        name = name or getattr(curves, "name", "") or ""
        mr = ModulusReductionCurve(name, curves.strains, curves.mod_reduc)
        d = DampingCurve(name, curves.strains, curves.damping)
        return cls(name=name, unit_wt=unit_wt, mod_reduc=mr, damping=d)

    @property
    def density(self) -> float:
        """Density of the soil in kg/m³."""
        return self.unit_wt / GRAVITY

    @property
    def damping_min(self) -> float:
        """Return the small-strain damping."""
        try:
            value = self.damping.values[0]
        except AttributeError:
            value = self.damping

        return value

    @property
    def quality(self) -> float:
        return 1 / (2 * self.damping_min)

    @property
    def unit_wt(self) -> float:
        """Unit weight of the soil in kN/m³."""
        return self._unit_wt

    @property
    def is_nonlinear(self) -> bool:
        """If nonlinear properties are specified."""
        return any(
            isinstance(p, NonlinearCurve) for p in [self.mod_reduc, self.damping]
        )

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.__dict__.values())

# TODO: for nonlinear site response this class wouldn't be used. Better way
# to do this? Maybe have the calculator create it?
class IterativeValue:
    def __init__(self, value: float | npt.ArrayLike):
        self._value = value
        self._previous = 1e-9

    @property
    def value(self) -> float | np.ndarray:
        return self._value

    @value.setter
    def value(self, value) -> float | np.ndarray:
        self._previous = self._value
        self._value = value

    @property
    def previous(self):
        return self._previous

    @property
    def relative_error(self) -> float:
        """The relative error, in decimal, between the two iterations."""
        if np.all(self.value > 0):
            err = np.max((self.previous - self.value) / self.value)
        elif np.isclose(self.value, self.previous).all():
            # When value is zero and close to previous
            err = 0
        else:
            err = np.inf

        return err

    def reset(self):
        self._previous = None


class Layer:
    """Docstring for Layer."""

    @convert_units(
        thickness="meter", shear_vel="meter / second", damping_min="dimensionless"
    )
    def __init__(
        self,
        soil_type: SoilType,
        thickness: float,
        shear_vel: float,
        damping_min: None | float = None,
        poissons_ratio: None | float = None,
    ):
        """@todo: to be defined!"""
        self._profile = None

        self._soil_type = soil_type

        self._thickness = thickness
        self._depth = 0
        self._stress_vert = 0

        # Need to set the initial dynamic properties prior to reseeting the
        # layer which creates the iterative values
        self._initial_shear_vel = shear_vel

        if damping_min is not None:
            self._damping_min = damping_min
        else:
            self._damping_min = soil_type.damping_min

        self._poissons_ratio = poissons_ratio

        self.reset()

    def __repr__(self) -> str:
        index = self._profile.index(self) if self._profile else None

        shear_vel = self._initial_shear_vel
        thickness = self._thickness
        st_name = self.soil_type.name
        damping_min = self._damping_min

        return (
            f"<Layer(index={index}, "
            f"shear_vel={shear_vel:0.1f} m/s, "
            f"thickness={thickness:0.1f} m, "
            f"soil_type={st_name}, "
            f"damping_min={damping_min:0.2f})>"
        )

    def __eq__(self, other) -> bool:
        attrs = ["_soil_type", "_thickness", "initial_shear_vel"]
        return (type(self) is type(other)) and all(
            [getattr(self, a) == getattr(other, a) for a in attrs]
        )

    def __hash__(self):
        return hash(self.__dict__.values())

    def copy(self) -> Layer:
        """Return a copy of the Layer instance with previously defined SoilType."""
        return Layer(
            self.soil_type,
            self.thickness,
            self.shear_vel,
            self.damping_min,
            self.poissons_ratio,
        )

    @property
    def depth(self) -> float:
        """Depth to the top of the layer [m]."""
        return self._depth

    @property
    def depth_mid(self) -> float:
        """Depth to the middle of the layer [m]."""
        return self._depth + self._thickness / 2

    @property
    def depth_base(self) -> float:
        """Depth to the base of the layer [m]."""
        return self._depth + self._thickness

    @property
    def poissons_ratio(self) -> float | None:
        """Poisson's ratio of the layer."""
        return self._poissons_ratio

    @poissons_ratio.setter
    def poissons_ratio(self, value: float | None):
        self._poissons_ratio = value

    @property
    def comp_vel(self) -> float | None:
        """Compression-wave velocity [m/s] derived from shear velocity and Poisson's
        ratio.

        Returns ``None`` if :attr:`poissons_ratio` is not set.
        """
        if self._poissons_ratio is None:
            return None
        nu = self._poissons_ratio
        return self.initial_shear_vel * np.sqrt(2 * (1 - nu) / (1 - 2 * nu))

    @property
    def density(self) -> float:
        """Density of soil in [kg/m³]."""
        return self.soil_type.density

    @property
    def damping_min(self) -> float:
        """Minimum damping of the soil [dec]"""
        return self._damping_min

    @damping_min.setter
    def damping_min(self, value: float):
        self._damping_min = value
        # Reset the iterated values
        self.reset()

    @property
    def damping(self) -> np.ndarray | float:
        """Strain-compatible damping."""
        try:
            value = self._damping.value
        except AttributeError:
            value = self._damping
        return value

    @property
    def initial_shear_mod(self) -> float:
        """Initial (small-strain) shear modulus [kN/m²]."""
        return self.density * self.initial_shear_vel**2

    @property
    def initial_shear_vel(self) -> float:
        """Initial (small-strain) shear-wave velocity [m/s]."""
        return self._initial_shear_vel

    @initial_shear_vel.setter
    def initial_shear_vel(self, value: float):
        """Set initial (small-strain) shear-wave velocity [m/s]."""

        self._initial_shear_vel = value
        # Reset the iterated values
        self.reset()

    @property
    def comp_shear_mod(self) -> complex:
        """Strain-compatible complex shear modulus [kN/m²]."""

        # Maximum damping value of less than 0.5
        damping = np.clip(self.damping, 0, 0.49)

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
    def comp_shear_vel(self) -> complex:
        """Strain-compatible complex shear-wave velocity [m/s]."""
        return np.sqrt(self.comp_shear_mod / self.density)

    @property
    def max_error(self) -> float:
        return max(self._shear_mod.relative_error, self._damping.relative_error)

    def reset(self):
        self._shear_mod = IterativeValue(self.initial_shear_mod)
        self._damping = IterativeValue(self.damping_min)
        # Use a small initial value
        self._strain = IterativeValue(1e-6)

        self.strain_max = None

    @property
    def shear_mod(self) -> np.ndarray | float:
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
        """Effective shear stress at layer midpoint."""
        return self.shear_mod * self.strain

    @property
    def stress_shear_max(self):
        """Maximum shear stress at layer midpoint."""
        return self.shear_mod * self.strain_max

    @property
    def strain(self):
        try:
            value = self._strain.value
        except AttributeError:
            value = self._strain

        return value

    def _compute_damping(self, strain) -> float:
        """Compute layer-adjusted damping at the given strain.

        The soil type's minimum damping is replaced with the layer-specific minimum
        damping.
        """
        try:
            damping = self.soil_type.damping(strain)
            damping -= self.soil_type.damping_min
        except TypeError:
            damping = 0.0

        return damping + self.damping_min

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

        # Update the damping value
        self._damping.value = self._compute_damping(strain)

    @property
    def adjusted_damping_curve(self) -> np.recarray:
        """Return the damping curve adjusted by the layer-specific minimum damping."""

        if isinstance(self.soil_type.damping, (float, int)):
            # No iteration provided by damping
            strains = np.asarray([np.nan])
            values = np.asarray([self.damping_min])
        else:
            strains = np.asarray(self.soil_type.damping.strains)
            values = np.array([self._compute_damping(s) for s in strains])

        return np.rec.array((strains, values), names=["strain", "damping"])

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
        return (2 * self.damping_min * self._thickness) / self.initial_shear_vel


class Location:
    """Location within a profile."""

    @convert_units(depth_within="meter")
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

    @convert_units(wt_depth="meter")
    def __init__(self, layers=None, wt_depth=0):
        super().__init__()
        self.layers = layers or []
        self.wt_depth = wt_depth
        if layers:
            self.update_layers()
            if logger.isEnabledFor(logging.DEBUG):
                max_depth = (
                    sum(layer.thickness for layer in self.layers[:-1])
                    if len(self.layers) > 1
                    else 0
                )
                logger.debug(
                    "Profile created: %d layers, max_depth=%.1fm, wt_depth=%.1fm",
                    len(self.layers),
                    max_depth,
                    self.wt_depth,
                )

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

    @classmethod
    @convert_units(layer_thickness="meter")
    def from_velocity_profile(
        cls,
        vp,
        soil_types,
        layer_thickness: float = 1.0,
        wt_depth: float = 0,
    ) -> "Profile":
        """Create from any object with .depth, .vs_median, .std_vs_ln.

        Duck-typed: accepts ``pygmm.contracts.VelocityProfile`` or any object
        with the same attributes.

        Parameters
        ----------
        vp :
            Velocity profile with ``.depth`` and ``.vs_median`` arrays.
        soil_types :
            A single :class:`SoilType` applied to every layer, or a list of
            :class:`SoilType` (one per depth point; last entry used for extras).
        layer_thickness :
            Fallback thickness [m] for the final half-space layer.
        wt_depth :
            Depth to the water table [m].
        """
        depth = np.asarray(vp.depth, dtype=float)
        vs_median = np.asarray(vp.vs_median, dtype=float)
        n = len(depth)

        thicknesses = np.diff(depth, append=depth[-1] + layer_thickness)

        def _get_st(i):
            if isinstance(soil_types, (list, tuple)):
                return soil_types[min(i, len(soil_types) - 1)]
            return soil_types

        layers = [
            Layer(_get_st(i), float(thicknesses[i]), float(vs_median[i]))
            for i in range(n)
        ]
        return cls(layers, wt_depth)

    def to_dataframe(self):
        records = []
        for layer in self:
            st = layer.soil_type
            records.append(
                (st.name, st.unit_wt, st.damping, layer.thickness, layer.shear_vel)
            )

        df = pd.DataFrame(
            records,
            columns=["soil_type", "unit_wt", "damping", "thickness", "shear_vel"],
        )
        df["depth"] = np.r_[0, df["thickness"].cumsum().iloc[:-1]]

        return df

    def __iter__(self):
        return iter(self.layers)

    def __contains__(self, value):
        return value in self.layers

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.layers[key]

    def copy(self):
        """Return a copy of the profile with new Layer instances."""
        return Profile([layer.copy() for layer in self], self.wt_depth)

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
        for layer in self[:-1]:
            if not nonlinear_only or layer.soil_type.is_nonlinear:
                opt_thickness = layer.shear_vel / max_freq * wave_frac
                count = max(np.ceil(layer.thickness / opt_thickness).astype(int), 1)
                thickness = layer.thickness / count
                for _ in range(count):
                    layers.append(
                        Layer(
                            layer.soil_type,
                            thickness,
                            layer.shear_vel,
                            layer.damping_min,
                        )
                    )
            else:
                layers.append(layer)
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

    def lookup_depth(self, depth: float) -> tuple[int, float]:
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

    def vs30(self):
        """Compute the Vs30 of the profile."""
        tot_time = np.r_[0, np.cumsum(self.thickness / self.initial_shear_vel)[:-1]]
        time = np.interp(30, self.depth, tot_time)
        return 30 / time

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

    @convert_units(freqs="hertz")
    def calc_dispersion(
        self,
        freqs,
        wave="rayleigh",
        mode=0,
        dc_type="phase",
    ):
        """Compute surface-wave dispersion curve via *disba*.

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz] at which to evaluate the dispersion curve.
        wave : str, optional
            Wave type: ``"rayleigh"`` (default) or ``"love"``.
        mode : int, optional
            Mode number (0 = fundamental, default).
        dc_type : str, optional
            ``"phase"`` (default) or ``"group"``.

        Returns
        -------
        np.ndarray
            Phase or group velocity [m/s] at each frequency.

        Raises
        ------
        ValueError
            If any layer is missing :attr:`Layer.poissons_ratio`.
        ImportError
            If *disba* is not installed.
        """
        try:
            from disba import GroupDispersion, PhaseDispersion
        except ImportError:
            raise ImportError(
                "The 'disba' package is required for dispersion calculations. "
                "Install it with: pip install disba"
            )

        if any(layer.poissons_ratio is None for layer in self):
            raise ValueError(
                "All layers must have poissons_ratio set to compute dispersion."
            )

        freqs = np.asarray(freqs, dtype=float)
        # disba expects periods sorted in ascending order
        periods = 1.0 / freqs
        sort_idx = np.argsort(periods)
        periods_sorted = periods[sort_idx]

        # Build velocity model: (thickness [km], Vp [km/s], Vs [km/s], density [g/cm³])
        thickness = self.thickness / 1e3
        comp_vel = self.comp_vel / 1e3
        shear_vel = self.initial_shear_vel / 1e3
        density = self.density / 1e3

        if dc_type == "phase":
            dc = PhaseDispersion(thickness, comp_vel, shear_vel, density)
        elif dc_type == "group":
            dc = GroupDispersion(thickness, comp_vel, shear_vel, density)
        else:
            raise ValueError(f"dc_type must be 'phase' or 'group', got {dc_type!r}")

        result = dc(periods_sorted, mode=mode, wave=wave)
        # result.velocity is in km/s; convert back to m/s and restore
        # original frequency ordering
        velocity = np.empty_like(result.velocity)
        velocity[sort_idx] = result.velocity
        return velocity * 1e3

    def plot(self, prop, ax=None, plot_kwds=None, axis_kwds=None):
        # Defaults
        xlabels = {
            "damping": "Damping (dec)",
            "density": "Density (kg/m³)",
            "initial_shear_vel": "Initial $V_s$ (m/s)",
            "max_error": "Max. Error (%)",
            "shear_vel": "$V_s$ (m/s)",
            "slowness": "Slowness (1/s)",
            "strain": "Strain (dec)",
            "travel_time": "Travel time (sec)",
            "unit_wt": "Unit Wt. (kN/m³)",
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
            _, ax = plt.subplots()

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
    def comp_vel(self):
        return self._get_values("comp_vel")

    @property
    def poissons_ratio(self):
        return self._get_values("poissons_ratio")

    @property
    def strain(self):
        return self._get_values("strain")

    @property
    def unit_wt(self):
        return self._get_values("unit_wt")

    @property
    def comp_shear_mod(self):
        return self._get_values("comp_shear_mod")

    @property
    def comp_shear_vel(self):
        return self._get_values("comp_shear_vel")

    def _get_values(self, attr):
        return np.array([getattr(layer, attr) for layer in self])
