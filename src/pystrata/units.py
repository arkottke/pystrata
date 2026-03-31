# The MIT License (MIT)
#
# Copyright (c) 2016-2026 Albert Kottke
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
"""Pint unit support for pystrata.

Provides a shared :class:`pint.UnitRegistry` and a :func:`convert_units`
decorator that transparently converts :class:`pint.Quantity` arguments to the
expected units and extracts their magnitude.
"""

from __future__ import annotations

import inspect
from functools import wraps

import pint

#: Shared unit registry — use this for all pystrata Quantity values.
ureg = pint.UnitRegistry()

# Shorthand units for seismology
ureg.define("gravity_second = standard_gravity * second = g_s")
ureg.define("gravity = standard_gravity = g_n")

#: Acceleration due to gravity [m/s²]
GRAVITY = (1 * ureg.standard_gravity).to("meter / second ** 2").magnitude

#: Conversion factor from kilopascals to atmospheres
KPA_TO_ATM = (1 * ureg.kilopascal).to(ureg.atmosphere).magnitude


def convert_units(**unit_specs: str):
    """Decorator that converts :class:`pint.Quantity` arguments to expected units.

    For each keyword argument in *unit_specs*, if the corresponding function
    argument is a :class:`pint.Quantity` it is converted to the target unit and
    its magnitude is extracted.  Plain numeric values pass through unchanged.

    Parameters
    ----------
    **unit_specs
        Mapping of parameter name → pint-compatible unit string.
        Example: ``@convert_units(thickness="meter", shear_vel="meter/second")``

    Notes
    -----
    * Works with both positional and keyword arguments via
      :meth:`inspect.Signature.bind`.
    * ``None`` values are passed through unchanged (for optional parameters).
    * Incompatible units raise :class:`pint.DimensionalityError` automatically.

    Examples
    --------
    >>> @convert_units(thickness="meter")
    ... def make_layer(thickness):
    ...     return thickness
    >>> make_layer(5 * ureg.feet)  # doctest: +SKIP
    1.524...
    """

    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name, target_unit in unit_specs.items():
                if name not in bound.arguments:
                    continue
                value = bound.arguments[name]
                if value is None:
                    continue
                if isinstance(value, pint.Quantity):
                    bound.arguments[name] = value.to(target_unit).magnitude

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def convert_kwds_units(**unit_specs: str):
    """Decorator that converts :class:`pint.Quantity` values inside ``**kwds``.

    This is for functions that accept ``**kwds`` dictionaries where some values
    may be :class:`pint.Quantity` objects (e.g. ``WangSoilType``).

    Parameters
    ----------
    **unit_specs
        Mapping of keyword name → pint-compatible unit string.
        Example: ``@convert_kwds_units(stress_mean="kilopascal")``
    """

    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Scan all bound arguments for matching keys
            for name, target_unit in unit_specs.items():
                if name in bound.arguments:
                    value = bound.arguments[name]
                    if isinstance(value, pint.Quantity):
                        bound.arguments[name] = value.to(target_unit).magnitude
                # Also check inside **kwargs captured as a dict
                for arg_name, arg_value in list(bound.arguments.items()):
                    if isinstance(arg_value, dict) and name in arg_value:
                        v = arg_value[name]
                        if isinstance(v, pint.Quantity):
                            arg_value[name] = v.to(target_unit).magnitude

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
