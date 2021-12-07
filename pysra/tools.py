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
import collections
import os
import re

import numpy as np
import pandas as pd
import scipy.constants as C

from . import motion
from . import propagation
from . import site


def to_str(s):
    """Parse a string and strip the extra characters."""
    return str(s).strip()


def to_float(s):
    """Try to parse a float."""
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_fixed_width(types, lines):
    """Parse a fixed width line."""
    values = []
    line = []
    for width, parser in types:
        if not line:
            line = lines.pop(0).replace("\n", "")

        values.append(parser(line[:width]))
        line = line[width:]

    return values


def split_line(line, parsers, sep=" "):
    """Split a line into pieces and parse the strings."""
    parts = [part for part in line.split(sep) if part]
    values = [parser(part) for parser, part in zip(parsers, parts)]
    return values if len(values) > 1 else values[0]


def _parse_curves(block, **kwargs):
    """Parse nonlinear curves block."""
    count = int(block.pop(0))

    curves = []
    for i in range(count):
        for param in ["mod_reduc", "damping"]:
            length, name = parse_fixed_width([(5, int), (65, to_str)], block)
            curves.append(
                site.NonlinearProperty(
                    name,
                    parse_fixed_width(length * [(10, float)], block),
                    parse_fixed_width(length * [(10, float)], block),
                    param,
                )
            )

    length = int(block[0][:5])
    soil_types = parse_fixed_width((length + 1) * [(5, int)], block)[1:]

    # Group soil type number and curves together
    return {(soil_types[i // 2], c.param): c for i, c in enumerate(curves)}


def _parse_soil_profile(block, units, curves, **kwargs):
    """Parse soil profile block."""
    wt_layer, length, _, name = parse_fixed_width(
        3 * [(5, int)] + [(55, to_str)], block
    )

    layers = []
    soil_types = []
    for i in range(length):
        (
            index,
            soil_idx,
            thickness,
            shear_mod,
            damping,
            unit_wt,
            shear_vel,
        ) = parse_fixed_width(
            [(5, int), (5, int), (15, to_float)] + 4 * [(10, to_float)], block
        )

        st = site.SoilType(
            soil_idx,
            unit_wt,
            curves[(soil_idx, "mod_reduc")],
            curves[(soil_idx, "damping")],
        )
        try:
            # Try to find previously added soil type
            st = soil_types[soil_types.index(st)]
        except ValueError:
            soil_types.append(st)

        layers.append(site.Layer(st, thickness, shear_vel))

    if units == "english":
        # Convert from English to metric
        for st in soil_types:
            st.unit_wt *= 0.00015708746

        for l in layers:
            l.thickness *= 0.3048
            l.shear_vel *= 0.3048

    p = site.Profile(layers)
    p.update_layers()
    p.wt_depth = p[wt_layer - 1].depth

    return p


def _parse_motion(block, **kwargs):
    """Parse motin specification block."""
    _, fa_length, time_step, name, fmt = parse_fixed_width(
        [(5, int), (5, int), (10, float), (30, to_str), (30, to_str)], block
    )

    scale, pga, _, header_lines, _ = parse_fixed_width(
        3 * [(10, to_float)] + 2 * [(5, int)], block
    )

    m = re.search(r"(\d+)\w(\d+)\.\d+", fmt)
    count_per_line = int(m.group(1))
    width = int(m.group(2))

    fname = os.path.join(os.path.dirname(kwargs["fname"]), name)
    accels = np.genfromtxt(
        fname,
        delimiter=(count_per_line * [width]),
        skip_header=header_lines,
    )

    if np.isfinite(scale):
        pass
    elif np.isfinite(pga):
        scale = pga / np.abs(accels).max()
    else:
        scale = 1.0

    accels *= scale
    m = motion.TimeSeriesMotion(fname, "", time_step, accels, fa_length)

    return m


def _parse_input_loc(block, profile, **kwargs):
    """Parse input location block."""
    layer, wave_field = parse_fixed_width(2 * [(5, int)], block)

    return profile.location(
        motion.WaveField[wave_field],
        index=(layer - 1),
    )


def _parse_run_control(block):
    """Parse run control block."""
    _, max_iterations, strain_ratio, _, _ = parse_fixed_width(
        2 * [(5, int)] + [(10, float)] + 2 * [(5, int)], block
    )

    return propagation.EquivalentLinearCalculation(
        strain_ratio, max_iterations, tolerance=10.0
    )


def _parse_output_accel(block):
    raise NotImplementedError


def _parse_output_stress(block):
    raise NotImplementedError


def _parse_output_spectra(block):
    raise NotImplementedError


def load_shake_inp(fname):
    with open(fname) as fp:
        lines = fp.readlines()

    lines.pop(0)
    units = lines.pop(0)

    # Parse the option blocks
    option, block = None, []
    options = []
    for l in lines:
        m = re.match(r"^\s+(\d+)$", l)

        if m:
            if option and not block:
                block.append(l)
            else:
                if option and block:
                    # Save the previous block
                    options.append((option, block))
                block = []
                option = int(m.group(1))
        else:
            block.append(l)

    parsers = {
        1: ("curves", _parse_curves),
        2: ("profile", _parse_soil_profile),
        3: ("motion", _parse_motion),
        4: ("input_loc", _parse_input_loc),
        5: ("run_control", _parse_run_control),
        6: ("output", _parse_output_accel),
        7: ("output", _parse_output_stress),
        9: ("output", _parse_output_spectra),
    }

    input = collections.OrderedDict(
        {
            "fname": fname,
            "units": units,
        }
    )
    for option, block in options:
        key, parser = parsers[option]
        input[key] = parser(block, **input)

    return input


def read_nrattle_ctl(fpath):
    """Read an nrattle control file."""
    lines = list(fpath.open())
    lines = [line for line in lines if line[0] != "!"]

    d = {k: lines.pop(0) for k in ["revision", "prefix"]}
    d["freq_count"], d["freq_max"] = split_line(lines.pop(0), [int, float])
    d["out_depth"] = split_line(
        lines.pop(0),
        [
            float,
        ],
    )

    profile = []
    while line := lines.pop(0):
        try:
            profile.append(split_line(line, [int, float, float, float, float]))
        except ValueError:
            break

    d["profile"] = np.rec.fromrecords(
        profile, names="layer,thickness,vel_shear,density,inv_qual"
    )
    d["hs_vel_shear"], d["hs_density"] = split_line(line, [float, float])
    d["hs_layer"], d["inci_angle"] = split_line(lines.pop(0), [int, float])

    return d


def profile_from_nrattle_ctl(ctl):
    df = pd.DataFrame(ctl["profile"]).set_index("layer")
    # Here the index is based on layer number which starts at 1 in Fortran
    # convention.
    df.loc[len(df) + 1] = [0, ctl["hs_vel_shear"], ctl["hs_density"], 0]

    # Scale from km to m
    df["vel_shear"] *= 1000
    df["thickness"] *= 1000
    # Convert Q to damping:
    # damping (dec) = 0.5 * 1 / Q = 1 / (2 * Q)
    df["damping"] = df["inv_qual"].apply(
        lambda iq: 0 if np.isclose(iq, 0) else 0.5 * 1 / (iq if iq > 1 else 1 / iq)
    )
    df["unit_wt"] = C.g * df["density"]

    return site.Profile.from_dataframe(df, 0)
