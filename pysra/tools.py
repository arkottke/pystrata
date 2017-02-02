#!/usr/bin/env python
# encoding: utf-8

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# Copyright (C) Albert Kottke, 2013-2016

import collections
import os
import re

import numpy as np

from . import site
from . import motion
from . import propagation


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
            line = lines.pop(0)

        values.append(parser(line[:width]))
        line = line[width:]

    return values


def _parse_curves(block, **kwargs):
    """Parse nonlinear curves block."""
    count = int(block.pop(0))

    curves = []
    for i in range(count):
        for param in ['mod_reduc', 'damping']:
            length, name = parse_fixed_width([(5, int), (65, to_str)], block)
            curves.append(
                site.NonlinearProperty(
                    name,
                    parse_fixed_width(length * [(10, float)], block),
                    parse_fixed_width(length * [(10, float)], block),
                    param
                )
            )

    length = int(block[0][:5])
    soil_types = parse_fixed_width((length + 1) * [(5, int)], block)[1:]

    # Group soil type number and curves together
    return {(soil_types[i // 2], c.param): c for i, c in enumerate(curves)}


def _parse_soil_profile(block, units, curves, **kwargs):
    """Parse soil profile block."""
    wt_layer, length, _, name = parse_fixed_width(
        3 * [(5, int)] + [(55, to_str)], block)

    layers = []
    soil_types = []
    for i in range(length):
        index, soil_idx, thickness, shear_mod, damping, unit_wt, shear_vel = \
            parse_fixed_width(
                [(5, int), (5, int), (15, to_float)] + 4 * [(10, to_float)],
                block
            )

        st = site.SoilType(
            soil_idx,
            unit_wt,
            curves[(soil_idx, 'mod_reduc')],
            curves[(soil_idx, 'damping')],
        )
        try:
            # Try to find previously added soil type
            st = soil_types[soil_types.index(st)]
        except ValueError:
            soil_types.append(st)

        layers.append(site.Layer(
            st, thickness, shear_vel
        ))

    if units == 'english':
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
        [(5, int), (5, int), (10, float), (30, to_str), (30, to_str)],
        block
    )

    scale, pga, _, header_lines, _ = parse_fixed_width(
        3 * [(10, to_float)] + 2 * [(5, int)],
        block
    )

    m = re.search(r'(\d+)\w(\d+)\.\d+', fmt)
    count_per_line = int(m.group(1))
    width = int(m.group(2))

    fname = os.path.join(os.path.dirname(kwargs['fname']), name)
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
        scale = 1.

    accels *= scale
    m = motion.TimeSeriesMotion(fname, '', time_step, accels, fa_length)

    return m


def _parse_input_loc(block, profile, **kwargs):
    """Parse input location block."""
    layer, wave_field = parse_fixed_width(
        2 * [(5, int)], block
    )

    return profile.location(
        motion.WaveField[wave_field],
        index=(layer - 1),
    )


def _parse_run_control(block):
    """Parse run control block."""
    _, max_iterations, strain_ratio, _, _ = parse_fixed_width(
        2 * [(5, int)] + [(10, float)] + 2 * [(5, int)],
        block
    )

    return propagation.EquivalentLinearCalculation(
        strain_ratio, max_iterations, tolerance=10.
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
        m = re.match(r'^\s+(\d+)$', l)

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
        1: ('curves', _parse_curves),
        2: ('profile', _parse_soil_profile),
        3: ('motion', _parse_motion),
        4: ('input_loc', _parse_input_loc),
        5: ('run_control', _parse_run_control),
        6: ('output', _parse_output_accel),
        7: ('output', _parse_output_stress),
        9: ('output', _parse_output_spectra),
    }

    input = collections.OrderedDict({
        'fname': fname,
        'units': units,
    })
    for option, block in options:
        key, parser = parsers[option]
        input[key] = parser(block, **input)

    return input
