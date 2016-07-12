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
# Copyright (C) Albert Kottke, 2013-2015

from pkg_resources import get_distribution

# Gravity in m/sec²
# Source: http://physics.nist.gov/cgi-bin/cuu/Value?gn
GRAVITY = 9.80665

from . import motion
from . import propagation
from . import site
from . import variation

__all__ = [
    'motion',
    'propagation',
    'site',
    'variation'
]

__author__ = 'Albert Kottke'
__copyright__ = 'Copyright 2016 Albert Kottke'
__license__ = 'MIT'
__title__ = 'pySRA'
__version__ = get_distribution('pySRA').version



