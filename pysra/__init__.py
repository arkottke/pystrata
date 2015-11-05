from numpy.testing import Tester

test = Tester().test

# Gravity in m/sec²
# Source: http://physics.nist.gov/cgi-bin/cuu/Value?gn
GRAVITY = 9.80665

from . import motion
from . import propagation
from . import site
from . import variation