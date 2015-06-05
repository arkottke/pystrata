
import matplotlib.pyplot as plt
import numpy as np

from .. import motion
from .. import propagation
from .. import site

gravity = 9.81

soil = site.SoilType('soil', 17, gravity, 1., 0.05)
rock = site.SoilType('rock', 23, gravity, 1., 0.01)
layers = [
    site.Layer(soil, 40, 400),
    site.Layer(rock, 0, 1500),
]
profile = site.Profile(layers)

motion = motion.SourceTheoryRvtMotion(6, 20, 'wna')
motion.compute_fourier_amps()

loc_in = profile.location(layers[0].depth_base, 'outcrop')
loc_out = profile.location(0, 'outcrop')

calculator = propagation.LinearElasticCalculator()
calculator.calc_waves(motion, profile.layers)

trans_func = calculator.calc_accel_tf(loc_in, loc_out)

fig, ax = plt.subplots(1, 1, subplot_kw=dict(xscale='log'))
ax.plot(motion.freqs, np.abs(trans_func))

fig.tight_layout()
fig.savefig('test.png')

