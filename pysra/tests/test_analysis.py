import matplotlib.pyplot as plt
import numpy as np

import pysra

soil = pysra.site.SoilType('soil', 17, None, 0.05)
rock = pysra.site.SoilType('rock', 23, None, 0.01)
layers = [
    pysra.site.Layer(soil, 40, 400),
    pysra.site.Layer(rock, 0, 1500),
]
profile = pysra.site.Profile(layers)

motion = pysra.motion.SourceTheoryRvtMotion(6, 20, 'wna')
motion.compute_fourier_amps()

loc_in = profile.location('outcrop', index=-1)
loc_out = profile.location('outcrop', index=0)

calculator = pysra.propagation.LinearElasticCalculator()
calculator.calc_waves(motion, profile, loc_in)

trans_func = calculator.calc_accel_tf(loc_in, loc_out)

fig, ax = plt.subplots(1, 1, subplot_kw=dict(xscale='log'))
ax.plot(motion.freqs, np.abs(trans_func))

fig.tight_layout()
fig.savefig('test.png')

