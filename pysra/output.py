


class Output(object):


    def __call__(self, site, motion):
        # Save results


class LocationOutput(Output):
    def __init__(self, depth, wave_field):
        super().__init__(self)

        self._depth = depth
        self._wave_field = wave_field

    @property
    def depth(self):
        return self._depth

    @property
    def wave_field(self):
        return self._wave_field


class ResponseSpectrumOutput(Output):
    def __init__(self, depth, wave_field):
        super().__init__(self)

    def __call__(self, profile, motion, calc):
        loc = profile.location( self.wave_field, depth=self.depth)
        tf = calc.calc_accel_tf(loc_bedrock, loc_surface))
ars_surface = motion.calc_osc_accels(
    osc_freqs, osc_damping,


