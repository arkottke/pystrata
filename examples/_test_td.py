"""Quick test of TD-MKZ and TD-HH after fixes."""

import numpy as np

import pystrata

MOTION_FILE = "data/NIS090.AT2"
_base = pystrata.motion.TimeSeriesMotion.load_at2_file(MOTION_FILE)
motion_low = pystrata.motion.TimeSeriesMotion.load_at2_file(
    MOTION_FILE, scale=0.01 / _base.pga
)
motion_high = pystrata.motion.TimeSeriesMotion.load_at2_file(
    MOTION_FILE, scale=0.50 / _base.pga
)

profile = pystrata.site.Profile(
    [
        pystrata.site.Layer(
            pystrata.site.DarendeliSoilType(
                unit_wt=18.0, plas_index=20, ocr=1.0, stress_mean=50
            ),
            thickness=10,
            shear_vel=200,
        ),
        pystrata.site.Layer(
            pystrata.site.DarendeliSoilType(
                unit_wt=19.0, plas_index=15, ocr=1.0, stress_mean=150
            ),
            thickness=10,
            shear_vel=350,
        ),
        pystrata.site.Layer(
            pystrata.site.SoilType("Rock", unit_wt=24.0, mod_reduc=None, damping=0.01),
            thickness=0,
            shear_vel=700,
        ),
    ]
).auto_discretize(max_freq=50, wave_frac=0.2)

loc_in = profile.location("outcrop", index=-1)
loc_surf = profile.location("outcrop", index=0)
freqs = np.array([1.0, 5.0, 10.0, 50.0])

for label, motion in [("low", motion_low), ("high", motion_high)]:
    print(f"\n--- {label} intensity (PGA={motion.pga:.2f}g) ---")
    sa_input = motion.calc_osc_accels(freqs, 0.05)

    calc_mkz = pystrata.propagation.TimeDomainCalculator(
        model="mkz", boundary="elastic"
    )
    calc_mkz(motion, profile, loc_in)
    pga_mkz = calc_mkz.calc_peak_accel(loc_surf)
    sa_mkz = calc_mkz.calc_osc_accels(loc_surf, freqs, 0.05)
    print(f"  TD-MKZ PGA: {pga_mkz:.4f} g ({pga_mkz / motion.pga:.2f}x)")

    calc_hh = pystrata.propagation.TimeDomainCalculator(model="hh", boundary="elastic")
    calc_hh(motion, profile, loc_in)
    pga_hh = calc_hh.calc_peak_accel(loc_surf)
    sa_hh = calc_hh.calc_osc_accels(loc_surf, freqs, 0.05)
    print(f"  TD-HH  PGA: {pga_hh:.4f} g ({pga_hh / motion.pga:.2f}x)")

    print("  Sa amplif (1/5/10/50 Hz):")
    for i, f in enumerate(freqs):
        if sa_input[i] > 1e-6:
            print(
                f"    {f:.0f}Hz: MKZ={sa_mkz[i] / sa_input[i]:.2f}x  HH={sa_hh[i] / sa_input[i]:.2f}x"
            )
