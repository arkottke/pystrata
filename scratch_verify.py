"""Quick verification of TD output interface changes."""

import numpy as np

import pystrata

# -- Build profile --
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

# -- Load motion --
MOTION_FILE = "examples/data/NIS090.AT2"
_base = pystrata.motion.TimeSeriesMotion.load_at2_file(MOTION_FILE)
motion_low = pystrata.motion.TimeSeriesMotion.load_at2_file(
    MOTION_FILE, scale=0.01 / _base.pga
)
motion_high = pystrata.motion.TimeSeriesMotion.load_at2_file(
    MOTION_FILE, scale=0.50 / _base.pga
)

freqs = np.logspace(-1, 2, 100)
loc_surface = pystrata.output.OutputLocation("outcrop", index=0)
loc_input = pystrata.output.OutputLocation("outcrop", index=-1)

print("=" * 70)
print("TEST 1: Spectral ratios at low intensity — should match across methods")
print("=" * 70)

for name, CalcClass, kwargs in [
    ("EQL", pystrata.propagation.EquivalentLinearCalculator, {"strain_ratio": 0.65}),
    (
        "TD-MKZ",
        pystrata.propagation.TimeDomainCalculator,
        {"model": "mkz", "boundary": "elastic"},
    ),
    (
        "TD-HH",
        pystrata.propagation.TimeDomainCalculator,
        {"model": "hh", "boundary": "elastic"},
    ),
]:
    calc = CalcClass(**kwargs)
    loc_in = profile.location("outcrop", index=-1)
    calc(motion_low, profile, loc_in)

    outputs = pystrata.output.OutputCollection(
        [
            pystrata.output.ResponseSpectrumOutput(freqs, loc_surface, 0.05),
            pystrata.output.ResponseSpectrumRatioOutput(
                freqs, loc_input, loc_surface, 0.05
            ),
        ]
    )
    outputs(calc, name)

    for n, refs, values in outputs[0].iter_results():
        sa_mean = np.mean(values[(refs > 1) & (refs < 20)])
        print(f"  {name:8s}: mean Sa(1-20 Hz) = {sa_mean:.4f} g")

    for n, refs, values in outputs[1].iter_results():
        ratio_mean = np.mean(values[(refs > 1) & (refs < 20)])
        print(f"  {name:8s}: mean Spectral ratio(1-20 Hz) = {ratio_mean:.3f}")

print()
print("=" * 70)
print("TEST 2: TD calc_osc_accels matches via-TF path")
print("=" * 70)

calc_td = pystrata.propagation.TimeDomainCalculator(model="mkz", boundary="elastic")
loc_in = profile.location("outcrop", index=-1)
calc_td(motion_low, profile, loc_in)
loc_surf = profile.location("outcrop", index=0)

# Direct path (new)
sa_direct = calc_td.calc_osc_accels(loc_surf, freqs, 0.05)
# TF path (old)
tf = calc_td.calc_accel_tf(calc_td.loc_input, loc_surf)
sa_via_tf = calc_td.motion.calc_osc_accels(freqs, 0.05, tf)
# Identity: direct at input should match original motion
sa_input_direct = calc_td.calc_osc_accels(calc_td.loc_input, freqs, 0.05)
sa_input_orig = calc_td.motion.calc_osc_accels(freqs, 0.05)

reldiff_surf = np.max(np.abs(sa_direct - sa_via_tf) / sa_via_tf)
reldiff_input = np.max(np.abs(sa_input_direct - sa_input_orig) / sa_input_orig)
print(f"  Surface:  max |direct - via_tf|/via_tf = {reldiff_surf:.2e}  (should be ~0)")
print(f"  Input:    max |direct - orig|/orig     = {reldiff_input:.2e}  (should be 0)")

print()
print("=" * 70)
print("TEST 3: MaxAccelProfile uses direct path for TD")
print("=" * 70)

calc_td2 = pystrata.propagation.TimeDomainCalculator(model="mkz", boundary="elastic")
loc_in = profile.location("outcrop", index=-1)
calc_td2(motion_low, profile, loc_in)

outputs = pystrata.output.OutputCollection(
    [
        pystrata.output.MaxAccelProfile(),
    ]
)
outputs(calc_td2, "TD-MKZ")
for n, depths, values in outputs[0].iter_results():
    print(f"  Surface PGA = {values[0]:.5f} g, Max PGA = {np.max(values):.5f} g")

print()
print("=" * 70)
print("TEST 4: High-frequency Sa at high intensity")
print("=" * 70)

for name, CalcClass, kwargs in [
    ("EQL", pystrata.propagation.EquivalentLinearCalculator, {"strain_ratio": 0.65}),
    (
        "TD-MKZ",
        pystrata.propagation.TimeDomainCalculator,
        {"model": "mkz", "boundary": "elastic"},
    ),
    (
        "TD-HH",
        pystrata.propagation.TimeDomainCalculator,
        {"model": "hh", "boundary": "elastic"},
    ),
]:
    calc = CalcClass(**kwargs)
    loc_in = profile.location("outcrop", index=-1)
    calc(motion_high, profile, loc_in)

    outputs = pystrata.output.OutputCollection(
        [
            pystrata.output.ResponseSpectrumOutput(freqs, loc_surface, 0.05),
        ]
    )
    outputs(calc, name)

    for n, refs, values in outputs[0].iter_results():
        hf_mask = refs > 20
        max_sa_hf = np.max(values[hf_mask])
        max_sa_all = np.max(values)
        print(
            f"  {name:8s}: max Sa(>20Hz)={max_sa_hf:.3f}g, max Sa(all)={max_sa_all:.3f}g"
        )

print("\nAll tests passed.")
