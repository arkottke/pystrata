"""Regenerate all reference data in tests/data/comparison/ with current code."""

import pathlib

import numpy as np

import pystrata
from pystrata.propagation import EquivalentLinearCalculator, TimeDomainCalculator

DATA_DIR = pathlib.Path("tests/data/comparison")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MOTION_FILE = "examples/data/NIS090.AT2"


def _build_profile():
    return pystrata.site.Profile(
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
                pystrata.site.SoilType(
                    "Rock", unit_wt=24.0, mod_reduc=None, damping=0.01
                ),
                thickness=0,
                shear_vel=700,
            ),
        ]
    )


# Load base motion
_base = pystrata.motion.TimeSeriesMotion.load_at2_file(MOTION_FILE)

TARGET_LOW = 0.01
TARGET_HIGH = 0.50

motion_low = pystrata.motion.TimeSeriesMotion.load_at2_file(
    MOTION_FILE, scale=TARGET_LOW / _base.pga
)
motion_high = pystrata.motion.TimeSeriesMotion.load_at2_file(
    MOTION_FILE, scale=TARGET_HIGH / _base.pga
)

freqs = np.logspace(-1, 2, 300)
loc_surface = pystrata.output.OutputLocation("outcrop", index=0)
loc_input = pystrata.output.OutputLocation("outcrop", index=-1)

for label, motion in [("low", motion_low), ("high", motion_high)]:
    print(f"\n=== {label.upper()} intensity (PGA={motion.pga:.2f} g) ===")
    profile = _build_profile()
    disc = profile.auto_discretize(max_freq=50, wave_frac=0.2)

    # Save input motion
    np.savez(
        DATA_DIR / f"motion_{label}.npz",
        times=motion.times,
        accels=motion.accels,
        dt=np.array([motion.time_step]),
        pga=np.array([motion.pga]),
    )

    rs_data = {"freqs": freqs}
    ratio_data = {"freqs": freqs}
    strain_data = {}

    # EQL
    print("  Running EQL...")
    calc_eql = EquivalentLinearCalculator(strain_ratio=0.65)
    loc_in = disc.location("outcrop", index=-1)
    calc_eql(motion, disc, loc_in)

    outputs_eql = pystrata.output.OutputCollection(
        [
            pystrata.output.ResponseSpectrumOutput(freqs, loc_surface, 0.05),
            pystrata.output.ResponseSpectrumRatioOutput(
                freqs, loc_input, loc_surface, 0.05
            ),
            pystrata.output.MaxStrainProfile(),
        ]
    )
    outputs_eql(calc_eql, "EQL")
    for name, refs, values in outputs_eql[0].iter_results():
        rs_data["EQL"] = values
    for name, refs, values in outputs_eql[1].iter_results():
        ratio_data["EQL"] = values
    for name, depths, strains in outputs_eql[2].iter_results():
        if "depths" not in strain_data:
            strain_data["depths"] = depths
        strain_data["EQL"] = strains

    # TD methods
    for model in ["mkz", "hh"]:
        print(f"  Running TD-{model.upper()}...")
        disc2 = _build_profile().auto_discretize(max_freq=50, wave_frac=0.2)
        calc_td = TimeDomainCalculator(model=model, boundary="elastic")
        loc_in = disc2.location("outcrop", index=-1)
        calc_td(motion, disc2, loc_in)

        outputs_td = pystrata.output.OutputCollection(
            [
                pystrata.output.ResponseSpectrumOutput(freqs, loc_surface, 0.05),
                pystrata.output.ResponseSpectrumRatioOutput(
                    freqs, loc_input, loc_surface, 0.05
                ),
                pystrata.output.MaxStrainProfile(),
            ]
        )
        outputs_td(calc_td, f"TD-{model.upper()}")

        key = f"TD_{model.upper()}"
        for name, refs, values in outputs_td[0].iter_results():
            rs_data[key] = values
        for name, refs, values in outputs_td[1].iter_results():
            ratio_data[key] = values
        for name, depths, strains in outputs_td[2].iter_results():
            if "depths" not in strain_data:
                strain_data["depths"] = depths
            strain_data[key] = strains

        # Save surface time series
        loc_surf = calc_td.profile.location("outcrop", index=0)
        ts_accels = calc_td.accel_ts(loc_surf)
        fname = f"surface_ts_{label}_td_{model}.npz"
        np.savez(DATA_DIR / fname, times=calc_td.times, accels=ts_accels)

    np.savez(DATA_DIR / f"response_spectra_{label}.npz", **rs_data)
    np.savez(DATA_DIR / f"spectral_ratios_{label}.npz", **ratio_data)
    np.savez(DATA_DIR / f"max_strains_{label}.npz", **strain_data)

# Save profile metadata
profile_data = {
    "thicknesses": np.array([10.0, 10.0, 0.0]),
    "shear_vels": np.array([200.0, 350.0, 700.0]),
    "unit_wts": np.array([18.0, 19.0, 24.0]),
    "plas_indexes": np.array([20.0, 15.0, 0.0]),
    "stress_means": np.array([50.0, 150.0, 0.0]),
    "ocrs": np.array([1.0, 1.0, 1.0]),
    "dampings": np.array([0.05, 0.05, 0.01]),
}
np.savez(DATA_DIR / "profile.npz", **profile_data)

print("\n\nSaved files:")
for f in sorted(DATA_DIR.glob("*.npz")):
    print(f"  {f.name}")
print("\nDone!")
