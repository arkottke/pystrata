# Process to Fix Time-Domain Site Response in pyStrata

This document outlines a systematic process to identify and resolve the current issues in the time-domain implementation of pyStrata, specifically focusing on the damping models proposed by **Ji and Archuleta (2007)** and the nonlinear formulations of **Shi and Asimaki (2017)**.

## 1. Problem Identification & Benchmarking

The current implementation shows excessive amplification (e.g., PGA > 7g for 0.1g input) and fails regression tests.

- **Establish a Baseline**: Use [PySeismoSoil](https://github.com/Asimaki-Group/PySeismoSoil), the official implementation of the HH model and the research from the Asimaki group.
- **Linear-Elastic Verification**: Disable nonlinear behavior and verify if the `propagate_time_domain` and `propagate_nonlinear` (with linear parameters) match exactly. If they don't, the issue is in the fundamental integration or boundary conditions.
- **Isolate Damping**: Run a case with 0% damping to verify the backbone stress-strain integration. Then add damping and observe the stability.

## 2. Implement Ji and Archuleta (2007) Damping

The current code uses the **Liu and Archuleta (2006)** approximation for weight coefficients ($\chi$), which is known to be inaccurate for $Q < 20$ (damping > 2.5%).

- **Least-Squares Weights**: Instead of the $\chi$ formula in `_compute_la_weights`, implement the least-squares fitting of mechanism weights $w_k$ to achieve a constant $Q$ over the target frequency range (typically 0.1–30 Hz).
- **Improved Coefficients**: Review Ji and Archuleta (2007) for the updated set of relaxation times ($\tau_k$) and weights that provide a more robust frequency-independent behavior.
- **Memory Variable Discretization**: Ensure the update for $anelastic\_strain$ (memory variables) uses a stable scheme for large time steps, as suggested in the 2007 paper.

## 3. Correct Nonlinear-Damping Coupling (Shi and Asimaki 2017)

The way viscous damping is added to the nonlinear stress is critical.

- **Modulus Scaling**: In `time_integration.py`, the damping deficit is currently scaled by `g_sec`.
    - **Verification**: Check if Shi and Asimaki (2017) recommend scaling by $G_{max}$ or $G_{sec}$. Often, if the viscous damping is intended to model small-strain energy loss that _persists_ or _fades_ at large strains, the scaling factor matters.
    - **Over-damping/Under-damping**: If `g_sec` goes to zero, viscous damping vanishes. If the HH backbone with non-Masing rules is already providing high hysteretic damping, this might be intended. However, if the amplification is too high, it suggests a lack of damping at critical frequencies.
- **Frequency-Dependent Damping (FDD)**: Shi and Asimaki (2017) often employ a transition between small-strain viscous damping and large-strain hysteretic damping. Verify if the "Liu & Archuleta" model needs a transition function $w(\gamma)$ similar to the HH backbone transition.

## 4. Numerical Stability and Boundaries

- **CFL Condition**: The `subcycles` must be sufficient even when the material is at its stiffest. Ensure that $dt < dz / V_{s,max}$ is strictly honored.
- **Base Boundary Conditions**: Verify the implementation of the "Elastic" (transmitting) boundary. The current force-based approach with `impedance_base` and `mass_base` must be checked against standard formulations (e.g., Lysmer-Kuhlemeyer).
- **Incident vs. Outcrop**: Ensure the factor of 2 for outcrop motions is applied correctly and consistently between linear and nonlinear dispatchers.

## 5. Recommended Open-Source Verification Tools

- **PySeismoSoil**: Use for direct comparison of HH model results.
- **OpenSees**: Use the `ASDEmbeddedCohesiveElement` or standard `quad` elements with `PressureIndependMultiYield` materials to verify 1D wave propagation.
- **DEEPSOIL**: Use the exported results from DEEPSOIL (which implements MKZ and various damping models) as a reference for regression tests.

## 6. Iterative Fix Plan

1. **Fix `_compute_la_weights`**: Implement the 2007 least-squares approach.
2. **Update Integration Loop**: Align the damping stress addition with the SeismoSoil formulation.
3. **Refactor `TimeDomainCalculator`**: Ensure that parameters from `prepare()` are correctly passed and used in the Numba-accelerated loop.
4. **Validation**: Run `tests/time_domain_test.py` and ensure the `RTOL_TIGHT` limits are met for low intensity.
