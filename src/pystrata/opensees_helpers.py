"""OpenSees 1D site response helpers for verification.

Provides thin wrappers around openseespy to run 1D linear-elastic and
nonlinear site response analyses using a column of quad elements.  The
results can be compared directly against :func:`pystrata.time_integration.propagate_time_domain`
and :func:`pystrata.time_integration.propagate_nonlinear`.

The functions build a standard 1D shear-beam model:

* Node pairs (left/right) at each layer interface, all vertical DOFs fixed,
  ``equalDOF`` on horizontal DOFs → pure 1D SH-wave propagation.
* Lysmer-Kuhlemeyer dashpot at the base for absorbing (elastic) boundary.
* Outcrop motion applied as a horizontal force on the base node:
  ``F(t) = 2 * c * v_incident(t)``.
* Rayleigh damping for viscous energy dissipation.

References
----------
.. [1] McKenna, F., Fenves, G. L., & Scott, M. H. (2000). Open system for
   earthquake engineering simulation. University of California, Berkeley.
.. [2] Lysmer, J., & Kuhlemeyer, R. L. (1969). Finite dynamic model for
   infinite media. Journal of the Engineering Mechanics Division, 95(4), 859-877.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import integrate


def _compute_rayleigh_coeffs(
    xi: float, f1: float = 0.5, f2: float = 20.0
) -> tuple[float, float]:
    """Compute Rayleigh damping coefficients *a0* and *a1*.

    Parameters
    ----------
    xi : float
        Target damping ratio (e.g. 0.02 for 2 %).
    f1, f2 : float
        Frequencies [Hz] at which the target damping is achieved exactly.

    Returns
    -------
    a0, a1 : float
        Mass-proportional and stiffness-proportional coefficients.
    """
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    a0 = 2 * xi * w1 * w2 / (w1 + w2)
    a1 = 2 * xi / (w1 + w2)
    return a0, a1


def run_opensees_linear(
    thicknesses: npt.NDArray[np.floating],
    densities: npt.NDArray[np.floating],
    shear_mods: npt.NDArray[np.floating],
    damping_ratios: npt.NDArray[np.floating],
    input_accel: npt.NDArray[np.floating],
    dt: float,
    rho_base: float,
    vs_base: float,
) -> npt.NDArray[np.floating]:
    """Run a 1D linear-elastic site response analysis in OpenSees.

    Parameters
    ----------
    thicknesses : array, shape (n_layers,)
        Layer thicknesses from surface to base [m].
    densities : array, shape (n_layers,)
        Layer mass densities [kg/m³].
    shear_mods : array, shape (n_layers,)
        Layer shear moduli [Pa].
    damping_ratios : array, shape (n_layers,)
        Layer viscous damping ratios (Rayleigh).
    input_accel : array, shape (n_times,)
        **Outcrop** acceleration time series [m/s²].
    dt : float
        Time step [s].
    rho_base : float
        Base (halfspace) mass density [kg/m³].
    vs_base : float
        Base shear-wave velocity [m/s].

    Returns
    -------
    surface_accel : array, shape (n_times,)
        Surface acceleration [m/s²].
    """
    import openseespy.opensees as ops

    ops.wipe()

    thicknesses = np.asarray(thicknesses, dtype=float)
    densities = np.asarray(densities, dtype=float)
    shear_mods = np.asarray(shear_mods, dtype=float)
    damping_ratios = np.asarray(damping_ratios, dtype=float)
    input_accel = np.asarray(input_accel, dtype=float)

    n_layers = len(thicknesses)
    n_nodes = n_layers + 1  # interfaces
    n_times = len(input_accel)
    width = 1.0  # unit column width [m]
    nu = 0.3  # Poisson's ratio (arbitrary for shear-dominated problem)

    # ------------------------------------------------------------------
    # 1. Model builder: 2D, 2 DOF per node (horizontal + vertical)
    # ------------------------------------------------------------------
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    # ------------------------------------------------------------------
    # 2. Nodes — pairs (left=2*i, right=2*i+1) at each layer interface
    #    y=0 at the surface, y<0 going downward.
    # ------------------------------------------------------------------
    depths = np.zeros(n_nodes)
    for i in range(n_layers):
        depths[i + 1] = depths[i] + thicknesses[i]

    for i in range(n_nodes):
        ops.node(2 * i, 0.0, -depths[i])
        ops.node(2 * i + 1, width, -depths[i])

    # Extra fixed reference node for dashpot
    dashpot_ref_node = 2 * n_nodes
    ops.node(dashpot_ref_node, 0.0, -depths[-1])
    ops.fix(dashpot_ref_node, 1, 1)

    # ------------------------------------------------------------------
    # 3. Boundary conditions
    #    - Fix all vertical DOFs (pure shear)
    #    - equalDOF on horizontal DOFs for left/right pairs
    #    - Base-left node: free horizontal (force applied), fixed vertical
    # ------------------------------------------------------------------
    for i in range(n_nodes):
        ops.fix(2 * i, 0, 1)  # free horizontal, fixed vertical
        ops.fix(2 * i + 1, 0, 1)

    for i in range(n_nodes):
        ops.equalDOF(2 * i, 2 * i + 1, 1)  # tie horizontal DOFs

    # ------------------------------------------------------------------
    # 4. Materials — nDMaterial ElasticIsotropic
    # ------------------------------------------------------------------
    for i in range(n_layers):
        G = shear_mods[i]
        E = 2 * G * (1 + nu)
        rho = densities[i]
        mat_tag = i + 1
        ops.nDMaterial("ElasticIsotropic", mat_tag, E, nu, rho)

    # ------------------------------------------------------------------
    # 5. Elements — quad (PlaneStrain)
    #    Element i connects nodes at interface i (top) and i+1 (bottom)
    # ------------------------------------------------------------------
    for i in range(n_layers):
        n1 = 2 * (i + 1)  # bottom-left
        n2 = 2 * (i + 1) + 1  # bottom-right
        n3 = 2 * i + 1  # top-right
        n4 = 2 * i  # top-left
        mat_tag = i + 1
        ele_tag = i + 1
        ops.element(
            "quad",
            ele_tag,
            n1,
            n2,
            n3,
            n4,
            1.0,  # thickness (out-of-plane)
            "PlaneStrain",
            mat_tag,
        )

    # ------------------------------------------------------------------
    # 6. Lysmer-Kuhlemeyer dashpot at the base
    # ------------------------------------------------------------------
    c_dashpot = rho_base * vs_base * width
    dashpot_mat_tag = n_layers + 1
    ops.uniaxialMaterial("Viscous", dashpot_mat_tag, c_dashpot, 1.0)

    base_left_node = 2 * (n_nodes - 1)
    dashpot_ele_tag = n_layers + 1
    ops.element(
        "zeroLength",
        dashpot_ele_tag,
        dashpot_ref_node,
        base_left_node,
        "-mat",
        dashpot_mat_tag,
        "-dir",
        1,
    )

    # ------------------------------------------------------------------
    # 7. Rayleigh damping (average damping ratio across layers)
    # ------------------------------------------------------------------
    xi_avg = float(np.mean(damping_ratios))
    a0, a1 = _compute_rayleigh_coeffs(xi_avg)
    ops.rayleigh(a0, a1, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 8. Compute incident velocity and applied force
    #    Outcrop → incident: divide by 2
    #    Force on base node: F(t) = 2 * c * v_incident = c * v_outcrop
    # ------------------------------------------------------------------
    v_outcrop = integrate.cumulative_trapezoid(input_accel, dx=dt, initial=0.0)
    force_hist = c_dashpot * v_outcrop  # = 2 * c * (v_outcrop / 2)

    # ------------------------------------------------------------------
    # 9. Time series and load pattern
    # ------------------------------------------------------------------
    force_values = list(force_hist)
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *force_values, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    ops.load(base_left_node, 1.0, 0.0)

    # ------------------------------------------------------------------
    # 10. Analysis setup — Newmark implicit (unconditionally stable)
    # ------------------------------------------------------------------
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-12, 30)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    # ------------------------------------------------------------------
    # 11. Run and record surface acceleration
    # ------------------------------------------------------------------
    surface_left_node = 0  # top-left node at surface
    accel_out = np.zeros(n_times)
    for step in range(n_times):
        ops.analyze(1, dt)
        accel_out[step] = ops.nodeAccel(surface_left_node, 1)

    ops.wipe()
    return accel_out


def run_opensees_nonlinear(
    thicknesses: npt.NDArray[np.floating],
    densities: npt.NDArray[np.floating],
    shear_mods: npt.NDArray[np.floating],
    damping_ratios: npt.NDArray[np.floating],
    shear_strengths: npt.NDArray[np.floating],
    ref_strains: npt.NDArray[np.floating],
    input_accel: npt.NDArray[np.floating],
    dt: float,
    rho_base: float,
    vs_base: float,
    *,
    n_surf: int = 20,
) -> npt.NDArray[np.floating]:
    """Run a 1D nonlinear site response analysis in OpenSees.

    Uses ``PressureIndependMultiYield`` with a hyperbolic backbone:

    .. math::
        \\tau = G_{max} \\gamma / (1 + \\gamma / \\gamma_{ref})

    This is equivalent to the pystrata MKZ model with ``beta=1, s=1``.

    Parameters
    ----------
    thicknesses : array, shape (n_layers,)
        Layer thicknesses from surface to base [m].
    densities : array, shape (n_layers,)
        Layer mass densities [kg/m³].
    shear_mods : array, shape (n_layers,)
        Layer shear moduli :math:`G_{max}` [Pa].
    damping_ratios : array, shape (n_layers,)
        Layer small-strain viscous damping ratios (Rayleigh).
    shear_strengths : array, shape (n_layers,)
        Layer peak shear strength :math:`\\tau_{max}` [Pa].
    ref_strains : array, shape (n_layers,)
        Layer reference strains :math:`\\gamma_{ref}` [-].
    input_accel : array, shape (n_times,)
        **Outcrop** acceleration time series [m/s²].
    dt : float
        Time step [s].
    rho_base : float
        Base (halfspace) mass density [kg/m³].
    vs_base : float
        Base shear-wave velocity [m/s].
    n_surf : int
        Number of yield surfaces for ``PressureIndependMultiYield``.

    Returns
    -------
    surface_accel : array, shape (n_times,)
        Surface acceleration [m/s²].
    """
    import openseespy.opensees as ops

    ops.wipe()

    thicknesses = np.asarray(thicknesses, dtype=float)
    densities = np.asarray(densities, dtype=float)
    shear_mods = np.asarray(shear_mods, dtype=float)
    damping_ratios = np.asarray(damping_ratios, dtype=float)
    shear_strengths = np.asarray(shear_strengths, dtype=float)
    ref_strains = np.asarray(ref_strains, dtype=float)
    input_accel = np.asarray(input_accel, dtype=float)

    n_layers = len(thicknesses)
    n_nodes = n_layers + 1
    n_times = len(input_accel)
    width = 1.0
    nu = 0.3

    # ------------------------------------------------------------------
    # 1. Model builder
    # ------------------------------------------------------------------
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    # ------------------------------------------------------------------
    # 2. Nodes
    # ------------------------------------------------------------------
    depths = np.zeros(n_nodes)
    for i in range(n_layers):
        depths[i + 1] = depths[i] + thicknesses[i]

    for i in range(n_nodes):
        ops.node(2 * i, 0.0, -depths[i])
        ops.node(2 * i + 1, width, -depths[i])

    dashpot_ref_node = 2 * n_nodes
    ops.node(dashpot_ref_node, 0.0, -depths[-1])
    ops.fix(dashpot_ref_node, 1, 1)

    # ------------------------------------------------------------------
    # 3. Boundary conditions
    # ------------------------------------------------------------------
    for i in range(n_nodes):
        ops.fix(2 * i, 0, 1)
        ops.fix(2 * i + 1, 0, 1)

    for i in range(n_nodes):
        ops.equalDOF(2 * i, 2 * i + 1, 1)

    # ------------------------------------------------------------------
    # 4. Materials — PressureIndependMultiYield
    # ------------------------------------------------------------------
    for i in range(n_layers):
        G = float(shear_mods[i])
        rho = float(densities[i])
        cohesion = float(shear_strengths[i])
        peak_strain = float(ref_strains[i])
        K = 2 * G * (1 + nu) / (3 * (1 - 2 * nu))  # bulk modulus
        mat_tag = i + 1
        # PressureIndependMultiYield args:
        # tag, nd, rho, refShearModul, refBulkModul, cohesi, peakShearStra
        # [, frictionAng=0, refPress=100, pressDependCoe=0, noYieldSurf=20]
        ops.nDMaterial(
            "PressureIndependMultiYield",
            mat_tag,
            2,  # ndm
            rho,
            G,
            K,
            cohesion,
            peak_strain,
            0.0,  # frictionAng
            1.0e5,  # refPress (Pa) — high value to make pressure-independent
            0.0,  # pressDependCoe
            n_surf,
        )

    # ------------------------------------------------------------------
    # 5. Elements — quad (PlaneStrain)
    # ------------------------------------------------------------------
    for i in range(n_layers):
        n1 = 2 * (i + 1)
        n2 = 2 * (i + 1) + 1
        n3 = 2 * i + 1
        n4 = 2 * i
        mat_tag = i + 1
        ele_tag = i + 1
        ops.element(
            "quad",
            ele_tag,
            n1,
            n2,
            n3,
            n4,
            1.0,
            "PlaneStrain",
            mat_tag,
        )

    # ------------------------------------------------------------------
    # 6. Dashpot at base
    # ------------------------------------------------------------------
    c_dashpot = rho_base * vs_base * width
    dashpot_mat_tag = n_layers + 1
    ops.uniaxialMaterial("Viscous", dashpot_mat_tag, c_dashpot, 1.0)

    base_left_node = 2 * (n_nodes - 1)
    dashpot_ele_tag = n_layers + 1
    ops.element(
        "zeroLength",
        dashpot_ele_tag,
        dashpot_ref_node,
        base_left_node,
        "-mat",
        dashpot_mat_tag,
        "-dir",
        1,
    )

    # ------------------------------------------------------------------
    # 7. Rayleigh damping
    # ------------------------------------------------------------------
    xi_avg = float(np.mean(damping_ratios))
    a0, a1 = _compute_rayleigh_coeffs(xi_avg)
    ops.rayleigh(a0, a1, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 8. Gravity analysis (initialize stresses)
    # ------------------------------------------------------------------
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-6, 30)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    # Allow material to initialize with a zero-load step
    ops.analyze(1, dt)

    # Update material to respond to soil elements
    for i in range(n_layers):
        ops.updateMaterialStage("-material", i + 1, "-stage", 1)

    # ------------------------------------------------------------------
    # 9. Dynamic loading
    # ------------------------------------------------------------------
    v_outcrop = integrate.cumulative_trapezoid(input_accel, dx=dt, initial=0.0)
    force_hist = c_dashpot * v_outcrop

    force_values = list(force_hist)
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *force_values, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    ops.load(base_left_node, 1.0, 0.0)

    # Reset analysis for dynamic phase
    ops.test("NormDispIncr", 1e-6, 30)
    ops.algorithm("Newton")

    # ------------------------------------------------------------------
    # 10. Run and record surface acceleration
    # ------------------------------------------------------------------
    surface_left_node = 0
    accel_out = np.zeros(n_times)
    for step in range(n_times):
        ok = ops.analyze(1, dt)
        if ok != 0:
            # Try modified Newton if standard Newton fails
            ops.algorithm("ModifiedNewton")
            ok = ops.analyze(1, dt)
            ops.algorithm("Newton")
        accel_out[step] = ops.nodeAccel(surface_left_node, 1)

    ops.wipe()
    return accel_out
