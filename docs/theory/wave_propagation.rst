Wave Propagation
================

This section covers the mathematical treatment of seismic wave propagation through layered media, which forms the computational foundation of site response analysis.

Governing Equations
-------------------

**One-Dimensional Wave Equation**

For vertically propagating shear waves in a layered medium, the equation of motion is:

.. math::

   \rho \frac{\partial^2 u}{\partial t^2} = G \frac{\partial^2 u}{\partial z^2}

where:
- :math:`u(z,t)` is the horizontal displacement
- :math:`\rho` is the mass density
- :math:`G` is the shear modulus
- :math:`z` is the vertical coordinate (positive upward)

**Complex Shear Modulus**

To incorporate material damping, the shear modulus is treated as complex:

.. math::

   G^* = G(1 + 2i\xi)

where :math:`\xi` is the damping ratio.

Solution Methods
----------------

**Frequency Domain Approach**

Taking the Fourier transform of the wave equation:

.. math::

   -\omega^2 \rho \hat{u} = G^* \frac{\partial^2 \hat{u}}{\partial z^2}

This leads to the general solution:

.. math::

   \hat{u}(z,\omega) = A e^{ik^*z} + B e^{-ik^*z}

where :math:`k^* = \omega/V_s^*` is the complex wavenumber and :math:`V_s^* = \sqrt{G^*/\rho}`.

**Transfer Matrix Method**

For layered systems, the Thomson-Haskell propagator matrix method relates the wave field at different depths. For a single layer of thickness :math:`h`, the transfer matrix is:

.. math::

   \mathbf{T} = \begin{pmatrix}
   \cos(k^* h) & \frac{i \sin(k^* h)}{Z^*} \\
   i Z^* \sin(k^* h) & \cos(k^* h)
   \end{pmatrix}

where :math:`Z^* = \rho V_s^*` is the complex impedance.

Boundary Conditions
-------------------

**Free Surface**

At the ground surface (:math:`z = 0`), the shear stress must vanish:

.. math::

   \tau = G^* \frac{\partial u}{\partial z} = 0

**Input Motion Specification**

Three common approaches for specifying input motion:

1. **Outcrop Motion**: Motion that would occur at a rock outcrop
2. **Within Motion**: Motion recorded within the rock formation
3. **Incident Motion**: Upward-traveling wave component only

**Radiation Damping**

For semi-infinite elastic bedrock, the radiation condition requires:

.. math::

   \tau = \rho V_s u̇

This represents energy radiation away from the site.

Numerical Implementation
------------------------

**Frequency Sampling**

Adequate frequency resolution is critical for accurate results:

.. math::

   \Delta f \leq \frac{V_{s,min}}{8 H_{max}}

where :math:`V_{s,min}` is the minimum shear wave velocity and :math:`H_{max}` is the maximum depth of interest.

**Stability Considerations**

The method is stable for all frequencies, but numerical precision may be lost for:
- Very thick, soft layers
- High frequencies
- Strong impedance contrasts

**Efficiency Optimizations**

- Pre-compute layer matrices for repeated analyses
- Use symmetry to reduce computation for real-valued inputs
- Implement fast convolution for time domain conversion

Validation Examples
-------------------

**Analytical Solutions**

1. **Uniform Half-Space**

   For a uniform elastic half-space, the exact amplification function is:

   .. math::

      H(\omega) = \frac{2Z_0}{Z_0 + Z_r}

   where :math:`Z_0` is the surface impedance and :math:`Z_r` is the reference impedance.

2. **Single Layer over Rigid Base**

   The transfer function has resonant peaks at:

   .. math::

      f_n = \frac{(2n-1)V_s}{4H}

   for :math:`n = 1, 2, 3, ...`

**Benchmark Comparisons**

PyStrata wave propagation algorithms have been validated against:
- Analytical solutions for simple geometries
- SHAKE91 for equivalent linear analysis
- DEEPSOIL for advanced nonlinear methods
- Recorded earthquake data from instrumented sites

Implementation Details
----------------------

The wave propagation calculations in PyStrata are implemented in the `propagation` module with careful attention to:

- **Numerical stability** through appropriate algorithmic choices
- **Computational efficiency** via optimized matrix operations
- **Physical accuracy** through proper boundary condition treatment
- **Flexibility** to accommodate various input motion types and analysis methods
