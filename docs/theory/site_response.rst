Site Response Fundamentals
==========================

Site response analysis predicts how seismic waves are modified as they propagate from bedrock through overlying soil layers to the ground surface. This modification occurs due to the impedance contrasts between layers and the dynamic properties of the soil materials.

Physical Principles
-------------------

**Wave Impedance**

The seismic impedance of a material is defined as:

.. math::

   Z = \rho V_s

where :math:`\rho` is the mass density and :math:`V_s` is the shear wave velocity.

When a wave encounters an interface between materials with different impedances, part of the energy is reflected and part is transmitted. The reflection and transmission coefficients depend on the impedance contrast:

.. math::

   R = \frac{Z_2 - Z_1}{Z_2 + Z_1}

.. math::

   T = \frac{2Z_2}{Z_2 + Z_1}

**Resonance and Amplification**

A soil deposit acts as a resonator, with natural frequencies determined by the layer thicknesses and wave velocities. For a single layer over bedrock, the fundamental resonant frequency is:

.. math::

   f_0 = \frac{V_s}{4H}

where :math:`H` is the layer thickness. Ground motions with frequencies near :math:`f_0` experience maximum amplification.

**Quality Factor and Damping**

The sharpness of the resonance peak is controlled by the quality factor:

.. math::

   Q = \frac{1}{2\xi}

where :math:`\xi` is the damping ratio. Higher damping (lower Q) results in broader, lower amplitude resonance peaks.

Transfer Function Approach
--------------------------

The site response transfer function relates the input motion at bedrock to the output motion at any depth or at the surface.

**Linear Elastic Transfer Function**

For a linear elastic medium, the transfer function between two points in a layered system can be computed using the Thomson-Haskell propagator matrix method :cite:p:`Thomson1950,Haskell1953`.

The displacement amplification function is:

.. math::

   H(\omega) = \frac{U_{surface}(\omega)}{U_{input}(\omega)}

where :math:`U(\omega)` represents the displacement in the frequency domain.

**Propagator Matrix Method**

Each layer is represented by a 2×2 propagator matrix that relates the wave field at the top and bottom of the layer:

.. math::

   \begin{pmatrix} u \\ \tau \end{pmatrix}_{top} =
   \mathbf{P} \begin{pmatrix} u \\ \tau \end{pmatrix}_{bottom}

where :math:`u` is displacement and :math:`\tau` is shear stress.

The propagator matrix for a single layer is:

.. math::

   \mathbf{P} = \begin{pmatrix}
   \cos(k^* h) & \frac{i \sin(k^* h)}{Z^*} \\
   i Z^* \sin(k^* h) & \cos(k^* h)
   \end{pmatrix}

where:
- :math:`k^* = \omega/V_s^*` is the complex wavenumber
- :math:`Z^* = \rho V_s^*` is the complex impedance
- :math:`V_s^* = V_s(1 + 2i\xi)` accounts for material damping
- :math:`h` is the layer thickness

**System Transfer Function**

For a multilayer system, the overall transfer function is obtained by multiplying the individual layer matrices:

.. math::

   \mathbf{P}_{system} = \mathbf{P}_1 \mathbf{P}_2 \cdots \mathbf{P}_n

Boundary Conditions
-------------------

**Surface Boundary Condition**

At the free surface, the shear stress must be zero:

.. math::

   \tau_{surface} = 0

**Bedrock Boundary Condition**

Two common assumptions for the bedrock boundary:

1. **Rigid bedrock** - zero displacement: :math:`u_{bedrock} = 0`
2. **Elastic bedrock** - specified input motion with radiation damping

**Within-Outcrop Motion**

For the within-outcrop condition, the input motion is specified as the motion that would occur at the bedrock outcrop in the absence of the overlying soil layers.

Time Domain vs. Frequency Domain
--------------------------------

**Frequency Domain Advantages**
- Computationally efficient for linear analysis
- Natural framework for transfer functions
- Easy incorporation of frequency-dependent damping
- Straightforward convolution operations

**Time Domain Advantages**
- Handles nonlinear soil behavior naturally
- Preserves phase relationships in motion
- Direct comparison with recorded time series
- Better for studying transient phenomena

Validation and Verification
---------------------------

**Analytical Solutions**

For simple cases, analytical solutions exist that can be used to verify numerical implementations:

- Single layer over rigid bedrock
- Two-layer systems with specific property ratios
- Infinite uniform medium (traveling wave solution)

**Benchmark Problems**

Standard benchmark problems include:
- LEAP-UCD centrifuge experiments :cite:p:`Kutter2017`
- PRENOLIN blind prediction exercise :cite:p:`Matasovic1995`
- Lotung downhole array data :cite:p:`Zeghal1995`

**Code-to-Code Comparisons**

PyStrata results have been compared against:
- SHAKE :cite:p:`Schnabel1972`
- DEEPSOIL :cite:p:`Hashash2016`
- OpenSees :cite:p:`McKenna2000`
- STRATA :cite:p:`Kottke2013`
