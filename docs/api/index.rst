API Reference
=============

This section provides detailed documentation for all public classes and functions in PyStrata.

.. toctree::
   :maxdepth: 2

   motion
   site
   propagation
   output
   variation

Overview
--------

PyStrata's API is organized into five main modules:

**Motion** (:doc:`motion`)
    Classes for representing and manipulating ground motion data

**Site** (:doc:`site`)
    Classes for defining layered soil profiles and material properties

**Propagation** (:doc:`propagation`)
    Site response analysis calculators and methods

**Output** (:doc:`output`)
    Output quantity computation and data handling

**Variation** (:doc:`variation`)
    Tools for parameter variation and uncertainty analysis

Design Philosophy
-----------------

The API follows several key design principles:

**Immutability**
    Most objects are immutable after creation to prevent accidental modification

**Composition**
    Complex functionality is built by composing simpler components

**Type Safety**
    Extensive use of type hints for better development experience

**Consistent Interfaces**
    Similar patterns across different modules for ease of use

**Extensibility**
    Clear extension points for custom functionality

Common Patterns
---------------

**Factory Methods**
    Many classes provide class methods for convenient creation:

    .. code-block:: python

        motion = pystrata.motion.TimeSeriesMotion.load_at2_file("motion.at2")

**Output Collections**
    Multiple outputs can be managed together:

    .. code-block:: python

        outputs = pystrata.output.OutputCollection([
            pystrata.output.ResponseSpectrumOutput(),
            pystrata.output.AccelTransferFunctionOutput(),
        ])

**Calculator Interface**
    All propagation calculators implement the same calling interface:

    .. code-block:: python

        calc = pystrata.propagation.EquivalentLinearCalculator()
        calc(motion, profile, output_location)

**Property Access**
    Read-only properties provide computed values:

    .. code-block:: python

        profile.vs30  # Time-averaged shear wave velocity
        motion.duration  # Motion duration
        output.values  # Computed output values
