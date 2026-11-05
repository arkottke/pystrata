Motion Module
=============

The motion module provides classes for representing and manipulating ground motion data used in site response analysis.

A recorded motion is a :class:`~pystrata.motion.TimeSeriesMotion`, which is read
from a file with :meth:`~pystrata.motion.TimeSeriesMotion.load_at2_file` or
:meth:`~pystrata.motion.TimeSeriesMotion.load_smc_file`. Random vibration theory
motions are an :class:`~pystrata.motion.RvtMotion`, from Fourier amplitudes and a
duration, or a :class:`~pystrata.motion.CompatibleRvtMotion`, from a target
acceleration-response spectrum. See the :doc:`../user-guide/motions` for examples.

.. currentmodule:: pystrata.motion

Classes
-------

.. autosummary::
   :nosignatures:

   TimeSeriesMotion
   RvtMotion
   CompatibleRvtMotion

Functions
---------

.. autofunction:: fit_attenuation

Time Series Motions
-------------------

.. autoclass:: TimeSeriesMotion
   :members:
   :show-inheritance:

Random Vibration Theory Motions
-------------------------------

.. autoclass:: RvtMotion
   :members:
   :show-inheritance:

.. autoclass:: CompatibleRvtMotion
   :members:
   :show-inheritance:

.. note::

   ``SourceTheoryRvtMotion`` was removed in pystrata v0.6+. Use
   ``pygmm.fourier_spectrum.SourceTheoryModel`` and pass the result to
   :meth:`~pystrata.motion.RvtMotion.from_fas` instead.
