Motion Module
=============

The motion module provides classes for representing and manipulating ground motion data used in site response analysis.

.. currentmodule:: pystrata.motion

Classes
-------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   TimeSeriesMotion
   RvtMotion
   CompatibleRvtMotion

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
