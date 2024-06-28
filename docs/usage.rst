.. _usage:

Usage
=====

`pyStrata` provides classes to programmatically perform site response analysis.
It does not provide a default workflow -- instead it is meant to be used to
define custom workflows. However, regardless of the process the follows parts need to be used:

#.  Define the :ref:`input motion <input-motion>`
#.  Construct the :ref:`site profile <site-profile>`
#.  Define the site response :ref:`calculator <calculator>`
#.  Select what :ref:`output <outputs>` should be computed

Once these components are assembled the calculations can be performed and the outputs plotted and saved.

.. _input-motion:

Input motion
------------

.. module:: pystrata.motion

`pyStrata` permits use of both time series and random vibration theory site
response. Time series motions are created using a ``TimeSeriesMotion`` and one
of three methods. Whereas, Random vibration theory motions are created through
three classes: ``RvtMotion`` for directly specifiying Fourier amplitudes and
durations, ``CompatibleRvtMotion`` for specifying the motion by the
acceleration-response, and ``SourceTheoryRvtMotion`` for computing the motion
by point-source parameters.

.. autoclass:: TimeSeriesMotion
   :special-members: __init__
   :members: load_at2_file, load_smc_file

.. autoclass:: RvtMotion
   :special-members: __init__

.. autoclass:: CompatibleRvtMotion
   :special-members: __init__

.. autoclass:: SourceTheoryRvtMotion
   :special-members: __init__


.. _site-profile:

Construct the site profile
--------------------------

.. _calculator:

Site response calcuator
-----------------------

.. _outputs:

Outputs
-------
