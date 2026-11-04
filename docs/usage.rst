.. _usage:

Usage
=====

pyStrata provides classes to programmatically perform site response analysis.
It does not provide a default workflow -- instead it is meant to be used to
define custom workflows. Regardless of the workflow, the following parts are
needed:

#.  Define the input motion, with :class:`~pystrata.motion.TimeSeriesMotion`,
    :class:`~pystrata.motion.RvtMotion` or
    :class:`~pystrata.motion.CompatibleRvtMotion`
    (:doc:`user guide <user-guide/motions>`, :doc:`API <api/motion>`).
#.  Construct the site profile from :class:`~pystrata.site.SoilType`,
    :class:`~pystrata.site.Layer` and :class:`~pystrata.site.Profile`
    (:doc:`user guide <user-guide/profiles>`, :doc:`API <api/site>`).
#.  Define the site response calculator, such as
    :class:`~pystrata.propagation.LinearElasticCalculator` or
    :class:`~pystrata.propagation.EquivalentLinearCalculator`
    (:doc:`user guide <user-guide/calculations>`, :doc:`API <api/propagation>`).
#.  Select what outputs should be computed
    (:doc:`user guide <user-guide/outputs>`, :doc:`API <api/output>`).

Once these components are assembled the calculations can be performed and the
outputs plotted and saved. The :doc:`quickstart` walks through a complete
analysis, and the :doc:`examples <examples/index>` cover the variations.

Random vibration theory motions are created through two classes:
:class:`~pystrata.motion.RvtMotion` for directly specifying Fourier amplitudes
and durations, and :class:`~pystrata.motion.CompatibleRvtMotion` for specifying
the motion by an acceleration-response spectrum. To create a source-theory
motion, use ``pygmm.fourier_spectrum.SourceTheoryModel`` and pass it to
:meth:`~pystrata.motion.RvtMotion.from_fas`.
