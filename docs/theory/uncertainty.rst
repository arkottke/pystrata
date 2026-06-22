Uncertainty Analysis
====================

Site response analysis involves numerous uncertain parameters. PyStrata provides tools for systematic uncertainty quantification using logic trees and Monte Carlo simulation.

Sources of Uncertainty
-----------------------

**Aleatory Uncertainty** (Natural Variability)
- Spatial variability in soil properties
- Earthquake source characteristics
- Ground motion variability

**Epistemic Uncertainty** (Knowledge Limitations)
- Model selection (equivalent linear vs. nonlinear)
- Parameter estimation uncertainty
- Methodological assumptions

Logic Tree Framework
---------------------

Logic trees provide a structured approach to capture epistemic uncertainties by:

1. **Defining Alternatives**: Different models or parameter values
2. **Assigning Weights**: Relative confidence in each alternative
3. **Computing Branches**: All possible combinations
4. **Aggregating Results**: Weighted ensemble statistics

**Example Logic Tree Structure**

::

   Site Response Method
   ├── Equivalent Linear (0.7)
   │   ├── Darendeli Curves (0.8)
   │   └── Zhang Curves (0.2)
   └── Frequency Domain (0.3)
       ├── Darendeli Curves (0.8)
       └── Zhang Curves (0.2)

Monte Carlo Simulation
----------------------

For aleatory uncertainties, Monte Carlo simulation generates random realizations:

.. code-block:: python

   # Example: Uncertain shear wave velocity
   vs_mean = 400  # m/s
   vs_std = 50    # m/s

   for i in range(1000):
       vs_sample = np.random.normal(vs_mean, vs_std)
       # Run site response analysis
       # Store results

Result Processing
-----------------

After running an ensemble of analyses, results are aggregated across realizations.
pyStrata stores multiple realizations in a 2-D array (realizations × frequency/period/depth),
so standard NumPy operations apply directly:

.. code-block:: python

    # Response spectrum output collected over N realizations
    import numpy as np

    values = rs_output.values          # shape (n_periods, n_realizations)
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)
    p16 = np.percentile(values, 16, axis=1)
    p84 = np.percentile(values, 84, axis=1)

For log-normal quantities (spectral acceleration, shear modulus) compute statistics in log space:

.. code-block:: python

    ln_vals = np.log(values)
    geomean = np.exp(np.mean(ln_vals, axis=1))
    sigma_ln = np.std(ln_vals, axis=1)

Exporting to xarray
~~~~~~~~~~~~~~~~~~~

:meth:`~pystrata.output.Output.to_xarray` collects realizations into a labeled dataset:

.. code-block:: python

    ds = rs_output.to_xarray()
    # ds["values"].mean("name")         → mean over realizations
    # ds["values"].std("name")          → standard deviation

Separation of Variance
----------------------

When using a logic tree, total variance in the results can be decomposed into
*between-branch* (epistemic) and *within-branch* (aleatory) components using
:func:`~pystrata.logic_tree.separation_of_variance`:

.. math::

   \sigma^2_\text{total} = \sigma^2_\text{epistemic} + \sigma^2_\text{aleatory}

Each term is computed as a weighted average over branches:

.. math::

   \sigma^2_\text{epistemic} = \sum_i w_i (\mu_i - \bar{\mu})^2

   \sigma^2_\text{aleatory} = \sum_i w_i \sigma^2_i

where :math:`w_i` is the branch weight, :math:`\mu_i` and :math:`\sigma^2_i` are the
mean and variance for branch *i*, and :math:`\bar{\mu}` is the weighted grand mean.

.. code-block:: python

    from pystrata.logic_tree import separation_of_variance

    # results: dict mapping branch → output array
    var_epistemic, var_aleatory = separation_of_variance(logic_tree, results)

Marginal Distributions
~~~~~~~~~~~~~~~~~~~~~~

:func:`~pystrata.logic_tree.compute_marginals` collapses the tree along each node to
show the contribution of individual nodes to the total spread:

.. code-block:: python

    from pystrata.logic_tree import compute_marginals
    marginals = compute_marginals(logic_tree, results)
    # marginals["calculator"] → weighted mean/std conditioned on each calculator alternative

Practical Recommendations
--------------------------

- Use **logic trees** for discrete epistemic choices (calculator type, curve model).
- Use **Monte Carlo** for continuous parameters (Vs variability, nonlinear curves) via
  :mod:`pystrata.variation`.
- Run at least 100 Monte Carlo realizations for stable percentile estimates.
- When combining both, nest Monte Carlo realizations inside each logic tree branch so
  aleatory and epistemic contributions remain separable.
