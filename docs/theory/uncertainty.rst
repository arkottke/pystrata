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

Statistical quantities of interest:
- **Mean**: Central tendency
- **Standard deviation**: Variability measure
- **Percentiles**: Confidence intervals
- **Sensitivity indices**: Parameter importance
