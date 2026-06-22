Logic Trees and Uncertainty
===========================

Logic trees provide a systematic framework for propagating epistemic uncertainty through a site
response analysis.  Each *node* represents a source of uncertainty; each *alternative* within a
node carries a weight.  Running all combinations (or a weighted sample) produces an ensemble of
results from which mean and fractile estimates can be derived.

.. currentmodule:: pystrata.logic_tree

Concepts
--------

The key classes are:

- :class:`Node` — a source of uncertainty with a name and a list of :class:`Alternative` objects.
- :class:`Alternative` — one possible value for a node, with an associated weight.
- :class:`Branch` — a single path through the tree (one alternative chosen per node).
- :class:`LogicTree` — the full tree; iterating over it yields :class:`Branch` objects.

A Simple Logic Tree
-------------------

.. code-block:: python

    import pystrata

    # Two nodes: calculator type and input motion
    lt = pystrata.logic_tree.LogicTree.from_list([
        {
            "name": "calculator",
            "alternatives": [
                {"value": "eql",     "weight": 0.6},
                {"value": "fd_eql",  "weight": 0.4},
            ],
        },
        {
            "name": "motion",
            "alternatives": [
                {"value": "motion_A", "weight": 0.5},
                {"value": "motion_B", "weight": 0.5},
            ],
        },
    ])

    for branch in lt:
        # branch["calculator"].value → "eql" or "fd_eql"
        # branch["motion"].value → "motion_A" or "motion_B"
        # branch.weight → product of individual weights
        print(branch.weight, branch["calculator"].value, branch["motion"].value)

Loading from JSON
-----------------

Logic trees can be stored in JSON and loaded with :meth:`LogicTree.from_json`:

.. code-block:: json

    {
      "nodes": [
        {
          "name": "calculator",
          "alternatives": [
            {"value": "eql",    "weight": 0.6},
            {"value": "fd_eql", "weight": 0.4}
          ]
        }
      ]
    }

.. code-block:: python

    lt = pystrata.logic_tree.LogicTree.from_json("my_logic_tree.json")

Conditional Alternatives
------------------------

An alternative can require or exclude specific values of other nodes using ``requires`` and
``excludes`` dictionaries.  This allows modeling correlated or mutually exclusive choices:

.. code-block:: python

    # Alternative valid only when "motion" node has value "motion_A"
    {
        "value": "eql",
        "weight": 0.8,
        "requires": {"motion": "motion_A"},
    }

Running a Full Logic Tree Analysis
-----------------------------------

A typical workflow iterates over all branches, runs the analysis for each, and collects outputs:

.. code-block:: python

    import pystrata

    outputs = pystrata.output.OutputCollection([
        pystrata.output.ResponseSpectrumOutput(),
    ])

    for branch in lt:
        calc_name = branch["calculator"].value
        if calc_name == "eql":
            calc = pystrata.propagation.EquivalentLinearCalculator()
        else:
            calc = pystrata.propagation.FrequencyDependentEqlCalculator()

        calc(motion, profile, profile.location("outcrop", index=-1))
        outputs(calc, name=str(branch.weight))

    # Summarise across realizations
    ds = outputs[0].to_xarray()

Separation of Variance
-----------------------

:func:`separation_of_variance` decomposes total variance across logic tree nodes into
between-node (epistemic) and within-node (aleatory) contributions:

.. code-block:: python

    from pystrata.logic_tree import separation_of_variance

    var_between, var_within = separation_of_variance(lt, results)

See :func:`compute_marginals` for computing weighted marginal distributions per node.
