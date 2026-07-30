import numpy as np
import pytest

from pystrata.logic_tree import (
    Alternative,
    LogicTree,
    Node,
    compute_marginals,
    plot_tornado,
    separation_of_variance,
)
from pystrata.output import Output

from . import FPATH_DATA


@pytest.fixture
def my_tree():
    tree = LogicTree(
        [
            Node("foo", "ab"),
            Node(
                "bar",
                [
                    Alternative("c"),
                    Alternative("d"),
                    Alternative("e", requires={"foo": "a"}),
                ],
            ),
            Node(
                "baz",
                [
                    Alternative("f", requires={"bar": ["c", "d"]}),
                    Alternative("g"),
                    Alternative("h", excludes={"foo": "a"}),
                ],
            ),
        ]
    )

    return tree


def test_parse_json():
    LogicTree.from_json(FPATH_DATA / "test_logic_tree.json")


def test_node_init():
    node = Node("foo", ["a", "b"])
    assert isinstance(node[0], Alternative)


def test_branch_count(my_tree):
    branches = list(my_tree)
    count = (2 * 3 * 3) - 3 - 1 - 3
    assert len(branches) == count


def test_len_conditional_tree(my_tree):
    """A conditional tree has to be enumerated, not computed from the product."""
    assert not my_tree.is_rectangular
    assert len(my_tree) == len(list(my_tree))
    assert len(my_tree) < np.prod([len(node) for node in my_tree.nodes])


def test_len_rectangular_tree(rectangular_tree):
    assert rectangular_tree.is_rectangular
    assert len(rectangular_tree) == len(list(rectangular_tree))
    assert len(rectangular_tree) == np.prod(
        [len(node) for node in rectangular_tree.nodes]
    )


def test_valid_branches(my_tree):
    branches = list(my_tree)

    def is_branch(values):
        for b in branches:
            if all(b[k].value == v for k, v in values.items()):
                return True
        else:
            return False

    assert is_branch({"foo": "a", "bar": "c", "baz": "f"})
    assert not is_branch({"foo": "a", "bar": "e", "baz": "f"})
    assert not is_branch({"foo": "a", "bar": "d", "baz": "h"})


def test_multiple_alternatives_same_value_different_requires():
    """Test that multiple alternatives with the same value but different 'requires'
    conditions are both included in the logic tree branches.

    This test addresses a bug where only the first alternative with a given value was
    being used in branch validation, causing branches with later alternatives to be
    incorrectly marked as invalid.
    """
    # Create a logic tree with the problematic pattern:
    # - Two alternatives with same value (0.05) but different requirements
    tree = LogicTree(
        [
            Node(
                "site_class",
                [
                    Alternative("D", weight=0.6),
                    Alternative("E", weight=0.4),
                ],
            ),
            Node(
                "kappa",
                [
                    # These two alternatives have the same value but different requirements
                    Alternative(0.05, weight=0.3, requires={"site_class": "D"}),
                    Alternative(0.05, weight=0.7, requires={"site_class": "E"}),
                    # Add another value to make it more interesting
                    Alternative(0.06, weight=1.0, requires={"site_class": "D"}),
                ],
            ),
        ]
    )

    branches = list(tree)

    # Should have 3 branches total:
    # - Site D with kappa 0.05 (weight = 0.6 * 0.3 = 0.18)
    # - Site D with kappa 0.06 (weight = 0.6 * 1.0 = 0.60)
    # - Site E with kappa 0.05 (weight = 0.4 * 0.7 = 0.28)
    assert len(branches) == 3

    # Extract branches by site class
    site_d_branches = [b for b in branches if b.value("site_class") == "D"]
    site_e_branches = [b for b in branches if b.value("site_class") == "E"]

    # Site D should have 2 branches (kappa 0.05 and 0.06)
    assert len(site_d_branches) == 2
    site_d_kappa_values = {b.value("kappa") for b in site_d_branches}
    assert site_d_kappa_values == {0.05, 0.06}

    # Site E should have 1 branch (kappa 0.05)
    assert len(site_e_branches) == 1
    assert site_e_branches[0].value("kappa") == 0.05

    # Check weights are calculated correctly
    site_d_kappa_05 = next(b for b in site_d_branches if b.value("kappa") == 0.05)
    site_d_kappa_06 = next(b for b in site_d_branches if b.value("kappa") == 0.06)
    site_e_kappa_05 = site_e_branches[0]

    assert site_d_kappa_05.weight == pytest.approx(0.6 * 0.3)  # 0.18
    assert site_d_kappa_06.weight == pytest.approx(0.6 * 1.0)  # 0.60
    assert site_e_kappa_05.weight == pytest.approx(0.4 * 0.7)  # 0.28


def test_complex_conditional_logic_tree():
    """Test a more complex case similar to the original bug report with multiple sites,
    kappa values, and methods."""
    # This reproduces the original bug scenario
    logic_tree_definition = [
        {
            "name": "site_classification",
            "alts": [
                {"value": "C", "weight": 0.3},
                {"value": "D", "weight": 0.6},
                {"value": "E", "weight": 0.1},
            ],
        },
        {
            "name": "kappa",
            "alts": [
                # Site C alternatives
                {
                    "value": 0.02,
                    "weight": 0.2,
                    "requires": {"site_classification": "C"},
                },
                {
                    "value": 0.03,
                    "weight": 0.6,
                    "requires": {"site_classification": "C"},
                },
                {
                    "value": 0.04,
                    "weight": 0.2,
                    "requires": {"site_classification": "C"},
                },
                # Site D alternatives
                {
                    "value": 0.03,
                    "weight": 0.2,
                    "requires": {"site_classification": "D"},
                },
                {
                    "value": 0.04,
                    "weight": 0.6,
                    "requires": {"site_classification": "D"},
                },
                {
                    "value": 0.05,
                    "weight": 0.2,
                    "requires": {"site_classification": "D"},
                },
                # Site E alternatives - this was the problematic case
                {
                    "value": 0.05,
                    "weight": 0.4,
                    "requires": {"site_classification": "E"},
                },
                {
                    "value": 0.06,
                    "weight": 0.6,
                    "requires": {"site_classification": "E"},
                },
            ],
        },
        {
            "name": "randomization_method",
            "alts": [
                {"value": "monte_carlo", "weight": 0.4},
                {"value": "latin_hypercube", "weight": 0.4},
                {"value": "deterministic", "weight": 0.2},
            ],
        },
    ]

    tree = LogicTree.from_list(logic_tree_definition)
    branches = list(tree)

    # Total expected branches:
    # Site C: 3 kappa × 3 methods = 9
    # Site D: 3 kappa × 3 methods = 9
    # Site E: 2 kappa × 3 methods = 6  (this was failing before the fix)
    # Total: 24 branches
    assert len(branches) == 24

    # Check site E specifically (this was the failing case)
    site_e_branches = [b for b in branches if b.value("site_classification") == "E"]
    assert len(site_e_branches) == 6  # 2 kappa values × 3 methods

    # Should have both kappa values for site E
    site_e_kappa_values = {b.value("kappa") for b in site_e_branches}
    assert site_e_kappa_values == {0.05, 0.06}

    # Count branches for each kappa value in site E
    kappa_05_count = len([b for b in site_e_branches if b.value("kappa") == 0.05])
    kappa_06_count = len([b for b in site_e_branches if b.value("kappa") == 0.06])

    # Should have 3 branches for each kappa value (one for each method)
    assert kappa_05_count == 3
    assert kappa_06_count == 3

    # Verify each method appears for both kappa values
    methods = {"monte_carlo", "latin_hypercube", "deterministic"}
    kappa_05_methods = {
        b.value("randomization_method")
        for b in site_e_branches
        if b.value("kappa") == 0.05
    }
    kappa_06_methods = {
        b.value("randomization_method")
        for b in site_e_branches
        if b.value("kappa") == 0.06
    }

    assert kappa_05_methods == methods
    assert kappa_06_methods == methods


# ---------------------------------------------------------------------------
# Fixtures and helpers for variance / tornado tests
# ---------------------------------------------------------------------------


def _make_output(tree, value_func, refs=None):
    """Build an Output with branches stored in names.

    Parameters
    ----------
    tree : LogicTree
        A rectangular logic tree.
    value_func : callable
        ``value_func(branch, ref) -> float`` producing the output value.
    refs : array_like, optional
        Reference values. Defaults to ``[1.0]``.

    Returns
    -------
    Output
        An Output instance ready for ``.to_xarray(tree)``.
    """
    if refs is None:
        refs = np.array([1.0])
    else:
        refs = np.asarray(refs)
    output = Output(refs)
    for branch in tree:
        vals = np.array([value_func(branch, r) for r in refs])
        output._add_values(vals)
        output._names.append(branch)
    return output


@pytest.fixture
def rectangular_tree():
    return LogicTree(
        [
            Node(
                "A",
                [Alternative(1.0, weight=0.3), Alternative(2.0, weight=0.7)],
            ),
            Node(
                "B",
                [
                    Alternative(10.0, weight=0.2),
                    Alternative(20.0, weight=0.5),
                    Alternative(30.0, weight=0.3),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# is_rectangular
# ---------------------------------------------------------------------------


def test_is_rectangular_true(rectangular_tree):
    assert rectangular_tree.is_rectangular


def test_is_rectangular_false(my_tree):
    assert not my_tree.is_rectangular


# ---------------------------------------------------------------------------
# Output.to_xarray
# ---------------------------------------------------------------------------


def test_to_xarray_shape(rectangular_tree):
    output = _make_output(
        rectangular_tree,
        lambda b, r: b.value("A") + b.value("B") + r,
        refs=np.arange(5, dtype=float),
    )

    da = output.to_xarray(rectangular_tree)

    assert da.dims == ("ref", "A", "B")
    assert da.shape == (5, 2, 3)


def test_to_xarray_values(rectangular_tree):
    output = _make_output(
        rectangular_tree,
        lambda b, r: b.value("A") * 100 + b.value("B") + r,
        refs=np.array([1.0, 2.0]),
    )
    branches = list(rectangular_tree)

    da = output.to_xarray(rectangular_tree)

    for branch in branches:
        a_val = branch.value("A")
        b_val = branch.value("B")
        expected = np.array([a_val * 100 + b_val + r for r in [1.0, 2.0]])
        np.testing.assert_allclose(da.sel(A=a_val, B=b_val).values, expected)


def test_to_xarray_rejects_conditional(my_tree):
    # Build a minimal output with string-named entries (not branches)
    output = Output(np.array([1.0]))
    for branch in my_tree:
        output._add_values(np.array([1.0]))
        output._names.append(branch)

    with pytest.raises(ValueError, match="rectangular"):
        output.to_xarray(my_tree)


# ---------------------------------------------------------------------------
# separation_of_variance
# ---------------------------------------------------------------------------


def test_separation_of_variance_single_varying_node():
    """When only one node varies, it should capture ~100% of the variance."""
    tree = LogicTree(
        [
            Node("X", [Alternative(1.0, weight=0.5), Alternative(2.0, weight=0.5)]),
            Node("Y", [Alternative(10.0, weight=0.5), Alternative(20.0, weight=0.5)]),
        ]
    )
    # Output depends only on X, not on Y → all variance from X
    output = _make_output(tree, lambda b, r: b.value("X"))
    da = output.to_xarray(tree)

    ds = separation_of_variance(da, tree, ref_value=1.0)

    assert ds["variance_fraction"].sel(node="X").item() == pytest.approx(1.0, abs=1e-10)
    assert ds["variance_fraction"].sel(node="Y").item() == pytest.approx(0.0, abs=1e-10)


def test_separation_of_variance_fractions_sum_to_one():
    """Variance fractions (including interaction) should sum to 1."""
    tree = LogicTree(
        [
            Node(
                "A",
                [Alternative(1.0, weight=0.3), Alternative(3.0, weight=0.7)],
            ),
            Node(
                "B",
                [Alternative(2.0, weight=0.4), Alternative(5.0, weight=0.6)],
            ),
        ]
    )
    output = _make_output(
        tree, lambda b, r: b.value("A") * b.value("B"), refs=np.array([0.5])
    )
    da = output.to_xarray(tree)

    ds = separation_of_variance(da, tree, ref_value=0.5)

    total = ds["variance_fraction"].values.sum()
    assert total == pytest.approx(1.0, abs=1e-10)


def test_separation_of_variance_attrs():
    tree = LogicTree(
        [
            Node("P", [Alternative(2.0, weight=0.5), Alternative(4.0, weight=0.5)]),
        ]
    )
    output = _make_output(tree, lambda b, r: b.value("P"))
    da = output.to_xarray(tree)

    ds = separation_of_variance(da, tree, ref_value=1.0)

    assert "weighted_mean" in ds.attrs
    assert "total_variance" in ds.attrs
    assert ds.attrs["total_variance"] > 0


# ---------------------------------------------------------------------------
# compute_marginals
# ---------------------------------------------------------------------------


def test_compute_marginals_single_node():
    """Marginals for a single-node tree equal the raw values."""
    tree = LogicTree(
        [Node("P", [Alternative(2.0, weight=0.5), Alternative(8.0, weight=0.5)])]
    )
    output = _make_output(tree, lambda b, r: b.value("P"))
    da = output.to_xarray(tree)

    ds = compute_marginals(da, tree, ref_value=1.0)

    np.testing.assert_allclose(ds["P"].values, [2.0, 8.0])
    # Geometric mean of 2 and 8 with equal weights = sqrt(16) = 4
    assert ds.attrs["weighted_mean"] == pytest.approx(4.0, rel=1e-10)


def test_compute_marginals_independent_nodes():
    """When output = A * B, marginals reflect each node independently."""
    tree = LogicTree(
        [
            Node("A", [Alternative(1.0, weight=0.5), Alternative(3.0, weight=0.5)]),
            Node("B", [Alternative(2.0, weight=0.5), Alternative(4.0, weight=0.5)]),
        ]
    )
    output = _make_output(tree, lambda b, r: b.value("A") * b.value("B"))
    da = output.to_xarray(tree)

    ds = compute_marginals(da, tree, ref_value=1.0)

    # Marginal for A: geometric mean over B of (A*B)
    # A=1: geom_mean(1*2, 1*4) = 1 * geom_mean(2,4) = 1 * sqrt(8)
    # A=3: geom_mean(3*2, 3*4) = 3 * sqrt(8)
    geom_b = np.sqrt(2.0 * 4.0)
    np.testing.assert_allclose(ds["A"].values, [1.0 * geom_b, 3.0 * geom_b], rtol=1e-10)


# ---------------------------------------------------------------------------
# plot_tornado
# ---------------------------------------------------------------------------


def test_plot_tornado_returns_axes():
    import matplotlib

    matplotlib.use("Agg")

    tree = LogicTree(
        [
            Node("A", [Alternative(1.0, weight=0.5), Alternative(2.0, weight=0.5)]),
            Node("B", [Alternative(3.0, weight=0.5), Alternative(6.0, weight=0.5)]),
        ]
    )
    output = _make_output(tree, lambda b, r: b.value("A") + b.value("B"))
    da = output.to_xarray(tree)
    ds = compute_marginals(da, tree, ref_value=1.0)

    ax = plot_tornado(ds)

    assert ax is not None
    # Should have one bar per node (A and B)
    assert len(ax.patches) == 2
