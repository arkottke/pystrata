import pytest

from pystrata.logic_tree import Alternative, LogicTree, Node

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
    """
    Test that multiple alternatives with the same value but different 'requires'
    conditions are both included in the logic tree branches.

    This test addresses a bug where only the first alternative with a given value
    was being used in branch validation, causing branches with later alternatives
    to be incorrectly marked as invalid.
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
    """
    Test a more complex case similar to the original bug report with multiple
    sites, kappa values, and methods.
    """
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
