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
