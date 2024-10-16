from __future__ import annotations

import gzip
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

AlternativeValueType = Union[str, int, float, tuple[str]]


@dataclass
class Realization:
    """
    Represents a realization of a node in a logic tree.

    Parameters
    ----------
    name : str
        The name of the realization.
    value : Union[str, float, int]
        The value of the realization.
    weight : float, optional
        The weight of the realization, by default 1.
    params : Dict[str, Any], optional
        Additional parameters for the realization, by default an empty dict.
    """

    name: str
    value: str | float | int
    weight: float = 1
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alternative:
    """
    Represents an alternative in a node of a logic tree.

    Parameters
    ----------
    value : AlternativeValueType
        The value of the alternative.
    weight : float, optional
        The weight of the alternative, by default 1.0.
    requires : Dict[str, Any], optional
        Requirements for this alternative to be valid, by default an empty dict.
    excludes : Dict[str, Any], optional
        Exclusions for this alternative to be valid, by default an empty dict.
    params : Dict[str, Any], optional
        Additional parameters for the alternative, by default an empty dict.
    """

    value: AlternativeValueType
    weight: float = 1.0
    requires: dict[str, Any] = field(default_factory=dict)
    excludes: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def is_valid(self, branch):
        """
        Check if this alternative is valid given a branch.

        Parameters
        ----------
        branch : Branch
            The branch to check against.

        Returns
        -------
        bool
            True if the alternative is valid, False otherwise.
        """

        def matches(ref, check):
            if isinstance(ref, list):
                ret = check in ref
            elif isinstance(ref, float):
                ret = np.isclose(ref, check)
            else:
                ret = ref == check
            return ret

        okay = True

        if self.requires:
            # Check that the required realizations are present
            okay = all(matches(v, branch[k].value) for k, v in self.requires.items())

        if okay and self.excludes:
            # Check that the excludes realizations are _not_ present
            okay &= not all(
                matches(v, branch[k].value) for k, v in self.excludes.items()
            )

        return okay


@dataclass
class Node:
    """
    Represents a node in a logic tree.

    Parameters
    ----------
    name : str
        The name of the node.
    alts : List[Union[Alternative, AlternativeValueType]]
        The alternatives for this node.
    """

    name: str
    alts: list[Alternative | AlternativeValueType]

    def __post_init__(self):
        self.alts = [
            a if isinstance(a, Alternative) else Alternative(a) for a in self.alts
        ]

    def __len__(self):
        return len(self.alts)

    def __getitem__(self, index):
        return self.alts[index]

    def by_value(self, value):
        """
        Get an alternative by its value.

        Parameters
        ----------
        value : Any
            The value to search for.

        Returns
        -------
        Alternative
            The alternative with the matching value.
        """
        for a in self.alts:
            if (
                isinstance(value, float) and np.isclose(a.value, value)
            ) or a.value == value:
                return a

    def __iter__(self):
        for a in self.alts:
            if a.weight > 0:
                yield Realization(self.name, a.value, a.weight, a.params)

    @property
    def options(self):
        """
        Get all alternative values for this node.

        Returns
        -------
        tuple
            A tuple of all alternative values.
        """
        return tuple(a.value for a in self.alts)

    @classmethod
    def from_dict(cls, d):
        """
        Create a Node from a dictionary.

        Parameters
        ----------
        d : dict
            A dictionary containing 'name' and 'alts' keys.

        Returns
        -------
        Node
            A new Node instance.
        """
        return cls(d["name"], [Alternative(**a) for a in d["alts"]])


@dataclass
class Branch:
    """
    Represents a branch in a logic tree.

    Parameters
    ----------
    params : Dict[str, Realization]
        A dictionary of parameter names to Realizations.
    """

    params: dict[str, Realization]

    def __getitem__(self, key):
        return self.params[key]

    def __iter__(self):
        yield from self.params.values()

    def __contains__(self, index):
        return index in self.params

    @property
    def weight(self):
        """
        Calculate the weight of this branch.

        Returns
        -------
        float
            The product of all realization weights in this branch.
        """
        return np.product([p.weight for p in self])

    def value(self, key):
        """
        Get the value of a realization by key.

        Parameters
        ----------
        key : str
            The key of the realization.

        Returns
        -------
        Any
            The value of the realization.
        """
        return self.params[key].value

    def as_dict(self):
        """
        Convert the branch to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the branch.
        """
        return {k: a.value for k, a in self.params.items()}


@dataclass
class LogicTree:
    """
    Represents a logic tree.

    Parameters
    ----------
    nodes : List[Node]
        A list of nodes in the logic tree.
    """

    nodes: list[Node]

    def __iter__(self) -> Branch:
        for reals in itertools.product(*self.nodes):
            branch = Branch({r.name: r for r in reals})
            if self.is_valid(branch):
                yield branch

    def is_valid(self, branch):
        """
        Check if a branch is valid according to the logic tree rules.

        Parameters
        ----------
        branch : Branch
            The branch to check.

        Returns
        -------
        bool
            True if the branch is valid, False otherwise.
        """
        for param in branch.params.values():
            # Select the alternative on the logic tree
            alt = self[param.name].by_value(param.value)
            if not alt.is_valid(branch):
                return False
        return True

    def __getitem__(self, key):
        for n in self.nodes:
            if n.name == key:
                return n

    @classmethod
    def from_json(cls, fname: str | Path) -> LogicTree:
        _open = gzip.open if str(fname).endswith(".gz") else open

        with _open(fname) as fp:
            items = json.load(fp)

        return cls.from_list(items)

    @classmethod
    def from_list(cls, dicts: list[dict[str, Any]]) -> LogicTree:
        """
        Create a LogicTree from a list of dictionaries.

        Parameters
        ----------
        dicts : List[Dict[str, Any]]
            A list of dictionaries, each representing a node.

        Returns
        -------
        LogicTree
            A new LogicTree instance.
        """
        nodes = [Node.from_dict(d) for d in dicts]
        return cls(nodes)
