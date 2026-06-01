from __future__ import annotations

import gzip
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
    requires : Dict[str, Any], optional
        Requirements for this realization to be valid, by default an empty dict.
    excludes : Dict[str, Any], optional
        Exclusions for this realization to be valid, by default an empty dict.
    """

    name: str
    value: str | float | int
    weight: float = 1
    params: dict[str, Any] = field(default_factory=dict)
    requires: dict[str, Any] = field(default_factory=dict)
    excludes: dict[str, Any] = field(default_factory=dict)

    def is_valid(self, branch):
        """
        Check if this realization is valid given a branch.

        Parameters
        ----------
        branch : Branch
            The branch to check against.

        Returns
        -------
        bool
            True if the realization is valid, False otherwise.
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
                yield Realization(
                    self.name, a.value, a.weight, a.params, a.requires, a.excludes
                )

    def to_xarray(self, dim_name: str, name: str = "") -> xr.DataArray:
        """
        Convert the node to a numpy array.

        Returns
        -------
        xr.DataArray
            A dataarray of the alternative values.
        """
        return xr.DataArray(
            np.array([a.weight for a in self.alts]),
            dims=(dim_name,),
            coords={dim_name: [a.value for a in self.alts]},
            name=name,
        )

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

    @classmethod
    def from_miller_rice(cls, name, nbranches):
        # Values from Table 3 in Miller and Rice (1983)
        DATA = {
            2: [(0.166667, 0.500000), (0.833333, 0.500000)],
            3: [(0.084669, 0.247614), (0.500000, 0.504771), (0.915331, 0.247614)],
            4: [
                (0.051621, 0.150361),
                (0.312208, 0.349639),
                (0.687792, 0.349639),
                (0.948379, 0.150361),
            ],
            5: [
                (0.034893, 0.101080),
                (0.211702, 0.244290),
                (0.500000, 0.309260),
                (0.788298, 0.244290),
                (0.965107, 0.101080),
            ],
            6: [
                (0.025219, 0.072713),
                (0.152820, 0.178624),
                (0.371852, 0.248663),
                (0.628148, 0.248663),
                (0.847180, 0.178624),
                (0.974781, 0.072713),
            ],
            7: [
                (0.019106, 0.054866),
                (0.115498, 0.135893),
                (0.285336, 0.198097),
                (0.500000, 0.222288),
                (0.714664, 0.198097),
                (0.884502, 0.135893),
                (0.980894, 0.054866),
            ],
            8: [
                (0.014992, 0.042899),
                (0.090374, 0.106727),
                (0.225240, 0.159700),
                (0.403080, 0.190674),
                (0.596920, 0.190674),
                (0.774760, 0.159700),
                (0.909626, 0.106727),
                (0.985008, 0.042899),
            ],
            9: [
                (0.012086, 0.034479),
                (0.072658, 0.085990),
                (0.182090, 0.130813),
                (0.330161, 0.162023),
                (0.500000, 0.173391),
                (0.669839, 0.162023),
                (0.817910, 0.130813),
                (0.927342, 0.085990),
                (0.987914, 0.034479),
            ],
            10: [
                (0.009957, 0.028328),
                (0.059697, 0.070740),
                (0.150157, 0.108826),
                (0.274659, 0.138009),
                (0.422152, 0.154097),
                (0.577848, 0.154097),
                (0.725341, 0.138009),
                (0.849843, 0.108826),
                (0.940303, 0.070740),
                (0.990043, 0.028328),
            ],
        }

        return cls(name, [Alternative(frac, wt) for frac, wt in DATA[nbranches]])


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
        return np.prod([p.weight for p in self])

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
        # Keep track of seen branches to avoid duplicates when multiple alternatives
        # have the same value but different conditional requirements
        seen_branches = set()

        for reals in itertools.product(*self.nodes):
            branch = Branch({r.name: r for r in reals})
            if self.is_valid(branch):
                # Create a hashable representation of the branch values
                branch_key = tuple(sorted((k, v) for k, v in branch.as_dict().items()))
                if branch_key not in seen_branches:
                    seen_branches.add(branch_key)
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
            # Check if this specific realization is valid for the branch
            if not param.is_valid(branch):
                return False

        return True

    def __getitem__(self, key):
        for n in self.nodes:
            if n.name == key:
                return n

    @property
    def is_rectangular(self) -> bool:
        """Check if the logic tree is fully crossed (no conditional branches).

        Returns
        -------
        bool
            True if no alternatives have ``requires`` or ``excludes`` conditions.
        """
        return all(
            not a.requires and not a.excludes
            for node in self.nodes
            for a in node.alts
        )

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


def separation_of_variance(
    da: xr.DataArray,
    tree: LogicTree,
    ref_value: float,
) -> xr.Dataset:
    """Compute weighted variance decomposition at a single reference value.

    For each node in the logic tree the marginal weighted variance is computed
    in log-space. The fraction of total weighted variance attributable to each
    node is returned, along with any residual interaction term.

    Parameters
    ----------
    da : xr.DataArray
        N-D DataArray as returned by :meth:`~pystrata.output.Output.to_xarray`.
    tree : LogicTree
        The rectangular logic tree.
    ref_value : float
        Reference value (frequency or depth) at which to evaluate.

    Returns
    -------
    xr.Dataset
        Dataset with dimension ``node`` and variables:

        - ``variance`` – absolute weighted variance per node
        - ``variance_fraction`` – fraction of total variance per node
        - ``weighted_mean`` – scalar, the overall weighted geometric mean
    """
    ref_dim = da.dims[0]
    node_names = [n.name for n in tree.nodes]

    # Select the slice at the requested reference value
    sliced = da.sel({ref_dim: ref_value}, method="nearest")
    ln_vals = np.log(sliced.values)

    # Build N-D weight array from outer product of node weights
    weight_arrays = []
    for node in tree.nodes:
        w = np.array([a.weight for a in node.alts])
        weight_arrays.append(w)

    weights = weight_arrays[0]
    for w in weight_arrays[1:]:
        weights = np.multiply.outer(weights, w)

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Total weighted mean and variance in log-space
    total_mean = np.sum(weights * ln_vals)
    total_var = np.sum(weights * (ln_vals - total_mean) ** 2)

    # Marginal variance for each node
    node_variances = []
    for k, node in enumerate(tree.nodes):
        # Weighted mean along all other dimensions
        marginal = ln_vals
        marginal_w = weights
        # Sum over all axes except axis k (iterating in reverse to keep indices stable)
        other_axes = [i for i in range(len(node_names)) if i != k]
        for ax in sorted(other_axes, reverse=True):
            # Weighted average along this axis
            w_ax = weight_arrays[ax]
            # Normalize the per-axis weights
            w_ax = w_ax / w_ax.sum()
            # Broadcast and sum
            shape = [1] * len(node_names)
            shape[ax] = len(w_ax)
            w_broadcast = w_ax.reshape(shape)
            marginal = np.sum(marginal * w_broadcast, axis=ax)
            marginal_w = np.sum(marginal_w, axis=ax)

        # Now marginal is 1-D along node k
        w_k = weight_arrays[k]
        w_k = w_k / w_k.sum()
        marginal_mean = np.sum(w_k * marginal)
        node_var = np.sum(w_k * (marginal - marginal_mean) ** 2)
        node_variances.append(node_var)

    node_variances = np.array(node_variances)

    # Interaction / residual
    interaction = max(0.0, total_var - node_variances.sum())

    labels = node_names + ["Interaction"]
    variances = np.append(node_variances, interaction)

    fractions = variances / total_var if total_var > 0 else np.zeros_like(variances)

    return xr.Dataset(
        {
            "variance": ("node", variances),
            "variance_fraction": ("node", fractions),
        },
        coords={"node": labels},
        attrs={
            "weighted_mean": float(np.exp(total_mean)),
            "total_variance": float(total_var),
            "ref_value": float(ref_value),
        },
    )


def compute_marginals(
    da: xr.DataArray,
    tree: LogicTree,
    ref_value: float,
) -> xr.Dataset:
    """Compute marginal weighted geometric means per node at a reference value.

    For each node, the output values are averaged (in log-space, weighted) over
    all other nodes to produce a 1-D marginal for each alternative.  The result
    is suitable for a tornado plot.

    Parameters
    ----------
    da : xr.DataArray
        N-D DataArray as returned by :meth:`~pystrata.output.Output.to_xarray`.
    tree : LogicTree
        The rectangular logic tree.
    ref_value : float
        Reference value (frequency or depth) at which to evaluate.

    Returns
    -------
    xr.Dataset
        Dataset with variables named after each node containing the marginal
        geometric means (one value per alternative).  The overall weighted
        geometric mean is stored in ``attrs["weighted_mean"]``.
    """
    ref_dim = da.dims[0]
    node_names = [n.name for n in tree.nodes]

    sliced = da.sel({ref_dim: ref_value}, method="nearest")
    ln_vals = np.log(sliced.values)

    # Per-node normalized weight arrays
    weight_arrays = []
    for node in tree.nodes:
        w = np.array([a.weight for a in node.alts])
        weight_arrays.append(w / w.sum())

    # Overall weighted mean
    full_weights = weight_arrays[0]
    for w in weight_arrays[1:]:
        full_weights = np.multiply.outer(full_weights, w)
    total_mean = float(np.exp(np.sum(full_weights * ln_vals)))

    # Marginal for each node
    data_vars = {}
    for k, node in enumerate(tree.nodes):
        marginal = ln_vals
        other_axes = [i for i in range(len(node_names)) if i != k]
        for ax in sorted(other_axes, reverse=True):
            shape = [1] * len(node_names)
            shape[ax] = len(weight_arrays[ax])
            w_broadcast = weight_arrays[ax].reshape(shape)
            marginal = np.sum(marginal * w_broadcast, axis=ax)
        # marginal is 1-D along node k — convert back from log-space
        data_vars[node.name] = (
            f"{node.name}_alt",
            np.exp(marginal),
            {"alternatives": list(node.options)},
        )

    ds = xr.Dataset(
        data_vars,
        attrs={"weighted_mean": total_mean, "ref_value": float(ref_value)},
    )
    return ds


def plot_tornado(ds: xr.Dataset, ax=None, **kwds):
    """Plot a tornado chart of marginal value ranges per logic-tree node.

    Each horizontal bar spans from the minimum to maximum marginal weighted
    geometric mean across a node's alternatives. A vertical line marks the
    overall weighted geometric mean.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset as returned by :func:`compute_marginals`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if not provided.
    **kwds
        Additional keyword arguments passed to ``ax.barh``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    mean = ds.attrs["weighted_mean"]

    # Collect (node_name, low, high) for each variable
    entries = []
    for name in ds.data_vars:
        vals = ds[name].values
        entries.append((name, vals.min(), vals.max()))

    # Sort by range width (smallest swing at bottom, largest at top)
    entries.sort(key=lambda e: e[2] - e[1])
    labels = [e[0] for e in entries]
    lows = np.array([e[1] for e in entries])
    highs = np.array([e[2] for e in entries])

    if ax is None:
        fig, ax = plt.subplots()

    bar_kwds = {"color": "C0", "edgecolor": "black", "height": 0.6} | kwds
    bars = ax.barh(
        range(len(labels)),
        highs - lows,
        left=lows,
        **bar_kwds,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Label bars with low / high values
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        ax.text(lo - 0.01 * mean, i, f"{lo:.3g}", va="center", ha="right")
        ax.text(hi + 0.01 * mean, i, f"{hi:.3g}", va="center", ha="left")

    # Reference line at overall weighted mean
    ax.axvline(mean, color="black", ls="--", lw=1, label=f"Mean = {mean:.3g}")
    ax.legend()

    ref_val = ds.attrs.get("ref_value", "")
    ax.set_title(f"Tornado Plot (ref = {ref_val})")

    return ax


def plot_separation_of_variance(da, tree, ref_values, ax=None, **kwds):
    """Plot variance fractions across multiple reference values.

    Produces a stacked area chart showing how the variance fraction
    attributable to each logic tree node changes across a range of reference
    values (e.g., frequencies).

    Parameters
    ----------
    da : xr.DataArray
        N-D DataArray as returned by :meth:`~pystrata.output.Output.to_xarray`.
    tree : LogicTree
        The rectangular logic tree.
    ref_values : array_like
        Reference values at which to evaluate the variance decomposition.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if not provided.
    **kwds
        Additional keyword arguments passed to ``ax.stackplot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ref_values = np.asarray(ref_values)
    datasets = [separation_of_variance(da, tree, rv) for rv in ref_values]

    labels = datasets[0]["node"].values.astype(str)
    fractions = np.array([ds["variance_fraction"].values for ds in datasets]).T

    if ax is None:
        fig, ax = plt.subplots()

    ax.stackplot(ref_values, fractions, labels=labels, **kwds)
    ax.set_xlabel(da.dims[0])
    ax.set_ylabel("Fraction of Total Variance")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Separation of Variance")

    return ax
