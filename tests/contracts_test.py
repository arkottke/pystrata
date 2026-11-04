"""Guard the private contract copies against drift from pygmm.contracts."""

import dataclasses

import pytest

from pystrata import _contracts

pygmm_contracts = pytest.importorskip("pygmm.contracts")


def _describe(cls):
    return {
        "frozen": cls.__dataclass_params__.frozen,
        "fields": [(f.name, str(f.type), f.default) for f in dataclasses.fields(cls)],
    }


@pytest.mark.parametrize("name", ["NonlinearSoilCurves", "VelocityProfile"])
def test_matches_pygmm(name):
    assert _describe(getattr(_contracts, name)) == _describe(
        getattr(pygmm_contracts, name)
    )
