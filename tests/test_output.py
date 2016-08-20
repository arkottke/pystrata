
import pytest
import numpy as np

from numpy.testing import assert_allclose, assert_array_equal

import pysra



def test_add_refs():
    output = pysra.output.Output()
    refs = [1.1, 2, 3]
    output._add_refs(refs)
    assert_allclose(refs, output.refs)


def test_add_refs_same():
    output = pysra.output.Output()
    # Force float arrays
    a = [1.1, 2, 3]
    b = [1.1, 2, 3]

    output._add_refs(a)
    output._add_refs(b)

    assert np.ndim(output.refs) == 1
    assert_array_equal(output.refs, a)


def test_add_refs_diff():
    output = pysra.output.Output()
    # Force float arrays
    a = [1.1, 2, 3]
    b = [1.1, 2, 3, 4, 5]

    output._add_refs(a)
    output._add_refs(b)

    assert np.ndim(output.refs) == 2
    assert len(output.refs) == len(b)
    assert_array_equal(output.refs[:, 0], a + 2 * [np.nan])
    assert_array_equal(output.refs[:, 1], b)

