import os
import json

import numpy as np
import scipy.constants

import pytest

from pysra import site

with open(os.path.join(
        os.path.dirname(__file__), 'data', 'kishida_2009.json')) as fp:
    kishida_cases = json.load(fp)


def format_kishida_case_id(case):
    fmt = "({mean_stress:.1f} kN/m², OC={organic_content:.0f} %)"
    return fmt.format(**case)


@pytest.mark.parametrize(
    'case',
    kishida_cases,
    ids=format_kishida_case_id
)
def test_kishida_unit_wt(case):
    st = site.KishidaSoilType(
        'test', unit_wt=None,
        mean_stress=case['mean_stress'],
        organic_content=case['organic_content'],
        strains=case['strains']
    )
    np.testing.assert_allclose(
        st.unit_wt, scipy.constants.g * case['density'], rtol=0.005)


@pytest.mark.parametrize(
    'case',
    kishida_cases,
    ids=format_kishida_case_id
)
@pytest.mark.parametrize(
    'curve,attr,key',
    [
        ('mod_reduc', 'strains', 'strains'),
        ('mod_reduc', 'values', 'mod_reducs'),
        ('damping', 'strains', 'strains'),
        ('damping', 'values', 'dampings'),
    ]
)
def test_kishida_nlc(case, curve, attr, key):
    st = site.KishidaSoilType(
        'test', unit_wt=None,
        mean_stress=case['mean_stress'],
        organic_content=case['organic_content'],
        strains=case['strains']
    )
    np.testing.assert_allclose(
        getattr(getattr(st, curve), attr), case[key],
        rtol=0.005, atol=0.0005
    )
