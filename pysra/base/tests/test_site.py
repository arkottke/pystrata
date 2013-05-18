import nose

from numpy.testing import assert_almost_equal

from pysra.base import site


def nlp_setup():
    """Setup for the NonlinearProperty tests"""
    global nlp
    nlp = site.NonlinearProperty('', [0.01, 1], [0., 1.])


def nlp_teardown():
    """Teardown for the NonlinearProperty tests"""
    global nlp
    del nlp


@nose.with_setup(nlp_setup, nlp_teardown)
def test_nlp_lowerbound():
    global nlp
    assert_almost_equal(nlp(0.001), 0.)


@nose.with_setup(nlp_setup, nlp_teardown)
def test_nlp_upperbound():
    global nlp
    assert_almost_equal(nlp(2.), 1.)


@nose.with_setup(nlp_setup, nlp_teardown)
def test_nlp_midpoint():
    global nlp
    assert_almost_equal(nlp(0.1), 0.5)


@nose.with_setup(nlp_setup, nlp_teardown)
def test_nlp_update():
    global nlp
    new_values = [0, 2]
    nlp.values = new_values
    assert_almost_equal(new_values, nlp.values)

    new_strains = [0.1, 10]
    nlp.strains = new_strains
    assert_almost_equal(new_strains, nlp.strains)

    assert_almost_equal(nlp(1.), 1.)


def test_iterative_value():
    """Test the iterative value and relative error."""
    iv = site.IterativeValue(11)
    value = 10
    iv.value = value
    assert_almost_equal(iv.value, value)
    assert_almost_equal(iv.relative_error(), 10.)


def test_soil_type_linear():
    """Test the soil type update process on a linear material."""
    damping = 1.0
    l = site.Layer(site.SoilType('', 18.0, 9.81, None, damping), 2., 500.)
    l.strain = 0.1

    assert_almost_equal(l.shear_mod.value, l.max_shear_mod())
    assert_almost_equal(l.damping.value, damping)


def test_soil_type_iterative():
    """Test the soil type update process on a nonlinear property."""
    mod_reduc = site.NonlinearProperty('', [0.01, 1.], [1, 0])
    damping = site.NonlinearProperty('', [0.01, 1.], [0, 10])

    st = site.SoilType('', 18.0, 9.81, mod_reduc, damping)
    l = site.Layer(st, 2., 500.)

    strain = 0.1
    l.strain = strain

    assert_almost_equal(l.strain.value, strain)
    assert_almost_equal(l.shear_mod.value, 0.5 * l.max_shear_mod())
    assert_almost_equal(l.damping.value, 5.0)

