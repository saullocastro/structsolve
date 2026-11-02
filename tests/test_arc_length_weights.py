"""Tests of the weights of the degrees of freedom in the arc-length norm

``Analysis.arc_length_weights``, see :func:`.arc_length._solver_arc_length`.

"""
import numpy as np
import pytest

from analytic_problems import von_mises_truss, hardening_springs
from test_arc_length import check_equilibrium, rescaled_dof_problem

METHODS = ['arc_length_riks', 'arc_length_crisfield']


@pytest.mark.parametrize('method', METHODS)
def test_unit_weights_are_the_default(method):
    results = []
    for weights in (None, np.ones(2)):
        p = von_mises_truss(frac=1.5, ks=0.5)
        p.analysis(NL_method=method, initialInc=0.05,
                   arc_length_weights=weights)
        p.an.static(NLgeom=True, silent=True)
        results.append((np.asarray(p.an.increments), np.asarray(p.an.cs),
                        len(p.fint_calls)))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]


@pytest.mark.parametrize('method', METHODS)
def test_null_weight_units_invariance(method):
    r"""The degree of freedom 1 measured in other units, e.g. a rotation:
    with its weight null in the arc-length norm and Crisfield's criterion,
    the whole analysis is independent of its units, whereas with the
    default weights the arc-length constraint depends on them, see
    ``test_arc_length.test_crisfield_criterion_units_invariance``"""
    scales = [1., 1.e3, 1.e-3]
    results = {}
    for weights in (None, np.array([1., 0.])):
        for s in scales:
            p = rescaled_dof_problem(von_mises_truss(frac=1.5, ks=1.), s)
            p.analysis(NL_method=method, initialInc=0.05,
                       arc_length_weights=weights)
            p.an.static(NLgeom=True, silent=True)
            check_equilibrium(p)
            cs = np.asarray(p.an.cs)/np.array([1., s])
            results[weights is None, s] = (np.asarray(p.an.increments), cs,
                                           len(p.fint_calls))
    inc0, cs0, calls0 = results[False, 1.]
    assert inc0[-1] == 1.
    for s in scales[1:]:
        inc, cs, calls = results[False, s]
        assert calls == calls0
        np.testing.assert_allclose(inc, inc0, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(cs, cs0, rtol=1e-9, atol=1e-12)
    calls = [results[True, s][2] for s in scales]
    lengths = [len(results[True, s][0]) for s in scales]
    assert len(set(calls)) > 1 or len(set(lengths)) > 1, (calls, lengths)


@pytest.mark.parametrize('method', METHODS)
def test_weights_with_snap_back(method):
    """Arc length on the apex deflection only: the snap-back in the
    displacement of the loaded point is traced"""
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method=method, initialInc=0.05,
               arc_length_weights=np.array([1., 0.]))
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    cs = np.asarray(p.an.cs)
    assert inc[-1] == 1.
    assert inc.min() < -0.5
    assert np.any(np.diff(cs[:, 1]) < 0)
    assert np.all(np.diff(cs[:, 0]) > 0)


def test_weights_errors():
    p = hardening_springs()
    p.analysis(NL_method='arc_length_riks', arc_length_weights=np.ones(3))
    with pytest.raises(ValueError, match='shape'):
        p.an.static(NLgeom=True, silent=True)
    p = hardening_springs()
    p.analysis(NL_method='arc_length_riks',
               arc_length_weights=np.array([1., -1., 1., 1.]))
    with pytest.raises(ValueError, match='negative'):
        p.an.static(NLgeom=True, silent=True)
