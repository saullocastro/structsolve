"""Tests of the end of the non-linear analyses

``Analysis.stop_criterion``, the criteria of :mod:`structsolve.stopping`
and ``Analysis.final_load_factor``.

"""
import numpy as np
import pytest

from analytic_problems import von_mises_truss, hardening_springs
from structsolve.stopping import value_reached
from test_arc_length import check_equilibrium

ARC_METHODS = ['arc_length_riks', 'arc_length_crisfield',
               'displacement_control']


def kwargs_of(method):
    if method == 'displacement_control':
        return dict(control_vector=np.array([1., 0.]))
    return {}


@pytest.mark.parametrize('method', ARC_METHODS)
def test_value_reached_beyond_unit_load_factor(method):
    """With ``final_load_factor=None`` the path is traced beyond the load
    factor 1 until the apex deflection reaches 3"""
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method=method, initialInc=0.05, final_load_factor=None,
               maxArcLength=1e3, stop_criterion=value_reached(lambda c: c[0], 3.),
               **kwargs_of(method))
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    w = np.array([c[0] for c in p.an.cs])
    assert w[-1] >= 3. > w[-2]
    assert p.an.increments[-1] > 1.


def test_value_reached_negative():
    stop = value_reached(lambda c: c[0], -1.)

    class An:
        cs = [np.array([-0.5])]
    assert not stop(An)
    An.cs = [np.array([-1.5])]
    assert stop(An)


@pytest.mark.parametrize('method', ['NR'] + ARC_METHODS)
def test_final_load_factor(method):
    p = hardening_springs()
    kwargs = {}
    if method == 'displacement_control':
        kwargs['control_vector'] = np.array([0., 0., 0., 1.])
    p.analysis(NL_method=method, initialInc=0.1, final_load_factor=1.7,
               **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.7
    np.testing.assert_allclose(p.an.cs[-1], p.solve_reference(lbd=1.7),
                               rtol=1e-6)


def test_final_load_factor_none_newton_raphson():
    p = hardening_springs()
    p.analysis(NL_method='NR', final_load_factor=None)
    with pytest.raises(ValueError, match='final_load_factor'):
        p.an.static(NLgeom=True, silent=True)


@pytest.mark.parametrize('method', ['NR'] + ARC_METHODS[:2])
def test_stop_criterion_receives_the_analysis(method):
    """The criterion sees every converged increment, with its inertia"""
    seen = []

    def stop(an):
        seen.append((len(an.increments), len(an.cs), len(an.det_signs)))
        return len(an.increments) == 3

    p = von_mises_truss(frac=0.99 if method == 'NR' else 1.5)
    p.analysis(NL_method=method, initialInc=0.05, inertia=True,
               stop_criterion=stop)
    p.an.static(NLgeom=True, silent=True)
    assert len(p.an.increments) == 3
    assert seen == [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
