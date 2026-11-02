"""Tests of the restart of the non-linear analyses

``Analysis.path_state`` and ``Analysis.restart_state``.

"""
import pickle

import numpy as np
import pytest

from analytic_problems import von_mises_truss, hardening_springs
from test_arc_length import check_equilibrium

ARC_METHODS = ['arc_length_riks', 'arc_length_crisfield',
               'displacement_control', 'generalized_displacement_control']


def kwargs_of(method):
    if method == 'displacement_control':
        return dict(control_vector=np.array([1., 0.]))
    return {}


@pytest.mark.parametrize('method', ARC_METHODS)
def test_restart_continues_the_path(method):
    """An analysis stopped after 6 increments and restarted from its
    ``path_state``, through pickle, traces the same path as without the
    stop"""
    p = von_mises_truss(frac=1.5, ks=0.5)
    kwargs = dict(NL_method=method, initialInc=0.05, inertia=True,
                  **kwargs_of(method))
    p.analysis(**kwargs)
    p.an.static(NLgeom=True, silent=True)
    ref_lbds = list(p.an.increments)
    ref_cs = list(p.an.cs)
    ref_dets = list(p.an.det_signs)
    assert len(ref_lbds) > 10
    p.analysis(stop_criterion=lambda an: len(an.increments) == 6, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    first = list(p.an.increments)
    state = pickle.loads(pickle.dumps(p.an.path_state))
    p.analysis(restart_state=state, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    np.testing.assert_allclose(first + p.an.increments, ref_lbds, rtol=1e-12)
    np.testing.assert_allclose(np.array(p.an.cs), np.array(ref_cs[6:]),
                               rtol=1e-12, atol=1e-14)
    assert p.an.det_signs == ref_dets[6:]


def test_restart_newton_raphson():
    p = hardening_springs()
    p.analysis(NL_method='NR', initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    ref = list(p.an.increments)
    p.analysis(NL_method='NR', initialInc=0.05,
               stop_criterion=lambda an: len(an.increments) == 4)
    p.an.static(NLgeom=True, silent=True)
    first = list(p.an.increments)
    p.analysis(NL_method='NR', initialInc=0.05,
               restart_state=p.an.path_state)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    np.testing.assert_allclose(first + p.an.increments, ref, rtol=1e-12)


def test_restart_method_mismatch():
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method='arc_length_riks',
               stop_criterion=lambda an: len(an.increments) == 2)
    p.an.static(NLgeom=True, silent=True)
    state = p.an.path_state
    p.analysis(NL_method='arc_length_crisfield', restart_state=state)
    with pytest.raises(ValueError, match='restart_state'):
        p.an.static(NLgeom=True, silent=True)
