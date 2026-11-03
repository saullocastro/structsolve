"""Tests of the generalized displacement control

``Analysis.NL_method = 'generalized_displacement_control'``, see
:func:`.arc_length._solver_arc_length`.

"""
import numpy as np
import pytest

from analytic_problems import (von_mises_truss, hardening_springs,
                               linear_problem, Problem)
from test_arc_length import check_equilibrium

METHOD = 'generalized_displacement_control'


def test_hardening_springs_final_load_factor():
    p = hardening_springs()
    p.analysis(NL_method=METHOD, initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.
    np.testing.assert_allclose(p.an.cs[-1], p.solve_reference(), rtol=1e-6)
    # the step lengths shrink with the stiffness parameter as the springs
    # harden
    assert np.all(np.diff(p.an.increments) > 0)


@pytest.mark.parametrize('ks', [0.5, 10.])
def test_von_mises_truss_limit_points(ks):
    """The load is reversed at the two limit points of the snap-through,
    where the stiffness parameter changes sign, and the path reaches the
    load factor 1 on the far side"""
    p = von_mises_truss(frac=1.5, ks=ks)
    p.analysis(NL_method=METHOD, initialInc=0.05, inertia=True)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    lbds = np.asarray(p.an.increments)
    assert lbds[-1] == 1.
    # the first limit point, the last increment before the load decreases
    ilim = np.flatnonzero(np.diff(lbds) < 0)[0]
    assert abs(lbds[ilim] - 1/1.5) < 0.05
    assert lbds.min() < -0.5
    w = np.array([c[0] for c in p.an.cs])
    assert w[-1] > 1.5
    # beyond the first limit point the branch is unstable, then stable
    dets = np.asarray(p.an.det_signs)
    assert dets[0] == 1 and dets[-1] == 1 and -1 in dets


def test_units_of_the_load():
    """The path does not depend on the units of the load, with the first
    load increment in the same units"""
    p = von_mises_truss(frac=1.5, ks=0.5)
    kwargs = dict(NL_method=METHOD, final_load_factor=None,
                  stop_criterion=lambda an: len(an.increments) == 25)
    p.analysis(initialInc=0.05, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    ref = np.asarray(p.an.increments)
    p.fext = 1e3*p.fext
    p.analysis(initialInc=0.05e-3, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    np.testing.assert_allclose(1e3*np.asarray(p.an.increments), ref,
                               rtol=1e-8)


def test_secant_not_valid():
    p = hardening_springs()
    p.analysis(NL_method=METHOD, arc_length_predictor='secant')
    with pytest.raises(ValueError, match='secant'):
        p.an.static(NLgeom=True, silent=True)


def test_exact_final_load_factor():
    """A linear problem, whose stiffness parameter is 1, reaches the final
    load factor exactly with steps of ``initialInc``"""
    p = linear_problem()
    p.analysis(NL_method=METHOD, initialInc=0.25)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p, rtol=1e-10)
    assert p.an.increments == [0.25, 0.5, 0.75, 1.]


def test_weighted_dofs_without_motion(capsys):
    """With ``arc_length_weights`` only on a degree of freedom that the load
    does not move, the stiffness parameter is not defined after the first
    step and the analysis is stopped"""
    K = np.diag([2., 3.])
    p = Problem(np.array([1., 0.]), lambda c: K @ c, K,
                lambda c: np.zeros_like(K))
    p.analysis(NL_method=METHOD, initialInc=0.2,
               arc_length_weights=np.array([0., 1.]))
    p.an.static(NLgeom=True, silent=False)
    assert 'Null generalized stiffness parameter' in capsys.readouterr().out
    check_equilibrium(p, rtol=1e-10)
    assert p.an.increments == [0.2]
