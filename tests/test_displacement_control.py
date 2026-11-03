"""Tests of the indirect displacement control

``Analysis.NL_method = 'displacement_control'`` with
``Analysis.control_vector``, see :func:`.arc_length._solver_arc_length`.
The von Mises truss loaded through a soft spring has a snap-through in the
load and a snap-back in the displacement of the loaded point, whereas the
apex deflection increases monotonically along the path: the control of the
apex traces the same path as the arc-length methods.

"""
import numpy as np
import pytest

from analytic_problems import (hardening_springs, von_mises_truss,
                               linear_problem, Problem)
from test_arc_length import check_equilibrium

FRAC = 1.5
FLIM = 2/(3*np.sqrt(3))


def truss_load_factor(w):
    """Load factor of the equilibrium of the truss at the apex deflection
    ``w``, the force of the spring being the force of the truss"""
    return (w**3 - 3*w**2 + 2*w)/(FRAC*FLIM)


@pytest.mark.parametrize('options', [{},
                                     dict(arc_length_line_search=True),
                                     dict(early_divergence=True)])
def test_snap_back_apex_control(options):
    p = von_mises_truss(frac=FRAC, ks=0.5)
    p.analysis(NL_method='displacement_control',
               control_vector=np.array([1., 0.]), initialInc=0.05, **options)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    cs = np.asarray(p.an.cs)
    w = cs[:, 0]
    assert inc[-1] == 1.0
    np.testing.assert_allclose(inc, truss_load_factor(w), atol=1e-9)
    # snap-through in the load: limit points at +-1/FRAC
    assert np.all(np.diff(w) > 0)
    assert inc[:inc.argmin()].max() > 0.95/FRAC
    assert inc.min() < -0.95/FRAC
    # snap-back in the displacement of the loaded point
    assert np.any(np.diff(cs[:, 1]) < 0)
    assert np.any(np.diff(cs[:, 1]) > 0)

    # the same path as the arc-length method, interpolated along the apex
    # deflection
    q = von_mises_truss(frac=FRAC, ks=0.5)
    q.analysis(NL_method='arc_length_riks', initialInc=0.01, maxInc=0.01)
    q.an.static(NLgeom=True, silent=True)
    wq = np.asarray(q.an.cs)[:, 0]
    lq = np.asarray(q.an.increments)
    assert np.all(np.diff(wq) > 0)
    inside = (w >= wq[0]) & (w <= wq[-1])
    np.testing.assert_allclose(inc[inside], np.interp(w[inside], wq, lq),
                               atol=2e-3)
    # fewer evaluations than the arc-length method with its default steps
    r = von_mises_truss(frac=FRAC, ks=0.5)
    r.analysis(NL_method='arc_length_riks', initialInc=0.05)
    r.an.static(NLgeom=True, silent=True)
    assert len(p.fint_calls) < len(r.fint_calls)


def test_loaded_point_control_misses_snap_back():
    """The displacement of the loaded point has limit points along the
    path, where `psi^T dc_q` vanishes: its control cannot trace the
    snap-back and misses the minimum of the load factor"""
    p = von_mises_truss(frac=FRAC, ks=0.5)
    p.analysis(NL_method='displacement_control',
               control_vector=np.array([0., 1.]), initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    assert inc.min() > -0.9/FRAC
    assert np.all(np.diff(np.asarray(p.an.cs)[:, 1]) > 0)


def test_first_step_and_scale():
    """The first step is ``initialInc`` times the scale, by default the
    controlled quantity in the linear solution: the load factor increment
    along the initial tangent; ``maxArcLength`` is also in units of the
    scale"""
    p = linear_problem()
    psi = np.array([0., 0., 1.])
    p.analysis(NL_method='displacement_control', control_vector=psi,
               initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p, rtol=1e-10)
    inc = np.asarray(p.an.increments)
    assert np.isclose(inc[0], 0.1)
    assert inc[-1] == 1.

    p = von_mises_truss(frac=FRAC, ks=0.5)
    p.analysis(NL_method='displacement_control',
               control_vector=np.array([1., 0.]), initialInc=0.1,
               control_scale=0.2)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    assert np.isclose(p.an.cs[0][0], 0.02)
    assert p.an.increments[-1] == 1.


def test_negative_control_direction():
    """The step keeps the sign for which the controlled quantity increases
    with the load factor at the origin, here a vector of the opposite sign
    of the displacement"""
    p = hardening_springs()
    psi = np.zeros(4)
    psi[-1] = -2.
    p.analysis(NL_method='displacement_control', control_vector=psi,
               initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    assert inc[-1] == 1.
    assert np.all(np.diff(inc) > 0)
    np.testing.assert_allclose(p.an.cs[-1], p.solve_reference(), rtol=1e-5)


def test_errors():
    p = hardening_springs()
    p.analysis(NL_method='displacement_control')
    with pytest.raises(ValueError, match='control_vector'):
        p.an.static(NLgeom=True, silent=True)
    p = hardening_springs()
    p.analysis(NL_method='displacement_control', control_vector=np.ones(3))
    with pytest.raises(ValueError, match='shape'):
        p.an.static(NLgeom=True, silent=True)
    # psi orthogonal to the linear solution
    p = von_mises_truss(frac=FRAC, ks=0.5)
    u = np.linalg.solve(p.kC, p.fext)
    p.analysis(NL_method='displacement_control',
               control_vector=np.array([u[1], -u[0]]))
    with pytest.raises(ValueError, match='control_scale'):
        p.an.static(NLgeom=True, silent=True)
    # orthogonal to round-off: cosine of 1e-14
    p = von_mises_truss(frac=FRAC, ks=0.5)
    p.analysis(NL_method='displacement_control',
               control_vector=np.array([u[1], -u[0]]) + 1e-14*u)
    with pytest.raises(ValueError, match='control_scale'):
        p.an.static(NLgeom=True, silent=True)


def test_control_without_motion(capsys):
    """A controlled quantity that the load does not change, given with
    ``control_scale``, cannot drive the analysis, which is stopped"""
    K = np.diag([2., 3.])
    p = Problem(np.array([1., 0.]), lambda c: K @ c, K,
                lambda c: np.zeros_like(K))
    p.analysis(NL_method='displacement_control',
               control_vector=np.array([0., 1.]), control_scale=1.)
    p.an.static(NLgeom=True, silent=False)
    assert 'The controlled quantity does not change' \
        in capsys.readouterr().out
    assert p.an.increments == []
