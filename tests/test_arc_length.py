"""Tests for the arc-length solvers using analytic problems

Both arc-length methods trace the equilibrium path until the load factor
reaches exactly 1.0, the cumulative arc length reaches ``maxArcLength`` or the
arc-length increment becomes smaller than ``minInc``.

"""
import numpy as np
import pytest

from analytic_problems import (hardening_springs, von_mises_truss,
                               linear_problem, convergence_orders, Problem)

METHODS = ['arc_length_riks', 'arc_length_crisfield']

#: apex deflection at the limit point of the von Mises truss
W_LIMIT = 1 - 1/np.sqrt(3)


def check_equilibrium(p, rtol=1e-5):
    an = p.an
    assert len(an.increments) == len(an.cs)
    assert len(an.increments) >= 1
    fext_norm = np.linalg.norm(p.fext)
    for lbd, c in zip(an.increments, an.cs):
        fint = p.fint(c)
        R = lbd*p.fext - fint
        ref = max(np.linalg.norm(lbd*p.fext), np.linalg.norm(fint))
        assert np.linalg.norm(R) <= rtol*ref + 1e-9*fext_norm, (lbd, R)


def scaled_problem(p, force, disp):
    """Same problem with forces multiplied by ``force`` and displacements
    divided by ``disp``"""
    return Problem(force*p.fext,
                   lambda c: force*p.fint(disp*c),
                   force*disp*p.kC,
                   lambda c: force*disp*p.kG(disp*c))


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_linear_problem(method):
    p = linear_problem()
    p.analysis(NL_method=method, initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p, rtol=1e-10)
    inc = np.asarray(p.an.increments)
    assert inc[-1] == 1.0
    assert len(inc) > 1
    assert np.all(np.diff(inc) > 0)
    # the tangent predictor is exact for a linear problem, the last step
    # includes the correction that lands exactly at a load factor of 1.0
    assert max(p.calls_per_step()) <= 2, p.calls_per_step()


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_hardening(method):
    p = hardening_springs()
    p.analysis(NL_method=method, initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    assert inc[-1] == 1.0
    assert np.all(np.diff(inc) > 0)
    np.testing.assert_allclose(p.an.cs[-1], p.solve_reference(), rtol=1e-5)
    assert max(p.calls_per_step()) <= 8, p.calls_per_step()


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_quadratic_convergence(method):
    p = hardening_springs()
    p.analysis(NL_method=method, initialInc=0.3, relTOL=1e-12)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p, rtol=1e-10)
    assert p.an.increments[0] < 1.
    # iterates of the first step, the last one is the converged solution
    cs = [c for step, c in p.fint_calls if step == 0]
    np.testing.assert_allclose(cs[-1], p.an.cs[0])
    ref = p.an.cs[0]
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref) for c in cs[:-1]]
    orders = convergence_orders(errors)
    assert len(orders) >= 1, errors
    assert min(orders) > 1.6, (orders, errors)


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_snap_through(method):
    """Load of 1.5 times the limit load of the von Mises truss

    The path goes through the limit point, the load factor becomes negative
    during the snap-through, and it increases again until the load factor
    reaches 1.0 on the inverted configuration.

    """
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    w = np.array([c[0] for c in p.an.cs])
    assert inc[-1] == 1.0
    # limit load factor is 1/1.5, the minimum along the path is -1/1.5
    assert inc[inc < 1.].max() < 1/1.5 + 1e-6
    assert inc[inc < 1.].max() > 0.6
    assert inc.min() < -0.5
    assert np.all(inc[:-1] < 1.)
    # continuous path along the apex deflection, without jumps
    assert np.all(np.diff(w) > 0)
    assert np.diff(w).max() < 0.5
    assert w[-1] > 2.
    np.testing.assert_allclose(p.an.cs[-1],
                               p.solve_reference(c0=p.an.cs[-1], tol=1e-12),
                               rtol=1e-6)


@pytest.mark.parametrize('convergence, relTOL', [('relative', 1.e-6),
                                                  ('crisfield', 1.e-9)])
@pytest.mark.parametrize('method', METHODS)
def test_arc_length_snap_back(method, convergence, relTOL):
    """A soft spring in series with the truss causes snap-back

    The displacement of the loaded point decreases along part of the path.
    The relative criterion with its former default tolerance,
    ``relTOL=1e-6``, and Crisfield's criterion with its default, ``1e-9``,
    both at the default ``maxInc=1``.

    """
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method=method, initialInc=0.05, convergence=convergence,
               relTOL=relTOL)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    cs = np.asarray(p.an.cs)
    assert inc[-1] == 1.0
    assert inc.min() < -0.5
    assert np.any(np.diff(cs[:, 1]) < 0)
    assert np.all(np.diff(cs[:, 0]) > 0)
    assert cs[-1, 0] > 2.


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_step_cutting(method):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.2, maxNumIter=3)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.0
    assert p.an.cs[-1][0] > 2.


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_modified_NR(method):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05, modified_NR=True,
               compute_every_n=3)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.0
    assert p.an.cs[-1][0] > 2.


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_maxArcLength(method):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05, maxArcLength=0.3)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    assert 0 < inc[-1] < 1.


@pytest.mark.parametrize('method', METHODS)
def test_arc_length_silent(method, capsys):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('method', METHODS + ['NR'])
def test_units_independence(method):
    frac = 1.5 if method != 'NR' else 0.99
    results = []
    for force, disp in [(1., 1.), (1.e6, 1.), (1.e-4, 1.e-3)]:
        p = scaled_problem(von_mises_truss(frac=frac), force, disp)
        p.analysis(NL_method=method, initialInc=0.05)
        p.an.static(NLgeom=True, silent=True)
        check_equilibrium(p)
        results.append((np.asarray(p.an.increments),
                        disp*np.asarray(p.an.cs), len(p.fint_calls)))
    inc0, cs0, calls0 = results[0]
    assert inc0[-1] == 1.0
    for inc, cs, calls in results[1:]:
        assert calls == calls0
        np.testing.assert_allclose(inc, inc0, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(cs, cs0, rtol=1e-6, atol=1e-9)


def rescaled_dof_problem(p, s):
    """Same problem with the degree of freedom 1 measured in other units,
    ``c1' = s c1``, e.g. a rotation: its force is ``F1' = F1/s`` and the
    stiffness matrices ``K' = S^-1 K S^-1``, ``S = diag(1, s)``"""
    Sinv = np.array([1., 1./s])
    return Problem(Sinv*p.fext,
                   lambda c: Sinv*p.fint(Sinv*c),
                   Sinv[:, None]*p.kC*Sinv[None, :],
                   lambda c: Sinv[:, None]*p.kG(Sinv*c)*Sinv[None, :])


def test_crisfield_criterion_units_invariance():
    r"""Measuring one degree of freedom in other units does not change
    Crisfield's `\zeta`, hence neither the number of iterations nor the
    increments of the Newton-Raphson method, whereas it changes the relative
    criterion, whose Euclidean norm adds the squares of forces of different
    units. The arc-length constraint, with the Euclidean norm of the
    displacements, is not invariant, see :func:`.arc_length._solver_arc_length`"""
    method, frac = 'NR', 0.99
    scales = [1., 1.e3, 1.e-3]
    results = {}
    for convergence in ('crisfield', 'relative'):
        for s in scales:
            p = rescaled_dof_problem(von_mises_truss(frac=frac), s)
            p.analysis(NL_method=method, initialInc=0.05,
                       convergence=convergence, relTOL=1e-9)
            p.an.static(NLgeom=True, silent=True)
            cs = np.asarray(p.an.cs)/np.array([1., s])
            results[convergence, s] = (np.asarray(p.an.increments), cs,
                                       len(p.fint_calls))
    inc0, cs0, calls0 = results['crisfield', 1.]
    assert inc0[-1] == 1.
    for s in scales[1:]:
        inc, cs, calls = results['crisfield', s]
        assert calls == calls0
        np.testing.assert_allclose(inc, inc0, rtol=1e-12)
        np.testing.assert_allclose(cs, cs0, rtol=1e-9, atol=1e-12)
    calls = [results['relative', s][2] for s in scales]
    assert len(set(calls)) > 1, calls


@pytest.mark.parametrize('method', METHODS + ['NR'])
def test_crisfield_criterion_free_dofs_only(method):
    r"""A support as a null row and column of the stiffness matrices, with
    the internal force vector of all the degrees of freedom, including the
    support reaction: Crisfield's criterion measures the free degrees of
    freedom only and gives the solution of the model without the support
    degree of freedom, whereas the reaction keeps the relative criterion
    from converging"""
    p = hardening_springs()
    n = p.kC.shape[0]
    # dof 0 is the support, connected to the dof 1 by the first spring
    K = np.zeros((n + 1, n + 1))
    K[1:, 1:] = p.kC

    def fint(c):
        f = np.zeros(n + 1)
        f[1:] = p.fint(c[1:])
        f[0] = -100.*c[1]
        return f

    def kG(c):
        G = np.zeros((n + 1, n + 1))
        G[1:, 1:] = p.kG(c[1:])
        return G

    q = Problem(np.concatenate(([0.], p.fext)), fint, K, kG)
    q.analysis(NL_method=method, initialInc=0.1, minInc=1e-3)
    q.an.static(NLgeom=True, silent=True)
    assert q.an.increments[-1] == 1.
    np.testing.assert_allclose(q.an.cs[-1][1:], p.solve_reference(),
                               rtol=1e-8)
    assert q.an.cs[-1][0] == 0.

    r = Problem(q.fext, fint, K, kG)
    r.analysis(NL_method=method, initialInc=0.1, minInc=1e-3,
               convergence='relative')
    r.an.static(NLgeom=True, silent=True)
    assert not r.an.increments or r.an.increments[-1] < 1.
