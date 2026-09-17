"""Tests for structsolve.newton_raphson using analytic problems

All problems have an exact tangent stiffness matrix ``kT = kC + kG`` that is
the Jacobian of ``fint``, such that full Newton-Raphson must converge
quadratically near the solution.

"""
import numpy as np

from analytic_problems import (hardening_springs, von_mises_truss,
                               linear_problem, convergence_orders, Problem)


def check_increments_and_equilibrium(p, rtol=1e-5):
    an = p.an
    inc = np.asarray(an.increments)
    assert len(an.increments) == len(an.cs)
    assert len(inc) >= 1
    assert inc[-1] == 1.0
    assert np.all(np.diff(inc) > 0)
    assert np.all(inc > 0)
    for lbd, c in zip(an.increments, an.cs):
        R = lbd*p.fext - p.fint(c)
        assert np.linalg.norm(R) <= rtol*np.linalg.norm(lbd*p.fext)


def test_nr_full_load_quadratic_convergence():
    p = hardening_springs()
    p.analysis(initialInc=1.)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    assert len(p.an.increments) == 1

    ref = p.solve_reference()
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref)
              for _, c in p.fint_calls]
    orders = convergence_orders(errors)
    assert len(orders) >= 1, errors
    assert min(orders) > 1.6, (orders, errors)
    # plain Newton from the linear predictor needs 9 residual evaluations to
    # reach machine precision; relTOL requires at most that
    assert len(p.fint_calls) <= 9


def test_nr_line_search_does_not_spoil_quadratic_convergence():
    p0 = hardening_springs()
    p0.analysis(initialInc=1., line_search=False)
    p0.an.static(NLgeom=True, silent=True)

    p = hardening_springs()
    p.analysis(initialInc=1., line_search=True)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)

    ref = p.solve_reference()
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref)
              for _, c in p.fint_calls]
    orders = convergence_orders(errors)
    assert len(orders) >= 1, errors
    assert min(orders) > 1.6, (orders, errors)
    # the Newton step is accepted as is, without extra fint evaluations
    assert len(p.fint_calls) == len(p0.fint_calls)


def test_nr_incremental_hardening():
    p = hardening_springs()
    p.analysis(initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    # a good predictor with an exact tangent needs few iterations per step
    assert max(p.calls_per_step()) <= 5, p.calls_per_step()


def test_nr_modified_NR_option():
    p = hardening_springs()
    p.analysis(initialInc=0.1, modified_NR=True, compute_every_n=3)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)


def test_nr_load_factor_bookkeeping():
    """Load steps must grow at most by the growth factor and end at 1.0"""
    for initialInc in [0.1, 0.25, 0.3, 0.5, 0.7, 1.]:
        p = linear_problem()
        p.analysis(initialInc=initialInc)
        p.an.static(NLgeom=True, silent=True)
        check_increments_and_equilibrium(p, rtol=1e-12)
        inc = np.asarray(p.an.increments)
        assert np.isclose(inc[0], initialInc)
        steps = np.diff(np.concatenate(([0.], inc)))
        assert np.all(steps[1:] <= 1.1111*steps[:-1] + 1e-12), inc


def test_nr_predictor_scaled_with_increment():
    """On a linear problem a scaled predictor is the exact solution

    Hence every step must converge at the first residual evaluation.

    """
    p = linear_problem()
    p.analysis(initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p, rtol=1e-12)
    assert len(p.an.increments) > 1
    assert p.calls_per_step() == [1]*len(p.an.increments)


def test_nr_softening_with_step_cutting():
    p = von_mises_truss(frac=0.99)
    # plain Newton needs 5 corrections to reach relTOL=1e-6 at full load, a
    # budget of 4 corrections forces the solver to cut the step
    p.analysis(initialInc=1., maxNumIter=4)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    assert len(p.an.increments) >= 2
    ref = p.solve_reference(c0=p.an.cs[-1])
    np.testing.assert_allclose(p.an.cs[-1], ref, rtol=1e-6)
    # the converged apex deflection is before the limit point
    assert p.an.cs[-1][0] < 1 - 1/np.sqrt(3)


def test_nr_softening_incremental():
    p = von_mises_truss(frac=0.99)
    p.analysis(initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    assert p.an.cs[-1][0] < 1 - 1/np.sqrt(3)


def test_nr_non_monotonic_residual_is_not_divergence():
    """Newton converges although the residual increases at iteration 5

    ``fint = u + 0.8 sin(u)`` has a positive tangent and a unique solution,
    but the Newton iterates oscillate before locking in, similarly to what
    happens in deep postbuckling.

    """
    b = 0.8
    p = Problem(np.array([8.]), lambda u: u + b*np.sin(u),
                np.array([[1. + b]]), lambda u: np.array([[b*np.cos(u[0]) - b]]))
    p.analysis(initialInc=1.)
    p.an.static(NLgeom=True, silent=True)
    check_increments_and_equilibrium(p)
    # solved in a single step, without step cutting
    assert len(p.an.increments) == 1
    assert len(p.fint_calls) <= 9
    # the residual did increase after the third iteration
    Rs = [abs(8. - p.fint(c)[0]) for _, c in p.fint_calls]
    assert any(Rs[i] > Rs[i-1] for i in range(3, len(Rs)))
