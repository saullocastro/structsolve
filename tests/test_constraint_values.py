"""Tests of the values of the linear constraints, ``Analysis.constraint_values``

The constraints `C^T c = lambda d` of ``Analysis.constraints`` and
``Analysis.constraint_values``, e.g. a displacement prescribed in proportion
to the load factor: the solutions of the load vector have the border `d`,
those of the residual a null border, see :func:`.static._factorize`.

"""
import pickle

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from analytic_problems import hardening_springs, Problem
from test_branch_switching import pitchforks
from structsolve import Analysis
from structsolve.static import _factorize
from structsolve.pardiso import available as pardiso_available

METHODS = ['NR', 'arc_length_riks', 'arc_length_crisfield',
           'displacement_control', 'generalized_displacement_control']


def has_pardiso():
    return pardiso_available()


SOLVERS = ['splu'] + (['pardiso'] if has_pardiso() else [])


def spring_chain(n=5, k=100.):
    """Chain of ``n`` springs of stiffness ``k``, the first one attached to
    the ground"""
    K = k*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[-1, -1] = k
    return K


@pytest.mark.parametrize('linear_solver', SOLVERS)
def test_factorize_border(linear_solver):
    """``solve_kT(b, d)`` satisfies ``C^T x = d``, ``solve_kT(b)`` the
    homogeneous constraints; a constraint dropped with the null rows and
    columns must have a null value"""
    rng = np.random.default_rng(0)
    n = 8
    K = np.zeros((n + 1, n + 1))
    K[1:, 1:] = spring_chain(n)
    C = np.zeros((n + 1, 3))
    C[[1, 4], 0] = [1., -1.]
    C[-1, 1] = 1.
    C[0, 2] = 1.
    f = _factorize(csc_matrix(K), linear_solver, constraints=C)
    assert f.num_constraints == 2
    b = rng.standard_normal(n + 1)
    b[0] = 0.
    d = np.array([0.3, -0.7, 0.])
    x = f(b, d)
    np.testing.assert_allclose(C.T @ x, d, atol=1e-12)
    np.testing.assert_allclose(C.T @ f(b), 0., atol=1e-12)
    assert np.array_equal(f(b, np.zeros(3)), f(b))
    # the same reactions balance the residual on the free directions
    mu = np.linalg.lstsq(C, b - K @ x, rcond=None)[0]
    np.testing.assert_allclose(K @ x + C @ mu, b, atol=1e-10)
    with pytest.raises(ValueError, match=r'\[2\]'):
        f(b, np.array([0., 0., 1.]))
    with pytest.raises(ValueError, match='shape'):
        f(b, np.zeros(2))
    g = _factorize(csc_matrix(K[1:, 1:]), linear_solver)
    with pytest.raises(ValueError, match='require'):
        g(b[1:], np.zeros(1))


@pytest.mark.parametrize('linear_solver', SOLVERS)
def test_linear_spring_chain(linear_solver):
    """Prescribed end displacement of a chain of springs: ``u_i = delta
    i/n``, the reaction ``k delta/n``, then combined with a load"""
    n, k, delta = 5, 100., 0.02
    K = spring_chain(n, k)
    C = np.zeros(n)
    C[-1] = 1.
    p = Problem(np.zeros(n), lambda c: K @ c, K, lambda c: 0*K)
    an = p.analysis(constraints=C, constraint_values=[delta],
                    linear_solver=linear_solver)
    an.static(silent=True)
    u = an.cs[0]
    np.testing.assert_allclose(u, delta*np.arange(1, n + 1)/n, rtol=1e-13)
    np.testing.assert_allclose(K @ u, np.r_[np.zeros(n - 1), k*delta/n],
                               atol=1e-12)
    # with a load P on the node 2: the superposition of the load on the
    # chain of 4 springs clamped at the end and of the prescribed motion
    P = 3.
    p.fext = np.zeros(n)
    p.fext[2] = P
    an = p.analysis(constraints=C, constraint_values=np.array([delta]))
    an.static(silent=True)
    u = an.cs[0]
    Kr = K[:-1, :-1]
    ur = np.linalg.solve(Kr, p.fext[:-1] - K[:-1, -1]*delta)
    np.testing.assert_allclose(u, np.r_[ur, delta], rtol=1e-12)


def reduced_reference(p, free, fixed, values, lbd=1.):
    """Dense Newton-Raphson with the degrees of freedom ``fixed`` at
    ``lbd*values``"""
    c = np.zeros(p.fext.size)
    c[fixed] = lbd*values
    for _ in range(100):
        R = (p.fint(c) - lbd*p.fext)[free]
        if np.linalg.norm(R) <= 1e-14*max(1., np.linalg.norm(p.fint(c))):
            return c
        Kt = (p.kC + p.kG(c))[np.ix_(free, free)]
        c[free] -= np.linalg.solve(Kt, R)
    raise RuntimeError('reference did not converge')


def kwargs_of(method, n, dof):
    if method == 'displacement_control':
        psi = np.zeros(n)
        psi[dof] = 1.
        return dict(control_vector=psi)
    return {}


@pytest.mark.parametrize('linear_solver', SOLVERS)
@pytest.mark.parametrize('method', METHODS)
def test_prescribed_displacement_only(method, linear_solver):
    """The hardening springs driven only by a prescribed displacement of
    their end, ``fext = 0``: the path against the dense solution of the
    free degrees of freedom, the constraint satisfied at every increment"""
    p = hardening_springs()
    p.fext = np.zeros(4)
    delta = 0.8
    C = np.zeros((4, 1))
    C[3, 0] = 1.
    an = p.analysis(NL_method=method, linear_solver=linear_solver,
                    constraints=C, constraint_values=[delta],
                    initialInc=0.1, **kwargs_of(method, 4, 3))
    an.static(NLgeom=True, silent=True)
    assert an.increments[-1] == 1.
    assert len(an.increments) > 3
    for lbd, c in zip(an.increments, an.cs):
        assert abs(c[3] - lbd*delta) <= 1e-12
        ref = reduced_reference(p, [0, 1, 2], [3], delta, lbd)
        np.testing.assert_allclose(c, ref, rtol=1e-7, atol=1e-12)
    # the reaction carries the load, far from linear
    lin = reduced_reference(Problem(p.fext, lambda c: p.kC @ c, p.kC,
                                    lambda c: 0*p.kC), [0, 1, 2], [3], delta)
    assert np.abs(an.cs[-1] - lin).max() > 0.05


@pytest.mark.parametrize('convergence', ['crisfield', 'relative'])
@pytest.mark.parametrize('method', METHODS)
def test_prescribed_displacement_and_load(method, convergence):
    """A load on the node 1 and a prescribed rotation-like relative
    displacement ``c2 - c0 = lbd d``, both proportional to the load factor"""
    p = hardening_springs(P=0.)
    p.fext = np.array([0., 30., 0., 0.])
    C = np.array([-1., 0., 1., 0.])
    d = 0.3
    an = p.analysis(NL_method=method, constraints=C, constraint_values=d,
                    convergence=convergence, initialInc=0.1,
                    **kwargs_of(method, 4, 2))
    an.static(NLgeom=True, silent=True)
    assert an.increments[-1] == 1.
    # the dense reference with c2 = c0 + d, c = T r + t
    T = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    t = np.array([0., 0., d, 0.])
    r = np.zeros(3)
    for _ in range(100):
        c = T @ r + t
        R = T.T @ (p.fint(c) - p.fext)
        r -= np.linalg.solve(T.T @ (p.kC + p.kG(c)) @ T, R)
    np.testing.assert_allclose(an.cs[-1], T @ r + t, rtol=1e-7)
    for lbd, c in zip(an.increments, an.cs):
        assert abs(C @ c - lbd*d) <= 1e-12


def test_modified_newton_raphson():
    p = hardening_springs()
    p.fext = np.zeros(4)
    C = np.zeros(4)
    C[3] = 1.
    an = p.analysis(constraints=C, constraint_values=[0.8], modified_NR=True,
                    kT_initial_state=False, initialInc=0.1)
    an.static(NLgeom=True, silent=True)
    assert an.increments[-1] == 1.
    np.testing.assert_allclose(an.cs[-1], reduced_reference(p, [0, 1, 2],
                                                            [3], 0.8),
                               rtol=1e-7)


@pytest.mark.parametrize('method', ['arc_length_riks', 'arc_length_crisfield'])
def test_branch_switching(method):
    """A column whose axial displacement is prescribed, ``u = -lbd delta``,
    without load: it buckles at ``lbd_cr = k/(EA delta)`` and the bifurcated
    branch is ``v**2 = (EA lbd delta - k)/(EA/2 + beta)``; the eigenvector
    predictor keeps the prescribed displacement"""
    p = pitchforks(betas=[0.5], ks=[1.], EA=10., P0=2.)
    p.fext = np.zeros(2)
    delta = 0.2
    an = p.analysis(NL_method=method, initialInc=0.1, inertia=True,
                    branch_switching=1, constraints=np.array([1., 0.]),
                    constraint_values=[-delta])
    an.static(NLgeom=True, silent=True)
    assert len(an.branch_points) == 1
    ib = an.branch_points[0]
    lbd_cr = 1./(10.*delta)
    assert an.increments[ib] > lbd_cr > an.increments[ib - 1]
    assert an.increments[-1] == 1.
    lbds = np.asarray(an.increments[ib + 1:])
    v = np.array([c[1] for c in an.cs[ib + 1:]])
    np.testing.assert_allclose(v**2, (10.*lbds*delta - 1.)/(5. + 0.5),
                               rtol=1e-6)
    for lbd, c in zip(an.increments, an.cs):
        assert abs(c[0] + lbd*delta) <= 1e-12


@pytest.mark.parametrize('method', METHODS)
def test_restart(method):
    p = hardening_springs()
    p.fext = np.zeros(4)
    C = np.zeros(4)
    C[3] = 1.
    kwargs = dict(NL_method=method, constraints=C, constraint_values=[0.8],
                  initialInc=0.05, **kwargs_of(method, 4, 3))
    p.analysis(**kwargs)
    p.an.static(NLgeom=True, silent=True)
    ref = list(p.an.increments)
    p.analysis(stop_criterion=lambda an: len(an.increments) == 4, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    first = list(p.an.increments)
    state = pickle.loads(pickle.dumps(p.an.path_state))
    p.analysis(restart_state=state, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    np.testing.assert_allclose(first + p.an.increments, ref, rtol=1e-12)
    for lbd, c in zip(p.an.increments, p.an.cs):
        assert abs(c[3] - lbd*0.8) <= 1e-12


def test_zero_values_unchanged():
    """Null values give the path of the homogeneous constraints, bitwise"""
    C = np.array([1., -1., 0., 0.])
    out = []
    for d in (None, [0.]):
        p = hardening_springs()
        an = p.analysis(NL_method='arc_length_riks', constraints=C,
                        constraint_values=d)
        an.static(NLgeom=True, silent=True)
        out.append(an)
    assert out[0].increments == out[1].increments
    for c0, c1 in zip(out[0].cs, out[1].cs):
        assert np.array_equal(c0, c1)


@pytest.mark.parametrize('NLgeom', [False, True])
def test_validation(NLgeom):
    p = hardening_springs()
    an = p.analysis(constraint_values=[1.])
    with pytest.raises(ValueError, match='requires Analysis.constraints'):
        an.static(NLgeom=NLgeom, silent=True)
    an = p.analysis(constraints=np.eye(4)[:, :2], constraint_values=[1.])
    with pytest.raises(ValueError, match=r'shape \(2,\)'):
        an.static(NLgeom=NLgeom, silent=True)
    an = p.analysis(constraints=np.eye(4)[:, :2],
                    constraint_values=[1., np.nan])
    with pytest.raises(ValueError, match='finite'):
        an.static(NLgeom=NLgeom, silent=True)
    assert Analysis().constraint_values is None
