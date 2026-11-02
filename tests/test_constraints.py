"""Tests of the linear constraints of the solution, ``Analysis.constraints``

The constraints `C^T c = 0` border every factorization of the stiffness
matrices, see :func:`.static._factorize`, and the convergence criteria
measure the residual without the reactions of the constraints, see
:func:`.newton_raphson._residual_projector`.

"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix

from analytic_problems import hardening_springs, Problem, convergence_orders
from structsolve.static import _factorize

METHODS = ['NR', 'arc_length_riks', 'arc_length_crisfield',
           'displacement_control']


def has_pypardiso():
    try:
        import pypardiso  # noqa: F401
    except ImportError:
        return False
    return True


def kwargs_of(method, n):
    if method == 'displacement_control':
        psi = np.zeros(n)
        psi[-1] = 1.
        return dict(control_vector=psi)
    return {}


def reduced_reference(p, T):
    """Dense Newton-Raphson on the reduced problem ``c = T r``"""
    r = np.linalg.solve(T.T @ p.kC @ T, T.T @ p.fext)
    for _ in range(100):
        R = T.T @ (p.fint(T @ r) - p.fext)
        if np.linalg.norm(R) <= 1e-15*np.linalg.norm(T.T @ p.fext):
            break
        r = r - np.linalg.solve(T.T @ (p.kC + p.kG(T @ r)) @ T, R)
    return T @ r


@pytest.mark.parametrize('form', ['1d', '2d', 'sparse'])
@pytest.mark.parametrize('method', METHODS)
def test_multipoint_constraint(method, form):
    """``c0 = c1`` on the hardening springs, against the elimination of the
    constraint, ``c = T r``; the reaction of the constraint is not null"""
    p = hardening_springs()
    C = np.array([1., -1., 0., 0.])
    Cin = {'1d': C, '2d': C[:, None], 'sparse': csc_matrix(C[:, None])}[form]
    p.analysis(NL_method=method, initialInc=0.1, constraints=Cin,
               **kwargs_of(method, 4))
    p.an.static(NLgeom=True, silent=True)
    assert p.an.increments[-1] == 1.
    for c in p.an.cs:
        assert abs(C @ c) < 1e-14
    T = np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    ref = reduced_reference(p, T)
    np.testing.assert_allclose(p.an.cs[-1], ref, rtol=1e-7)
    # the reaction of the constraint
    R = p.fint(ref) - p.fext
    mu = -R @ C/(C @ C)
    np.testing.assert_allclose(R + mu*C, 0., atol=1e-9*np.abs(p.fext).max())
    assert abs(mu) > 1e-3*np.abs(p.fext).max()


def free_floating_springs(P=20., n=4):
    """Chain of ``n`` nodes and springs with cubic hardening between them,
    without support, loaded by ``-P`` and ``+P`` at its ends: the stiffness
    matrix is singular, its null space the rigid translation"""
    k = 100.
    a = 50.

    def fint(u):
        d = np.diff(u)
        t = k*d + a*d**3
        f = np.zeros(n)
        f[:-1] -= t
        f[1:] += t
        return f

    def kG(u):
        d = np.diff(u)
        g = 3*a*d**2
        G = np.zeros((n, n))
        for i, gi in enumerate(g):
            G[i:i + 2, i:i + 2] += gi*np.array([[1., -1.], [-1., 1.]])
        return G

    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i:i + 2, i:i + 2] += k*np.array([[1., -1.], [-1., 1.]])
    fext = np.zeros(n)
    fext[0], fext[-1] = -P, P
    return Problem(fext, fint, K, kG), k, a


@pytest.mark.parametrize('linear_solver', ['splu'] + (['pardiso'] if
                                                       has_pypardiso() else []))
@pytest.mark.parametrize('method', METHODS)
def test_inertia_relief(method, linear_solver):
    """A structure without support under a self-equilibrated load: the
    constraint of a null mean displacement removes the rigid translation,
    the reaction is null; every spring stretches by ``d``, ``k d + a d^3 =
    P``"""
    P = 20.
    p, k, a = free_floating_springs(P)
    n = 4
    p.analysis(NL_method=method, initialInc=0.1, constraints=np.ones(n),
               linear_solver=linear_solver, inertia=True,
               **kwargs_of(method, n))
    p.an.static(NLgeom=True, silent=True)
    assert p.an.increments[-1] == 1.
    u = p.an.cs[-1]
    d = np.roots([a, 0., k, -P]).real[np.abs(np.roots([a, 0., k, -P]).imag) < 1e-12][0]
    np.testing.assert_allclose(np.diff(u), d, rtol=1e-8)
    assert abs(u.sum()) < 1e-12
    # stable on the space of the constraint
    assert set(p.an.det_signs) == {1}
    if linear_solver == 'pardiso':
        assert set(p.an.num_negative_eigvals) == {0}
    # without the constraint the stiffness matrix is singular: no increment
    # converges, or the linear solution of the controlled quantity is null
    q, _, _ = free_floating_springs(P)
    q.analysis(NL_method=method, initialInc=0.1, **kwargs_of(method, n))
    with pytest.warns(Warning, match='singular'):
        if method == 'displacement_control':
            with pytest.raises(ValueError, match='control_scale'):
                q.an.static(NLgeom=True, silent=True)
        else:
            q.an.static(NLgeom=True, silent=True)
            assert q.an.increments == []


def test_quadratic_convergence():
    """Newton-Raphson with the bordered tangent stays quadratic"""
    p = hardening_springs()
    C = np.array([1., -1., 0., 0.])
    p.analysis(NL_method='NR', initialInc=1., constraints=C, relTOL=1e-13)
    p.an.static(NLgeom=True, silent=True)
    cs = [c for _, c in p.fint_calls]
    ref = p.an.cs[-1]
    errors = [np.linalg.norm(c - ref)/np.linalg.norm(ref) for c in cs[:-1]]
    orders = convergence_orders(errors)
    assert len(orders) >= 1, errors
    assert min(orders) > 1.6, (orders, errors)


def test_linear_static():
    p = hardening_springs()
    C = np.array([1., -1., 0., 0.])
    an = p.analysis(constraints=C)
    an.static(silent=True)
    c = an.cs[0]
    T = np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    ref = T @ np.linalg.solve(T.T @ p.kC @ T, T.T @ p.fext)
    np.testing.assert_allclose(c, ref, rtol=1e-12)


@pytest.mark.parametrize('linear_solver', ['splu'] + (['pardiso'] if
                                                       has_pypardiso() else []))
def test_bordered_factorization(linear_solver):
    """Solution on the space of the constraints, with null rows and
    columns and a constraint on them only, which is dropped; the inertia
    of the tangent on the space of the constraints"""
    from scipy.linalg import null_space
    rng = np.random.default_rng(1)
    n = 30
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    K = Q @ np.diag(rng.standard_normal(n)) @ Q.T
    K = (K + K.T)/2
    Kf = np.zeros((n + 2, n + 2))
    Kf[2:, 2:] = K
    C = np.zeros((n + 2, 3))
    C[2:, :2] = rng.standard_normal((n, 2))
    C[0, 2] = 1.
    Z = null_space(C[2:, :2].T)
    KZ = Z.T @ K @ Z
    b = rng.standard_normal(n + 2)
    b[:2] = 0.
    f = _factorize(csc_matrix(Kf), linear_solver, constraints=C)
    x = f(b)
    assert f.num_constraints == 2
    assert np.abs(C.T @ x).max() < 1e-12
    r = Kf @ x - b
    mu = np.linalg.lstsq(C[:, :2], -r, rcond=None)[0]
    assert np.abs(r + C[:, :2] @ mu).max() < 1e-10
    assert f.det_sign == np.linalg.slogdet(KZ)[0]
    if linear_solver == 'pardiso':
        assert f.num_negative_eigvals == np.count_nonzero(
            np.linalg.eigvalsh(KZ) < 0)
