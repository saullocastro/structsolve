"""Tests of the linear solvers and of the inertia of the tangent stiffness
matrix along the path

``Analysis.linear_solver`` and ``Analysis.inertia``, see
:func:`.static._factorize` and :func:`.newton_raphson._converged_state`.
The tests of ``'pardiso'`` need ``pypardiso``.

"""
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import structsolve.static as static
from structsolve.static import _factorize, _permutation_sign
from analytic_problems import von_mises_truss, linear_problem, Problem
from test_arc_length import check_equilibrium

METHODS = ['arc_length_riks', 'arc_length_crisfield']


def has_pypardiso():
    try:
        import pypardiso  # noqa: F401
    except ImportError:
        return False
    return True


needs_pardiso = pytest.mark.skipif(not has_pypardiso(),
                                   reason='pypardiso is not installed')


def symmetric_indefinite(n, num_null, seed, spread=3.):
    """Symmetric indefinite matrix of size ``n + num_null``, the first
    ``num_null`` rows and columns null, its diagonal scaled over
    ``10**+-spread``, and its number of negative eigenvalues"""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = rng.standard_normal(n)
    ev[np.abs(ev) < 0.05] = 0.3
    A = Q @ np.diag(ev) @ Q.T
    S = np.diag(10.**rng.uniform(-spread, spread, n))
    A = S @ ((A + A.T)/2) @ S
    full = np.zeros((n + num_null, n + num_null))
    full[num_null:, num_null:] = A
    return full, int(np.count_nonzero(ev < 0)), int(np.linalg.slogdet(A)[0])


def test_permutation_sign():
    assert _permutation_sign([0, 1, 2]) == 1
    assert _permutation_sign([1, 0, 2]) == -1
    assert _permutation_sign([1, 2, 0]) == 1
    assert _permutation_sign([3, 2, 1, 0]) == 1


@pytest.mark.parametrize('seed', range(5))
def test_splu_determinant_sign(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((30, 30))
    f = _factorize(csr_matrix(A))
    assert f.linear_solver == 'splu'
    assert f.det_sign == np.linalg.slogdet(A)[0]
    assert f.num_negative_eigvals is None
    assert f.num_perturbed_pivots is None
    A, nneg, det = symmetric_indefinite(40, 3, seed)
    f = _factorize(csr_matrix(A))
    assert f.det_sign == det == (-1)**nneg


@needs_pardiso
@pytest.mark.parametrize('seed', range(5))
def test_pardiso_inertia(seed):
    """Number of negative eigenvalues, with null rows and columns and a
    diagonal over 6 orders of magnitude"""
    A, nneg, det = symmetric_indefinite(60, 3, seed)
    f = _factorize(csr_matrix(A), linear_solver='pardiso')
    assert f.linear_solver == 'pardiso'
    assert f.num_negative_eigvals == nneg
    assert f.num_perturbed_pivots == 0
    assert f.det_sign == det
    b = np.random.default_rng(seed).standard_normal(A.shape[0])
    b[:3] = 0.
    x = f(b)
    assert np.all(x[:3] == 0.)
    np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-8*np.abs(b).max())
    assert np.all(f(np.zeros_like(b)) == 0.)


@needs_pardiso
def test_pardiso_gmres_refinement(monkeypatch):
    """With a refinement tolerance below any residual, every solution is
    improved by GMRES, preconditioned by the factorization"""
    monkeypatch.setattr(static, 'PARDISO_REFINE_TOL', 0.)
    A, _, _ = symmetric_indefinite(60, 0, 1)
    f = _factorize(csr_matrix(A), linear_solver='pardiso')
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        x = f(b)
    np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-8*np.abs(b).max())


@needs_pardiso
def test_pardiso_unsymmetric_falls_back_to_splu():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 20)) + 20*np.eye(20)
    with pytest.warns(RuntimeWarning, match='unsymmetric'):
        f = _factorize(csr_matrix(A), linear_solver='pardiso')
    assert f.linear_solver == 'splu'
    b = rng.standard_normal(20)
    np.testing.assert_allclose(f(b), np.linalg.solve(A, b))


def test_invalid_linear_solver():
    with pytest.raises(ValueError, match='linear_solver'):
        _factorize(csr_matrix(np.eye(2)), linear_solver='mumps')


def truss_det_sign(c):
    r"""Sign of the determinant of the tangent stiffness matrix of the von
    Mises truss with a spring, `k_s (3 w^2 - 6 w + 2)`, negative between
    the two limit points of the load"""
    w = c[0]
    return int(np.sign(3*w**2 - 6*w + 2))


def solvers():
    return ['splu'] + (['pardiso'] if has_pypardiso() else [])


@pytest.mark.parametrize('linear_solver', solvers())
@pytest.mark.parametrize('method', METHODS + ['displacement_control'])
def test_inertia_along_snap_back(method, linear_solver):
    """Snap-through of the von Mises truss: one negative eigenvalue between
    the two limit points of the load, none elsewhere"""
    p = von_mises_truss(frac=1.5, ks=0.5)
    kwargs = {}
    if method == 'displacement_control':
        kwargs['control_vector'] = np.array([1., 0.])
    p.analysis(NL_method=method, initialInc=0.05, inertia=True,
               linear_solver=linear_solver, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    an = p.an
    assert an.increments[-1] == 1.
    n = len(an.increments)
    assert len(an.det_signs) == len(an.num_negative_eigvals) == n
    assert len(an.num_perturbed_pivots) == n
    expected = [truss_det_sign(c) for c in an.cs]
    assert an.det_signs == expected
    assert -1 in expected and 1 in expected
    if linear_solver == 'pardiso':
        assert an.num_negative_eigvals == [(1 - s)//2 for s in expected]
        assert an.num_perturbed_pivots == [0]*n
    else:
        assert an.num_negative_eigvals == [None]*n


@pytest.mark.parametrize('linear_solver', solvers())
def test_inertia_newton_raphson(linear_solver):
    p = von_mises_truss(frac=0.99, ks=0.5)
    p.analysis(NL_method='NR', initialInc=0.1, inertia=True,
               linear_solver=linear_solver)
    p.an.static(NLgeom=True, silent=True)
    assert p.an.increments[-1] == 1.
    assert p.an.det_signs == [1]*len(p.an.increments)


@pytest.mark.parametrize('method', METHODS)
def test_inertia_same_path(method):
    """The factorization at the converged state is the one the next step
    would compute: the same path and evaluations of ``calc_fint`` with and
    without ``inertia``, and with ``'pardiso'``"""
    results = []
    for inertia, linear_solver in [(False, 'splu'), (True, 'splu')] + (
            [(True, 'pardiso')] if has_pypardiso() else []):
        p = von_mises_truss(frac=1.5, ks=0.5)
        p.analysis(NL_method=method, initialInc=0.05, inertia=inertia,
                   linear_solver=linear_solver)
        p.an.static(NLgeom=True, silent=True)
        results.append((np.asarray(p.an.increments), np.asarray(p.an.cs),
                        len(p.fint_calls)))
    for inc, cs, calls in results[1:]:
        assert calls == results[0][2]
        np.testing.assert_allclose(inc, results[0][0], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(cs, results[0][1], rtol=1e-8, atol=1e-12)


def test_defaults():
    p = linear_problem()
    an = p.analysis(NL_method='arc_length_riks')
    assert an.linear_solver == 'splu'
    assert an.inertia is False
    an.static(NLgeom=True, silent=True)
    assert an.num_negative_eigvals == []
    assert an.det_signs == []


@pytest.mark.parametrize('linear_solver', solvers())
def test_unsymmetric_tangent(linear_solver):
    """A follower-like load stiffness makes the tangent unsymmetric:
    ``'pardiso'`` falls back to ``'splu'``, and the count is ``None``"""
    K = np.array([[4., -1.], [0.5, 3.]])
    p = Problem(np.array([1., 2.]), lambda c: K @ c, K,
                lambda c: np.zeros_like(K))
    p.analysis(NL_method='arc_length_riks', inertia=True,
               linear_solver=linear_solver)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        p.an.static(NLgeom=True, silent=True)
    assert p.an.increments[-1] == 1.
    np.testing.assert_allclose(p.an.cs[-1], np.linalg.solve(K, p.fext))
    assert set(p.an.det_signs) == {1}
    assert p.an.num_negative_eigvals == [None]*len(p.an.increments)
    unsym = [x for x in w if 'unsymmetric' in str(x.message)]
    assert bool(unsym) == (linear_solver == 'pardiso')
