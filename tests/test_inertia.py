"""Tests of the linear solvers and of the inertia of the tangent stiffness
matrix along the path

``Analysis.linear_solver`` and ``Analysis.inertia``, see
:func:`.static._factorize` and :func:`.newton_raphson._converged_state`.
The tests of ``'pardiso'`` need oneMKL, see :mod:`structsolve.pardiso`.

"""
import importlib
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from structsolve.static import _factorize, _permutation_sign
from analytic_problems import von_mises_truss, linear_problem, Problem
from test_arc_length import check_equilibrium
from structsolve.pardiso import available as pardiso_available

METHODS = ['arc_length_riks', 'arc_length_crisfield']


def has_pardiso():
    return pardiso_available()


#NOTE structsolve.static is the function static(), exported by the package
static_module = importlib.import_module('structsolve.static')

needs_pardiso = pytest.mark.skipif(not has_pardiso(),
                                   reason='oneMKL is not found')


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
    monkeypatch.setattr(static_module, 'PARDISO_REFINE_TOL', 0.)
    A, _, _ = symmetric_indefinite(60, 0, 1)
    f = _factorize(csr_matrix(A), linear_solver='pardiso')
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        x = f(b)
    np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-8*np.abs(b).max())



@needs_pardiso
def test_pardiso_gmres_warning(monkeypatch):
    """A solution whose residual remains above ``PARDISO_WARN_TOL`` after
    GMRES is returned with a RuntimeWarning"""
    monkeypatch.setattr(static_module, 'PARDISO_REFINE_TOL', 0.)
    monkeypatch.setattr(static_module, 'PARDISO_WARN_TOL', 0.)
    A, _, _ = symmetric_indefinite(20, 0, 1)
    f = _factorize(csr_matrix(A), linear_solver='pardiso')
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    with pytest.warns(RuntimeWarning, match='after GMRES'):
        x = f(b)
    np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-8*np.abs(b).max())


@needs_pardiso
@pytest.mark.parametrize('with_cache', [False, True])
def test_pardiso_null_matrix(with_cache):
    """A matrix without any non-zero term: 'splu' solves the empty reduced
    system, the solutions are null, whereas PARDISO fails, which is
    reported as a singular matrix, a RuntimeError"""
    from structsolve.static import _FactorizationCache
    Z = csr_matrix((3, 3))
    cache = _FactorizationCache() if with_cache else None
    np.testing.assert_array_equal(_factorize(Z, cache=cache)(np.ones(3)),
                                  np.zeros(3))
    with pytest.raises(RuntimeError, match='PARDISO factorization failed'):
        _factorize(Z, linear_solver='pardiso', cache=cache)
    if with_cache:
        cache.close()


def test_singular_converged_state(capsys):
    """A converged state whose tangent stiffness matrix is singular has no
    factorization: its inertia is recorded as not known, with the sign 0
    of the determinant"""
    from structsolve.newton_raphson import _converged_state
    K = np.array([[1., 1.], [1., 1.]])
    p = Problem(np.array([1., 1.]), lambda c: K @ c, K,
                lambda c: np.zeros_like(K))
    an = p.analysis(inertia=True)
    # as at the start of Analysis.static
    an.num_negative_eigvals, an.det_signs, an.num_perturbed_pivots = [], [], []
    assert _converged_state(an, np.zeros(2), 0.5) is None
    assert 'Singular tangent stiffness matrix at the converged state'         in capsys.readouterr().out
    assert an.num_negative_eigvals == [None]
    assert an.det_signs == [0]
    assert an.num_perturbed_pivots == [None]


def dense_indefinite(n, nneg, seed=0):
    """Dense symmetric matrix with ``nneg`` negative eigenvalues, the others
    from 1e-4 to 1, in a random orthonormal basis"""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.r_[-np.logspace(-3, -2, nneg), np.logspace(-4, 0, n - nneg)]
    A = (Q*ev) @ Q.T
    return 0.5*(A + A.T)


@needs_pardiso
@pytest.mark.parametrize('with_cache', [False, True])
def test_pardiso_fallback_dense_indefinite(with_cache):
    """A dense indefinite matrix of 114 degrees of freedom, whose
    factorization with the default parameters of PARDISO perturbs 34
    pivots with MKL 2026.1, has then a wrong inertia (0 negative eigenvalues
    instead of 1) and solutions of relative residual about 1; the second
    factorization with ``PARDISO_FALLBACK_IPARM`` is exact. MKL 2026.2
    perturbs no pivot"""
    from scipy.sparse import triu
    from structsolve.pardiso import PardisoSolver
    from structsolve.static import _FactorizationCache
    A = dense_indefinite(114, 1)
    d = 1/np.sqrt(np.abs(A.diagonal()))
    As = A*d[:, None]*d[None, :]
    solver = PardisoSolver()
    solver.factorize(triu(csr_matrix(As), format='csr'))
    default_perturbed = int(solver.iparm[13])
    solver.free(everything=True)
    cache = _FactorizationCache() if with_cache else None
    b = np.random.default_rng(1).standard_normal(A.shape[0])
    for _ in range(2):
        f = _factorize(csr_matrix(A), linear_solver='pardiso', cache=cache)
        assert f.num_negative_eigvals == 1
        assert f.num_perturbed_pivots == 0
        assert f.pardiso_fallback == (default_perturbed > 0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            x = f(b)
        np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-9*np.abs(b).max())
    if cache is not None:
        cache.close()


def fake_perturbed_pivots(monkeypatch, default, fallback, fallback_fails=False):
    """``PardisoSolver.factorize`` reporting ``default`` perturbed pivots
    with the default parameters and ``fallback`` with those of
    ``PARDISO_FALLBACK_IPARM``, which raises when ``fallback_fails``; returns
    the list of the factorizations, ``(phase, is_fallback)``"""
    from structsolve.pardiso import PardisoSolver, PardisoError
    orig = PardisoSolver.factorize
    calls = []

    def factorize(self, A, phase=12):
        is_fallback = bool(self.iparm[0]) and self.iparm[23] == 1
        calls.append((phase, is_fallback))
        if is_fallback and fallback_fails:
            raise PardisoError(phase, -4)
        orig(self, A, phase)
        self.iparm[13] = fallback if is_fallback else default

    monkeypatch.setattr(PardisoSolver, 'factorize', factorize)
    return calls


@needs_pardiso
@pytest.mark.parametrize('with_cache', [False, True])
def test_pardiso_fallback_used(monkeypatch, with_cache):
    """The fallback, with fewer perturbed pivots, is kept; with a cache, the
    next factorizations of the pattern use its parameters directly"""
    from structsolve.static import _FactorizationCache
    calls = fake_perturbed_pivots(monkeypatch, 34, 0)
    A = csr_matrix(dense_indefinite(30, 1))
    b = np.random.default_rng(1).standard_normal(A.shape[0])
    cache = _FactorizationCache() if with_cache else None
    for _ in range(2):
        f = _factorize(A, linear_solver='pardiso', cache=cache)
        assert f.pardiso_fallback
        assert f.num_perturbed_pivots == 0
        assert f.num_negative_eigvals == 1
        np.testing.assert_allclose(A @ f(b), b, rtol=0, atol=1e-9*np.abs(b).max())
    if with_cache:
        assert calls == [(12, False), (12, True), (12, True)]
        cache.close()
    else:
        assert calls == [(12, False), (12, True)]*2


@needs_pardiso
@pytest.mark.parametrize('with_cache', [False, True])
@pytest.mark.parametrize('fallback_fails', [False, True])
def test_pardiso_fallback_discarded(monkeypatch, fallback_fails, with_cache):
    """When the fallback fails or still perturbs pivots, the matrix is
    factorized by SuperLU, with a warning, and the count of negative
    eigenvalues is unknown; with a cache, the next factorizations of the
    pattern use SuperLU directly"""
    from structsolve.static import _FactorizationCache
    calls = fake_perturbed_pivots(monkeypatch, 3, 3, fallback_fails)
    A = csr_matrix(dense_indefinite(30, 1))
    b = np.ones(A.shape[0])
    cache = _FactorizationCache() if with_cache else None
    with pytest.warns(RuntimeWarning, match='PARDISO_FALLBACK_IPARM'):
        f = _factorize(A, linear_solver='pardiso', cache=cache)
    assert calls == [(12, False), (12, True)]
    assert f.linear_solver == 'splu'
    assert f.num_negative_eigvals is None
    assert f.det_sign == -1
    np.testing.assert_allclose(A @ f(b), b, rtol=0, atol=1e-9)
    if with_cache:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            f = _factorize(A, linear_solver='pardiso', cache=cache)
        assert len(calls) == 2
        assert f.linear_solver == 'splu'
        cache.close()


@needs_pardiso
def test_pardiso_inaccurate_fallback(monkeypatch):
    """A fallback without perturbed pivots but with a residual above
    PARDISO_ACCURACY_RTOL is not used, e.g. the stiffness matrix of
    tests/data/cylinder_5panels with MKL 2025.0 (residual 1.3e3)"""
    static_mod = importlib.import_module('structsolve.static')
    calls = fake_perturbed_pivots(monkeypatch, 3, 0)
    monkeypatch.setattr(static_mod, 'PARDISO_ACCURACY_RTOL', 0.)
    A = csr_matrix(dense_indefinite(30, 1))
    with pytest.warns(RuntimeWarning, match='inaccurate'):
        f = _factorize(A, linear_solver='pardiso')
    assert calls == [(12, False), (12, True)]
    assert f.linear_solver == 'splu'
    assert f.num_negative_eigvals is None


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
    return ['splu'] + (['pardiso'] if has_pardiso() else [])


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
            [(True, 'pardiso')] if has_pardiso() else []):
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
