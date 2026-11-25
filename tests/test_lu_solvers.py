"""Tests of the LU factorizations ``linear_solver='pardiso_lu'`` and
``'dense_lu'`` of :func:`.static._factorize`

Both are checked against the sparse LU factorization of SuperLU,
``'splu'``: the solutions to round-off, the sign of the determinant where
available, the constraints, and the increments of complete non-linear
analyses. For ``'pardiso_lu'``: the fallbacks of an inaccurate
factorization or solution, forced by monkeypatching, the analysis of
PARDISO kept per pattern by the cache, and its behaviour without oneMKL.

"""
import importlib
import warnings
from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from structsolve.static import _factorize, _FactorizationCache
from structsolve.pardiso import available as pardiso_available
from analytic_problems import Problem, von_mises_truss

#NOTE structsolve.static is the function static(), exported by the package
static_module = importlib.import_module('structsolve.static')
linear_buckling_module = importlib.import_module('structsolve.linear_buckling')

needs_pardiso = pytest.mark.skipif(not pardiso_available(),
                                   reason='oneMKL is not found')


def solvers():
    return ['dense_lu'] + (['pardiso_lu'] if pardiso_available() else [])


def unsymmetric_matrix(n=120, seed=0):
    """Sparse unsymmetric indefinite matrix, well conditioned, with a null
    row and column (degree of freedom 3) removed by the factorizations"""
    rng = np.random.default_rng(seed)
    R = sp.random(n, n, density=0.05, random_state=seed, format='csr')
    A = R - 0.4*R.T + sp.diags(rng.choice([-1., 1.], n)*(2 + rng.random(n)))
    A = A.tolil()
    A[3, :] = 0.
    A[:, 3] = 0.
    return csr_matrix(A)


def rhs(n, seed=1):
    b = np.random.default_rng(seed).standard_normal(n)
    b[3] = 0.
    return b


@pytest.mark.parametrize('linear_solver', solvers())
def test_against_splu(linear_solver):
    """The solution of an unsymmetric sparse matrix and the sign of its
    determinant ('dense_lu') are those of SuperLU, to round-off; the null
    row and column give a null value of the solution"""
    A = unsymmetric_matrix()
    b = rhs(A.shape[0])
    ref = _factorize(A, 'splu')
    f = _factorize(A, linear_solver)
    x = f(b)
    assert f.linear_solver == linear_solver
    assert x[3] == 0.
    np.testing.assert_allclose(x, ref(b), rtol=0, atol=1e-12*np.abs(
        ref(b)).max())
    assert f.num_negative_eigvals is None
    assert f.num_constraints == 0
    if linear_solver == 'dense_lu':
        assert f.det_sign == ref.det_sign
        assert f.num_perturbed_pivots is None
        # the eigenvalues of smallest magnitude of SuperLU's
        np.testing.assert_allclose(np.sort_complex(f.smallest_eigvals(4)),
                                   np.sort_complex(ref.smallest_eigvals(4)),
                                   rtol=1e-8)
    else:
        assert f.det_sign is None
        assert f.num_perturbed_pivots == 0
        assert not f.pardiso_fallback


def test_dense_lu_det_sign():
    """The sign of the determinant of 'dense_lu' is that of the dense
    determinant of numpy, for matrices of both signs, also with row
    interchanges by the pivoting"""
    rng = np.random.default_rng(3)
    for k in range(10):
        M = rng.standard_normal((12, 12))
        f = _factorize(csr_matrix(M), 'dense_lu')
        assert f.det_sign == np.sign(np.linalg.det(M))


@pytest.mark.parametrize('linear_solver', ['dense_lu', 'pardiso_lu'])
def test_singular(linear_solver):
    """A singular matrix raises a RuntimeError, as SuperLU; for
    'pardiso_lu' through its fallback to SuperLU, whose warning is issued"""
    if linear_solver == 'pardiso_lu' and not pardiso_available():
        pytest.skip('oneMKL is not found')
    A = csr_matrix(np.array([[1., 2., 0.], [2., 4., 0.], [0., 0., 1.]]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(RuntimeError):
            _factorize(A, linear_solver)


@pytest.mark.parametrize('linear_solver', solvers())
def test_constraints(linear_solver):
    """With constraints, the solution of the bordered system, also with
    values of the constraints, and the sign of the determinant on the
    space of the constraints ('dense_lu') are those of SuperLU"""
    A = unsymmetric_matrix()
    n = A.shape[0]
    b = rhs(n)
    C = np.zeros((n, 3))
    C[0, 0] = 1.
    C[5, 1] = 1.
    C[9, 1] = -1.
    C[3, 2] = 1.  # on the null degree of freedom: dropped
    d = np.array([0.1, -0.2, 0.])
    ref = _factorize(A, 'splu', constraints=C)
    f = _factorize(A, linear_solver, constraints=C)
    assert f.num_constraints == ref.num_constraints == 2
    for dd in (None, d):
        x = f(b, dd)
        np.testing.assert_allclose(x, ref(b, dd), rtol=0,
                                   atol=1e-12*np.abs(x).max())
        np.testing.assert_allclose(C.T @ x, 0 if dd is None else dd,
                                   atol=1e-12)
    if linear_solver == 'dense_lu':
        assert f.det_sign == ref.det_sign


@needs_pardiso
def test_solution_fallback(monkeypatch):
    """A solution whose relative residual exceeds PARDISO_LU_RTOL (forced
    negative) is replaced by that of SuperLU, with a RuntimeWarning, and
    SuperLU is used by the next calls"""
    A = unsymmetric_matrix()
    b = rhs(A.shape[0])
    ref = _factorize(A, 'splu')
    f = _factorize(A, 'pardiso_lu')
    monkeypatch.setattr(static_module, 'PARDISO_LU_RTOL', -1.)
    with pytest.warns(RuntimeWarning, match='pardiso_lu'):
        x = f(b)
    assert f.linear_solver == 'splu'
    np.testing.assert_allclose(x, ref(b), rtol=1e-14, atol=1e-14)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        np.testing.assert_allclose(f(2*b), ref(2*b), rtol=1e-14, atol=1e-14)


@pytest.fixture
def phases(monkeypatch):
    """Count the phases of the PARDISO factorizations of mtype 11, the
    default and the fallback parameters apart; with ``force_perturbed``,
    the default factorizations report perturbed pivots"""
    from structsolve.pardiso import PardisoSolver
    count = Counter()
    count.force_perturbed = False
    factorize = PardisoSolver.factorize

    def counted(self, A, phase=12):
        out = factorize(self, A, phase)
        fallback = bool(self.iparm[0]) and self.iparm[23] == 10
        count[('fallback' if fallback else 'default', phase)] += 1
        if count.force_perturbed and not fallback:
            self.iparm[13] = 5
        return out

    monkeypatch.setattr(PardisoSolver, 'factorize', counted)
    return count


@needs_pardiso
def test_analysis_kept_per_pattern(phases):
    """With the cache, the second factorization of a pattern is a phase 22
    of a solver that keeps the analysis, and two live factorizations of
    the same pattern do not overwrite each other: each still solves its
    own matrix, compared with SuperLU"""
    A = unsymmetric_matrix()
    b = rhs(A.shape[0])
    cache = _FactorizationCache()
    f1 = _factorize(A, 'pardiso_lu', cache=cache)
    f2 = _factorize(2*A, 'pardiso_lu', cache=cache)
    assert phases == {('default', 12): 2}
    x1 = _factorize(A, 'splu')(b)
    np.testing.assert_allclose(f1(b), x1, atol=1e-12*np.abs(x1).max())
    np.testing.assert_allclose(f2(b), x1/2, atol=1e-12*np.abs(x1).max())
    del f1
    assert len(cache.free_solvers) == 1
    f3 = _factorize(3*A, 'pardiso_lu', cache=cache)
    assert phases == {('default', 12): 2, ('default', 22): 1}
    # the factorization of f2, alive, is not overwritten by that of f3
    np.testing.assert_allclose(f2(b), x1/2, atol=1e-12*np.abs(x1).max())
    np.testing.assert_allclose(f3(b), x1/3, atol=1e-12*np.abs(x1).max())
    # the solvers of 'pardiso' are not those of 'pardiso_lu'
    del f2, f3
    assert len(cache.free_solvers) == 2
    S = csr_matrix(A + A.T + 10*sp.eye(A.shape[0]))
    cache2 = _FactorizationCache()
    g = _factorize(S, 'pardiso_lu', cache=cache2)
    del g
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        g = _factorize(2*S, 'pardiso', cache=cache2)
    assert len(cache2.free_solvers) == 1
    del g
    cache.close()
    cache2.close()
    assert not cache.free_solvers


@needs_pardiso
def test_factorization_fallbacks(phases, monkeypatch):
    """A factorization with perturbed pivots (forced) whose residual check
    fails is made again with PARDISO_LU_FALLBACK_IPARM, used directly by
    the next factorizations of the pattern; when that also fails, SuperLU
    is used, with a RuntimeWarning, as with 'splu'"""
    phases.force_perturbed = True
    A = unsymmetric_matrix()
    b = rhs(A.shape[0])
    ref = _factorize(A, 'splu')
    residual = static_module._lu_residual

    def fails_default(solve, A, m):
        # the default factorizations, with the forced perturbed pivots, fail
        return 1. if solve.__self__.iparm[23] != 10 else residual(solve, A, m)

    monkeypatch.setattr(static_module, '_lu_residual', fails_default)
    cache = _FactorizationCache()
    f = _factorize(A, 'pardiso_lu', cache=cache)
    assert f.pardiso_fallback and f.linear_solver == 'pardiso_lu'
    np.testing.assert_allclose(f(b), ref(b), atol=1e-12*np.abs(ref(b)).max())
    assert phases == {('default', 12): 1, ('fallback', 12): 1}
    del f
    f = _factorize(2*A, 'pardiso_lu', cache=cache)
    assert f.pardiso_fallback
    assert phases == {('default', 12): 1, ('fallback', 12): 1,
                      ('fallback', 22): 1}
    # both fail: SuperLU, also directly for the next factorizations
    del f
    monkeypatch.setattr(static_module, '_lu_residual', lambda s, A, m: 1.)
    cache = _FactorizationCache()
    with pytest.warns(RuntimeWarning, match='pardiso_lu'):
        f = _factorize(A, 'pardiso_lu', cache=cache)
    assert f.linear_solver == 'splu'
    assert f.det_sign == ref.det_sign
    np.testing.assert_array_equal(f(b), ref(b))
    n = sum(phases.values())
    f = _factorize(2*A, 'pardiso_lu', cache=cache)
    assert f.linear_solver == 'splu'
    assert sum(phases.values()) == n


def test_without_mkl(monkeypatch):
    """Without oneMKL, 'pardiso_lu' raises the ImportError of 'pardiso'"""
    def no_mkl():
        raise ImportError('no mkl_rt')
    monkeypatch.setattr(static_module, 'load_mkl', no_mkl)
    with pytest.raises(ImportError):
        _factorize(unsymmetric_matrix(), 'pardiso_lu')


@pytest.mark.parametrize('linear_solver', ['pardiso_lu', 'dense_lu'])
def test_not_for_eigenvalue_analyses(linear_solver):
    """lb and freq accept only 'splu' and 'pardiso'"""
    with pytest.raises(ValueError, match='Invalid linear_solver'):
        linear_buckling_module._check_linear_solver(linear_solver)


def circulatory_springs(P=40.):
    """Chain of 6 hardening springs with a circulatory (unsymmetric) linear
    coupling, ``fint = (K + G) u + a u**3``, ``G`` antisymmetric: the
    tangent stiffness matrix is unsymmetric"""
    n = 6
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[-1, -1] = 100.
    G = 15.*(np.eye(n, k=1) - np.eye(n, k=-1))
    a = 30.*np.arange(1, n + 1)
    fext = np.zeros(n)
    fext[-1] = P
    return Problem(fext, lambda u: (K + G) @ u + a*u**3, K + G,
                   lambda u: np.diag(3*a*u**2))


@pytest.mark.parametrize('linear_solver', solvers())
@pytest.mark.parametrize('NL_method', ['NR', 'arc_length_riks'])
def test_analysis_same_as_splu(linear_solver, NL_method):
    """A non-linear analysis with an unsymmetric tangent stiffness matrix
    gives the increments and the Newton-Raphson iterations of 'splu', and
    its solutions to round-off"""
    results = {}
    for ls in ('splu', linear_solver):
        p = circulatory_springs()
        an = p.analysis(linear_solver=ls, NL_method=NL_method,
                        initialInc=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            increments, cs = an.static(NLgeom=True, silent=True)
        results[ls] = (increments, cs, p.calls_per_step())
    inc0, cs0, calls0 = results['splu']
    inc1, cs1, calls1 = results[linear_solver]
    assert calls1 == calls0
    np.testing.assert_allclose(inc1, inc0, rtol=1e-12, atol=1e-14)
    for c0, c1 in zip(cs0, cs1):
        np.testing.assert_allclose(c1, c0, rtol=1e-10, atol=1e-14)


def test_inertia_options():
    """'pardiso_lu' rejects inertia=True with a ValueError; 'dense_lu'
    records the signs of the determinant of 'splu' and, with
    inertia_num_eigvals, the same counts of negative eigenvalues along the
    path of the von Mises truss through its limit point"""
    p = von_mises_truss(frac=0.99)
    an = p.analysis(linear_solver='pardiso_lu', inertia=True)
    with pytest.raises(ValueError, match='pardiso_lu'):
        an.static(NLgeom=True, silent=True)
    out = {}
    for ls in ('splu', 'dense_lu'):
        p = von_mises_truss(frac=0.99)
        an = p.analysis(linear_solver=ls, inertia=True, inertia_num_eigvals=1,
                        NL_method='arc_length_riks', initialInc=0.1)
        an.static(NLgeom=True, silent=True)
        out[ls] = (an.increments, an.det_signs, an.num_negative_eigvals)
    assert out['dense_lu'][1] == out['splu'][1]
    assert out['dense_lu'][2] == out['splu'][2]
    np.testing.assert_allclose(out['dense_lu'][0], out['splu'][0],
                               rtol=1e-12)
