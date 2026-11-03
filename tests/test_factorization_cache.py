"""Tests of the cache of the pattern-dependent work of the factorizations

:class:`.static._FactorizationCache`, created by :meth:`.Analysis.static`
for the non-linear analyses and passed to :func:`.static._factorize`, must
give exactly the factorizations without the cache, and must be rebuilt when
the pattern, the null rows and columns or the constraints change.

"""
import importlib
import warnings
from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix

from structsolve.static import _factorize, _FactorizationCache
from analytic_problems import von_mises_truss, Problem
from structsolve.pardiso import available as pardiso_available

#NOTE structsolve.static is the function static(), exported by the package
static_module = importlib.import_module('structsolve.static')
analysis_module = importlib.import_module('structsolve.analysis')

LINEAR_SOLVERS = ['splu', 'pardiso']


def has_pardiso():
    return pardiso_available()


def solvers():
    return [s for s in LINEAR_SOLVERS if s == 'splu' or has_pardiso()]


@pytest.fixture
def phases(monkeypatch):
    """Count the phases of the PARDISO factorizations"""
    from structsolve.pardiso import PardisoSolver
    count = Counter()
    factorize = PardisoSolver.factorize

    count.fallbacks = 0

    def counted(self, A, phase=12):
        # the factorizations with PARDISO_FALLBACK_IPARM, which depend on
        # the version of MKL, counted apart
        if self.iparm[0] and self.iparm[23] == 1:
            count.fallbacks += 1
        else:
            count[phase] += 1
        return factorize(self, A, phase)

    monkeypatch.setattr(PardisoSolver, 'factorize', counted)
    return count


def random_matrix(rng, n, symmetric):
    """Sparse matrix with null rows and columns, a row and a column of
    explicit zeros, and an explicit zero off the diagonal"""
    A = sp.random(n, n, density=0.15, random_state=rng, format='csr')
    if symmetric:
        A = A + A.T
    A = csr_matrix(A + sp.diags(rng.standard_normal(n)))
    keep = np.ones(n)
    keep[rng.choice(n, n//5, replace=False)] = 0
    A = csr_matrix(sp.diags(keep) @ A @ sp.diags(keep))
    A.eliminate_zeros()
    i = int(np.flatnonzero(keep == 0)[0])
    j = int(np.flatnonzero(keep)[0])
    A = A.tolil()
    A[i, j] = 1.
    A[j, i] = 1.
    A = A.tocsr()
    A.data[A.indices == i] = 0.
    A.data[A.indptr[i]:A.indptr[i + 1]] = 0.
    A.sort_indices()
    return A, keep


def same(f0, f1, b):
    x0, x1 = f0(b), f1(b)
    assert np.array_equal(x0, x1)
    for attr in ('linear_solver', 'det_sign', 'num_negative_eigvals',
                 'num_perturbed_pivots', 'num_constraints'):
        assert getattr(f0, attr) == getattr(f1, attr)


@pytest.mark.parametrize('linear_solver', solvers())
@pytest.mark.parametrize('symmetric', [True, False])
@pytest.mark.parametrize('border', [False, True])
@pytest.mark.parametrize('seed', range(4))
def test_same_factorization(seed, border, symmetric, linear_solver):
    """Bitwise the factorization without the cache, for matrices of the
    same pattern and different values"""
    rng = np.random.default_rng(seed)
    n = 40
    A, keep = random_matrix(rng, n, symmetric)
    C = None
    if border:
        C = np.zeros((n, 2))
        C[:, 0] = keep
        C[np.flatnonzero(keep)[-1], 1] = 1.
    b = rng.standard_normal(n)
    cache = _FactorizationCache()
    for k in range(3):
        K = A.copy()
        K.data = K.data*(1 + 0.1*k)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            f0 = _factorize(K, linear_solver, constraints=C)
            f1 = _factorize(K, linear_solver, constraints=C, cache=cache)
        same(f0, f1, b)
    assert cache.entry is not None
    cache.close()


@pytest.mark.skipif(not has_pardiso(), reason='oneMKL is not found')
def test_pardiso_phases(phases):
    """The analysis of PARDISO is reused while the pattern does not change,
    also by the factorizations that coexist, and redone when it changes"""
    rng = np.random.default_rng(0)
    A, keep = random_matrix(rng, 50, True)
    cache = _FactorizationCache()
    f1 = _factorize(A, 'pardiso', cache=cache)
    f2 = _factorize(2*A, 'pardiso', cache=cache)
    assert phases == {12: 2}
    del f1
    f3 = _factorize(3*A, 'pardiso', cache=cache)
    assert phases == {12: 2, 22: 1}
    del f2, f3
    assert len(cache.free_solvers) == 2
    for k in range(5):
        f = _factorize((4 + k)*A, 'pardiso', cache=cache)
    assert phases == {12: 2, 22: 6}
    # a null row and column by its values, same pattern: new analysis
    j = np.flatnonzero(keep)[0]
    B = A.copy()
    B.data[B.indices == j] = 0.
    B.data[B.indptr[j]:B.indptr[j + 1]] = 0.
    del f
    f = _factorize(B, 'pardiso', cache=cache)
    assert phases == {12: 3, 22: 6}
    # new pattern
    del f
    f = _factorize(A + sp.eye(50, k=7) + sp.eye(50, k=-7), 'pardiso',
                   cache=cache)
    assert phases == {12: 4, 22: 6}
    # constraints
    del f
    f = _factorize(A, 'pardiso', constraints=keep, cache=cache)
    assert phases == {12: 5, 22: 6}
    del f
    f = _factorize(2*A, 'pardiso', constraints=keep, cache=cache)
    assert phases == {12: 5, 22: 7}
    del f
    f = _factorize(2*A, 'pardiso', constraints=2*keep, cache=cache)
    assert phases == {12: 6, 22: 7}
    del f
    cache.close()
    assert not cache.free_solvers


def test_pattern_change_splu():
    """Rebuilt when the pattern or the null rows and columns change"""
    rng = np.random.default_rng(1)
    A, keep = random_matrix(rng, 30, False)
    b = rng.standard_normal(30)
    cache = _FactorizationCache()
    same(_factorize(A, cache=cache), _factorize(A), b)
    entry = cache.entry
    same(_factorize(3*A, cache=cache), _factorize(3*A), b)
    assert cache.entry is entry
    B = (A + sp.eye(30, k=3)).tocsr()
    same(_factorize(B, cache=cache), _factorize(B), b)
    assert cache.entry is not entry
    entry = cache.entry
    num_used = entry.used_cols.size
    # a null row and column by its values, same pattern
    j = np.flatnonzero(keep)[0]
    B2 = B.copy()
    B2.data[B2.indices == j] = 0.
    B2.data[B2.indptr[j]:B2.indptr[j + 1]] = 0.
    same(_factorize(B2, cache=cache), _factorize(B2), b)
    assert cache.entry is not entry
    assert cache.entry.used_cols.size == num_used - 1


def test_not_cached():
    """Matrices in another format, ``float32`` or with duplicated terms"""
    A = csr_matrix(np.array([[2., 1.], [1., 3.]]))
    cache = _FactorizationCache()
    same(_factorize(csc_matrix(A), cache=cache), _factorize(A), np.ones(2))
    assert cache.entry is not None
    cache = _FactorizationCache()
    _factorize(A.astype(np.float32), cache=cache)
    assert cache.entry is None
    D = csr_matrix((np.array([1., 1., 1., 3.]), np.array([0, 0, 1, 1]),
                    np.array([0, 2, 4])), shape=(2, 2))
    assert not D.has_canonical_format
    same(_factorize(D, cache=cache), _factorize(D), np.ones(2))
    assert cache.entry is None


def chain(n=30, P=20.):
    """Chain of springs with cubic hardening, sparse, with two null degrees
    of freedom at the end, the pattern of the tangent constant"""
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[-1, -1] = 100.
    a = 50.*np.linspace(1, 2, n)
    Kf = np.zeros((n + 2, n + 2))
    Kf[:n, :n] = K
    af = np.r_[a, 0., 0.]
    fext = np.zeros(n + 2)
    fext[n - 1] = P
    return Problem(fext, lambda u: Kf @ u + af*u**3, Kf,
                   lambda u: np.diag(3*af*u**2))


class NoCache(_FactorizationCache):
    def get(self, K, C):
        return None


def run(p, cached, monkeypatch, **kwargs):
    an = p.analysis(**kwargs)
    with monkeypatch.context() as m:
        if not cached:
            m.setattr(analysis_module, '_FactorizationCache', NoCache)
        an.static(NLgeom=True, silent=True)
    assert an._factorization_cache is None
    return an


@pytest.mark.parametrize('linear_solver', solvers())
@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield',
                                    'generalized_displacement_control'])
def test_analysis_identical(method, linear_solver, monkeypatch):
    """Bitwise the analysis without the cache, with constraints and the
    inertia"""
    C = np.zeros(32)
    C[[3, 7]] = [1., -1.]
    kwargs = dict(NL_method=method, linear_solver=linear_solver,
                  inertia=True, constraints=C, initialInc=0.05)
    if method != 'NR':
        kwargs.update(final_load_factor=None,
                      stop_criterion=lambda an: len(an.increments) == 8)
    an0 = run(chain(), False, monkeypatch, **kwargs)
    an1 = run(chain(), True, monkeypatch, **kwargs)
    assert len(an1.increments) > 3
    assert an0.increments == an1.increments
    for c0, c1 in zip(an0.cs, an1.cs):
        assert np.array_equal(c0, c1)
    assert an0.num_negative_eigvals == an1.num_negative_eigvals
    assert an0.det_signs == an1.det_signs
    assert an0.num_perturbed_pivots == an1.num_perturbed_pivots


@pytest.mark.skipif(not has_pardiso(), reason='oneMKL is not found')
def test_analysis_limit_point(monkeypatch, phases):
    """Riks beyond the limit point of the von Mises truss with PARDISO:
    the same path and inertia, the analysis of PARDISO reused"""
    kwargs = dict(NL_method='arc_length_riks', linear_solver='pardiso',
                  inertia=True, final_load_factor=None, maxArcLength=6.)
    an0 = run(von_mises_truss(frac=1.5), False, monkeypatch, **kwargs)
    n12 = phases[12]
    assert phases[22] == 0
    an1 = run(von_mises_truss(frac=1.5), True, monkeypatch, **kwargs)
    assert 1 in an1.num_negative_eigvals
    assert an0.num_negative_eigvals == an1.num_negative_eigvals
    assert an0.increments == an1.increments
    for c0, c1 in zip(an0.cs, an1.cs):
        assert np.array_equal(c0, c1)
    assert phases[22] > 10*(phases[12] - n12)
