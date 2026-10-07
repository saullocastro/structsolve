"""Solver policy of lb() and freq(): with sparse_solver=True the sparse
solvers always run first, whatever the size of the problem, and the dense
solvers are only a verified last resort within max_dense_size"""
import importlib
import os
import warnings

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, load_npz, random as sprandom

import structsolve.arpackutils as arpackutils
import structsolve.linear_buckling as linear_buckling
from structsolve import freq, lb
from structsolve.linear_buckling import DenseFallbackWarning

#NOTE structsolve.freq is the function freq(), exported by the package
freq_module = importlib.import_module('structsolve.freq')

DATA = os.path.join(os.path.dirname(__file__), 'data')


def load_saved_matrices(name):
    K = load_npz(os.path.join(DATA, '%s_K.npz' % name)).tocsr()
    KG = load_npz(os.path.join(DATA, '%s_KG.npz' % name)).tocsr()
    return K, KG


def spy(monkeypatch, module, names):
    """Replace the functions ``names`` of ``module`` by wrappers that record
    their calls in the returned list"""
    calls = []
    for name in names:
        func = getattr(module, name)

        def wrapper(*args, _name=name, _func=func, **kwargs):
            calls.append(_name)
            return _func(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)
    return calls


LB_SPARSE = ['_eigsh_cayley', '_eigs_shift_search']
LB_DENSE = ['_eigh_condensed', '_eig_condensed', '_eig_complex', 'eigh']
FREQ_SPARSE = ['_eigsh_freq', '_eigs_freq']
FREQ_DENSE = ['_dense_freq', '_eig_complex', 'eigh']


def freq_matrices(symmetric, n=120):
    rng = np.random.RandomState(0)
    A = sprandom(n, n, density=0.05, random_state=rng)
    K = A @ A.T + diags(np.linspace(1., 10., n))
    if not symmetric:
        K = K + 1e-3*sprandom(n, n, density=0.05, random_state=rng)
    M = diags(1. + rng.rand(n))
    return csr_matrix(K), csr_matrix(M)


@pytest.mark.parametrize('max_dense_size', [0, 2000, 10**9])
@pytest.mark.parametrize('name, symmetric', [
    ('plate_ssss_Nxx', True), ('cylinder_5panels', True),
    ('s1B1_m14n8', False), ('s3B2_m14n8', False)])
def test_lb_sparse_solver_runs_first(monkeypatch, name, symmetric,
                                     max_dense_size):
    """The sparse solver is called first, also for the small problems that
    the version 0.5.3 solved with a dense solver, for any max_dense_size,
    and no dense solver is called when it succeeds"""
    K, KG = load_saved_matrices(name)
    calls = spy(monkeypatch, linear_buckling, LB_SPARSE + LB_DENSE)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, _ = lb(K, KG, silent=True, max_dense_size=max_dense_size)
    assert calls == [LB_SPARSE[0] if symmetric else LB_SPARSE[1]]
    assert np.isreal(eigvals[0]) and eigvals[0].real > 0


@pytest.mark.parametrize('name', ['plate_ssss_Nxx', 's1B1_m14n8'])
def test_lb_explicit_dense(monkeypatch, name):
    """sparse_solver=False calls the dense solver only"""
    K, KG = load_saved_matrices(name)
    calls = spy(monkeypatch, linear_buckling, LB_SPARSE + LB_DENSE)
    lb(K, KG, silent=True, sparse_solver=False)
    assert calls == (['eigh'] if name.startswith('plate')
                     else ['_eig_complex'])


@pytest.mark.parametrize('max_dense_size', [0, 2000, 10**9])
@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_sparse_solver_runs_first(monkeypatch, symmetric,
                                       max_dense_size):
    K, M = freq_matrices(symmetric)
    calls = spy(monkeypatch, freq_module, FREQ_SPARSE + FREQ_DENSE)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        freq(K, M, silent=True, symmetric=symmetric,
             max_dense_size=max_dense_size)
    assert calls == [FREQ_SPARSE[0] if symmetric else FREQ_SPARSE[1]]


@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_explicit_dense(monkeypatch, symmetric):
    K, M = freq_matrices(symmetric)
    calls = spy(monkeypatch, freq_module, FREQ_SPARSE + FREQ_DENSE)
    freq(K, M, silent=True, symmetric=symmetric, sparse_solver=False)
    assert calls == (['eigh'] if symmetric else ['_eig_complex'])


def wrong_eigenpairs(K, k):
    rng = np.random.RandomState(0)
    return -rng.rand(k) - 1., rng.randn(K.shape[0], k)


@pytest.mark.parametrize('name, sparse', [
    ('plate_ssss_Nxx', '_eigsh_cayley'), ('s3B2_m14n8', '_eigs_shift_search')])
def test_lb_dense_fallback(monkeypatch, name, sparse):
    """Sparse solvers returning wrong eigenpairs: the dense fallback returns
    the correct result with a DenseFallbackWarning, a RuntimeError listing
    every attempted solver is raised without the fallback"""
    K, KG = load_saved_matrices(name)
    ref, _ = lb(K, KG, silent=True)
    monkeypatch.setattr(linear_buckling, sparse,
                        lambda K, KG, k, *args, **kwargs:
                        wrong_eigenpairs(K, k))
    with pytest.warns(DenseFallbackWarning, match='relative residual'):
        eigvals, eigvecs = lb(K, KG, silent=True)
    np.testing.assert_allclose(eigvals[0], ref[0], rtol=1e-8)
    for max_dense_size in (0, 10):
        with pytest.raises(RuntimeError) as e:
            lb(K, KG, silent=True, max_dense_size=max_dense_size)
        message = str(e.value)
        assert 'Attempted:' in message and 'relative residual' in message
        assert ('disabled' if max_dense_size == 0 else 'skipped') in message


@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_dense_fallback(monkeypatch, symmetric):
    """As test_lb_dense_fallback, for freq(): the dense fallback returns the
    lowest eigenvalues, as the sparse solver"""
    K, M = freq_matrices(symmetric)
    l_ref, _ = freq(K, M, silent=True, symmetric=symmetric, num_eigvalues=10,
                    sort=False)
    sparse = '_eigsh_freq' if symmetric else '_eigs_freq'
    monkeypatch.setattr(freq_module, sparse,
                        lambda K, M, k, *args: wrong_eigenpairs(K, k))
    with pytest.warns(DenseFallbackWarning, match='relative residual'):
        lambda2, _ = freq(K, M, silent=True, symmetric=symmetric,
                          num_eigvalues=10, sort=False)
    np.testing.assert_allclose(np.sort((-lambda2).real),
                               np.sort((-l_ref).real), rtol=1e-8)
    for max_dense_size in (0, 50):
        with pytest.raises(RuntimeError, match='Attempted:'):
            freq(K, M, silent=True, symmetric=symmetric, num_eigvalues=10,
                 max_dense_size=max_dense_size)


def test_freq_missing_mode_detected(monkeypatch):
    """The inertia of K - s M detects a missing lowest mode"""
    K, M = freq_matrices(True)
    sparse = freq_module._eigsh_freq

    def missing_lowest(K, M, k, tol, sigma):
        lambda2, eigvecs = sparse(K, M, k + 1, tol, sigma)
        order = np.argsort(-lambda2)
        return lambda2[order[1:]], eigvecs[:, order[1:]]

    monkeypatch.setattr(freq_module, '_eigsh_freq', missing_lowest)
    with pytest.raises(RuntimeError, match='eigenvalues omega'):
        freq(K, M, silent=True, symmetric=True, num_eigvalues=10,
             max_dense_size=0)
    with pytest.warns(DenseFallbackWarning, match='eigenvalues omega'):
        lambda2, _ = freq(K, M, silent=True, symmetric=True,
                          num_eigvalues=10)
    w_ref = eigh(K.toarray(), M.toarray(), eigvals_only=True)[:10]
    np.testing.assert_allclose(-lambda2, w_ref, rtol=1e-10)


def mkl_bug_present():
    return bool(arpackutils.dsteqr_is_faulty())


@pytest.mark.parametrize('name', ['plate_ssss_Nxx', 'cylinder_5panels'])
def test_lb_mkl_without_cap_uses_dense_fallback(monkeypatch, name):
    """The real-world case of the dense fallback: with a faulty dsteqr and
    the cap of ncv disabled, eigsh() fails and the dense fallback gives the
    correct load multipliers, with a warning. Without the bug, eigsh()
    succeeds"""
    monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', 0)
    K, KG = load_saved_matrices(name)
    ref, _ = lb(K, KG, silent=True, sparse_solver=False)
    if mkl_bug_present():
        with pytest.warns(DenseFallbackWarning, match='eigsh'):
            eigvals, _ = lb(K, KG, silent=True)
        with pytest.raises(RuntimeError, match='Attempted:'):
            lb(K, KG, silent=True, max_dense_size=0)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('error', DenseFallbackWarning)
            eigvals, _ = lb(K, KG, silent=True)
    np.testing.assert_allclose(eigvals, ref[:25], rtol=1e-6)


def test_freq_mkl_without_cap_uses_dense_fallback(monkeypatch):
    """As test_lb_mkl_without_cap_uses_dense_fallback, for
    freq(symmetric=True)"""
    monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', 0)
    K, _ = load_saved_matrices('plate_ssss_Nxx')
    M = diags(1. + np.random.RandomState(3).rand(K.shape[0])).tocsr()
    w_ref = np.sort(1/eigh(M.toarray(), K.toarray(),
                           eigvals_only=True))[:25]
    if mkl_bug_present():
        with pytest.warns(DenseFallbackWarning):
            lambda2, _ = freq(K, M, silent=True, symmetric=True, sort=False)
    else:
        lambda2, _ = freq(K, M, silent=True, symmetric=True, sort=False)
    np.testing.assert_allclose(np.sort(-lambda2), w_ref, rtol=1e-8)
