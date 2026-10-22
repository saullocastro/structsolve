"""Tests for structsolve.freq: freq()"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix

from structsolve import freq


def test_freq_simple():
    """Test frequency analysis with known natural frequencies

    For a diagonal system: [K]{u} + omega^2 [M]{u} = 0
    K = diag(k1, k2, ...), M = diag(m1, m2, ...)
    omega_i = sqrt(ki/mi)
    """
    n = 10
    k_vals = np.arange(1, n + 1, dtype=float) * 100
    m_vals = np.ones(n)
    K = csc_matrix(np.diag(k_vals))
    M = csc_matrix(np.diag(m_vals))

    eigvals, eigvecs = freq(K, M, silent=True, num_eigvalues=5,
                            sparse_solver=True)

    # freq returns lambda2 where [K] + lambda2 [M] = 0
    # so lambda2 = -omega^2 = -k/m
    # The sorted natural frequencies should match sqrt(k_vals)
    assert eigvecs.shape[0] == n


def test_freq_dense_solver():
    """Test freq with sparse_solver=False"""
    n = 8
    k_vals = np.array([1, 4, 9, 16, 25, 36, 49, 64], dtype=float)
    m_vals = np.ones(n)
    K = csc_matrix(np.diag(k_vals))
    M = csc_matrix(np.diag(m_vals))

    eigvals, eigvecs = freq(K, M, silent=True, num_eigvalues=5,
                            sparse_solver=False)
    assert eigvecs.shape[0] == n


def test_freq_with_null_dofs():
    """Test frequency analysis with constrained (null) DOFs"""
    n = 8
    K_dense = np.zeros((n, n))
    M_dense = np.zeros((n, n))
    active = [1, 2, 3, 4, 5]
    for i in active:
        K_dense[i, i] = float(i) * 100
        M_dense[i, i] = 1.0
    K = csc_matrix(K_dense)
    M = csc_matrix(M_dense)

    eigvals, eigvecs = freq(K, M, silent=True, num_eigvalues=3,
                            sparse_solver=True)
    assert eigvecs.shape[0] == n


def freq_problem(symmetric, n=300, seed=0):
    from scipy.sparse import diags, random as sprandom
    rng = np.random.RandomState(seed)
    A = sprandom(n, n, density=0.02, random_state=rng)
    K = A @ A.T + diags(np.linspace(1., 100., n))
    if not symmetric:
        K = K + 1e-2*sprandom(n, n, density=0.02, random_state=rng)
    M = diags(1. + rng.rand(n))
    return csc_matrix(K), csc_matrix(M)


@pytest.mark.parametrize('num_eigvalues', [5, 25, 40])
@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_lowest_frequencies(symmetric, num_eigvalues):
    """The sparse solver, without dense fallback, returns the lowest
    frequencies of the full dense solution, also for many modes, which
    eigsh() computes in several passes when ncv is capped"""
    K, M = freq_problem(symmetric)
    lambda2, eigvecs = freq(K, M, silent=True, symmetric=symmetric,
                            num_eigvalues=num_eigvalues, max_dense_size=0,
                            sort=False)
    ref, _ = freq(K, M, silent=True, symmetric=symmetric,
                  sparse_solver=False, sort=False)
    w_ref = np.sort((-ref).real)[:num_eigvalues]
    assert lambda2.shape == (num_eigvalues,)
    np.testing.assert_allclose(np.sort((-lambda2).real), w_ref, rtol=1e-8)
    if symmetric:
        assert np.isrealobj(lambda2)


@pytest.mark.parametrize('sparse_solver', [True, False])
@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_sort(sparse_solver, symmetric):
    """sort=True sorts and filters the eigenvalues together with the
    eigenvectors: increasing natural frequencies, without the rigid-body
    mode"""
    rng = np.random.RandomState(3)
    n = 12
    k = rng.permutation(np.arange(1, n + 1, dtype=float))*100
    K = np.diag(k)
    # a free-free pair of springs: one rigid-body mode
    K[:2, :2] = 50.*np.array([[1., -1.], [-1., 1.]])
    M = np.eye(n)
    K, M = csc_matrix(K), csc_matrix(M)
    lambda2, eigvecs = freq(K, M, silent=True, num_eigvalues=6, sort=True,
                            sparse_solver=sparse_solver, symmetric=symmetric)
    omega2 = (-lambda2).real
    assert lambda2.shape[0] == eigvecs.shape[1]
    assert np.all(np.diff(omega2) >= 0)
    assert np.all(omega2 > 1e-6)
    lambda2_u, _ = freq(K, M, silent=True, num_eigvalues=6, sort=False,
                        sparse_solver=sparse_solver, symmetric=symmetric)
    w2_u = np.sort((-lambda2_u).real)
    np.testing.assert_allclose(omega2, w2_u[w2_u > 1e-6][:omega2.size],
                               rtol=1e-8)
    # each eigenvalue with its eigenvector
    res = np.linalg.norm(K @ eigvecs + (M @ eigvecs)*lambda2, axis=0)
    assert np.all(res <= 1e-6*np.linalg.norm(K @ eigvecs, axis=0))


def test_freq_prestressed_plate():
    """Natural frequencies of the plate of panels saved in tests/data, with
    a lumped mass matrix, under a compressive prestress lam*KG: they
    decrease with lam, and the lowest one vanishes linearly at the critical
    load multiplier of lb"""
    import os
    from scipy.sparse import diags, load_npz
    from structsolve import lb

    data = os.path.join(os.path.dirname(__file__), 'data')
    K = load_npz(os.path.join(data, 'plate_ssss_Nxx_K.npz')).tocsr()
    KG = load_npz(os.path.join(data, 'plate_ssss_Nxx_KG.npz')).tocsr()
    rng = np.random.RandomState(3)
    M = diags(1. + rng.rand(K.shape[0])).tocsr()
    # uniaxial compression: KG is negative semi-definite
    assert np.linalg.eigvalsh(KG.toarray()).max() <= 1e-10*abs(KG).max()
    lambda_1 = lb(K, KG, silent=True, sparse_solver=False)[0][0]

    omega2 = []
    for frac in [0., 0.5, 0.9, 1 - 1e-6, 1 - 1e-7]:
        lambda2, eigvecs = freq(K + frac*lambda_1*KG, M, silent=True,
                                num_eigvalues=5, symmetric=True)
        omega = np.sqrt(-lambda2)        # rad/s
        f_Hz = omega/(2*np.pi)           # Hz
        assert np.all(np.diff(f_Hz) >= 0)
        # symmetric=True: eigenvectors normalized with respect to M
        np.testing.assert_allclose(eigvecs.T @ M @ eigvecs, np.eye(5),
                                   atol=1e-8)
        omega2.append(omega**2)
    omega2 = np.array(omega2)
    # the frequencies decrease with the compressive prestress
    assert np.all(np.diff(omega2, axis=0) <= 1e-12*omega2[0, 0])
    # the lowest one vanishes linearly at the critical load multiplier
    np.testing.assert_allclose(omega2[-2, 0]/omega2[-1, 0], 10., rtol=1e-2)


@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_rigid_body_modes(symmetric):
    """A free-free chain of springs has a rigid-body mode, lambda2 = 0 up to
    the round-off, which passes the residual check of the sparse solvers
    without the dense fallback, the matrix-norm floor of the relative
    residual being meaningful when lambda2 vanishes"""
    import warnings
    from structsolve import DenseFallbackWarning
    from structsolve.freq import _residuals

    n = 30
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[0, 0] = K[-1, -1] = 100.
    K, M = csc_matrix(K), csc_matrix(np.eye(n))
    w2_ref = np.linalg.eigvalsh(K.toarray())
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lambda2, eigvecs = freq(K, M, silent=True, num_eigvalues=5,
                                symmetric=symmetric, sort=False,
                                max_dense_size=0)
    w2 = np.sort((-lambda2).real)
    assert abs(w2[0]) <= 1e-10*w2_ref[1]
    np.testing.assert_allclose(w2[1:], w2_ref[1:5], rtol=1e-8)
    res = _residuals(K, M, lambda2, eigvecs)
    assert np.all(res <= 1e-6)
    # sort=True removes the rigid-body mode
    lambda2_s, eigvecs_s = freq(K, M, silent=True, num_eigvalues=5,
                                symmetric=symmetric)
    assert lambda2_s.shape == (4,) and eigvecs_s.shape == (n, 4)
