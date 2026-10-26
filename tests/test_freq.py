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
    assert np.all(omega2 > 1e-6)
    # exactly non-decreasing, which implies the non-decreasing rounded
    # omega_n of the primary sort key: the double eigenvalue omega2 = 100,
    # of the springs and of k = 100, which the dense eig() returns with a
    # round-off difference in an order depending on the LAPACK build, is
    # ordered by the exact tie-break of the sort key
    assert np.all(np.diff(omega2) >= 0)
    lambda2_u, _ = freq(K, M, silent=True, num_eigvalues=6, sort=False,
                        sparse_solver=sparse_solver, symmetric=symmetric)
    w2_u = np.sort((-lambda2_u).real)
    np.testing.assert_allclose(omega2, w2_u[w2_u > 1e-6][:omega2.size],
                               rtol=1e-8)
    # each eigenvalue with its eigenvector
    res = np.linalg.norm(K @ eigvecs + (M @ eigvecs)*lambda2, axis=0)
    assert np.all(res <= 1e-6*np.linalg.norm(K @ eigvecs, axis=0))


@pytest.mark.parametrize('w2_tie', [(100.0001, 100.),
                                    (100., np.nextafter(100., 0.))])
def test_freq_sort_tie(monkeypatch, w2_tie):
    """Eigenpairs with the same rounded sort key are returned by increasing
    exact natural frequency, not in the order of the solver: the solver is
    replaced by one returning the tied pair of omega2 ``w2_tie`` in
    decreasing order. With 100 and the omega2 one unit in the last place
    below it, the double eigenvalue returned by the dense eig() with
    OpenBLAS, both omega_n are 10.0 after the square root, and the tie is
    broken by the exact omega2"""
    import importlib
    freq_module = importlib.import_module('structsolve.freq')
    w2 = np.array([400., w2_tie[0], w2_tie[1], 900.])
    K = csc_matrix(np.diag(w2))
    M = csc_matrix(np.eye(w2.size))

    def eig_decreasing_tie(a, b):
        # eigenvalues 1/omega2 of M u = (1/omega2) K u, in the order of w2
        return 1./w2.astype(complex), np.eye(w2.size, dtype=complex)

    monkeypatch.setattr(freq_module, '_eig_complex', eig_decreasing_tie)
    lambda2_u, _ = freq(K, M, silent=True, sort=False, sparse_solver=False,
                        symmetric=False)
    w2_u = (-lambda2_u).real
    # the solver order has the tie in decreasing order of omega2
    assert w2_u[1] > w2_u[2]
    assert np.round(np.sqrt(w2_u[1]), 1) == np.round(np.sqrt(w2_u[2]), 1)
    lambda2, eigvecs = freq(K, M, silent=True, sort=True,
                            sparse_solver=False, symmetric=False)
    omega2 = (-lambda2).real
    assert np.all(np.diff(omega2) > 0)
    np.testing.assert_array_equal(omega2, np.sort(w2_u))
    # each eigenvalue with its eigenvector, e_i for omega2 = w2[i]
    np.testing.assert_array_equal(np.argmax(np.abs(eigvecs), axis=0),
                                  np.argsort(w2, kind='stable'))


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
    without the dense fallback, its residual being relative to the
    matrix-norm scale ``(||K||_F + |lambda2| ||M||_F) ||u||``"""
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


def penalty_chains(n=20, kp=1e8, free=False):
    """Two chains of ``n`` unit springs, each fixed at its outer end unless
    ``free=True``, connected by a penalty spring ``kp``, which sets
    ``||K||_F``, and a unit mass matrix"""
    K = 2*np.eye(2*n) - np.eye(2*n, k=1) - np.eye(2*n, k=-1)
    K[n - 1, n] = K[n, n - 1] = -kp
    K[n - 1, n - 1] = K[n, n] = 1 + kp
    if free:
        K[0, 0] = K[-1, -1] = 1.
    return csc_matrix(K), csc_matrix(np.eye(2*n))


def test_freq_check_penalty():
    """Regression test: with a penalty spring setting ``||K||_F``, an
    eigenvalue ``omega**2`` 0.01 % too low, relative residual 5e-5, is
    rejected with ``check_rtol=1e-6``. A floor of the denominator of 1e-8
    times the matrix-norm scale ``(||K||_F + |lambda2| ||M||_F) ||u||``,
    about 170 times ``||K u|| + |lambda2| ||M u||`` here, accepted it"""
    from structsolve.freq import _check_freq, _residuals

    K, M = penalty_chains()
    w2, u = np.linalg.eigh(K.toarray())
    w2, u = w2[:1], u[:, :1]
    assert _check_freq(K, M, -w2, u, 1e-6, True) is None
    lambda2 = -w2*(1 - 1e-4)
    # the residual of the eigenpair of eigh(), up to 2e-3 of 5e-5 with
    # Apple's LAPACK, adds to the one of the perturbation
    np.testing.assert_allclose(_residuals(K, M, lambda2, u), 1e-4/(2 - 1e-4),
                               rtol=1e-2)
    error = _check_freq(K, M, lambda2, u, 1e-6, True)
    assert 'relative residual' in error


@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_rigid_body_mode_penalty(symmetric):
    """Two free-free chains of springs connected by a penalty spring: the
    rigid-body mode, whose ``||K u|| + |lambda2| ||M u||`` is at the
    round-off of the matrix-norm scale, passes the residual check without
    the dense fallback, while the other eigenpairs keep the relative
    residual"""
    import warnings
    from structsolve import DenseFallbackWarning
    from structsolve.freq import _residuals
    from structsolve.linear_buckling import RIGID_BODY_RTOL, _frobenius

    K, M = penalty_chains(kp=1e6, free=True)
    w2_ref = np.linalg.eigvalsh(K.toarray())
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lambda2, eigvecs = freq(K, M, silent=True, num_eigvalues=5,
                                symmetric=symmetric, sort=False,
                                max_dense_size=0)
    order = np.argsort((-lambda2).real)
    w2 = (-lambda2).real[order]
    u = eigvecs[:, order]
    # null up to the round-off of ||K||_F
    assert abs(w2[0]) <= RIGID_BODY_RTOL*_frobenius(K)
    np.testing.assert_allclose(w2[1:], w2_ref[1:5], rtol=1e-8)
    Ku = K @ u
    Mu = M @ u
    scale = (_frobenius(K) + np.abs(w2)*_frobenius(M))*np.linalg.norm(u,
                                                                       axis=0)
    d = np.linalg.norm(Ku, axis=0) + np.abs(w2)*np.linalg.norm(Mu, axis=0)
    rigid = d <= RIGID_BODY_RTOL*scale
    assert rigid.tolist() == [True, False, False, False, False]
    res = _residuals(K, M, -w2, u)
    assert np.all(res <= 1e-6)
    plain = np.linalg.norm(Ku - Mu*w2, axis=0)/d
    np.testing.assert_allclose(res[1:], plain[1:], rtol=1e-12)


def beam_cantilever():
    """Cantilever of pyfe3d, saved in tests/data: 49 ``BeamC`` elements of
    a steel beam of length 3 m with a 0.05 x 0.05 m square cross-section,
    clamped at one end, ``K`` and ``M`` (consistent, ``mtype=0``) of the
    free dofs, and ``KG`` of a unit axial compression at the tip, as in
    pyfe3d's tests/test_beamc_natural_freq_cantilever_pre_stress.py"""
    import os
    from scipy.sparse import load_npz

    data = os.path.join(os.path.dirname(__file__), 'data')
    return [load_npz(os.path.join(data, 'beam_cantilever_%s.npz' % name)
                     ).tocsr() for name in ('K', 'KG', 'M')]


@pytest.mark.parametrize('symmetric', [True, False])
def test_freq_near_buckling(symmetric):
    """Regression test: the cantilever pre-stressed to 0.9999 of its
    buckling load has a nearly singular ``K``. Its two lowest eigenpairs,
    ``omega**2 = 0.0895``, a double eigenvalue of the square cross-section,
    have relative residuals of 2.5e-6 to 9.1e-6 with the sparse and the
    dense solvers, larger than the default ``check_rtol=1e-6``, but backward
    errors at the round-off, and pass the verification of the sparse solver
    without the dense fallback and without a larger ``check_rtol``"""
    import warnings
    from scipy.linalg import eigh
    from structsolve import DenseFallbackWarning, lb
    from structsolve.freq import _residuals
    from structsolve.linear_buckling import (BACKWARD_ERROR_RTOL,
                                             _backward_errors, _frobenius)

    K0, KG, M = beam_cantilever()
    lambda_cr = lb(K0, KG, silent=True, num_eigvalues=1)[0][0]
    K = (K0 + 0.9999*lambda_cr*KG).tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lambda2, eigvecs = freq(K, M, silent=True, num_eigvalues=3,
                                symmetric=symmetric, max_dense_size=0)
    w2 = (-lambda2).real
    # reference: M u = (1/omega**2) K u, accurate for the lowest omega**2
    mu = eigh(M.toarray(), K.toarray(), eigvals_only=True)
    w2_ref = np.sort(1/mu[mu > 0])[:3]
    np.testing.assert_allclose(w2, w2_ref, rtol=1e-4)
    assert np.all(w2[:2] < 1e-5*w2[2])
    # ||K u|| + omega**2 ||M u|| is about 5000 eps times the matrix-norm
    # scale, which amplifies the relative residual of the lowest eigenpairs
    u = eigvecs[:, :2]
    d = (np.linalg.norm(K @ u, axis=0)
         + w2[:2]*np.linalg.norm(M @ u, axis=0))
    scale = (_frobenius(K) + w2[:2]*_frobenius(M))*np.linalg.norm(u, axis=0)
    assert np.all(d < 1e4*np.finfo(float).eps*scale)
    res = _residuals(K, M, lambda2, eigvecs)
    be = _backward_errors(K, M, lambda2, eigvecs)
    assert np.all((res <= 1e-6) | (be <= BACKWARD_ERROR_RTOL))


def test_freq_check_backward_error():
    """The residual check of freq on both sides of
    ``BACKWARD_ERROR_RTOL``, with eigenvalues perturbed from the dense
    eigenpairs: the lowest eigenpair of the cantilever pre-stressed to
    0.9999 of its buckling load, ``omega**2`` 1e-5 too large, relative
    residual about 5e-6, backward error about 0.03 eps, is accepted; the
    lowest eigenpairs of the blade-stiffened plate with penalty connections
    of tests/data, ``lambda`` 0.1 % too low as the inaccurate eigenpairs of
    eigsh, relative residuals about 5e-4, backward errors of 16 eps and
    more, are rejected"""
    from scipy.linalg import eigh
    from scipy.sparse import load_npz
    import os
    from structsolve.freq import _check_freq, _residuals
    from structsolve.linear_buckling import (BACKWARD_ERROR_RTOL,
                                             _backward_errors,
                                             _eigh_condensed)

    K0, KG, M = beam_cantilever()
    mu, u = eigh(KG.toarray(), K0.toarray())
    lambda_cr = -1/mu[0]
    K = (K0 + 0.9999*lambda_cr*KG).tocsr()
    mu, u = eigh(M.toarray(), K.toarray())
    w2, u = 1/mu[-1:], u[:, -1:]
    lambda2 = -w2*(1 + 1e-5)
    res = _residuals(K, M, lambda2, u)
    be = _backward_errors(K, M, lambda2, u)
    assert np.all(res > 1e-6) and np.all(be <= 0.1*BACKWARD_ERROR_RTOL)
    assert _check_freq(K, M, lambda2, u, 1e-6, True) is None

    data = os.path.join(os.path.dirname(__file__), 'data')
    K = load_npz(os.path.join(data, 'blade_stiffened_penalty_K.npz')).tocsr()
    KG = load_npz(os.path.join(data, 'blade_stiffened_penalty_KG.npz')).tocsr()
    mu, u = _eigh_condensed(K, KG, 4)
    lam = -1/mu
    # (K + lam KG) u = 0 checked as (K + lambda2 M) u = 0
    assert _check_freq(K, KG, lam, u, 1e-6, False) is None
    lam = lam*(1 - 1e-3)
    res = _residuals(K, KG, lam, u)
    be = _backward_errors(K, KG, lam, u)
    assert np.all(res > 1e-4) and np.all(be >= 10*BACKWARD_ERROR_RTOL)
    error = _check_freq(K, KG, lam, u, 1e-6, False)
    assert 'relative residual' in error and 'backward error' in error
