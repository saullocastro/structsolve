"""Tests for structsolve.linear_buckling: lb()"""
import os

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from structsolve import lb


def test_lb_simple_eigenvalue():
    """Test linear buckling with a well-conditioned system

    lb solves: [K] + lambda[KG] = 0
    For K = diag(2,4,6,...) and KG = I, lambda_i = -K_ii
    lb returns -1/eigvals from eigsh(A=KG, M=K), giving positive load factors.
    """
    n = 10
    k_diag = np.arange(2, 2 * n + 2, 2, dtype=float)
    K = csc_matrix(np.diag(k_diag))
    KG = csc_matrix(np.eye(n))

    eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=5,
                          sparse_solver=True)
    # lb returns eigenvalues (load multipliers), verify we get results
    assert len(eigvals) >= 5
    assert eigvecs.shape[0] == n


def test_lb_dense_solver():
    """Test lb with sparse_solver=False

    lb solves [K] + lambda[KG] = 0, returning -1/eigvals.
    For K=2I, KG=I: eigsh gives eigval=0.5, so lb returns -1/0.5 = -2.
    The dense solver (eigh) with b=K gives eigvals with the same sign convention.
    """
    n = 10
    K = csc_matrix(np.eye(n) * 2.0)
    KG = csc_matrix(np.eye(n))

    eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=5,
                          sparse_solver=False)
    np.testing.assert_allclose(eigvals[:5], -2.0, rtol=1e-6)


@pytest.mark.parametrize('max_dense_size', [0, 2000])
def test_lb_sparse_shift_mixed_spectrum(max_dense_size):
    """Sparse and dense lb agree when the load multipliers have mixed signs

    Regression test: the shift of the sparse solver was a single Rayleigh
    quotient, in which the positive and negative eigenvalues of
    ``KG u = mu K u`` cancel. The shift landed below the critical ``|mu|``
    and eigsh returned the load multipliers near ``1/sigma`` instead of the
    lowest ones. With ``max_dense_size=0`` eigsh is always used.
    """
    from scipy.sparse import diags
    from structsolve.linear_buckling import _estimate_sigma

    n = 80
    k = np.linspace(1e6, 5e6, n)
    lambdas = np.concatenate([[1.0, 1.0, 1.1, 1.2],
                              np.linspace(1.5, 4., 36),
                              -np.linspace(1.5, 4., 40)])
    mu = -1. / lambdas
    K = diags(k).tocsc()
    KG = diags(mu * k).tocsc()

    # the shift must bound the largest |mu| of the critical eigenvalues
    assert _estimate_sigma(K, KG) >= np.abs(mu).max()

    eig_sparse, _ = lb(K, KG, silent=True, num_eigvalues=10,
                       max_dense_size=max_dense_size)
    eig_dense, _ = lb(K, KG, silent=True, sparse_solver=False)
    np.testing.assert_allclose(eig_sparse[:4], [1.0, 1.0, 1.1, 1.2],
                               rtol=1e-8)
    np.testing.assert_allclose(eig_sparse[:4], eig_dense[:4], rtol=1e-8)


DATA = os.path.join(os.path.dirname(__file__), 'data')

# K and KG of panels models, null rows and columns removed, with the
# critical load multipliers of the dense solver
SAVED_MATRICES = [
    ('plate_ssss_Nxx', [157.5386, 226.3470, 232.5623]),
    ('cylinder_5panels', [473.1924, 473.1924, 478.6168]),
]


def load_saved_matrices(name):
    from scipy.sparse import load_npz
    K = load_npz(os.path.join(DATA, '%s_K.npz' % name)).tocsr()
    KG = load_npz(os.path.join(DATA, '%s_KG.npz' % name)).tocsr()
    return K, KG


def relative_residuals(K, KG, eigvals, eigvecs):
    Ku = K @ eigvecs
    KGu = KG @ eigvecs
    return (np.linalg.norm(Ku + KGu*eigvals, axis=0)
            / (np.linalg.norm(Ku, axis=0)
               + np.abs(eigvals)*np.linalg.norm(KGu, axis=0)))


@pytest.mark.parametrize('name, ref', SAVED_MATRICES)
def test_lb_saved_matrices_sparse_solver(name, ref):
    """Regression test: ARPACK linked against Intel MKL 2024.2.0 to 2025.0.0
    returned wrong load multipliers at random, e.g. with the Anaconda builds
    of SciPy. The default sparse solver condenses out the dofs where KG is
    null and verifies the eigenpairs"""
    K, KG = load_saved_matrices(name)
    eig_dense, _ = lb(K, KG, silent=True, sparse_solver=False)
    np.testing.assert_allclose(eig_dense[:3], ref, rtol=1e-5)
    for trial in range(3):
        eigvals, eigvecs = lb(K, KG, silent=True)
        assert eigvals.shape == (25,)
        assert eigvecs.shape == (K.shape[0], 25)
        np.testing.assert_allclose(eigvals, eig_dense[:25], rtol=1e-6)
        assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)


@pytest.mark.parametrize('name, ref', SAVED_MATRICES)
def test_lb_saved_matrices_eigsh_never_wrong(name, ref):
    """With max_dense_size=0 only eigsh is used, which either returns the
    correct load multipliers or raises a RuntimeError, when the eigenpairs
    returned by ARPACK are wrong"""
    K, KG = load_saved_matrices(name)
    for trial in range(3):
        try:
            eigvals, eigvecs = lb(K, KG, silent=True, max_dense_size=0)
        except RuntimeError as e:
            assert 'verification' in str(e)
            continue
        np.testing.assert_allclose(eigvals[:3], ref, rtol=1e-5)
        assert np.all(eigvals[:-1] <= eigvals[1:])


def test_lb_check_eigenpairs():
    """The verification detects wrong eigenpairs and a missing lowest load
    multiplier"""
    from structsolve.linear_buckling import _check_eigenpairs, _eigh_condensed

    K, KG = load_saved_matrices('plate_ssss_Nxx')
    mu, eigvecs = _eigh_condensed(K, KG, 10)
    eigvals = -1./mu
    assert _check_eigenpairs(K, KG, eigvals, eigvecs, 1e-3) is None

    # eigenvectors of other eigenvalues
    error = _check_eigenpairs(K, KG, eigvals, eigvecs[:, ::-1], 1e-3)
    assert 'relative residual' in error

    # wrong eigenvalues
    error = _check_eigenpairs(K, KG, eigvals*1.01, eigvecs, 1e-3)
    assert 'relative residual' in error

    # lowest load multiplier missing
    error = _check_eigenpairs(K, KG, eigvals[1:], eigvecs[:, 1:], 1e-3)
    assert 'missing' in error

    # both equal critical load multipliers missing
    K, KG = load_saved_matrices('cylinder_5panels')
    mu, eigvecs = _eigh_condensed(K, KG, 10)
    eigvals = -1./mu
    assert _check_eigenpairs(K, KG, eigvals, eigvecs, 1e-3) is None
    error = _check_eigenpairs(K, KG, eigvals[2:], eigvecs[:, 2:], 1e-3)
    assert 'missing' in error


def test_lb_sparse_raises_when_verification_fails(monkeypatch):
    """A RuntimeError is raised instead of returning wrong eigenpairs"""
    import structsolve.linear_buckling as linear_buckling

    K, KG = load_saved_matrices('plate_ssss_Nxx')

    def wrong_eigsh(K, KG, k, tol, silent):
        rng = np.random.RandomState(0)
        return -rng.rand(k), rng.randn(K.shape[0], k)

    monkeypatch.setattr(linear_buckling, '_eigsh_cayley', wrong_eigsh)
    with pytest.raises(RuntimeError, match='verification'):
        lb(K, KG, silent=True, max_dense_size=0)
    # the condensed dense solution is used first
    eigvals, _ = lb(K, KG, silent=True)
    np.testing.assert_allclose(eigvals[0], 157.5386, rtol=1e-5)


@pytest.mark.parametrize('max_dense_size', [0, 2000])
def test_lb_sparse_sorted_as_dense(max_dense_size):
    """The sparse solver returns the positive load multipliers first, in
    increasing order, followed by the negative ones, as the dense solver"""
    from scipy.sparse import diags

    n = 40
    k = np.linspace(1., 3., n)
    rng = np.random.RandomState(1)
    lambdas = rng.permutation(np.concatenate([np.linspace(1., 20., 5),
                                              -np.linspace(1., 20., n - 5)]))
    K = diags(k).tocsr()
    KG = diags(-k/lambdas).tocsr()
    eig_sparse, eigvecs = lb(K, KG, silent=True, num_eigvalues=10,
                             max_dense_size=max_dense_size)
    eig_dense, _ = lb(K, KG, silent=True, sparse_solver=False)
    np.testing.assert_allclose(eig_sparse, eig_dense[:10], rtol=1e-8)
    np.testing.assert_allclose(eig_sparse[:5], np.linspace(1., 20., 5),
                               rtol=1e-8)
    assert np.all(eig_sparse[5:] < 0)
    assert np.all(relative_residuals(K, KG, eig_sparse, eigvecs) < 1e-8)


def test_lb_sparse_null_KG_rows_and_null_cols():
    """Condensed solution with null rows of KG and null rows of K, returning
    eigenvectors with the size of the original matrices"""
    from scipy.sparse import random as sprandom, identity, bmat

    n = 60
    A = sprandom(n, n, density=0.1, random_state=2)
    K = (A @ A.T + identity(n)).tocsr()
    B = sprandom(n//3, n//3, density=0.3, random_state=3)
    KG = bmat([[-(B @ B.T + 0.1*identity(n//3)), None],
               [None, csc_matrix((n - n//3, n - n//3))]]).tocsr()
    # adding two null dofs
    Z = csc_matrix((2, 2))
    K2 = bmat([[K, None], [None, Z]]).tocsr()
    KG2 = bmat([[KG, None], [None, Z]]).tocsr()
    eig_sparse, eigvecs = lb(K2, KG2, silent=True, num_eigvalues=10)
    eig_dense, _ = lb(K2, KG2, silent=True, sparse_solver=False)
    np.testing.assert_allclose(eig_sparse, eig_dense[:10], rtol=1e-8)
    assert eigvecs.shape == (n + 2, 10)
    assert np.all(eigvecs[n:, :] == 0)
    assert np.all(relative_residuals(K2, KG2, eig_sparse, eigvecs) < 1e-8)


def test_lb_sparse_singular_K():
    """The shift falls back to 1 when K is singular"""
    from structsolve.linear_buckling import _estimate_sigma

    K = csc_matrix(np.diag([1., 2., 0., 4., 5.]))
    KG = csc_matrix(np.eye(5))
    assert _estimate_sigma(K, KG) == 1.


def test_lb_plate_buckling_fsdt():
    """Test plate buckling (FSDT) based on semi-analytical Ritz method.

    Reference: notebook BucklingPlates-FSDT.ipynb from the buckling repository.
    Uses Legendre polynomials as basis functions with Gauss-Legendre quadrature.
    """
    from scipy.special import roots_legendre
    from composites import isotropic_plate
    from buckling.legendre import vecf, vecfxi

    m1 = 20
    m2 = 10
    N = 3 * m1 * m2

    pts1, weights1 = roots_legendre(2 * m1 - 1)
    pts2, weights2 = roots_legendre(2 * m2 - 1)

    E = 200.e9
    nu = 0.3

    # BCs: w simply supported, phi free
    wxit1, wxir1, wxit2, wxir2 = 0, 1, 0, 1
    wetat1, wetar1, wetat2, wetar2 = 0, 1, 0, 1
    xit1, xir1, xit2, xir2 = 1, 1, 1, 1
    etat1, etar1, etat2, etar2 = 1, 1, 1, 1

    a, b, h = 0.3, 0.1, 0.003
    prop = isotropic_plate(thickness=h, E=E, nu=nu)

    Nxxhat = -100.

    Swx = np.zeros(N)
    Swy = np.zeros(N)
    Sphix = np.zeros(N)
    Sphiy = np.zeros(N)
    Sphixx = np.zeros(N)
    Sphixy = np.zeros(N)
    Sphiyx = np.zeros(N)
    Sphiyy = np.zeros(N)

    buff = np.zeros((N, N))
    K = np.zeros((N, N))
    KG = np.zeros((N, N))

    def addouter(matrix, vec1, vec2):
        np.outer(vec1, vec2, out=buff)
        matrix += buff

    for xi, wxi in zip(pts1, weights1):
        wP_xi = vecf(m1, xi, wxit1, wxir1, wxit2, wxir2)
        wPx_xi = vecfxi(m1, xi, wxit1, wxir1, wxit2, wxir2)
        P_xi = vecf(m1, xi, xit1, xir1, xit2, xir2)
        Px_xi = vecfxi(m1, xi, xit1, xir1, xit2, xir2)

        for eta, weta in zip(pts2, weights2):
            wP_eta = vecf(m2, eta, wetat1, wetar1, wetat2, wetar2)
            wPx_eta = vecfxi(m2, eta, wetat1, wetar1, wetat2, wetar2)
            P_eta = vecf(m2, eta, etat1, etar1, etat2, etar2)
            Px_eta = vecfxi(m2, eta, etat1, etar1, etat2, etar2)

            weight = wxi * weta

            Pi, Pj = np.meshgrid(P_xi, P_eta, indexing='ij')
            Sphix[m1*m2:2*m1*m2] = (Pi * Pj).flatten()
            Sphiy[2*m1*m2:] = (Pi * Pj).flatten()

            Pxi, Pj = np.meshgrid(wPx_xi, wP_eta, indexing='ij')
            Swx[:m1*m2] = (Pxi * Pj * (2 / a)).flatten()

            Pxi, Pj = np.meshgrid(Px_xi, P_eta, indexing='ij')
            Sphixx[m1*m2:2*m1*m2] = (Pxi * Pj * (2 / a)).flatten()
            Sphiyx[2*m1*m2:] = (Pxi * Pj * (2 / a)).flatten()

            Pi, Pxj = np.meshgrid(wP_xi, wPx_eta, indexing='ij')
            Swy[:m1*m2] = (Pi * Pxj * (2 / b)).flatten()

            Pi, Pxj = np.meshgrid(P_xi, Px_eta, indexing='ij')
            Sphixy[m1*m2:2*m1*m2] = (Pi * Pxj * (2 / b)).flatten()
            Sphiyy[2*m1*m2:] = (Pi * Pxj * (2 / b)).flatten()

            e1xx = Sphixx
            e1yy = Sphiyy
            e1xy = Sphixy + Sphiyx

            g0yz = Sphiy + Swy
            g0xz = Sphix + Swx

            Mxx = prop.D11 * e1xx + prop.D12 * e1yy + prop.D16 * e1xy
            Myy = prop.D12 * e1xx + prop.D22 * e1yy + prop.D26 * e1xy
            Mxy = prop.D16 * e1xx + prop.D26 * e1yy + prop.D66 * e1xy

            Qy = prop.A44 * g0yz + prop.A45 * g0xz
            Qx = prop.A45 * g0yz + prop.A55 * g0xz

            detJ = a * b / 4

            addouter(K, detJ * weight * Mxx, e1xx)
            addouter(K, detJ * weight * Myy, e1yy)
            addouter(K, detJ * weight * Mxy, e1xy)
            addouter(K, detJ * weight * Qy, g0yz)
            addouter(K, detJ * weight * Qx, g0xz)

            addouter(KG, detJ * weight * Nxxhat * Swx, Swx)

    eigvals, eigvecs = lb(csc_matrix(K), csc_matrix(KG), silent=True)

    D = E * h**3 / (12 * (1 - nu**2))
    Ncr_analytical = 4 * np.pi**2 * D / b**2
    Ncr_computed = eigvals[0] * abs(Nxxhat)

    # FSDT gives slightly lower Ncr than CLPT due to shear deformation
    np.testing.assert_allclose(Ncr_computed, Ncr_analytical, rtol=0.02)
