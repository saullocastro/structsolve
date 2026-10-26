"""Tests for structsolve.linear_buckling: lb()"""
import os

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from structsolve import lb

from ritz import isotropic_plate, vecf, vecf_xi


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
    lowest ones. eigsh is always used first, ``max_dense_size=0`` disables
    the dense fallback.
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


@pytest.mark.parametrize('max_dense_size', [0, 2000])
@pytest.mark.parametrize('name, ref', SAVED_MATRICES)
def test_lb_saved_matrices_sparse_solver(name, ref, max_dense_size):
    """Regression test: ARPACK linked against Intel MKL 2024.2.0 to 2025.0.0
    returned wrong load multipliers at random, e.g. with the Anaconda builds
    of SciPy, whenever ncv > 32. eigsh() with ncv <= 32 returns the dense
    load multipliers, without the dense fallback (max_dense_size=0) and
    without any DenseFallbackWarning"""
    import warnings
    from structsolve.linear_buckling import DenseFallbackWarning

    K, KG = load_saved_matrices(name)
    eig_dense, _ = lb(K, KG, silent=True, sparse_solver=False)
    np.testing.assert_allclose(eig_dense[:3], ref, rtol=1e-5)
    for trial in range(3):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DenseFallbackWarning)
            eigvals, eigvecs = lb(K, KG, silent=True,
                                  max_dense_size=max_dense_size)
        assert eigvals.shape == (25,)
        assert eigvecs.shape == (K.shape[0], 25)
        np.testing.assert_allclose(eigvals, eig_dense[:25], rtol=1e-6)
        assert np.all(eigvals[:-1] <= eigvals[1:])
        # eigsh() is less accurate than the dense solution, but far below
        # the default check_rtol=1e-3
        assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-4)


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


def test_lb_repeated_load_multipliers():
    """Two equal critical load multipliers: the sparse and the dense solvers
    agree on the load multipliers and on the eigenspace of the pair, but
    not necessarily on the individual eigenvectors, any basis of the
    eigenspace being valid. With num_eigvalues=1 only one of them is
    returned"""
    from scipy.linalg import subspace_angles

    K, KG = load_saved_matrices('cylinder_5panels')
    eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=4)
    eig_dense, eigvecs_dense = lb(K, KG, silent=True, sparse_solver=False,
                                  num_eigvalues=4)
    np.testing.assert_allclose(eigvals, eig_dense[:4], rtol=1e-7)
    np.testing.assert_allclose(eigvals[0], eigvals[1], rtol=1e-7)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-3)
    # same eigenspace of the pair
    assert subspace_angles(eigvecs[:, :2], eigvecs_dense[:, :2]).max() < 1e-4
    # the individual eigenvectors of both solvers are equally valid
    assert np.all(relative_residuals(K, KG, eig_dense[:2],
                                     eigvecs_dense[:, :2]) < 1e-8)

    eigvals1, eigvecs1 = lb(K, KG, silent=True, num_eigvalues=1)
    assert eigvals1.shape == (1,)
    assert eigvecs1.shape == (K.shape[0], 1)
    np.testing.assert_allclose(eigvals1[0], eigvals[0], rtol=1e-7)


def test_lb_sparse_raises_when_verification_fails(monkeypatch):
    """A RuntimeError is raised instead of returning wrong eigenpairs"""
    import structsolve.linear_buckling as linear_buckling

    K, KG = load_saved_matrices('plate_ssss_Nxx')

    def wrong_eigsh(K, KG, k, tol, silent, safety=10., mu_max=False):
        rng = np.random.RandomState(0)
        return -rng.rand(k), rng.randn(K.shape[0], k)

    monkeypatch.setattr(linear_buckling, '_eigsh_cayley', wrong_eigsh)
    with pytest.raises(RuntimeError, match='verification') as e:
        lb(K, KG, silent=True, max_dense_size=0)
    # every attempted solver is listed
    assert str(e.value).count('relative residual') == 2
    assert 'dense fallback disabled with max_dense_size=0' in str(e.value)
    # the condensed dense solution is the last resort, with a warning
    with pytest.warns(linear_buckling.DenseFallbackWarning,
                      match='relative residual'):
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


def mixed_sign_problem(lambdas, num_null=0):
    """Diagonal ``K`` and ``KG`` with the load multipliers ``lambdas``,
    coupled by a sparse congruence, and ``num_null`` dofs where ``KG`` is
    null, e.g. in-plane dofs"""
    from scipy.sparse import diags, identity, random as sprandom, tril

    lambdas = np.concatenate([lambdas, np.full(num_null, np.inf)])
    n = lambdas.size
    k = np.linspace(1., 3., n)
    T = (identity(n) + 0.1*tril(sprandom(n, n, density=2/n, random_state=1),
                                -1)).tocsc()
    K = (T.T @ diags(k) @ T).tocsr()
    KG = (T.T @ diags(-k/lambdas) @ T).tocsr()
    return K, KG


@pytest.mark.parametrize('max_dense_size', [0, 2000])
def test_lb_negative_multipliers_closest_to_zero(max_dense_size):
    """After the positive load multipliers, the negative ones closest to
    zero, i.e. the critical load multipliers of the reversed load, for the
    sparse and the dense solvers"""
    positive = np.linspace(1., 20., 5)
    negative = -np.linspace(2., 40., 35)
    K, KG = mixed_sign_problem(np.concatenate([positive, negative]),
                               num_null=10)
    eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=10,
                          max_dense_size=max_dense_size)
    np.testing.assert_allclose(eigvals[:5], positive, rtol=1e-8)
    np.testing.assert_allclose(eigvals[5:], negative[:5], rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)
    eig_dense, _ = lb(K, KG, silent=True, sparse_solver=False)
    np.testing.assert_allclose(eig_dense[:10], eigvals, rtol=1e-8)
    # the infinite load multipliers of the null rows of KG come last
    assert np.all(np.abs(eig_dense[-10:]) > 1e8)
    # the same order with the solvers of unsymmetric matrices
    eig_gen, _ = lb(K, KG, silent=True, num_eigvalues=10, symmetric=False,
                    max_dense_size=max_dense_size)
    pos, neg = eig_gen[eig_gen > 0], eig_gen[eig_gen < 0]
    np.testing.assert_allclose(pos, positive[:pos.size], rtol=1e-8)
    np.testing.assert_allclose(neg, negative[:neg.size], rtol=1e-8)
    np.testing.assert_array_equal(eig_gen, np.concatenate((pos, neg)))


def test_lb_tensile_reference_load():
    """No positive load multiplier, e.g. a tensile reference load: the
    sparse solver returns the negative load multipliers closest to zero,
    without the dense fallback"""
    import warnings
    from structsolve.linear_buckling import DenseFallbackWarning

    negative = -np.linspace(2., 40., 40)
    K, KG = mixed_sign_problem(negative, num_null=10)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=10,
                              max_dense_size=0)
    np.testing.assert_allclose(eigvals, negative[:10], rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)
    # the reversed load
    eigvals_rev, _ = lb(K, -KG, silent=True, num_eigvalues=10,
                        max_dense_size=0)
    np.testing.assert_allclose(eigvals_rev, -negative[:10], rtol=1e-8)


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

    Uses Legendre polynomials as basis functions with Gauss-Legendre quadrature.
    Compares the critical load with the classical solution ``4*pi**2*D/b**2``,
    with a relative tolerance of 2 %.
    """
    from scipy.special import roots_legendre

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

    Sw_x = np.zeros(N)
    Sw_y = np.zeros(N)
    Sphix = np.zeros(N)
    Sphiy = np.zeros(N)
    Sphix_x = np.zeros(N)
    Sphix_y = np.zeros(N)
    Sphiy_x = np.zeros(N)
    Sphiy_y = np.zeros(N)

    buff = np.zeros((N, N))
    K = np.zeros((N, N))
    KG = np.zeros((N, N))

    def addouter(matrix, vec1, vec2):
        np.outer(vec1, vec2, out=buff)
        matrix += buff

    for xi, wxi in zip(pts1, weights1):
        wPxi = vecf(m1, xi, wxit1, wxir1, wxit2, wxir2)
        wPxi_xi = vecf_xi(m1, xi, wxit1, wxir1, wxit2, wxir2)
        Pxi = vecf(m1, xi, xit1, xir1, xit2, xir2)
        Pxi_xi = vecf_xi(m1, xi, xit1, xir1, xit2, xir2)

        for eta, weta in zip(pts2, weights2):
            wPeta = vecf(m2, eta, wetat1, wetar1, wetat2, wetar2)
            wPeta_eta = vecf_xi(m2, eta, wetat1, wetar1, wetat2, wetar2)
            Peta = vecf(m2, eta, etat1, etar1, etat2, etar2)
            Peta_eta = vecf_xi(m2, eta, etat1, etar1, etat2, etar2)

            weight = wxi * weta

            Pi, Pj = np.meshgrid(Pxi, Peta, indexing='ij')
            Sphix[m1*m2:2*m1*m2] = (Pi * Pj).flatten()
            Sphiy[2*m1*m2:] = (Pi * Pj).flatten()

            Pi_xi, Pj = np.meshgrid(wPxi_xi, wPeta, indexing='ij')
            Sw_x[:m1*m2] = (Pi_xi * Pj * (2 / a)).flatten()

            Pi_xi, Pj = np.meshgrid(Pxi_xi, Peta, indexing='ij')
            Sphix_x[m1*m2:2*m1*m2] = (Pi_xi * Pj * (2 / a)).flatten()
            Sphiy_x[2*m1*m2:] = (Pi_xi * Pj * (2 / a)).flatten()

            Pi, Pj_eta = np.meshgrid(wPxi, wPeta_eta, indexing='ij')
            Sw_y[:m1*m2] = (Pi * Pj_eta * (2 / b)).flatten()

            Pi, Pj_eta = np.meshgrid(Pxi, Peta_eta, indexing='ij')
            Sphix_y[m1*m2:2*m1*m2] = (Pi * Pj_eta * (2 / b)).flatten()
            Sphiy_y[2*m1*m2:] = (Pi * Pj_eta * (2 / b)).flatten()

            e1xx = Sphix_x
            e1yy = Sphiy_y
            e1xy = Sphix_y + Sphiy_x

            g0yz = Sphiy + Sw_y
            g0xz = Sphix + Sw_x

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

            addouter(KG, detJ * weight * Nxxhat * Sw_x, Sw_x)

    eigvals, eigvecs = lb(csc_matrix(K), csc_matrix(KG), silent=True)

    D = E * h**3 / (12 * (1 - nu**2))
    Ncr_analytical = 4 * np.pi**2 * D / b**2
    Ncr_computed = eigvals[0] * abs(Nxxhat)

    # FSDT gives slightly lower Ncr than CLPT due to shear deformation
    np.testing.assert_allclose(Ncr_computed, Ncr_analytical, rtol=0.02)


@pytest.mark.parametrize('ratio', [1., 0.1, 0.01, 0.001])
def test_lb_sparse_mixed_spectrum_coupled(ratio):
    """Harder variants of test_lb_sparse_shift_mixed_spectrum, as the study
    of the sparse solvers: negative load multipliers down to -ratio, which
    made eigsh() without a shift (which='SA') slow or not converge, coupled
    by a sparse congruence such that the problem is not diagonal. The
    sparse solver, without dense fallback, returns the critical load
    multipliers 1.0, 1.0, 1.1, 1.2"""
    from scipy.sparse import diags, identity, random as sprandom, tril

    n = 600
    k = np.linspace(1e6, 5e6, n)
    lambdas = np.concatenate([[1.0, 1.0, 1.1, 1.2],
                              np.linspace(1.5, 4., n//2 - 4),
                              -ratio*np.linspace(1., 4., n//2)])
    T = (identity(n) + 0.1*tril(sprandom(n, n, density=2/n, random_state=1),
                                -1)).tocsc()
    K = (T.T @ diags(k) @ T).tocsr()
    KG = (T.T @ diags(-k/lambdas) @ T).tocsr()
    eigvals, eigvecs = lb(K, KG, silent=True, max_dense_size=0)
    np.testing.assert_allclose(eigvals[:4], [1.0, 1.0, 1.1, 1.2], rtol=1e-8)
    np.testing.assert_allclose(eigvals, np.sort(lambdas[lambdas > 0])[:25],
                               rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)


def test_lb_reference_load():
    """The load multipliers multiply the reference load of KG: a reference
    load twice as large halves them, a constant preload belongs to K, and
    the reversed, tensile, reference load gives the negative load
    multipliers closest to zero, without the dense fallback. Plate of
    panels under a uniaxial compression, saved in tests/data"""
    import warnings
    from structsolve.linear_buckling import DenseFallbackWarning

    K, KG = load_saved_matrices('plate_ssss_Nxx')
    ref = np.array([157.5386, 226.3470, 232.5623])
    eigvals, _ = lb(K, KG, silent=True, num_eigvalues=5)
    np.testing.assert_allclose(eigvals[:3], ref, rtol=1e-5)
    lambda_1 = eigvals[0]

    # a reference load twice as large
    eigvals2, _ = lb(K, 2*KG, silent=True, num_eigvalues=5)
    np.testing.assert_allclose(eigvals2, eigvals/2, rtol=1e-8)

    # a constant preload of 0.4 lambda_1 times the reference load
    eigvals_pre, _ = lb(K + 0.4*lambda_1*KG, KG, silent=True,
                        num_eigvalues=5)
    np.testing.assert_allclose(eigvals_pre[0], 0.6*lambda_1, rtol=1e-8)

    # the reversed reference load: no positive load multiplier
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals_rev, eigvecs_rev = lb(K, -KG, silent=True, num_eigvalues=5)
    np.testing.assert_allclose(eigvals_rev, -eigvals, rtol=1e-8)
    assert np.all(relative_residuals(K, -KG, eigvals_rev, eigvecs_rev)
                  < 1e-3)


def penalty_chains(n=20, kp=1e8):
    """Two chains of ``n`` unit springs, each fixed at its outer end,
    connected by a penalty spring ``kp``, which sets ``||K||_F``, and
    ``KG = -I``: the load multipliers are the eigenvalues of ``K``"""
    K = 2*np.eye(2*n) - np.eye(2*n, k=1) - np.eye(2*n, k=-1)
    K[n - 1, n] = K[n, n - 1] = -kp
    K[n - 1, n - 1] = K[n, n] = 1 + kp
    return csc_matrix(K), csc_matrix(-np.eye(2*n))


def test_lb_check_eigenpairs_penalty():
    """Regression test: with a penalty spring setting ``||K||_F``, load
    multipliers 2 % too low, relative residuals of 1 %, are rejected. A
    floor of the denominator of 1e-8 times the matrix-norm scale
    ``(||K||_F + |lambda| ||KG||_F) ||u||``, about 170 times
    ``||K u|| + |lambda| ||KG u||`` here, accepted them"""
    from structsolve.linear_buckling import (_check_eigenpairs,
                                             _relative_residuals)

    K, KG = penalty_chains()
    lam, u = np.linalg.eigh(K.toarray())
    lam, u = lam[:4], u[:, :4]
    assert _check_eigenpairs(K, KG, lam, u, 1e-3) is None
    res = _relative_residuals(K, KG, 0.98*lam, u)
    np.testing.assert_allclose(res, 0.02/1.98, rtol=1e-3)
    error = _check_eigenpairs(K, KG, 0.98*lam, u, 1e-3)
    assert 'relative residual' in error


def test_lb_penalty_connections():
    """Regression test: blade-stiffened plate of panels with penalty
    connections, saved in tests/data, whose stiff penalty springs set
    ``||K||_F = 5.1e12``. The lowest load multiplier is that certified by
    the inertia of ``K + s KG``. eigsh() returned eigenpairs with relative
    residuals of 1 to 2 % and a lowest load multiplier 0.08 % too low,
    which a floor of the denominator of the relative residual, 1e-8 times
    the matrix-norm scale ``(||K||_F + |lambda| ||KG||_F) ||u||``, accepted;
    they fail the verification, and the dense fallback is used

    The model of panels for Figure 7 (Section 3.2.2) of Stamatelos and
    Labeas (2023): a square plate, ``a = b = 0.5`` m, of 11 ``[0_3/90_3]s``
    skin strips, ply thickness 0.134 mm, with 10 ``[0/90]s`` blades of
    height 2 mm, every domain a ``plate_clpt_donnell`` with 12 x 6 (skin)
    and 12 x 4 (blades) terms, connected by penalty springs 100 times those
    of ``calc_kt_kr``, under a uniform compression of 1 N/m.

    Reference: Stamatelos, D.G. and Labeas, G.N., Buckling analysis of
    laminated stiffened plates with material anisotropy using the
    Rayleigh-Ritz approach, Computation, 11(6), 110, 2023,
    doi:10.3390/computation11060110.

    """
    import warnings
    from structsolve.linear_buckling import (DenseFallbackWarning,
                                             _negative_pivots)

    K, KG = load_saved_matrices('blade_stiffened_penalty')
    lambda_1 = 2583.916728
    # one load multiplier between lambda_1 (1 -/+ 1e-6)
    assert _negative_pivots(K + lambda_1*(1 - 1e-6)*KG) == 0
    assert _negative_pivots(K + lambda_1*(1 + 1e-6)*KG) == 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DenseFallbackWarning)
        eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=4)
    np.testing.assert_allclose(eigvals[0], lambda_1, rtol=1e-6)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-3)
