"""Tests for structsolve.linear_buckling: lb()"""
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
