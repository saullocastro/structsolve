"""Tests for static deflection using structsolve.solve()

Reference: Static-deflection-plate-CLPT.ipynb from the buckling repository.
"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix

from structsolve import solve


def test_static_deflection_plate_clpt():
    """Test static deflection of a simply supported plate under point load (CLPT).

    Uses Legendre polynomials as shape functions with Gauss-Legendre quadrature.
    Compares the center deflection with the Navier analytical solution.
    """
    from scipy.special import roots_legendre
    from composites import isotropic_plate
    from buckling.legendre import vecf, vecfxi, vecfxixi

    m1 = m2 = 20
    N = m1 * m2

    pts1, weights1 = roots_legendre(2 * m1 - 1)
    pts2, weights2 = roots_legendre(2 * m2 - 1)

    E = 200.e9
    nu = 0.3
    a = 3
    b = 7
    h = 0.005

    # Simply supported BCs
    xit1, xir1, xit2, xir2 = 0, 1, 0, 1
    etat1, etar1, etat2, etar2 = 0, 1, 0, 1

    prop = isotropic_plate(thickness=h, E=E, nu=nu)

    K = np.zeros((N, N))
    detJ = a * b / 4

    for xi, wxi in zip(pts1, weights1):
        P_xi = vecf(m1, xi, xit1, xir1, xit2, xir2)
        Px_xi = vecfxi(m1, xi, xit1, xir1, xit2, xir2)
        Pxx_xi = vecfxixi(m1, xi, xit1, xir1, xit2, xir2)

        for eta, weta in zip(pts2, weights2):
            P_eta = vecf(m2, eta, etat1, etar1, etat2, etar2)
            Px_eta = vecfxi(m2, eta, etat1, etar1, etat2, etar2)
            Pxx_eta = vecfxixi(m2, eta, etat1, etar1, etat2, etar2)

            Swxx = (Pxx_xi[:, None] * P_eta[None, :] * (2 / a)**2).ravel()
            Swyy = (P_xi[:, None] * Pxx_eta[None, :] * (2 / b)**2).ravel()
            Swxy = (Px_xi[:, None] * Px_eta[None, :] * (2 / a) * (2 / b)).ravel()

            e1xx = -Swxx
            e1yy = -Swyy
            e1xy = -2 * Swxy

            Mxx = prop.D11 * e1xx + prop.D12 * e1yy + prop.D16 * e1xy
            Myy = prop.D12 * e1xx + prop.D22 * e1yy + prop.D26 * e1xy
            Mxy = prop.D16 * e1xx + prop.D26 * e1yy + prop.D66 * e1xy

            weight = wxi * weta

            K += np.outer(detJ * weight * Mxx, e1xx)
            K += np.outer(detJ * weight * Myy, e1yy)
            K += np.outer(detJ * weight * Mxy, e1xy)

    # Point load at center (xi=0, eta=0 maps to a/2, b/2)
    xi, eta = 0, 0
    P_xi = vecf(m1, xi, xit1, xir1, xit2, xir2)
    P_eta = vecf(m2, eta, etat1, etar1, etat2, etar2)
    Sw = (P_xi[:, None] * P_eta[None, :]).ravel()

    Pforce = 1.0
    Fext = Pforce * Sw

    u = solve(csc_matrix(K), Fext, silent=True)

    # Reconstruct deflection at center
    w_center = Sw @ u

    # Navier analytical solution for SS plate under central point load:
    # w(a/2,b/2) = (4*P)/(pi^4*D*a*b) * sum_m sum_n sin^2(m*pi/2)*sin^2(n*pi/2) / (m^2/a^2 + n^2/b^2)^2
    D = E * h**3 / (12 * (1 - nu**2))
    w_navier = 0.0
    for m in range(1, 80, 2):  # only odd m, n contribute
        for n in range(1, 80, 2):
            w_navier += 1.0 / (m**2 / a**2 + n**2 / b**2)**2

    w_navier *= 4 * Pforce / (np.pi**4 * D * a * b)

    np.testing.assert_allclose(w_center, w_navier, rtol=0.02)
