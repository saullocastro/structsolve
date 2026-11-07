"""Approximation functions and plate properties of the Ritz models of the tests

The Legendre-based approximation functions of Castro and Donadon (2017) and
a NumPy version of ``composites.isotropic_plate``, such that the tests only
depend on NumPy and SciPy.

References
----------
.. [1] Castro, S.G.P. and Donadon, M.V., Assembly of semi-analytical models
   to address linear buckling and vibration of stiffened composite panels
   with debonding defect, Composite Structures, 160, 232-247, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026

"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from numpy.polynomial import legendre


#: Power series coefficients of the cubic Hermite functions, which control
#: the displacement and the rotation at each extremity
_HERMITE = np.array([[0.5, -0.75, 0., 0.25],
                     [0.125, -0.125, -0.125, 0.125],
                     [0.5, 0.75, 0., -0.25],
                     [-0.125, -0.125, 0.125, 0.125]])


@lru_cache(maxsize=None)
def _coefficients(n, d1t, d1r, d2t, d2r, der):
    """Legendre series coefficients, one column per function"""
    C = np.zeros((max(n, 4), max(n, 4)))
    for i, flag in enumerate((d1t, d1r, d2t, d2r)):
        C[:4, i] = flag*legendre.poly2leg(_HERMITE[i])
    for i in range(4, n):
        Pi_2 = np.zeros(i - 1)
        Pi_2[-1] = 1.
        C[:i+1, i] = legendre.legint(Pi_2, m=2, lbnd=-1)
    C = legendre.legder(C[:, :n], m=der, axis=0) if der else C[:, :n]
    C.flags.writeable = False
    return C


def vecf(n, xi, d1t, d1r, d2t, d2r):
    r"""Hierarchical functions of Bardell at the natural coordinate ``xi``

    Returns the values of the ``n`` functions at ``xi``, in `[-1, 1]`. The
    flags ``d1t``, ``d1r``, ``d2t`` and ``d2r``, 1 or 0, keep or remove the
    displacement and the rotation at `\xi=-1` and at `\xi=+1`.

    Notes
    -----
    The first four functions are cubic Hermite functions, each one with a
    displacement or a rotation at one extremity only, multiplied by the
    flags ``d1t``, ``d1r``, ``d2t`` and ``d2r``. The following ones are the
    Legendre polynomials `P_{i-2}` integrated twice from `-1`, which vanish
    with their first derivative at both extremities, as in [1]_.

    References
    ----------
    .. [1] Castro, S.G.P. and Donadon, M.V., Assembly of semi-analytical
       models to address linear buckling and vibration of stiffened
       composite panels with debonding defect, Composite Structures, 160,
       232-247, 2017. https://doi.org/10.1016/j.compstruct.2016.10.026

    """
    return legendre.legval(xi, _coefficients(n, d1t, d1r, d2t, d2r, 0))


def vecf_xi(n, xi, d1t, d1r, d2t, d2r):
    """First derivative of :func:`vecf` with respect to ``xi``"""
    return legendre.legval(xi, _coefficients(n, d1t, d1r, d2t, d2r, 1))


def vecf_xixi(n, xi, d1t, d1r, d2t, d2r):
    """Second derivative of :func:`vecf` with respect to ``xi``"""
    return legendre.legval(xi, _coefficients(n, d1t, d1r, d2t, d2r, 2))


def isotropic_plate(thickness, E, nu):
    r"""Bending and transverse shear stiffnesses of an isotropic plate

    The transverse shear stiffnesses use the shear correction factor 5/6.
    Same as the ``D`` and the transverse shear terms of the ``Laminate``
    returned by ``composites.isotropic_plate``.

    Notes
    -----
    With the thickness `h`, `D_{11} = D_{22} = D = E h^3/(12 (1 - \nu^2))`,
    `D_{12} = \nu D`, `D_{66} = (1 - \nu) D/2`, and `A_{44} = A_{55} = k G
    h`, with `G = E/(2 (1 + \nu))` and `k = 5/6`.

    """
    D = E*thickness**3/(12*(1 - nu**2))
    G = E/(2*(1 + nu))
    return SimpleNamespace(
        D11=D, D12=nu*D, D16=0., D22=D, D26=0., D66=(1 - nu)*D/2,
        A44=5/6*G*thickness, A45=0., A55=5/6*G*thickness)
