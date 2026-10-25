r"""Arc-length methods on the modal model of a low sinusoidal arch

Fung and Kaplan [1]_ expand the initial and the deflected center lines of a
low arch hinged at both ends in the sine series `y_0 = \sum a_m L \sin(m \pi
x/L)` and `y = \sum b_m L \sin(m \pi x/L)`, and give the equations of
equilibrium in terms of the nondimensional rises `\lambda_m` and `B_m` of
the harmonics and of the nondimensional load `R`, their Eqs. (14) and (15).
For a sinusoidal arch, `\lambda_1 > 0`, under the sinusoidal load `q = q_0
\sin(\pi x/L)`, positive downward, without initial thrust, `S = 0`, and
with rigid supports, `\beta = 1`, their Eqs. (15) with the first two
harmonics are

.. math::

    B_1 \left(s + 1\right) = \lambda_1 - R, \qquad
    B_2 \left(s + 4\right) = 4 \lambda_2, \qquad
    s = B_1^2 + 4 B_2^2 - \lambda_1^2 - 4 \lambda_2^2

which are their Eqs. (22) with `\lambda_2 = 0`; `\lambda_2 \neq 0` is an
imperfection of the second harmonic. The degrees of freedom are `\{c\} =
\{\lambda_1 - B_1, B_2 - \lambda_2\}`, null in the unloaded state, the first
one positive in the direction of the load. The second equation is
multiplied by `m^2 = 4`, such that the tangent stiffness matrix is
symmetric, and the load factor is `R/F`, with `\{F_{ext}\} = \{F, 0\}`.

The symmetric path, `B_2 = 0`, is their Eq. (23),

.. math::

    B_1^3 - B_1 (\lambda_1^2 - 1) = \lambda_1 - R

with the limit points of their Eq. (25), `B_1 = \pm\sqrt{(\lambda_1^2 -
1)/3}` and `R = \lambda_1 \pm \sqrt{4 (\lambda_1^2 - 1)^3/27}`, and null
loads at `B_1 = \lambda_1` and `B_1 = (-\lambda_1 \pm \sqrt{\lambda_1^2 -
4})/2`. The asymmetric path, `B_2 \neq 0`, their Eq. (27), `B_1 = (R -
\lambda_1)/3`, branches from the symmetric one at `B_1 = \pm\sqrt{\lambda_1^2
- 4}` and `R = \lambda_1 \pm 3\sqrt{\lambda_1^2 - 4}`. The arch buckles at
the upper limit point when `\lambda_1^2 < 5.5` and at the bifurcation point
otherwise, their Eqs. (30) and (31).

References
----------
.. [1] Fung, Y. C. and Kaplan, A., Buckling of low arches or curved beams of
   small curvature, NACA Technical Note 2840, National Advisory Committee
   for Aeronautics, Washington, 1952.
   https://ntrs.nasa.gov/citations/19930083579

"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from analytic_problems import Problem

METHODS = ['arc_length_riks', 'arc_length_crisfield']
CONVERGENCE = ['crisfield', 'relative']

#: maximum residual of Eqs. (22) or (15) at the converged points, the default
#: relTOL = 1e-9 times the references of the convergence criteria, which are
#: of the order of the load ``F``, about 10
EQ_ATOL = 1e-7


def limit_load(lam1):
    """Upper limit load of the symmetric path, Eq. (25) of Fung and Kaplan"""
    return lam1 + np.sqrt(4/27*(lam1**2 - 1)**3)


def arch(lam1, lam2=0., F=None):
    """Modal model of Fung and Kaplan with the harmonics 1 and 2

    ``F`` is the reference load, by default 1.5 times the upper limit load,
    such that the load factor reaches 1.0 on the inverted configuration
    only.

    """
    F = 1.5*limit_load(lam1) if F is None else F

    def B(c):
        return lam1 - c[0], lam2 + c[1]

    def fint(c):
        B1, B2 = B(c)
        s = B1**2 + 4*B2**2 - lam1**2 - 4*lam2**2
        return np.array([lam1 - B1*(s + 1), 4*(B2*(s + 4) - 4*lam2)])

    def kT(c):
        B1, B2 = B(c)
        s = B1**2 + 4*B2**2 - lam1**2 - 4*lam2**2
        return np.array([[s + 1 + 2*B1**2, -8*B1*B2],
                         [-8*B1*B2, 4*(s + 4) + 32*B2**2]])

    kC = kT(np.zeros(2))
    p = Problem(np.array([F, 0.]), fint, kC, lambda c: kT(c) - kC)
    p.lam1, p.lam2, p.F, p.kT = lam1, lam2, F, kT
    return p


def path(p):
    """Load ``R``, ``B1`` and ``B2`` at the converged points"""
    cs = np.asarray(p.an.cs)
    R = p.F*np.asarray(p.an.increments)
    return R, p.lam1 - cs[:, 0], p.lam2 + cs[:, 1]


def check_symmetric_path(p):
    """Eq. (23), both limit points, Eq. (25), and the null loads

    The exact limit point lies between the neighbours of the converged point
    of extreme load, at a distance ``d <= h`` of it in ``B1``, and Eq. (23)
    gives exactly ``|R - R_lim| = 3 B1_lim d**2 + d**3`` with ``B1_lim > 0``
    the modulus of ``B1`` at the limit points, hence the tolerance ``3 B1_lim
    h**2 + h**3`` justified by the step size.

    Returns the index of the converged point of maximum load before the
    snap-through.

    """
    lam1 = p.lam1
    R, B1, B2 = path(p)
    assert p.an.increments[-1] == 1.0
    assert np.all(B2 == 0.)
    # continuous path, without doubling back
    assert np.all(np.diff(np.r_[lam1, B1]) < 0)
    np.testing.assert_allclose(B1**3 - B1*(lam1**2 - 1), lam1 - R, rtol=0,
                               atol=EQ_ATOL)
    B1_lim = np.sqrt((lam1**2 - 1)/3)
    kmin = np.argmin(R)
    kmax = np.argmax(R[:kmin])
    for k, sign in [(kmax, 1), (kmin, -1)]:
        assert B1[k + 1] < sign*B1_lim < B1[k - 1]
        h = max(B1[k - 1] - B1[k], B1[k] - B1[k + 1])
        R_lim = lam1 + sign*np.sqrt(4/27*(lam1**2 - 1)**3)
        assert -EQ_ATOL <= sign*(R_lim - R[k]) <= 3*B1_lim*h**2 + h**3
    # null loads between the limit points and after the lower one
    i = np.flatnonzero(R[:-1]*R[1:] < 0)
    assert len(i) == 2
    roots = (-lam1 + np.array([1, -1])*np.sqrt(lam1**2 - 4))/2
    assert np.all((B1[i + 1] < roots) & (roots < B1[i]))
    assert kmax <= i[0] < kmin <= i[1]
    return kmax


@pytest.mark.parametrize('lam1', [2.2, 2.5])
@pytest.mark.parametrize('convergence', CONVERGENCE)
@pytest.mark.parametrize('method', METHODS)
def test_fung_kaplan_symmetric_path(method, convergence, lam1):
    r"""Symmetric path through both limit points and the null loads

    The arc length is limited to 0.05, about the increment of ``B1`` of each
    step, hence a tolerance of the limit loads of about 1e-2. With
    `\lambda_1^2 = 4.84 < 5.5`, the stiffness of the second harmonic,
    ``kT[1, 1]``, is positive up to the limit point: no bifurcation before
    it. With `\lambda_1^2 = 6.25`,
    it changes sign before the limit point, at the bifurcation point `B_1 =
    1.5`, `R = 7.0`, Eq. (31); the perfect arch remains on the symmetric
    path.

    """
    p = arch(lam1)
    p.analysis(NL_method=method, convergence=convergence, initialInc=0.05,
               maxInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    kmax = check_symmetric_path(p)
    R, B1, _ = path(p)
    k22 = np.array([p.kT(c)[1, 1] for c in p.an.cs])
    j = np.flatnonzero(k22 < 0)[0] - 1
    B1_bif = np.sqrt(lam1**2 - 4)
    assert B1[j + 1] < B1_bif < B1[j]
    if lam1**2 < 5.5:
        assert j > kmax
    else:
        assert j < kmax
        R_bif = lam1 + 3*B1_bif
        assert R[j] < R_bif < R[j + 1]
        assert R_bif == 7.0


@pytest.mark.parametrize('lam1', [2.2, 2.5])
@pytest.mark.parametrize('convergence', CONVERGENCE)
@pytest.mark.parametrize('method', METHODS)
def test_fung_kaplan_default_settings(method, convergence, lam1):
    """The symmetric path at the default step sizes and relTOL = 1e-9

    The steps grow up to ``maxInc = 1``, about 0.7 in ``B1``, and the
    tolerance of the limit loads, of the same step-size bound, is much
    looser. The path reaches the load factor 1.0 on the inverted
    configuration, crossing zero twice.

    """
    p = arch(lam1)
    p.analysis(NL_method=method, convergence=convergence)
    assert p.an.relTOL == 1e-9
    p.an.static(NLgeom=True, silent=True)
    check_symmetric_path(p)


def imperfect_extremes(lam1, lam2):
    r"""Maximum and minimum loads of the path of the imperfect arch

    With `B_2 \neq 0`, Eq. (15) of Fung and Kaplan gives `B_1^2 = \lambda_1^2
    + 4\lambda_2^2 - 4 + 4\lambda_2/B_2 - 4B_2^2` and `R = \lambda_1 + B_1
    (3 - 4\lambda_2/B_2)`, with `B_1 > 0` before the snap-through and `B_1 <
    0` after it.

    """
    def B1R(B2, sign):
        B1 = sign*np.sqrt(lam1**2 + 4*lam2**2 - 4 + 4*lam2/B2 - 4*B2**2)
        return B1, lam1 + B1*(3 - 4*lam2/B2)

    out = []
    for sign in [1, -1]:
        res = minimize_scalar(lambda B2: -sign*B1R(B2, sign)[1],
                              bounds=(0.01, 0.7), method='bounded',
                              options=dict(xatol=1e-12))
        out.append((res.x, B1R(res.x, sign)[1], lambda B2, s=sign:
                    B1R(B2, s)[1]))
    return out


@pytest.mark.parametrize('convergence', CONVERGENCE)
@pytest.mark.parametrize('method', METHODS)
def test_fung_kaplan_asymmetric_path(method, convergence):
    r"""Imperfect arch, `\lambda_1 = 2.5`, `\lambda_2 = 10^{-3}`

    The path follows the asymmetric path, Eq. (27), with a maximum load
    below the bifurcation load of the perfect arch, `R = 7.0`, Eq. (31).
    The extreme loads are compared with those of Eq. (15) as a function of
    ``B2``, ``R(B2)``: the exact extreme lies between the neighbours of the
    converged point of extreme load, at a distance ``d <= h`` of it in
    ``B2``, and ``|R - R_ext| <= max|R''| h**2/2`` over this interval.

    """
    lam1, lam2 = 2.5, 1e-3
    p = arch(lam1, lam2, F=10.5)
    p.analysis(NL_method=method, convergence=convergence, initialInc=0.05,
               maxInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    R, B1, B2 = path(p)
    assert p.an.increments[-1] == 1.0
    assert B1[-1] < -2.
    s = B1**2 + 4*B2**2 - lam1**2 - 4*lam2**2
    np.testing.assert_allclose(B1*(s + 1), lam1 - R, rtol=0, atol=EQ_ATOL)
    np.testing.assert_allclose(B2*(s + 4), 4*lam2, rtol=0, atol=EQ_ATOL)
    # the second harmonic reaches its maximum on the asymmetric path,
    # sqrt(lam1**2 - 4)/2 at B1 = 0, Eq. (27)
    assert abs(B2.max() - 0.75) < 0.01
    kmin = np.argmin(R)
    kmax = np.argmax(R[:kmin])
    (B2max, Rmax, Rfun_max), (B2min, Rmin, Rfun_min) = imperfect_extremes(
        lam1, lam2)
    assert Rmax < 7.0 and Rmin > -2.
    for k, B2_ext, R_ext, Rfun, sign in [(kmax, B2max, Rmax, Rfun_max, 1),
                                         (kmin, B2min, Rmin, Rfun_min, -1)]:
        lo, hi = sorted([B2[k - 1], B2[k + 1]])
        assert lo < B2_ext < hi
        h = max(abs(B2[k] - B2[k - 1]), abs(B2[k + 1] - B2[k]))
        x = np.linspace(lo, hi, 201)
        d = 1e-5
        R2 = np.abs(Rfun(x + d) - 2*Rfun(x) + Rfun(x - d))/d**2
        assert -EQ_ATOL <= sign*(R_ext - R[k]) <= 0.5*R2.max()*h**2
