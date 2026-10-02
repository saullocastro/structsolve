r"""Follower forces and unsymmetric matrices in :func:`freq` and the static
analyses

Ziegler's double pendulum: two rigid bars of length ``l``, rotational springs
``k`` at the base and at the middle hinge, masses ``2m`` at the middle hinge
and ``m`` at the tip, and a follower force ``P`` at the tip, tangential to the
upper bar. With the absolute rotations of the bars as degrees of freedom:

.. math::

    [M] = m l^2 \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
    [K(P)] = \begin{bmatrix} 2k - P l & -k + P l \\ -k & k \end{bmatrix}

The stiffness matrix is unsymmetric and never singular, ``det K = k**2``, so
the static (divergence) criterion of :func:`lb` finds no critical load, and
the system loses stability by flutter at ``P_f = (7/2 - sqrt(2)) k/l``, where
two natural frequencies coalesce (Ziegler, 1952).
"""
import warnings

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

from structsolve import Analysis, freq, lb, solve, static

K_SPRING = 3.
L_BAR = 0.5
MASS = 2.
P_FLUTTER = (3.5 - np.sqrt(2))*K_SPRING/L_BAR


def ziegler(P, extra=0):
    r"""``K(P)`` and ``M``, with ``extra`` decoupled unit oscillators added
    such that the sparse solvers, limited to ``k < size - 1``, can be used"""
    k, l, m = K_SPRING, L_BAR, MASS
    K = np.array([[2*k - P*l, -k + P*l], [-k, k]])
    M = m*l**2*np.array([[3., 1.], [1., 1.]])
    if extra:
        rng = np.random.default_rng(0)
        K = block_diag(K, np.diag(1e3 + rng.random(extra)))
        M = block_diag(M, np.eye(extra))
    return K, M


def max_imag(lambda2):
    return np.max(np.abs(np.imag(lambda2)))


@pytest.mark.parametrize('sparse_solver,extra', [(False, 0), (True, 10)])
def test_freq_ziegler_flutter(sparse_solver, extra):
    r"""The general solver of :func:`freq` gives real frequencies below the
    flutter load and complex ones above it; the flutter load found by
    bisection is the analytical one"""
    kw = dict(silent=True, sparse_solver=sparse_solver, num_eigvalues=4,
              sort=False)
    below = freq(*map(csr_matrix, ziegler(0.9*P_FLUTTER, extra)), **kw)[0]
    above = freq(*map(csr_matrix, ziegler(1.1*P_FLUTTER, extra)), **kw)[0]
    assert max_imag(below) <= 1e-8*np.abs(below).max()
    assert max_imag(above) > 1e-3*np.abs(above).max()
    a, b = 0.9*P_FLUTTER, 1.1*P_FLUTTER
    for _ in range(40):
        c = 0.5*(a + b)
        lam2 = freq(*map(csr_matrix, ziegler(c, extra)), **kw)[0]
        if max_imag(lam2) > 1e-6*np.abs(lam2).max():
            b = c
        else:
            a = c
    assert np.isclose(0.5*(a + b), P_FLUTTER, rtol=1e-4)


def test_freq_ziegler_frequencies():
    r"""Below the flutter load the frequencies are the roots of
    ``det(K - omega**2 M) = 0``"""
    P = 0.5*P_FLUTTER
    K, M = ziegler(P)
    lam2 = freq(csr_matrix(K), csr_matrix(M), silent=True,
                sparse_solver=False, sort=False)[0]
    # det(K - w2 M) = a w2**2 + b w2 + c
    a = np.linalg.det(M)
    b = -(K[0, 0]*M[1, 1] + K[1, 1]*M[0, 0] - K[0, 1]*M[1, 0]
          - K[1, 0]*M[0, 1])
    c = np.linalg.det(K)
    w2 = np.sort(np.roots([a, b, c]).real)
    assert np.allclose(np.sort((-lam2).real), w2, rtol=1e-10)


def test_lb_ziegler_finds_no_divergence():
    r"""``K(P) = K0 + P KF`` is never singular: the static criterion of
    :func:`lb` finds no load multiplier, only the kinetic one of
    :func:`freq` detects the instability"""
    K0, _ = ziegler(0.)
    K1, _ = ziegler(1.)
    KF = K1 - K0
    # K0^-1 KF is nilpotent: the eigenvalues mu of KF u = mu K0 u are zero,
    # a defective pair that the round-off splits into +-sqrt(eps), i.e. load
    # multipliers that are infinite or 1e8 times the flutter load
    for sparse_solver in (False, True):
        try:
            eigvals = lb(csr_matrix(K0), csr_matrix(KF), silent=True,
                         sparse_solver=sparse_solver)[0]
        except RuntimeError:
            continue
        finite = np.isfinite(eigvals)
        assert np.all(np.abs(eigvals[finite]) > 1e6*P_FLUTTER)


def test_freq_symmetric_option():
    r"""``symmetric=True`` (eigsh/eigh) against the general solvers"""
    rng = np.random.default_rng(1)
    n = 30
    X = rng.standard_normal((n, n))
    K = csr_matrix(X @ X.T + n*np.eye(n))
    M = csr_matrix(np.diag(1 + rng.random(n)))
    for sparse_solver in (True, False):
        l_gen = freq(K, M, silent=True, sparse_solver=sparse_solver,
                     num_eigvalues=6)[0]
        l_sym, v_sym = freq(K, M, silent=True, sparse_solver=sparse_solver,
                            num_eigvalues=6, symmetric=True)
        assert np.isrealobj(l_sym)
        w_gen = np.sort((-l_gen).real)[:5]
        w_sym = np.sort(-l_sym)[:5]
        assert np.allclose(w_gen, w_sym, rtol=1e-8)
        l_auto = freq(K, M, silent=True, sparse_solver=sparse_solver,
                      num_eigvalues=6, symmetric=None)[0]
        assert np.isrealobj(l_auto)


def test_freq_unsymmetric_residuals_are_checked():
    r"""A residual check that cannot pass issues a warning"""
    K, M = ziegler(0.5*P_FLUTTER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        freq(csr_matrix(K), csr_matrix(M), silent=False, sparse_solver=False,
             check_rtol=-1.)
    # structsolve.logger.warn prints and may also use the warnings module
    freq(csr_matrix(K), csr_matrix(M), silent=True, sparse_solver=False,
         check_rtol=None)


def test_solve_unsymmetric_with_null_row_and_column():
    r"""Null rows and columns are removed only together, an equation whose
    row is not null is kept"""
    A = np.array([[4., 1., 0., 0.],
                  [2., 3., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 5.]])
    b = np.array([1., 2., 0., 10.])
    x = solve(csr_matrix(A), b, silent=True)
    assert np.allclose(x, [0.1, 0.6, 0., 2.])
    # a row that is not null with a null column: singular, not dropped
    B = A.copy()
    B[2, 0] = 1.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = solve(csr_matrix(B), b, silent=True)
    assert not np.all(np.isfinite(x)) or not np.allclose(B @ x, b)


def test_static_unsymmetric():
    r"""``static(K0 + KF, F0)``, the linear static solution of a follower
    load affine in ``c``, ``F(c) = F0 - KF c``"""
    K0, _ = ziegler(0.)
    P = 0.4*P_FLUTTER
    K, _ = ziegler(P)
    F0 = np.array([0., 1.])
    increments, cs = static(csr_matrix(K), F0, silent=True)
    assert np.allclose(K @ cs[0], F0)


class LinearFollower(object):
    r"""Geometrically linear structure under a configuration-dependent load
    ``F(c) = F0 + Q c``"""
    def __init__(self, K0, Q, F0, with_c=True):
        self.K0 = K0
        self.Q = Q
        self.F0 = F0
        self.with_c = with_c

    def calc_fext(self, inc=1., silent=True, c=None):
        if c is None:
            return inc*self.F0
        return inc*(self.F0 + self.Q @ c)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        return csr_matrix(self.K0)

    def analysis(self):
        if self.with_c:
            return Analysis(calc_fext=self.calc_fext, calc_kC=self.calc_kC)
        return Analysis(calc_fext=lambda inc=1., silent=True:
                        self.calc_fext(inc), calc_kC=self.calc_kC)


def linear_follower_problem(with_c=True, factor=0.3):
    rng = np.random.default_rng(4)
    n = 6
    X = rng.standard_normal((n, n))
    K0 = X @ X.T + n*np.eye(n)
    W = rng.standard_normal((n, n))
    # unsymmetric load stiffness, scaled to a fraction of the critical one
    Q = W - 0.5*W.T
    rho = np.max(np.abs(np.linalg.eigvals(np.linalg.solve(K0, Q))))
    Q *= factor/rho
    F0 = rng.standard_normal(n)
    return LinearFollower(K0, Q, F0, with_c=with_c)


def test_analysis_linear_static_with_configuration_dependent_load():
    pb = linear_follower_problem()
    an = pb.analysis()
    an.relTOL = 1e-12
    increments, cs = an.static(NLgeom=False, silent=True)
    ref = np.linalg.solve(pb.K0 - pb.Q, pb.F0)
    assert np.allclose(cs[0], ref, rtol=1e-9)
    # without c the load is the reference one, as before
    pb_old = linear_follower_problem(with_c=False)
    increments, cs = pb_old.analysis().static(NLgeom=False, silent=True)
    assert np.allclose(cs[0], np.linalg.solve(pb.K0, pb.F0))


def test_analysis_linear_static_beyond_the_critical_load():
    r"""Beyond the critical load of the linearized problem the load
    correction iterations diverge and a warning is issued, the analysis
    returns the last iterate"""
    pb = linear_follower_problem(factor=1.5)
    an = pb.analysis()
    an.maxNumIter = 20
    increments, cs = an.static(NLgeom=False, silent=True)
    ref = np.linalg.solve(pb.K0 - pb.Q, pb.F0)
    assert not np.allclose(cs[0], ref, rtol=1e-6)
