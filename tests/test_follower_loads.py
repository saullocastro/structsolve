r"""Configuration-dependent (follower) loads and unsymmetric eigenproblems

The analytic problem has the internal force of hardening springs and a load
that depends on the configuration through an unsymmetric matrix ``Q``:

.. math::

    \{F_{int}(c)\} = [K]\{c\} + \{a c^3\}, \qquad
    \{F(c, \lambda)\} = \lambda (\{q_0\} + [Q]\{c\})

passed to :class:`structsolve.Analysis` as described in
:mod:`structsolve.callbacks`: ``calc_fint(c, inc)`` contains ``-inc Q c``,
``calc_kC(c, NLgeom, inc)`` contains ``-inc Q``, and ``calc_fext(inc, c)`` is
the load of the configuration ``c``.
"""
import numpy as np
import pytest
from scipy.linalg import eig
from scipy.sparse import csr_matrix, random as sprandom

from structsolve import Analysis, lb
from structsolve import callbacks
from structsolve.linear_buckling import is_symmetric


class FollowerProblem(object):
    def __init__(self, K, a, q0, Q, with_c=True):
        self.K = K
        self.a = a
        self.q0 = q0
        self.Q = Q
        self.with_c = with_c
        self.an = None
        # (inc, ||R||/||lbd*q||, number of converged increments) of each
        # calc_fint call
        self.calls = []

    def fint_el(self, c):
        return self.K @ c + self.a*c**3

    def residual(self, c, lbd):
        return lbd*(self.q0 + self.Q @ c) - self.fint_el(c)

    def calc_fext(self, inc=1., silent=True, c=None):
        if c is None:
            return inc*self.q0
        return inc*(self.q0 + self.Q @ c)

    def calc_fint(self, c, inc=1., silent=True):
        R = self.residual(c, inc)
        ref = max(np.linalg.norm(inc*(self.q0 + self.Q @ c)), 1e-300)
        step = len(self.an.increments) if self.an is not None else 0
        self.calls.append((inc, np.linalg.norm(R)/ref, step))
        return self.fint_el(c) - inc*(self.Q @ c)

    def calc_kC(self, c=None, NLgeom=False, inc=1., silent=True):
        if not NLgeom:
            return csr_matrix(self.K)
        return csr_matrix(self.K + np.diag(3*self.a*c**2) - inc*self.Q)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return csr_matrix(self.K.shape)

    def analysis(self, **kwargs):
        if self.with_c:
            calc_fext = self.calc_fext
        else:
            calc_fext = lambda inc=1., silent=True: self.calc_fext(inc)
        an = Analysis(calc_fext, self.calc_fint, self.calc_kC, self.calc_kG)
        for k, v in kwargs.items():
            setattr(an, k, v)
        self.an = an
        return an


def make_problem(with_c=True):
    n = 3
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[-1, -1] = 100.
    a = np.array([20., 40., 60.])
    q0 = np.array([0., 0., 30.])
    # unsymmetric load stiffness, of the order of K
    Q = np.array([[0., 30., -20.],
                  [-30., 10., 40.],
                  [10., -40., 20.]])
    return FollowerProblem(K, a, q0, Q, with_c=with_c)


def test_accepts_kwarg():
    def f(c, inc=1., silent=True):
        pass
    def g(c, silent=True, **kwargs):
        pass
    class A(object):
        def m(self, c, inc=1.):
            pass
    assert callbacks.accepts_kwarg(f, 'inc')
    assert not callbacks.accepts_kwarg(f, 'c_cte')
    # **kwargs does not count, such that old callables are never surprised
    assert not callbacks.accepts_kwarg(g, 'inc')
    assert callbacks.accepts_kwarg(A().m, 'inc')
    assert not callbacks.accepts_kwarg(len, 'inc')


def test_callables_without_inc_are_called_as_before():
    calls = []
    def calc_fint(c, silent=True):
        calls.append('fint')
        return 2*c
    def calc_kC(c=None, NLgeom=False, silent=True):
        calls.append('kC')
        return csr_matrix(2*np.eye(2))
    def calc_kG(c=None, NLgeom=False, silent=True):
        calls.append('kG')
        return csr_matrix((2, 2))
    def calc_fext(inc=1., silent=True):
        return inc*np.array([1., 2.])
    an = Analysis(calc_fext, calc_fint, calc_kC, calc_kG)
    an.static(NLgeom=True, silent=True)
    assert np.allclose(an.cs[-1], [0.5, 1.])
    an.NL_method = 'arc_length_riks'
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    assert np.allclose(an.cs[-1], [0.5, 1.])


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
def test_equilibrium_with_follower_load(method):
    pb = make_problem()
    an = pb.analysis(NL_method=method, initialInc=0.2, relTOL=1e-10)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    for lbd, c in zip(an.increments, an.cs):
        R = pb.residual(c, lbd)
        assert np.linalg.norm(R) <= 1e-8*np.linalg.norm(lbd*pb.q0), (lbd, R)
    # the follower terms matter: the dead-load solution is different
    pb_dead = make_problem()
    pb_dead.Q = 0*pb.Q
    an_dead = pb_dead.analysis(relTOL=1e-10)
    an_dead.static(NLgeom=True, silent=True)
    assert np.linalg.norm(an_dead.cs[-1] - an.cs[-1]) > 0.1*np.linalg.norm(an.cs[-1])


def orders(errors, floor=1e-12):
    """Convergence orders log(e2/e1)/log(e1/e0) above the round-off floor"""
    return [np.log(e2/e1)/np.log(e1/e0)
            for e0, e1, e2 in zip(errors[:-2], errors[1:-1], errors[2:])
            if e2 > floor and e0 < 5e-2]


def test_newton_raphson_quadratic_with_follower_load():
    pb = make_problem()
    an = pb.analysis(initialInc=1., relTOL=1e-14)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    errors = [e for inc, e, _ in pb.calls if inc == 1.]
    ps = orders(errors)
    assert len(ps) >= 1, errors
    assert min(ps) > 1.7, errors


def test_riks_quadratic_only_with_configuration_load_vector():
    r"""The bordered system of Riks is the Newton-Raphson method of the
    augmented system only when dR/dlbd = q0 + Q c, i.e. with ``calc_fext``
    accepting ``c``; with the constant reference vector the corrections
    converge linearly"""
    worst = {}
    for with_c in (True, False):
        pb = make_problem(with_c=with_c)
        an = pb.analysis(NL_method='arc_length_riks', initialInc=0.5,
                         relTOL=1e-13, maxNumIter=60)
        an.static(NLgeom=True, silent=True)
        assert np.isclose(an.increments[-1], 1.)
        # residuals of the iterations of the first step, whose first value
        # is the one of the tangent predictor
        first = [e for inc, e, step in pb.calls if step == 0]
        worst[with_c] = (min(orders(first)) if orders(first) else None,
                         first)
    p_c, errors_c = worst[True]
    assert p_c is not None and p_c > 1.7, errors_c
    p_const, errors_const = worst[False]
    assert len(errors_const) > len(errors_c), (errors_c, errors_const)


def test_is_symmetric():
    A = sprandom(30, 30, density=0.2, random_state=1)
    S = A + A.T
    assert is_symmetric(S)
    assert not is_symmetric(A)
    assert is_symmetric(S + 1e-14*A)


def spd(n, rng):
    X = rng.standard_normal((n, n))
    return X @ X.T + n*np.eye(n)


@pytest.mark.parametrize('sparse_solver,max_dense_size',
                         [(True, 2000), (True, 0), (False, 2000)])
def test_lb_unsymmetric_real(sparse_solver, max_dense_size):
    r"""Unsymmetric KG with real eigenvalues, against scipy.linalg.eig"""
    rng = np.random.default_rng(3)
    n = 40
    K = spd(n, rng)
    # KG = S + small skew part keeps the eigenvalues real
    S = -spd(n, rng)/n
    W = rng.standard_normal((n, n))
    KG = S + 1e-3*(W - W.T)
    assert not is_symmetric(KG)
    eigvals, eigvecs = lb(csr_matrix(K), csr_matrix(KG), silent=True,
                          sparse_solver=sparse_solver, num_eigvalues=6,
                          max_dense_size=max_dense_size)
    assert np.isrealobj(eigvals) and np.isrealobj(eigvecs)
    mu = eig(KG, K, right=False)
    lam = np.sort((-1/mu).real)
    lam = lam[lam > 0]
    assert np.allclose(eigvals[:5], lam[:5], rtol=1e-8)
    for i in range(5):
        u = eigvecs[:, i]
        r = K @ u + eigvals[i]*(KG @ u)
        assert np.linalg.norm(r) <= 1e-8*np.linalg.norm(K @ u)


def test_lb_unsymmetric_complex_pairs():
    r"""A strongly skew KG gives complex conjugate pairs, returned last"""
    rng = np.random.default_rng(5)
    n = 20
    K = spd(n, rng)
    W = rng.standard_normal((n, n))
    KG = -np.eye(n) + 2*(W - W.T)
    eigvals, eigvecs = lb(csr_matrix(K), csr_matrix(KG), silent=True,
                          sparse_solver=False)
    assert np.iscomplexobj(eigvals)
    mu = eig(KG, K, right=False)
    ref = -1/mu
    assert np.allclose(np.sort_complex(eigvals), np.sort_complex(ref))
    real = np.isreal(eigvals)
    # the real ones come first
    assert not np.any(real[np.argmin(real):]) or np.all(real)


def test_lb_symmetric_path_unchanged():
    rng = np.random.default_rng(7)
    n = 30
    K = spd(n, rng)
    KG = -spd(n, rng)/n
    e1, v1 = lb(csr_matrix(K), csr_matrix(KG), silent=True, num_eigvalues=5)
    e2, v2 = lb(csr_matrix(K), csr_matrix(KG), silent=True, num_eigvalues=5,
                symmetric=True)
    e3, v3 = lb(csr_matrix(K), csr_matrix(KG), silent=True, num_eigvalues=5,
                symmetric=False)
    assert np.array_equal(e1, e2)
    assert np.allclose(e1[:5], e3[:5], rtol=1e-9)
