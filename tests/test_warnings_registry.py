"""The solvers do not reset the registry of the warnings already shown

A warning of the caller under the ``'default'`` action is shown once per
location. The exit of :func:`warnings.catch_warnings` resets the registry of
the warnings already shown, of every module, such that a solver using it
would make the warning of the caller be shown again after each call.

"""
import importlib
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags

import structsolve.arpackutils as arpackutils
import structsolve.linear_buckling as linear_buckling
from structsolve import freq, lb, static
from structsolve.linear_buckling import DenseFallbackWarning

from analytic_problems import von_mises_truss

#NOTE structsolve.freq is the function freq(), exported by the package
freq_module = importlib.import_module('structsolve.freq')


def emit():
    warnings.warn('caller warning', UserWarning)


def symmetric_matrices(n=40, num_active=20):
    """Positive definite ``K`` and ``KG`` with null rows, which the dense
    fallback condenses"""
    K = diags([-1., 4., -1.], [-1, 0, 1], shape=(n, n)).tocsr()
    KG = diags(-np.r_[np.ones(num_active), np.zeros(n - num_active)]).tocsr()
    return K, KG


def unsymmetric_matrices(n=20):
    r"""Upper triangular `\boldsymbol{K}^{-1} \boldsymbol{K}_G`, load multipliers 1 to ``n``"""
    K = diags(np.arange(1., n + 1)).tocsr()
    KG = csr_matrix(-np.eye(n) + 0.05*np.triu(np.ones((n, n)), k=1))
    return K, KG


def complex_pair_matrices(n=30, nb=12):
    """Unsymmetric ``KG`` without a positive real load multiplier: the shift
    search fails and the dense fallback, with condensed dofs, returns the
    complex load multipliers, see
    ``test_lb_unsymmetric_complex_eigenpairs``"""
    K = (diags(np.linspace(4., 8., n)) + diags(0.5*np.ones(n - 1), 1)
         + diags(0.5*np.ones(n - 1), -1)).toarray()
    A = np.random.RandomState(4).randn(nb, nb)
    KG = np.zeros((n, n))
    KG[:nb, :nb] = -np.eye(nb) + 0.5*(A - A.T)
    return csr_matrix(K), csr_matrix(KG)


def freq_matrices(symmetric, n=60):
    K = diags([-1., 4., -1.], [-1, 0, 1], shape=(n, n)).tocsr()
    if not symmetric:
        K = K + diags(1e-3*np.ones(n - 2), 2)
    M = diags(1. + np.random.RandomState(0).rand(n)).tocsr()
    return csr_matrix(K), M


def test_warnings_shown_once(monkeypatch):
    r"""A warning of the caller, shown once per location with the
    ``'default'`` action, is not repeated after the calls of :func:`lb`
    (symmetric, unsymmetric, dense fallbacks with condensed dofs), of
    :func:`freq` (symmetric with the deflation passes of
    :func:`.arpackutils.capped_eigsh`, unsymmetric) and of the linear and
    nonlinear static analyses"""
    K, KG = symmetric_matrices()
    lb_ref = lb(K, KG, silent=True, sparse_solver=False)[0][:3]
    Ku, KGu = unsymmetric_matrices()
    Kc, KGc = complex_pair_matrices()

    def lb_symmetric():
        eigvals, _ = lb(K, KG, num_eigvalues=3, silent=True)
        np.testing.assert_allclose(eigvals, lb_ref, rtol=1e-8)

    def lb_unsymmetric():
        eigvals, _ = lb(Ku, KGu, num_eigvalues=3, symmetric=False,
                        silent=True)
        np.testing.assert_allclose(eigvals, [1., 2., 3.], rtol=1e-8)

    def lb_symmetric_dense_fallback():
        with monkeypatch.context() as m:
            rng = np.random.RandomState(0)
            m.setattr(linear_buckling, '_eigsh_cayley',
                      lambda K, KG, k, *args, **kwargs:
                      (-rng.rand(k) - 1., rng.randn(K.shape[0], k)))
            eigvals, _ = lb(K, KG, num_eigvalues=3, silent=True)
        np.testing.assert_allclose(eigvals, lb_ref, rtol=1e-8)

    def lb_unsymmetric_dense_fallback():
        eigvals, _ = lb(Kc, KGc, num_eigvalues=4, symmetric=False,
                        silent=True)
        assert np.all(np.abs(eigvals.imag) > 1e-3*np.abs(eigvals))

    def freq_symmetric():
        with monkeypatch.context() as m:
            # deflation passes of at most (8 - 1)//2 = 3 eigenpairs
            m.setattr(arpackutils, 'ARPACK_MAX_NCV', 8)
            Kf, Mf = freq_matrices(True)
            lambda2, _ = freq(Kf, Mf, num_eigvalues=5, symmetric=True,
                              silent=True)
        ref = eigh(Kf.toarray(), Mf.toarray(), eigvals_only=True)
        np.testing.assert_allclose(-lambda2, ref[:5], rtol=1e-8)

    def freq_unsymmetric():
        Kf, Mf = freq_matrices(False)
        lambda2, _ = freq(Kf, Mf, num_eigvalues=5, symmetric=False,
                          silent=True)
        ref = np.sort(np.linalg.eigvals(np.linalg.solve(Mf.toarray(),
                                                        Kf.toarray())).real)
        np.testing.assert_allclose(np.sort(-lambda2.real), ref[:5],
                                   rtol=1e-8)

    def linear_static():
        Ks, _ = symmetric_matrices()
        f = np.ones(Ks.shape[0])
        _, cs = static(Ks, f, silent=True)
        u = cs[0]
        np.testing.assert_allclose(Ks @ u, f, rtol=1e-12)

    def nonlinear(method):
        def run():
            p = von_mises_truss()
            p.analysis(NL_method=method, initialInc=0.2)
            p.an.static(NLgeom=True, silent=True)
            assert p.an.increments[-1] == 1.
        return run

    cases = [('lb symmetric', lb_symmetric),
             ('lb unsymmetric', lb_unsymmetric),
             ('lb symmetric, dense fallback', lb_symmetric_dense_fallback),
             ('lb unsymmetric, dense fallback', lb_unsymmetric_dense_fallback),
             ('freq symmetric, deflation passes', freq_symmetric),
             ('freq unsymmetric', freq_unsymmetric),
             ('linear static', linear_static),
             ('Newton-Raphson', nonlinear('NR')),
             ('Riks arc-length', nonlinear('arc_length_riks')),
             ('Crisfield arc-length', nonlinear('arc_length_crisfield'))]
    shown = {}
    for name, run in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('default')
            for _ in range(3):
                emit()
                run()
        shown[name] = [str(w.message) for w in caught].count('caller warning')
        if 'dense fallback' in name:
            # the warning of the solver is also shown once only
            shown[name + ' (DenseFallbackWarning)'] = len(
                [w for w in caught
                 if issubclass(w.category, DenseFallbackWarning)])
    assert shown == dict.fromkeys(shown, 1)
