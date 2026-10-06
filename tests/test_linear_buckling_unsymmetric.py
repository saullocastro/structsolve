"""Tests for structsolve.linear_buckling: lb() with unsymmetric matrices"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy.linalg import eig
from scipy.sparse import csr_matrix, diags, random as sprandom, identity

from structsolve import lb


def relative_residuals(K, KG, eigvals, eigvecs):
    Ku = K @ eigvecs
    KGu = KG @ eigvecs
    return (np.linalg.norm(Ku + KGu*eigvals, axis=0)
            / (np.linalg.norm(Ku, axis=0)
               + np.abs(eigvals)*np.linalg.norm(KGu, axis=0)))


def dense_load_multipliers(K, KG):
    """Finite load multipliers of the full problem with a direct dense eig"""
    mu = eig(a=np.asarray(KG, dtype=complex), b=np.asarray(K, dtype=complex),
             right=False)
    mu = mu[np.isfinite(mu)]
    # the eigenvalues of the null rows and columns of KG
    mu = mu[np.abs(mu) > 1e-10*np.abs(mu).max()]
    return -1./mu


def matrices_complex_KG_block():
    """Dofs 3 to 5 have null rows and columns of KG, which are condensed out,
    and the KG block has a complex pair of eigenvalues"""
    n = 6
    K = np.diag([4., 5., 6., 7., 8., 9.]) + 0.1*(np.eye(n, k=1)
                                                  + np.eye(n, k=-1))
    KG = np.zeros((n, n))
    KG[:3, :3] = [[-1., 0.5, 0.], [-0.5, -1., 0.], [0., 0., -2.]]
    return K, KG


def test_lb_unsymmetric_complex_pair_condensed():
    """Regression test: a complex eigenpair of the condensed problem, even
    when not returned, made the back-substitution of the condensed dofs
    raise a TypeError in the real SuperLU factorization"""
    K, KG = matrices_complex_KG_block()
    ref = dense_load_multipliers(K, KG)
    real = np.abs(ref.imag) <= 1e-8*np.abs(ref)
    assert np.sum(real) == 1 and np.sum(~real) == 2
    lam_real = ref[real].real[0]
    lam_cplx = ref[~real]

    # only the real eigenpair is returned: real arrays
    eigvals, eigvecs = lb(csr_matrix(K), csr_matrix(KG), num_eigvalues=1,
                          symmetric=False, silent=True)
    assert eigvals.dtype == np.float64 and eigvecs.dtype == np.float64
    assert eigvecs.shape == (6, 1)
    np.testing.assert_allclose(eigvals, [lam_real], rtol=1e-10)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-10)

    # the real eigenpair first, followed by one of the complex pair
    eigvals, eigvecs = lb(csr_matrix(K), csr_matrix(KG), num_eigvalues=2,
                          symmetric=False, silent=True)
    assert eigvals.shape == (2,) and eigvecs.shape == (6, 2)
    assert np.iscomplexobj(eigvals) and np.iscomplexobj(eigvecs)
    assert eigvals[0].imag == 0 and np.all(eigvecs[:, 0].imag == 0)
    np.testing.assert_allclose(eigvals[0].real, lam_real, rtol=1e-10)
    assert np.min(np.abs(lam_cplx - eigvals[1])) <= 1e-10*abs(eigvals[1])
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-10)


def test_lb_unsymmetric_general_eigenpairs_real_when_kept_real():
    """The real/complex decision applies to the returned eigenpairs"""
    from structsolve.linear_buckling import _general_eigenpairs

    mu = np.array([-0.5 + 0.2j, -0.5 - 0.2j, -0.25 + 1e-14j, -0.1, np.inf])
    v = np.random.RandomState(0).randn(4, 5)*(1 + 1j)
    m, u = _general_eigenpairs(mu, v, 2)
    assert m.dtype == np.float64 and u.dtype == np.float64
    np.testing.assert_allclose(m, [-0.25, -0.1])
    # rotated eigenvector with a positive largest component
    r = v[:, 2].real
    np.testing.assert_allclose(u[:, 0],
                               np.sqrt(2)*r*np.sign(r[np.argmax(abs(r))]))
    m, u = _general_eigenpairs(mu, v)
    assert np.iscomplexobj(m) and m.shape == (4,)
    np.testing.assert_allclose(m[:2], [-0.25, -0.1])
    assert np.all(u[:, :2].imag == 0)


@pytest.mark.parametrize('max_dense_size', [2000, 0])
def test_lb_unsymmetric_complex_eigenpairs(max_dense_size):
    """Genuinely complex eigenpairs, with condensed dofs that receive complex
    eigenvectors from the back-substitution"""
    n, nb = 30, 12
    K = (diags(np.linspace(4., 8., n)) + diags(0.5*np.ones(n - 1), 1)
         + diags(0.5*np.ones(n - 1), -1)).toarray()
    rng = np.random.RandomState(4)
    A = rng.randn(nb, nb)
    KG = np.zeros((n, n))
    KG[:nb, :nb] = -np.eye(nb) + 0.5*(A - A.T)
    ref = dense_load_multipliers(K, KG)
    assert np.all(np.abs(ref.imag) > 1e-3*np.abs(ref))
    k = 4
    eigvals, eigvecs = lb(csr_matrix(K), csr_matrix(KG), num_eigvalues=k,
                          symmetric=False, silent=True,
                          max_dense_size=max_dense_size)
    assert np.iscomplexobj(eigvals) and np.iscomplexobj(eigvecs)
    assert eigvals.shape == (k,) and eigvecs.shape == (n, k)
    assert np.all(np.abs(eigvals.imag) > 1e-3*np.abs(eigvals))
    # the k load multipliers of smallest modulus, by increasing modulus
    np.testing.assert_allclose(np.abs(eigvals), np.sort(np.abs(ref))[:k],
                               rtol=1e-8)
    for lam in eigvals:
        assert np.min(np.abs(ref - lam)) <= 1e-8*abs(lam)
    res = relative_residuals(K, KG, eigvals, eigvecs)
    assert np.all(res < 1e-8)
    # the condensed dofs are not null
    assert np.all(np.linalg.norm(eigvecs[nb:, :], axis=0) > 0)


def test_lb_check_eigenpairs_real_and_complex():
    """_check_eigenpairs with check_inertia=False accepts real and complex
    eigenpairs, and detects wrong ones"""
    from structsolve.linear_buckling import _check_eigenpairs, _eig_condensed

    K, KG = matrices_complex_KG_block()
    K = csr_matrix(K)
    KG = csr_matrix(KG)
    mu, eigvecs = _eig_condensed(K, KG, 3)
    assert np.iscomplexobj(mu) and np.iscomplexobj(eigvecs)
    lam = -1./mu
    assert _check_eigenpairs(K, KG, lam, eigvecs, 1e-8,
                             check_inertia=False) is None
    error = _check_eigenpairs(K, KG, lam, eigvecs[:, ::-1], 1e-8,
                              check_inertia=False)
    assert 'relative residual' in error
    # complex load multipliers are ignored by the inertia check
    assert _check_eigenpairs(K, KG, lam, eigvecs, 1e-8) is None

    mu, eigvecs = _eig_condensed(K, KG, 1)
    assert mu.dtype == np.float64 and eigvecs.dtype == np.float64
    lam = -1./mu
    assert _check_eigenpairs(K, KG, lam, eigvecs, 1e-8,
                             check_inertia=False) is None
    error = _check_eigenpairs(K, KG, lam*1.01, eigvecs, 1e-8,
                              check_inertia=False)
    assert 'relative residual' in error


@pytest.mark.parametrize('symmetric, solver', [
    (False, '_eig_condensed'), (True, '_eigh_condensed')])
def test_lb_solver_type_error_falls_back(monkeypatch, symmetric, solver):
    """A TypeError of the first solver is recorded and the next solver,
    eigs() or eigsh(), is used"""
    import structsolve.linear_buckling as linear_buckling

    n = 40
    k = np.linspace(1., 3., n)
    lambdas = np.linspace(1., 20., n)
    K = diags(k).tocsr()
    KG = diags(-k/lambdas).tocsr()

    def raise_type_error(*args, **kwargs):
        raise TypeError('Cannot cast array data from dtype complex128')

    monkeypatch.setattr(linear_buckling, solver, raise_type_error)
    eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=5,
                          symmetric=symmetric)
    np.testing.assert_allclose(eigvals, lambdas[:5], rtol=1e-8)
    assert eigvals.dtype == np.float64 and eigvecs.dtype == np.float64
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-8)


@pytest.mark.parametrize('sparse_solver, max_dense_size',
                         [(True, 2000), (True, 0), (False, 2000)])
def test_lb_nearly_symmetric(sparse_solver, max_dense_size):
    """A symmetric positive definite K with 1e-12 asymmetric noise, forced
    to the solvers of unsymmetric matrices, gives the load multipliers of
    the symmetric solver, with real eigenvectors"""
    n = 60
    A = sprandom(n, n, density=0.1, random_state=2)
    K = (A @ A.T + identity(n)).tocsr()
    B = sprandom(n//3, n//3, density=0.3, random_state=3)
    KG = np.zeros((n, n))
    KG[:n//3, :n//3] = -(B @ B.T + 0.1*identity(n//3)).toarray()
    KG = csr_matrix(KG)
    noise = sprandom(n, n, density=0.2, random_state=5).toarray()
    Kn = csr_matrix(K.toarray() + 1e-12*abs(K).max()*noise)
    assert np.linalg.norm((Kn - Kn.T).toarray()) > 1e-13*abs(K).max()

    ref, _ = lb(K, KG, silent=True, num_eigvalues=10)
    eigvals, eigvecs = lb(Kn, KG, silent=True, num_eigvalues=10,
                          symmetric=False, sparse_solver=sparse_solver,
                          max_dense_size=max_dense_size)
    if not sparse_solver:
        # all eigenvalues are returned, also the spurious ones of the null
        # rows of KG, which may be complex
        assert np.all(eigvals[:10].imag == 0)
        eigvals = eigvals[:10].real
    assert eigvals.dtype == np.float64 and eigvecs.dtype == np.float64
    assert eigvecs.shape == (n, 10)
    np.testing.assert_allclose(eigvals, ref, rtol=1e-8)
    assert np.all(relative_residuals(Kn, KG, eigvals, eigvecs) < 1e-8)


def test_eig_complex_mkl_crash_regression():
    """Regression test: with Intel MKL 2025.0.0, scipy.linalg.eig on these
    real 48 x 48 matrices (LAPACK dggev) corrupts the heap and the process
    exits with 0xc0000374, deterministically and with any number of threads.
    lb() calls the complex driver zggev. Run in a subprocess, which a crash
    does not take down with the test session"""
    code = textwrap.dedent("""
        import numpy as np
        from scipy.sparse import csr_matrix
        from structsolve import lb
        rng = np.random.RandomState(4)
        KG = rng.randn(48, 48)
        K = rng.randn(48, 48) + 48*np.eye(48)
        eigvals, eigvecs = lb(csr_matrix(K), csr_matrix(KG), silent=True,
                              num_eigvalues=5, symmetric=False)
        Ku = K @ eigvecs
        KGu = KG @ eigvecs
        res = (np.linalg.norm(Ku + KGu*eigvals, axis=0)
               / (np.linalg.norm(Ku, axis=0)
                  + np.abs(eigvals)*np.linalg.norm(KGu, axis=0)))
        assert eigvecs.shape == (48, 5), eigvecs.shape
        assert np.all(res < 1e-8), res
        print('ok')
        """)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [root] + [p for p in [env.get('PYTHONPATH')] if p])
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (hex(result.returncode & 0xffffffff),
                                    result.stderr[-2000:])
    assert result.stdout.strip().endswith('ok')
