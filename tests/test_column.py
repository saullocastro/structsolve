r"""Pinned column of :mod:`column`: boundary conditions, linear buckling with
a prestress and frequencies, against the exact solution of the discrete
problem"""
import warnings

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix

from structsolve import DenseFallbackWarning, freq, lb, solve

from column import Column, apply_null_rows


def test_null_rows_free_partition_and_null_space():
    r"""Three ways of imposing ``u_0 = w_0 = w_N = 0`` give the same load
    multipliers; only the null rows and columns give eigenvectors with the
    size of the model"""
    col = Column()
    K, f = col.K(), col.fext(P=1.)
    lambda_exact = col.load_multipliers(np.arange(1, 6))

    # 1) null rows and columns, removed by structsolve
    K1 = apply_null_rows(K, col.bc)
    c1 = solve(K1, f, silent=True)
    KG1 = apply_null_rows(col.KG(c1), col.bc)
    eigvals1, eigvecs1 = lb(K1, KG1, silent=True, num_eigvalues=5)
    assert eigvecs1.shape == (col.size, 5)
    assert np.all(eigvecs1[col.bc, :] == 0)

    # 2) partition of the free dofs, by the user
    free = np.setdiff1d(np.arange(col.size), col.bc)
    K2 = K[free, :][:, free]
    c2 = np.zeros(col.size)
    c2[free] = solve(K2, f[free], silent=True)
    KG2 = col.KG(c2)[free, :][:, free]
    eigvals2, eigvecs2_free = lb(K2, KG2, silent=True, num_eigvalues=5)
    assert eigvecs2_free.shape == (free.size, 5)
    eigvecs2 = np.zeros((col.size, 5))
    eigvecs2[free, :] = eigvecs2_free

    # 3) null space of the constraints [C]{c} = {0}, {c} = [T]{c_r}
    C = np.zeros((col.bc.size, col.size))
    C[np.arange(col.bc.size), col.bc] = 1.
    T = null_space(C)
    Kr = csr_matrix(T.T @ K @ T)
    c3 = T @ solve(Kr, T.T @ f, silent=True)
    KGr = csr_matrix(T.T @ col.KG(c3) @ T)
    eigvals3, eigvecs3_r = lb(Kr, KGr, silent=True, num_eigvalues=5)
    assert eigvecs3_r.shape == (T.shape[1], 5)
    eigvecs3 = T @ eigvecs3_r

    for eigvals in (eigvals1, eigvals2, eigvals3):
        np.testing.assert_allclose(eigvals, lambda_exact, rtol=1e-8)
    # same modes, up to the normalization and the sign
    for eigvecs in (eigvecs2, eigvecs3):
        cos = np.abs(np.einsum('ij,ij->j', eigvecs1, eigvecs))
        cos /= (np.linalg.norm(eigvecs1, axis=0)
                *np.linalg.norm(eigvecs, axis=0))
        np.testing.assert_allclose(cos, 1., rtol=1e-8)


def test_lb_prestress():
    r"""Linear static solution of a reference load, geometric stiffness
    matrix of its prestress, and load multipliers of the reference load"""
    col = Column()
    K = apply_null_rows(col.K(), col.bc)
    P_ref = 2.
    c = solve(K, col.fext(P=P_ref), silent=True)
    KG = apply_null_rows(col.KG(c), col.bc)
    eigvals, eigvecs = lb(K, KG, silent=True, num_eigvalues=5)
    np.testing.assert_allclose(eigvals,
                               col.load_multipliers(np.arange(1, 6), P_ref),
                               rtol=1e-8)
    # critical load: lambda_1 times the reference load, close to the Euler
    # load for N = 20 segments
    P_cr = eigvals[0]*P_ref
    np.testing.assert_allclose(P_cr, np.pi**2*col.EI/col.L**2, rtol=3e-3)

    # a constant preload P0, not multiplied by lambda, belongs to K
    P0 = 0.4*P_cr
    c0 = solve(K, col.fext(P=P0), silent=True)
    K_pre = K + apply_null_rows(col.KG(c0), col.bc)
    eigvals_pre, _ = lb(K_pre, KG, silent=True, num_eigvalues=5)
    np.testing.assert_allclose(eigvals_pre[0], (P_cr - P0)/P_ref,
                               rtol=1e-8)


def test_lb_reversed_load():
    r"""Under a tensile reference load all the load multipliers are negative:
    the column buckles only under the reversed load. The negative load
    multipliers returned are not those of smallest modulus; reversing the
    reference load gives the critical one"""
    col = Column()
    K = apply_null_rows(col.K(), col.bc)
    c = solve(K, col.fext(P=-1.), silent=True)
    KG = apply_null_rows(col.KG(c), col.bc)
    lambda_1 = col.load_multipliers(1)
    with warnings.catch_warnings():
        # the sparse solvers search positive load multipliers
        warnings.simplefilter('ignore', DenseFallbackWarning)
        eigvals, _ = lb(K, KG, silent=True, num_eigvalues=5)
    assert np.all(eigvals < 0)
    assert np.all(np.abs(eigvals) > 10*lambda_1)
    # all the load multipliers of the dense solver: -lambda_1 is the last
    eigvals_all, _ = lb(K, KG, silent=True, sparse_solver=False)
    np.testing.assert_allclose(eigvals_all[-1], -lambda_1, rtol=1e-8)
    # the reversed reference load
    eigvals_rev, _ = lb(K, -KG, silent=True, num_eigvalues=5)
    np.testing.assert_allclose(eigvals_rev[0], lambda_1, rtol=1e-8)


def test_freq_prestressed_column():
    r"""Natural frequencies with and without a compressive prestress,
    eigenvectors normalized with respect to the mass matrix, and the
    sorting of the eigenpairs"""
    col = Column()
    K = apply_null_rows(col.K(), col.bc)
    M = col.M()
    k = np.arange(1, 4)

    lambda2, eigvecs = freq(K, M, silent=True, num_eigvalues=3,
                            symmetric=True, sort=False)
    assert np.isrealobj(lambda2)
    order = np.argsort(-lambda2)
    lambda2, eigvecs = lambda2[order], eigvecs[:, order]
    omega = np.sqrt(-lambda2)        # rad/s
    f_Hz = omega/(2*np.pi)           # Hz
    np.testing.assert_allclose(omega**2, col.omega2(k), rtol=1e-8)
    np.testing.assert_allclose(f_Hz, np.sqrt(col.omega2(k))/(2*np.pi),
                               rtol=1e-8)
    # symmetric=True: eigenvectors normalized with respect to M
    np.testing.assert_allclose(eigvecs.T @ M @ eigvecs, np.eye(3),
                               atol=1e-8)

    # the general solvers, symmetric=False by default, give the same
    # eigenvalues as a complex array
    lambda2_gen, _ = freq(K, M, silent=True, num_eigvalues=3, sort=False)
    assert np.iscomplexobj(lambda2_gen)
    np.testing.assert_allclose(np.sort((-lambda2_gen).real), omega**2,
                               rtol=1e-8)

    # compressive prestress lam*KG: omega_k**2 (1 - lam/lambda_k)
    c = solve(K, col.fext(P=1.), silent=True)
    KG = apply_null_rows(col.KG(c), col.bc)
    lam = 0.5*col.load_multipliers(1)
    lambda2_pre, _ = freq(K + lam*KG, M, silent=True, num_eigvalues=3,
                          symmetric=True, sort=False)
    np.testing.assert_allclose(
        np.sort(-lambda2_pre),
        col.omega2(k)*(1 - lam/col.load_multipliers(k)), rtol=1e-8)

    # sort=True sorts and filters the eigenvectors only, lambda2 keeps the
    # order of the solver
    lambda2_s, eigvecs_s = freq(K, M, silent=True, num_eigvalues=3,
                                symmetric=True, sort=True)
    lambda2_u, eigvecs_u = freq(K, M, silent=True, num_eigvalues=3,
                                symmetric=True, sort=False)
    np.testing.assert_array_equal(lambda2_s, lambda2_u)
    assert eigvecs_s.shape == eigvecs_u.shape
