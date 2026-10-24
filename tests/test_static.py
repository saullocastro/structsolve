"""Tests for structsolve.static: solve() and static()"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix

from structsolve import solve
from structsolve.static import static


def test_solve_diagonal_system():
    """Solve a simple diagonal system K*u = f"""
    n = 10
    diag = np.arange(1, n + 1, dtype=float)
    K = csc_matrix(np.diag(diag))
    f = np.ones(n)
    u = solve(K, f, silent=True)
    expected = 1.0 / diag
    np.testing.assert_allclose(u, expected)


def test_solve_with_null_columns():
    """Solve a system where some DOFs are constrained (null rows/cols)"""
    n = 6
    K_dense = np.zeros((n, n))
    # Only DOFs 1,2,3,4 are active
    active = np.array([1, 2, 3, 4])
    for i in active:
        K_dense[i, i] = float(i + 1)
    K = csc_matrix(K_dense)
    f = np.zeros(n)
    f[active] = 1.0
    u = solve(K, f, silent=True)
    # DOFs 0 and 5 should remain zero
    assert u[0] == 0.0
    assert u[5] == 0.0
    for i in active:
        np.testing.assert_allclose(u[i], 1.0 / (i + 1))


def test_solve_symmetric_system():
    """Solve a symmetric positive definite system"""
    A = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ])
    K = csc_matrix(A)
    f = np.array([1.0, 2.0, 3.0])
    u = solve(K, f, silent=True)
    expected = np.linalg.solve(A, f)
    np.testing.assert_allclose(u, expected, rtol=1e-10)


def test_static_linear():
    """Test the static() convenience function"""
    n = 5
    K = csc_matrix(np.eye(n) * 2.0)
    f = np.ones(n) * 4.0
    increments, cs = static(K, f, silent=True)
    assert len(increments) == 1
    assert increments[0] == 1.0
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0], 2.0 * np.ones(n))


def test_boundary_conditions_three_ways():
    """The chain of hardening springs of analytic_problems with its free end,
    dof 3, constrained to zero: null rows and columns removed by the
    solvers, a partition of the free dofs, and the null space of the
    constraint give the same static solution and natural frequencies (unit
    masses); only the null rows and columns give vectors with the size of
    the model"""
    from scipy.linalg import null_space
    from scipy.sparse import csr_matrix

    from structsolve import freq
    from analytic_problems import hardening_springs

    p = hardening_springs()
    K = p.kC.copy()
    n = K.shape[0]
    M = np.eye(n)
    f = np.array([0., 0., 10., 0.])
    bc = [3]
    free = np.array([0, 1, 2])
    u_ref = np.linalg.solve(K[np.ix_(free, free)], f[free])
    w2_ref = np.linalg.eigvalsh(K[np.ix_(free, free)])

    # 1) null rows and columns, removed by structsolve
    K1 = K.copy()
    K1[bc, :] = 0.
    K1[:, bc] = 0.
    u1 = solve(csr_matrix(K1), f, silent=True)
    assert u1.shape == (n,) and u1[3] == 0
    lambda2, eigvecs1 = freq(csr_matrix(K1), csr_matrix(M), silent=True,
                             sparse_solver=False, symmetric=True)
    # the dof removed from K is removed from M too
    assert eigvecs1.shape == (n, 3)
    assert np.all(eigvecs1[3, :] == 0)
    np.testing.assert_allclose(u1[free], u_ref)
    np.testing.assert_allclose(-lambda2, w2_ref)

    # 2) partition of the free dofs, by the user
    K2 = csr_matrix(K[np.ix_(free, free)])
    u2 = solve(K2, f[free], silent=True)
    lambda2_2, eigvecs2 = freq(K2, csr_matrix(M[np.ix_(free, free)]),
                               silent=True, sparse_solver=False,
                               symmetric=True)
    assert eigvecs2.shape == (3, 3)
    np.testing.assert_allclose(u2, u_ref)
    np.testing.assert_allclose(-lambda2_2, w2_ref)

    # 3) null space of the constraint [C]{c} = {0}, {c} = [T]{c_r}
    C = np.zeros((1, n))
    C[0, bc] = 1.
    T = null_space(C)
    u3 = T @ solve(csr_matrix(T.T @ K @ T), T.T @ f, silent=True)
    lambda2_3, eigvecs3_r = freq(csr_matrix(T.T @ K @ T),
                                 csr_matrix(T.T @ M @ T), silent=True,
                                 sparse_solver=False, symmetric=True)
    eigvecs3 = T @ eigvecs3_r
    np.testing.assert_allclose(u3[free], u_ref)
    np.testing.assert_allclose(u3[3], 0., atol=1e-12)
    np.testing.assert_allclose(-lambda2_3, w2_ref)

    # same modes, up to the sign
    for eigvecs in (eigvecs1[free], eigvecs2, eigvecs3[free]):
        np.testing.assert_allclose(np.abs(eigvecs1[free].T @ eigvecs),
                                   np.eye(3), atol=1e-10)
