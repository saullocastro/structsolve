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
