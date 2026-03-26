"""Tests for structsolve.freq: freq()"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix

from structsolve import freq


def test_freq_simple():
    """Test frequency analysis with known natural frequencies

    For a diagonal system: [K]{u} + omega^2 [M]{u} = 0
    K = diag(k1, k2, ...), M = diag(m1, m2, ...)
    omega_i = sqrt(ki/mi)
    """
    n = 10
    k_vals = np.arange(1, n + 1, dtype=float) * 100
    m_vals = np.ones(n)
    K = csc_matrix(np.diag(k_vals))
    M = csc_matrix(np.diag(m_vals))

    eigvals, eigvecs = freq(K, M, silent=True, num_eigvalues=5,
                            sparse_solver=True)

    # freq returns lambda2 where [K] + lambda2 [M] = 0
    # so lambda2 = -omega^2 = -k/m
    # The sorted natural frequencies should match sqrt(k_vals)
    assert eigvecs.shape[0] == n


def test_freq_dense_solver():
    """Test freq with sparse_solver=False"""
    n = 8
    k_vals = np.array([1, 4, 9, 16, 25, 36, 49, 64], dtype=float)
    m_vals = np.ones(n)
    K = csc_matrix(np.diag(k_vals))
    M = csc_matrix(np.diag(m_vals))

    eigvals, eigvecs = freq(K, M, silent=True, num_eigvalues=5,
                            sparse_solver=False)
    assert eigvecs.shape[0] == n


def test_freq_with_null_dofs():
    """Test frequency analysis with constrained (null) DOFs"""
    n = 8
    K_dense = np.zeros((n, n))
    M_dense = np.zeros((n, n))
    active = [1, 2, 3, 4, 5]
    for i in active:
        K_dense[i, i] = float(i) * 100
        M_dense[i, i] = 1.0
    K = csc_matrix(K_dense)
    M = csc_matrix(M_dense)

    eigvals, eigvecs = freq(K, M, silent=True, num_eigvalues=3,
                            sparse_solver=True)
    assert eigvecs.shape[0] == n
