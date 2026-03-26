"""Tests for structsolve.sparseutils"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

from structsolve.sparseutils import remove_null_cols, make_symmetric, make_skew_symmetric


class TestRemoveNullCols:
    def test_single_matrix(self):
        """Test removing null cols from a single matrix"""
        A = np.zeros((5, 5))
        A[1, 1] = 1.0
        A[3, 3] = 2.0
        A[1, 3] = 0.5
        A[3, 1] = 0.5
        K = csr_matrix(A)

        K_reduced, used_cols = remove_null_cols(K, silent=True)
        assert K_reduced.shape == (2, 2)
        assert len(used_cols) == 2
        np.testing.assert_array_equal(used_cols, [1, 3])

    def test_two_matrices(self):
        """Test removing null cols from two matrices simultaneously"""
        n = 6
        A = np.zeros((n, n))
        B = np.zeros((n, n))
        active = [0, 2, 4]
        for i in active:
            A[i, i] = float(i + 1)
            B[i, i] = float(i + 1) * 10
        K = csr_matrix(A)
        KG = csr_matrix(B)

        K_red, KG_red, used_cols = remove_null_cols(K, KG, silent=True)
        assert K_red.shape == (3, 3)
        assert KG_red.shape == (3, 3)
        np.testing.assert_array_equal(used_cols, active)

    def test_no_null_cols(self):
        """Test when no columns are null"""
        K = csr_matrix(np.eye(4))
        K_red, used_cols = remove_null_cols(K, silent=True)
        assert K_red.shape == (4, 4)
        assert len(used_cols) == 4


class TestMakeSymmetric:
    def test_upper_triangle(self):
        """Test making symmetric from upper triangle"""
        n = 4
        m = coo_matrix(np.array([
            [1, 2, 0, 0],
            [0, 3, 4, 0],
            [0, 0, 5, 6],
            [0, 0, 0, 7],
        ], dtype=float))
        sym = make_symmetric(m)
        result = sym.toarray()
        np.testing.assert_array_equal(result, result.T)
        assert result[0, 1] == 2.0
        assert result[1, 0] == 2.0
        assert result[2, 3] == 6.0
        assert result[3, 2] == 6.0

    def test_non_square_raises(self):
        """Test that non-square matrix raises ValueError"""
        m = coo_matrix(np.ones((3, 4)))
        with pytest.raises(ValueError):
            make_symmetric(m)


class TestMakeSkewSymmetric:
    def test_upper_triangle(self):
        """Test making skew-symmetric from upper triangle"""
        n = 3
        m = coo_matrix(np.array([
            [0, 2, 3],
            [0, 0, 4],
            [0, 0, 0],
        ], dtype=float))
        skew = make_skew_symmetric(m)
        result = skew.toarray()
        # Off-diagonal: m[i,j] = -m[j,i]
        np.testing.assert_allclose(result[0, 1], 2.0)
        np.testing.assert_allclose(result[1, 0], -2.0)
        np.testing.assert_allclose(result[0, 2], 3.0)
        np.testing.assert_allclose(result[2, 0], -3.0)
