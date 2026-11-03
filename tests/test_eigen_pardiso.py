"""Tests of ``linear_solver='pardiso'`` in the symmetric eigenvalue analyses

:func:`.lb`, :func:`.freq`, :func:`.arpackutils.capped_eigsh` and the
inertia counts of :func:`.linear_buckling._negative_pivots`, compared with
``'splu'``. They need oneMKL, see :mod:`structsolve.pardiso`.

"""
import warnings

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from scipy.sparse import block_diag, csr_matrix, diags

import structsolve.arpackutils as au
from structsolve import freq, lb, pardiso
from structsolve.linear_buckling import (_check_eigenpairs, _eigh_condensed,
                                         _estimate_sigma, _negative_pivots,
                                         _sort_eigenpairs)
from test_linear_buckling import load_saved_matrices
from test_sparse_first import freq_matrices
from test_inertia import fake_perturbed_pivots

needs_pardiso = pytest.mark.skipif(not pardiso.available(),
                                   reason='oneMKL is not found')


def assert_same_eigenpairs(w1, v1, w2, v2, rtol=1e-8, max_angle=1e-5):
    """Same eigenvalues to ``rtol`` and the same eigenvectors up to the sign,
    or the same subspaces of the groups of equal eigenvalues; a group that
    reaches the last eigenvalue may be cut and is not compared"""
    np.testing.assert_allclose(w2, w1, rtol=rtol)
    n = len(w1)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(w1[j] - w1[i]) <= 1e-6*abs(w1[i]):
            j += 1
        if j < n or j - i == 1:
            angles = subspace_angles(v1[:, i:j], v2[:, i:j])
            assert angles.max() <= max_angle, (i, j, angles)
        i = j


@needs_pardiso
def test_lb_pardiso_as_splu():
    """The factorizations of PARDISO, accurate for the beam, without any
    fallback to SuperLU"""
    K, KG = load_saved_matrices('beam_cantilever')
    w1, v1 = lb(K, KG, silent=True, num_eigvalues=10)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        w2, v2 = lb(K, KG, silent=True, num_eigvalues=10,
                    linear_solver='pardiso')
    assert_same_eigenpairs(w1, v1, w2, v2, rtol=1e-7)


@needs_pardiso
@pytest.mark.filterwarnings('ignore:linear_solver=.pardiso.:RuntimeWarning')
@pytest.mark.parametrize('name', ['plate_ssss_Nxx', 'cylinder_5panels'])
def test_lb_pardiso_inaccurate_as_splu(name):
    """The Ritz models, whose factorizations by PARDISO may be inaccurate
    and then replaced by SuperLU, see the Notes of lb(): the load
    multipliers of SuperLU, to the accuracy that the check of the
    factorizations allows for these ill-conditioned matrices (with MKL
    2026.2 PARDISO is accepted for cylinder_5panels, 1.5e-8 apart)"""
    K, KG = load_saved_matrices(name)
    w1, v1 = lb(K, KG, silent=True, num_eigvalues=10)
    w2, v2 = lb(K, KG, silent=True, num_eigvalues=10, linear_solver='pardiso')
    assert_same_eigenpairs(w1, v1, w2, v2, rtol=1e-6, max_angle=1e-3)


@needs_pardiso
def test_negative_pivots_pardiso_inaccurate():
    """K + s KG of the plate close to its 10th load multiplier: PARDISO
    perturbs pivots, its fallback reports none but has a wrong inertia, and
    the count is that of SuperLU"""
    K, KG = load_saved_matrices('plate_ssss_Nxx')
    w, _ = lb(K, KG, silent=True, sparse_solver=False)
    A = (K + w[9]*(1 - 1e-4)*KG).tocsr()
    assert np.count_nonzero(np.linalg.eigvalsh(A.toarray()) < 0) == 9
    assert _negative_pivots(A, 'pardiso') == 9


@needs_pardiso
def test_freq_pardiso_as_splu():
    K, M = freq_matrices(True)
    w1, v1 = freq(K, M, silent=True, symmetric=True, num_eigvalues=10)
    w2, v2 = freq(K, M, silent=True, symmetric=True, num_eigvalues=10,
                  linear_solver='pardiso')
    assert_same_eigenpairs(w1, v1, w2, v2)


@needs_pardiso
def test_freq_pardiso_null_dofs():
    """The null rows and columns removed by freq() are not seen by PARDISO"""
    K, M = freq_matrices(True, n=60)
    K = block_diag((K, csr_matrix((3, 3)))).tocsr()
    M = block_diag((M, csr_matrix((3, 3)))).tocsr()
    w1, v1 = freq(K, M, silent=True, symmetric=True, num_eigvalues=8)
    w2, v2 = freq(K, M, silent=True, symmetric=True, num_eigvalues=8,
                  linear_solver='pardiso')
    assert_same_eigenpairs(w1, v1, w2, v2)


@needs_pardiso
def test_capped_eigsh_passes_pardiso(monkeypatch):
    """With the cap of ncv, the k > 15 eigenpairs of the passes with the
    factorizations of PARDISO, of the first call and of each pass"""
    monkeypatch.setattr(au, 'ARPACK_MAX_NCV', 32)
    calls = []
    scipy_eigsh = au._scipy_eigsh

    def spy(**kwargs):
        calls.append(kwargs.get('OPinv') is not None)
        return scipy_eigsh(**kwargs)

    factorizations = []
    pardiso_solve = au.pardiso_solve

    def spy_solve(A, refine=True):
        solve = pardiso_solve(A, refine)
        factorizations.append(solve.linear_solver)
        return solve

    monkeypatch.setattr(au, '_scipy_eigsh', spy)
    monkeypatch.setattr(au, 'pardiso_solve', spy_solve)
    K, KG = load_saved_matrices('beam_cantilever')
    k = 25
    mu_ref, _ = _eigh_condensed(K, KG, k)
    sigma = _estimate_sigma(K, KG)
    mu, v = au.capped_eigsh(A=KG, k=k, M=K, sigma=sigma, which='SM',
                            mode='cayley', linear_solver='pardiso')
    mu, v = _sort_eigenpairs(mu, v)
    assert calls == [True, True]
    assert factorizations == ['pardiso']*2
    np.testing.assert_allclose(mu, mu_ref, rtol=1e-6)
    assert _check_eigenpairs(K, KG, -1/mu, v, 1e-4,
                             linear_solver='pardiso') is None
    np.testing.assert_allclose(v.T @ (K @ v), np.eye(k), atol=1e-6)


@needs_pardiso
def test_lb_pardiso_passes_as_splu(monkeypatch):
    monkeypatch.setattr(au, 'ARPACK_MAX_NCV', 32)
    K, KG = load_saved_matrices('beam_cantilever')
    w1, v1 = lb(K, KG, silent=True, num_eigvalues=20)
    w2, v2 = lb(K, KG, silent=True, num_eigvalues=20, linear_solver='pardiso')
    assert_same_eigenpairs(w1, v1, w2, v2, rtol=1e-7, max_angle=1e-4)


def swapped_blocks(num_blocks=3):
    """Symmetric matrix of ``[[0, 1], [1, 0]]`` blocks, whose zero pivots
    force row interchanges in SuperLU, and of the diagonal ``2, -3``:
    ``num_blocks + 1`` negative eigenvalues"""
    swap = csr_matrix(np.array([[0., 1.], [1., 0.]]))
    return block_diag([swap]*num_blocks + [diags([2., -3.])]).tocsr()


@needs_pardiso
def test_negative_pivots_pardiso_row_interchange():
    A = swapped_blocks()
    assert _negative_pivots(A) is None
    assert _negative_pivots(A, 'pardiso') == 4
    # congruent, the same inertia
    D = diags(np.arange(1., A.shape[0] + 1))
    assert _negative_pivots(D @ A @ D, 'pardiso') == 4
    # SPD
    assert _negative_pivots(A @ A, 'pardiso') == 0


@needs_pardiso
def test_negative_pivots_pardiso_unknown(monkeypatch):
    """None for a null row, exactly singular, and the count of SuperLU for
    pivots perturbed also after the fallback"""
    A = swapped_blocks()
    assert _negative_pivots(block_diag((A, csr_matrix((1, 1)))).tocsr(),
                            'pardiso') is None
    fake_perturbed_pivots(monkeypatch, 3, 3)
    # SuperLU instead
    assert _negative_pivots(A, 'pardiso') is None
    assert _negative_pivots(A @ A, 'pardiso') == 0


@needs_pardiso
def test_pardiso_solve_inaccurate(monkeypatch):
    """SuperLU, with a warning, when the factorization of PARDISO is not
    accurate enough"""
    A = swapped_blocks() + 0.5*diags(np.ones(8))
    b = np.arange(1., A.shape[0] + 1)
    assert au.pardiso_solve(A).linear_solver == 'pardiso'
    monkeypatch.setattr(au, 'PARDISO_ACCURACY_RTOL', 0.)
    with pytest.warns(RuntimeWarning, match="'splu' is used instead"):
        solve = au.pardiso_solve(A)
    assert solve.linear_solver == 'splu'
    np.testing.assert_allclose(A @ solve(b), b, atol=1e-12)
    # the count of SuperLU
    assert _negative_pivots(A, 'pardiso') == 4


@needs_pardiso
def test_pardiso_solve_complex():
    A = swapped_blocks()
    b = np.arange(1., A.shape[0] + 1) + 1j*np.arange(A.shape[0])
    for refine in [True, False]:
        x = au.pardiso_solve(A, refine)(b)
        np.testing.assert_allclose(A @ x, b, atol=1e-12)


def test_invalid_linear_solver():
    K, KG = load_saved_matrices('plate_ssss_Nxx')
    with pytest.raises(ValueError, match='linear_solver'):
        lb(K, KG, silent=True, linear_solver='umfpack')
    with pytest.raises(ValueError, match='linear_solver'):
        freq(K, K, silent=True, symmetric=True, linear_solver='umfpack')
    with pytest.raises(ValueError, match='linear_solver'):
        au.capped_eigsh(A=KG, k=2, M=K, sigma=1., which='SM', mode='cayley',
                        linear_solver='umfpack')


def test_pardiso_missing_mkl(monkeypatch, tmp_path):
    monkeypatch.setattr(pardiso, '_mkl', None)
    monkeypatch.setenv(pardiso.MKL_RT_ENV, str(tmp_path/'missing_mkl_rt'))
    K, KG = load_saved_matrices('plate_ssss_Nxx')
    with pytest.raises(ImportError, match=pardiso.MKL_RT_ENV):
        lb(K, KG, silent=True, linear_solver='pardiso')
    with pytest.raises(ImportError, match=pardiso.MKL_RT_ENV):
        freq(K, K, silent=True, symmetric=True, linear_solver='pardiso')
