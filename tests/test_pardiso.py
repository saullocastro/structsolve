"""Tests of :class:`structsolve.pardiso.PardisoSolver`, the interface to
PARDISO of oneMKL; they need ``mkl_rt``, except those of its location

"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix, triu

import structsolve.pardiso as pardiso
from structsolve.pardiso import PardisoSolver, PardisoError
from test_inertia import dense_indefinite

needs_pardiso = pytest.mark.skipif(not pardiso.available(),
                                   reason='oneMKL is not found')


def upper(A):
    return triu(csr_matrix(A), format='csr')


@needs_pardiso
def test_factorize_solve_inertia():
    A = dense_indefinite(40, 3)
    s = PardisoSolver()
    assert not s.analyzed
    s.factorize(upper(A))
    assert s.analyzed and s.factorized
    assert s.iparm[22] == 3
    assert s.iparm[21] == 37
    b = np.random.default_rng(0).standard_normal((40, 2))
    x = s.solve(b)
    assert x.shape == (40, 2) and x.flags.c_contiguous
    np.testing.assert_allclose(A @ x, b, atol=1e-8)
    np.testing.assert_allclose(s.solve(b[:, 0]), x[:, 0], rtol=1e-12)
    s.free(everything=True)
    assert not s.analyzed and not s.factorized


@needs_pardiso
def test_phase_22_reuses_the_analysis():
    """The phase 22 of a matrix of the same pattern, after the memory of the
    factorization was freed and the analysis kept, gives the factorization
    of the phase 12"""
    A = dense_indefinite(40, 2)
    B = A + 0.01*np.eye(40)
    s = PardisoSolver()
    s.factorize(upper(A))
    s.free()
    assert s.analyzed and not s.factorized
    with pytest.raises(ValueError, match='no factorization'):
        s.solve(np.ones(40))
    s.factorize(upper(B), phase=22)
    r = PardisoSolver()
    r.factorize(upper(B))
    b = np.linspace(1., 2., 40)
    np.testing.assert_allclose(s.solve(b), r.solve(b), rtol=1e-10)
    assert s.iparm[22] == r.iparm[22]
    s.free(everything=True)
    r.free(everything=True)
    # free without any analysis does nothing
    s.free()


@needs_pardiso
def test_unsorted_indices():
    A = upper(dense_indefinite(10, 1))
    U = csr_matrix((A.data[::-1].copy(), A.indices[::-1].copy(), A.indptr),
                   shape=A.shape)
    # the rows reversed within each row: not sorted
    for i in range(10):
        a, b = A.indptr[i], A.indptr[i + 1]
        U.indices[a:b] = A.indices[a:b][::-1]
        U.data[a:b] = A.data[a:b][::-1]
    U.has_sorted_indices = False
    s = PardisoSolver()
    s.factorize(U)
    r = PardisoSolver()
    r.factorize(A)
    np.testing.assert_allclose(s.solve(np.ones(10)), r.solve(np.ones(10)),
                               rtol=1e-12)


@needs_pardiso
def test_errors():
    s = PardisoSolver()
    with pytest.raises(ValueError, match='square'):
        s.factorize(csr_matrix(np.ones((2, 3))))
    with pytest.raises(ValueError, match='float64'):
        s.factorize(csr_matrix(np.eye(3, dtype=np.float32)))
    E = csr_matrix(np.diag([1., 0., 1.]))
    E.eliminate_zeros()
    with pytest.raises(ValueError, match='empty row'):
        s.factorize(E)
    with pytest.raises(ValueError, match='phase 22'):
        s.factorize(upper(np.eye(3)), phase=22)
    with pytest.raises(ValueError, match='phase 12 or 22'):
        s.factorize(upper(np.eye(3)), phase=13)
    s.factorize(upper(np.eye(3)))
    with pytest.raises(ValueError, match='same pattern'):
        s.factorize(upper(np.ones((3, 3))), phase=22)
    with pytest.raises(ValueError, match='rows'):
        s.solve(np.ones(4))
    # an error code of PARDISO: the type of matrix
    s.free(everything=True)
    s.mtype = 99
    with pytest.raises(PardisoError, match='phase 12 with the error') as e:
        s.factorize(upper(np.eye(3)))
    assert e.value.phase == 12 and e.value.error < 0


def test_mkl_rt_not_found(monkeypatch, tmp_path):
    """The path of the environment variable is used, and a library without
    ``pardiso`` is not ``mkl_rt``"""
    monkeypatch.setattr(pardiso, '_mkl', None)
    monkeypatch.setenv(pardiso.MKL_RT_ENV, str(tmp_path/'missing_mkl_rt'))
    assert not pardiso.available()
    with pytest.raises(ImportError, match=pardiso.MKL_RT_ENV):
        PardisoSolver()


def test_mkl_rt_searched(monkeypatch, tmp_path):
    """Without the environment variable and ``find_library``, ``mkl_rt`` is
    searched under ``sys.prefix`` and then the user site"""
    monkeypatch.setattr(pardiso, '_mkl', None)
    monkeypatch.delenv(pardiso.MKL_RT_ENV, raising=False)
    monkeypatch.setattr(pardiso, 'find_library', lambda name: None)
    lib = tmp_path/'lib'/'sub'
    lib.mkdir(parents=True)
    (lib/'libmkl_rt.so').write_bytes(b'not a library')
    monkeypatch.setattr(pardiso.sys, 'prefix', str(tmp_path/'empty'))
    monkeypatch.setattr(pardiso.site, 'USER_BASE', str(tmp_path))
    with pytest.raises(ImportError):
        pardiso.load_mkl()
    assert pardiso._mkl is None


@needs_pardiso
def test_mkl_rt_loaded_once():
    assert pardiso.load_mkl() is pardiso.load_mkl()
