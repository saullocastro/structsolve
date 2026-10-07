"""Tests for structsolve.arpackutils: ncv <= 32 against the dsteqr bug of
Intel MKL 2024.2.0 to 2025.0.0"""
import os

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import csc_matrix, diags, load_npz
from scipy.sparse.linalg import ArpackError, LinearOperator, eigs, eigsh, splu

import structsolve.arpackutils as arpackutils
from structsolve import freq
from structsolve.linear_buckling import (_check_eigenpairs, _eigh_condensed,
                                         _estimate_sigma, _sort_eigenpairs)

DATA = os.path.join(os.path.dirname(__file__), 'data')


def load_saved_matrices(name):
    K = load_npz(os.path.join(DATA, '%s_K.npz' % name)).tocsr()
    KG = load_npz(os.path.join(DATA, '%s_KG.npz' % name)).tocsr()
    return K, KG


@pytest.fixture
def detection(monkeypatch):
    """Mock the inputs of the automatic cap of ncv"""
    def set_detection(faulty, is_mkl, version):
        monkeypatch.setattr(arpackutils, 'dsteqr_is_faulty', lambda: faulty)
        monkeypatch.setattr(arpackutils, 'lapack_is_mkl', lambda: is_mkl)
        monkeypatch.setattr(arpackutils, 'mkl_version', lambda: version)
        monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', None)
        monkeypatch.delenv('STRUCTSOLVE_ARPACK_MAX_NCV', raising=False)
        arpackutils._auto_max_ncv.cache_clear()
    yield set_detection
    arpackutils._auto_max_ncv.cache_clear()


@pytest.mark.parametrize('faulty, is_mkl, version, max_ncv', [
    (True, None, None, 32),
    (True, False, None, 32),
    (False, True, None, 32),
    (False, True, (2024, 2, 0), 32),
    (False, True, (2025, 0, 0), 32),
    (False, True, (2024, 1, 0), None),
    (False, True, (2025, 0, 1), None),
    (False, True, (2025, 3, 0), None),
    (False, False, None, None),
    (False, None, None, None),
    (None, None, None, None),
])
def test_arpack_max_ncv_auto(detection, faulty, is_mkl, version, max_ncv):
    """The cap applies to a faulty dsteqr and to any MKL build that is not
    known to be outside 2024.2.0 to 2025.0.0"""
    detection(faulty, is_mkl, version)
    assert arpackutils.arpack_max_ncv() == max_ncv


@pytest.mark.parametrize('capped', [False, True])
def test_arpack_ncv(detection, capped):
    """SciPy's default ncv, capped at 32 only when the detection says so"""
    detection(capped, capped, None)
    assert arpackutils.arpack_ncv(5, 1000) == 20
    assert arpackutils.arpack_ncv(15, 1000) == 31
    assert arpackutils.arpack_ncv(25, 1000) == (32 if capped else 51)
    assert arpackutils.arpack_ncv(25, 30) == 30
    assert arpackutils.arpack_ncv(3, 10) == 10


def test_arpack_max_ncv_override(detection, monkeypatch):
    """The module attribute takes precedence over the environment variable,
    which takes precedence over the automatic detection"""
    detection(True, True, (2025, 0, 0))
    assert arpackutils.arpack_max_ncv() == 32
    for value, expected in [('0', None), ('none', None), ('48', 48),
                            ('auto', 32), ('', 32)]:
        monkeypatch.setenv('STRUCTSOLVE_ARPACK_MAX_NCV', value)
        assert arpackutils.arpack_max_ncv() == expected
    monkeypatch.setenv('STRUCTSOLVE_ARPACK_MAX_NCV', '0')
    for value, expected in [(0, None), (40, 40), ('auto', 32), (None, None)]:
        monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', value)
        assert arpackutils.arpack_max_ncv() == expected
    monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', 2)
    with pytest.raises(ValueError):
        arpackutils.arpack_max_ncv()


def test_lapack_name_from_config():
    """The LAPACK of the build configuration, robust to missing keys and to
    the numpy.distutils builds of older SciPy versions"""
    from types import SimpleNamespace as NS

    def show_config(lapack):
        def f(mode='stdout'):
            assert mode == 'dicts'
            return {'Build Dependencies': {'lapack': lapack}}
        return f

    name = arpackutils._lapack_name_from_config
    assert name(NS(show_config=show_config(
        {'name': 'mkl-sdl', 'version': '2025'}))) == 'mkl-sdl'
    assert name(NS(show_config=show_config({'name': 'scipy-openblas'}))) \
        == 'scipy-openblas'
    assert name(NS(show_config=show_config({}))) is None

    def old_show_config():
        print('old style')
    assert name(NS(show_config=old_show_config)) is None
    assert name(NS(show_config=old_show_config, __config__=NS(
        lapack_mkl_info={'libraries': ['mkl_rt']}))) == 'mkl'
    assert name(NS(show_config=old_show_config, __config__=NS(
        lapack_mkl_info={}, blas_mkl_info={}))) is None
    assert isinstance(arpackutils.lapack_is_mkl(), (bool, type(None)))


def test_parse_mkl_version():
    parse = arpackutils._parse_mkl_version
    assert parse('2025.0-Product') == (2025, 0, 0)
    assert parse('Version 2024.2.2-Product Build 20240823') == (2024, 2, 2)
    assert parse('unknown') is None


def faulty_environment():
    """MKL 2024.2.0 to 2025.0.0 identified at runtime"""
    version = arpackutils.mkl_version()
    first, last = arpackutils.MKL_FAULTY_VERSIONS
    return version is not None and first <= version <= last


def test_dsteqr_probe_agrees_with_mkl_version():
    """The functional probe of dsteqr detects the faulty MKL versions"""
    if faulty_environment():
        assert arpackutils.dsteqr_is_faulty() is True
        assert arpackutils.arpack_max_ncv() == 32
    else:
        assert arpackutils.dsteqr_is_faulty() in (False, None)


@pytest.mark.parametrize('name', ['plate_ssss_Nxx', 'cylinder_5panels'])
def test_eigsh_ncv_boundary(name):
    """Root cause: eigsh in Cayley mode is correct with ncv=32 and, with a
    faulty dsteqr, wrong or raising with ncv=33, i.e. as soon as ARPACK
    computes the eigenvectors of a tridiagonal matrix larger than 32 x 32"""
    K, KG = load_saved_matrices(name)
    mu_ref, _ = _eigh_condensed(K, KG, 10)
    sigma = _estimate_sigma(K, KG)

    def solve(ncv):
        mu, v = eigsh(A=KG, M=K, k=10, ncv=ncv, which='SM', sigma=sigma,
                      mode='cayley', tol=0,
                      v0=arpackutils.start_vector(K.shape[0]))
        mu, v = _sort_eigenpairs(mu, v)
        error = _check_eigenpairs(K, KG, -1/mu, v, 1e-3)
        return error is None and np.allclose(mu, mu_ref, rtol=1e-6)

    assert solve(32)
    if arpackutils.dsteqr_is_faulty():
        try:
            assert not solve(33)
        except ArpackError as e:
            assert 'dsteqr' in str(e)


@pytest.mark.parametrize('k', [16, 25, 40])
def test_capped_eigsh_passes(monkeypatch, k):
    """With the cap, k > 15 eigenpairs are computed in passes of at most 15,
    with ncv <= 32, giving the dense load multipliers and K-orthonormal
    eigenvectors"""
    import structsolve.arpackutils as au

    monkeypatch.setattr(au, 'ARPACK_MAX_NCV', 32)
    calls = []
    scipy_eigsh = au._scipy_eigsh

    def spy(**kwargs):
        calls.append((kwargs['k'], kwargs['ncv']))
        return scipy_eigsh(**kwargs)

    monkeypatch.setattr(au, '_scipy_eigsh', spy)
    K, KG = load_saved_matrices('cylinder_5panels')
    mu_ref, _ = _eigh_condensed(K, KG, k)
    sigma = _estimate_sigma(K, KG)
    mu, v = au.capped_eigsh(A=KG, k=k, M=K, sigma=sigma, which='SM',
                            mode='cayley')
    mu, v = _sort_eigenpairs(mu, v)
    num_passes = -(-k//15)
    assert len(calls) == num_passes
    assert all(kp <= 15 and ncv <= 32 for kp, ncv in calls)
    assert sum(kp for kp, ncv in calls) == k
    np.testing.assert_allclose(mu, mu_ref, rtol=1e-6)
    assert _check_eigenpairs(K, KG, -1/mu, v, 1e-4) is None
    np.testing.assert_allclose(v.T @ (K @ v), np.eye(k), atol=1e-6)


def test_capped_eigsh_single_call_without_cap(monkeypatch):
    """Without the cap, SciPy's default ncv in a single call, whose result is
    only correct when dsteqr is not faulty"""
    import structsolve.arpackutils as au

    monkeypatch.setattr(au, 'ARPACK_MAX_NCV', 0)
    calls = []
    scipy_eigsh = au._scipy_eigsh

    def spy(**kwargs):
        calls.append((kwargs['k'], kwargs['ncv']))
        return scipy_eigsh(**kwargs)

    monkeypatch.setattr(au, '_scipy_eigsh', spy)
    n = 200
    K = diags(np.linspace(1., 2., n)).tocsc()
    M = diags(np.ones(n)).tocsc()
    w, v = au.capped_eigsh(A=K, k=25, M=M, sigma=-1., which='LM')
    assert calls == [(25, 51)]
    if not arpackutils.dsteqr_is_faulty():
        np.testing.assert_allclose(np.sort(w), np.linspace(1., 2., n)[:25])


def test_eigs_not_affected():
    """The non-symmetric drivers of eigs() do not call dsteqr: eigs() on
    K^-1 KG with the default ncv=51 is correct with any LAPACK"""
    K, KG = load_saved_matrices('plate_ssss_Nxx')
    lu = splu(csc_matrix(K))
    n = K.shape[0]
    OP = LinearOperator((n, n), matvec=lambda x: lu.solve(KG @ x),
                        dtype=float)
    mu, v = eigs(OP, k=25, which='LM', tol=0)
    mu_ref = eigh(KG.toarray(), K.toarray(), eigvals_only=True)
    mu_ref = mu_ref[np.argsort(-np.abs(mu_ref))[:25]]
    np.testing.assert_allclose(np.sort(mu.real), np.sort(mu_ref), rtol=1e-8)
    assert np.all(np.abs(mu.imag) <= 1e-10*np.abs(mu))


def plate_freq_matrices():
    """K of the plate and a lumped, positive definite mass matrix"""
    K, _ = load_saved_matrices('plate_ssss_Nxx')
    rng = np.random.RandomState(3)
    M = diags(1. + rng.rand(K.shape[0])).tocsr()
    return K, M


def test_freq_symmetric_mkl():
    """freq(symmetric=True) uses eigsh() in shift-invert mode, which is also
    affected by the dsteqr bug: with the default ncv=51 the eigenpairs are
    wrong when dsteqr is faulty, freq() with the cap of ncv is correct"""
    from structsolve.freq import _estimate_sigma as estimate_sigma_freq

    K, M = plate_freq_matrices()
    w_ref = eigh(K.toarray(), M.toarray(), eigvals_only=True)[:25]
    sigma = estimate_sigma_freq(K, M)
    if arpackutils.dsteqr_is_faulty():
        try:
            w, v = eigsh(A=K, M=M, k=25, ncv=51, sigma=sigma, which='LM',
                         tol=0)
            res = (np.linalg.norm(K @ v - (M @ v)*w, axis=0)
                   / np.linalg.norm(K @ v, axis=0))
            assert not (np.allclose(np.sort(w), w_ref, rtol=1e-8)
                        and np.all(res < 1e-6))
        except ArpackError:
            pass
    for trial in range(3):
        lambda2, eigvecs = freq(K, M, silent=True, symmetric=True,
                                max_dense_size=0, sort=False)
        np.testing.assert_allclose(np.sort(-lambda2), w_ref, rtol=1e-8)
