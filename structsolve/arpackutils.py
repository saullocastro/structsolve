r"""Utilities for ARPACK, against the Intel MKL bug of ``dsteqr``

The symmetric drivers of ARPACK, used by :func:`scipy.sparse.linalg.eigsh`,
compute the Ritz vectors with the LAPACK routine ``dsteqr`` applied to the
``ncv x ncv`` tridiagonal matrix of the Lanczos process. The ``dsteqr`` of
Intel MKL 2024.2.0 to 2025.0.0 returns wrong eigenvectors for matrices
larger than 32 x 32, such that :func:`scipy.sparse.linalg.eigsh` returns
wrong eigenpairs, without any error, or raises ``ArpackError -8``, whenever
``ncv > 32``, e.g. with SciPy's default ``ncv = 2 k + 1`` for ``k >= 16``.
MKL 2025.0.1 or newer is not affected, neither are the non-symmetric drivers
of :func:`scipy.sparse.linalg.eigs`, which do not call ``dsteqr``.

:func:`capped_eigsh` keeps ``ncv <= 32`` when :func:`arpack_max_ncv` says so,
computing more than 15 eigenpairs in several calls with deflation.

The cap is decided by :func:`arpack_max_ncv` and can be overridden with
the module attribute :data:`ARPACK_MAX_NCV` or the environment variable
``STRUCTSOLVE_ARPACK_MAX_NCV``, e.g. to lift it with a fixed MKL that cannot
be identified as such::

    import structsolve.arpackutils
    structsolve.arpackutils.ARPACK_MAX_NCV = 0  # no cap

"""
import contextlib
import functools
import io
import os
import re
import warnings

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, splu
from scipy.sparse.linalg import eigsh as _scipy_eigsh

#: Maximum ``ncv`` of :func:`capped_eigsh`. ``None``, the default, reads the
#: environment variable ``STRUCTSOLVE_ARPACK_MAX_NCV`` and, when it is not
#: defined, uses ``'auto'``: the cap :data:`DSTEQR_MAX_SIZE` is applied when
#: the LAPACK of SciPy is detected as faulty, see :func:`arpack_max_ncv`.
#: ``0`` (or ``'none'``) disables the cap, an integer of at least ``3`` sets
#: it.
ARPACK_MAX_NCV = None

#: Largest tridiagonal matrix for which the faulty ``dsteqr`` is correct
DSTEQR_MAX_SIZE = 32

#: First and last Intel MKL versions with the faulty ``dsteqr``
MKL_FAULTY_VERSIONS = ((2024, 2, 0), (2025, 0, 0))


def _lapack_name_from_config(module):
    """Name of the LAPACK library in the build configuration of ``module``
    (``scipy`` or ``numpy``), ``None`` when unknown"""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            config = module.show_config(mode='dicts')
        name = config['Build Dependencies']['lapack']['name']
        return str(name).lower() if name else None
    except Exception:
        pass
    # builds with numpy.distutils, e.g. SciPy < 1.9
    config = getattr(module, '__config__', None)
    if config is None:
        return None
    for key in ('lapack_mkl_info', 'blas_mkl_info'):
        info = getattr(config, key, None)
        if info is None and hasattr(config, 'get_info'):
            try:
                info = config.get_info(key)
            except Exception:
                info = None
        if info:
            return 'mkl'
    return None


@functools.lru_cache(maxsize=None)
def lapack_is_mkl():
    """Tell whether the LAPACK of SciPy is Intel MKL

    ``scipy.show_config(mode='dicts')`` is read first, NumPy's build
    configuration only when SciPy's does not name the LAPACK library. Older
    versions of SciPy without ``mode='dicts'`` are checked through
    ``scipy.__config__``.

    Returns
    -------
    is_mkl : bool or None
        ``None`` when the build configuration does not tell.

    """
    import numpy
    import scipy
    for module in (scipy, numpy):
        name = _lapack_name_from_config(module)
        if name is not None:
            return 'mkl' in name
    return None


def _parse_mkl_version(text):
    found = re.search(r'(\d{4})\.(\d+)(?:\.(\d+))?', str(text))
    if found is None:
        return None
    return tuple(int(g) if g is not None else 0 for g in found.groups())


@functools.lru_cache(maxsize=None)
def mkl_version():
    """Version of the Intel MKL loaded in the process, as a tuple
    ``(major, minor, update)``

    The build configuration of SciPy only gives the major version, e.g.
    ``'2025'``, therefore the version is read at runtime from the optional
    packages ``mkl-service`` (``mkl.get_version()``) or ``threadpoolctl``.

    Returns
    -------
    version : tuple or None
        ``None`` when neither package is installed or no MKL is loaded.

    """
    try:
        import mkl
        v = mkl.get_version()
        return (int(v['MajorVersion']), int(v['MinorVersion']),
                int(v['UpdateVersion']))
    except Exception:
        pass
    try:
        import threadpoolctl
        for info in threadpoolctl.threadpool_info():
            if info.get('internal_api') == 'mkl':
                return _parse_mkl_version(info.get('version'))
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=None)
def dsteqr_is_faulty():
    """Tell whether the LAPACK ``dsteqr`` of SciPy returns wrong eigenvectors

    ``scipy.linalg.lapack.dstev`` computes the eigenvectors of a symmetric
    tridiagonal matrix with ``dsteqr``. It is called for three random
    matrices of sizes 33, 40 and 64, whose eigenpairs must have relative
    residuals and an orthogonality error below ``1e-10``. With Intel MKL
    2025.0.0 every size above 32 fails, with rare exceptions for sizes close
    to 32, hence the three sizes.

    Returns
    -------
    faulty : bool or None
        ``None`` when ``dstev`` is not available.

    """
    try:
        from scipy.linalg.lapack import dstev
    except ImportError:
        return None
    rng = np.random.RandomState(0)
    for n in (33, 40, 64):
        d = rng.randn(n)
        e = rng.randn(n - 1)
        try:
            with np.errstate(all='ignore'):
                w, z, info = dstev(d, e, compute_v=1)
                T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
                residual = np.linalg.norm(T @ z - z*w)/np.linalg.norm(T)
                orthogonality = np.linalg.norm(z.T @ z - np.eye(n))
        except Exception:
            return True
        if not (info == 0 and residual <= 1e-10 and orthogonality <= 1e-10):
            return True
    return False


@functools.lru_cache(maxsize=None)
def _auto_max_ncv():
    if dsteqr_is_faulty():
        return DSTEQR_MAX_SIZE
    if lapack_is_mkl():
        version = mkl_version()
        first, last = MKL_FAULTY_VERSIONS
        #NOTE the probe of dsteqr_is_faulty() passed, but the version of an
        #     MKL build must be known to lift the cap
        if version is None or first <= version <= last:
            return DSTEQR_MAX_SIZE
    return None


def arpack_max_ncv():
    r"""Maximum ``ncv`` of :func:`capped_eigsh`, ``None`` when not capped

    The value of :data:`ARPACK_MAX_NCV` is used, or, when it is ``None``,
    the environment variable ``STRUCTSOLVE_ARPACK_MAX_NCV``. Both accept
    ``'auto'``, ``0`` or ``'none'`` (no cap), or an integer of at least
    ``3``. In the ``'auto'`` mode, the default, ``ncv`` is capped at
    :data:`DSTEQR_MAX_SIZE` when:

    - :func:`dsteqr_is_faulty` is true; or
    - :func:`lapack_is_mkl` is true and :func:`mkl_version` is unknown or
      within :data:`MKL_FAULTY_VERSIONS`.

    The detection runs once per process.

    """
    value = ARPACK_MAX_NCV
    if value is None:
        value = os.environ.get('STRUCTSOLVE_ARPACK_MAX_NCV', 'auto')
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('', 'auto'):
            return _auto_max_ncv()
        if value in ('none', 'off', 'no', 'false'):
            return None
        value = int(value)
    if not value:
        return None
    if value < 3:
        raise ValueError('The maximum ncv of ARPACK must be 0 (no cap) or at '
                         'least 3, got {0}'.format(value))
    return int(value)


def arpack_ncv(k, n):
    r"""``ncv`` of :func:`scipy.sparse.linalg.eigsh` for ``k`` eigenpairs of
    a problem of size ``n``

    SciPy's default ``min(n, max(2 k + 1, 20))``, capped at
    :func:`arpack_max_ncv`. With a cap, ``k`` should not exceed
    ``(max_ncv - 1)//2``, see :func:`capped_eigsh`.

    """
    ncv = min(n, max(2*k + 1, 20))
    max_ncv = arpack_max_ncv()
    if max_ncv is not None:
        ncv = min(ncv, max_ncv)
    return ncv


def start_vector(n):
    """Fixed starting vector of ARPACK, uniform in ``[-1, 1]`` as ARPACK's
    random one, such that the results are reproducible"""
    return np.random.RandomState(42).uniform(-1., 1., n)


def capped_eigsh(A, k, M, sigma, which, mode='normal', tol=0):
    r""":func:`scipy.sparse.linalg.eigsh` with ``ncv`` from :func:`arpack_ncv`

    Solves `[A]\{x\} = w [M]\{x\}` with a positive definite ``M`` about the
    shift ``sigma``, in the shift-invert (``mode='normal'``) or Cayley
    (``mode='cayley'``) mode.

    When ``ncv`` is capped at ``max_ncv``, see :func:`arpack_max_ncv`, and
    ``k`` is larger than ``(max_ncv - 1)//2``, i.e. 15, the eigenpairs are
    computed in passes of at most 15 eigenpairs, each with ``ncv=max_ncv``.
    With ``k`` close to ``ncv`` ARPACK converges very slowly or not at all,
    e.g. ``k=31`` and ``ncv=32``. The first pass is a call in the requested
    mode. The following passes use the shift-invert mode about ``sigma``, or
    ``-sigma`` in the Cayley mode, with the eigenvectors `[V]` of the previous
    passes deflated from the operator:

    .. math::

        \{y\} = ([I] - [V][V]^T[M])([A] - \sigma_d [M])^{-1}\{x\}

    such that the eigenvalues of `[V]`, already found, become null and are
    not found again. Each pass therefore returns the eigenvalues nearest
    `\sigma_d` among those not found yet. For the Cayley mode with ``sigma``
    larger than all `|w|`, the order of selection is the same as in the
    first pass: by increasing `w`. One sparse LU factorization of
    `[A] - \sigma_d [M]` is shared by the following passes. Every call
    starts from :func:`start_vector`.

    Returns
    -------
    w, x : ndarray
        The eigenvalues and the `[M]`-orthonormal eigenvectors, in the order
        of the passes.

    """
    n = A.shape[0]
    max_ncv = arpack_max_ncv()
    max_k = n if max_ncv is None else (max_ncv - 1)//2
    if k <= max_k or min(n, max(2*k + 1, 20)) <= max_ncv:
        return _scipy_eigsh(A=A, k=k, M=M, sigma=sigma, which=which,
                            mode=mode, tol=tol, ncv=arpack_ncv(k, n),
                            v0=start_vector(n))
    num_passes = -(-k//max_k)
    k_pass = -(-k//num_passes)
    ncv = min(n, max_ncv)
    w, V = _scipy_eigsh(A=A, k=k_pass, M=M, sigma=sigma, which=which,
                        mode=mode, tol=tol, ncv=ncv, v0=start_vector(n))
    sigma_d = -sigma if mode == 'cayley' else sigma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lu = splu(csc_matrix(A - sigma_d*M))
    dtype = np.result_type(A.dtype, M.dtype)
    while w.shape[0] < k:
        MV = M @ V

        def matvec(x, V=V, MV=MV):
            y = lu.solve(np.asarray(x, dtype=dtype).ravel())
            return y - V @ (MV.T @ y)

        OPinv = LinearOperator((n, n), matvec=matvec, dtype=dtype)
        wi, Vi = _scipy_eigsh(A=A, k=min(k_pass, k - w.shape[0]), M=M,
                              sigma=sigma_d, which='LM', mode='normal',
                              tol=tol, ncv=ncv, OPinv=OPinv,
                              v0=start_vector(n))
        w = np.concatenate((w, wi))
        V = np.hstack((V, Vi))
    return w, V
