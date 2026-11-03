r"""Utilities for ARPACK, against the Intel MKL bug of ``dsteqr``

With Intel MKL 2024.2.0 to 2025.0.0, :func:`scipy.sparse.linalg.eigsh`
returns wrong eigenpairs, without any error, or raises ``ArpackError -8``,
whenever its number of Lanczos vectors is ``ncv > 32``, e.g. with SciPy's
default ``ncv = 2 k + 1`` for ``k >= 16`` eigenpairs. MKL 2025.0.1 or newer
is not affected, neither is :func:`scipy.sparse.linalg.eigs`.

:func:`capped_eigsh` keeps ``ncv <= 32`` when :func:`arpack_max_ncv` says so,
computing more than 15 eigenpairs in several calls with deflation, whose
eigenvectors are combined with a Rayleigh-Ritz projection such that they
are orthonormal also across the calls.

The cap is decided by :func:`arpack_max_ncv` and can be overridden with
the module attribute :data:`ARPACK_MAX_NCV` or the environment variable
``STRUCTSOLVE_ARPACK_MAX_NCV``, e.g. to lift it with a fixed MKL that cannot
be identified as such::

    import structsolve.arpackutils
    structsolve.arpackutils.ARPACK_MAX_NCV = 0  # no cap

Notes
-----
The symmetric drivers of ARPACK, used by :func:`scipy.sparse.linalg.eigsh`,
compute the Ritz vectors with the LAPACK routine ``dsteqr`` applied to the
``ncv x ncv`` tridiagonal matrix of the Lanczos process. The ``dsteqr`` of
Intel MKL 2024.2.0 to 2025.0.0 returns wrong eigenvectors for matrices
larger than 32 x 32, hence the wrong eigenpairs whenever ``ncv > 32``. The
non-symmetric drivers of :func:`scipy.sparse.linalg.eigs` do not call
``dsteqr``.

"""
import contextlib
import functools
import gc
import io
import os
import re
import sys
import warnings

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, splu
from scipy.sparse.linalg import eigsh as _scipy_eigsh

from .static import PARDISO_ACCURACY_RTOL as _static_accuracy_rtol

#: Maximum ``ncv`` of :func:`capped_eigsh`. ``None``, the default, reads the
#: environment variable ``STRUCTSOLVE_ARPACK_MAX_NCV`` and, when it is not
#: defined, uses ``'auto'``: the cap :data:`DSTEQR_MAX_SIZE` is applied when
#: the LAPACK of SciPy is detected as faulty, see :func:`arpack_max_ncv`.
#: ``0`` (or ``'none'``) disables the cap, an integer of at least ``3`` sets
#: it.
ARPACK_MAX_NCV = None

#: ``linear_solver='pardiso'``: largest relative residual of the solution
#: of a factorization of PARDISO, without refinement, for a random
#: right-hand side, above which SuperLU is used instead, see
#: :func:`pardiso_solve`, and the count of negative pivots of PARDISO is
#: not trusted, see :func:`.linear_buckling._negative_pivots`; that of
#: :func:`.static._factorize` for the factorizations of the non-linear
#: analyses
PARDISO_ACCURACY_RTOL = _static_accuracy_rtol

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

    Returns
    -------
    max_ncv : int or None
        The cap of ``ncv``, at least ``3``, or ``None`` when ``ncv`` is not
        capped.

    Raises
    ------
    ValueError
        When the value given is an integer lower than ``3`` and not ``0``.

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

    Parameters
    ----------
    k : int
        Number of eigenpairs.
    n : int
        Size of the problem, the number of rows of the matrices.

    Returns
    -------
    ncv : int
        Number of Lanczos vectors.

    """
    ncv = min(n, max(2*k + 1, 20))
    max_ncv = arpack_max_ncv()
    if max_ncv is not None:
        ncv = min(ncv, max_ncv)
    return ncv


def release_memory():
    """Collect the reference cycles of SciPy's ARPACK wrappers in the browser

    The objects of :func:`scipy.sparse.linalg.eigsh` and
    :func:`scipy.sparse.linalg.eigs` keep the SuperLU factorizations of the
    shift-invert modes in reference cycles, freed only by the cyclic garbage
    collector. Natively this is harmless, but the WebAssembly heap of
    Pyodide (``sys.platform == 'emscripten'``) is exhausted before the
    collector runs, e.g. after about ten ``eigs`` calls on a problem with
    1,000 dofs, raising ``MemoryError``. :func:`gc.collect` is then called
    after each ARPACK call; it does nothing on other platforms.

    """
    if sys.platform == 'emscripten':
        gc.collect()


def start_vector(n):
    """Fixed starting vector of ARPACK, uniform in ``[-1, 1]`` as ARPACK's
    random one, such that the results are reproducible

    The vector is drawn from :class:`numpy.random.RandomState` with the seed
    ``42``, so that the same ``n`` always gives the same vector. The
    calls of ARPACK use :func:`start_kwargs`, which also fixes the vectors
    drawn when ARPACK restarts.

    Parameters
    ----------
    n : int
        Size of the problem.

    Returns
    -------
    v0 : ndarray
        1-D array of size ``n``.

    """
    return np.random.RandomState(42).uniform(-1., 1., n)


@functools.lru_cache(maxsize=None)
def _arpack_takes_rng():
    """Tell whether :func:`scipy.sparse.linalg.eigs` and
    :func:`scipy.sparse.linalg.eigsh` accept the argument ``rng``, i.e.
    SciPy 1.17 or newer"""
    import inspect
    from scipy.sparse.linalg import eigs
    try:
        return 'rng' in inspect.signature(eigs).parameters
    except (TypeError, ValueError):
        return False


def start_kwargs(n):
    """Arguments of :func:`scipy.sparse.linalg.eigs` and
    :func:`scipy.sparse.linalg.eigsh` such that the results are
    reproducible

    ``v0`` is :func:`start_vector` and, with SciPy 1.17 or newer, ``rng``
    is a :class:`numpy.random.Generator` with the seed ``42``, from which
    ARPACK draws a new starting vector whenever the Krylov subspace is
    invariant, e.g. when the operator is nilpotent.

    Parameters
    ----------
    n : int
        Size of the problem.

    Returns
    -------
    kwargs : dict
        ``{'v0': start_vector(n)}``, with the key ``'rng'`` when supported.

    Notes
    -----
    Since SciPy 1.17, ARPACK asks the caller for a random vector when the
    Arnoldi or Lanczos process breaks down, i.e. finds an invariant
    subspace, and SciPy draws it from ``rng``. With ``rng=None`` it comes
    from the entropy of the operating system: with the same ``v0`` the
    results then differ from one call to the next. Older versions of SciPy
    draw it inside ARPACK, with a seed kept from one call to the next in
    the same process, which ``rng`` cannot fix.

    """
    kwargs = {'v0': start_vector(n)}
    if _arpack_takes_rng():
        kwargs['rng'] = np.random.default_rng(42)
    return kwargs


def _deflation_shift(w, far_shift):
    """Shift of the passes after the first one of :func:`capped_eigsh`

    The middle of the largest gap between the last five eigenvalues found,
    sorted by increasing value. ``far_shift`` when they are all equal, i.e.
    when their largest gap is not larger than ``1e-8`` times their largest
    modulus.

    Notes
    -----
    The shift-invert mode finds first the eigenvalues nearest the shift.
    The middle of the largest gap is close to the next eigenvalues and away
    from the deflated ones.

    """
    last = np.sort(w)[-5:]
    gaps = np.diff(last)
    if not gaps.size or gaps.max() <= 1e-8*np.abs(last).max():
        return far_shift
    i = np.argmax(gaps)
    return 0.5*(last[i] + last[i + 1])


def _rayleigh_ritz(A, M, V):
    r"""Rayleigh-Ritz projection of ``A x = w M x`` onto the columns of ``V``


    Returns the Ritz values and the ``M``-orthonormal Ritz vectors of the
    eigenvalue problem of the symmetric ``A`` and the positive definite
    ``M`` in the subspace spanned by the columns of ``V``, which must be
    linearly independent. Used by :func:`capped_eigsh` to combine the
    eigenvectors of its passes. The projected problem is solved with the
    LAPACK driver ``dsyevr`` (``driver='evr'``) of
    :func:`scipy.linalg.eigh`.

    Returns
    -------
    w, x : ndarray
        The Ritz values by increasing value and the Ritz vectors.

    Notes
    -----
    The eigenvectors of different passes of :func:`capped_eigsh` are only
    `\boldsymbol{M}`-orthogonal to the accuracy of ARPACK, e.g. `10^{-6}` between
    close eigenvalues of different passes. `\boldsymbol{V}` is `\boldsymbol{M}`-orthonormalized
    with the Cholesky factor `\boldsymbol{L}` of `\boldsymbol{V}^\top \boldsymbol{M} \boldsymbol{V}`, and the projected
    problem `\boldsymbol{W}^\top \boldsymbol{A} \boldsymbol{W} \boldsymbol{y} = w \boldsymbol{y}`, with `\boldsymbol{W} = \boldsymbol{V} \boldsymbol{L}^{-\top}`, is
    solved with the LAPACK driver ``dsyevr`` (``driver='evr'``), which does
    not use the ``dsteqr`` of Intel MKL broken above 32 x 32 matrices, see
    :func:`dsteqr_is_faulty`. The returned eigenvectors `\boldsymbol{W} \boldsymbol{y}` are
    `\boldsymbol{M}`-orthonormal to the rounding errors, and the Ritz values are at
    least as accurate as the eigenvalues of the passes. `\boldsymbol{V}^\top \boldsymbol{M} \boldsymbol{V}` and
    `\boldsymbol{W}^\top \boldsymbol{A} \boldsymbol{W}` are symmetrized, `\frac{1}{2}(\boldsymbol{B} + \boldsymbol{B}^\top)`, before the
    Cholesky factorization and the eigenvalue solution.

    The Ritz values are the optimal approximations of the eigenvalues that
    can be obtained from the subspace spanned by the columns of `\boldsymbol{V}`
    ([1]_, Chapter 11, "Approximations from a Subspace", Section 11.4,
    "Optimality", Eq. (11.5) and Theorem 11.4.1, p. 236): the `j`-th
    Ritz value is the minimum over the `j`-dimensional subspaces of the span
    of the maximum of the Rayleigh quotient, so it can only improve when the
    span grows, and the span contains the eigenvectors of every pass. The
    theorem is stated for the standard problem; it holds for the pencil in
    the `\boldsymbol{M}` inner product, through the Cholesky factor of `\boldsymbol{V}^\top \boldsymbol{M} \boldsymbol{V}`.

    References
    ----------
    .. [1] Parlett, B.N., The Symmetric Eigenvalue Problem, SIAM,
       Philadelphia, 1998. https://doi.org/10.1137/1.9781611971163

    """
    MV = M @ V
    L = cholesky(0.5*(V.T @ MV + MV.T @ V), lower=True)
    W = solve_triangular(L, V.T, lower=True).T
    H = W.T @ (A @ W)
    w, Y = eigh(0.5*(H + H.T), driver='evr')
    return w, W @ Y


def pardiso_factorization(A):
    r"""Factorization of the symmetric sparse matrix ``A`` by PARDISO and
    the relative residual of its solution for a random right-hand side

    ``A`` is factorized by :func:`.static._factorize` with
    ``linear_solver='pardiso'``: the symmetric indefinite factorization of
    PARDISO of the matrix scaled to a unit diagonal, with the fallback on
    perturbed pivots, see its Notes.

    Returns
    -------
    factorization : callable or None
        The ``solve_kT`` of :func:`.static._factorize`, with
        ``num_negative_eigvals`` and ``num_perturbed_pivots``; ``None`` when
        ``A`` has a null row, i.e. it is exactly singular, PARDISO failed,
        or ``A`` is unsymmetric for :func:`.static._factorize`.
    residual : float or None
        `\|\boldsymbol{A} \boldsymbol{x} - \boldsymbol{b}\|/\|\boldsymbol{b}\|`
        of the solution `\boldsymbol{x}` of the factorization, without
        refinement, for a fixed random `\boldsymbol{b}`; ``None`` with
        ``factorization``.

    """
    from .static import _factorize
    A = csr_matrix(A, dtype=np.float64)
    if np.count_nonzero(np.asarray(abs(A).sum(axis=1)).ravel()) < A.shape[0]:
        return None, None
    with warnings.catch_warnings():
        # the warning of an unsymmetric matrix, for which SuperLU is used
        warnings.simplefilter('ignore', RuntimeWarning)
        try:
            factorization = _factorize(A, linear_solver='pardiso')
        except RuntimeError:
            return None, None
    if factorization.linear_solver != 'pardiso':
        return None, None
    b = np.random.RandomState(42).randn(A.shape[0])
    residual = (np.linalg.norm(A @ factorization.unrefined(b) - b)
                /np.linalg.norm(b))
    return factorization, residual


def pardiso_solve(A, refine=True):
    r"""Solution function of the symmetric sparse matrix ``A`` factorized by
    PARDISO, for the eigenvalue analyses with ``linear_solver='pardiso'``

    ``A`` is factorized by :func:`pardiso_factorization`. With
    ``refine=True`` every solution whose relative residual is larger than
    ``static.PARDISO_REFINE_TOL`` is improved by GMRES, see
    :func:`.static._factorize`; with ``refine=False`` the solution of the
    factorization is returned as is, as by SuperLU. When PARDISO fails, or
    its factorization is inaccurate, i.e. the relative residual of
    :func:`pardiso_factorization` is larger than ``PARDISO_ACCURACY_RTOL``,
    ``A`` is factorized by SuperLU instead, with a :class:`RuntimeWarning`.

    Returns
    -------
    solve : callable
        ``solve(b)`` returns `\boldsymbol{A}^{-1} \boldsymbol{b}` for a real
        or complex ``b`` of shape ``(n,)``; its attribute ``linear_solver``
        is ``'pardiso'`` or ``'splu'``, the solver used, and its attribute
        ``factorization`` the ``solve_kT`` of :func:`.static._factorize`
        with PARDISO, ``None`` with SuperLU, which holds the memory of
        PARDISO, freed when it is deleted.

    Raises
    ------
    RuntimeError
        When SuperLU is used and ``A`` is exactly singular.
    ImportError
        When oneMKL is not found.

    """
    factorization, residual = pardiso_factorization(A)
    if factorization is not None and residual <= PARDISO_ACCURACY_RTOL:
        f = factorization if refine else factorization.unrefined
        linear_solver = 'pardiso'
    else:
        if factorization is not None:
            warnings.warn("linear_solver='pardiso': inaccurate factorization, "
                          "relative residual %1.1e with %d perturbed pivots, "
                          "'splu' is used instead"
                          % (residual, factorization.num_perturbed_pivots),
                          RuntimeWarning, stacklevel=2)
        factorization = None
        f = splu(csc_matrix(A)).solve
        linear_solver = 'splu'

    def solve(b):
        b = np.asarray(b).ravel()
        if np.iscomplexobj(b):
            return (f(np.ascontiguousarray(b.real))
                    + 1j*f(np.ascontiguousarray(b.imag)))
        return f(b)

    solve.linear_solver = linear_solver
    solve.factorization = factorization
    return solve


def _pardiso_opinv(A, dtype):
    """``LinearOperator`` of the solutions of ``A`` by :func:`pardiso_solve`,
    the ``OPinv`` of :func:`scipy.sparse.linalg.eigsh`"""
    solve = pardiso_solve(A)
    n = A.shape[0]
    return LinearOperator((n, n), dtype=dtype,
                          matvec=lambda x: solve(np.asarray(x, dtype=dtype)))


def capped_eigsh(A, k, M, sigma, which, mode='normal', tol=0,
                 linear_solver='splu'):
    r""":func:`scipy.sparse.linalg.eigsh` with ``ncv`` from :func:`arpack_ncv`

    Solves `\boldsymbol{A} \boldsymbol{x} = w \boldsymbol{M} \boldsymbol{x}` with a positive definite ``M`` about the
    shift ``sigma``, in the shift-invert (``mode='normal'``) or Cayley
    (``mode='cayley'``) mode, selecting the eigenvalues by increasing `w`,
    e.g. the Cayley mode with ``which='SM'`` and ``sigma`` larger than all
    `|w|`, or the shift-invert mode with ``which='LM'`` and ``sigma`` lower
    than all `w`.

    When ``ncv`` is capped at ``max_ncv``, see :func:`arpack_max_ncv`, and
    ``k`` is larger than ``(max_ncv - 1)//2``, i.e. 15, the eigenpairs are
    computed in passes of at most 15 eigenpairs, each with ``ncv=max_ncv``.
    The first pass is a call in the requested mode. The following passes
    use the shift-invert mode about the shift ``sigma_d`` given by
    :func:`_deflation_shift`, with the eigenvectors of the previous passes
    deflated from the operator, and each of them returns the next
    eigenvalues by increasing `w`. Each pass factorizes ``A - sigma_d*M``
    with SuperLU, and every call starts from :func:`start_kwargs`. The
    eigenvectors of all the passes are finally combined with a
    Rayleigh-Ritz projection, see :func:`_rayleigh_ritz`, such that they are
    ``M``-orthonormal also across the passes.

    With ``linear_solver='pardiso'``, ``A - sigma*M`` of the first call and
    ``A - sigma_d*M`` of the passes are factorized by PARDISO instead, see
    :func:`pardiso_solve`, and passed to :func:`scipy.sparse.linalg.eigsh`
    as ``OPinv``, `(\boldsymbol{A} - \sigma \boldsymbol{M})^{-1}` in both the
    shift-invert and the Cayley modes; SuperLU is used for a matrix whose
    factorization by PARDISO is inaccurate.

    Parameters
    ----------
    A : sparse matrix
        Square matrix of the left-hand side, of size ``n x n``.
    k : int
        Number of eigenpairs.
    M : sparse matrix
        Positive definite matrix of the right-hand side, of size ``n x n``.
    sigma : float
        Shift of the shift-invert or Cayley mode, in the units of `w`.
    which : str
        Eigenvalues to find, passed to :func:`scipy.sparse.linalg.eigsh`, e.g.
        ``'LM'`` or ``'SM'``.
    mode : str, optional
        ``'normal'`` (shift-invert, default) or ``'cayley'``, passed to
        :func:`scipy.sparse.linalg.eigsh`.
    tol : float, optional
        Relative accuracy of the eigenvalues, passed to
        :func:`scipy.sparse.linalg.eigsh`; ``0`` (default) means the machine
        precision.
    linear_solver : str, optional
        ``'splu'`` (default), the factorizations of SuperLU, or
        ``'pardiso'``, those of PARDISO, see above.

    Returns
    -------
    w, x : ndarray
        The eigenvalues and the `\boldsymbol{M}`-orthonormal eigenvectors, by
        increasing `w` when computed in passes. ``w`` is a 1-D array of size
        ``k`` and ``x`` an array of shape ``(n, k)``.

    See Also
    --------
    arpack_max_ncv, arpack_ncv, start_kwargs, start_vector

    Notes
    -----
    In the shift-invert mode ARPACK computes the eigenvalues
    `\nu = 1/(w - \sigma)` of largest modulus of the operator
    `(\boldsymbol{A} - \sigma \boldsymbol{M})^{-1} \boldsymbol{M}`, i.e. the `w` nearest `\sigma` ([1]_,
    Chapter 3, Section 3.2, Eq. (3.2.2)).
    The passes keep ``k <= (max_ncv - 1)//2``, i.e. ``ncv >= 2 k + 1``.
    With ``k`` close to ``ncv`` ARPACK converges very slowly, not at all, or
    misses eigenvalues, e.g. ``k=31`` and ``ncv=32``.

    The passes after the first one use the shift-invert mode about
    `\sigma_d`, the middle of the largest gap between the last five
    eigenvalues found, with the eigenvectors `\boldsymbol{V}` of the previous passes
    deflated from the operator:

    .. math::

        \boldsymbol{y} = (\boldsymbol{I} - \boldsymbol{V} \boldsymbol{V}^\top \boldsymbol{M})(\boldsymbol{A} - \sigma_d \boldsymbol{M})^{-1} \boldsymbol{x}

    such that the eigenvalues of `\boldsymbol{V}`, already found, become null and are
    not found again. Each pass therefore returns the eigenvalues nearest
    `\sigma_d` among those not found yet, i.e. the next ones by increasing
    `w`. A shift close to them, instead of ``sigma``, separates them better:
    in the study of the panels cylinders it was 30 to 45% faster and gave
    eigenvalues 5 to 10 times more accurate. ARPACK applies the operator
    to `\boldsymbol{M} \boldsymbol{x}`; for an eigenvector `\boldsymbol{v}_i` of `\boldsymbol{V}`,
    `(\boldsymbol{A} - \sigma_d \boldsymbol{M})^{-1} \boldsymbol{M} \boldsymbol{v}_i = \boldsymbol{v}_i/(w_i - \sigma_d)` and,
    `\boldsymbol{V}` being `\boldsymbol{M}`-orthonormal, `(\boldsymbol{I} - \boldsymbol{V} \boldsymbol{V}^\top \boldsymbol{M}) \boldsymbol{v}_i = \boldsymbol{0}`.

    References
    ----------
    .. [1] Lehoucq, R.B., Sorensen, D.C., Yang, C., ARPACK Users' Guide:
       Solution of Large-Scale Eigenvalue Problems with Implicitly
       Restarted Arnoldi Methods, SIAM, Philadelphia, 1998.
       https://doi.org/10.1137/1.9780898719628

    """

    if linear_solver not in ('splu', 'pardiso'):
        raise ValueError('Invalid linear_solver: %s' % linear_solver)
    n = A.shape[0]
    dtype = np.result_type(A.dtype, M.dtype)
    opinv = {}
    if linear_solver == 'pardiso':
        opinv['OPinv'] = _pardiso_opinv(A - sigma*M, dtype)
    max_ncv = arpack_max_ncv()
    max_k = n if max_ncv is None else (max_ncv - 1)//2
    if k <= max_k or min(n, max(2*k + 1, 20)) <= max_ncv:
        result = _scipy_eigsh(A=A, k=k, M=M, sigma=sigma, which=which,
                              mode=mode, tol=tol, ncv=arpack_ncv(k, n),
                              **opinv, **start_kwargs(n))
        del opinv
        release_memory()
        return result
    num_passes = -(-k//max_k)
    k_pass = -(-k//num_passes)
    ncv = min(n, max_ncv)
    w, V = _scipy_eigsh(A=A, k=k_pass, M=M, sigma=sigma, which=which,
                        mode=mode, tol=tol, ncv=ncv, **opinv,
                        **start_kwargs(n))
    del opinv
    release_memory()
    far_shift = -sigma if mode == 'cayley' else sigma
    while w.shape[0] < k:
        order = np.argsort(w, kind='stable')
        w, V = w[order], V[:, order]
        sigma_d = _deflation_shift(w, far_shift)
        if linear_solver == 'pardiso':
            solve = pardiso_solve(A - sigma_d*M)
        else:
            lu = splu(csc_matrix(A - sigma_d*M))
            solve = lu.solve
        MV = M @ V

        def matvec(x, V=V, MV=MV, solve=solve):
            y = solve(np.asarray(x, dtype=dtype).ravel())
            return y - V @ (MV.T @ y)

        OPinv = LinearOperator((n, n), matvec=matvec, dtype=dtype)
        wi, Vi = _scipy_eigsh(A=A, k=min(k_pass, k - w.shape[0]), M=M,
                              sigma=sigma_d, which='LM', mode='normal',
                              tol=tol, ncv=ncv, OPinv=OPinv,
                              **start_kwargs(n))
        w = np.concatenate((w, wi))
        V = np.hstack((V, Vi))
        release_memory()
    return _rayleigh_ritz(A, M, V)
