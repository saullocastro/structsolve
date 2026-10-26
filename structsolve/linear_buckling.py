"""Linear buckling analysis (:func:`lb`)

Load multipliers and buckling modes from the constitutive and geometric
stiffness matrices, computed with the sparse eigenvalue solvers first and
with a dense solver as a last resort, with a :class:`DenseFallbackWarning`.
:func:`is_symmetric` selects the solvers of symmetric or unsymmetric
matrices.

"""
import sys
import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, splu, LinearOperator
from scipy.linalg import eigh, eig, LinAlgError

from .arpackutils import capped_eigsh, release_memory, start_kwargs
from .logger import msg, warn
from .sparseutils import remove_null_cols


class DenseFallbackWarning(RuntimeWarning):
    """Issued by :func:`lb` and :func:`.freq` when the sparse eigenvalue
    solvers failed and the dense solver was used as a last resort

    A subclass of :class:`RuntimeWarning`, issued with :func:`warnings.warn`,
    whose message gives the reasons of the failures of the sparse solvers.
    It can be turned into an error with
    ``warnings.simplefilter('error', DenseFallbackWarning)``.

    """


#NOTE eigenvalues per step of the shift search of unsymmetric problems,
#     at least: few eigenvalues give small discs and many steps
SHIFT_SEARCH_MIN_EIGVALUES = 6
#NOTE the reversed follower load cantilevers of the study, whose critical
#     load multiplier is 1e4 times the modulus of the multipliers near zero,
#     need up to 12 steps
SHIFT_SEARCH_MAX_STEPS = 40
SHIFT_SEARCH_MAX_GROWTH = 1e10
PARITY_REL_GAP = 1e-4
#: An eigenvalue `\mu` of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` with `|\mu|` not
#: larger than this fraction of the largest `|\mu|` is null, i.e. its load
#: multiplier is infinite up to the round-off, e.g. at the dofs where
#: `\boldsymbol{K}_G` is null; also the fraction of the scale of the null `\nu` of
#: :func:`_eigs_shifted`
NULL_MU_RTOL = 1e-12
#: An eigenvalue `\mu` of the dense solvers is also null when `|\mu|` is
#: not larger than this factor times its first-order perturbation bound
#: for a backward error of the machine precision, see
#: :func:`_null_mu_pencil`
NULL_MU_PERTURBATION_FACTOR = 10.
#: The load multipliers of each sign are counted with the inertia of
#: `\boldsymbol{K} + s \boldsymbol{K}_G` below the modulus `|s|`, this factor divided by the
#: largest `|\mu|`, i.e. this factor times the lowest modulus that a load
#: multiplier may have
INERTIA_COUNT_FACTOR = 1e6
#: An eigenpair of :func:`.freq` is a rigid-body mode, null up to the
#: round-off, when `\|\boldsymbol{K} \boldsymbol{u}\| + |\lambda^2| \, \|\boldsymbol{M} \boldsymbol{u}\|` is not larger
#: than this fraction of the matrix-norm scale `(\|\boldsymbol{K}\|_F + |\lambda^2| \,
#: \|\boldsymbol{M}\|_F)\|\boldsymbol{u}\|`, 100 times the machine precision, see
#: :func:`_relative_residuals`
RIGID_BODY_RTOL = 100*np.finfo(float).eps
#: An eigenpair of :func:`.freq` whose relative residual is larger than
#: ``check_rtol`` passes the residual check when its normwise backward error
#: is not larger than this value, the machine precision, i.e. when it is
#: accurate up to the round-off, see :func:`_backward_errors`
BACKWARD_ERROR_RTOL = np.finfo(float).eps


def _estimate_mu_max(K, KG, max_iter=50, rel_tol=1e-3):
    r"""Largest `|\mu|` of ``KG u = mu K u``, estimated by power
    iterations, see :func:`_estimate_sigma`, or ``None`` when ``K`` is
    singular, not positive definite, or the linear solution is
    inaccurate"""
    try:
        lu = splu(csc_matrix(K))
        x = np.random.RandomState(42).randn(K.shape[0])
        Kx = K @ x
        mu_max = 0.
        for i in range(max_iter):
            xKx = x @ Kx
            if not np.isfinite(xKx) or xKx <= 0:
                return None
            x = x / np.sqrt(xKx)
            rhs = KG @ x
            y = lu.solve(rhs)
            Ky = K @ y
            if i == 0:
                residual = np.linalg.norm(Ky - rhs)
                if not residual <= 1e-6 * np.linalg.norm(rhs):
                    return None
            # ||y||_K / ||x||_K, with ||x||_K = 1
            ratio = np.sqrt(abs(y @ Ky))
            if not np.isfinite(ratio) or ratio == 0:
                return None
            converged = i > 0 and ratio - mu_max <= rel_tol * ratio
            mu_max = max(mu_max, ratio)
            if converged:
                break
            x, Kx = y, Ky
        return mu_max
    except Exception:
        return None


def _estimate_sigma(K, KG, safety=10., max_iter=50, rel_tol=1e-3,
                   mu_max=False):
    r"""Shift of the Cayley mode used by :func:`lb`

    With ``sigma > 0``, ``eigsh(A=KG, M=K, sigma=sigma, mode='cayley',
    which='SM')`` returns the eigenvalues `\mu` of ``KG u = mu K u`` with the
    smallest `|(\mu + \sigma)/(\mu - \sigma)|`. This ordering selects the most
    negative `\mu`, i.e. the lowest positive load multipliers `-1/\mu`, only
    when `\sigma` is larger than `|\mu|` of these critical eigenvalues.
    Otherwise the eigenvalues closest to `-\sigma` are returned.

    The largest `|\mu|` is estimated with power iterations on
    `\boldsymbol{K}^{-1} \boldsymbol{K}_G`, whose norm ratio in the `\boldsymbol{K}`-norm increases towards the
    largest `|\mu|`, and multiplied by ``safety``. The default shift ``1.`` is
    returned when ``K`` is singular, not positive definite, or the linear
    solution is inaccurate. An estimate of :func:`_estimate_mu_max` may be
    given as ``mu_max``, e.g. to try another ``safety`` factor.

    """
    if mu_max is False:
        mu_max = _estimate_mu_max(K, KG, max_iter=max_iter, rel_tol=rel_tol)
    if mu_max is None:
        return 1.
    return safety * mu_max


def _negative_pivots(A):
    r"""Number of negative eigenvalues of the symmetric sparse matrix ``A``

    ``A`` is factorized with a symmetric permutation and without row
    interchanges, i.e. `\boldsymbol{\Pi} \boldsymbol{A} \boldsymbol{\Pi}^\top = \boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top` with the pivots `\boldsymbol{D}` on
    the diagonal of `\boldsymbol{U} = \boldsymbol{D} \boldsymbol{L}^\top`. By Sylvester's law of inertia, the
    number of negative pivots is the number of negative eigenvalues of
    ``A``.

    Returns
    -------
    num : int or None
        ``None`` when the count is unknown: ``A`` is exactly singular, or a
        zero pivot forced a row interchange.

    """
    try:
        lu = splu(csc_matrix(A), permc_spec='MMD_AT_PLUS_A',
                  diag_pivot_thresh=0., options=dict(SymmetricMode=True))
    except RuntimeError:
        # exactly singular
        return None
    if not np.array_equal(lu.perm_r, lu.perm_c):
        # a zero pivot forced a row interchange
        return None
    d = lu.U.diagonal()
    if not np.all(np.isfinite(d)) or np.any(d == 0):
        return None
    return int(np.count_nonzero(d < 0))


def _is_positive_definite(A):
    """Tells if the symmetric sparse matrix ``A`` is positive definite

    The Gaussian elimination of a positive definite matrix never finds a
    zero or negative pivot, see :func:`_negative_pivots`.

    """
    return _negative_pivots(A) == 0


def _lu_solve(lu, b):
    """``lu.solve(b)`` for a real or complex right-hand side ``b``

    The SuperLU factorization of a real matrix rejects a complex ``b``, whose
    real and imaginary parts are then solved separately.

    """
    b = np.asarray(b)
    if np.iscomplexobj(b):
        return (lu.solve(np.ascontiguousarray(b.real))
                + 1j*lu.solve(np.ascontiguousarray(b.imag)))
    return lu.solve(b)


def _check_eigenpairs(K, KG, eigvals, eigvecs, rtol, min_rel_gap=1e-4,
                      check_inertia=True, K_positive_definite=None):
    r"""Verify the eigenpairs of ``(K + lambda*KG) @ u = 0``


    The eigenpairs may be real or complex.

    - The relative residual
      ``||K u + lambda KG u|| / (||K u|| + |lambda| ||KG u||)`` of every
      eigenpair with a finite load multiplier ``lambda``, see
      :func:`_relative_residuals`, must not be larger than ``rtol``.
    - With ``check_inertia=True``, only for symmetric matrices, and when
      ``K`` is positive definite: no positive load multiplier is missing
      below the highest positive one of ``eigvals``, checked with the
      number of negative pivots of ``K + s KG`` for ``s`` slightly lower,
      by the relative gap ``min_rel_gap``. When this count
      is unknown, see :func:`_negative_pivots`, the weaker check that no
      load multiplier is missing below the lowest positive one is done. The
      complex load multipliers are then ignored. Missing negative load
      multipliers are not detected.

    Whether ``K`` is positive definite may be given as
    ``K_positive_definite``, a boolean or a function returning it, e.g. to
    verify several solutions with one factorization of ``K``.

    Returns
    -------
    error : str or None
        Description of the failed check, ``None`` when all checks passed.

    Notes
    -----
    - No load multiplier is missing below the highest positive `\lambda_n`
      of ``eigvals``: when ``K`` is positive definite, the number of
      negative pivots of `\boldsymbol{K} + s \boldsymbol{K}_G` with
      `s = \lambda_n (1 - min\_rel\_gap)`, i.e. the number of load
      multipliers in `(0, s)` by Sylvester's law of inertia, see
      :func:`_negative_pivots`, must be the number of positive ``eigvals``
      lower than `s`. When this count is unknown, the weaker check of the
      lowest positive `\lambda_1` is done: `\boldsymbol{K} + s \boldsymbol{K}_G` with
      `s = \lambda_1 (1 - min\_rel\_gap)` must be positive definite. Only for
      symmetric matrices, ``check_inertia=True``, the complex `\lambda` are
      ignored. Missing negative load multipliers are not detected.

    For symmetric `\boldsymbol{K}` and `\boldsymbol{K}_G` with a positive definite `\boldsymbol{K}`,
    `\boldsymbol{K} + s \boldsymbol{K}_G` is congruent to `\boldsymbol{I} + s \boldsymbol{K}^{-1/2} \boldsymbol{K}_G \boldsymbol{K}^{-1/2}`,
    whose eigenvalues are `1 + s \mu_i = 1 - s/\lambda_i`, with `\mu_i =
    -1/\lambda_i` the eigenvalues of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}`. For
    `s > 0`, `1 - s/\lambda_i < 0` exactly when `0 < \lambda_i < s`, hence
    the number of negative eigenvalues of `\boldsymbol{K} + s \boldsymbol{K}_G`, equal to its
    number of negative pivots by Sylvester's law of inertia, is the number
    of load multipliers in `(0, s)`. This is the Sturm sequence check of
    the eigenvalue solvers of structural analysis ([1]_, Section 10.2.2,
    p. 849, and Section 11.6, p. 954), where the number of negative
    elements of `\boldsymbol{D}` in the factorization `\boldsymbol{L} \boldsymbol{D} \boldsymbol{L}^\top` of `\boldsymbol{K} - \mu \boldsymbol{M}`
    is the number of eigenvalues of the pencil `(\boldsymbol{K}, \boldsymbol{M})` smaller than
    `\mu`. When only one
    positive load multiplier was returned, none is expected in `(0, s)` and
    the weaker check is done. The gap ``min_rel_gap`` keeps `s` away from
    `\lambda_n`, where `\boldsymbol{K} + s \boldsymbol{K}_G` is singular.

    References
    ----------
    .. [1] Bathe, K.J., Finite Element Procedures, 2nd ed., K.J. Bathe,
       Watertown, MA, 2014.

    """

    eigvals = np.asarray(eigvals)
    finite = np.isfinite(eigvals)
    if not np.any(finite):
        return 'no finite load multiplier was found'
    lam = eigvals[finite]
    u = eigvecs[:, finite]
    residual = _relative_residuals(K, KG, lam, u)
    if not np.all(residual <= rtol):
        i = np.argmax(np.nan_to_num(residual, nan=np.inf))
        return ('relative residual {0:.2e} > {1:.1e} for the load multiplier '
                '{2}'.format(residual[i], rtol, lam[i]))
    if not check_inertia:
        return None
    lam = lam[np.isreal(lam)].real
    positive = lam[lam > 0]
    if K_positive_definite is None:
        K_positive_definite = _is_positive_definite
    if positive.size and (K_positive_definite(K) if
                          callable(K_positive_definite) else
                          K_positive_definite):
        s = positive.max()*(1 - min_rel_gap)
        expected = int(np.count_nonzero(positive < s))
        count = _negative_pivots(K + s*KG) if expected else None
        if count is not None:
            if count > expected:
                return ('{0} load multiplier(s) lower than {1} are missing'
                        .format(count - expected, s))
            if count < expected:
                return ('{0} load multipliers lower than {1} were found, but '
                        'only {2} exist'.format(expected, s, count))
            return None
        s = positive.min()*(1 - min_rel_gap)
        if not _is_positive_definite(K + s*KG):
            return ('at least one load multiplier lower than {0} is missing'
                    .format(positive.min()))
    return None


def _frobenius(A):
    """Frobenius norm of a dense or sparse matrix"""
    data = A.data if hasattr(A, 'nnz') else np.asarray(A)
    return float(np.sqrt(np.sum(np.abs(data)**2)))


def _relative_residuals(A, B, lam, u, rigid_rtol=None):
    r"""Relative residuals of the eigenpairs of `(\boldsymbol{A} + \lambda \boldsymbol{B}) \boldsymbol{u}
    = \boldsymbol{0}`, used by :func:`_check_eigenpairs` and :func:`.freq`

    Parameters
    ----------
    A, B : sparse_matrix or ndarray
        Matrices, shape ``(n, n)``.
    lam : array_like
        Finite eigenvalues, real or complex, shape ``(k,)``.
    u : ndarray
        Eigenvectors, ``u[:, i]`` that of ``lam[i]``, shape ``(n, k)``.
    rigid_rtol : float or None, optional
        ``None`` gives the relative residual
        ``||A u + lam B u|| / (||A u|| + |lam| ||B u||)`` of every eigenpair.
        Otherwise the eigenpairs with
        ``||A u|| + |lam| ||B u|| <= rigid_rtol (||A||_F + |lam| ||B||_F)
        ||u||``, null up to the round-off, e.g. the rigid-body modes of
        :func:`.freq` with ``rigid_rtol=RIGID_BODY_RTOL``, give
        ``||A u + lam B u|| / ((||A||_F + |lam| ||B||_F) ||u||)``, not larger
        than ``rigid_rtol``.

    Returns
    -------
    residual : ndarray
        One relative residual per eigenpair, shape ``(k,)``, ``inf`` when
        the denominator is null.

    Notes
    -----
    The relative residual

    .. math::

        \frac{\|\boldsymbol{A} \boldsymbol{u} + \lambda \boldsymbol{B} \boldsymbol{u}\|}{\|\boldsymbol{A} \boldsymbol{u}\|
        + |\lambda| \, \|\boldsymbol{B} \boldsymbol{u}\|}

    compares the residual with the terms of the eigenvalue problem for the
    eigenvector itself. For an eigenvector in the null space of `\boldsymbol{A}` with
    a null eigenvalue, e.g. a rigid-body mode of :func:`.freq`, both the
    residual and the denominator are round-off errors, and their ratio is
    meaningless, of the order of 1. With ``rigid_rtol`` `= \epsilon_r`, an
    eigenpair with

    .. math::

        \|\boldsymbol{A} \boldsymbol{u}\| + |\lambda| \, \|\boldsymbol{B} \boldsymbol{u}\| \le \epsilon_r
        (\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F) \|\boldsymbol{u}\|

    is null up to the round-off: its residual is relative to the
    matrix-norm scale, i.e. it is the normwise backward error
    `\|r\|/((\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F) \|\boldsymbol{u}\|)`, with `r` the
    residual vector, not larger than `\epsilon_r`. Every other eigenpair
    keeps the relative residual, such that the check of a genuine eigenpair
    is never loosened.

    A backward error at the round-off does not tell that a genuine
    eigenpair is accurate when `\|\boldsymbol{A}\|_F` is set by stiff springs, e.g. the
    penalty connections of multi-domain models: the eigenpairs that satisfy
    the connections give `\|\boldsymbol{A} \boldsymbol{u}\|` many orders of magnitude smaller
    than `\|\boldsymbol{A}\|_F \|\boldsymbol{u}\|`. Measured with ``eigsh`` for the lowest load
    multipliers of a blade-stiffened panel with penalty connections,
    `\|\boldsymbol{K}\|_F = 5.1 \cdot 10^{12}` and `5.1 \cdot 10^{13}`, the inaccurate
    eigenpairs, with relative residuals of 1 to 5 % and eigenvalues up to 1 %
    too low, have backward errors of `3 \cdot 10^{-15}` to `3 \cdot
    10^{-13}` and `(\|\boldsymbol{K} \boldsymbol{u}\| + |\lambda| \, \|\boldsymbol{K}_G \boldsymbol{u}\|)/((\|\boldsymbol{K}\|_F
    + |\lambda| \, \|\boldsymbol{K}_G\|_F)\|\boldsymbol{u}\|) = 1.6 \cdot 10^{-13}` to `1.6 \cdot
    10^{-11}`, i.e. 720 times the machine precision and more. A floor
    `\epsilon_f (\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F)\|\boldsymbol{u}\|` of the
    denominator with `\epsilon_f = 10^{-8}` would accept them. The
    rigid-body modes of :func:`.freq`, with the sparse and the dense
    solvers, give at most 19 times the machine precision for the chains of
    springs of the tests and for free plates of ``panels`` with penalty
    connections up to `10^4` times the penalty constants of ``calc_kt_kr``,
    against at least 540 times for the genuine eigenpairs of these plates,
    inaccurate ones included, hence ``RIGID_BODY_RTOL``. :func:`lb` uses the
    relative residual only: a finite load multiplier is never null. The
    eigenpairs of :func:`.freq` whose relative residual is larger than
    ``check_rtol`` are also accepted when their backward error is at the
    round-off, see :func:`_backward_errors`.

    """
    Au = A @ u
    Bu = B @ u
    lam = np.asarray(lam)
    num = np.linalg.norm(Au + Bu*lam, axis=0)
    den = np.linalg.norm(Au, axis=0) + np.abs(lam)*np.linalg.norm(Bu, axis=0)
    if rigid_rtol is not None:
        scale = (_frobenius(A) + np.abs(lam)*_frobenius(B)
                 )*np.linalg.norm(u, axis=0)
        den = np.where(den <= rigid_rtol*scale, scale, den)
    residual = np.full(lam.shape, np.inf)
    np.divide(num, den, out=residual, where=den > 0)
    return residual


def _backward_errors(A, B, lam, u):
    r"""Normwise backward errors of the eigenpairs of `(\boldsymbol{A} + \lambda \boldsymbol{B})
    \boldsymbol{u} = \boldsymbol{0}`, used by :func:`.freq`

    Parameters
    ----------
    A, B : sparse_matrix or ndarray
        Matrices, shape ``(n, n)``.
    lam : array_like
        Finite eigenvalues, real or complex, shape ``(k,)``.
    u : ndarray
        Eigenvectors, ``u[:, i]`` that of ``lam[i]``, shape ``(n, k)``.

    Returns
    -------
    backward_error : ndarray
        ``||A u + lam B u|| / ((||A||_F + |lam| ||B||_F) ||u||)``, one per
        eigenpair, shape ``(k,)``, ``inf`` when the denominator is null.

    Notes
    -----
    The normwise backward error of an approximate eigenpair `(\lambda,
    \boldsymbol{u})`

    .. math::

        \eta = \frac{\|\boldsymbol{A} \boldsymbol{u} + \lambda \boldsymbol{B} \boldsymbol{u}\|}{(\|\boldsymbol{A}\|_F + |\lambda|
        \, \|\boldsymbol{B}\|_F) \|\boldsymbol{u}\|}

    is the smallest `\epsilon` such that `(\lambda, \boldsymbol{u})` is an exact
    eigenpair of `(\boldsymbol{A} + \Delta \boldsymbol{A}) + \lambda (\boldsymbol{B} + \Delta \boldsymbol{B})` with
    `\|\Delta \boldsymbol{A}\| \le \epsilon \|\boldsymbol{A}\|_F` and `\|\Delta \boldsymbol{B}\| \le \epsilon
    \|\boldsymbol{B}\|_F` ([1]_, Theorem 1, Eq. (2.3), with the tolerances `\|\boldsymbol{A}\|_F`
    and `\|\boldsymbol{B}\|_F`). The perturbations that attain it are of rank one, so
    that their Frobenius norm is their 2-norm, and for symmetric matrices and
    a real `\lambda` the minimum is the same with symmetric perturbations
    ([1]_, Theorem 2).

    The relative residual of :func:`_relative_residuals` is `\eta` times
    `(\|\boldsymbol{A}\|_F + |\lambda| \, \|\boldsymbol{B}\|_F)\|\boldsymbol{u}\|/(\|\boldsymbol{A} \boldsymbol{u}\| + |\lambda|
    \, \|\boldsymbol{B} \boldsymbol{u}\|)`, a ratio that is large for an eigenvalue close to zero
    of a nearly singular `\boldsymbol{A}`, e.g. a frequency analysis pre-stressed close
    to buckling, whose eigenvalue is then sensitive to the round-off of the
    entries of `\boldsymbol{A}`. :func:`.freq` accepts an eigenpair whose relative
    residual is larger than ``check_rtol`` when `\eta \le`
    ``BACKWARD_ERROR_RTOL`` `= \epsilon`, the machine precision: the
    eigenpair is exact for matrices that differ from `\boldsymbol{K}` and `\boldsymbol{M}` by
    perturbations of the order of the round-off of their entries, and it is
    as accurate as the matrices allow.

    Measured with SciPy 1.17.1 and Intel MKL 2025.0.0 for the cantilever of
    49 ``BeamC`` elements of ``pyfe3d`` pre-stressed to 0.9999 of its
    buckling load, ``tests/data/beam_cantilever_*.npz``, whose two lowest
    eigenvalues are `\omega_n^2 = 0.0895`: the relative residuals of these
    eigenpairs are `2.5 \cdot 10^{-6}` to `9.1 \cdot 10^{-6}`, larger than
    the default ``check_rtol=1e-6``, with ``eigsh``, ``eigs`` and the dense
    solvers, and `(\|\boldsymbol{K} \boldsymbol{u}\| + |\lambda^2| \, \|\boldsymbol{M} \boldsymbol{u}\|)/((\|\boldsymbol{K}\|_F
    + |\lambda^2| \, \|\boldsymbol{M}\|_F)\|\boldsymbol{u}\|) = 5.0 \cdot 10^3 \epsilon`, while
    their backward errors are `0.012 \epsilon` to `0.046 \epsilon`, i.e.
    ``BACKWARD_ERROR_RTOL`` is 22 times the largest one. The inaccurate
    eigenpairs of ``eigsh`` for the lowest load multipliers of the
    blade-stiffened panel with penalty connections of
    ``tests/data/blade_stiffened_penalty_*.npz``, see
    :func:`_relative_residuals`, and of six variants with other blade
    heights, numbers of terms and 1000 times the penalty constants of
    ``calc_kt_kr``, have relative residuals of `1.6 \cdot 10^{-3}` to
    `9.7 \cdot 10^{-2}` and backward errors of `14 \epsilon` to `1200
    \epsilon`: the smallest one is 14 times ``BACKWARD_ERROR_RTOL``, and they
    are rejected. The backward error of an inaccurate eigenpair is its
    relative residual times the ratio above, which decreases as the penalty
    constants increase; the smallest one, `14 \epsilon`, was measured with
    1000 times the penalty constants of ``calc_kt_kr``.

    :func:`lb` keeps the relative residual only. For the cantilever above,
    with `\boldsymbol{K} = \boldsymbol{K}_{C0} + f \lambda_{cr} \boldsymbol{K}_G`, the load multipliers
    `(1 - f) \lambda_{cr}` close to zero of the dense solver have relative
    residuals from `4.7 \cdot 10^{-7}` for `f = 0.9999` to `5.1 \cdot
    10^{-4}` for `f = 1 - 10^{-7}`, below its default ``check_rtol=1e-3``.
    With ``check_rtol=1e-3`` the backward error would only accept eigenpairs
    whose ratio above is lower than `1000 \epsilon`, the range of the penalty
    connections with 1000 times the penalty constants, where the inaccurate
    eigenpairs above lie.

    References
    ----------
    .. [1] Tisseur, F., Backward error and condition of polynomial eigenvalue
       problems, Linear Algebra and its Applications, 309(1-3), 339-361,
       2000. https://doi.org/10.1016/S0024-3795(99)00063-4

    """
    Au = A @ u
    Bu = B @ u
    lam = np.asarray(lam)
    num = np.linalg.norm(Au + Bu*lam, axis=0)
    scale = (_frobenius(A) + np.abs(lam)*_frobenius(B)
             )*np.linalg.norm(u, axis=0)
    backward_error = np.full(lam.shape, np.inf)
    np.divide(num, scale, out=backward_error, where=scale > 0)
    return backward_error


def _null_mu(mu, scale=None):
    r"""Tell which eigenvalues `\mu` are null, `|\mu| \le`
    ``NULL_MU_RTOL`` ``scale``, ``scale`` being the largest finite `|\mu|`
    when not given"""
    absmu = np.abs(np.asarray(mu))
    finite = np.isfinite(absmu)
    if scale is None:
        scale = absmu[finite].max() if np.any(finite) else 0.
    return finite & (absmu <= NULL_MU_RTOL*scale)


def _lb_order(mu, scale=None):
    r"""Order of the load multipliers `\lambda = -1/\mu` returned by
    :func:`lb`

    The positive real `\lambda` first, in increasing order, then the
    negative real `\lambda` by increasing modulus, i.e. the critical load
    multipliers of the reversed load first, then the complex `\lambda` by
    increasing modulus, and at the end the null `\mu`, see
    :func:`_null_mu`, and the eigenvalues that are not finite.

    """
    mu = np.asarray(mu)
    null = _null_mu(mu, scale)
    finite = np.isfinite(mu)
    real = np.isreal(mu) & finite & ~null
    re = np.real(mu)
    neg = np.flatnonzero(real & (re < 0))
    neg = neg[np.argsort(re[neg], kind='stable')]
    pos = np.flatnonzero(real & (re > 0))
    pos = pos[np.argsort(-re[pos], kind='stable')]
    cplx = np.flatnonzero(~np.isreal(mu) & finite & ~null)
    cplx = cplx[np.argsort(-np.abs(mu[cplx]), kind='stable')]
    first = np.concatenate((neg, pos, cplx))
    rest = np.setdiff1d(np.arange(mu.size), first)
    return np.concatenate((first, rest)).astype(int)


def _sort_eigenpairs(mu, eigvecs, scale=None):
    r"""Sort the eigenpairs of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` in the order of
    the load multipliers returned by :func:`lb`, see :func:`_lb_order`"""
    order = _lb_order(mu, scale)
    return mu[order], eigvecs[:, order]


def _eigh_condensed(K, KG, k):
    r"""Dense solution after condensing out the dofs where ``KG`` is null

    With the dofs split in `b`, where `\boldsymbol{K}_G` has non-null rows, and `a`,
    where it does not, the static condensation of the `a` dofs gives:

    .. math::

        (\boldsymbol{K}_{bb} - \boldsymbol{K}_{ba} \boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab}
         + \lambda \boldsymbol{K}_{G_{bb}}) \boldsymbol{u}_b = \boldsymbol{0}, \qquad
        \boldsymbol{u}_a = -\boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab} \boldsymbol{u}_b

    solved with :func:`scipy.linalg.eigh`, which requires a positive
    definite Schur complement. Returns the first ``k`` eigenvalues `\mu` of
    `K_G u = \mu K u` in the order of :func:`_sort_eigenpairs`, i.e. the
    positive load multipliers first, and the corresponding eigenvectors.
    The null `\mu` of :func:`_null_mu_pencil` are set to 0.

    """
    active = np.asarray(abs(KG).sum(axis=1)).ravel() > 0
    b = np.flatnonzero(active)
    a = np.flatnonzero(~active)
    if b.size == 0:
        raise ValueError('KG is a null matrix')
    K = csr_matrix(K)
    KGbb = csr_matrix(KG)[b, :][:, b].toarray()
    S = K[b, :][:, b].toarray()
    if a.size:
        Kab = csc_matrix(K[a, :][:, b])
        Kba = K[b, :][:, a]
        lu = splu(csc_matrix(K[a, :][:, a]))
        # solving blocks of right-hand sides with at most ~1e7 entries
        chunk = max(1, int(1e7) // a.size)
        for j0 in range(0, b.size, chunk):
            j1 = min(j0 + chunk, b.size)
            X = lu.solve(Kab[:, j0:j1].toarray())
            S[:, j0:j1] -= Kba @ X
    KGbb = 0.5*(KGbb + KGbb.T)
    S = 0.5*(S + S.T)
    mu, ub = eigh(a=KGbb, b=S)
    mu[_null_mu_pencil(KGbb, S, mu, ub, ub)] = 0.
    mu, ub = _sort_eigenpairs(mu, ub)
    k = min(k, b.size)
    ub = ub[:, :k]
    eigvecs = np.zeros((K.shape[0], k), dtype=ub.dtype)
    eigvecs[b, :] = ub
    if a.size:
        eigvecs[a, :] = -lu.solve(np.asarray(Kab @ ub))
    return mu[:k], eigvecs


def is_symmetric(A, rtol=1e-10):
    r"""Tell whether ``||A - A^T|| <= rtol ||A||``, in the Frobenius norm

    A cheap check, of the order of the number of non-zero entries of ``A``,
    used by :func:`lb` to select the solvers of symmetric or unsymmetric
    matrices.

    Parameters
    ----------
    A : array or sparse matrix
        Square matrix, converted to CSR format.
    rtol : float, optional
        Relative tolerance, dimensionless. Default ``1e-10``.

    Returns
    -------
    symmetric : bool
        ``True`` when the Frobenius norm of ``A - A^T`` is not larger than
        ``rtol`` times the Frobenius norm of ``A``.

    See Also
    --------
    structsolve.sparseutils.is_symmetric : entry-wise check.

    """
    A = csr_matrix(A)
    normA = np.sqrt(np.sum(np.abs(A.data)**2))
    D = (A - A.T).tocsr()
    normD = np.sqrt(np.sum(np.abs(D.data)**2))
    return bool(normD <= rtol*normA)


def _real_if_close(mu, eigvecs, rtol=1e-8):
    r"""Real eigenpairs when the imaginary parts are negligible

    Each eigenvector of an eigenvalue whose imaginary part is below ``rtol``
    of its modulus is rotated in the complex plane to make its largest
    component real, and its imaginary part is dropped, as the imaginary part
    of the eigenvalue. Complex arrays are returned, see
    :func:`_general_eigenpairs`.

    """
    mu = np.array(mu, dtype=complex)
    eigvecs = np.array(eigvecs, dtype=complex)
    real = np.abs(mu.imag) <= rtol*np.abs(mu)
    mu[real] = mu[real].real
    for i in np.flatnonzero(real):
        v = eigvecs[:, i]
        j = np.argmax(np.abs(v))
        if v[j] != 0:
            v = v*(abs(v[j])/v[j])
        eigvecs[:, i] = v.real
    return mu, eigvecs


def _sort_eigenpairs_general(mu, eigvecs):
    r"""As :func:`_sort_eigenpairs`, for real or complex `\mu`: the
    positive real load multipliers first, then the negative real ones by
    increasing modulus, then the complex ones by increasing `|\lambda| =
    1/|\mu|`, see :func:`_lb_order`"""
    mu = np.asarray(mu)
    order = _lb_order(mu)
    return mu[order], eigvecs[:, order]


def _general_eigenpairs(mu, eigvecs, k=None):
    r"""The first ``k`` finite eigenpairs of a general eigenvalue solver

    The eigenvalues with a negligible imaginary part are made real with
    :func:`_real_if_close`, the finite eigenpairs are sorted with
    :func:`_sort_eigenpairs_general` and the first ``k`` are kept, all of
    them with ``k=None``. Real arrays are returned when the kept eigenpairs
    are real, even when discarded ones, e.g. spurious eigenvalues far from
    the critical ones, are complex.

    """
    mu = np.asarray(mu)
    finite = np.isfinite(mu)
    mu, eigvecs = _real_if_close(mu[finite], eigvecs[:, finite])
    mu, eigvecs = _sort_eigenpairs_general(mu, eigvecs)
    if k is not None:
        mu = mu[:k]
        eigvecs = eigvecs[:, :k]
    if np.all(mu.imag == 0):
        return mu.real, eigvecs.real
    return mu, eigvecs


def _null_mu_pencil(a, b, mu, vl, vr):
    r"""Tell which eigenvalues `\mu` of `[a]\boldsymbol{v} = \mu [b]\boldsymbol{v}` are null up
    to the round-off of a dense solver

    An eigenvalue is null when it is null for :func:`_null_mu`, or when
    `|\mu| \le c\,\epsilon\,\|[a]\|_F\,\kappa`, with `c` =
    :data:`NULL_MU_PERTURBATION_FACTOR`, `\epsilon` the machine precision
    and `\kappa = \|\boldsymbol{y}\| \|\boldsymbol{x}\| / |\boldsymbol{y}^H [b] \boldsymbol{x}|` the condition
    number of `\mu`, with its left and right eigenvectors `\boldsymbol{y}` and
    `\boldsymbol{x}`.

    Parameters
    ----------
    a, b : ndarray
        Dense matrices, shape ``(n, n)``; ``b=None`` for the identity.
    mu : ndarray
        Eigenvalues, shape ``(m,)``.
    vl, vr : ndarray
        Left and right eigenvectors, ``vl[:, i]`` and ``vr[:, i]`` those of
        ``mu[i]``, shape ``(n, m)``. For symmetric matrices ``vl = vr``.

    Returns
    -------
    null : ndarray
        Boolean array, shape ``(m,)``.

    Notes
    -----
    A backward stable solver, e.g. the QZ algorithm of LAPACK ``zggev``,
    returns the exact eigenvalues of `[a] + [\delta a]`, `[b] + [\delta b]`
    with `\|[\delta a]\| \approx \gamma \epsilon \|[a]\|`, `\gamma` of order
    one. Multiplying `([a] + [\delta a] - (\mu + \delta\mu)([b] + [\delta
    b]))(\boldsymbol{x} + \delta \boldsymbol{x}) = 0` by `\boldsymbol{y}^H`, with `\boldsymbol{y}^H ([a] - \mu
    [b]) = 0`, gives to first order `\delta\mu = \boldsymbol{y}^H ([\delta a] - \mu
    [\delta b]) \boldsymbol{x} / (\boldsymbol{y}^H [b] \boldsymbol{x})`: a null eigenvalue moves by up
    to `\gamma \epsilon \|[a]\| \kappa`.

    A null eigenvalue of a Jordan block of size `m`, e.g. `m = 2` for the
    nilpotent `\boldsymbol{K}^{-1} \boldsymbol{K}_G` of Ziegler's double pendulum, is split by a
    perturbation of size `\eta` into `m` eigenvalues with `\mu^m \propto
    \eta`, i.e. `|\mu| \approx (\gamma \epsilon)^{1/m} \|[a]\|`, and
    `d\mu/d\eta = \mu/(m \eta)`: their condition number gives `\epsilon
    \|[a]\| \kappa \approx |\mu|/(m \gamma)`, i.e. `|\mu|` is about `m
    \gamma` times the bound whatever `m`. Measured with ``zggev``: 0.14 for
    Ziegler's double pendulum, `|\mu| = 9.4 \cdot 10^{-10}`, and 0.005 to
    2.5 for Jordan blocks of size 2 to 4 in pencils of size 2 to 600,
    where the genuine eigenvalues down to `10^{-8} \max |\mu|` were at
    least 73 times their bound, except those within the modulus of the
    split null eigenvalues, which the round-off does not determine and
    which are null as well. For the cantilevers under a follower pressure
    of the tests, the eigenvalues null by this test only are below `2
    \cdot 10^{-9} \max |\mu|`. A well-conditioned
    genuine eigenvalue is null only when `|\mu| \le 2.2 \cdot 10^{-15}
    \|[a]\|`, below the :func:`_null_mu` threshold when `\|[a]\|` is of
    the order of `\max |\mu|`. For symmetric matrices no Jordan block
    exists, and `\boldsymbol{x}^\top [b] \boldsymbol{x} = 1` of :func:`scipy.linalg.eigh` gives
    `\kappa = \|\boldsymbol{x}\|^2`.

    """
    mu = np.asarray(mu)
    null = _null_mu(mu)
    a = np.asarray(a)
    vl = np.asarray(vl)
    vr = np.asarray(vr)
    bvr = vr if b is None else np.asarray(b) @ vr
    den = np.abs(np.einsum('ij,ij->j', vl.conj(), bvr))
    num = np.linalg.norm(vl, axis=0)*np.linalg.norm(vr, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(den > 0, num/den, np.inf)
    bound = (NULL_MU_PERTURBATION_FACTOR*np.finfo(float).eps
             * np.linalg.norm(a)*kappa)
    return null | (np.isfinite(mu) & (np.abs(mu) <= bound))


def _eig_complex(a, b=None, left=False):
    r"""All eigenpairs of `[a]\boldsymbol{v} = w [b]\boldsymbol{v}` with :func:`scipy.linalg.eig`

    The matrices are converted to complex, such that LAPACK ``zgeev`` or
    ``zggev`` is used instead of ``dgeev`` or ``dggev``. With Intel MKL
    2025.0.0 the real drivers crash the Python process when computing the
    eigenvectors of some matrices, with an access violation or a heap
    corruption that cannot be caught, deterministically for a given matrix,
    already for 48 x 48 matrices and with any number of threads. The complex
    drivers are not affected, at about twice the cost of the real ones. The
    complex eigenvectors of real eigenvalues are made real by
    :func:`_general_eigenpairs`. In the browser, ``sys.platform ==
    'emscripten'`` (Pyodide), where SciPy is not linked to MKL, the real
    drivers are used. Returns ``w, vr``, or ``w, vl, vr`` with the left
    eigenvectors ``vl`` when ``left=True``.

    """
    if sys.platform != 'emscripten':
        a = np.asarray(a, dtype=complex)
        if b is not None:
            b = np.asarray(b, dtype=complex)
    return eig(a=a, b=b, left=left, right=True)


def _eig_condensed(K, KG, k):
    r"""As :func:`_eigh_condensed`, for unsymmetric matrices

    The dofs where both the row and the column of ``KG`` are null are
    condensed out with an LU factorization of ``K``, and the condensed
    problem is solved with :func:`scipy.linalg.eig`, through the complex
    LAPACK driver ``zggev``, see :func:`_eig_complex`. Neither converting to
    the standard problem `(\boldsymbol{K}_{bb} - \boldsymbol{K}_{ba} \boldsymbol{K}_{aa}^{-1} \boldsymbol{K}_{ab})^{-1} \boldsymbol{K}_{G_{bb}}` (``dgeev``) nor limiting the
    threads of BLAS/LAPACK avoids the crash of ``dggev`` with Intel MKL
    2025.0.0. The ``k`` returned eigenpairs, see :func:`_general_eigenpairs`,
    are real when they are all real, and complex eigenvectors are
    back-substituted with the real factorization of `\boldsymbol{K}_{aa}`. The null
    `\mu` of :func:`_null_mu_pencil`, e.g. the round-off splitting of a
    defective null eigenvalue, are set to 0.

    """
    KG = csr_matrix(KG)
    active = ((np.asarray(abs(KG).sum(axis=1)).ravel() > 0)
              | (np.asarray(abs(KG).sum(axis=0)).ravel() > 0))
    b = np.flatnonzero(active)
    a = np.flatnonzero(~active)
    if b.size == 0:
        raise ValueError('KG is a null matrix')
    K = csr_matrix(K)
    KGbb = KG[b, :][:, b].toarray()
    S = K[b, :][:, b].toarray()
    if a.size:
        Kab = csc_matrix(K[a, :][:, b])
        Kba = K[b, :][:, a]
        lu = splu(csc_matrix(K[a, :][:, a]))
        chunk = max(1, int(1e7) // a.size)
        for j0 in range(0, b.size, chunk):
            j1 = min(j0 + chunk, b.size)
            X = lu.solve(Kab[:, j0:j1].toarray())
            S[:, j0:j1] -= Kba @ X
    mu, vl, ub = _eig_complex(a=KGbb, b=S, left=True)
    mu[_null_mu_pencil(KGbb, S, mu, vl, ub)] = 0.
    mu, ub = _general_eigenpairs(mu, ub, k)
    eigvecs = np.zeros((K.shape[0], mu.shape[0]), dtype=ub.dtype)
    eigvecs[b, :] = ub
    if a.size:
        eigvecs[a, :] = -_lu_solve(lu, np.asarray(Kab @ ub))
    return mu, eigvecs


def _splu(A):
    """SuperLU factorization of ``A`` with partial pivoting, raising a
    ``RuntimeError`` when ``A`` is exactly singular

    :func:`scipy.sparse.linalg.splu` warns only when it converts its input
    to the CSC format, :class:`scipy.sparse.SparseEfficiencyWarning`, which
    the conversion here avoids, as in the other calls of this package. No
    :func:`warnings.catch_warnings` block is used: its exit resets the
    registry of the warnings already shown, of every module, and the
    warnings of the caller shown once per location would be shown again
    after each call.

    """
    return splu(csc_matrix(A))


def _permutation_parity(p):
    """``1`` for an even permutation ``p``, ``-1`` for an odd one"""
    p = np.asarray(p)
    seen = np.zeros(p.size, dtype=bool)
    odd = False
    for i in range(p.size):
        if seen[i]:
            continue
        j = i
        length = 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        odd ^= bool((length - 1) % 2)
    return -1 if odd else 1


def _det_sign(lu):
    r"""Sign of the determinant of a real matrix from its SuperLU
    factorization `\boldsymbol{\Pi}_r \boldsymbol{A} \boldsymbol{\Pi}_c = \boldsymbol{L} \boldsymbol{U}`, with a unit `\boldsymbol{L}`"""
    d = lu.U.diagonal()
    return (int(np.prod(np.sign(d))) * _permutation_parity(lu.perm_r)
            * _permutation_parity(lu.perm_c))


def _norm_estimate(lu, KG, v):
    r"""Estimate from below of `\|\boldsymbol{A}^{-1} \boldsymbol{K}_G\|_2`, with `\boldsymbol{A}` the matrix
    factorized in ``lu``, from the vector ``v``

    Returns
    -------
    norm : float
        `\|\boldsymbol{O}^H \boldsymbol{O} \boldsymbol{v}\|/\|\boldsymbol{O} \boldsymbol{v}\|`, with `\boldsymbol{O} = \boldsymbol{A}^{-1} \boldsymbol{K}_G`, or
        ``0.`` when `\boldsymbol{O} \boldsymbol{v} = 0`.

    Notes
    -----
    One step of the power iteration of `\boldsymbol{O}^H \boldsymbol{O}`, whose largest
    eigenvalue is `\|\boldsymbol{O}\|_2^2`, with two solutions with ``lu``. The
    estimate is not larger than `\|\boldsymbol{O}\|_2` and, by the Cauchy-Schwarz
    inequality `\|\boldsymbol{O} \boldsymbol{v}\|^2 = \boldsymbol{v}^H \boldsymbol{O}^H \boldsymbol{O} \boldsymbol{v} \le \|\boldsymbol{v}\|
    \|\boldsymbol{O}^H \boldsymbol{O} \boldsymbol{v}\|`, not smaller than `\|\boldsymbol{O} \boldsymbol{v}\|/\|\boldsymbol{v}\|`. The
    vector `\boldsymbol{O} \boldsymbol{v}` is in the range of `\boldsymbol{O}`, i.e. of `\boldsymbol{A}^{-1} \boldsymbol{K}_G`,
    such that the estimate does not decrease with the number of dofs where
    `\boldsymbol{K}_G` is null, as `\|\boldsymbol{O} \boldsymbol{v}\|/\|\boldsymbol{v}\|` does.

    """
    y = _lu_solve(lu, KG @ v)
    norm_y = np.linalg.norm(y)
    if norm_y == 0:
        return 0.
    z = KG.conj().T @ lu.solve(np.asarray(y), trans='H')
    return float(np.linalg.norm(z)/norm_y)


def _eigs_shifted(K, KG, s, k, tol, lu):
    r"""The ``k`` load multipliers nearest the shift ``s``, for real
    unsymmetric matrices

    With `\lambda = s + \delta`, `(\boldsymbol{K} + s \boldsymbol{K}_G + \delta \boldsymbol{K}_G) \boldsymbol{u} = 0`
    gives the standard problem
    `(\boldsymbol{K} + s \boldsymbol{K}_G)^{-1} \boldsymbol{K}_G \boldsymbol{u} = \nu \boldsymbol{u}`, `\nu = -1/(\lambda - s)`,
    whose ``k`` eigenvalues of largest `|\nu|` are found with
    :func:`scipy.sparse.linalg.eigs`, using the SuperLU factorization ``lu``
    of `\boldsymbol{K} + s \boldsymbol{K}_G`. The eigenvalues `|\nu| \le \epsilon_0 \max(\max
    |\nu|, \|\boldsymbol{O}\|)`, with `\epsilon_0` = :data:`NULL_MU_RTOL` and
    `\|\boldsymbol{O}\|` the estimate of the norm of the operator `\boldsymbol{O} = (\boldsymbol{K} + s
    \boldsymbol{K}_G)^{-1} \boldsymbol{K}_G` of :func:`_norm_estimate` from the starting vector of
    ARPACK, are taken as null, i.e. infinite load multipliers, e.g. of the
    null rows of `\boldsymbol{K}_G`, and are not returned.

    Returns
    -------
    mu, eigvecs : ndarray
        The eigenpairs, with `\mu = -1/\lambda = \nu/(1 - s \nu)`, sorted by
        :func:`_general_eigenpairs`, empty when all the `\nu` are null.
    radius : float
        `\max |\lambda - s|` of the returned load multipliers. All the load
        multipliers in the disc `|\lambda - s| < radius` are returned.
        ``inf`` when a null `\nu` was found, i.e. all the finite load
        multipliers were found, none when all the `\nu` are null.

    Notes
    -----
    The null `\nu` are found as round-off: each product with `\boldsymbol{O}` is
    computed with an error of the order of the machine precision times
    `\|\boldsymbol{O}\|`, and a null `\nu` comes out with a modulus up to this order,
    or larger for a defective null eigenvalue. When `\boldsymbol{O}` has non-null
    eigenvalues, `\max |\nu|` is a scale of the round-off; when it has
    none, e.g. when `\boldsymbol{K}^{-1} \boldsymbol{K}_G` is nilpotent as with the follower load
    of Ziegler's double pendulum, all the `\nu` are round-off, e.g. about
    `10^{-33}` or `10^{-17}` instead of 0 depending on the restarts of
    ARPACK, and `\max |\nu|` is not a scale: the largest `\nu` would not be
    null and their spurious load multipliers, about `10^{33}` or
    `10^{17}`, would be returned. The norm `\|\boldsymbol{O}\|` is a scale that does
    not depend on the eigenvalues. The estimate is not larger than `\max
    |\nu|` for a normal `\boldsymbol{O}`, whose largest `|\nu|` is its norm, and both
    scales then give the same null `\nu`.

    """
    n = K.shape[0]
    OP = LinearOperator((n, n), matvec=lambda x: _lu_solve(lu, KG @ x),
                        dtype=np.result_type(K.dtype, KG.dtype))
    kwargs = start_kwargs(n)
    nu, eigvecs = eigs(OP, k=k, which='LM', tol=tol, **kwargs)
    release_memory()
    scale = max(float(np.abs(nu).max()), _norm_estimate(lu, KG, kwargs['v0']))
    null = np.abs(nu) <= NULL_MU_RTOL*scale
    if np.any(null):
        #NOTE the infinite load multipliers are not returned, as by
        #     _eig_condensed()
        radius = np.inf
        nu = nu[~null]
        eigvecs = eigvecs[:, ~null]
    else:
        radius = float(np.max(1./np.abs(nu)))
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = nu/(1 - s*nu)
    mu, eigvecs = _general_eigenpairs(mu, eigvecs)
    return mu, eigvecs, radius


def _covered(discs):
    r"""Largest `c` such that `(0, c)` is inside the union of the intervals
    `(s - r, s + r)` of the discs ``(s, r)`` on the real axis"""
    c = 0.
    for left, right in sorted((s - r, s + r) for s, r in discs):
        if left > c:
            break
        c = max(c, right)
    return c


def _eigs_shift_search(K, KG, k, tol, check_rtol, silent):
    r"""Lowest positive real load multiplier of unsymmetric matrices, with a
    search of shifts along the positive real axis

    At each shift, :func:`_eigs_shifted` finds the ``k`` load multipliers
    nearest the shift, ``k`` being at least
    :data:`SHIFT_SEARCH_MIN_EIGVALUES` and at most the size minus 2, and
    their residuals are verified with :func:`_check_eigenpairs` and
    ``check_inertia=False``. The shift moves along the positive real axis
    until no positive real load multiplier lower than the lowest one found
    can have been missed, at most :data:`SHIFT_SEARCH_MAX_STEPS` times, and
    a parity check of the signs of determinants confirms the result.

    Returns
    -------
    mu, eigvecs : ndarray
        The eigenpairs of the step that found `\lambda_{cr}`, i.e. the ``k``
        multipliers nearest its shift, sorted by :func:`_general_eigenpairs`
        with `\lambda_{cr}` first.

    Raises
    ------
    RuntimeError
        When ``K`` is singular, a verification failed, no positive real load
        multiplier was found, or the parity check failed.

    Notes
    -----
    The eigenvalue solver :func:`scipy.sparse.linalg.eigs` finds the load
    multipliers of smallest modulus, see :func:`_eigs_shifted` with
    `s = 0`, but with follower loads many negative or complex multipliers
    may have a smaller modulus than the critical one, e.g. under an internal
    pressure. The search therefore moves the shift `s` along the positive
    real axis:

    1. `s_0 = 0`: the ``k`` multipliers nearest `s_0`, within the radius
       `r_0`, are found with :func:`_eigs_shifted`, ``k`` being at least
       :data:`SHIFT_SEARCH_MIN_EIGVALUES`. The residual of every step is
       verified with :func:`_check_eigenpairs` and ``check_inertia=False``.
    2. Each step `j` covers the interval `(s_j - r_j, s_j + r_j)` of the real
       axis, inside which all the multipliers were found. With `c` the end of
       the union of these intervals starting at 0, i.e. `(0, c)` is covered,
       the next shift is `s_{j+1} = c + r_j`. When the new interval does not
       reach `c`, the next shifts move back towards `c` and fill the gap. The
       shift is perturbed when `\boldsymbol{K} + s \boldsymbol{K}_G` is exactly singular.
    3. The search stops when the lowest positive real multiplier found,
       `\lambda_{cr}`, is not larger than `c`: no multiplier of `(0,
       \lambda_{cr})` was missed. It fails after
       :data:`SHIFT_SEARCH_MAX_STEPS` steps, or when the shift exceeds
       :data:`SHIFT_SEARCH_MAX_GROWTH` times `r_0`.
    4. Parity check: `\det(\boldsymbol{K} + 0.9999 \lambda_{cr} \boldsymbol{K}_G)` must have the
       sign of `\det \boldsymbol{K}`, i.e. an even number, normally zero, of real
       multipliers in `(0, 0.9999 \lambda_{cr})`. The signs come from the
       SuperLU factorizations, see :func:`_det_sign`. This check cannot
       detect two missing real multipliers, nor a missing complex pair.

    The parity check follows from `\det(\boldsymbol{K} + s \boldsymbol{K}_G) = \det \boldsymbol{K}
    \prod_i (1 + s \mu_i)`, with `\mu_i = -1/\lambda_i` the eigenvalues of
    `\boldsymbol{K}^{-1} \boldsymbol{K}_G`: a complex conjugate pair gives the positive factor
    `|1 + s \mu_i|^2`, a null `\mu_i` the factor 1, and a real `\mu_i` a
    negative factor exactly when `0 < \lambda_i < s`. The sign of the
    determinant therefore changes with the parity of the number of real
    load multipliers in `(0, s)`.

    """
    K = csc_matrix(K)
    KG = csr_matrix(KG)
    n = K.shape[0]
    k = min(max(k, SHIFT_SEARCH_MIN_EIGVALUES), n - 2)
    if k < 1:
        raise ValueError('eigs() requires at least 3 dofs')
    try:
        lu0 = _splu(K)
    except RuntimeError:
        raise RuntimeError('K is singular')
    discs = []
    best = None
    s = 0.
    lu = lu0
    for step in range(SHIFT_SEARCH_MAX_STEPS):
        mu, eigvecs, radius = _eigs_shifted(K, KG, s, k, tol, lu)
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = -1./mu
        error = _check_eigenpairs(K, KG, lam, eigvecs, check_rtol,
                                  check_inertia=False)
        if error is not None:
            raise RuntimeError('shift {0:g}: {1}'.format(s, error))
        discs.append((s, radius))
        real = np.isfinite(lam) & (np.imag(lam) == 0)
        positive = np.real(lam[real])
        positive = positive[positive > 0]
        if positive.size and (best is None or positive.min() < best[0]):
            best = (positive.min(), mu, eigvecs)
        covered = _covered(discs)
        msg('shift {0:.6g}: radius {1:.6g}, (0, {2:.6g}) covered{3}'.format(
            s, radius, covered, '' if best is None else
            ', lowest positive real load multiplier {0:.6g}'.format(best[0])),
            level=4, silent=silent)
        if best is not None and best[0] <= covered:
            break
        if not np.isfinite(covered):
            raise RuntimeError('no positive real load multiplier exists, '
                    'all the finite ones were found')
        if step == 0:
            r0 = radius
        s_next = covered + radius
        if s_next > SHIFT_SEARCH_MAX_GROWTH*r0:
            raise RuntimeError('no positive real load multiplier was found '
                    'below {0:g}'.format(covered))
        for perturbation in (0., 1e-6, 1e-3):
            s = s_next + perturbation*radius
            try:
                lu = _splu(K + s*KG)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError('K + s KG is singular for s={0:g}'.format(s))
    else:
        raise RuntimeError('no positive real load multiplier was found below '
                '{0:g} after {1} shifts'.format(covered,
                                                SHIFT_SEARCH_MAX_STEPS))
    lam_cr, mu, eigvecs = best
    s = (1 - PARITY_REL_GAP)*lam_cr
    if _det_sign(_splu(K + s*KG)) != _det_sign(lu0):
        raise RuntimeError('parity check failed: an odd number of real load '
                'multipliers lower than {0:g} was missed'.format(lam_cr))
    msg('{0} shifts, parity check passed'.format(len(discs)), level=4,
        silent=silent)
    return mu, eigvecs


def _eigsh_cayley(K, KG, k, tol, silent, safety=10., mu_max=False):
    r"""Solution with :func:`scipy.sparse.linalg.eigsh` in Cayley mode

    The shift is estimated with :func:`_estimate_sigma` and the given
    ``safety`` factor, from the estimate ``mu_max`` of
    :func:`_estimate_mu_max` when given. ARPACK's ``ncv`` is chosen by
    :func:`.arpackutils.capped_eigsh`, which avoids the Intel MKL bug.
    Returns the ``min(k, n - 2)`` eigenvalues ``mu`` of ``KG u = mu K u``
    selected by the Cayley mode and their eigenvectors, sorted by
    :func:`_sort_eigenpairs`.

    Notes
    -----
    :func:`scipy.sparse.linalg.eigsh` solves `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}`,
    with `\mu = -1/\lambda`, which requires symmetric matrices and a
    positive definite `\boldsymbol{K}`. In the Cayley mode, ARPACK computes the
    eigenvalues `(\mu + \sigma)/(\mu - \sigma)` of the operator
    `(\boldsymbol{K}_G - \sigma \boldsymbol{K})^{-1}(\boldsymbol{K}_G + \sigma \boldsymbol{K})` ([1]_, Appendix A,
    Section A.1, "Cayley Transformation Mode", with `\boldsymbol{A} = \boldsymbol{K}_G` and
    `\boldsymbol{M} = \boldsymbol{K}`), and ``which='SM'`` selects those of smallest modulus. When `\sigma` is larger than `|\mu|`
    of all the eigenvalues, `(\mu + \sigma)/(\mu - \sigma)` increases in
    modulus with `\mu` in `(-\sigma, \sigma)`, such that the most negative
    `\mu`, i.e. the lowest positive load multipliers, are selected, see
    :func:`_estimate_sigma`.

    References
    ----------
    .. [1] Lehoucq, R.B., Sorensen, D.C., Yang, C., ARPACK Users' Guide:
       Solution of Large-Scale Eigenvalue Problems with Implicitly
       Restarted Arnoldi Methods, SIAM, Philadelphia, 1998.
       https://doi.org/10.1137/1.9780898719628

    """

    sigma = _estimate_sigma(K, KG, safety=safety, mu_max=mu_max)
    msg('sigma={0}'.format(sigma), level=4, silent=silent)
    k = min(k, K.shape[0] - 2)
    mu, eigvecs = capped_eigsh(A=KG, k=k, M=K, sigma=sigma, which='SM',
                               mode='cayley', tol=tol)
    return _sort_eigenpairs(mu, eigvecs)


def _count_multipliers(K, KG, mu_max, sign, K_positive_definite):
    r"""Number of load multipliers of the given ``sign`` whose modulus is
    lower than ``INERTIA_COUNT_FACTOR/mu_max``, or ``None`` when unknown

    By Sylvester's law of inertia, for a positive definite ``K``, the number
    of negative pivots of `\boldsymbol{K} + s \boldsymbol{K}_G` is the number of load multipliers
    in `(0, s)` for `s > 0`, and in `(s, 0)` for `s < 0`, see
    :func:`_check_eigenpairs` and :func:`_negative_pivots`. ``None`` is
    returned when ``mu_max`` is unknown, ``K`` is not positive definite, or
    the count is unknown.

    """
    if mu_max is None or not K_positive_definite(K):
        return None
    s = sign*INERTIA_COUNT_FACTOR/mu_max
    return _negative_pivots(K + s*KG)


def _eigsh_both_signs(K, KG, k, tol, silent, safety=10., mu_max=False,
                      K_positive_definite=None):
    r"""The lowest positive load multipliers, completed by the negative ones
    closest to zero, with :func:`_eigsh_cayley`

    Returns the eigenpairs of at most ``min(k, n - 2)`` finite load
    multipliers, in the order of :func:`_sort_eigenpairs`.

    Notes
    -----
    The Cayley mode with a shift `\sigma` larger than all the `|\mu|`
    selects the eigenvalues `\mu` of `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}` by
    increasing `(\mu + \sigma)/(\mu - \sigma)`, i.e. all the negative
    `\mu` before the null and the positive ones, see :func:`_eigsh_cayley`.
    ARPACK may not converge when fewer negative `\mu` exist than the number
    of requested eigenvalues, the remaining ones being then in the cluster
    of the null `\mu`, e.g. for a tensile reference load. The number of
    requested eigenvalues of each sign is therefore limited by the number
    of load multipliers of that sign, counted with the inertia of `\boldsymbol{K} + s
    \boldsymbol{K}_G` below the modulus ``INERTIA_COUNT_FACTOR`` divided by the largest
    `|\mu|`, see :func:`_count_multipliers`:

    1. ``KG`` gives the positive load multipliers `\lambda = -1/\mu`,
       `\mu < 0`, at most as many as counted. The null `\mu`, see
       :func:`_null_mu`, and the positive `\mu` are discarded.
    2. When fewer than ``k`` positive load multipliers were found, ``-KG``,
       i.e. the reversed load, gives the remaining ones, the negative load
       multipliers closest to zero, at most as many as counted.

    When the counts are unknown, e.g. ``K`` is not positive definite, ``k``
    eigenvalues are requested. A tensile reference load, without any
    positive load multiplier, thus gives the negative load multipliers
    closest to zero without a dense solution. The load multipliers whose
    modulus is larger than the limit of the counts, i.e. `|\mu|` below
    ``1/INERTIA_COUNT_FACTOR`` of the largest one, are only returned when
    the counts are unknown.

    """
    if mu_max is False:
        mu_max = _estimate_mu_max(K, KG)
    if K_positive_definite is None:
        K_positive_definite = _is_positive_definite
    kmax = min(k, K.shape[0] - 2)
    num_pos = _count_multipliers(K, KG, mu_max, 1, K_positive_definite)
    msg('{0} positive load multipliers counted'.format(num_pos), level=4,
        silent=silent)
    mu = np.zeros(0)
    eigvecs = np.zeros((K.shape[0], 0))
    if num_pos != 0:
        k1 = kmax if num_pos is None else min(kmax, num_pos)
        mu, eigvecs = _eigsh_cayley(K, KG, k1, tol, silent, safety=safety,
                                    mu_max=mu_max)
        keep = (mu < 0) & ~_null_mu(mu, mu_max)
        mu, eigvecs = mu[keep], eigvecs[:, keep]
        if mu.size >= kmax:
            return mu[:kmax], eigvecs[:, :kmax]
    # the reversed load
    num_neg = _count_multipliers(K, -KG, mu_max, 1, K_positive_definite)
    msg('{0} negative load multipliers counted'.format(num_neg), level=4,
        silent=silent)
    if num_neg != 0:
        k2 = kmax - mu.size
        if num_neg is not None:
            k2 = min(k2, num_neg)
        mu2, eigvecs2 = _eigsh_cayley(K, -KG, k2, tol, silent, safety=safety,
                                      mu_max=mu_max)
        keep2 = (mu2 < 0) & ~_null_mu(mu2, mu_max)
        mu = np.concatenate((mu, -mu2[keep2]))
        eigvecs = np.concatenate((eigvecs, eigvecs2[:, keep2]), axis=1)
    if mu.size == 0:
        raise RuntimeError('no finite load multiplier was found')
    return mu, eigvecs


def _check_both_signs(K, KG, mu, eigvecs, rtol, K_positive_definite=None):
    r"""Verify the eigenpairs of :func:`lb` with :func:`_check_eigenpairs`,
    the positive load multipliers with ``KG``, the negative ones with
    ``-KG``, i.e. the reversed load, such that the inertia check also tells
    that no negative load multiplier closer to zero than the returned ones
    is missing. The null `\mu`, see :func:`_null_mu`, are not verified."""
    mu = np.asarray(mu)
    kept = np.isfinite(mu) & ~_null_mu(mu)
    if not np.any(kept):
        return 'no finite load multiplier was found'
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = -1./mu
    positive = kept & (lam > 0)
    negative = kept & (lam < 0)
    if np.any(positive):
        error = _check_eigenpairs(K, KG, lam[positive], eigvecs[:, positive],
                                  rtol, K_positive_definite=K_positive_definite)
        if error is not None:
            return error
    if np.any(negative):
        error = _check_eigenpairs(K, -KG, -lam[negative],
                                  eigvecs[:, negative], rtol,
                                  K_positive_definite=K_positive_definite)
        if error is not None:
            return 'reversed load: ' + error
    return None


def _sparse_first(analysis, strategies, verify, dense, dense_size,
                  max_dense_size, silent, hint=''):
    r"""Solver policy of :func:`lb` and :func:`.freq`: the sparse solvers
    first, the dense solver as a last resort

    Parameters
    ----------
    analysis : str
        Name of the analysis, used in the messages.
    strategies : list
        The sparse strategies as ``(name, solver)`` pairs, tried in this
        order. ``solver()`` returns the eigenpairs as a tuple.
    verify : callable
        ``verify(*eigenpairs)`` returns ``None`` when the eigenpairs pass
        the verification, otherwise the description of the failure.
    dense : tuple
        The dense solver, a ``(name, solver)`` pair, used when all
        strategies failed and ``0 < dense_size <= max_dense_size``. A
        :class:`DenseFallbackWarning` is then issued with the reasons of the
        failures, and the dense eigenpairs are verified as well.
    dense_size, max_dense_size : int
        Size of the dense problem and its maximum.
    silent : bool
        Whether the log messages are printed.
    hint : str, optional
        Appended to the message of the ``RuntimeError``.

    Returns
    -------
    eigenpairs : tuple
        The returned value of the solver that passed the verification.
    name : str
        The name of this solver.

    Raises
    ------
    RuntimeError
        When no solver passed the verification, listing every attempted
        solver and the reason of its failure.

    """
    errors = []

    def attempt(name, solver):
        msg('{0}...'.format(name), level=3, silent=silent)
        try:
            result = solver()
            error = verify(*result)
        except (LinAlgError, RuntimeError, ValueError, TypeError,
                ArithmeticError) as e:
            result = None
            error = '{0}: {1}'.format(type(e).__name__, e)
        if error is None:
            msg('finished!', level=3, silent=silent)
        else:
            warn('{0} failed: {1}'.format(name, error), level=3,
                 silent=silent)
            errors.append('{0}: {1}'.format(name, error))
        return result, error

    for name, solver in strategies:
        result, error = attempt(name, solver)
        if error is None:
            return result, name
    name, solver = dense
    if max_dense_size <= 0:
        errors.append('{0}: dense fallback disabled with max_dense_size={1}'
                      .format(name, max_dense_size))
    elif dense_size > max_dense_size:
        errors.append('{0}: dense fallback skipped, size {1} > '
                      'max_dense_size={2}'.format(name, dense_size,
                                                  max_dense_size))
    else:
        text = ('{0}: the sparse solvers failed ({1}), using the dense solver '
                '{2} of size {3} as a last resort'.format(analysis,
                    '; '.join(errors), name, dense_size))
        warn(text, level=3, silent=silent)
        warnings.warn(text, DenseFallbackWarning, stacklevel=4)
        result, error = attempt(name, solver)
        if error is None:
            return result, name
    raise RuntimeError('{0} failed, no eigenvalue solver passed the '
                       'verification. Attempted: {1}.{2}'.format(analysis,
                           '; '.join(errors), hint))


def lb(K, KG, tol=0, sparse_solver=True, silent=False,
       num_eigvalues=25, num_eigvalues_print=5,
       skip_null_cols=False, max_dense_size=2000, check_rtol=1e-3,
       symmetric=None):
    r"""Linear buckling analysis

    Calculates the load multipliers ``lambda`` and the buckling modes ``u``
    of the linear buckling problem of ``K`` and ``KG``, i.e. the eigenpairs
    with ``(K + lambda*KG) @ u = 0``, where ``lambda`` is the load
    multiplier of the load state that generated ``KG``. It can also be used
    for more general eigenvalue analyses, e.g. if ``K`` is the tangent
    stiffness matrix of a given load state. The null rows and columns of
    ``K`` are removed from both matrices before solving the eigenvalue
    problem, unless ``skip_null_cols=True``.

    **Solver policy.** With ``sparse_solver=True`` the sparse solvers run
    first for every problem size, in the order given below, and each result
    must pass the verification: the relative residual of each eigenpair
    must not be larger than ``check_rtol`` and, for symmetric matrices with
    a positive definite ``K``, no positive load multiplier below the highest
    one returned, and no negative one closer to zero than the returned
    ones, may be missing, see :func:`_check_both_signs`. Only when
    all of them failed (verification, ARPACK error or no convergence,
    singular factorization, no positive real load multiplier...) a dense
    solver is used, if the dense problem has at most ``max_dense_size``
    dofs, with a :class:`DenseFallbackWarning` that gives the reasons of the
    failures. Its result is verified as well. A ``RuntimeError`` listing
    every attempted solver is raised when the problem is too large for the
    dense fallback, or when the dense solution also failed.

    **Symmetric matrices.** The sparse solvers are:

    1. :func:`scipy.sparse.linalg.eigsh` in Cayley mode, with the shift
       estimated from the matrices by :func:`_estimate_sigma`;
    2. the same with a 10 times larger shift.

    Each one computes the lowest positive load multipliers and, when fewer
    than ``num_eigvalues`` exist, the negative ones closest to zero, i.e.
    the critical load multipliers of the reversed load, with ``-KG``, see
    :func:`_eigsh_both_signs`. A reference load without any positive load
    multiplier, e.g. a tensile load, is solved by the sparse solvers too.
    The dense fallback condenses out the dofs where ``KG`` has null rows
    and solves the condensed problem with :func:`scipy.linalg.eigh`, see
    :func:`_eigh_condensed`.

    **Unsymmetric matrices**, e.g. ``KG`` with the load stiffness of a
    follower pressure, may have complex load multipliers. The sparse solver
    is a search of shifts along the positive real axis with
    :func:`scipy.sparse.linalg.eigs`, see :func:`_eigs_shift_search`, which
    finds the lowest positive real load multiplier. The returned eigenpairs
    are the ``num_eigvalues`` load multipliers nearest the last shift of the
    search, which is zero unless the multipliers of smallest modulus did not
    include a positive real one, not necessarily the load multipliers of
    smallest modulus. The dense fallback, :func:`_eig_condensed`, keeps the
    dofs where either the row or the column of ``KG`` is not null, and
    returns the ``num_eigvalues`` first load multipliers in the order
    described in Returns; when none of them is positive and real, they are
    returned with a warning. Eigenvalues whose imaginary part is below
    ``1e-8`` of their modulus are returned as real, with real eigenvectors.
    The arrays are real when the returned eigenpairs are real, even if other
    eigenvalues of the solver are complex.

    **Intel MKL.** When SciPy is linked against Intel MKL 2024.2.0 to
    2025.0.0, ``ncv`` of :func:`scipy.sparse.linalg.eigsh` is capped at 32
    and more than 15 eigenpairs are computed in several passes, see
    :mod:`structsolve.arpackutils`, where the cap can be overridden, e.g.
    with the environment variable ``STRUCTSOLVE_ARPACK_MAX_NCV=0`` to lift
    it with MKL 2025.0.1 or newer. The dense solvers of unsymmetric
    matrices call :func:`scipy.linalg.eig` with complex matrices.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix. Should include all constant terms of the initial
        stress stiffness matrix, aerodynamic matrix and so forth when
        applicable.
    KG : sparse_matrix
        Initial stress stiffness matrix that multiplies the load multiplier
        `\lambda` of the eigenvalue problem.
    tol : float, optional
        A float tolerance passed to :func:`scipy.sparse.linalg.eigsh` and
        :func:`scipy.sparse.linalg.eigs`.
    sparse_solver : bool, optional
        With ``True``, the default, the sparse solvers always run first,
        whatever the size of the problem, and a dense solver is only used as
        a last resort, see above. ``num_eigvalues`` load multipliers are
        returned. With ``False``, the full problem is explicitly solved with
        a dense solver, without trying the sparse solvers:
        :func:`scipy.linalg.eigh`, which requires a positive definite ``K``,
        or :func:`scipy.linalg.eig` for unsymmetric matrices, returning all
        the load multipliers. It is a reference for validations and for the
        callers that need all of them.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    num_eigvalues : int, optional
        Number of load multipliers calculated with the sparse solver, and
        number of returned eigenvectors. Fewer are returned by the sparse
        solver of symmetric matrices when fewer finite load multipliers
        exist, the infinite ones, e.g. of the null rows of ``KG``, being
        discarded.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.
    skip_null_cols : bool, optional
        If True, skip the removal of null columns from the matrices.
        Use only when K is known to be non-singular.
    max_dense_size : int, optional
        Maximum size of the dense problem solved as a last resort, when all
        the sparse solvers failed, i.e. the number of non-null rows of
        ``KG``, or of non-null rows or columns for unsymmetric matrices, see
        above. Larger problems raise a ``RuntimeError`` instead, and
        ``0`` disables the dense fallback. Until version 0.5.3, it was the
        size below which the dense solver was tried first.
    check_rtol : float, optional
        Maximum relative residual
        ``||K u + lambda KG u|| / (||K u|| + |lambda| ||KG u||)`` of the
        eigenpairs of the sparse solver, see above and
        :func:`_relative_residuals`.
    symmetric : bool or None, optional
        Whether ``K`` and ``KG`` are symmetric. ``None`` checks it with
        :func:`is_symmetric`, ``True`` skips the check and uses the solvers
        of symmetric matrices, ``False`` forces the solvers of unsymmetric
        matrices, required e.g. when ``KG`` includes the load stiffness of
        follower loads, see above.

    Returns
    -------
    eigvals : ndarray
        The load multipliers `\lambda`, calculated as ``-1/eigval`` from the
        eigenvalues ``eigval`` of ``KG u = eigval K u``. The positive load
        multipliers come first in increasing order, followed by the negative
        ones in order of increasing modulus, i.e. the critical load
        multipliers of the reversed load first, for both solvers. For
        unsymmetric matrices the array is complex when some eigenvalues are
        complex, which come after the real ones by increasing modulus. The
        load multipliers of the null ``eigval``, ``|eigval|`` not larger
        than ``NULL_MU_RTOL`` times the largest ``|eigval|``, infinite up to
        the round-off, come last; the dense solvers also take as null the
        ``eigval`` not determined by the round-off, see
        ``linear_buckling._null_mu_pencil``, returned as ``-inf``.
    eigvecs : ndarray
        The `i^{th}` eigenvector is ``eigvecs[:, i]``, with the size of the
        original matrices. Only ``num_eigvalues`` eigenvectors are returned.

    Raises
    ------
    RuntimeError
        When no eigenvalue solver passed the verification, with the list of
        the attempted solvers and their failures.

    Warns
    -----
    DenseFallbackWarning
        When the sparse solvers failed and the dense solver was used.
    RuntimeWarning
        For unsymmetric matrices, when the dense fallback found no positive
        real load multiplier.

    Notes
    -----
    The eigenvalue problem is

    .. math::

        (\boldsymbol{K} + \lambda \boldsymbol{K}_G) \boldsymbol{u} = \boldsymbol{0}

    where `\lambda` is the load multiplier of the load state that generated
    `\boldsymbol{K}_G`. It is solved as `\boldsymbol{K}_G \boldsymbol{u} = \mu \boldsymbol{K} \boldsymbol{u}`, with `\lambda =
    -1/\mu`; the null eigenvalues `\mu`, e.g. of the dofs where `\boldsymbol{K}_G` is
    null, give infinite load multipliers.

    **Solver policy.** The dense fallback keeps structsolve robust where
    ARPACK is unreliable, normal runs never pay for a dense solution.

    **Symmetric matrices.** The shift of the Cayley mode is estimated by
    power iterations on `\boldsymbol{K}^{-1} \boldsymbol{K}_G`, see :func:`_estimate_sigma` and
    :func:`_eigsh_cayley`. The dense fallback condenses out the dofs where
    ``KG`` has null rows, typically the in-plane dofs, with a sparse
    factorization of ``K``, see :func:`_eigh_condensed`. The eigenpairs are
    verified as follows:

    - the relative residual
      `||K u + \lambda K_G u|| / (||K u|| + |\lambda| ||K_G u||)` of each
      eigenpair must not be larger than ``check_rtol``, see
      :func:`_relative_residuals`;
    - when ``K`` is positive definite, no load multiplier below the highest
      positive one returned may be missing: the number of negative pivots of
      `\boldsymbol{K} + s \boldsymbol{K}_G`, for `s` slightly below that multiplier, is the number
      of load multipliers in `(0, s)` (Sylvester's law of inertia). The
      factorization is done with SuperLU, without row interchanges. When a
      row interchange was needed and the count is unknown, `\boldsymbol{K} + s \boldsymbol{K}_G`
      must be positive definite for `s` slightly below the lowest positive
      load multiplier. See :func:`_check_eigenpairs`;
    - the same for the negative load multipliers with `-\boldsymbol{K}_G`, i.e. the
      positive load multipliers of the reversed load, see
      :func:`_check_both_signs`.

    The Cayley mode selects all the negative `\mu`, i.e. the positive load
    multipliers, before the null and the positive `\mu`. The load
    multipliers of each sign are first counted with the inertia of `\boldsymbol{K} + s
    \boldsymbol{K}_G`, for `|s|` equal to ``INERTIA_COUNT_FACTOR`` times the lowest
    modulus that a load multiplier may have, and at most as many are
    requested, since ARPACK may not converge when it must return
    eigenvalues of the cluster of the null `\mu`. When fewer than
    ``num_eigvalues`` positive load multipliers exist, the Cayley mode with
    `-\boldsymbol{K}_G` gives the negative load multipliers closest to zero, see
    :func:`_eigsh_both_signs`.

    **Unsymmetric matrices**, e.g. ``KG`` with the load stiffness of a
    follower pressure that is not conservative, may have complex
    eigenvalues. The search of shifts of :func:`_eigs_shift_search` finds
    the lowest positive real load multiplier even when many negative or
    complex multipliers have a smaller modulus, verifies the residuals, and
    checks the parity of the number of real multipliers below the critical
    one. The dofs kept by the dense fallback, where either the row or the
    column of ``KG`` is not null, are for follower loads almost all of
    them. When no returned load multiplier is positive and real, no
    divergence load exists, e.g. for a system that only loses stability by
    flutter, see :func:`.freq`. The inertia check requires symmetric
    matrices and is skipped.

    **Intel MKL.** ARPACK, used by :func:`scipy.sparse.linalg.eigsh`, returns
    wrong eigenpairs at random, without any error, or raises
    ``ArpackError -8`` when linked against Intel MKL 2024.2.0 to 2025.0.0,
    e.g. in some Anaconda builds of SciPy: the ``dsteqr`` routine of these
    MKL versions returns wrong eigenvectors for matrices larger than
    32 x 32, which ARPACK uses with the size ``ncv``. When such a LAPACK is
    detected, ``ncv`` is capped at 32 and more than 15 eigenpairs are
    computed in several passes, combined with a Rayleigh-Ritz projection,
    see :mod:`structsolve.arpackutils`.
    The non-symmetric drivers of :func:`scipy.sparse.linalg.eigs` are not
    affected.

    With Intel MKL 2025.0.0, the real LAPACK drivers ``dgeev`` and ``dggev``
    of :func:`scipy.linalg.eig` crash the Python process for some matrices
    when computing the eigenvectors, with any number of threads. The dense
    solvers of unsymmetric matrices therefore call :func:`scipy.linalg.eig`
    with complex matrices, i.e. ``zgeev`` and ``zggev``, which are not
    affected.

    """
    msg('Running linear buckling analysis...', silent=silent)

    msg('Eigenvalue solver... ', level=2, silent=silent)

    size = KG.shape[0]
    if skip_null_cols:
        used_cols = None
    else:
        K, KG, used_cols = remove_null_cols(K, KG, silent=silent)
    if symmetric is None:
        symmetric = is_symmetric(K) and is_symmetric(KG)
        if not symmetric:
            msg('Unsymmetric K or KG, using the general eigenvalue solvers',
                level=2, silent=silent)
    if not symmetric:
        eigvals, peigvecs = _lb_unsymmetric(K, KG, tol, sparse_solver,
                silent, num_eigvalues, max_dense_size, check_rtol)
    elif sparse_solver:
        K = csr_matrix(K)
        KG = csr_matrix(KG)
        num_active = int(np.count_nonzero(abs(KG).sum(axis=1)))
        msg('{0} of {1} rows of KG are non-null'.format(num_active,
            K.shape[0]), level=3, silent=silent)

        # the estimate of the shift and the inertia of K are computed once
        # for all the attempts
        once = {}

        def mu_max():
            if 'mu_max' not in once:
                once['mu_max'] = _estimate_mu_max(K, KG)
            return once['mu_max']

        def K_positive_definite(K):
            if 'pd' not in once:
                once['pd'] = _is_positive_definite(K)
            return once['pd']

        def verify(mu, eigvecs):
            return _check_both_signs(K, KG, mu, eigvecs, check_rtol,
                    K_positive_definite=K_positive_definite)

        strategies = [
            ('eigsh() in Cayley mode',
             lambda: _eigsh_both_signs(K, KG, num_eigvalues, tol, silent,
                    mu_max=mu_max(), K_positive_definite=K_positive_definite)),
            ('eigsh() in Cayley mode with a 10 times larger shift',
             lambda: _eigsh_both_signs(K, KG, num_eigvalues, tol, silent,
                    safety=100., mu_max=mu_max(),
                    K_positive_definite=K_positive_definite)),
        ]
        dense = ('eigh() with condensed dofs',
                 lambda: _eigh_condensed(K, KG, num_eigvalues))
        (eigvals, peigvecs), _ = _sparse_first('Linear buckling analysis',
                strategies, verify, dense, num_active, max_dense_size, silent,
                hint=(' If ARPACK is linked against Intel MKL 2024.2.0 to '
                      '2025.0.0, check that the cap of ncv in '
                      'structsolve.arpackutils is not disabled, or update '
                      'MKL to 2025.0.1 or newer. Otherwise, try '
                      'sparse_solver=False or a larger max_dense_size.'))

    else:
        K = csr_matrix(K).toarray()
        KG = csr_matrix(KG).toarray()
        msg('eigh() solver...', level=3, silent=silent)
        eigvals, peigvecs = eigh(a=KG, b=K)
        eigvals[_null_mu_pencil(KG, K, eigvals, peigvecs, peigvecs)] = 0.
        eigvals, peigvecs = _sort_eigenpairs(eigvals, peigvecs)
        msg('finished!', level=3, silent=silent)

    num_eigvecs = min(num_eigvalues, peigvecs.shape[1])
    if (np.iscomplexobj(peigvecs)
            and np.all(np.imag(eigvals[:num_eigvecs]) == 0)):
        # all eigenvalues are returned by eig(), only the returned
        # eigenvectors must be real
        peigvecs = peigvecs[:, :num_eigvecs].real
    if used_cols is not None:
        eigvecs = np.zeros((size, num_eigvecs), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs[:, :num_eigvecs]
    else:
        eigvecs = peigvecs[:, :num_eigvecs]

    with np.errstate(divide='ignore', invalid='ignore'):
        null = eigvals == 0
        eigvals = -1./eigvals
    eigvals[null] = -np.inf

    msg('finished!', level=2, silent=silent)

    msg('first {0} eigenvalues:'.format(num_eigvalues_print), level=1,
        silent=silent)

    for eigval in eigvals[:num_eigvalues_print]:
        msg('{0}'.format(eigval), level=2, silent=silent)

    return eigvals, eigvecs


def _lb_unsymmetric(K, KG, tol, sparse_solver, silent, num_eigvalues,
                    max_dense_size, check_rtol):
    r"""Eigenvalues `\mu` of ``KG u = mu K u`` for unsymmetric matrices, see
    the Notes of :func:`lb`"""
    if not sparse_solver:
        msg('eig() solver...', level=3, silent=silent)
        KG = csr_matrix(KG).toarray()
        K = csr_matrix(K).toarray()
        mu, vl, eigvecs = _eig_complex(a=KG, b=K, left=True)
        mu[_null_mu_pencil(KG, K, mu, vl, eigvecs)] = 0.
        mu, eigvecs = _general_eigenpairs(mu, eigvecs)
        msg('finished!', level=3, silent=silent)
        return mu, eigvecs
    K = csr_matrix(K)
    KG = csr_matrix(KG)
    num_active = int(np.count_nonzero(
        (np.asarray(abs(KG).sum(axis=1)).ravel() > 0)
        | (np.asarray(abs(KG).sum(axis=0)).ravel() > 0)))
    msg('{0} of {1} rows or columns of KG are non-null'.format(num_active,
        K.shape[0]), level=3, silent=silent)

    def verify(mu, eigvecs):
        with np.errstate(divide='ignore', invalid='ignore'):
            return _check_eigenpairs(K, KG, -1./mu, eigvecs, check_rtol,
                                     check_inertia=False)

    strategies = [('eigs() shift search along the positive real axis',
                   lambda: _eigs_shift_search(K, KG, num_eigvalues, tol,
                                              check_rtol, silent))]
    dense = ('eig() with condensed dofs',
             lambda: _eig_condensed(K, KG, num_eigvalues))
    (mu, eigvecs), name = _sparse_first(
        'Linear buckling analysis of unsymmetric matrices', strategies,
        verify, dense, num_active, max_dense_size, silent,
        hint=' Try sparse_solver=False or a larger max_dense_size.')
    if name == dense[0] and not np.any(np.isreal(mu) & (np.real(mu) < 0)):
        text = ('Linear buckling analysis of unsymmetric matrices: no positive '
                'real load multiplier exists, i.e. no divergence load, the '
                'returned load multipliers are negative or complex')
        warn(text, level=3, silent=silent)
        warnings.warn(text, RuntimeWarning, stacklevel=3)
    return _general_eigenpairs(mu, eigvecs, num_eigvalues)
