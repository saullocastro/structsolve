"""Frequency analysis (:func:`freq`)

Natural frequencies and modes from the stiffness and mass matrices, computed
with the sparse eigenvalue solvers first and with a dense solver as a last
resort, with a :class:`.linear_buckling.DenseFallbackWarning`.

"""
import warnings

import numpy as np
import scipy
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, splu
from scipy.linalg import eigh, LinAlgError

from .arpackutils import capped_eigsh, release_memory, start_kwargs
from .linear_buckling import (BACKWARD_ERROR_RTOL, RIGID_BODY_RTOL,
                              _backward_errors, _check_linear_solver,
                              _eig_complex, _is_positive_definite,
                              _negative_pivots, _relative_residuals,
                              _solve_function, _sparse_first, is_symmetric)
from .logger import msg, warn
from .sparseutils import remove_null_cols


def _estimate_sigma(K, M, linear_solver='splu'):
    """Negative shift of :func:`freq`, see its Notes; ``K`` is factorized
    by :func:`.linear_buckling._solve_function`"""
    try:
        rhs = M @ np.random.RandomState(42).randn(K.shape[0])
        try:
            if linear_solver == 'pardiso':
                y = _solve_function(K, linear_solver)(rhs)
            else:
                y = splu(csc_matrix(K)).solve(rhs)
        except RuntimeError:
            # exactly singular
            return -1.
        residual = np.linalg.norm(K @ y - rhs)
        if residual > 1e-6 * np.linalg.norm(rhs):
            return -1.
        sigma = -abs((y @ K @ y) / (y @ M @ y))
        if not np.isfinite(sigma) or sigma == 0:
            return -1.
        return sigma
    except Exception:
        return -1.


def _residuals(K, M, lambda2, eigvecs):
    r"""Relative residuals `||K u + \lambda^2 M u||/(||K u|| + |\lambda^2|
    ||M u||)` of the finite eigenpairs, relative to the matrix-norm scale
    `(||K||_F + |\lambda^2| ||M||_F) ||u||` instead for the rigid-body
    modes, see :func:`.linear_buckling._relative_residuals` with
    ``rigid_rtol=RIGID_BODY_RTOL``"""
    lambda2 = np.asarray(lambda2)
    finite = np.isfinite(lambda2)
    return _relative_residuals(K, M, lambda2[finite], eigvecs[:, finite],
                               rigid_rtol=RIGID_BODY_RTOL)


def _inertia_confirmed(K, M, w2, candidates, rtol, linear_solver='splu'):
    r"""Eigenvalues `\omega^2` confirmed by the inertia of `\boldsymbol{K}
    - s \boldsymbol{M}`, for symmetric ``K`` and ``M`` with a positive
    definite ``M``

    Parameters
    ----------
    K, M : sparse_matrix
        Stiffness and mass matrices, shape ``(n, n)``.
    w2 : ndarray
        All returned real eigenvalues `\omega^2`, shape ``(k,)``.
    candidates : array_like
        Indices of ``w2`` to confirm.
    rtol : float
        Relative half-width of the interval around each `\omega^2`.
    linear_solver : str, optional
        Passed to :func:`.linear_buckling._negative_pivots`.

    Returns
    -------
    confirmed : ndarray
        Boolean array, shape ``(len(candidates),)``: ``True`` when the
        number of eigenvalues in the interval of half-width ``rtol``
        `|\omega^2|` around `\omega^2` is known and not lower than the
        number of returned eigenvalues ``w2`` in it; ``False`` for
        `\omega^2 = 0`.

    Notes
    -----
    By Sylvester's law of inertia, see :func:`_check_freq`, the number of
    negative pivots `\nu(s)` of `\boldsymbol{K} - s \boldsymbol{M}` is the
    number of eigenvalues lower than `s`, and `\nu(b) - \nu(a)` the number
    of eigenvalues in `[a, b)`. Each candidate costs two factorizations,
    shared by candidates with the same bounds.

    """
    counts = {}

    def count(s):
        if s not in counts:
            counts[s] = _negative_pivots(K - s*M, linear_solver)
        return counts[s]

    confirmed = np.zeros(len(candidates), dtype=bool)
    for j, i in enumerate(candidates):
        if w2[i] == 0:
            continue
        a = w2[i] - rtol*abs(w2[i])
        b = w2[i] + rtol*abs(w2[i])
        ca, cb = count(a), count(b)
        inside = int(np.count_nonzero((w2 >= a) & (w2 <= b)))
        confirmed[j] = ca is not None and cb is not None and cb - ca >= inside
    return confirmed


def _failed_residuals(K, M, lambda2, eigvecs, check_rtol, definite=False,
                      linear_solver='splu', max_inertia=None):
    r"""Finite eigenpairs that fail the residual check of :func:`freq`

    An eigenpair passes when

    - its relative residual of :func:`_residuals` is not larger than
      ``check_rtol``, or
    - its normwise backward error of
      :func:`.linear_buckling._backward_errors` is not larger than
      ``linear_buckling.BACKWARD_ERROR_RTOL``, the machine precision, i.e.
      it is accurate up to the round-off, e.g. an eigenvalue close to zero
      of a model pre-stressed close to buckling, or
    - for a symmetric definite pencil, see ``definite``, its real eigenvalue
      `\omega^2 = -\lambda^2` is confirmed by the inertia within the
      relative distance ``check_rtol``, see :func:`_inertia_confirmed`.

    Parameters
    ----------
    K, M : sparse_matrix
        Stiffness and mass matrices, shape ``(n, n)``.
    lambda2 : array_like
        Eigenvalues `\lambda^2`, shape ``(k,)``.
    eigvecs : ndarray
        Eigenvectors, ``eigvecs[:, i]`` that of ``lambda2[i]``, shape
        ``(n, k)``.
    check_rtol : float
        Maximum relative residual, see :func:`freq`.
    definite : bool or None, optional
        ``True`` for symmetric ``K`` and ``M`` with a positive definite
        ``M``, ``False`` to skip the inertia, ``None`` to check it with
        :func:`.linear_buckling.is_symmetric` and
        :func:`.linear_buckling._is_positive_definite` when an eigenpair
        fails the first two conditions.
    linear_solver : str, optional
        Passed to :func:`.linear_buckling._negative_pivots`.
    max_inertia : int or None, optional
        Maximum number of eigenpairs checked with the inertia, the lowest
        `\omega^2` first; ``None`` for all.

    Returns
    -------
    failed : ndarray
        Boolean array, ``True`` for the finite eigenpairs that fail, shape
        ``(k,)`` with ``k`` the number of finite ``lambda2``.
    res, backward_error : ndarray
        Relative residuals and backward errors of the finite eigenpairs,
        shape ``(k,)``.

    """
    lambda2 = np.asarray(lambda2)
    finite = np.isfinite(lambda2)
    res = _residuals(K, M, lambda2, eigvecs)
    backward_error = _backward_errors(K, M, lambda2[finite],
                                      eigvecs[:, finite])
    failed = ~((res <= check_rtol) | (backward_error <= BACKWARD_ERROR_RTOL))
    if np.any(failed) and definite is None:
        definite = (is_symmetric(K) and is_symmetric(M)
                    and _is_positive_definite(csr_matrix(M), linear_solver))
    if np.any(failed) and definite:
        w2 = -lambda2[finite]
        real = np.abs(np.imag(w2)) <= check_rtol*np.abs(w2)
        w2 = np.real(w2)
        candidates = np.flatnonzero(failed & real)
        candidates = candidates[np.argsort(w2[candidates], kind='stable')]
        if max_inertia is not None:
            candidates = candidates[:max_inertia]
        if candidates.size:
            index = np.searchsorted(np.flatnonzero(real), candidates)
            confirmed = _inertia_confirmed(csr_matrix(K), csr_matrix(M),
                                           w2[real], index, check_rtol,
                                           linear_solver)
            failed[candidates[confirmed]] = False
    return failed, res, backward_error


def _check_freq(K, M, lambda2, eigvecs, check_rtol, symmetric,
                min_rel_gap=1e-4, linear_solver='splu'):
    r"""Verify the eigenpairs of ``(K + lambda2*M) @ u = 0``


    - Unless ``check_rtol`` is ``None``, the relative residuals of
      :func:`_residuals` must not be larger than ``check_rtol``, or the
      backward errors not larger than
      ``linear_buckling.BACKWARD_ERROR_RTOL``, or, for symmetric ``K`` and
      ``M`` (``symmetric=True``, or matrices found symmetric with a positive
      definite ``M``), the eigenvalues must be confirmed by the inertia, see
      :func:`_failed_residuals`, called with ``linear_solver``.
    - With ``symmetric=True``, no eigenvalue ``omega**2 = -lambda2`` is
      missing below the highest one returned, checked with the number of
      negative pivots of ``K - s M`` for ``s`` slightly lower, by the
      relative gap ``min_rel_gap``. The check is skipped
      when the count is unknown, see
      :func:`.linear_buckling._negative_pivots`, called with
      ``linear_solver``, or when ``s`` is zero.

    Returns
    -------
    error : str or None
        Description of the failed check, ``None`` when all checks passed.

    Notes
    -----
    With ``symmetric=True``, no eigenvalue `\omega^2 = -\lambda^2` is
    missing below the highest one returned, `\omega^2_n`: the number of
    negative pivots of `\boldsymbol{K} - s \boldsymbol{M}`, with
    `s = \omega^2_n - min\_rel\_gap |\omega^2_n|`, i.e. the number of
    eigenvalues lower than `s` by Sylvester's law of inertia for a
    positive definite ``M``, must be the number of returned eigenvalues
    lower than `s`.

    For symmetric `\boldsymbol{K}` and `\boldsymbol{M}` with a positive definite `\boldsymbol{M}`, `\boldsymbol{K} -
    s \boldsymbol{M}` is congruent to `\boldsymbol{M}^{-1/2} \boldsymbol{K} \boldsymbol{M}^{-1/2} - s \boldsymbol{I}`, whose
    eigenvalues are `\omega_i^2 - s`; its number of negative eigenvalues,
    equal to its number of negative pivots by Sylvester's law of inertia,
    is therefore the number of `\omega_i^2 < s`. ``K`` need not be positive
    definite.

    """

    lambda2 = np.asarray(lambda2)
    if not eigvecs.shape[1]:
        return 'no eigenpair was found'
    if check_rtol is not None:
        failed, res, backward_error = _failed_residuals(
            K, M, lambda2, eigvecs, check_rtol,
            definite=True if symmetric else None,
            linear_solver=linear_solver)
        if np.any(failed):
            i = np.argmax(np.where(failed, np.nan_to_num(res, nan=np.inf),
                                   -np.inf))
            return ('relative residual {0:.2e} > {1:.1e} and backward error '
                    '{2:.1e} > {3:.1e}, not confirmed by the inertia, for '
                    'the eigenvalue lambda**2={4}'
                    .format(res[i], check_rtol, backward_error[i],
                            BACKWARD_ERROR_RTOL,
                            lambda2[np.isfinite(lambda2)][i]))
    if symmetric:
        w2 = -np.real(lambda2[np.isfinite(lambda2)])
        if not w2.size:
            return 'no finite eigenvalue was found'
        s = w2.max() - min_rel_gap*abs(w2.max())
        expected = int(np.count_nonzero(w2 < s))
        count = (_negative_pivots(K - s*M, linear_solver) if s != 0
                 else None)
        if count is not None and count != expected:
            return ('{0} eigenvalues omega**2 lower than {1} exist, {2} were '
                    'found'.format(count, s, expected))
    return None


def _eigsh_freq(K, M, k, tol, sigma, linear_solver='splu'):
    """:func:`.arpackutils.capped_eigsh` in shift-invert mode"""
    eigvals, eigvecs = capped_eigsh(A=K, k=k, M=M, sigma=sigma, which='LM',
                                    mode='normal', tol=tol,
                                    linear_solver=linear_solver)
    return -eigvals, eigvecs


def _eigs_freq(K, M, k, tol, sigma):
    """:func:`scipy.sparse.linalg.eigs` in shift-invert mode"""
    eigvals, eigvecs = eigs(A=K, M=M, k=k, which='LM', tol=tol, sigma=sigma,
                            **start_kwargs(K.shape[0]))
    release_memory()
    #NOTE eigs solves: [K] {u} = eigval [M] {u}
    #     therefore we must correct he sign of lambda^2 here:
    return -eigvals, eigvecs


def _dense_eigh(Kd, Md, sigma):
    r"""All eigenpairs of the dense symmetric ``Kd`` and ``Md``, by
    :func:`scipy.linalg.eigh`, see :func:`freq`

    Parameters
    ----------
    Kd, Md : ndarray
        Stiffness and mass matrices, shape ``(n, n)``.
    sigma : float
        Negative shift of :func:`_estimate_sigma`.

    Returns
    -------
    w2 : ndarray
        The eigenvalues `\omega^2`, increasing, ``inf`` for those beyond
        the resolution of the shifted inverse problem, see the Notes, shape
        ``(n,)``.
    eigvecs : ndarray
        The eigenvectors, normalized with respect to ``Md``, shape
        ``(n, n)``.

    Notes
    -----
    The shifted inverse problem `\boldsymbol{M} \boldsymbol{u} = \mu
    (\boldsymbol{K} - \sigma \boldsymbol{M}) \boldsymbol{u}`, with `\mu =
    1/(\omega^2 - \sigma)`, is solved, which requires a positive definite
    `\boldsymbol{K} - \sigma \boldsymbol{M}`, i.e. no `\omega^2 \le
    \sigma`; a positive semi-definite `\boldsymbol{K}`, e.g. with
    rigid-body modes, is allowed with `\sigma < 0` ([2]_, Section 10.2,
    p. 860, Section 10.2.3). A computed `\mu \le 0`,
    a round-off of the smallest `\mu`, gives `\omega^2 = \infty`. When the
    Cholesky factorization of `\boldsymbol{K} - \sigma \boldsymbol{M}`
    fails, `\boldsymbol{K} \boldsymbol{u} = \omega^2 \boldsymbol{M}
    \boldsymbol{u}` is solved instead, which requires a positive definite
    `\boldsymbol{M}`.

    For `\boldsymbol{A} \boldsymbol{z} = \lambda \boldsymbol{B}
    \boldsymbol{z}`, the approximate error bound of LAPACK for every
    computed eigenvalue is `\epsilon (\|\boldsymbol{A}\|/\|\boldsymbol{B}\|
    + |\lambda|)/r_B^2`, with `\epsilon` the machine precision and `r_B`
    the reciprocal condition number of the Cholesky factor of
    `\boldsymbol{B}` ([1]_, section "Error Bounds for the Generalized
    Symmetric Definite Eigenproblem"): an absolute error of nearly the same
    size for all eigenvalues, hence the largest relative errors for the
    eigenvalues closest to zero. These are the lowest `\omega^2` of
    `\boldsymbol{K} \boldsymbol{u} = \omega^2 \boldsymbol{M}
    \boldsymbol{u}`, and the highest `\omega^2`, the smallest `\mu`, of the
    shifted inverse problem, which is therefore accurate for the lowest
    natural frequencies and less accurate for the highest ones. Measured for
    a free-free beam of 400 Euler-Bernoulli elements split into two halves
    joined by a penalty spring `10^6 EI/l_e^3`, with `\sigma = -1`: the
    lowest elastic `\omega_n = 230.0` rad/s from `\boldsymbol{K}
    \boldsymbol{u} = \omega^2 \boldsymbol{M} \boldsymbol{u}`, with a
    relative residual of 0.58, against 326.0 rad/s from the shifted inverse
    problem, the sparse solver and the inertia count; the relative
    residuals of the shifted inverse problem are larger than `10^{-6}` for
    768 of its 804 eigenpairs, the highest, the lowest 36 passing.

    References
    ----------
    .. [1] Anderson, E., Bai, Z., Bischof, C., Blackford, S., Demmel, J.,
       Dongarra, J., Du Croz, J., Greenbaum, A., Hammarling, S., McKenney,
       A., Sorensen, D., LAPACK Users' Guide, 3rd ed., SIAM, Philadelphia,
       1999. https://doi.org/10.1137/1.9780898719604
    .. [2] Bathe, K.J., Finite Element Procedures, 2nd ed., K.J. Bathe,
       Watertown, MA, 2014.

    """
    try:
        mu, eigvecs = eigh(a=Md, b=Kd - sigma*Md)
    except LinAlgError:
        return eigh(a=Kd, b=Md)
    order = np.argsort(-mu, kind='stable')
    mu = mu[order]
    eigvecs = eigvecs[:, order]
    eigvecs /= np.sqrt(np.einsum('ij,ij->j', eigvecs, Md @ eigvecs))
    w2 = np.full(mu.shape, np.inf)
    positive = mu > 0
    w2[positive] = sigma + 1./mu[positive]
    return w2, eigvecs


def _dense_freq(K, M, k, sigma, symmetric):
    r"""Dense fallback of the sparse solvers of :func:`freq`

    With ``symmetric=True``, the ``k`` lowest `\omega^2` of
    :func:`_dense_eigh`, with eigenvectors normalized with respect to ``M``.
    Otherwise :func:`scipy.linalg.eig`, through the complex LAPACK driver
    ``zggev``, see :func:`.linear_buckling._eig_complex`, and the ``k``
    finite `\omega^2` nearest ``sigma``, as the sparse solver, by increasing
    distance.

    """
    Kd = csr_matrix(K).toarray()
    Md = csr_matrix(M).toarray()
    if symmetric:
        w2, eigvecs = _dense_eigh(Kd, Md, sigma)
        return -w2[:k], eigvecs[:, :k]
    eigvals, eigvecs = _eig_complex(a=Md, b=Kd)
    with np.errstate(divide='ignore', invalid='ignore'):
        w2 = 1./eigvals
    finite = np.flatnonzero(np.isfinite(w2))
    order = finite[np.argsort(np.abs(w2[finite] - sigma), kind='stable')[:k]]
    return -w2[order], eigvecs[:, order]


def freq(K, M, tol=0, sparse_solver=True,
        silent=False, sort=True, num_eigvalues=25,
        num_eigvalues_print=5, skip_null_cols=False, symmetric=False,
        check_rtol=1e-6, max_dense_size=2000, linear_solver='splu'):
    r"""Frequency analysis

    Calculates the eigenvalues ``lambda2`` and eigenvectors ``u`` of the
    free-vibration problem of ``K`` and ``M``, i.e. the eigenpairs with
    ``(K + lambda2*M) @ u = 0``, where ``lambda2 = -omega_n**2`` and
    ``omega_n`` is the natural frequency in rad/s. The null rows and columns
    of ``K`` are removed from both matrices before solving the eigenvalue
    problem, unless ``skip_null_cols=True``.

    **Solver policy.** As in :func:`.lb`, with ``sparse_solver=True`` the
    sparse solver runs first for every problem size, and its eigenpairs are
    verified:

    - the relative residual of each eigenpair must not be larger than
      ``check_rtol``, unless its backward error is at the round-off or, for
      symmetric matrices, its eigenvalue is confirmed by the inertia, see
      ``check_rtol``;
    - with ``symmetric=True``, no eigenvalue ``omega_n**2`` lower than the
      highest one returned may be missing, see :func:`_check_freq`.

    When the verification fails or the sparse solver raises an error, e.g.
    ARPACK did not converge, the dense solver is used as a last resort if
    the problem has at most ``max_dense_size`` dofs, with a
    :class:`.linear_buckling.DenseFallbackWarning` giving the reasons. It
    returns the ``num_eigvalues`` lowest eigenvalues with ``symmetric=True``,
    otherwise the ``num_eigvalues`` eigenvalues closest to the shift, as the
    sparse solver, and its eigenpairs are verified as well. Otherwise a
    ``RuntimeError`` is raised.

    With Intel MKL 2024.2.0 to 2025.0.0, ``ncv`` of
    :func:`scipy.sparse.linalg.eigsh` is capped, see :func:`.lb` and
    :mod:`structsolve.arpackutils`.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix. Should include initial stress stiffness matrix,
        aerodynamic matrix and so forth when applicable.
    M : sparse_matrix
        Mass matrix.
    tol : float, optional
        A tolerance value passed to :func:`scipy.sparse.linalg.eigs` and
        :func:`scipy.sparse.linalg.eigsh`.
    sparse_solver : bool, optional
        With ``True``, the default, the sparse solver always runs first,
        whatever the size of the problem: :func:`scipy.sparse.linalg.eigs`,
        or :func:`scipy.sparse.linalg.eigsh` with ``symmetric=True``, in
        shift-invert mode with a negative shift estimated from the matrices,
        calculating the ``num_eigvalues`` eigenvalues closest to the shift.
        Its eigenpairs are verified, and a dense solver is only used as a
        last resort, see above. With ``False``, the full problem is
        explicitly solved with a dense solver, without trying the sparse
        solver, which calculates all eigenvalues, e.g. for the kinetic
        criterion of stability: :func:`scipy.linalg.eig`, or, with
        ``symmetric=True``, :func:`scipy.linalg.eigh` for the shifted
        inverse problem, accurate for the lowest natural frequencies and
        less accurate for the highest ones, see the Notes. With
        ``symmetric=True``, a ``UserWarning`` is issued when one of the
        ``num_eigvalues`` lowest eigenpairs fails the residual check, see
        ``check_rtol``.

    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    sort : bool, optional
        Sort the eigenvalues and the eigenvectors together by increasing
        natural frequency, keeping only those with a natural frequency
        larger than ``1e-6`` rad/s, e.g. without the rigid-body modes. The
        sort key is the real part of `\omega_n = \sqrt{-\lambda^2}` rounded
        to one decimal, then its imaginary part rounded to one decimal, then,
        between equal rounded keys, the exact real and imaginary parts of
        `\omega_n`, then those of `\omega_n^2 = -\lambda^2`, which separate
        two `\omega_n^2` differing at the round-off that have the same
        `\omega_n` after the square root. Only identical eigenvalues keep
        the order of the solver, such that a multiple eigenvalue returned
        with round-off differences, in an order that may depend on the
        LAPACK build, is sorted by increasing exact value. With ``False``
        the eigenpairs are returned in the order of the solver.
    num_eigvalues : int, optional
        Number of calculated eigenvalues with the sparse solver, and of the
        lowest eigenpairs checked with ``sparse_solver=False`` and
        ``symmetric=True``, limited to the size of ``M`` minus 2.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.
    skip_null_cols : bool, optional
        If True, skip the removal of null columns from the matrices.
        Use only when ``K`` and ``M`` are known to be non-singular.
    symmetric : bool or None, optional
        ``False``, the default, uses the general solvers described above,
        valid for unsymmetric matrices, e.g. ``K`` with an aerodynamic matrix
        or with the load stiffness of follower loads. ``True`` uses the
        solvers of symmetric matrices, :func:`scipy.sparse.linalg.eigsh` in
        shift-invert mode or :func:`scipy.linalg.eigh`, which require
        symmetric ``K`` and ``M`` and a positive definite ``M``, and return
        real eigenvalues and eigenvectors normalized with respect to ``M``.
        ``None`` selects them when :func:`.linear_buckling.is_symmetric` is
        true for both matrices.
    check_rtol : float or None, optional
        Maximum relative residual of an eigenpair,
        `||K u + \lambda^2 M u||/(||K u|| + |\lambda^2| ||M u||)`. For a
        rigid-body mode, whose `||K u|| + |\lambda^2| ||M u||` is not larger
        than ``linear_buckling.RIGID_BODY_RTOL`` times the matrix-norm scale
        `(||K||_F + |\lambda^2| ||M||_F) ||u||`, the residual is relative to
        this scale, see :func:`.linear_buckling._relative_residuals`. An
        eigenpair with a larger relative residual is also accepted when its
        normwise backward error `||K u + \lambda^2 M u||/((||K||_F +
        |\lambda^2| ||M||_F) ||u||)` is not larger than
        ``linear_buckling.BACKWARD_ERROR_RTOL``, the machine precision, e.g.
        an eigenvalue close to zero of a model pre-stressed close to
        buckling, see :func:`.linear_buckling._backward_errors`, or, for
        symmetric ``K`` and ``M`` with a positive definite ``M``
        (``symmetric=True``, or matrices found symmetric by
        :func:`.linear_buckling.is_symmetric`), when the inertia of
        ``K - s*M`` confirms an eigenvalue within the relative distance
        ``check_rtol`` of its ``omega_n**2``, see the Notes. Any other
        larger residual of the sparse solver is a failed verification, see
        above. For ``sparse_solver=False``, a log message is printed, and,
        with ``symmetric=True``, a ``UserWarning`` is issued for the
        ``num_eigvalues`` lowest eigenpairs, see ``sparse_solver``. ``None``
        skips the check.
    max_dense_size : int, optional
        Maximum size of the dense problem solved as a last resort when the
        sparse solver failed, see above. Larger problems raise a
        ``RuntimeError`` instead, and ``0`` disables the dense fallback.
    linear_solver : str, optional
        ``'splu'`` (default) for the sparse factorizations of SuperLU, or
        ``'pardiso'`` for those of PARDISO, which requires the shared
        library ``mkl_rt`` of Intel oneMKL, see :mod:`structsolve.pardiso`,
        with ``symmetric=True`` (or ``None`` detecting symmetric matrices)
        only; see the Notes.

    Returns
    -------
    lambda2 : ndarray
        Complex array with the eigenvalues `\lambda^2 = -\omega_n^2`, real
        with ``symmetric=True``, sorted and filtered as described in
        ``sort``, otherwise in the order returned by the eigenvalue solver.
    eigvecs : ndarray
        The `i^{th}` eigenvector is ``eigvecs[:, i]``, the eigenvector of
        ``lambda2[i]``, with the size of the original matrices. The solvers
        of symmetric matrices return eigenvectors normalized with respect to
        ``M``, :func:`scipy.linalg.eig` returns eigenvectors with unit
        Euclidean norm.

    Raises
    ------
    RuntimeError
        When the sparse solver failed and the dense fallback was not
        possible or failed too, listing the attempted solvers and their
        failures.

    Warns
    -----
    .linear_buckling.DenseFallbackWarning
        When the sparse solver failed and the dense solver was used.
    UserWarning
        With ``sparse_solver=False`` and ``symmetric=True``, when one of the
        ``num_eigvalues`` lowest eigenpairs fails the residual check.

    Notes
    -----
    The free-vibration eigenvalue problem is

    .. math::

        (\boldsymbol{K} + \lambda^2 \boldsymbol{M}) \boldsymbol{u} = \boldsymbol{0}

    where `\lambda^2 = -\omega_n^2` and `\omega_n` is the natural frequency
    in rad/s. The sparse solvers solve `\boldsymbol{K} \boldsymbol{u} = \omega_n^2 \boldsymbol{M} \boldsymbol{u}`
    in shift-invert mode, which returns the `\omega_n^2` nearest the shift
    `\sigma` ([1]_, Chapter 3, Section 3.2, Eq. (3.2.2)). The negative
    shift is `\sigma = -|\boldsymbol{y}^\top \boldsymbol{K} \boldsymbol{y} /
    \boldsymbol{y}^\top \boldsymbol{M} \boldsymbol{y}|`, the Rayleigh quotient of `\boldsymbol{y} = \boldsymbol{K}^{-1} \boldsymbol{M} \boldsymbol{x}`
    for a fixed random vector `\boldsymbol{x}`, i.e. one step of inverse iteration
    towards the lowest `\omega_n^2`, or `\sigma = -1` when ``K`` is
    singular or the solution is inaccurate. With all `\omega_n^2 \ge 0`,
    the eigenvalues nearest a negative shift are the lowest ones.

    **Verification.** With ``symmetric=True``, the number of negative
    pivots of `\boldsymbol{K} - s \boldsymbol{M}`, for `s` slightly lower than the highest
    `\omega_n^2` returned, is the number of eigenvalues below `s`
    (Sylvester's law of inertia), see :func:`_check_freq`.

    For an accurate eigenpair, the residual `\boldsymbol{K} \boldsymbol{u}
    - \omega_n^2 \boldsymbol{M} \boldsymbol{u}` computed in floating point
    is of the order of the round-off of `\boldsymbol{K} \boldsymbol{u}`,
    `\epsilon \|\boldsymbol{K}\| \|\boldsymbol{u}\|`, with `\epsilon` the
    machine precision, while `\|\boldsymbol{K} \boldsymbol{u}\| =
    \omega_n^2 \|\boldsymbol{M} \boldsymbol{u}\|`: the relative residuals of
    the lowest modes of stiff models are large. Measured with
    :func:`scipy.sparse.linalg.eigsh` for a free-free beam of 400
    Euler-Bernoulli elements (802 dofs, lowest elastic `\omega_n = 326.00`
    rad/s, analytic 326.00 rad/s): relative residuals up to `7.6 \cdot
    10^{-4}` and normwise backward errors of 0.4 to `32 \epsilon`, see
    :func:`.linear_buckling._backward_errors` and [3]_, while the inertia
    confirms each eigenvalue within the relative distance `10^{-6}`. No
    constant multiple of `\epsilon` separates these from inaccurate
    eigenpairs, because the backward error is relative to
    `\|\boldsymbol{K}\|_F`, set by the stiffest entries: eigenvalues 0.1 %
    too low of the blade-stiffened plate with penalty connections of
    ``tests/data``, with `\boldsymbol{K}_G` in place of `\boldsymbol{M}`,
    have backward errors of 16 to `53 \epsilon`, and the inaccurate
    eigenpairs of :func:`scipy.sparse.linalg.eigsh` for two decoupled beams
    of 400 elements, two of the four lowest elastic `\omega_n^2` wrong by
    `-9 \cdot 10^{-6}` and `5 \cdot 10^{-5}`, up to `716 \epsilon`, below
    `n \epsilon = 1604 \epsilon`.

    Hence a relative residual larger than ``check_rtol`` is accepted, besides
    a backward error not larger than `\epsilon`, only when the eigenvalue is
    confirmed independently of the residual. For symmetric `\boldsymbol{K}`
    and `\boldsymbol{M}` with a positive definite `\boldsymbol{M}`, the
    numbers of negative pivots `\nu(a)` and `\nu(b)` of `\boldsymbol{K} - a
    \boldsymbol{M}` and `\boldsymbol{K} - b \boldsymbol{M}`, with `a, b =
    \omega_n^2 \mp \tau |\omega_n^2|` and `\tau` = ``check_rtol``, give the
    number `\nu(b) - \nu(a)` of eigenvalues in `[a, b)` (the Sturm sequence
    check of [4]_, Section 11.6, p. 964), which must not be lower than the
    number of returned eigenvalues in this interval, see
    :func:`_inertia_confirmed`. The count costs two factorizations per
    eigenpair, only for the eigenpairs that fail the other two criteria. It
    confirms the eigenvalue, not the eigenvector, which is verified only as
    far as the residual allows: the lowest elastic mode of the beam above
    mixed with `10^{-3}` times the next one, with the exact eigenvalue, has
    a relative residual of `3.4 \cdot 10^{-3}`, against `7.6 \cdot 10^{-4}`
    for the accurate one. The resolution of the count is limited by the
    round-off of the factorization: for the cantilever pre-stressed to
    0.9999 of its buckling load, see :func:`.linear_buckling._backward_errors`,
    the count confirms the two lowest eigenvalues within `10^{-5}` but not
    `10^{-6}`; these eigenpairs pass with their backward errors. The
    componentwise backward error `\max_i |r_i|/((|\boldsymbol{K}|
    |\boldsymbol{u}| + |\lambda^2| |\boldsymbol{M}| |\boldsymbol{u}|)_i)`,
    with `\boldsymbol{r}` the residual vector, as for linear systems ([5]_,
    section "Error Bounds for Linear Equation Solving"), does not accept the
    accurate eigenpairs either: `6 \cdot 10^3` to `4 \cdot 10^5 \epsilon`
    for the beam above.

    **PARDISO.** With ``linear_solver='pardiso'`` and symmetric matrices,
    ``K`` of the estimate of the shift, `\boldsymbol{K} - \sigma
    \boldsymbol{M}` of the shift-invert mode, passed to
    :func:`scipy.sparse.linalg.eigsh` as ``OPinv`` (also in the passes of
    :func:`.arpackutils.capped_eigsh`), and `\boldsymbol{K} - s
    \boldsymbol{M}` of the inertia count are factorized by PARDISO, the
    symmetric indefinite factorization of :func:`.static._factorize`, see
    :func:`.lb`; ARPACK remains the eigenvalue solver. The count of negative
    eigenvalues of PARDISO is exact also when SuperLU would need a row
    interchange. As in :func:`.lb`, SuperLU is used instead for a matrix
    whose factorization by PARDISO is inaccurate or, for the count, keeps
    perturbed pivots. The general solver of unsymmetric matrices,
    :func:`scipy.sparse.linalg.eigs`, and the dense solvers are unchanged.

    :func:`scipy.sparse.linalg.eigsh` uses ARPACK's symmetric drivers, which
    return wrong eigenpairs when linked against Intel MKL 2024.2.0 to
    2025.0.0 and ``ncv > 32``, see :func:`.lb` and
    :mod:`structsolve.arpackutils`, where ``ncv`` is capped accordingly.

    Non-conservative systems, e.g. under follower loads, may lose stability
    by flutter, which the static criterion of :func:`.lb` does not detect:
    with ``K`` the tangent stiffness matrix at a load level, including the
    load stiffness, the eigenvalues `\lambda^2` of the general solvers are
    real and negative below the flutter load and become complex conjugate
    pairs above it, when two natural frequencies coalesce, a double root of
    the frequency equation. This is the dynamic method of [2]_
    (Introduction, Sections 6 and 7, pp. 12-17), where `\lambda` is the
    characteristic exponent `s = i\Omega`. A real positive `\lambda^2`,
    i.e. a real positive `\lambda`, indicates divergence ([2]_, p. 17).

    References
    ----------
    .. [1] Lehoucq, R.B., Sorensen, D.C., Yang, C., ARPACK Users' Guide:
       Solution of Large-Scale Eigenvalue Problems with Implicitly
       Restarted Arnoldi Methods, SIAM, Philadelphia, 1998.
       https://doi.org/10.1137/1.9780898719628
    .. [2] Bolotin, V.V., Nonconservative Problems of the Theory of Elastic
       Stability, Pergamon Press, Oxford, 1963.
    .. [3] Tisseur, F., Backward error and condition of polynomial eigenvalue
       problems, Linear Algebra and its Applications, 309(1-3), 339-361,
       2000. https://doi.org/10.1016/S0024-3795(99)00063-4
    .. [4] Bathe, K.J., Finite Element Procedures, 2nd ed., K.J. Bathe,
       Watertown, MA, 2014.
    .. [5] Anderson, E., Bai, Z., Bischof, C., Blackford, S., Demmel, J.,
       Dongarra, J., Du Croz, J., Greenbaum, A., Hammarling, S., McKenney,
       A., Sorensen, D., LAPACK Users' Guide, 3rd ed., SIAM, Philadelphia,
       1999. https://doi.org/10.1137/1.9780898719604

    Examples
    --------
    The two lowest natural frequencies, in rad/s:

    >>> import numpy as np
    >>> from scipy.sparse import diags
    >>> from structsolve import freq
    >>> K = diags([4., 9., 16., 25.]).tocsr()
    >>> M = diags([1., 1., 1., 1.]).tocsr()
    >>> lambda2, eigvecs = freq(K, M, num_eigvalues=2, symmetric=True,
    ...                         silent=True)
    >>> np.sqrt(-lambda2)
    array([2., 3.])

    """
    _check_linear_solver(linear_solver)
    msg('Running frequency analysis...', silent=silent)

    msg('Eigenvalue solver... ', level=2, silent=silent)

    k = min(num_eigvalues, M.shape[0]-2)
    size = M.shape[0]
    if skip_null_cols:
        used_cols = None
        Keff, Meff = K, M
    else:
        Keff, Meff, used_cols = remove_null_cols(K, M, silent=silent,
                level=3)
    if symmetric is None:
        symmetric = is_symmetric(Keff) and is_symmetric(Meff)
    if sparse_solver:
        Keff = csr_matrix(Keff)
        Meff = csr_matrix(Meff)
        if symmetric:
            sigma = _estimate_sigma(Keff, Meff, linear_solver)
            strategies = [('eigsh() in shift-invert mode (sigma={0})'.format(
                           sigma), lambda: _eigsh_freq(Keff, Meff, k, tol,
                                                       sigma, linear_solver))]
            dense = ('eigh()', lambda: _dense_freq(Keff, Meff, k, sigma,
                                                   True))
        else:
            sigma = _estimate_sigma(Keff, Meff)
            #NOTE Looking for better performance with symmetric matrices, I
            #     tried using sparseutils.sparse.is_symmetric and eigsh, but
            #     it seems not to improve speed (I did not try passing only
            #     half of the sparse matrices to the solver)
            strategies = [('eigs() in shift-invert mode (sigma={0})'.format(
                           sigma), lambda: _eigs_freq(Keff, Meff, k, tol,
                                                      sigma))]
            dense = ('eig()', lambda: _dense_freq(Keff, Meff, k, sigma,
                                                  False))
        (lambda2, peigvecs), _ = _sparse_first('Frequency analysis',
                strategies, lambda l2, v: _check_freq(Keff, Meff, l2, v,
                    check_rtol, symmetric, linear_solver=linear_solver),
                dense, Keff.shape[0],
                max_dense_size, silent,
                hint=' Try sparse_solver=False or a larger max_dense_size.')
    else:
        if symmetric:
            msg('eigh() solver...', level=3, silent=silent)
            Kd = (Keff.toarray() if hasattr(Keff, 'toarray')
                  else np.asarray(Keff))
            Md = (Meff.toarray() if hasattr(Meff, 'toarray')
                  else np.asarray(Meff))
            sigma = _estimate_sigma(csr_matrix(Kd), csr_matrix(Md),
                                    linear_solver)
            w2, peigvecs = _dense_eigh(Kd, Md, sigma)
            lambda2 = -w2
        else:
            if isinstance(Meff, scipy.sparse.spmatrix):
                Meff = Meff.toarray()
            else:
                Meff = np.asarray(Meff)
            if isinstance(Keff, scipy.sparse.spmatrix):
                Keff = Keff.toarray()
            else:
                Keff = np.asarray(Keff)

            # for effiency reasons, solving:
            #    [M]{u} = (-1/lambda2)[K]{u}
            #    [M]{u} = eigval [K]{u}
            #NOTE complex matrices to call zggev instead of dggev, which
            #     crashes the process for some matrices with Intel MKL
            #     2025.0.0, see linear_buckling._eig_complex
            msg('eig() solver...', level=3, silent=silent)
            eigvals, peigvecs = _eig_complex(a=Meff, b=Keff)
            lambda2 = -1./eigvals

        if check_rtol is not None and peigvecs.shape[1]:
            failed, res, _ = _failed_residuals(Keff, Meff, lambda2, peigvecs,
                                               check_rtol)
            if np.any(failed):
                warn('freq: relative residual {0:.1e} of an eigenpair is '
                     'larger than check_rtol={1:.1e}'.format(
                         res[failed].max(), check_rtol), level=2,
                     silent=silent)
            if symmetric and np.any(failed[:k]):
                # the k lowest finite eigenpairs, with the inertia
                finite = np.flatnonzero(np.isfinite(lambda2))[:k]
                lowest, res, _ = _failed_residuals(Keff, Meff,
                        lambda2[finite], peigvecs[:, finite], check_rtol,
                        definite=True, linear_solver=linear_solver)
                if np.any(lowest):
                    warnings.warn('freq: {0} of the {1} lowest eigenpairs of '
                        'eigh() fail the residual check, relative residual '
                        'up to {2:.1e} > check_rtol={3:.1e}, and are not '
                        'confirmed by the inertia, the lowest with '
                        'omega**2={4:.6g}; check them with the sparse '
                        'solver'.format(int(np.count_nonzero(lowest)),
                            finite.size, res[lowest].max(), check_rtol,
                            -lambda2[finite][lowest].max()),
                        UserWarning, stacklevel=2)

    if used_cols is not None:
        eigvecs = np.zeros((size, peigvecs.shape[1]), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs
    else:
        eigvecs = peigvecs

    msg('finished!', level=3, silent=silent)

    if sort:
        #NOTE complex square root, a positive real lambda2 (divergence) has
        #     an imaginary natural frequency and is filtered out
        omegan2 = -np.asarray(lambda2, dtype=complex)
        omegan = np.sqrt(omegan2)
        #NOTE np.lexsort: the last key is the primary one. The exact
        #     omega_n, then the exact omega_n**2, break the ties of the
        #     rounded keys, e.g. a double eigenvalue returned with a
        #     round-off difference in an order that depends on the LAPACK
        #     build. omega_n**2 = 100 and 100 minus one unit in the last
        #     place have the same omega_n = 10.0 after the square root
        sort_ind = np.lexsort((omegan2.imag, omegan2.real,
                               omegan.imag, omegan.real,
                               np.round(omegan.imag, 1),
                               np.round(omegan.real, 1)))
        omegan = omegan[sort_ind]
        higher_zero = omegan.real > 1e-6
        # the eigenvalues are sorted and filtered with their eigenvectors
        lambda2 = lambda2[sort_ind][higher_zero]
        eigvecs = eigvecs[:, sort_ind][:, higher_zero]

    msg('finished!', level=2, silent=silent)

    msg('first {0} eigenvalues:'.format(num_eigvalues_print), level=1,
        silent=silent)
    for lambda2i in lambda2[:num_eigvalues_print]:
        #NOTE complex for non-conservative systems, e.g. beyond flutter
        omegan = np.sqrt(-complex(lambda2i))
        if np.iscomplexobj(lambda2) and abs(complex(lambda2i).imag) > 0:
            msg('lambda**2: {0:.5g}, natural frequency: {1:.5g} rad/s'.format(
                complex(lambda2i), omegan), level=2, silent=silent)
        else:
            msg('lambda**2: %1.5f, natural frequency: %1.5f rad/s'
                % (np.real(lambda2i), omegan.real), level=2, silent=silent)

    return lambda2, eigvecs
