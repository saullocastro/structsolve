import warnings

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from scipy.linalg import eigh, LinAlgError

from .arpackutils import capped_eigsh, release_memory, start_vector
from .linear_buckling import (_eig_complex, _negative_pivots, _sparse_first,
                              is_symmetric)
from .logger import msg, warn
from .sparseutils import remove_null_cols


def _estimate_sigma(K, M):
    try:
        rhs = M @ np.random.RandomState(42).randn(K.shape[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            y = spsolve(K, rhs)
        if any(issubclass(w.category, (RuntimeWarning, Warning))
               and "singular" in str(w.message).lower() for w in caught):
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
    ||M u||)` of the finite eigenpairs"""
    finite = np.isfinite(lambda2)
    u = eigvecs[:, finite]
    lam = lambda2[finite]
    Ku = K @ u
    Mu = M @ u
    num = np.linalg.norm(Ku + Mu*lam, axis=0)
    den = np.linalg.norm(Ku, axis=0) + np.abs(lam)*np.linalg.norm(Mu, axis=0)
    out = np.zeros(num.shape)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _check_freq(K, M, lambda2, eigvecs, check_rtol, symmetric,
                min_rel_gap=1e-4):
    r"""Verify the eigenpairs of `([K] + \lambda^2 [M])\{u\} = \{0\}`

    - The relative residuals of :func:`_residuals` must not be larger than
      ``check_rtol``, unless it is ``None``.
    - With ``symmetric=True``, no eigenvalue `\omega^2 = -\lambda^2` is
      missing below the highest one returned, `\omega^2_n`: the number of
      negative pivots of `[K] - s [M]`, with
      `s = \omega^2_n - min\_rel\_gap |\omega^2_n|`, i.e. the number of
      eigenvalues lower than `s` by Sylvester's law of inertia for a
      positive definite ``M``, must be the number of returned eigenvalues
      lower than `s`. The check is skipped when the count is unknown, see
      :func:`.linear_buckling._negative_pivots`.

    Returns
    -------
    error : str or None
        Description of the failed check, ``None`` when all checks passed.

    """
    lambda2 = np.asarray(lambda2)
    if not eigvecs.shape[1]:
        return 'no eigenpair was found'
    if check_rtol is not None:
        res = _residuals(K, M, lambda2, eigvecs)
        if not np.all(res <= check_rtol):
            i = np.argmax(np.nan_to_num(res, nan=np.inf))
            return ('relative residual {0:.2e} > {1:.1e} for the eigenvalue '
                    'lambda**2={2}'.format(res[i], check_rtol,
                                           lambda2[np.isfinite(lambda2)][i]))
    if symmetric:
        w2 = -np.real(lambda2[np.isfinite(lambda2)])
        if not w2.size:
            return 'no finite eigenvalue was found'
        s = w2.max() - min_rel_gap*abs(w2.max())
        expected = int(np.count_nonzero(w2 < s))
        count = _negative_pivots(K - s*M) if s != 0 else None
        if count is not None and count != expected:
            return ('{0} eigenvalues omega**2 lower than {1} exist, {2} were '
                    'found'.format(count, s, expected))
    return None


def _eigsh_freq(K, M, k, tol, sigma):
    """:func:`.arpackutils.capped_eigsh` in shift-invert mode"""
    eigvals, eigvecs = capped_eigsh(A=K, k=k, M=M, sigma=sigma, which='LM',
                                    mode='normal', tol=tol)
    return -eigvals, eigvecs


def _eigs_freq(K, M, k, tol, sigma):
    """:func:`scipy.sparse.linalg.eigs` in shift-invert mode"""
    eigvals, eigvecs = eigs(A=K, M=M, k=k, which='LM', tol=tol, sigma=sigma,
                            v0=start_vector(K.shape[0]))
    release_memory()
    #NOTE eigs solves: [K] {u} = eigval [M] {u}
    #     therefore we must correct he sign of lambda^2 here:
    return -eigvals, eigvecs


def _dense_freq(K, M, k, sigma, symmetric):
    r"""Dense fallback of the sparse solvers of :func:`freq`

    With ``symmetric=True``, :func:`scipy.linalg.eigh` and the ``k`` lowest
    `\omega^2`, with eigenvectors normalized with respect to ``M``.
    Otherwise :func:`scipy.linalg.eig`, through the complex LAPACK driver
    ``zggev``, see :func:`.linear_buckling._eig_complex`, and the ``k``
    finite `\omega^2` nearest ``sigma``, as the sparse solver, by increasing
    distance.

    """
    Kd = csr_matrix(K).toarray()
    Md = csr_matrix(M).toarray()
    if symmetric:
        #NOTE [M]{u} = (1/omega**2) [K]{u} gives the lowest frequencies with
        #     relative residuals of ~1e-10, against ~1e-6 for
        #     eigh(a=K, b=M) for stiff models, but requires a positive
        #     definite K
        try:
            eigvals, eigvecs = eigh(a=Md, b=Kd)
            positive = np.flatnonzero(eigvals > 0)[::-1][:k]
            w2 = 1./eigvals[positive]
            eigvecs = eigvecs[:, positive]
            eigvecs /= np.sqrt(np.einsum('ij,ij->j', eigvecs, Md @ eigvecs))
        except LinAlgError:
            eigvals, eigvecs = eigh(a=Kd, b=Md)
            w2 = eigvals[:k]
            eigvecs = eigvecs[:, :k]
        return -w2, eigvecs
    eigvals, eigvecs = _eig_complex(a=Md, b=Kd)
    with np.errstate(divide='ignore', invalid='ignore'):
        w2 = 1./eigvals
    finite = np.flatnonzero(np.isfinite(w2))
    order = finite[np.argsort(np.abs(w2[finite] - sigma), kind='stable')[:k]]
    return -w2[order], eigvecs[:, order]


def freq(K, M, tol=0, sparse_solver=True,
        silent=False, sort=True, num_eigvalues=25,
        num_eigvalues_print=5, skip_null_cols=False, symmetric=False,
        check_rtol=1e-6, max_dense_size=2000):
    r"""Frequency analysis

    Calculates the eigenvalues `\lambda^2` and eigenvectors `\{u\}` of the
    free-vibration eigenvalue problem:

    .. math::

        ([K] + \lambda^2 [M])\{u\} = \{0\}

    where `\lambda^2 = -\omega_n^2` and `\omega_n` is the natural frequency
    in rad/s. The null rows and columns of ``K`` are removed from both
    matrices before solving the eigenvalue problem, unless
    ``skip_null_cols=True``.

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
        last resort, see the Notes. With ``False``, the full problem is
        explicitly solved with a dense solver, :func:`scipy.linalg.eig`, or
        :func:`scipy.linalg.eigh` with ``symmetric=True``, without trying the
        sparse solver, which calculates all eigenvalues, e.g. for the
        kinetic criterion of stability, and is a reference for validations.

    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    sort : bool, optional
        Sort the eigenvectors by increasing natural frequency, keeping only
        those with a natural frequency larger than ``1e-6`` rad/s. The
        returned eigenvalues are not affected by this option.
    num_eigvalues : int, optional
        Number of calculated eigenvalues with the sparse solver, limited to
        the size of ``M`` minus 2.
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
        `||K u + \lambda^2 M u||/(||K u|| + |\lambda^2| ||M u||)`. A larger
        residual of the sparse solver is a failed verification, see the
        Notes. For ``sparse_solver=False``, a warning is issued. ``None``
        skips the check.
    max_dense_size : int, optional
        Maximum size of the dense problem solved as a last resort when the
        sparse solver failed, see the Notes. Larger problems raise a
        ``RuntimeError`` instead, and ``0`` disables the dense fallback.

    Returns
    -------
    lambda2 : ndarray
        Complex array with the eigenvalues `\lambda^2 = -\omega_n^2`, in the
        order returned by the eigenvalue solver, real with
        ``symmetric=True``.
    eigvecs : ndarray
        The `i^{th}` eigenvector is ``eigvecs[:, i]``, with the size of the
        original matrices. The sparse solver returns eigenvectors normalized
        with respect to ``M``, the dense solver returns eigenvectors with
        unit Euclidean norm. If ``sort=True`` the columns are sorted and
        filtered as described above, in which case they correspond to the
        eigenvalues in ``lambda2`` only if the solver already returned them
        in the order of increasing natural frequency.

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

    Notes
    -----
    **Solver policy.** As in :func:`.lb`, with ``sparse_solver=True`` the
    sparse solver runs first for every problem size, and its eigenpairs are
    verified:

    - the relative residual of each eigenpair must not be larger than
      ``check_rtol``;
    - with ``symmetric=True``, no eigenvalue `\omega_n^2` lower than the
      highest one returned may be missing: the number of negative pivots of
      `[K] - s [M]`, for `s` slightly lower than this eigenvalue, is the
      number of eigenvalues below `s` (Sylvester's law of inertia).

    When the verification fails or the sparse solver raises an error, e.g.
    ARPACK did not converge, the dense solver is used as a last resort if
    the problem has at most ``max_dense_size`` dofs, with a
    :class:`.linear_buckling.DenseFallbackWarning` giving the reasons. It
    returns the ``num_eigvalues`` lowest eigenvalues with ``symmetric=True``,
    otherwise the ``num_eigvalues`` eigenvalues closest to the shift, as the
    sparse solver, and its eigenpairs are verified as well. Otherwise a
    ``RuntimeError`` is raised. Until version 0.5.3, the sparse solver
    only issued a warning when the residual check failed.

    :func:`scipy.sparse.linalg.eigsh` uses ARPACK's symmetric drivers, which
    return wrong eigenpairs when linked against Intel MKL 2024.2.0 to
    2025.0.0 and ``ncv > 32``, see :func:`.lb` and
    :mod:`structsolve.arpackutils`, where ``ncv`` is capped accordingly.

    Non-conservative systems, e.g. under follower loads, may lose stability
    by flutter, which the static criterion of :func:`.lb` does not detect:
    with ``K`` the tangent stiffness matrix at a load level, including the
    load stiffness, the eigenvalues `\lambda^2` of the general solvers are
    real and negative below the flutter load and become complex conjugate
    pairs above it, when two natural frequencies coalesce (the kinetic
    criterion). A real positive `\lambda^2` indicates divergence.

    """
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
        sigma = _estimate_sigma(Keff, Meff)
        if symmetric:
            strategies = [('eigsh() in shift-invert mode (sigma={0})'.format(
                           sigma), lambda: _eigsh_freq(Keff, Meff, k, tol,
                                                       sigma))]
            dense = ('eigh()', lambda: _dense_freq(Keff, Meff, k, sigma,
                                                   True))
        else:
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
                    check_rtol, symmetric), dense, Keff.shape[0],
                max_dense_size, silent,
                hint=' Try sparse_solver=False or a larger max_dense_size.')
    else:
        if symmetric:
            msg('eigh() solver...', level=3, silent=silent)
            Kd = (Keff.toarray() if hasattr(Keff, 'toarray')
                  else np.asarray(Keff))
            Md = (Meff.toarray() if hasattr(Meff, 'toarray')
                  else np.asarray(Meff))
            eigvals, peigvecs = eigh(a=Kd, b=Md)
            lambda2 = -eigvals
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
            res = _residuals(Keff, Meff, lambda2, peigvecs)
            if res.size and res.max() > check_rtol:
                warn('freq: relative residual {0:.1e} of an eigenpair is '
                     'larger than check_rtol={1:.1e}'.format(res.max(),
                         check_rtol), level=2, silent=silent)

    if used_cols is not None:
        eigvecs = np.zeros((size, peigvecs.shape[1]), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs
    else:
        eigvecs = peigvecs

    msg('finished!', level=3, silent=silent)

    if sort:
        omegan = np.sqrt(-lambda2)
        sort_ind = np.lexsort((np.round(omegan.imag, 1),
                               np.round(omegan.real, 1)))
        omegan = omegan[sort_ind]
        eigvecs = eigvecs[:, sort_ind]

        higher_zero = omegan.real > 1e-6

        omegan = omegan[higher_zero]
        eigvecs = eigvecs[:, higher_zero]

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
