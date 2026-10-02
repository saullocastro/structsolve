import warnings

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh, spsolve
from scipy.linalg import eig, eigh

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


def freq(K, M, tol=0, sparse_solver=True,
        silent=False, sort=True, num_eigvalues=25,
        num_eigvalues_print=5, skip_null_cols=False, symmetric=False,
        check_rtol=1e-6):
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
        A tolerance value passed to :func:`scipy.sparse.linalg.eigs`.
    sparse_solver : bool, optional
        Tells if solver :func:`scipy.sparse.linalg.eigs` (``True``) or
        :func:`scipy.linalg.eig` (``False``) should be used. The sparse
        solver uses the shift-invert mode, with a negative shift estimated
        from the matrices, and calculates the ``num_eigvalues`` eigenvalues
        closest to the shift. The dense solver calculates all eigenvalues.

        .. note:: The sparse solver is faster, but it was verified to become
                  unstable for some cases, where ``sparse_solver=False`` is
                  recommended.

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
        A warning is issued when the relative residual of an eigenpair,
        `||K u + \lambda^2 M u||/(||K u|| + |\lambda^2| ||M u||)`, is larger
        than ``check_rtol``. ``None`` skips the check.

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

    Notes
    -----
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
        from .linear_buckling import is_symmetric
        symmetric = is_symmetric(Keff) and is_symmetric(Meff)
    if symmetric and sparse_solver:
        sigma = _estimate_sigma(Keff, Meff)
        msg('eigsh() solver (sigma={0})...'.format(sigma), level=3,
            silent=silent)
        eigvals, peigvecs = eigsh(A=csr_matrix(Keff), M=csr_matrix(Meff),
                                  k=k, which='LM', tol=tol, sigma=sigma)
        lambda2 = -eigvals
    elif symmetric:
        msg('eigh() solver...', level=3, silent=silent)
        Kd = Keff.toarray() if hasattr(Keff, 'toarray') else np.asarray(Keff)
        Md = Meff.toarray() if hasattr(Meff, 'toarray') else np.asarray(Meff)
        eigvals, peigvecs = eigh(a=Kd, b=Md)
        lambda2 = -eigvals
    elif sparse_solver:
        #NOTE Looking for better performance with symmetric matrices, I tried
        #     using sparseutils.sparse.is_symmetric and eigsh, but it seems not
        #     to improve speed (I did not try passing only half of the sparse
        #     matrices to the solver)
        sigma = _estimate_sigma(Keff, Meff)
        msg('eigs() solver (sigma={0})...'.format(sigma), level=3, silent=silent)
        eigvals, peigvecs = eigs(A=Keff, M=Meff, k=k, which='LM', tol=tol,
                                 sigma=sigma)
        #NOTE eigs solves: [K] {u} = eigval [M] {u}
        #     therefore we must correct he sign of lambda^2 here:
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

        #TODO did not try using eigh when input is symmetric to see if there
        #     will be speed improvements
        # for effiency reasons, solving:
        #    [M]{u} = (-1/lambda2)[K]{u}
        #    [M]{u} = eigval [K]{u}
        msg('eig() solver...', level=3, silent=silent)
        eigvals, peigvecs = eig(a=Meff, b=Keff)
        lambda2 = -1./eigvals

    if check_rtol is not None and peigvecs.shape[1]:
        res = _residuals(Keff, Meff, lambda2, peigvecs)
        if res.size and res.max() > check_rtol:
            warn('freq: relative residual {0:.1e} of an eigenpair is larger '
                 'than check_rtol={1:.1e}'.format(res.max(), check_rtol),
                 level=2, silent=silent)

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
