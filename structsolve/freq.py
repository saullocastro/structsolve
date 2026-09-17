import warnings

import numpy as np
import scipy
from scipy.sparse.linalg import eigs, spsolve
from scipy.linalg import eig

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


def freq(K, M, tol=0, sparse_solver=True,
        silent=False, sort=True, num_eigvalues=25,
        num_eigvalues_print=5, skip_null_cols=False):
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

    Returns
    -------
    lambda2 : ndarray
        Complex array with the eigenvalues `\lambda^2 = -\omega_n^2`, in the
        order returned by the eigenvalue solver.
    eigvecs : ndarray
        The `i^{th}` eigenvector is ``eigvecs[:, i]``, with the size of the
        original matrices. The sparse solver returns eigenvectors normalized
        with respect to ``M``, the dense solver returns eigenvectors with
        unit Euclidean norm. If ``sort=True`` the columns are sorted and
        filtered as described above, in which case they correspond to the
        eigenvalues in ``lambda2`` only if the solver already returned them
        in the order of increasing natural frequency.

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
    if sparse_solver:
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
        msg('lambda**2: %1.5f, natural frequency: %1.5f rad/s' % (lambda2i, (-lambda2i)**0.5), level=2, silent=silent)

    return lambda2, eigvecs
