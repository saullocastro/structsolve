import warnings

import numpy as np
from scipy.sparse.linalg import eigsh, spsolve
from scipy.linalg import eigh

from .logger import msg, warn
from .sparseutils import remove_null_cols


def _estimate_sigma(K, KG):
    try:
        rhs = KG @ np.random.RandomState(42).randn(K.shape[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            y = spsolve(K, rhs)
        if any(issubclass(w.category, (RuntimeWarning, Warning))
               and "singular" in str(w.message).lower() for w in caught):
            return 1.
        residual = np.linalg.norm(K @ y - rhs)
        if residual > 1e-6 * np.linalg.norm(rhs):
            return 1.
        sigma = abs((y @ KG @ y) / (y @ K @ y))
        if not np.isfinite(sigma) or sigma == 0:
            return 1.
        return sigma
    except Exception:
        return 1.

def lb(K, KG, tol=0, sparse_solver=True, silent=False,
       num_eigvalues=25, num_eigvalues_print=5,
       skip_null_cols=False):
    r"""Linear buckling analysis

    Calculates the eigenvalues `\lambda` and eigenvectors `\{u\}` of the
    eigenvalue problem:

    .. math::

        ([K] + \lambda [K_G])\{u\} = \{0\}

    where `\lambda` is the load multiplier of the load state that generated
    `[K_G]`. It can also be used for more general eigenvalue analyses, e.g.
    if ``K`` is the tangent stiffness matrix of a given load state. The null
    rows and columns of ``K`` are removed from both matrices before solving
    the eigenvalue problem, unless ``skip_null_cols=True``.

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
        A float tolerance passed to the eigenvalue solver.
    sparse_solver : bool, optional
        Tells if solver :func:`scipy.sparse.linalg.eigsh` (``True``) or
        :func:`scipy.linalg.eigh` (``False``) should be used. The sparse
        solver uses the Cayley mode with a shift estimated from the matrices
        and calculates ``num_eigvalues`` eigenvalues. The dense solver
        calculates all eigenvalues and requires a positive definite ``K``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    num_eigvalues : int, optional
        Number of calculated eigenvalues with the sparse solver, limited to
        the size of ``KG`` minus 2, and number of returned eigenvectors.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.
    skip_null_cols : bool, optional
        If True, skip the removal of null columns from the matrices.
        Use only when K is known to be non-singular.

    Returns
    -------
    eigvals : ndarray
        The load multipliers `\lambda`, calculated as ``-1/eigval`` from the
        eigenvalues ``eigval`` of ``KG u = eigval K u``. The dense solver
        returns all load multipliers, the positive ones first in increasing
        order.
    eigvecs : ndarray
        The `i^{th}` eigenvector is ``eigvecs[:, i]``, with the size of the
        original matrices. Only ``num_eigvalues`` eigenvectors are returned.

    """
    msg('Running linear buckling analysis...', silent=silent)

    msg('Eigenvalue solver... ', level=2, silent=silent)

    k = min(num_eigvalues, KG.shape[0]-2)
    size = KG.shape[0]
    if skip_null_cols:
        used_cols = None
    else:
        K, KG, used_cols = remove_null_cols(K, KG, silent=silent)
    if sparse_solver:
        mode = 'cayley'
        sigma = _estimate_sigma(K, KG)
        msg('eigsh() solver (sigma={0})...'.format(sigma), level=3, silent=silent)
        eigvals, peigvecs = eigsh(A=KG, k=k,
                which='SM', M=K, tol=tol, sigma=sigma, mode=mode)
        msg('finished!', level=3, silent=silent)

    else:
        K = K.toarray()
        KG = KG.toarray()
        msg('eigh() solver...', level=3, silent=silent)
        eigvals, peigvecs = eigh(a=KG, b=K)
        msg('finished!', level=3, silent=silent)

    if used_cols is not None:
        eigvecs = np.zeros((size, num_eigvalues), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs[:, :num_eigvalues]
    else:
        eigvecs = peigvecs[:, :num_eigvalues]

    eigvals = -1./eigvals

    eigvals = eigvals
    eigvecs = eigvecs

    msg('finished!', level=2, silent=silent)

    msg('first {0} eigenvalues:'.format(num_eigvalues_print), level=1,
        silent=silent)

    for eig in eigvals[:num_eigvalues_print]:
        msg('{0}'.format(eig), level=2, silent=silent)

    return eigvals, eigvecs
