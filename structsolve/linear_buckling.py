import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, splu
from scipy.linalg import eigh, LinAlgError

from .logger import msg, warn
from .sparseutils import remove_null_cols


def _estimate_sigma(K, KG, safety=10., max_iter=50, rel_tol=1e-3):
    r"""Shift of the Cayley mode used by :func:`lb`

    With ``sigma > 0``, ``eigsh(A=KG, M=K, sigma=sigma, mode='cayley',
    which='SM')`` returns the eigenvalues `\mu` of ``KG u = mu K u`` with the
    smallest `|(\mu + \sigma)/(\mu - \sigma)|`. This ordering selects the most
    negative `\mu`, i.e. the lowest positive load multipliers `-1/\mu`, only
    when `\sigma` is larger than `|\mu|` of these critical eigenvalues.
    Otherwise the eigenvalues closest to `-\sigma` are returned.

    The largest `|\mu|` is estimated with power iterations on
    `[K]^{-1}[K_G]`, whose norm ratio in the `[K]`-norm increases towards the
    largest `|\mu|`, and multiplied by ``safety``. The default shift ``1.`` is
    returned when ``K`` is singular, not positive definite, or the linear
    solution is inaccurate.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lu = splu(csc_matrix(K))
        x = np.random.RandomState(42).randn(K.shape[0])
        Kx = K @ x
        mu_max = 0.
        for i in range(max_iter):
            xKx = x @ Kx
            if not np.isfinite(xKx) or xKx <= 0:
                return 1.
            x = x / np.sqrt(xKx)
            rhs = KG @ x
            y = lu.solve(rhs)
            Ky = K @ y
            if i == 0:
                residual = np.linalg.norm(Ky - rhs)
                if not residual <= 1e-6 * np.linalg.norm(rhs):
                    return 1.
            # ||y||_K / ||x||_K, with ||x||_K = 1
            ratio = np.sqrt(abs(y @ Ky))
            if not np.isfinite(ratio) or ratio == 0:
                return 1.
            converged = i > 0 and ratio - mu_max <= rel_tol * ratio
            mu_max = max(mu_max, ratio)
            if converged:
                break
            x, Kx = y, Ky
        return safety * mu_max
    except Exception:
        return 1.


def _is_positive_definite(A):
    """Tells if the symmetric sparse matrix ``A`` is positive definite

    ``A`` is factorized with a symmetric permutation and without row
    interchanges. The Gaussian elimination of a positive definite matrix
    never finds a zero or negative pivot, and by Sylvester's law of inertia
    any negative pivot means a negative eigenvalue of ``A``.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lu = splu(csc_matrix(A), permc_spec='MMD_AT_PLUS_A',
                      diag_pivot_thresh=0., options=dict(SymmetricMode=True))
    except RuntimeError:
        # exactly singular
        return False
    if not np.array_equal(lu.perm_r, lu.perm_c):
        # a zero pivot forced a row interchange
        return False
    return bool(np.all(lu.U.diagonal() > 0))


def _check_eigenpairs(K, KG, eigvals, eigvecs, rtol, min_rel_gap=1e-4):
    r"""Verify the eigenpairs of `([K] + \lambda [K_G])\{u\} = \{0\}`

    - The relative residual
      `||K u + \lambda K_G u|| / (||K u|| + |\lambda| ||K_G u||)` of every
      eigenpair with a finite `\lambda` must not be larger than ``rtol``.
    - No load multiplier is missing below the lowest positive `\lambda_1`
      of ``eigvals``: when ``K`` is positive definite, `[K] + s [K_G]` with
      `s = \lambda_1 (1 - min\_rel\_gap)` must also be positive definite.

    Returns
    -------
    error : str or None
        Description of the failed check, ``None`` when all checks passed.

    """
    finite = np.isfinite(eigvals)
    if not np.any(finite):
        return 'no finite load multiplier was found'
    lam = eigvals[finite]
    u = eigvecs[:, finite]
    Ku = K @ u
    KGu = KG @ u
    num = np.linalg.norm(Ku + KGu*lam, axis=0)
    den = np.linalg.norm(Ku, axis=0) + np.abs(lam)*np.linalg.norm(KGu, axis=0)
    residual = np.full(lam.shape, np.inf)
    np.divide(num, den, out=residual, where=den > 0)
    if not np.all(residual <= rtol):
        i = np.argmax(np.nan_to_num(residual, nan=np.inf))
        return ('relative residual {0:.2e} > {1:.1e} for the load multiplier '
                '{2}'.format(residual[i], rtol, lam[i]))
    positive = lam[lam > 0]
    if positive.size and _is_positive_definite(K):
        s = positive.min()*(1 - min_rel_gap)
        if not _is_positive_definite(K + s*KG):
            return ('at least one load multiplier lower than {0} is missing'
                    .format(positive.min()))
    return None


def _sort_eigenpairs(mu, eigvecs):
    r"""Sort by increasing `\mu`, i.e. the positive `\lambda = -1/\mu` first
    in increasing order, as returned by :func:`scipy.linalg.eigh`"""
    order = np.argsort(mu, kind='stable')
    return mu[order], eigvecs[:, order]


def _eigh_condensed(K, KG, k):
    r"""Dense solution after condensing out the dofs where ``KG`` is null

    With the dofs split in `b`, where `[K_G]` has non-null rows, and `a`,
    where it does not, the static condensation of the `a` dofs gives:

    .. math::

        ([K_{bb}] - [K_{ba}][K_{aa}]^{-1}[K_{ab}]
         + \lambda [K_{G_{bb}}])\{u_b\} = \{0\}, \qquad
        \{u_a\} = -[K_{aa}]^{-1}[K_{ab}]\{u_b\}

    solved with :func:`scipy.linalg.eigh`, which requires a positive
    definite Schur complement. Returns the ``k`` lowest eigenvalues `\mu` of
    `K_G u = \mu K u` and the corresponding eigenvectors.

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lu = splu(csc_matrix(K[a, :][:, a]))
        # solving blocks of right-hand sides with at most ~1e7 entries
        chunk = max(1, int(1e7) // a.size)
        for j0 in range(0, b.size, chunk):
            j1 = min(j0 + chunk, b.size)
            X = lu.solve(Kab[:, j0:j1].toarray())
            S[:, j0:j1] -= Kba @ X
    mu, ub = eigh(a=0.5*(KGbb + KGbb.T), b=0.5*(S + S.T))
    k = min(k, b.size)
    ub = ub[:, :k]
    eigvecs = np.zeros((K.shape[0], k), dtype=ub.dtype)
    eigvecs[b, :] = ub
    if a.size:
        eigvecs[a, :] = -lu.solve(np.asarray(Kab @ ub))
    return mu[:k], eigvecs


def _eigsh_cayley(K, KG, k, tol, silent):
    """Solution with :func:`scipy.sparse.linalg.eigsh` in Cayley mode"""
    sigma = _estimate_sigma(K, KG)
    msg('sigma={0}'.format(sigma), level=4, silent=silent)
    k = min(k, K.shape[0] - 2)
    mu, eigvecs = eigsh(A=KG, k=k, which='SM', M=K, tol=tol, sigma=sigma,
                        mode='cayley')
    return _sort_eigenpairs(mu, eigvecs)


def lb(K, KG, tol=0, sparse_solver=True, silent=False,
       num_eigvalues=25, num_eigvalues_print=5,
       skip_null_cols=False, max_dense_size=2000, check_rtol=1e-3):
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
        A float tolerance passed to :func:`scipy.sparse.linalg.eigsh`.
    sparse_solver : bool, optional
        With ``True``, when ``KG`` has at most ``max_dense_size`` non-null
        rows, the other dofs (typically the in-plane dofs) are condensed out
        with a sparse factorization of ``K`` and the condensed problem is
        solved with :func:`scipy.linalg.eigh`. Otherwise, or when this
        solution fails, :func:`scipy.sparse.linalg.eigsh` solves the full
        problem in Cayley mode, with a shift estimated from the matrices.
        The eigenpairs are verified as described in the Notes, and
        ``num_eigvalues`` load multipliers are returned.
        With ``False``, :func:`scipy.linalg.eigh` solves the full problem,
        returning all load multipliers, and requires a positive definite
        ``K``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    num_eigvalues : int, optional
        Number of load multipliers calculated with the sparse solver, and
        number of returned eigenvectors.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.
    skip_null_cols : bool, optional
        If True, skip the removal of null columns from the matrices.
        Use only when K is known to be non-singular.
    max_dense_size : int, optional
        Maximum number of non-null rows of ``KG`` for which the sparse solver
        solves the condensed problem with :func:`scipy.linalg.eigh`. Use
        ``0`` to always use :func:`scipy.sparse.linalg.eigsh`.
    check_rtol : float, optional
        Maximum relative residual of the eigenpairs of the sparse solver,
        see the Notes.

    Returns
    -------
    eigvals : ndarray
        The load multipliers `\lambda`, calculated as ``-1/eigval`` from the
        eigenvalues ``eigval`` of ``KG u = eigval K u``. The positive load
        multipliers come first in increasing order, followed by the negative
        ones, for both solvers.
    eigvecs : ndarray
        The `i^{th}` eigenvector is ``eigvecs[:, i]``, with the size of the
        original matrices. Only ``num_eigvalues`` eigenvectors are returned.

    Raises
    ------
    RuntimeError
        When no eigenpairs of the sparse solver pass the verification.

    Notes
    -----
    The eigenpairs of the sparse solver are verified as follows:

    - the relative residual
      `||K u + \lambda K_G u|| / (||K u|| + |\lambda| ||K_G u||)` of each
      eigenpair must not be larger than ``check_rtol``;
    - when ``K`` is positive definite, `[K] + s [K_G]` must also be positive
      definite for `s` slightly lower than the lowest positive load
      multiplier found, i.e. no lower load multiplier was missed. The
      factorization is done with SuperLU, without row interchanges.

    When the condensed dense solution fails the verification,
    :func:`scipy.sparse.linalg.eigsh` is tried, and a ``RuntimeError`` is
    raised when it also fails.

    ARPACK, used by :func:`scipy.sparse.linalg.eigsh`, returns wrong
    eigenpairs at random, without any error, when linked against Intel MKL
    2024.2.0 to 2025.0.0, e.g. in some Anaconda builds of SciPy: the
    ``dsteqr`` routine of these MKL versions returns wrong eigenvectors for
    matrices larger than 32 x 32. MKL 2025.0.1 or newer is not affected.

    """
    msg('Running linear buckling analysis...', silent=silent)

    msg('Eigenvalue solver... ', level=2, silent=silent)

    size = KG.shape[0]
    if skip_null_cols:
        used_cols = None
    else:
        K, KG, used_cols = remove_null_cols(K, KG, silent=silent)
    if sparse_solver:
        K = csr_matrix(K)
        KG = csr_matrix(KG)
        num_active = int(np.count_nonzero(abs(KG).sum(axis=1)))
        solvers = []
        if num_active <= max_dense_size:
            solvers.append(('eigh() with condensed dofs',
                            lambda: _eigh_condensed(K, KG, num_eigvalues)))
        solvers.append(('eigsh()',
                        lambda: _eigsh_cayley(K, KG, num_eigvalues, tol,
                                              silent)))
        errors = []
        for name, solver in solvers:
            msg('{0} solver ({1} of {2} rows of KG are non-null)...'.format(
                name, num_active, K.shape[0]), level=3, silent=silent)
            try:
                eigvals, peigvecs = solver()
                with np.errstate(divide='ignore'):
                    error = _check_eigenpairs(K, KG, -1./eigvals, peigvecs,
                                              check_rtol)
            except (LinAlgError, RuntimeError, ValueError) as e:
                error = '{0}: {1}'.format(type(e).__name__, e)
            if error is None:
                msg('finished!', level=3, silent=silent)
                break
            warn('{0} solver failed: {1}'.format(name, error), level=3,
                 silent=silent)
            errors.append('{0} solver: {1}'.format(name, error))
        else:
            raise RuntimeError(
                'Linear buckling analysis failed, no eigenvalue solver passed '
                'the verification ({0}). When ARPACK is linked against Intel '
                'MKL 2024.2.0 to 2025.0.0, update MKL to 2025.0.1 or newer. '
                'Otherwise, try sparse_solver=False or a larger '
                'max_dense_size.'.format('; '.join(errors)))

    else:
        K = K.toarray()
        KG = KG.toarray()
        msg('eigh() solver...', level=3, silent=silent)
        eigvals, peigvecs = eigh(a=KG, b=K)
        msg('finished!', level=3, silent=silent)

    num_eigvecs = min(num_eigvalues, peigvecs.shape[1])
    if used_cols is not None:
        eigvecs = np.zeros((size, num_eigvecs), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs[:, :num_eigvecs]
    else:
        eigvecs = peigvecs[:, :num_eigvecs]

    with np.errstate(divide='ignore'):
        eigvals = -1./eigvals

    msg('finished!', level=2, silent=silent)

    msg('first {0} eigenvalues:'.format(num_eigvalues_print), level=1,
        silent=silent)

    for eig in eigvals[:num_eigvalues_print]:
        msg('{0}'.format(eig), level=2, silent=silent)

    return eigvals, eigvecs
