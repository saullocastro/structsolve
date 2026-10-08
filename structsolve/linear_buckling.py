import sys
import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, splu, LinearOperator
from scipy.linalg import eigh, eig, LinAlgError

from .arpackutils import capped_eigsh, release_memory, start_vector
from .logger import msg, warn
from .sparseutils import remove_null_cols


class DenseFallbackWarning(RuntimeWarning):
    """Issued by :func:`lb` and :func:`.freq` when the sparse eigenvalue
    solvers failed and the dense solver was used as a last resort"""


#NOTE eigenvalues per step of the shift search of unsymmetric problems,
#     at least: few eigenvalues give small discs and many steps
SHIFT_SEARCH_MIN_EIGVALUES = 6
#NOTE the reversed follower load cantilevers of the study, whose critical
#     load multiplier is 1e4 times the modulus of the multipliers near zero,
#     need up to 12 steps
SHIFT_SEARCH_MAX_STEPS = 40
SHIFT_SEARCH_MAX_GROWTH = 1e10
PARITY_REL_GAP = 1e-4


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


def _negative_pivots(A):
    """Number of negative eigenvalues of the symmetric sparse matrix ``A``

    ``A`` is factorized with a symmetric permutation and without row
    interchanges, i.e. `[P][A][P]^T = [L][D][L]^T` with the pivots `[D]` on
    the diagonal of `[U] = [D][L]^T`. By Sylvester's law of inertia, the
    number of negative pivots is the number of negative eigenvalues of
    ``A``.

    Returns
    -------
    num : int or None
        ``None`` when the count is unknown: ``A`` is exactly singular, or a
        zero pivot forced a row interchange.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                      check_inertia=True):
    r"""Verify the eigenpairs of `([K] + \lambda [K_G])\{u\} = \{0\}`

    The eigenpairs may be real or complex.

    - The relative residual
      `||K u + \lambda K_G u|| / (||K u|| + |\lambda| ||K_G u||)` of every
      eigenpair with a finite `\lambda` must not be larger than ``rtol``.
    - No load multiplier is missing below the highest positive `\lambda_n`
      of ``eigvals``: when ``K`` is positive definite, the number of
      negative pivots of `[K] + s [K_G]` with
      `s = \lambda_n (1 - min\_rel\_gap)`, i.e. the number of load
      multipliers in `(0, s)` by Sylvester's law of inertia, see
      :func:`_negative_pivots`, must be the number of positive ``eigvals``
      lower than `s`. When this count is unknown, the weaker check of the
      lowest positive `\lambda_1` is done: `[K] + s [K_G]` with
      `s = \lambda_1 (1 - min\_rel\_gap)` must be positive definite. Only for
      symmetric matrices, ``check_inertia=True``, the complex `\lambda` are
      ignored. Missing negative load multipliers are not detected.

    Returns
    -------
    error : str or None
        Description of the failed check, ``None`` when all checks passed.

    """
    eigvals = np.asarray(eigvals)
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
    if not check_inertia:
        return None
    lam = lam[np.isreal(lam)].real
    positive = lam[lam > 0]
    if positive.size and _is_positive_definite(K):
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


def is_symmetric(A, rtol=1e-10):
    r"""Tell whether ``||A - A^T|| <= rtol ||A||``, in the Frobenius norm

    A cheap check, of the order of the number of non-zero entries of ``A``,
    used by :func:`lb` to select the solvers of symmetric or unsymmetric
    matrices.

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
    r"""As :func:`_sort_eigenpairs` for the real `\mu`, with the complex
    ones at the end, by increasing `|\lambda| = 1/|\mu|`"""
    mu = np.asarray(mu)
    real = np.isreal(mu)
    ireal = np.flatnonzero(real)
    icplx = np.flatnonzero(~real)
    ireal = ireal[np.argsort(mu[ireal].real, kind='stable')]
    icplx = icplx[np.argsort(-np.abs(mu[icplx]), kind='stable')]
    order = np.concatenate((ireal, icplx))
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


def _eig_complex(a, b=None):
    r"""All eigenpairs of `[a]\{v\} = w [b]\{v\}` with :func:`scipy.linalg.eig`

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
    drivers are used.

    """
    if sys.platform == 'emscripten':
        return eig(a=a, b=b)
    a = np.asarray(a, dtype=complex)
    if b is not None:
        b = np.asarray(b, dtype=complex)
    return eig(a=a, b=b)


def _eig_condensed(K, KG, k):
    r"""As :func:`_eigh_condensed`, for unsymmetric matrices

    The dofs where both the row and the column of ``KG`` are null are
    condensed out with an LU factorization of ``K``, and the condensed
    problem is solved with :func:`scipy.linalg.eig`, through the complex
    LAPACK driver ``zggev``, see :func:`_eig_complex`. Neither converting to
    the standard problem `[S]^{-1}[K_{G_{bb}}]` (``dgeev``) nor limiting the
    threads of BLAS/LAPACK avoids the crash of ``dggev`` with Intel MKL
    2025.0.0. The ``k`` returned eigenpairs, see :func:`_general_eigenpairs`,
    are real when they are all real, and complex eigenvectors are
    back-substituted with the real factorization of `[K_{aa}]`.

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lu = splu(csc_matrix(K[a, :][:, a]))
        chunk = max(1, int(1e7) // a.size)
        for j0 in range(0, b.size, chunk):
            j1 = min(j0 + chunk, b.size)
            X = lu.solve(Kab[:, j0:j1].toarray())
            S[:, j0:j1] -= Kba @ X
    mu, ub = _eig_complex(a=KGbb, b=S)
    mu, ub = _general_eigenpairs(mu, ub, k)
    eigvecs = np.zeros((K.shape[0], mu.shape[0]), dtype=ub.dtype)
    eigvecs[b, :] = ub
    if a.size:
        eigvecs[a, :] = -_lu_solve(lu, np.asarray(Kab @ ub))
    return mu, eigvecs


def _splu(A):
    """SuperLU factorization of ``A`` with partial pivoting, raising a
    ``RuntimeError`` when ``A`` is exactly singular"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    """Sign of the determinant of a real matrix from its SuperLU
    factorization `[P_r][A][P_c] = [L][U]`, with a unit `[L]`"""
    d = lu.U.diagonal()
    return (int(np.prod(np.sign(d))) * _permutation_parity(lu.perm_r)
            * _permutation_parity(lu.perm_c))


def _eigs_shifted(K, KG, s, k, tol, lu):
    r"""The ``k`` load multipliers nearest the shift ``s``, for real
    unsymmetric matrices

    With `\lambda = s + \delta`, `([K] + s[K_G] + \delta [K_G])\{u\} = 0`
    gives the standard problem
    `([K] + s [K_G])^{-1} [K_G] \{u\} = \nu \{u\}`, `\nu = -1/(\lambda - s)`,
    whose ``k`` eigenvalues of largest `|\nu|` are found with
    :func:`scipy.sparse.linalg.eigs`, using the SuperLU factorization ``lu``
    of `[K] + s [K_G]`. The eigenvalues `|\nu| \le 10^{-12} \max |\nu|` are
    taken as null, i.e. infinite load multipliers, e.g. of the null rows of
    `[K_G]`, and are not returned.

    Returns
    -------
    mu, eigvecs : ndarray
        The eigenpairs, with `\mu = -1/\lambda = \nu/(1 - s \nu)`, sorted by
        :func:`_general_eigenpairs`.
    radius : float
        `\max |\lambda - s|` of the returned load multipliers. All the load
        multipliers in the disc `|\lambda - s| < radius` are returned.
        ``inf`` when a null `\nu` was returned, i.e. all the finite load
        multipliers were found.

    """
    n = K.shape[0]
    OP = LinearOperator((n, n), matvec=lambda x: _lu_solve(lu, KG @ x),
                        dtype=np.result_type(K.dtype, KG.dtype))
    nu, eigvecs = eigs(OP, k=k, which='LM', tol=tol, v0=start_vector(n))
    release_memory()
    null = np.abs(nu) <= 1e-12*np.abs(nu).max()
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
       shift is perturbed when `[K] + s [K_G]` is exactly singular.
    3. The search stops when the lowest positive real multiplier found,
       `\lambda_{cr}`, is not larger than `c`: no multiplier of `(0,
       \lambda_{cr})` was missed. It fails after
       :data:`SHIFT_SEARCH_MAX_STEPS` steps, or when the shift exceeds
       :data:`SHIFT_SEARCH_MAX_GROWTH` times `r_0`.
    4. Parity check: `\det([K] + 0.9999 \lambda_{cr} [K_G])` must have the
       sign of `\det [K]`, i.e. an even number, normally zero, of real
       multipliers in `(0, 0.9999 \lambda_{cr})`. The signs come from the
       SuperLU factorizations, see :func:`_det_sign`. This check cannot
       detect two missing real multipliers, nor a missing complex pair.

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


def _eigsh_cayley(K, KG, k, tol, silent, safety=10.):
    r"""Solution with :func:`scipy.sparse.linalg.eigsh` in Cayley mode

    The shift is estimated with :func:`_estimate_sigma` and the given
    ``safety`` factor. ARPACK's ``ncv`` is chosen by
    :func:`.arpackutils.capped_eigsh`, which avoids the Intel MKL bug.

    """
    sigma = _estimate_sigma(K, KG, safety=safety)
    msg('sigma={0}'.format(sigma), level=4, silent=silent)
    k = min(k, K.shape[0] - 2)
    mu, eigvecs = capped_eigsh(A=KG, k=k, M=K, sigma=sigma, which='SM',
                               mode='cayley', tol=tol)
    return _sort_eigenpairs(mu, eigvecs)


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
        A float tolerance passed to :func:`scipy.sparse.linalg.eigsh` and
        :func:`scipy.sparse.linalg.eigs`.
    sparse_solver : bool, optional
        With ``True``, the default, the sparse solvers always run first,
        whatever the size of the problem, and a dense solver is only used as
        a last resort, see the Notes. ``num_eigvalues`` load multipliers are
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
        number of returned eigenvectors.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.
    skip_null_cols : bool, optional
        If True, skip the removal of null columns from the matrices.
        Use only when K is known to be non-singular.
    max_dense_size : int, optional
        Maximum size of the dense problem solved as a last resort, when all
        the sparse solvers failed, i.e. the number of non-null rows of
        ``KG``, or of non-null rows or columns for unsymmetric matrices, see
        the Notes. Larger problems raise a ``RuntimeError`` instead, and
        ``0`` disables the dense fallback. Until version 0.5.3, it was the
        size below which the dense solver was tried first.
    check_rtol : float, optional
        Maximum relative residual of the eigenpairs of the sparse solver,
        see the Notes.
    symmetric : bool or None, optional
        Whether ``K`` and ``KG`` are symmetric. ``None`` checks it with
        :func:`is_symmetric`, ``True`` skips the check and uses the solvers
        of symmetric matrices, ``False`` forces the solvers of unsymmetric
        matrices, required e.g. when ``KG`` includes the load stiffness of
        follower loads, see the Notes.

    Returns
    -------
    eigvals : ndarray
        The load multipliers `\lambda`, calculated as ``-1/eigval`` from the
        eigenvalues ``eigval`` of ``KG u = eigval K u``. The positive load
        multipliers come first in increasing order, followed by the negative
        ones, for both solvers. For unsymmetric matrices the array is
        complex when some eigenvalues are complex, which come last.
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
    **Solver policy.** With ``sparse_solver=True`` the sparse solvers run
    first for every problem size, in the order given below, and each result
    must pass the verification. Only when all of them failed (verification,
    ARPACK error or no convergence, singular factorization, no positive real
    load multiplier...) a dense solver is used, if the dense problem has at
    most ``max_dense_size`` dofs, with a :class:`DenseFallbackWarning` that
    gives the reasons of the failures. Its result is verified as well. A
    ``RuntimeError`` listing every attempted solver is raised when the
    problem is too large for the dense fallback, or when the dense solution
    also failed. The dense fallback keeps structsolve robust where ARPACK is
    unreliable, normal runs never pay for a dense solution.

    **Symmetric matrices.** The sparse solvers are:

    1. :func:`scipy.sparse.linalg.eigsh` in Cayley mode, with the shift
       estimated from the matrices by :func:`_estimate_sigma`;
    2. the same with a 10 times larger shift.

    The dense fallback condenses out the dofs where ``KG`` has null rows,
    typically the in-plane dofs, with a sparse factorization of ``K``, and
    solves the condensed problem with :func:`scipy.linalg.eigh`, see
    :func:`_eigh_condensed`. The eigenpairs are verified as follows:

    - the relative residual
      `||K u + \lambda K_G u|| / (||K u|| + |\lambda| ||K_G u||)` of each
      eigenpair must not be larger than ``check_rtol``;
    - when ``K`` is positive definite, no load multiplier below the highest
      positive one returned may be missing: the number of negative pivots of
      `[K] + s [K_G]`, for `s` slightly below that multiplier, is the number
      of load multipliers in `(0, s)` (Sylvester's law of inertia). The
      factorization is done with SuperLU, without row interchanges. When a
      row interchange was needed and the count is unknown, `[K] + s [K_G]`
      must be positive definite for `s` slightly below the lowest positive
      load multiplier. See :func:`_check_eigenpairs`.

    **Unsymmetric matrices**, e.g. ``KG`` with the load stiffness of a
    follower pressure that is not conservative, may have complex
    eigenvalues. The sparse solver is a search of shifts along the positive
    real axis with :func:`scipy.sparse.linalg.eigs`, see
    :func:`_eigs_shift_search`, which finds the lowest positive real load
    multiplier even when many negative or complex multipliers have a
    smaller modulus, verifies the residuals, and checks the parity of the
    number of real multipliers below the critical one. The returned
    eigenpairs are the ``num_eigvalues`` load multipliers nearest the last
    shift of the search, which is zero unless the multipliers of smallest
    modulus did not include a positive real one, not necessarily the load
    multipliers of smallest modulus. The dense fallback,
    :func:`_eig_condensed`, keeps the dofs where either the row or the
    column of ``KG`` is not null, which for follower loads are almost all
    of them, and returns the ``num_eigvalues`` first load multipliers in the
    order described above; when none of them is positive and real, i.e. no
    divergence load exists, e.g. for a system that only loses stability by
    flutter, they are returned with a warning. The inertia check requires
    symmetric matrices and is skipped. Eigenvalues whose imaginary part is
    below ``1e-8`` of their modulus are returned as real, with real
    eigenvectors. The arrays are real when the returned eigenpairs are real,
    even if other eigenvalues of the solver are complex.

    **Intel MKL.** ARPACK, used by :func:`scipy.sparse.linalg.eigsh`, returns
    wrong eigenpairs at random, without any error, or raises
    ``ArpackError -8`` when linked against Intel MKL 2024.2.0 to 2025.0.0,
    e.g. in some Anaconda builds of SciPy: the ``dsteqr`` routine of these
    MKL versions returns wrong eigenvectors for matrices larger than
    32 x 32, which ARPACK uses with the size ``ncv``. When such a LAPACK is
    detected, ``ncv`` is capped at 32 and more than 15 eigenpairs are
    computed in several passes, combined with a Rayleigh-Ritz projection,
    see :mod:`structsolve.arpackutils`, where the cap can be overridden,
    e.g. with the environment variable ``STRUCTSOLVE_ARPACK_MAX_NCV=0`` to
    lift it with MKL 2025.0.1 or newer.
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

        def verify(mu, eigvecs):
            with np.errstate(divide='ignore', invalid='ignore'):
                return _check_eigenpairs(K, KG, -1./mu, eigvecs, check_rtol)

        strategies = [
            ('eigsh() in Cayley mode',
             lambda: _eigsh_cayley(K, KG, num_eigvalues, tol, silent)),
            ('eigsh() in Cayley mode with a 10 times larger shift',
             lambda: _eigsh_cayley(K, KG, num_eigvalues, tol, silent,
                                   safety=100.)),
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
        eigvals = -1./eigvals

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
        mu, eigvecs = _eig_complex(a=csr_matrix(KG).toarray(),
                                   b=csr_matrix(K).toarray())
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
