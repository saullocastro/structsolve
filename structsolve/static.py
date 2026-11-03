"""Linear static analysis and solution of linear systems

:func:`solve` solves a sparse linear system after removing the null rows and
columns, and :func:`static` is a linear static analysis based on it. For
analyses based on callables, linear or non-linear, see :class:`.Analysis`.

"""
import warnings
import weakref

import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse.linalg import (spsolve, splu, gmres, LinearOperator, eigs,
                                  ArpackError, ArpackNoConvergence)

from .logger import msg
from .sparseutils import remove_null_cols, _used_cols


#: Linear solvers of the factorizations of the tangent stiffness matrix,
#: ``Analysis.linear_solver``
LINEAR_SOLVERS = ('splu', 'pardiso')
#: ``'pardiso'``: a matrix is symmetric when the largest absolute value of
#: ``A - A.T`` is not larger than this factor times the largest absolute
#: value of ``A``
SYMMETRY_RTOL = 1.e-10
#: ``'pardiso'``: relative residual of a solution above which it is improved
#: by GMRES, the factorization being the preconditioner
PARDISO_REFINE_TOL = 1.e-9
#: ``'pardiso'``: relative tolerance of GMRES
PARDISO_GMRES_RTOL = 1.e-11
#: ``'pardiso'``: relative residual after GMRES above which a warning is
#: issued
PARDISO_WARN_TOL = 1.e-6
#: ``'pardiso'``: parameters, ``{i: iparm(i)}``, of the second
#: factorization of a matrix whose first factorization, with the default
#: parameters, perturbed pivots: ``iparm(24) = 1``, the two-level
#: factorization algorithm instead of the classic one, see
#: :func:`._factorize`
PARDISO_FALLBACK_IPARM = {24: 1}
#: ``Analysis.inertia_num_eigvals``: relative tolerance of ARPACK on the
#: eigenvalues of smallest magnitude of a SuperLU factorization, see
#: :func:`._smallest_eigvals`
INERTIA_ARPACK_TOL = 1.e-10
#: ``Analysis.inertia_num_eigvals``: an eigenvalue `\lambda` computed by
#: ARPACK is real when `|\mathrm{Im}(\lambda)| \le` this factor times
#: `|\lambda|`, see :func:`._count_negative_real`
INERTIA_REAL_RTOL = 1.e-8

# instances of pypardiso.PyPardisoSolver whose memory was freed, reused by
# the next factorizations, since creating one may take seconds
_pardiso_pool = []
#: Maximum number of PARDISO solvers that a cache of the factorizations
#: keeps with the analysis of its pattern, for the phase 22, see
#: :func:`._factorize`
CACHE_MAX_PARDISO_SOLVERS = 4


def _pardiso_release(solver):
    """Free the memory of a PARDISO solver and keep it for reuse, with the
    default parameters of PARDISO, ``iparm(1) = 0``"""
    solver.free_memory(everything=True)
    solver.iparm[:] = 0
    _pardiso_pool.append(solver)


def _pardiso_factorize_fallback(A, iparm):
    """Factorization of phase 12 of ``A`` by a new or pooled PARDISO solver
    with the parameters ``iparm`` (the defaults filled by a previous
    factorization), ``iparm(1) = 1``, and those of
    ``PARDISO_FALLBACK_IPARM``; ``None`` when it fails"""
    import pypardiso
    solver = (_pardiso_pool.pop() if _pardiso_pool
              else pypardiso.PyPardisoSolver(mtype=-2))
    solver.iparm[:] = iparm
    solver.set_iparm(1, 1)
    for i, v in PARDISO_FALLBACK_IPARM.items():
        solver.set_iparm(i, v)
    try:
        solver.factorize(A)
    except Exception:
        _pardiso_release(solver)
        return None
    return solver


#: State of the private API of ``pypardiso`` used by the phase 22 of
#: :func:`._pardiso_factorize`: ``None`` before its first use,
#: ``'unverified'`` when it was found, ``True`` when its first phase 22 was
#: verified, ``False`` when it is missing or failed, the public
#: ``factorize`` (phase 12) being then used, see
#: :func:`._pardiso_phase22_api`
_pardiso_private_api = None
#: The private methods and attributes of ``pypardiso.PyPardisoSolver`` used
#: by the phase 22, with the names of the parameters of the methods, those
#: of ``pypardiso`` 0.4.7
_PARDISO_PRIVATE_API = dict(_check_A=('A',), _call_pardiso=('A', 'b'),
                            _hash_csr_matrix=('matrix',),
                            _is_already_factorized=('A',), set_phase=('phase',),
                            factorized_A=None, size_limit_storage=None)


def _pardiso_disable_private_api(reason):
    """Use the public ``factorize`` of ``pypardiso`` from now on, with one
    warning"""
    global _pardiso_private_api
    _pardiso_private_api = False
    warnings.warn("linear_solver='pardiso': the private API of pypardiso "
                  "that reuses the analysis of PARDISO (phase 22) is not "
                  "usable (%s); the public factorize (phase 12), which "
                  "repeats the analysis, is used instead" % reason,
                  RuntimeWarning, stacklevel=4)


def _pardiso_phase22_api(solver):
    """Whether the private API of ``pypardiso`` of the phase 22 is usable,
    checked once per process on the first solver: the attributes of
    ``_PARDISO_PRIVATE_API`` present, the methods callable with the
    parameters of ``pypardiso`` 0.4.7"""
    global _pardiso_private_api
    if _pardiso_private_api is not None:
        return _pardiso_private_api
    import inspect
    for name, params in _PARDISO_PRIVATE_API.items():
        if not hasattr(solver, name):
            _pardiso_disable_private_api('PyPardisoSolver.%s is missing'
                                         % name)
            return False
        if params is None:
            continue
        try:
            found = tuple(inspect.signature(getattr(solver, name)).parameters)
        except (TypeError, ValueError):
            found = None
        if found != params:
            _pardiso_disable_private_api('PyPardisoSolver.%s%s instead of '
                                         '%s' % (name, found, params))
            return False
    # verified by the first factorization of phase 22, see
    # _pardiso_factorize
    _pardiso_private_api = 'unverified'
    return _pardiso_private_api


def _pardiso_factorize_22(solver, A):
    """Phase 22 with the private API of ``pypardiso``, as
    ``pypardiso.PyPardisoSolver.factorize`` with the phase 22 instead of
    12; the copy or the hash of ``A`` is stored such that ``solve`` finds
    it factorized and solves with the phase 33"""
    solver._check_A(A)
    if A.nnz > solver.size_limit_storage:
        solver.factorized_A = solver._hash_csr_matrix(A)
    else:
        solver.factorized_A = A.copy()
    solver.set_phase(22)
    solver._call_pardiso(A, np.zeros((A.shape[0], 1)))
    if not solver._is_already_factorized(A):
        raise RuntimeError('the matrix is not recognized as factorized')


def _pardiso_factorize(solver, A, phase):
    r"""Numerical factorization of ``A`` by the ``pypardiso.PyPardisoSolver``
    ``solver``, with the phase 12 of PARDISO (analysis and factorization), or
    22 (factorization only), which reuses the analysis of the last phase 12
    of ``solver``, valid for a matrix of the same pattern

    ``pypardiso`` 0.4.7 has no public route to the phase 22: its
    ``factorize`` sets the phase 12, then calls the private ``_check_A``
    and ``_call_pardiso``, and stores a copy of ``A``, or its hash for more
    than ``size_limit_storage`` terms, which ``solve`` compares with its
    matrix to solve with the phase 33, otherwise with the phase 13, i.e. a
    new analysis and factorization, correct but slow. The phase 22 is that
    sequence with ``set_phase(22)``, see :func:`._pardiso_factorize_22`.
    The private API is checked once per process, see
    :func:`._pardiso_phase22_api`, and the first factorization of phase 22
    is verified against a factorization of phase 12 of the same matrix by
    the same solver: the same number of negative and perturbed pivots
    (``iparm(23)``, ``iparm(22)`` and ``iparm(14)``) and the same solution
    of one right-hand side, to a relative difference of ``1e-8``. When the
    API is missing, raises or fails this check, the public ``factorize``
    (phase 12) is used from then on, with one :class:`RuntimeWarning`; an
    exception that the phase 12 raises as well is a failure of the
    factorization, not of the API, and is raised.

    """
    global _pardiso_private_api
    if phase == 12 or not _pardiso_phase22_api(solver):
        solver.factorize(A)
        return
    try:
        _pardiso_factorize_22(solver, A)
        if _pardiso_private_api is True:
            return
        # first use: verified against the phase 12
        iparm = solver.get_iparms()
        b = np.linspace(1., 2., A.shape[0])
        x22 = solver.solve(A, b)
        ok = solver.phase == 33
    except Exception as e:
        solver.factorize(A)
        _pardiso_disable_private_api('%s: %s' % (type(e).__name__, e))
        return
    solver.factorize(A)
    iparm12 = solver.get_iparms()
    x12 = solver.solve(A, b)
    ok = (ok and all(iparm[i] == iparm12[i] for i in (14, 22, 23))
          and np.linalg.norm(x22 - x12) <= 1e-8*np.linalg.norm(x12))
    if ok:
        _pardiso_private_api = True
    else:
        _pardiso_disable_private_api('the factorization of phase 22 differs '
                                     'from that of phase 12')


def _release_solvers(solvers):
    """Free the memory of the PARDISO ``solvers`` and empty the list"""
    while solvers:
        _pardiso_release(solvers.pop())


class _FactorizationCache(object):
    r"""Work of :func:`._factorize` that depends only on the sparsity pattern
    of the matrix, reused while the pattern does not change

    Created by :meth:`.Analysis.static` for the factorizations of the
    tangent stiffness matrix along a non-linear analysis, whose pattern is
    usually the same at every iteration, and passed to :func:`._factorize`
    as ``cache``. It keeps the last pattern only; see :func:`._factorize`
    for what is reused and for the checks that keep it valid. :meth:`close`
    frees the memory of its PARDISO solvers.

    """
    def __init__(self):
        self.entry = None
        # generation of the pattern of entry, of the analyses of PARDISO
        self.gen = 0
        # PARDISO solvers with the analysis of the pattern of generation gen,
        # whose factorization is not used by any solve_kT
        self.free_solvers = []
        self.closed = False
        weakref.finalize(self, _release_solvers, self.free_solvers)

    def get(self, K, C):
        r"""The :class:`_PatternEntry` of the CSR matrix ``K``, canonical and
        ``float64``, with the constraints ``C``, CSR or ``None``, built anew
        when the pattern of ``K``, its degrees of freedom whose row or
        column is not null, or ``C`` changed"""
        e = self.entry
        rows = None
        if (e is not None and e.shape == K.shape
                and np.array_equal(e.indptr, K.indptr)
                and np.array_equal(e.indices, K.indices)):
            rows = e.rows
        used_cols = _used_cols(K, rows)
        if (rows is not None and np.array_equal(e.used_cols, used_cols)
                and e.same_constraints(C)):
            return e
        _release_solvers(self.free_solvers)
        self.gen += 1
        self.entry = None
        try:
            self.entry = _PatternEntry(K, used_cols, C)
        except ValueError:
            # duplicated terms summed: not cached
            pass
        return self.entry

    def pardiso_solver(self):
        """A PARDISO solver and the phase of its factorization: 22 for a
        solver that keeps the analysis of the current pattern, 12 otherwise"""
        if self.free_solvers:
            return self.free_solvers.pop(), 22
        if _pardiso_pool:
            return _pardiso_pool.pop(), 12
        import pypardiso
        return pypardiso.PyPardisoSolver(mtype=-2), 12

    def release(self, solver, gen):
        """Free the factorization of ``solver`` and keep its analysis when it
        is of the current pattern, otherwise free all its memory"""
        if (not self.closed and gen == self.gen
                and len(self.free_solvers) < CACHE_MAX_PARDISO_SOLVERS):
            solver.free_memory(everything=False)
            self.free_solvers.append(solver)
        else:
            _pardiso_release(solver)

    def close(self):
        """Free the memory of the PARDISO solvers and forget the pattern"""
        self.closed = True
        self.entry = None
        _release_solvers(self.free_solvers)


class _PatternEntry(object):
    r"""Pattern-dependent data of :func:`._factorize` for one pattern, see
    :class:`_FactorizationCache`

    The reduced and bordered matrix, its CSC form for SuperLU and the scaled
    upper triangle for PARDISO are built once, by the operations of
    :func:`._factorize` applied to a matrix whose stored terms are their own
    positions, 1-based, which gives the position in ``K.data``, or in the
    data of the border, of each of their terms; a factorization then only
    gathers ``K.data``.

    """
    def __init__(self, K, used_cols, C):
        self.shape = K.shape
        self.indptr = K.indptr.copy()
        self.indices = K.indices.copy()
        self.rows = np.repeat(np.arange(K.shape[0]), np.diff(K.indptr))
        self.used_cols = used_cols
        self.C = None if C is None else (C.shape, C.indptr.copy(),
                                         C.indices.copy(), C.data.copy())
        nnz = K.nnz
        tags = csr_matrix((np.arange(1, nnz + 1, dtype=np.float64),
                           K.indices, K.indptr), shape=K.shape)
        if used_cols.shape[0] != K.shape[1]:
            # as remove_null_cols
            tags = tags[used_cols, :]
            tags = tags[:, used_cols]
        Cr, m, kept = _reduced_constraints(C, used_cols)
        self.m = m
        self.border_rhs = _Border(0 if C is None else C.shape[1], kept)
        self.border = None
        if m:
            self.border = Cr.data.copy()
            Ct = csc_matrix((np.arange(nnz + 1, nnz + 1 + Cr.nnz,
                                       dtype=np.float64),
                             Cr.indices, Cr.indptr), shape=Cr.shape)
            tags = bmat([[None, Ct.T], [Ct, tags]], format='csr')
        A = csr_matrix(tags)
        self.A_tags = A
        self.A_gather = _positions(A.data)
        self.A_sorted = bool(A.has_sorted_indices)
        self.csc = None
        self.pardiso = None

    def same_constraints(self, C):
        """Whether ``C`` is the matrix of the constraints of the entry"""
        if C is None or self.C is None:
            return C is None and self.C is None
        shape, indptr, indices, data = self.C
        return (shape == C.shape and np.array_equal(indptr, C.indptr)
                and np.array_equal(indices, C.indices)
                and np.array_equal(data, C.data))

    def _ext(self, K):
        """The data of ``K`` followed by the data of the border"""
        if self.m:
            return np.concatenate((K.data, self.border))
        return K.data

    def reduced_csr(self, K):
        """The reduced and bordered matrix of ``K``, CSR"""
        A = csr_matrix((self._ext(K)[self.A_gather], self.A_tags.indices,
                        self.A_tags.indptr), shape=self.A_tags.shape)
        if self.A_sorted:
            A.has_sorted_indices = True
        return A

    def reduced_csc(self, K):
        """The reduced and bordered matrix of ``K``, CSC"""
        if self.csc is None:
            At = csc_matrix(self.A_tags)
            self.csc = (_positions(At.data), At.indices, At.indptr,
                        bool(At.has_sorted_indices))
        gather, indices, indptr, srt = self.csc
        A = csc_matrix((self._ext(K)[gather], indices, indptr),
                       shape=self.A_tags.shape)
        if srt:
            A.has_sorted_indices = True
        return A

    def pardiso_data(self):
        """Pattern data of the symmetry check and of the scaled upper
        triangle of :func:`._factorize_pardiso`"""
        if self.pardiso is None:
            A = self.A_tags
            N = A.shape[0]
            nnzA = A.nnz
            Ad = csr_matrix((np.arange(1, nnzA + 1, dtype=np.float64),
                             A.indices, A.indptr), shape=A.shape)
            # position of the transposed term of each term, nnzA when absent
            At = Ad.T.tocsr()
            if (np.array_equal(At.indptr, A.indptr)
                    and np.array_equal(At.indices, A.indices)):
                tpos = _positions(At.data)
            else:
                rows = np.repeat(np.arange(N, dtype=np.int64),
                                 np.diff(A.indptr))
                cols = A.indices.astype(np.int64)
                keys = rows*N + cols
                tpos = None
                if nnzA and np.all(np.diff(keys) > 0):
                    tkeys = cols*N + rows
                    pos = np.minimum(np.searchsorted(keys, tkeys), nnzA - 1)
                    tpos = np.where(keys[pos] == tkeys, pos, nnzA)
            # positions of the diagonal, nnzA when absent
            dpos = _positions(Ad.diagonal())
            dpos[dpos < 0] = nnzA
            U = _scaled_upper(Ad, np.ones(N))
            upos = _positions(U.data)
            urows = np.repeat(np.arange(N), np.diff(U.indptr))
            # the diagonal terms of A, to which an explicit zero is added
            dup = np.flatnonzero((urows == U.indices) & (upos >= 0))
            upos[upos < 0] = nnzA
            self.pardiso = dict(tpos=tpos, dpos=dpos, upos=upos, urows=urows,
                                ucols=U.indices, dup=dup, U=U,
                                U_sorted=bool(U.has_sorted_indices))
        return self.pardiso

    def is_symmetric(self, A):
        """:func:`_is_symmetric` of the reduced matrix ``A`` of this entry"""
        p = self.pardiso_data()
        if p['tpos'] is None:
            return _is_symmetric(A)
        a = A.data
        if a.size == 0:
            return True
        ae = np.append(a, 0.)
        return np.abs(a - ae[p['tpos']]).max() <= \
            SYMMETRY_RTOL*np.abs(a).max()

    def scaled_upper(self, A):
        """The scaling and the scaled upper triangle of
        :func:`._scaled_upper` of the reduced matrix ``A`` of this entry"""
        p = self.pardiso_data()
        ae = np.append(A.data, 0.)
        d = np.abs(ae[p['dpos']])
        d[d == 0] = 1.
        s = 1/np.sqrt(d)
        data = ae[p['upos']]*s[p['urows']]*s[p['ucols']]
        data[p['dup']] += 0.
        U = csr_matrix((data, p['U'].indices, p['U'].indptr), shape=A.shape)
        if p['U_sorted']:
            U.has_sorted_indices = True
        return s, U


def _positions(tags):
    """Positions, 0-based, given by the ``float`` ``tags``, 1-based, of the
    terms built by :class:`_PatternEntry`, ``-1`` for the tag 0 of a term
    that is not in the original matrix"""
    pos = tags.astype(np.int64)
    if not np.array_equal(pos, tags):
        raise ValueError('_FactorizationCache: terms were summed')
    return pos - 1


def _reduced_constraints(C, used_cols):
    """The matrix of the constraints on the degrees of freedom
    ``used_cols``, CSC, without the constraints left without any term,
    their number and the indices of the constraints kept"""
    if C is None:
        return None, 0, np.zeros(0, dtype=np.intp)
    Cr = csc_matrix(C[used_cols])
    kept = np.flatnonzero(np.diff(Cr.indptr))
    Cr = Cr[:, kept]
    return Cr, Cr.shape[1], kept


class _Border(object):
    """The right-hand side of the border of a factorization, see
    :func:`._factorize`"""
    def __init__(self, num_constraints, kept):
        self.num_constraints = num_constraints
        self.kept = kept

    def __call__(self, d, dtype):
        """The values ``d`` of the constraints kept, zeros when ``None``"""
        m = self.kept.shape[0]
        if d is None:
            return np.zeros(m, dtype=dtype)
        if not self.num_constraints:
            raise ValueError('The values of the constraints require '
                             'constraints')
        d = np.asarray(d)
        if d.shape != (self.num_constraints,):
            raise ValueError('The values of the constraints must have the '
                             'shape (%d,)' % self.num_constraints)
        dropped = np.ones(self.num_constraints, dtype=bool)
        dropped[self.kept] = False
        if np.any(d[dropped] != 0):
            raise ValueError('The constraints %s have no term on the '
                             'degrees of freedom whose row or column of the '
                             'matrix is not null, and cannot have a non-zero '
                             'value' % np.flatnonzero(dropped).tolist())
        return d[self.kept]


def _scaled_upper(A, s):
    """Upper triangle of ``D A D``, ``D = diag(s)``, CSR, the diagonal
    stored explicitly, also its zeros, of the border"""
    Acoo = A.tocoo()
    upper = Acoo.row <= Acoo.col
    row = Acoo.row[upper]
    col = Acoo.col[upper]
    n = A.shape[0]
    return csr_matrix((np.r_[Acoo.data[upper]*s[row]*s[col], np.zeros(n)],
                       (np.r_[row, np.arange(n)],
                        np.r_[col, np.arange(n)])), shape=A.shape)


def solve(a, b, silent=False, **kwargs):
    r"""Solve a linear system of equations removing null rows and columns


    Wrapper for :func:`scipy.sparse.linalg.spsolve`. The rows and columns of
    ``a`` without any non-zero term, typically the constrained degrees of
    freedom of a Ritz or finite element model, are removed before solving
    the linear system of equations. The corresponding values of the solution
    ``x`` are zero.

    Parameters
    ----------
    a : ndarray or sparse matrix
        A square matrix, converted to CSR form in the solution. It may be
        unsymmetric, e.g. with the load stiffness of follower loads, since
        :func:`scipy.sparse.linalg.spsolve` uses an LU factorization. The
        degrees of freedom whose row and column are both null are removed.
    b : ndarray
        1-D array representing the right-hand side of the system of equations.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    kwargs : keyword arguments, optional
        Other arguments directly passed to :func:`scipy.sparse.linalg.spsolve`.

    Returns
    -------
    x : ndarray
        The solution of the system of equations, a 1-D array of size
        ``a.shape[1]`` with the same ``dtype`` as ``b``.

    Notes
    -----
    With `u` the set of the degrees of freedom whose row or column of ``a``
    is not null, the reduced system `[a_{uu}]\boldsymbol{x}_u = \boldsymbol{b}_u` is solved
    and `x = 0` at the other degrees of freedom. The terms of ``b`` at the
    removed degrees of freedom are ignored: ``x`` solves the full system
    only when they are null, otherwise the full system has no solution,
    since a null row and column make ``a`` singular.

    """
    a, used_cols = remove_null_cols(a, silent=silent)
    px = spsolve(a, b[used_cols], **kwargs)
    x = np.zeros(b.shape[0], dtype=b.dtype)
    x[used_cols] = px

    return x


def _permutation_sign(perm):
    """Sign of a permutation given as an array of indices, from the number
    of its cycles"""
    perm = np.asarray(perm)
    seen = np.zeros(perm.size, dtype=bool)
    num_cycles = 0
    for i in range(perm.size):
        if not seen[i]:
            num_cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return -1 if (perm.size - num_cycles) % 2 else 1


def _is_symmetric(a):
    """Whether the sparse matrix ``a`` is symmetric to ``SYMMETRY_RTOL``"""
    d = (a - a.T).data
    if d.size == 0:
        return True
    return np.abs(d).max() <= SYMMETRY_RTOL*np.abs(a.data).max()


def _factorize(kT, linear_solver='splu', constraints=None, cache=None):
    r"""Factorization of the tangent stiffness matrix

    Null rows and columns are removed, the corresponding values of the
    solution are zero, as in :func:`.solve`. The factorization is reused by
    the non-linear solvers while the tangent stiffness matrix is not
    updated.

    Parameters
    ----------
    kT : sparse matrix
        Tangent stiffness matrix.
    linear_solver : str, optional
        ``'splu'`` (default) for the sparse LU factorization of SuperLU, or
        ``'pardiso'`` for the symmetric indefinite factorization of PARDISO,
        which requires the package ``pypardiso``, see ``LINEAR_SOLVERS``.
    constraints : array-like, sparse matrix or None, optional
        The matrix `\boldsymbol{C}` of the linear constraints
        `\boldsymbol{C}^\top \boldsymbol{x} = \boldsymbol{0}`, of shape ``(n, m)``
        for ``m`` constraints, see the Notes. ``None`` (default) for none.
    cache : :class:`_FactorizationCache` or None, optional
        Cache of the work that depends only on the sparsity pattern of
        ``kT``, reused by the next factorizations of a matrix of the same
        pattern, see the Notes. ``None`` (default) for none.

    Returns
    -------
    solve_kT : callable
        ``solve_kT(b, d=None)`` returns the solution ``x`` of ``kT x = b``,
        or of the system bordered by the constraints, whose values are
        ``d``, of shape ``(m,)``, zeros when ``None``, see the Notes; it
        has the attributes:

        - ``linear_solver``: the linear solver used, ``'splu'`` when
          ``'pardiso'`` fell back to it for an unsymmetric matrix;
        - ``det_sign``: the sign of the determinant of the reduced matrix,
          ``1`` or ``-1``, on the space of the constraints when given;
        - ``num_negative_eigvals``: the number of negative eigenvalues of the
          reduced matrix with ``'pardiso'``, on the space of the
          constraints when given, ``None`` with ``'splu'``;
        - ``num_constraints``: the number ``m`` of constraints of the
          border;
        - ``num_perturbed_pivots``: the number of pivots perturbed by
          PARDISO, ``None`` with ``'splu'``;
        - ``smallest_eigvals``, with ``'splu'`` only: ``smallest_eigvals(k)``
          returns the ``k`` eigenvalues of smallest magnitude of the reduced
          matrix, on the space of the constraints, by shift-invert with
          ARPACK and this factorization, see :func:`._smallest_eigvals`.

    Raises
    ------
    RuntimeError
        When the matrix is singular, for ``'splu'``, or when PARDISO fails.
    ImportError
        When ``linear_solver='pardiso'`` and ``pypardiso`` is not installed.
    ValueError
        For an invalid ``linear_solver``.

    Notes
    -----
    With ``'splu'``, the reduced matrix is factorized once with SuperLU,
    :func:`scipy.sparse.linalg.splu` with its default options, i.e. a
    sparse LU factorization `\boldsymbol{P}_r \boldsymbol{A} \boldsymbol{P}_c = \boldsymbol{L}
    \boldsymbol{U}` with row and column permutations, valid for unsymmetric
    matrices. Each call of ``solve_kT`` is then only a forward and a
    backward substitution, which is why the factorization is reused while
    the tangent stiffness matrix is not updated, e.g. in the modified
    Newton-Raphson method and in the two solutions of the bordered system
    of the arc-length methods. `\boldsymbol{L}` having a unit diagonal, the sign
    of the determinant is the product of the signs of the diagonal of
    `\boldsymbol{U}` and of the two permutations; for a symmetric matrix it is
    `(-1)^{n_{neg}}`, with `n_{neg}` its number of negative eigenvalues,
    which LU does not give.

    With ``'pardiso'``, the reduced matrix `\boldsymbol{A}` is scaled to a unit
    diagonal, `\boldsymbol{A}_s = \boldsymbol{D} \boldsymbol{A} \boldsymbol{D}`, `\boldsymbol{D} =
    \mathrm{diag}(|a_{ii}|^{-1/2})`, a null diagonal term being replaced by
    1, and the upper triangle of `\boldsymbol{A}_s` is factorized by PARDISO as
    a real symmetric indefinite matrix (``mtype=-2``), `\boldsymbol{L} \boldsymbol{D}
    \boldsymbol{L}^\top` with Bunch-Kaufman pivoting. By Sylvester's law of
    inertia, the number of negative pivots, ``iparm(23)`` of PARDISO, is
    the number of negative eigenvalues of `\boldsymbol{A}_s`, hence of
    `\boldsymbol{A}`, the congruence by `\boldsymbol{D}` preserving the inertia: zero
    on the stable parts of an equilibrium path, it changes at the limit
    points and at the bifurcation points crossed. PARDISO perturbs the
    pivots that are too small, ``iparm(14)`` of them, which makes the count
    uncertain when it is not zero. When the first factorization perturbed
    pivots, the matrix is factorized again with the parameters
    ``PARDISO_FALLBACK_IPARM``, ``iparm(24) = 1``, the two-level
    factorization algorithm, and the factorization with fewer perturbed
    pivots is kept, ``solve_kT.pardiso_fallback`` telling whether it is
    the second one; with ``cache``, the next factorizations of the same
    pattern are then done with these parameters directly, with the phase
    12. The tangent stiffness matrices of a Ritz model of a cylinder
    (panels 0.15.0, ``cylshell_clpt_donnell``, 8 x 8 terms, 114 degrees
    of freedom, a dense matrix) and dense random indefinite matrices of
    114 and 200 degrees of freedom had 18 to 40 perturbed pivots with the
    default parameters (pypardiso 0.4.7, MKL 2026.1, ``iparm(10) = 8``,
    ``iparm(11) = iparm(13) = 0``, ``iparm(21) = 1``, ``iparm(24) = 0``),
    solutions of relative residual 0.4 to 1, recovered by the GMRES
    below, and a number of negative pivots short by one, e.g. 0 from the
    load factor 0.827 of the critical point, where a dense eigenvalue
    analysis has one negative eigenvalue, up to 0.903; neither
    ``iparm(10) = 13`` or ``20``, nor the scaling and the weighted
    matching, ``iparm(11) = iparm(13) = 1``, nor ``iparm(21) = 3`` changed
    it, and ``iparm(24) = 1`` gave no perturbed pivot, the counts of the
    dense eigenvalue analysis along the whole path, and residuals of
    ``1e-13``. The tangent of the DOE09 cylinder below, sparse, had no
    perturbed pivot. The diagonal of the tangent of a model
    with derivatives of the displacements among its unknowns may span many
    orders of magnitude, which the scaling compensates. Every solution is
    checked against its relative residual `\|\boldsymbol{A} \boldsymbol{x} -
    \boldsymbol{b}\|/\|\boldsymbol{b}\|` and, when it is larger than
    ``PARDISO_REFINE_TOL``, improved by GMRES with the factorization as the
    preconditioner, to the relative tolerance ``PARDISO_GMRES_RTOL``; a
    :class:`RuntimeWarning` is issued when the residual remains larger than
    ``PARDISO_WARN_TOL``. These are the settings of the Riks analyses of a
    tow-steered cylinder for the DCSE website (2026), where the
    unsymmetric LU of PARDISO was wrong and its Cholesky factorization
    failed beyond the limit point. The symmetric factorization would be
    wrong for an unsymmetric matrix, e.g. with the load stiffness of a
    follower load: ``'splu'`` is then used, with a :class:`RuntimeWarning`.
    With ``constraints``, the bordered system

    .. math::

        \begin{bmatrix} \boldsymbol{0} & \boldsymbol{C}^\top \\
        \boldsymbol{C} & \boldsymbol{K}_T \end{bmatrix}
        \begin{Bmatrix} \boldsymbol{\mu} \\ \boldsymbol{x} \end{Bmatrix} =
        \begin{Bmatrix} \boldsymbol{d} \\ \boldsymbol{b} \end{Bmatrix}

    is factorized instead, the border first, such that no row of the upper
    triangle factorized by PARDISO is empty, and ``solve_kT`` returns
    `\boldsymbol{x}`, which satisfies `\boldsymbol{C}^\top \boldsymbol{x} =
    \boldsymbol{d}`, `\boldsymbol{\mu}` being the Lagrange multipliers,
    the reactions of the constraints, `\boldsymbol{d} = \boldsymbol{0}`
    unless ``d`` is given, e.g. a prescribed displacement. The rows of
    `\boldsymbol{C}` of the null rows and columns of ``kT`` are removed,
    and the constraints left without any non-zero term, satisfied with
    `d_j = 0`, are dropped; a non-zero `d_j` of such a constraint raises a
    :class:`ValueError`, the degrees of freedom of the null rows and
    columns being zero. The
    bordered matrix is indefinite even when `\boldsymbol{K}_T` is positive
    definite on the space of the constraints: for `\boldsymbol{C}` of full
    rank its inertia is that of `\boldsymbol{Z}^\top \boldsymbol{K}_T
    \boldsymbol{Z}` plus `m` positive and `m` negative eigenvalues,
    `\boldsymbol{Z}` a basis of the null space of `\boldsymbol{C}^\top`
    (Gould, N.I.M., On practical conditions for the existence and
    uniqueness of solutions to the general equality quadratic programming
    problem, Mathematical Programming, 32, 90-99, 1985), hence
    ``num_negative_eigvals`` is the number of negative pivots minus `m`,
    and ``det_sign`` is multiplied by `(-1)^m`. This is the border of the
    inertia relief of the Riks analyses of a tow-steered cylinder for the
    DCSE website (2026), with a single constraint.

    The memory of PARDISO is freed when ``solve_kT`` is deleted, and its
    solver object is kept for the next factorizations, since creating one
    took about 6.5 s on Windows with ``pypardiso`` 0.4.7, against
    milliseconds for the factorization of a small matrix.

    With ``cache``, created by :meth:`.Analysis.static` for each non-linear
    analysis, along which the pattern of the tangent stiffness matrix
    usually does not change, the work that depends only on the pattern is
    done once per pattern: the indices of the reduced and bordered matrix,
    of its CSC form, of the transposed terms of the symmetry check, of its
    diagonal and of the scaled upper triangle, from which each
    factorization only gathers the values of ``kT``; and the analysis
    (reordering and symbolic factorization) of PARDISO, its phase 11, the
    next factorizations being its phase 22 instead of 12 with a solver
    that keeps the analysis of the pattern, through the private API of
    ``pypardiso``, checked and verified by :func:`._pardiso_factorize`,
    which falls back to the phase 12 when it is not usable; up to
    ``CACHE_MAX_PARDISO_SOLVERS`` such solvers are kept, since the
    factorizations of the last converged state and of the current
    iteration coexist. The pattern is the shape, ``indptr`` and
    ``indices`` of ``kT`` in CSR form, compared at every call, the
    degrees of freedom whose row or column is not null, which depend on
    the values, since explicit zeros do not count, computed at every call,
    and ``constraints``, compared with its values; any change rebuilds the
    cache. The values enter only through the gathered terms, and the
    symmetry check, the scaling and the null rows and columns are evaluated
    at every call, such that the factorization is bitwise the one without
    the cache, verified by the tests, also the inertia and the perturbed
    pivots of PARDISO (with ``MKL_NUM_THREADS=1``; the analysis of PARDISO
    depends only on the pattern, with its default ``iparm(11) = iparm(13)
    = 0`` for symmetric indefinite matrices). A matrix that is not
    ``float64`` or whose CSR form has duplicated or unsorted terms is
    factorized without the cache. On a symmetric matrix of 57000 degrees
    of freedom with 4.5 million terms, the factorization took 1.65 s
    without and 0.40 s with the cache with ``'pardiso'``, with or without
    one constraint, and 1.1 s with ``'splu'``, whose preprocessing went
    from 0.11 s to 0.05 s, SuperLU computing its ordering at every
    factorization.

    """
    if linear_solver not in LINEAR_SOLVERS:
        raise ValueError('Invalid linear_solver: %s' % linear_solver)
    C = None
    if constraints is not None:
        C = csr_matrix(constraints)
        if C.shape[0] == 1 and kT.shape[0] != 1:
            C = C.T.tocsr()
    entry = None
    if cache is not None:
        K = kT if isinstance(kT, csr_matrix) else csr_matrix(kT)
        if K.dtype == np.float64 and K.has_canonical_format:
            entry = cache.get(K, C)
    if entry is not None:
        used_cols = entry.used_cols
        m = entry.m
        border = entry.border_rhs
        if linear_solver == 'pardiso':
            A = entry.reduced_csr(K)
            if entry.is_symmetric(A):
                return _factorize_pardiso(A, used_cols, border,
                                          scaled=entry.scaled_upper(A),
                                          cache=cache)
            warnings.warn("linear_solver='pardiso': the matrix is "
                          "unsymmetric, 'splu' is used instead",
                          RuntimeWarning, stacklevel=3)
        lu = splu(entry.reduced_csc(K))
    else:
        kT_red, used_cols = remove_null_cols(kT, silent=True)
        Cr, m, kept = _reduced_constraints(C, used_cols)
        border = _Border(0 if C is None else C.shape[1], kept)
        if m:
            kT_red = bmat([[None, Cr.T], [Cr, kT_red]], format='csr')
        if linear_solver == 'pardiso':
            A = csr_matrix(kT_red)
            if _is_symmetric(A):
                return _factorize_pardiso(A, used_cols, border)
            warnings.warn("linear_solver='pardiso': the matrix is "
                          "unsymmetric, 'splu' is used instead",
                          RuntimeWarning, stacklevel=3)
        lu = splu(csc_matrix(kT_red))

    def solve_kT(b, d=None):
        bd = border(d, b.dtype)
        x = np.zeros(b.shape[0], dtype=np.result_type(b, bd, lu.U.dtype))
        if m:
            x[used_cols] = lu.solve(np.concatenate((bd, b[used_cols])))[m:]
        else:
            x[used_cols] = lu.solve(b[used_cols])
        return x

    diagU = lu.U.diagonal()
    solve_kT.smallest_eigvals = lambda k: _smallest_eigvals(lu, m, k)
    solve_kT.linear_solver = 'splu'
    solve_kT.det_sign = int(np.prod(np.sign(diagU))
                            *_permutation_sign(lu.perm_r)
                            *_permutation_sign(lu.perm_c)*(-1)**m)
    solve_kT.num_negative_eigvals = None
    solve_kT.num_perturbed_pivots = None
    solve_kT.num_constraints = m

    return solve_kT


def _smallest_eigvals(lu, m, k):
    r"""The ``k`` eigenvalues of smallest magnitude of the matrix of the
    SuperLU factorization ``lu``, on the space of its ``m`` constraints

    The matrix is the reduced matrix `\boldsymbol{K}` of
    :func:`._factorize`, bordered first by ``m`` constraints
    `\boldsymbol{C}`. Shift-invert at `\sigma = 0` with ARPACK,
    :func:`scipy.sparse.linalg.eigs`, on the operator `\boldsymbol{T}
    \boldsymbol{b} = \boldsymbol{x}`, the solution of the bordered system
    `\boldsymbol{K} \boldsymbol{x} + \boldsymbol{C} \boldsymbol{\mu} =
    \boldsymbol{b}`, `\boldsymbol{C}^\top \boldsymbol{x} = \boldsymbol{0}`,
    by the forward and backward substitutions of ``lu``, which is not
    factorized again: an eigenvalue `\nu \ne 0` of `\boldsymbol{T}` has its
    eigenvector `\boldsymbol{b}` in the null space of
    `\boldsymbol{C}^\top`, and `\boldsymbol{P} \boldsymbol{K} \boldsymbol{b}
    = \boldsymbol{b}/\nu`, `\boldsymbol{P}` the orthogonal projection on
    that space, i.e. `\lambda = 1/\nu` is an eigenvalue of
    `\boldsymbol{Z}^\top \boldsymbol{K} \boldsymbol{Z}`, `\boldsymbol{Z}` an
    orthonormal basis of the null space of `\boldsymbol{C}^\top`; the range
    of `\boldsymbol{C}`, of `\nu = 0`, is not among the largest `|\nu|`.
    With ``k`` larger than the size minus 2, the limit of ARPACK, all the
    eigenvalues are computed with :func:`numpy.linalg.eigvals` from
    `\boldsymbol{T}` built column by column.

    Returns
    -------
    lam : ndarray or None
        The eigenvalues, complex, sorted by increasing magnitude, ``None``
        when ARPACK did not converge.

    """
    from .arpackutils import arpack_ncv, release_memory, start_kwargs
    n = lu.shape[0] - m
    k = int(k)
    if n == 0:
        return np.zeros(0, dtype=complex)
    if k >= n - 1:
        B = np.zeros((m + n, n))
        B[m:] = np.eye(n)
        nu = np.linalg.eigvals(lu.solve(B)[m:])
        nu = nu[np.abs(nu) > n*np.finfo(float).eps*np.abs(nu).max()]
    else:
        z = np.zeros(m)
        op = LinearOperator((n, n), dtype=np.float64,
                            matvec=lambda v: lu.solve(np.concatenate(
                                (z, np.asarray(v, dtype=np.float64).ravel())))[m:])
        try:
            nu = eigs(op, k=k, which='LM', tol=INERTIA_ARPACK_TOL,
                      ncv=arpack_ncv(k, n), return_eigenvectors=False,
                      **start_kwargs(n))
        except (ArpackNoConvergence, ArpackError):
            return None
        finally:
            release_memory()
        nu = nu[nu != 0]
    lam = 1./nu.astype(complex)
    return lam[np.argsort(np.abs(lam), kind='stable')][:k]


def _count_negative_real(lam):
    r"""The number of real eigenvalues ``lam`` that are negative, and of the
    complex ones with a negative real part, see ``INERTIA_REAL_RTOL``"""
    real = np.abs(lam.imag) <= INERTIA_REAL_RTOL*np.abs(lam)
    neg = lam.real < 0
    return int(np.count_nonzero(real & neg)), int(np.count_nonzero(~real
                                                                   & neg))


def _factorize_pardiso(A, used_cols, border, scaled=None, cache=None):
    """Symmetric indefinite factorization of PARDISO of the reduced matrix
    ``A``, bordered first by the constraints of the :class:`_Border`
    ``border``, see :func:`._factorize`; ``scaled``, the scaling and the
    scaled upper triangle of ``A``, is computed when ``None``"""
    m = border.kept.shape[0]
    try:
        import pypardiso
    except ImportError:
        raise ImportError("linear_solver='pardiso' requires the package "
                          "pypardiso")
    if scaled is None:
        d = np.abs(A.diagonal())
        d[d == 0] = 1.
        s = 1/np.sqrt(d)
        As_upper = _scaled_upper(A, s)
    else:
        s, As_upper = scaled
    fallback = (cache is not None
                and getattr(cache, 'pardiso_fallback_gen', None) == cache.gen)
    solver = None
    if fallback:
        # the first factorization of this pattern perturbed pivots
        solver = _pardiso_factorize_fallback(As_upper, cache.pardiso_iparm)
        cached = False
    if solver is None:
        if cache is not None:
            solver, phase = cache.pardiso_solver()
            gen = cache.gen
        else:
            solver = (_pardiso_pool.pop() if _pardiso_pool
                      else pypardiso.PyPardisoSolver(mtype=-2))
            phase = 12
        cached = cache is not None
        try:
            _pardiso_factorize(solver, As_upper, phase)
        except Exception as e:
            _pardiso_release(solver)
            raise RuntimeError('PARDISO factorization failed: %s' % e)
        if not fallback and int(solver.iparm[13]) > 0:
            # perturbed pivots: the parameters of the fallback
            solver2 = _pardiso_factorize_fallback(As_upper, solver.iparm)
            if (solver2 is not None
                    and int(solver2.iparm[13]) < int(solver.iparm[13])):
                if cache is not None:
                    cache.pardiso_fallback_gen = cache.gen
                    cache.pardiso_iparm = solver.iparm.copy()
                    cache.release(solver, gen)
                else:
                    _pardiso_release(solver)
                solver = solver2
                cached = False
            elif solver2 is not None:
                _pardiso_release(solver2)
    iparms = solver.get_iparms()

    def precond(r):
        return s*solver.solve(As_upper, np.ascontiguousarray(s*r))

    def solve_kT(b, d=None):
        x = np.zeros(b.shape[0], dtype=np.result_type(b, np.float64))
        bu = np.concatenate((border(d, np.float64),
                             np.asarray(b[used_cols], dtype=np.float64)))
        bnorm = np.linalg.norm(bu)
        if bnorm == 0:
            return x
        xu = precond(bu)
        res = np.linalg.norm(A @ xu - bu)/bnorm
        if not res <= PARDISO_REFINE_TOL:
            M = LinearOperator(A.shape, matvec=precond, dtype=np.float64)
            xu, _ = gmres(A, bu, x0=xu, M=M, rtol=PARDISO_GMRES_RTOL, atol=0.,
                          restart=30, maxiter=10)
            res = np.linalg.norm(A @ xu - bu)/bnorm
            if not res <= PARDISO_WARN_TOL:
                warnings.warn("linear_solver='pardiso': relative residual "
                              "%1.1e of a solution after GMRES" % res,
                              RuntimeWarning, stacklevel=2)
        x[used_cols] = xu[m:]
        return x

    solve_kT.linear_solver = 'pardiso'
    solve_kT.num_constraints = m
    solve_kT.num_negative_eigvals = int(iparms[23]) - m
    solve_kT.num_perturbed_pivots = int(iparms[14])
    solve_kT.det_sign = -1 if solve_kT.num_negative_eigvals % 2 else 1
    solve_kT.pardiso_fallback = int(iparms[1]) == 1
    if cached:
        weakref.finalize(solve_kT, cache.release, solver, gen)
    else:
        weakref.finalize(solve_kT, _pardiso_release, solver)

    return solve_kT


def static(K, fext, silent=False):
    r"""Linear static analysis

    Solves the linear system ``K u = fext`` using :func:`.solve`. For linear and

    non-linear static analyses based on callables that calculate the force
    vectors and stiffness matrices, see :class:`.Analysis`.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix. Should include initial stress stiffness matrix,
        aerodynamic matrix and so forth when applicable. It may be
        unsymmetric, e.g. ``K0 + kCfollower`` for a follower load on a
        geometrically linear structure, whose load stiffness is ``kCfollower
        = -dF/dc``.
    fext : array-like
        Vector of external loads.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.

    Returns
    -------
    increments : list
        ``[1.]``, the load factor of the solution.
    cs : list
        List with the solution vector.

    """
    increments = []
    cs = []

    NLgeom=False
    if NLgeom:
        raise NotImplementedError('Independent static function not ready for NLgeom')
    else:
        msg('Started Linear Static Analysis', silent=silent)
        c = solve(K, fext, silent=silent)
        increments.append(1.)
        cs.append(c)
        msg('Finished Linear Static Analysis', silent=silent)

    return increments, cs
