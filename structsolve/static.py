"""Linear static analysis and solution of linear systems

:func:`solve` solves a sparse linear system after removing the null rows and
columns, and :func:`static` is a linear static analysis based on it. For
analyses based on callables, linear or non-linear, see :class:`.Analysis`.

"""
import warnings
import weakref

import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags, triu
from scipy.sparse.linalg import spsolve, splu, gmres, LinearOperator

from .logger import msg
from .sparseutils import remove_null_cols


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

# instances of pypardiso.PyPardisoSolver whose memory was freed, reused by
# the next factorizations, since creating one may take seconds
_pardiso_pool = []


def _pardiso_release(solver):
    """Free the memory of a PARDISO solver and keep it for reuse"""
    solver.free_memory(everything=True)
    _pardiso_pool.append(solver)


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
    d = abs(a - a.T)
    if d.nnz == 0:
        return True
    return d.max() <= SYMMETRY_RTOL*abs(a).max()


def _factorize(kT, linear_solver='splu', constraints=None):
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

    Returns
    -------
    solve_kT : callable
        Function that returns the solution ``x`` of ``kT x = b``, or of the
        system bordered by the constraints, with the attributes:

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
          PARDISO, ``None`` with ``'splu'``.

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
    uncertain when it is not zero. The diagonal of the tangent of a model
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
        \begin{Bmatrix} \boldsymbol{0} \\ \boldsymbol{b} \end{Bmatrix}

    is factorized instead, the border first, such that no row of the upper
    triangle factorized by PARDISO is empty, and ``solve_kT`` returns
    `\boldsymbol{x}`, which satisfies `\boldsymbol{C}^\top \boldsymbol{x} =
    \boldsymbol{0}`, `\boldsymbol{\mu}` being the Lagrange multipliers,
    the reactions of the constraints. The rows of `\boldsymbol{C}` of the
    null rows and columns of ``kT`` are removed, and the constraints left
    without any non-zero term, already satisfied, are dropped. The
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

    """
    if linear_solver not in LINEAR_SOLVERS:
        raise ValueError('Invalid linear_solver: %s' % linear_solver)
    kT_red, used_cols = remove_null_cols(kT, silent=True)
    m = 0
    if constraints is not None:
        C = csr_matrix(constraints)
        if C.shape[0] == 1 and kT.shape[0] != 1:
            C = C.T.tocsr()
        Cr = csc_matrix(C[used_cols])
        Cr = Cr[:, np.flatnonzero(np.diff(Cr.indptr))]
        m = Cr.shape[1]
        if m:
            kT_red = bmat([[None, Cr.T], [Cr, kT_red]], format='csr')
    if linear_solver == 'pardiso':
        A = csr_matrix(kT_red)
        if _is_symmetric(A):
            return _factorize_pardiso(A, used_cols, m)
        warnings.warn("linear_solver='pardiso': the matrix is unsymmetric, "
                      "'splu' is used instead", RuntimeWarning, stacklevel=3)
    lu = splu(csc_matrix(kT_red))

    def solve_kT(b):
        x = np.zeros(b.shape[0], dtype=np.result_type(b, lu.U.dtype))
        if m:
            x[used_cols] = lu.solve(np.concatenate((np.zeros(m, dtype=b.dtype),
                                                    b[used_cols])))[m:]
        else:
            x[used_cols] = lu.solve(b[used_cols])
        return x

    diagU = lu.U.diagonal()
    solve_kT.linear_solver = 'splu'
    solve_kT.det_sign = int(np.prod(np.sign(diagU))
                            *_permutation_sign(lu.perm_r)
                            *_permutation_sign(lu.perm_c)*(-1)**m)
    solve_kT.num_negative_eigvals = None
    solve_kT.num_perturbed_pivots = None
    solve_kT.num_constraints = m

    return solve_kT


def _factorize_pardiso(A, used_cols, m=0):
    """Symmetric indefinite factorization of PARDISO of the reduced matrix
    ``A``, bordered by ``m`` constraints first, see :func:`._factorize`"""
    try:
        import pypardiso
    except ImportError:
        raise ImportError("linear_solver='pardiso' requires the package "
                          "pypardiso")
    d = np.abs(A.diagonal())
    d[d == 0] = 1.
    s = 1/np.sqrt(d)
    D = diags(s)
    # the diagonal stored explicitly, also its zeros, of the border
    As_upper = triu(D @ A @ D, format='coo')
    n = A.shape[0]
    As_upper = csr_matrix((np.r_[As_upper.data, np.zeros(n)],
                           (np.r_[As_upper.row, np.arange(n)],
                            np.r_[As_upper.col, np.arange(n)])), shape=A.shape)
    if _pardiso_pool:
        solver = _pardiso_pool.pop()
    else:
        solver = pypardiso.PyPardisoSolver(mtype=-2)
    try:
        solver.factorize(As_upper)
    except Exception as e:
        _pardiso_release(solver)
        raise RuntimeError('PARDISO factorization failed: %s' % e)
    iparms = solver.get_iparms()

    def precond(r):
        return s*solver.solve(As_upper, np.ascontiguousarray(s*r))

    def solve_kT(b):
        x = np.zeros(b.shape[0], dtype=np.result_type(b, np.float64))
        bu = np.concatenate((np.zeros(m),
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
