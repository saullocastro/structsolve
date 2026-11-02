"""Linear static analysis and solution of linear systems

:func:`solve` solves a sparse linear system after removing the null rows and
columns, and :func:`static` is a linear static analysis based on it. For
analyses based on callables, linear or non-linear, see :class:`.Analysis`.

"""
import warnings
import weakref

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, triu
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


def _factorize(kT, linear_solver='splu'):
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

    Returns
    -------
    solve_kT : callable
        Function that returns the solution ``x`` of ``kT x = b``, with the
        attributes:

        - ``linear_solver``: the linear solver used, ``'splu'`` when
          ``'pardiso'`` fell back to it for an unsymmetric matrix;
        - ``det_sign``: the sign of the determinant of the reduced matrix,
          ``1`` or ``-1``;
        - ``num_negative_eigvals``: the number of negative eigenvalues of the
          reduced matrix with ``'pardiso'``, ``None`` with ``'splu'``;
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
    The memory of PARDISO is freed when ``solve_kT`` is deleted.

    """
    if linear_solver not in LINEAR_SOLVERS:
        raise ValueError('Invalid linear_solver: %s' % linear_solver)
    kT_red, used_cols = remove_null_cols(kT, silent=True)
    if linear_solver == 'pardiso':
        A = csr_matrix(kT_red)
        if _is_symmetric(A):
            return _factorize_pardiso(A, used_cols)
        warnings.warn("linear_solver='pardiso': the matrix is unsymmetric, "
                      "'splu' is used instead", RuntimeWarning, stacklevel=3)
    lu = splu(csc_matrix(kT_red))

    def solve_kT(b):
        x = np.zeros(b.shape[0], dtype=np.result_type(b, lu.U.dtype))
        x[used_cols] = lu.solve(b[used_cols])
        return x

    diagU = lu.U.diagonal()
    solve_kT.linear_solver = 'splu'
    solve_kT.det_sign = int(np.prod(np.sign(diagU))
                            *_permutation_sign(lu.perm_r)
                            *_permutation_sign(lu.perm_c))
    solve_kT.num_negative_eigvals = None
    solve_kT.num_perturbed_pivots = None

    return solve_kT


def _factorize_pardiso(A, used_cols):
    """Symmetric indefinite factorization of PARDISO of the reduced matrix
    ``A``, see :func:`._factorize`"""
    try:
        import pypardiso
    except ImportError:
        raise ImportError("linear_solver='pardiso' requires the package "
                          "pypardiso")
    d = np.abs(A.diagonal())
    d[d == 0] = 1.
    s = 1/np.sqrt(d)
    D = diags(s)
    As_upper = triu(D @ A @ D, format='csr')
    solver = pypardiso.PyPardisoSolver(mtype=-2)
    try:
        solver.factorize(As_upper)
    except Exception as e:
        solver.free_memory(everything=True)
        raise RuntimeError('PARDISO factorization failed: %s' % e)
    iparms = solver.get_iparms()

    def precond(r):
        return s*solver.solve(As_upper, np.ascontiguousarray(s*r))

    def solve_kT(b):
        x = np.zeros(b.shape[0], dtype=np.result_type(b, np.float64))
        bu = np.asarray(b[used_cols], dtype=np.float64)
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
        x[used_cols] = xu
        return x

    solve_kT.linear_solver = 'pardiso'
    solve_kT.num_negative_eigvals = int(iparms[23])
    solve_kT.num_perturbed_pivots = int(iparms[14])
    solve_kT.det_sign = -1 if solve_kT.num_negative_eigvals % 2 else 1
    weakref.finalize(solve_kT, solver.free_memory, everything=True)

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
