"""Linear static analysis and solution of linear systems

:func:`solve` solves a sparse linear system after removing the null rows and
columns, and :func:`static` is a linear static analysis based on it. For
analyses based on callables, linear or non-linear, see :class:`.Analysis`.

"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, splu

from .logger import msg
from .sparseutils import remove_null_cols


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
    is not null, the reduced system `[a_{uu}]\{x_u\} = \{b_u\}` is solved
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


def _factorize(kT):
    """LU factorization of the tangent stiffness matrix

    Null rows and columns are removed, the corresponding values of the
    solution are zero, as in :func:`.solve`. The factorization is reused by
    the non-linear solvers while the tangent stiffness matrix is not
    updated.

    Parameters
    ----------
    kT : sparse matrix
        Tangent stiffness matrix.

    Returns
    -------
    solve_kT : callable
        Function that returns the solution ``x`` of ``kT x = b``.

    Notes
    -----
    The reduced matrix is factorized once with SuperLU,
    :func:`scipy.sparse.linalg.splu` with its default options, i.e. a
    sparse LU factorization with row and column permutations, valid for
    unsymmetric matrices. Each call of ``solve_kT`` is then only a forward
    and a backward substitution, which is why the factorization is reused
    while the tangent stiffness matrix is not updated, e.g. in the
    modified Newton-Raphson method and in the two solutions of the bordered
    system of the arc-length methods.

    """
    kT_red, used_cols = remove_null_cols(kT, silent=True)
    lu = splu(csc_matrix(kT_red))

    def solve_kT(b):
        x = np.zeros(b.shape[0], dtype=np.result_type(b, lu.U.dtype))
        x[used_cols] = lu.solve(b[used_cols])
        return x

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
