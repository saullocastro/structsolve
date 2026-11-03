"""Newton-Raphson solver for non-linear static analyses

The module-level constants control the step size and the divergence checks,
and they are shared with the arc-length solvers of
:mod:`structsolve.arc_length`.

"""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from .logger import msg, warn
from .static import solve, _factorize, _count_negative_real
from . import callbacks
from .convergence import check_convergence, crisfield_weights, scaled_norm


#: Number of iterations of a step before the divergence and too-slow checks
#: are activated. A non-monotonic residual in the first iterations is normal,
#: especially in postbuckling
NUM_ITER_BEFORE_DIVERGENCE_CHECK = 5
#: A step diverged when its residual norm becomes larger than this factor
#: times the residual norm at the start of the step
DIVERGENCE_FACTOR = 1.e3
#: The convergence is too slow when the smallest residual norm of the last
#: ``TOO_SLOW_WINDOW`` iterations is not at least ``too_slow_TOL`` (relative)
#: smaller than the smallest residual norm of the previous iterations
TOO_SLOW_WINDOW = 5
#: With ``Analysis.early_divergence = True``: number of iterations of a step
#: before the early divergence check against the smallest residual norm
EARLY_DIVERGENCE_NUM_ITER = 3
#: With ``Analysis.early_divergence = True``: a step diverged when its
#: residual norm becomes larger than this factor times the smallest residual
#: norm of its previous iterations
EARLY_DIVERGENCE_FACTOR = 10.
#: With ``Analysis.early_divergence = True``: a step diverged when its
#: relative residual, Crisfield's `\zeta` or the relative residual of
#: ``convergence='relative'``, becomes larger than this value
EARLY_DIVERGENCE_MAX_RREL = 1.e3
#: Constant of the sufficient-decrease (Armijo) test of the line search
LINE_SEARCH_ALPHA = 1.e-4
#: Factor used to increase the load increment after a step that converged
#: without being cut
INC_GROWTH_FACTOR = 1.1111
#: Factor used to reduce the load increment when a step fails to converge
INC_CUT_FACTOR = 0.5


def _weights(an, kC0):
    r"""Weights of Crisfield's criterion from the linear stiffness matrix
    ``kC0``, see :func:`.convergence.crisfield_weights`, or ``None`` with
    ``an.convergence = 'relative'``"""
    if an.convergence == 'crisfield':
        return crisfield_weights(kC0)
    return None


def _abs_matrix(kC0):
    r"""Elementwise absolute value `|K_{C0}|` of the linear stiffness matrix
    ``kC0``, sparse, for the round-off floor of the convergence criteria,
    see :func:`.convergence.roundoff_floor`"""
    return abs(csr_matrix(kC0))


def _constraint_matrix(an):
    r"""The matrix `\boldsymbol{C}` of ``an.constraints``, sparse, of shape
    ``(n, m)``, a 1-D array being one constraint, or ``None``"""
    if an.constraints is None:
        return None
    C = np.asarray(an.constraints) if not hasattr(an.constraints, 'tocsc') \
        else an.constraints
    if getattr(C, 'ndim', 2) == 1:
        C = np.asarray(C)[:, None]
    return csc_matrix(C)


def _cache(an):
    r"""The cache of the factorizations of the tangent stiffness matrix of
    the running analysis, see :func:`.static._factorize`, ``None`` outside
    :meth:`.Analysis.static`"""
    return getattr(an, '_factorization_cache', None)


def _constraint_values(an):
    r"""The values `\boldsymbol{d}` of ``an.constraint_values``, of the
    constraints `\boldsymbol{C}^\top \boldsymbol{c} = \lambda \boldsymbol{d}`,
    validated, ``None`` when not given

    Raises
    ------
    ValueError
        When ``an.constraints`` is ``None``, or when the values are not a
        finite vector of one value per constraint.

    """
    if an.constraint_values is None:
        return None
    C = _constraint_matrix(an)
    if C is None:
        raise ValueError('Analysis.constraint_values requires '
                         'Analysis.constraints')
    d = np.atleast_1d(np.asarray(an.constraint_values, dtype=float))
    if d.shape != (C.shape[1],):
        raise ValueError('Analysis.constraint_values must have the shape '
                         '(%d,), one value per constraint' % C.shape[1])
    if not np.all(np.isfinite(d)):
        raise ValueError('Analysis.constraint_values must be finite')
    return d


def _linear_solve(an, kC0, b, d=None):
    r"""Linear solution of ``kC0 x = b`` with ``an.linear_solver`` and the
    constraints of ``an.constraints``, of values ``d``, see
    :func:`.static._factorize`, or with :func:`.static.solve` with
    ``'splu'`` and without constraints"""
    if an.constraints is None and an.linear_solver == 'splu':
        return solve(kC0, b, silent=True)
    return _factorize(kC0, linear_solver=an.linear_solver,
                      constraints=_constraint_matrix(an))(b, d)


def _residual_projector(an, w):
    r"""Projection of the residual that removes the reactions of the
    constraints ``an.constraints``, ``None`` without constraints

    With the constraints `\boldsymbol{C}^\top \boldsymbol{c} = \boldsymbol{0}`,
    the equilibrium is `\boldsymbol{R} + \boldsymbol{C} \boldsymbol{\mu} =
    \boldsymbol{0}`, `\boldsymbol{\mu}` the reactions of the constraints. The
    convergence criteria and the line searches measure

    .. math::

        \boldsymbol{R}_p = \boldsymbol{R} - \boldsymbol{C} (\boldsymbol{C}^\top
        \boldsymbol{W} \boldsymbol{C})^{+} \boldsymbol{C}^\top \boldsymbol{W}
        \boldsymbol{R}

    the residual minus the reactions that minimize its norm, with the
    weights `\boldsymbol{W} = \mathrm{diag}(\boldsymbol{w})` of Crisfield's
    criterion, `\boldsymbol{W} = \boldsymbol{I}` with ``w=None``, i.e. the
    projection of `\boldsymbol{R}` orthogonal to the columns of
    `\boldsymbol{C}` in the inner product of the criterion, `^{+}` the
    pseudo-inverse, for constraints that are not independent.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object.
    w : array-like or None
        Weights of Crisfield's criterion, see :func:`._weights`.

    Returns
    -------
    proj : callable or None
        ``proj(R)`` returns `\boldsymbol{R}_p`.

    """
    C = _constraint_matrix(an)
    if C is None:
        return None
    WC = C if w is None else csc_matrix(C.multiply(np.asarray(w)[:, None]))
    Ginv = np.linalg.pinv((C.T @ WC).toarray())

    def proj(R):
        return R - C @ (Ginv @ (WC.T @ R))

    return proj


def _norm(v, w):
    r"""Norm of the convergence criterion: `\|v\|_{\boldsymbol{d}}` with the
    weights ``w`` of Crisfield's criterion, the Euclidean norm when ``w`` is
    ``None``"""
    if w is None:
        return float(np.linalg.norm(v))
    return scaled_norm(v, w)


def _log_iteration(an, iteration, R, zeta, silent, lbd=None,
                   roundoff=False):
    r"""Message of one equilibrium iteration, with `\zeta` of Crisfield's
    criterion or the relative residual of the relative criterion, marked
    when the iteration converged by the round-off floor only, see
    :func:`.convergence.check_convergence`"""
    name = 'zeta' if an.convergence == 'crisfield' else 'relative ||R||'
    load = '' if lbd is None else ', lbd %1.5f' % lbd
    floor = ', converged at the round-off floor' if roundoff else ''
    msg('Iteration: %d%s, max(|R|) = %1.3e, %s = %1.3e%s'
        % (iteration, load, np.abs(R).max(), name, zeta, floor), level=2,
        silent=silent)


def _check_convergence(an, R, fint, fext_total, fext_ref=None, q_ref=None,
                       w=None, Kc_abs=None, full_output=False):
    r"""Check the convergence of the residual force vector

    With ``an.convergence = 'crisfield'``, the default, the convergence is
    achieved when Crisfield's `\zeta \le` ``an.relTOL``, the residual and
    the reference force vectors being measured with the scaled norm of the
    weights ``w`` over the free degrees of freedom. With ``an.convergence =
    'relative'`` it is achieved when either:

    - ``||R|| <= relTOL*max(||fext_total||, ||fint||, ||fext_ref||,
      ||q_ref||)``, using the Euclidean norm, if ``an.relTOL`` is not
      ``None``
    - ``max(|R|) <= absTOL``, if ``an.absTOL`` is not ``None``

    ``an.absTOL`` is also accepted as an additional criterion with
    ``'crisfield'``, and, with both criteria, the round-off floor of
    :func:`.convergence.roundoff_floor` when ``Kc_abs`` is given. See
    :func:`.convergence.check_convergence`.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the attributes ``convergence``, ``relTOL`` and
        ``absTOL``.
    R : array-like
        Residual force vector.
    fint : array-like
        Internal force vector, of the same evaluation as ``R``.
    fext_total : array-like
        External force vector at the current load factor.
    fext_ref : array-like or None, optional
        Additional reference force vector, e.g. the external force vector of
        the current increment of the arc-length solvers.
    q_ref : array-like or None, optional
        Additional reference force vector, the load factor times
        `-\partial \boldsymbol{R}/\partial \lambda` when ``calc_fext`` accepts ``lbd``,
        see :mod:`.callbacks`.
    w : array-like or None, optional
        Weights of Crisfield's criterion, see
        :func:`.convergence.crisfield_weights`, required with
        ``an.convergence = 'crisfield'``.
    Kc_abs : array-like or None, optional
        `|K_{C0}| |c|` of the round-off floor, see
        :func:`.convergence.check_convergence`; ``None`` disables it.
    full_output : bool, optional
        Whether to also return ``roundoff``.

    Returns
    -------
    converged : bool
        Whether the convergence criterion is satisfied.
    Rnorm : float
        Norm of the residual of the criterion, `\|\boldsymbol{R}\|_{\boldsymbol{d}}` with
        ``'crisfield'``, the Euclidean norm with ``'relative'``, used by the
        divergence checks and the line search.
    Rrel : float
        Relative residual, Crisfield's `\zeta` with ``'crisfield'``,
        ``||R||/max(||fext_total||, ||fint||, ||fext_ref||, ||q_ref||)``
        with ``'relative'``.
    roundoff : bool
        Only with ``full_output=True``: whether the convergence is achieved
        by the round-off floor only.

    Raises
    ------
    ValueError
        When ``an.convergence`` is not ``'crisfield'`` or ``'relative'``,
        when ``an.relTOL`` and ``an.absTOL`` are both ``None``, or when
        ``w`` is missing with ``'crisfield'``.

    Notes
    -----
    Crisfield's criterion, see :mod:`.convergence`, does not depend on the
    units of each degree of freedom, e.g. translations and rotations,
    whereas the Euclidean norm of the relative criterion adds the squares
    of forces and of moments. The reference ``fext_ref`` is useful when the
    load factor is close to zero, where ``fext_total`` and ``fint`` are
    small. The reference ``q_ref`` is needed under a prescribed displacement
    alone, where the external and the internal force vectors of the reduced
    problem both vanish at equilibrium and cannot be the reference.

    """
    return check_convergence(an, R, fint, fext_total, fext_ref=fext_ref,
                             q_ref=q_ref, w=w, Kc_abs=Kc_abs,
                             full_output=full_output)


def _check_divergence(an, Rnorm, Rnorms, iteration, silent=False, Rrel=None):
    r"""Check if the iterations of a step must be stopped without convergence


    The iterations are stopped when the residual is not finite, when
    ``an.maxNumIter`` corrections were performed, and, after
    ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` iterations, when the residual
    diverges or when the convergence is too slow. With
    ``an.early_divergence = True`` a diverging step is also stopped earlier.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the convergence criteria.
    Rnorm : float
        Norm of the residual at the current iteration, the norm of the
        convergence criterion, see :func:`._check_convergence`.
    Rnorms : list
        Norms of the residual at the previous iterations of the step, the
        current norm is appended to it.
    iteration : int
        Current iteration, starting at 1.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    Rrel : float or None, optional
        Relative residual of the current iteration, see
        :func:`._check_convergence`, used by the early divergence check.

    Returns
    -------
    stop : bool
        Whether the iterations must be stopped.

    Notes
    -----
    With `r_i` the residual norm at the iteration `i` of the step, `r_1`
    being the residual of the predictor, the iterations are stopped at the
    iteration `j` when:

    - `r_j` is not finite;
    - `j >` ``an.maxNumIter``, i.e. ``an.maxNumIter`` corrections were done;
    - `j >` ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` and `r_j >`
      ``DIVERGENCE_FACTOR`` `r_1` (divergence);
    - `j >` ``NUM_ITER_BEFORE_DIVERGENCE_CHECK``, `j > W` and
      `\min(r_{j-W+1}, ..., r_j) > (1 -` ``an.too_slow_TOL``
      `) \min(r_1, ..., r_{j-W})`, with `W` = ``TOO_SLOW_WINDOW`` (too slow
      convergence).

    The divergence and too-slow checks are delayed because a non-monotonic
    residual in the first iterations is normal, especially in
    postbuckling. The checks only compare norms of the same step with each
    other, and with the scaled norm of Crisfield's criterion they do not
    depend on the units of the degrees of freedom.

    With ``an.early_divergence = True``, a diverging step is stopped
    without waiting for ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` iterations,
    such that its load or arc-length increment is cut before the
    iterations are spent, when:

    - the relative residual ``Rrel`` is larger than
      ``EARLY_DIVERGENCE_MAX_RREL``, at any iteration;
    - `j >` ``EARLY_DIVERGENCE_NUM_ITER`` and `r_j >`
      ``EARLY_DIVERGENCE_FACTOR`` `\min(r_1, ..., r_{j-1})`.

    These are the criteria with which the Riks analyses of a tow-steered
    cylinder of the DCSE website (2026) gave up the increments that would
    not converge beyond the limit point.

    """
    if not np.isfinite(Rnorm):
        warn('Diverged - residual is not finite', level=2, silent=silent)
        return True
    if iteration > an.maxNumIter:
        warn('Maximum number of iterations achieved!', level=2, silent=silent)
        return True
    if an.early_divergence:
        if Rrel is not None and Rrel > EARLY_DIVERGENCE_MAX_RREL:
            warn('Diverged - relative residual larger than %1.0e'
                 % EARLY_DIVERGENCE_MAX_RREL, level=2, silent=silent)
            return True
        if (iteration > EARLY_DIVERGENCE_NUM_ITER and
                Rnorm > EARLY_DIVERGENCE_FACTOR*min(Rnorms)):
            warn('Diverged - residual increased %1.0f times its minimum in the step'
                 % EARLY_DIVERGENCE_FACTOR, level=2, silent=silent)
            return True
    Rnorms.append(Rnorm)
    if iteration > NUM_ITER_BEFORE_DIVERGENCE_CHECK:
        if Rnorm > DIVERGENCE_FACTOR*Rnorms[0]:
            warn('Diverged - residual increased %1.0e times since the start of the step'
                 % DIVERGENCE_FACTOR, level=2, silent=silent)
            return True
        if (len(Rnorms) > TOO_SLOW_WINDOW and
                min(Rnorms[-TOO_SLOW_WINDOW:]) >
                (1 - an.too_slow_TOL)*min(Rnorms[:-TOO_SLOW_WINDOW])):
            warn('Diverged - convergence too slow', level=2, silent=silent)
            return True
    return False


def _converged_state(an, c, lbd, silent=False):
    r"""Factorization of the tangent stiffness matrix at a converged state,
    recording its inertia

    Used with ``an.inertia = True``. The tangent stiffness matrix at ``c``
    and the load factor ``lbd`` is factorized with ``an.linear_solver``, see
    :func:`.static._factorize`, and the number of its negative eigenvalues,
    the sign of its determinant and the number of perturbed pivots are
    appended to ``an.num_negative_eigvals``, ``an.det_signs`` and
    ``an.num_perturbed_pivots``; for a singular matrix ``None``, ``0`` and
    ``None``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object.
    c : array-like
        Converged solution.
    lbd : float
        Converged load factor.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.

    Returns
    -------
    solve_kT : callable or None
        The factorization, see :func:`.static._factorize`, ``None`` when the
        matrix is singular.

    """
    solve_kT = _factorize_state(an, c, lbd, silent=silent)
    _record_inertia(an, solve_kT, silent=silent)
    return solve_kT


def _factorize_state(an, c, lbd, silent=False):
    r"""Factorization of the tangent stiffness matrix at ``c`` and ``lbd``
    with ``an.linear_solver`` and ``an.constraints``, see
    :func:`.static._factorize`, ``None`` when it is singular; with
    ``an.inertia_num_eigvals``, the count of the negative real eigenvalues
    of a SuperLU factorization, see :func:`._eigvals_inertia`"""
    try:
        solve_kT = _factorize(callbacks.calc_kT(an, c, lbd),
                              linear_solver=an.linear_solver,
                              constraints=_constraint_matrix(an),
                              cache=_cache(an))
    except RuntimeError:
        warn('Singular tangent stiffness matrix at the converged state',
             level=2, silent=silent)
        return None
    if an.inertia_num_eigvals is not None:
        _eigvals_inertia(solve_kT, an.inertia_num_eigvals, silent=silent)
    return solve_kT


def _eigvals_inertia(solve_kT, k, silent=False):
    r"""The number of negative real eigenvalues among the ``k`` of smallest
    magnitude of the SuperLU factorization ``solve_kT``, set as its
    ``num_negative_eigvals``, see ``Analysis.inertia_num_eigvals``

    Nothing is done when ``solve_kT`` already has a count, the exact one of
    PARDISO. The eigenvalues are computed by
    :func:`.static._smallest_eigvals` with the LU factorization of
    ``solve_kT``, on the degrees of freedom whose row or column is not
    null and on the space of the constraints. The attributes set on
    ``solve_kT`` are:

    - ``num_negative_eigvals``: the number of real eigenvalues that are
      negative, ``None`` when ARPACK did not converge;
    - ``num_complex_negative``: the number of complex eigenvalues with a
      negative real part, among the ``k``;
    - ``inertia_num_eigvals``: ``k``;
    - ``inertia_complete``: ``False`` when the parity of the count differs
      from the sign of the determinant, i.e. an odd number of negative real
      eigenvalues lies beyond the ``k`` of smallest magnitude, with a
      warning.

    Returns
    -------
    nneg : int or None
        ``solve_kT.num_negative_eigvals``.

    """
    if solve_kT.num_negative_eigvals is not None:
        return solve_kT.num_negative_eigvals
    if not hasattr(solve_kT, 'smallest_eigvals'):
        return None
    solve_kT.inertia_num_eigvals = k
    solve_kT.num_complex_negative = None
    solve_kT.inertia_complete = None
    lam = solve_kT.smallest_eigvals(k)
    if lam is None:
        warn('ARPACK did not converge, the number of negative eigenvalues '
             'is not known', level=2, silent=silent)
        return None
    nneg, ncomplex = _count_negative_real(lam)
    solve_kT.num_negative_eigvals = nneg
    solve_kT.num_complex_negative = ncomplex
    # complex pairs have a positive product: the sign of the determinant is
    # (-1)**(number of negative real eigenvalues)
    solve_kT.inertia_complete = (-1)**nneg == solve_kT.det_sign
    if not solve_kT.inertia_complete:
        warn('An odd number of negative real eigenvalues lies beyond the '
             '%d of smallest magnitude: increase Analysis.inertia_num_eigvals'
             % k, level=2, silent=silent)
    return nneg


def _record_inertia(an, solve_kT, silent=False):
    r"""Append the inertia of the factorization ``solve_kT`` of a converged
    state to ``an.num_negative_eigvals``, ``an.det_signs`` and
    ``an.num_perturbed_pivots``, see :func:`._converged_state`"""
    if solve_kT is None:
        an.num_negative_eigvals.append(None)
        an.det_signs.append(0)
        an.num_perturbed_pivots.append(None)
        return
    nneg = solve_kT.num_negative_eigvals
    an.num_negative_eigvals.append(nneg)
    an.det_signs.append(solve_kT.det_sign)
    an.num_perturbed_pivots.append(solve_kT.num_perturbed_pivots)
    if nneg is None:
        msg('Tangent stiffness matrix: sign of the determinant %+d'
            % solve_kT.det_sign, level=2, silent=silent)
    elif solve_kT.num_perturbed_pivots is None:
        msg('Tangent stiffness matrix: %d negative real eigenvalues and %d '
            'complex with a negative real part among the %d of smallest '
            'magnitude, sign of the determinant %+d'
            % (nneg, solve_kT.num_complex_negative,
               solve_kT.inertia_num_eigvals, solve_kT.det_sign), level=2,
            silent=silent)
    else:
        msg('Tangent stiffness matrix: %d negative eigenvalues, %d perturbed pivots'
            % (nneg, solve_kT.num_perturbed_pivots), level=2, silent=silent)


def _check_max_inertia_jump(an):
    r"""Validate ``an.inertia_num_eigvals``, which requires ``an.inertia``,
    and ``an.max_inertia_jump``, which requires ``an.inertia`` and the
    counts of negative eigenvalues of ``an.linear_solver='pardiso'`` or of
    ``an.inertia_num_eigvals``"""
    k = an.inertia_num_eigvals
    if k is not None:
        if not an.inertia:
            raise ValueError('Analysis.inertia_num_eigvals requires '
                             'Analysis.inertia = True')
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError('Analysis.inertia_num_eigvals must be a '
                             'positive integer')
    max_jump = an.max_inertia_jump
    if max_jump is None:
        return None
    if not an.inertia or (an.linear_solver != 'pardiso' and k is None):
        raise ValueError("Analysis.max_inertia_jump requires Analysis.inertia "
                         "= True and Analysis.linear_solver = 'pardiso' or "
                         "Analysis.inertia_num_eigvals")
    if int(max_jump) != max_jump or max_jump < 1:
        raise ValueError('Analysis.max_inertia_jump must be a positive '
                         'integer')
    return int(max_jump)


def _inertia_jump(solve_last, solve_new, max_jump):
    r"""Whether the number of negative eigenvalues changed by more than
    ``max_jump`` between the factorizations ``solve_last`` and ``solve_new``
    of two converged states, ``False`` when a count is not known

    The count of ``solve_last`` is computed when it has none but
    ``solve_new`` has one of ``an.inertia_num_eigvals``, e.g. for the
    factorization at the origin of the arc-length solvers, made before
    the count was asked for, see :func:`._eigvals_inertia`."""
    if max_jump is None or solve_last is None or solve_new is None:
        return False
    n1 = solve_new.num_negative_eigvals
    n0 = solve_last.num_negative_eigvals
    k = getattr(solve_new, 'inertia_num_eigvals', None)
    if n0 is None and n1 is not None and k is not None:
        n0 = _eigvals_inertia(solve_last, k, silent=True)
    if n0 is None or n1 is None:
        return False
    return abs(n1 - n0) > max_jump


def _restart_state(an, method):
    r"""``an.restart_state`` validated for the solver ``method``, ``None``
    when not given"""
    state = an.restart_state
    if state is None:
        return None
    if state.get('method') != method:
        raise ValueError('Analysis.restart_state was saved by the method %r, '
                         'not %r' % (state.get('method'), method))
    return state


def _stop(an, silent=False):
    r"""Whether ``an.stop_criterion``, when given, stops the analysis after
    the last converged increment, see :mod:`.stopping`"""
    if an.stop_criterion is not None and an.stop_criterion(an):
        msg('Analysis stopped by the stop criterion at load factor %1.5f'
            % an.increments[-1], level=1, silent=silent)
        return True
    return False


def _NR_iterations(an, c, dc, total, fext, kT0=None, silent=False,
                   lbd_dependent=False, q=None, w=None, K0_abs=None,
                   proj=None):
    r"""Newton-Raphson iterations at a fixed load factor

    Corrects the predictor ``dc`` at the load factor ``total`` until the
    convergence criteria of :func:`._check_convergence` are satisfied, or
    until the iterations are stopped by :func:`._check_divergence` or by a
    singular tangent stiffness matrix. The tangent stiffness matrix is
    ``calc_kC + calc_kG`` at the current iterate, see
    :func:`.callbacks.calc_kT`; its updates and the line search are
    controlled by ``an.modified_NR``, ``an.compute_every_n``,
    ``an.line_search`` and ``an.max_iter_line_search``, the early divergence
    check by ``an.early_divergence``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object.
    c : array-like
        Solution at the last converged state.
    dc : array-like
        Predictor of the solution increment.
    total : float
        Load factor.
    fext : array-like
        External force vector for a unit load factor.
    kT0 : sparse matrix or None, optional
        Tangent stiffness matrix used in the first iteration when
        ``an.modified_NR=True``. If ``None`` it is calculated at ``c + dc``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    lbd_dependent : bool, optional
        Whether ``calc_fext`` accepts ``lbd``, see :mod:`.callbacks`;
        ``total*q``, with `\boldsymbol{q} = -\partial \boldsymbol{R}/\partial \lambda`, is then an
        additional reference of the convergence criterion, see
        :func:`._check_convergence`. `\boldsymbol{q}` is evaluated together with the
        tangent stiffness matrix, at the same iterate, and the last one
        evaluated is the reference until the next update of the tangent.
    q : array-like or None, optional
        The last `\boldsymbol{q}` evaluated before these iterations, the reference
        until the first update of the tangent stiffness matrix, used only
        when ``lbd_dependent=True``.
    w : array-like or None, optional
        Weights of Crisfield's criterion, see
        :func:`.convergence.crisfield_weights`, computed from ``calc_kC()``
        when ``None`` and ``an.convergence = 'crisfield'``.
    K0_abs : sparse matrix or None, optional
        `|K_{C0}|` of the round-off floor of the convergence criteria, see
        :func:`.convergence.check_convergence`, computed from ``calc_kC()``
        when ``None``.
    proj : callable or None, optional
        Projection of the residual that removes the reactions of
        ``an.constraints``, see :func:`._residual_projector`, computed when
        ``None`` and ``an.constraints`` is given.

    Returns
    -------
    converged : bool
        Whether the iterations converged.
    dc : array-like
        Solution increment, such that ``c + dc`` is the solution.
    q : array-like or None
        The last `\boldsymbol{q}` evaluated, ``None`` when ``lbd_dependent=False``.

    Notes
    -----
    At each iteration, with the load factor `\lambda` = ``total`` fixed,

    .. math::

        \boldsymbol{R} = \boldsymbol{F}_{int}(\boldsymbol{c} + \Delta \boldsymbol{c}, \lambda) - \lambda \boldsymbol{F}_{ext},
        \qquad \boldsymbol{K}_T \delta \boldsymbol{c} = -\boldsymbol{R}, \qquad
        \Delta \boldsymbol{c} \leftarrow \Delta \boldsymbol{c} + \eta \delta \boldsymbol{c}

    With full Newton-Raphson, `\boldsymbol{K}_T` is computed and factorized at every
    iteration at `c + \Delta c`, and the convergence is quadratic when it is
    the exact Jacobian `\partial \boldsymbol{R}/\partial \boldsymbol{c}` ([1]_, Section 1.2.2,
    p. 9). With
    ``an.modified_NR = True``, `\boldsymbol{K}_T` is computed at the first iteration,
    or given as ``kT0``, and then every ``an.compute_every_n`` iterations,
    its LU factorization being reused in the other iterations.

    `\eta = 1` without line search. With ``an.line_search = True``, with
    `\phi(\eta) = \|R(c + \Delta c + \eta \, \delta c)\|^2`, the step `\eta`
    is accepted when it satisfies the sufficient-decrease (Armijo) test

    .. math::

        \phi(\eta) \le (1 - 2 \alpha \eta) \phi(0)

    with `\alpha` = ``LINE_SEARCH_ALPHA``, which assumes `\phi'(0) =
    -2\phi(0)`, exact for a Newton direction, i.e. when `\boldsymbol{K}_T` is the exact
    Jacobian at `c + \Delta c`; with ``an.modified_NR = True`` it is an
    approximation. The full step `\eta = 1` is tried first. Otherwise `\eta`
    is replaced by the minimizer `\phi(0)/A` of the quadratic interpolation
    of `\phi(0)`, `\phi'(0)` and `\phi(\eta)`, with `A = (\phi(\eta) -
    \phi(0) + 2\phi(0)\eta)/\eta^2`, or by `\eta/2` when `A` is not
    finite or not positive, safeguarded in `[0.1 \eta, 0.5 \eta]`. After
    ``an.max_iter_line_search`` reductions the last `\eta` is accepted. The
    internal force vector of the accepted point is reused in the residual
    of the next iteration.

    The line search is adapted from the backtracking line search of [2]_
    (Section 6.3.2, Algorithm 6.3.5, pp. 126-129, and Algorithm A6.3.1,
    pp. 325-327), applied to nonlinear equations with the merit function
    `f = \frac{1}{2}\|D_F F\|^2` ([2]_, Section 6.5, Eq. (6.5.3), p. 147,
    and Algorithm NEFN, p. 303), whose directional derivative along the
    Newton step is `-F^\top F = -2f` (p. 148, with `D_F` the identity). With
    `F = R`, whose Jacobian is `\boldsymbol{K}_T`, and `\phi = 2f`, `D_F` being
    the identity with the Euclidean norm or `\mathrm{diag}(\sqrt{w_j})` with Crisfield's criterion, the test
    above is their condition (6.3.3) (p. 118) with their `\alpha = 10^{-4}`
    (p. 126), and the first reduction is their quadratic backtrack,
    Eq. (6.3.17) (p. 127). It differs from Algorithm A6.3.1 in that: the
    quadratic interpolation through the last trial `\eta` is used at every
    reduction, whereas [2]_ uses, from the second backtrack on, the cubic
    interpolation of `f(0)`, `f'(0)` and the last two trial values,
    Eq. (6.3.18) (p. 128); the bound `0.5 \eta` is also imposed at the
    first reduction, where [2]_ imposes only the lower bound `0.1`, the
    quadratic backtrack being then smaller than `1/(2(1 - \alpha))`
    (p. 127; Algorithm A6.3.1, steps 10.3c.1E.4 and 10.3c.4); and [2]_ has
    no maximum number of reductions, but limits the length of the step to
    ``maxstep`` and stops when the step length becomes smaller than a
    minimum, returning the current point.

    References
    ----------
    .. [1] Crisfield, M.A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.
    .. [2] Dennis, J.E., Jr., Schnabel, R.B., Numerical Methods for
       Unconstrained Optimization and Nonlinear Equations, SIAM,
       Philadelphia, 1996 (first published by Prentice-Hall, Englewood
       Cliffs, NJ, 1983), pages of the SIAM edition.
       https://doi.org/10.1137/1.9781611971200

    """

    modified_NR = an.modified_NR
    if K0_abs is None or (w is None and an.convergence == 'crisfield'):
        kC0 = an.calc_kC(silent=True)
        if w is None:
            w = _weights(an, kC0)
        if K0_abs is None:
            K0_abs = _abs_matrix(kC0)
    if proj is None:
        proj = _residual_projector(an, w)
    C = _constraint_matrix(an)

    def residual(fint):
        R = fint - fext_total
        return R if proj is None else proj(R)

    fext_total = total*fext
    if not lbd_dependent:
        q = None
    q_ref = None
    Rnorms = []
    iter_kT = 0
    solve_kT = None
    fint = None

    iteration = 0
    while True:
        iteration += 1
        if fint is None:
            fint = callbacks.calc_fint(an, c + dc, total)
        R = residual(fint)
        if q is not None:
            q_ref = total*q

        conv, Rnorm, Rrel, roundoff = _check_convergence(an, R, fint,
                fext_total, q_ref=q_ref, w=w, Kc_abs=K0_abs @ np.abs(c + dc),
                full_output=True)
        _log_iteration(an, iteration, R, Rrel, silent, roundoff=roundoff)
        if conv:
            return True, dc, q
        if _check_divergence(an, Rnorm, Rnorms, iteration, silent=silent,
                             Rrel=Rrel):
            return False, dc, q

        # the factorization of the tangent stiffness matrix is reused while
        # the matrix is not updated
        try:
            if modified_NR and iteration > 1 and iter_kT < an.compute_every_n:
                iter_kT += 1
            elif modified_NR and iteration == 1 and kT0 is not None:
                solve_kT = _factorize(kT0, linear_solver=an.linear_solver,
                                      constraints=C, cache=_cache(an))
                iter_kT = 1
            else:
                solve_kT = _factorize(callbacks.calc_kT(an, c + dc, total),
                                      linear_solver=an.linear_solver,
                                      constraints=C, cache=_cache(an))
                iter_kT = 1
                if lbd_dependent:
                    # -dR/dlbd at the iterate of the tangent stiffness matrix
                    q = callbacks.calc_load_vector(an, c + dc, fext, total)
        except RuntimeError:
            warn('Singular tangent stiffness matrix', level=2, silent=silent)
            return False, dc, q

        varc = solve_kT(-R)

        eta = 1.
        fint = None
        if an.line_search:
            # sufficient decrease of phi(eta) = ||R(eta)||**2, assuming
            # phi'(0) = -2*phi(0), exact for a Newton direction
            fint = callbacks.calc_fint(an, c + dc + varc, total)
            phi0 = Rnorm**2
            phi1 = _norm(residual(fint), w)**2
            iter_line_search = 0
            while not phi1 <= (1 - 2*LINE_SEARCH_ALPHA*eta)*phi0:
                if iter_line_search == an.max_iter_line_search:
                    warn('Line-search: maximum number of iterations achieved',
                         level=3, silent=silent)
                    break
                iter_line_search += 1
                # minimum of the quadratic interpolation of phi, safeguarded
                A = (phi1 - phi0 + 2*phi0*eta)/eta**2
                eta_new = phi0/A if (np.isfinite(A) and A > 0) else 0.5*eta
                eta = min(max(eta_new, 0.1*eta), 0.5*eta)
                fint = callbacks.calc_fint(an, c + dc + eta*varc, total)
                phi1 = _norm(residual(fint), w)**2
            if eta != 1.:
                msg('Line-search: eta = %1.5f' % eta, level=3, silent=silent)
        dc = dc + eta*varc


def _solver_NR(an, silent=False, initialInc=None):
    r"""Newton-Raphson solver with load control

    Used by :meth:`.Analysis.static` when ``NL_method='NR'``. The load factor
    goes from zero to one in increments; at each increment the equilibrium
    of the internal force vector with the external force vector scaled by
    the load factor is solved with the iterations of :func:`._NR_iterations`,
    starting from a predictor extrapolated from the previous converged
    increment, and the converged load factors and solutions are appended to
    ``an.increments`` and ``an.cs``.

    The variant of the iterations is selected by ``an.modified_NR``,
    ``an.compute_every_n``, ``an.kT_initial_state``, ``an.line_search`` and
    ``an.max_iter_line_search``. See :func:`._check_convergence` for the
    convergence criteria. A step that does not converge within
    ``an.maxNumIter`` iterations, that diverges or that converges too slowly
    is repeated with the load increment multiplied by ``INC_CUT_FACTOR``.
    The divergence and too-slow checks are only activated after
    ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` iterations, see
    :func:`._check_divergence`.

    After a step that converged without being cut, the load increment is
    multiplied by ``INC_GROWTH_FACTOR``, limited by ``an.maxInc``. The last
    step is adjusted such that the analysis finishes at a load factor of
    exactly ``an.final_load_factor``, ``1.0`` by default. The analysis stops
    earlier, keeping the converged increments, when the load increment
    becomes smaller than ``an.minInc`` or when ``an.stop_criterion(an)``
    returns ``True`` after a converged step, see :mod:`.stopping`.

    The callables of ``an`` that declare the optional keyword arguments
    ``inc``, ``c`` or ``lbd`` receive the current load factor and
    configuration, see :mod:`.callbacks`.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object, the converged load factors and solutions are appended
        to ``an.increments`` and ``an.cs``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    initialInc : float or None, optional
        Initial load increment, ``an.initialInc`` is used if ``None``.

    Notes
    -----
    The load factor `\lambda` goes from zero to one in increments, solving
    at each increment

    .. math::

        \boldsymbol{R} = \boldsymbol{F}_{int}(\boldsymbol{c}) - \lambda \boldsymbol{F}_{ext} = \boldsymbol{0}

    Each increment starts from a predictor that extrapolates the previous
    converged increment of the solution, scaled by the ratio between the new
    and the previous load increments. The first increment uses the linear
    solution as predictor: `\Delta\lambda \boldsymbol{K}_{C0}^{-1} \boldsymbol{F}_{ext}`, with
    `\boldsymbol{K}_{C0}` the linear stiffness matrix ``calc_kC()``.

    The callables that accept the keyword argument ``inc`` receive the load
    factor `\lambda` of the step, see :mod:`.callbacks`, which allows
    configuration-dependent loads such as follower pressures.

    When ``calc_fext`` accepts the keyword argument ``lbd``, e.g. for
    prescribed displacements imposed exactly, see :mod:`.callbacks`, the
    load vector `\boldsymbol{q} = -\partial \boldsymbol{R}/\partial \lambda` at `\boldsymbol{c} = \boldsymbol{0}`,
    `\lambda = 0` gives the predictor of the first increment, its linear
    solution `\boldsymbol{K}_{C0}^{-1} \boldsymbol{q}`, instead of the linear solution of the
    reference vector, which is null under a prescribed displacement alone,
    and `\lambda \|q\|` is an additional reference of the relative
    convergence criterion. `\boldsymbol{q}` is evaluated together with the tangent
    stiffness matrix, at the same iterate, i.e. ``calc_fext`` is called once
    per update of the tangent stiffness matrix, and the last `\boldsymbol{q}`
    evaluated is the reference of the next convergence checks, also in the
    next step.

    By default full Newton-Raphson is used, i.e. the tangent stiffness matrix
    ``kT = kC + kG`` is rebuilt at every iteration, which gives quadratic
    convergence when ``kT`` is the exact Jacobian of ``fint``. With
    ``an.modified_NR = True`` the tangent stiffness matrix is rebuilt only at
    the first iteration of each step and every ``an.compute_every_n``
    iterations, and its LU factorization is reused in the other
    iterations, at the cost of a linear convergence rate; at the first step
    with ``an.kT_initial_state = False``, the linear constitutive stiffness
    matrix `\boldsymbol{K}_{C0}` is used instead of the tangent at the first iteration.

    With ``an.line_search = True``, the full Newton step (``eta = 1``) is tried
    first and it is accepted if it satisfies a sufficient-decrease test on the
    residual norm. Otherwise ``eta`` is reduced by backtracking, see
    :func:`._NR_iterations`.

    The full and modified Newton-Raphson methods, combined with incremental
    predictors, are described in [1]_ (Sections 1.2.2 and 1.2.3,
    pp. 8-13), with the quadratic and the linear convergence rates.

    The step that would end within ``max(an.minInc, 1e-12)`` of the load
    factor 1.0, or beyond it, is replaced by the step to exactly 1.0.

    With ``an.constraint_values``, the values `\boldsymbol{d}` of the
    constraints `\boldsymbol{C}^\top \boldsymbol{c} = \lambda \boldsymbol{d}`
    of ``an.constraints``, the linear solution of the first predictor is
    that of the bordered system with the values `\boldsymbol{d}`, see
    :func:`.static._factorize`, such that `\boldsymbol{C}^\top \Delta
    \boldsymbol{c} = \Delta\lambda \boldsymbol{d}` for every predictor,
    scaled from the previous converged increment, and the corrections keep
    it, their border being null. A prescribed displacement then needs no
    external load: the reactions of the constraints, part of the internal
    force vector, are the reference of the convergence criterion.

    References
    ----------
    .. [1] Crisfield, M.A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.

    """
    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Newton-Raphson solver', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)

    if initialInc is None:
        initialInc = an.initialInc
    target = an.final_load_factor
    if target is None:
        raise ValueError("Analysis.final_load_factor = None is not valid "
                         "with NL_method='NR'")
    inc = min(initialInc, target)
    min_last_inc = max(an.minInc, 1.e-12)
    max_jump = _check_max_inertia_jump(an)

    fext = an.calc_fext(inc=1., silent=True)
    kC0 = an.calc_kC(silent=silent)
    w = _weights(an, kC0)
    K0_abs = _abs_matrix(kC0)
    proj = _residual_projector(an, w)
    q = None
    lbd_dependent = callbacks.is_load_factor_dependent(an)
    if lbd_dependent:
        # -dR/dlbd at c = 0 and lbd = 0
        q = callbacks.calc_load_vector(an, np.zeros_like(fext), fext, 0.)
    d = _constraint_values(an)
    # predictor of the first step: linear solution scaled by the load
    # factor, C^T dc_last = d
    dc_last = _linear_solve(an, kC0, fext if q is None else q, d)
    inc_last = 1.
    c = np.zeros_like(dc_last)
    total_last = 0.
    step_cut = False
    step_num = 1

    state = _restart_state(an, 'NR')
    if state is not None:
        c = np.array(state['c'], dtype=float)
        total_last = state['lbd']
        inc = min(state['inc'], target - total_last)
        dc_last = np.array(state['dc_last'], dtype=float)
        inc_last = state['inc_last']
        q = state['q']
        step_num = state['step_num']
        msg('Restarting at load factor %1.5f' % total_last, level=1,
            silent=silent)
    # factorization of the last converged state, with max_inertia_jump
    solve_last = None
    if max_jump is not None:
        solve_last = _factorize_state(an, c, total_last, silent=silent)

    while True:
        if total_last + inc >= target - min_last_inc:
            total = target
            inc = target - total_last
        else:
            total = total_last + inc
        msg('Step %d, attempting load factor %1.5f' % (step_num, total), level=1, silent=silent)

        dc = (inc/inc_last)*dc_last
        kT0 = None
        if an.modified_NR and step_num == 1 and not an.kT_initial_state:
            kT0 = kC0
        converged, dc, q_new = _NR_iterations(an, c, dc, total, fext,
                kT0=kT0, silent=silent, lbd_dependent=lbd_dependent, q=q,
                w=w, K0_abs=K0_abs, proj=proj)

        solve_new = None
        if converged and an.inertia:
            solve_new = _factorize_state(an, c + dc, total, silent=silent)
            if (_inertia_jump(solve_last, solve_new, max_jump)
                    and inc*INC_CUT_FACTOR >= an.minInc):
                warn('The number of negative eigenvalues changed from %d to '
                     '%d, by more than max_inertia_jump'
                     % (solve_last.num_negative_eigvals,
                        solve_new.num_negative_eigvals), level=2,
                     silent=silent)
                converged = False

        if converged:
            msg('Converged at load factor %1.5f' % total, level=2, silent=silent)
            # the -dR/dlbd of a step that failed is not kept
            q = q_new
            c = c + dc
            total_last = total
            an.cs.append(c.copy()) #NOTE copy required
            an.increments.append(total)
            if an.inertia:
                _record_inertia(an, solve_new, silent=silent)
                solve_last = solve_new
            dc_last = dc
            inc_last = inc
            if not step_cut:
                inc = min(INC_GROWTH_FACTOR*inc, an.maxInc)
            step_cut = False
            step_num += 1
            an.path_state = dict(method='NR', c=c.copy(), lbd=total_last,
                                 inc=inc, dc_last=dc_last.copy(),
                                 inc_last=inc_last, q=q, step_num=step_num)
            if total == target:
                break
            if _stop(an, silent):
                break

        else:
            inc *= INC_CUT_FACTOR
            step_cut = True
            msg('Reseting step with reduced load increment %1.5f' % inc, level=1, silent=silent)
            if inc < an.minInc:
                warn('Minimum step size of %1.5f achieved! Analysis stopped at load factor %1.5f'
                     % (an.minInc, total_last), level=1, silent=silent)
                break

    msg('Finished Non-Linear Static Analysis', silent=silent)
    msg('    load factor %1.5f' % total_last, level=1, silent=silent)
