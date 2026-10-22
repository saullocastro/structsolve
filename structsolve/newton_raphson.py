"""Newton-Raphson solver for non-linear static analyses

The module-level constants control the step size and the divergence checks,
and they are shared with the arc-length solvers of
:mod:`structsolve.arc_length`.

"""
import numpy as np

from .logger import msg, warn
from .static import solve, _factorize
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


def _norm(v, w):
    r"""Norm of the convergence criterion: `\|v\|_D` with the weights ``w``
    of Crisfield's criterion, the Euclidean norm when ``w`` is ``None``"""
    if w is None:
        return float(np.linalg.norm(v))
    return scaled_norm(v, w)


def _log_iteration(an, iteration, R, zeta, silent, lbd=None):
    r"""Message of one equilibrium iteration, with `\zeta` of Crisfield's
    criterion or the relative residual of the relative criterion"""
    name = 'zeta' if an.convergence == 'crisfield' else 'relative ||R||'
    load = '' if lbd is None else ', lbd %1.5f' % lbd
    msg('Iteration: %d%s, max(|R|) = %1.3e, %s = %1.3e'
        % (iteration, load, np.abs(R).max(), name, zeta), level=2,
        silent=silent)


def _check_convergence(an, R, fint, fext_total, fext_ref=None, q_ref=None,
                       w=None):
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
    ``'crisfield'``. See :func:`.convergence.check_convergence`.

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
        `\partial R/\partial \lambda` when ``calc_fext`` accepts ``lbd``,
        see :mod:`.callbacks`.
    w : array-like or None, optional
        Weights of Crisfield's criterion, see
        :func:`.convergence.crisfield_weights`, required with
        ``an.convergence = 'crisfield'``.

    Returns
    -------
    converged : bool
        Whether the convergence criterion is satisfied.
    Rnorm : float
        Norm of the residual of the criterion, `\|R\|_D` with
        ``'crisfield'``, the Euclidean norm with ``'relative'``, used by the
        divergence checks and the line search.
    Rrel : float
        Relative residual, Crisfield's `\zeta` with ``'crisfield'``,
        ``||R||/max(||fext_total||, ||fint||, ||fext_ref||, ||q_ref||)``
        with ``'relative'``.

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
                             q_ref=q_ref, w=w)


def _check_divergence(an, Rnorm, Rnorms, iteration, silent=False):
    r"""Check if the iterations of a step must be stopped without convergence


    The iterations are stopped when the residual is not finite, when
    ``an.maxNumIter`` corrections were performed, and, after
    ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` iterations, when the residual
    diverges or when the convergence is too slow.

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

    """
    if not np.isfinite(Rnorm):
        warn('Diverged - residual is not finite', level=2, silent=silent)
        return True
    if iteration > an.maxNumIter:
        warn('Maximum number of iterations achieved!', level=2, silent=silent)
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


def _NR_iterations(an, c, dc, total, fext, kT0=None, silent=False,
                   lbd_dependent=False, q=None, w=None):
    r"""Newton-Raphson iterations at a fixed load factor

    Corrects the predictor ``dc`` at the load factor ``total`` until the
    convergence criteria of :func:`._check_convergence` are satisfied, or
    until the iterations are stopped by :func:`._check_divergence` or by a
    singular tangent stiffness matrix. The tangent stiffness matrix is
    ``calc_kC + calc_kG`` at the current iterate, see
    :func:`.callbacks.calc_kT`; its updates and the line search are
    controlled by ``an.modified_NR``, ``an.compute_every_n``,
    ``an.line_search`` and ``an.max_iter_line_search``.

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
        ``total*q``, with `\{q\} = \partial R/\partial \lambda`, is then an
        additional reference of the convergence criterion, see
        :func:`._check_convergence`. `\{q\}` is evaluated together with the
        tangent stiffness matrix, at the same iterate, and the last one
        evaluated is the reference until the next update of the tangent.
    q : array-like or None, optional
        The last `\{q\}` evaluated before these iterations, the reference
        until the first update of the tangent stiffness matrix, used only
        when ``lbd_dependent=True``.
    w : array-like or None, optional
        Weights of Crisfield's criterion, see
        :func:`.convergence.crisfield_weights`, computed from ``calc_kC()``
        when ``None`` and ``an.convergence = 'crisfield'``.

    Returns
    -------
    converged : bool
        Whether the iterations converged.
    dc : array-like
        Solution increment, such that ``c + dc`` is the solution.
    q : array-like or None
        The last `\{q\}` evaluated, ``None`` when ``lbd_dependent=False``.

    Notes
    -----
    At each iteration, with the load factor `\lambda` = ``total`` fixed,

    .. math::

        \{R\} = \lambda \{F_{ext}\} - \{F_{int}(c + \Delta c, \lambda)\},
        \qquad \{\delta c\} = [K_T]^{-1}\{R\}, \qquad
        \{\Delta c\} \leftarrow \{\Delta c\} + \eta \{\delta c\}

    With full Newton-Raphson, `[K_T]` is computed and factorized at every
    iteration at `c + \Delta c`, and the convergence is quadratic when it is
    the exact Jacobian `-\partial R/\partial c` [1]_. With
    ``an.modified_NR = True``, `[K_T]` is computed at the first iteration,
    or given as ``kT0``, and then every ``an.compute_every_n`` iterations,
    its LU factorization being reused in the other iterations.

    `\eta = 1` without line search. With ``an.line_search = True``, with
    `\phi(\eta) = \|R(c + \Delta c + \eta \, \delta c)\|^2`, the step `\eta`
    is accepted when it satisfies the sufficient-decrease (Armijo) test

    .. math::

        \phi(\eta) \le (1 - 2 \alpha \eta) \phi(0)

    with `\alpha` = ``LINE_SEARCH_ALPHA``, which assumes `\phi'(0) =
    -2\phi(0)`, exact for a Newton direction, i.e. when `[K_T]` is the exact
    Jacobian at `c + \Delta c`; with ``an.modified_NR = True`` it is an
    approximation. The full step `\eta = 1` is tried first. Otherwise `\eta`
    is replaced by the minimizer `\phi(0)/A` of the quadratic interpolation
    of `\phi(0)`, `\phi'(0)` and `\phi(\eta)`, with `A = (\phi(\eta) -
    \phi(0) + 2\phi(0)\eta)/\eta^2`, or by `\eta/2` when `A` is not
    positive, safeguarded in `[0.1 \eta, 0.5 \eta]` [2]_. After
    ``an.max_iter_line_search`` reductions the last `\eta` is accepted. The
    internal force vector of the accepted point is reused in the residual
    of the next iteration.

    References
    ----------
    .. [1] Crisfield, M.A., Non-linear Finite Element Analysis of Solids and
       Structures, Vol. 1: Essentials, Wiley, 1991.
       TODO(author): confirm this source
    .. [2] Dennis, J.E., Schnabel, R.B., Numerical Methods for Unconstrained
       Optimization and Nonlinear Equations, Prentice-Hall, 1983.
       TODO(author): confirm this source

    """

    modified_NR = an.modified_NR
    if w is None:
        w = _weights(an, an.calc_kC(silent=True))
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
        R = fext_total - fint
        if q is not None:
            q_ref = total*q

        conv, Rnorm, Rrel = _check_convergence(an, R, fint, fext_total,
                                               q_ref=q_ref, w=w)
        _log_iteration(an, iteration, R, Rrel, silent)
        if conv:
            return True, dc, q
        if _check_divergence(an, Rnorm, Rnorms, iteration, silent=silent):
            return False, dc, q

        # the factorization of the tangent stiffness matrix is reused while
        # the matrix is not updated
        try:
            if modified_NR and iteration > 1 and iter_kT < an.compute_every_n:
                iter_kT += 1
            elif modified_NR and iteration == 1 and kT0 is not None:
                solve_kT = _factorize(kT0)
                iter_kT = 1
            else:
                solve_kT = _factorize(callbacks.calc_kT(an, c + dc, total))
                iter_kT = 1
                if lbd_dependent:
                    # dR/dlbd at the iterate of the tangent stiffness matrix
                    q = callbacks.calc_load_vector(an, c + dc, fext, total)
        except RuntimeError:
            warn('Singular tangent stiffness matrix', level=2, silent=silent)
            return False, dc, q

        varc = solve_kT(R)

        eta = 1.
        fint = None
        if an.line_search:
            # sufficient decrease of phi(eta) = ||R(eta)||**2, assuming
            # phi'(0) = -2*phi(0), exact for a Newton direction
            fint = callbacks.calc_fint(an, c + dc + varc, total)
            phi0 = Rnorm**2
            phi1 = _norm(fext_total - fint, w)**2
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
                phi1 = _norm(fext_total - fint, w)**2
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
    exactly ``1.0``. The analysis stops earlier, keeping the converged
    increments, when the load increment becomes smaller than ``an.minInc``.

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

        \{R\} = \lambda \{F_{ext}\} - \{F_{int}(c)\} = \{0\}

    Each increment starts from a predictor that extrapolates the previous
    converged increment of the solution, scaled by the ratio between the new
    and the previous load increments. The first increment uses the linear
    solution as predictor: `\Delta\lambda [K_0]^{-1}\{F_{ext}\}`, with
    `[K_0]` the linear stiffness matrix ``calc_kC()``.

    The callables that accept the keyword argument ``inc`` receive the load
    factor `\lambda` of the step, see :mod:`.callbacks`, which allows
    configuration-dependent loads such as follower pressures.

    When ``calc_fext`` accepts the keyword argument ``lbd``, e.g. for
    prescribed displacements imposed exactly, see :mod:`.callbacks`, the
    load vector `\{q\} = \partial R/\partial \lambda` at `c = 0`,
    `\lambda = 0` gives the predictor of the first increment, its linear
    solution `[K_0]^{-1}\{q\}`, instead of the linear solution of the
    reference vector, which is null under a prescribed displacement alone,
    and `\lambda \|q\|` is an additional reference of the relative
    convergence criterion. `\{q\}` is evaluated together with the tangent
    stiffness matrix, at the same iterate, i.e. ``calc_fext`` is called once
    per update of the tangent stiffness matrix, and the last `\{q\}`
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
    matrix `[K_0]` is used instead of the tangent at the first iteration.

    With ``an.line_search = True``, the full Newton step (``eta = 1``) is tried
    first and it is accepted if it satisfies a sufficient-decrease test on the
    residual norm. Otherwise ``eta`` is reduced by backtracking, see
    :func:`._NR_iterations`.

    The full and modified Newton-Raphson methods are described in [1]_.

    The step that would end within ``max(an.minInc, 1e-12)`` of the load

    factor 1.0, or beyond it, is replaced by the step to exactly 1.0.

    References
    ----------
    .. [1] Crisfield, M.A., Non-linear Finite Element Analysis of Solids and
       Structures, Vol. 1: Essentials, Wiley, 1991.
       TODO(author): confirm this source

    """
    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Newton-Raphson solver', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)

    if initialInc is None:
        initialInc = an.initialInc
    inc = min(initialInc, 1.)
    min_last_inc = max(an.minInc, 1.e-12)

    fext = an.calc_fext(inc=1., silent=True)
    kC0 = an.calc_kC(silent=silent)
    w = _weights(an, kC0)
    q = None
    lbd_dependent = callbacks.is_load_factor_dependent(an)
    if lbd_dependent:
        # dR/dlbd at c = 0 and lbd = 0
        q = callbacks.calc_load_vector(an, np.zeros_like(fext), fext, 0.)
    # predictor of the first step: linear solution scaled by the load factor
    dc_last = solve(kC0, fext if q is None else q, silent=True)
    inc_last = 1.
    c = np.zeros_like(dc_last)
    total_last = 0.
    step_cut = False

    step_num = 1

    while True:
        if total_last + inc >= 1. - min_last_inc:
            total = 1.
            inc = 1. - total_last
        else:
            total = total_last + inc
        msg('Step %d, attempting load factor %1.5f' % (step_num, total), level=1, silent=silent)

        dc = (inc/inc_last)*dc_last
        kT0 = None
        if an.modified_NR and step_num == 1 and not an.kT_initial_state:
            kT0 = kC0
        converged, dc, q_new = _NR_iterations(an, c, dc, total, fext,
                kT0=kT0, silent=silent, lbd_dependent=lbd_dependent, q=q,
                w=w)

        if converged:
            msg('Converged at load factor %1.5f' % total, level=2, silent=silent)
            # the dR/dlbd of a step that failed is not kept
            q = q_new
            c = c + dc
            total_last = total
            an.cs.append(c.copy()) #NOTE copy required
            an.increments.append(total)
            if total == 1.:
                break
            dc_last = dc
            inc_last = inc
            if not step_cut:
                inc = min(INC_GROWTH_FACTOR*inc, an.maxInc)
            step_cut = False
            step_num += 1

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
