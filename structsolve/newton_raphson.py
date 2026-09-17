"""Newton-Raphson solver for non-linear static analyses

The module-level constants control the step size and the divergence checks,
and they are shared with the arc-length solvers of
:mod:`structsolve.arc_length`.

"""
import numpy as np

from .logger import msg, warn
from .static import solve


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


def _check_convergence(an, R, fint, fext_total, fext_ref=None):
    """Check the convergence of the residual force vector

    The convergence is achieved when either:

    - ``||R|| <= relTOL*max(||fext_total||, ||fint||, ||fext_ref||)``, using
      the Euclidean norm, if ``an.relTOL`` is not ``None``
    - ``max(|R|) <= absTOL``, if ``an.absTOL`` is not ``None``

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the convergence criteria.
    R : array-like
        Residual force vector.
    fint : array-like
        Internal force vector.
    fext_total : array-like
        External force vector at the current load factor.
    fext_ref : array-like or None, optional
        Additional reference force vector, e.g. the external force vector of
        the current increment, useful when the load factor is close to zero.

    Returns
    -------
    converged : bool
        Whether the convergence criteria are satisfied.
    Rnorm : float
        Euclidean norm of the residual.
    Rrel : float
        Relative residual ``||R||/max(||fext_total||, ||fint||, ||fext_ref||)``.

    """
    if an.relTOL is None and an.absTOL is None:
        raise ValueError('At least one of relTOL or absTOL must be defined')
    Rnorm = np.linalg.norm(R)
    ref = max(np.linalg.norm(fext_total), np.linalg.norm(fint))
    if fext_ref is not None:
        ref = max(ref, np.linalg.norm(fext_ref))
    Rrel = Rnorm/ref if ref > 0 else Rnorm
    converged = False
    if an.relTOL is not None and Rnorm <= an.relTOL*ref:
        converged = True
    if an.absTOL is not None and np.abs(R).max() <= an.absTOL:
        converged = True
    return converged, Rnorm, Rrel


def _check_divergence(an, Rnorm, Rnorms, iteration, silent=False):
    """Check if the iterations of a step must be stopped without convergence

    The iterations are stopped when the residual is not finite, when
    ``an.maxNumIter`` corrections were performed, and, after
    ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` iterations, when the residual
    diverges or when the convergence is too slow.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object with the convergence criteria.
    Rnorm : float
        Norm of the residual at the current iteration.
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


def _NR_iterations(an, c, dc, total, fext, kT0=None, silent=False):
    """Newton-Raphson iterations at a fixed load factor

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

    Returns
    -------
    converged : bool
        Whether the iterations converged.
    dc : array-like
        Solution increment, such that ``c + dc`` is the solution.

    """
    modified_NR = an.modified_NR
    fext_total = total*fext
    Rnorms = []
    iter_kT = 0
    kT = None
    fint = None

    iteration = 0
    while True:
        iteration += 1
        if fint is None:
            fint = an.calc_fint(c=(c + dc), silent=True)
        R = fext_total - fint

        conv, Rnorm, Rrel = _check_convergence(an, R, fint, fext_total)
        msg('Iteration: %d, max(|R|) = %1.3e, relative ||R|| = %1.3e'
            % (iteration, np.abs(R).max(), Rrel), level=2, silent=silent)
        if conv:
            return True, dc
        if _check_divergence(an, Rnorm, Rnorms, iteration, silent=silent):
            return False, dc

        if modified_NR and iteration > 1 and iter_kT < an.compute_every_n:
            iter_kT += 1
        elif modified_NR and iteration == 1 and kT0 is not None:
            kT = kT0
            iter_kT = 1
        else:
            kC = an.calc_kC(c=(c + dc), NLgeom=True, silent=True)
            kG = an.calc_kG(c=(c + dc), NLgeom=True, silent=True)
            kT = kC + kG
            iter_kT = 1

        varc = solve(kT, R, silent=True)

        eta = 1.
        fint = None
        if an.line_search:
            # sufficient decrease of phi(eta) = ||R(eta)||**2, assuming
            # phi'(0) = -2*phi(0), exact for a Newton direction
            fint = an.calc_fint(c=(c + dc + varc), silent=True)
            phi0 = Rnorm**2
            phi1 = np.linalg.norm(fext_total - fint)**2
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
                fint = an.calc_fint(c=(c + dc + eta*varc), silent=True)
                phi1 = np.linalg.norm(fext_total - fint)**2
            if eta != 1.:
                msg('Line-search: eta = %1.5f' % eta, level=3, silent=silent)
        dc = dc + eta*varc


def _solver_NR(an, silent=False, initialInc=None):
    r"""Newton-Raphson solver with load control

    Used by :meth:`.Analysis.static` when ``NL_method='NR'``. The load factor
    `\lambda` goes from zero to one in increments, solving at each increment

    .. math::

        \{R\} = \lambda \{F_{ext}\} - \{F_{int}(c)\} = \{0\}

    Each increment starts from a predictor that extrapolates the previous
    converged increment of the solution, scaled by the ratio between the new
    and the previous load increments. The first increment uses the linear
    solution as predictor.

    By default full Newton-Raphson is used, i.e. the tangent stiffness matrix
    ``kT = kC + kG`` is rebuilt at every iteration, which gives quadratic
    convergence when ``kT`` is the exact Jacobian of ``fint``. With
    ``an.modified_NR = True`` the tangent stiffness matrix is rebuilt only at
    the first iteration of each step and every ``an.compute_every_n``
    iterations.

    With ``an.line_search = True``, the full Newton step (``eta = 1``) is tried
    first and it is accepted if it satisfies a sufficient-decrease test on the
    residual norm. Otherwise ``eta`` is reduced by backtracking.

    See :func:`._check_convergence` for the convergence criteria. A step that
    does not converge within ``an.maxNumIter`` iterations, that diverges or
    that converges too slowly is repeated with the load increment multiplied
    by ``INC_CUT_FACTOR``. The divergence and too-slow checks are only
    activated after ``NUM_ITER_BEFORE_DIVERGENCE_CHECK`` iterations, see
    :func:`._check_divergence`.

    After a step that converged without being cut, the load increment is
    multiplied by ``INC_GROWTH_FACTOR``, limited by ``an.maxInc``. The last
    step is adjusted such that the analysis finishes at a load factor of
    exactly ``1.0``. The analysis stops earlier, keeping the converged
    increments, when the load increment becomes smaller than ``an.minInc``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object, the converged load factors and solutions are appended
        to ``an.increments`` and ``an.cs``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    initialInc : float or None, optional
        Initial load increment, ``an.initialInc`` is used if ``None``.

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
    # predictor of the first step: linear solution scaled by the load factor
    dc_last = solve(kC0, fext, silent=True)
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
        converged, dc = _NR_iterations(an, c, dc, total, fext, kT0=kT0,
                                       silent=silent)

        if converged:
            msg('Converged at load factor %1.5f' % total, level=2, silent=silent)
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
