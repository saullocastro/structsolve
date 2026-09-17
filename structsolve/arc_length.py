import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .logger import msg, warn
from .static import solve
from .sparseutils import remove_null_cols
from .newton_raphson import (_check_convergence, _check_divergence,
                             _NR_iterations, INC_CUT_FACTOR)


#: Scaling factor of the load-factor term in the arc-length constraint
PSI = 1.
#: Desired number of corrections per step, used to adapt the arc length
DESIRED_NUM_ITER = 4
#: Maximum factor to increase the arc length after a converged step
MAX_GROWTH_FACTOR = 1.5
#: Minimum factor to reduce the arc length after a converged step
MIN_GROWTH_FACTOR = 0.5
#: Maximum number of converged steps of an arc-length analysis
MAX_NUM_STEPS = 1000


def _factorize(kT):
    """LU factorization of the tangent stiffness matrix

    Null rows and columns are removed, the corresponding values of the
    solution are zero.

    Returns
    -------
    solve_kT : callable
        Function that returns the solution ``x`` of ``kT x = b``.

    """
    kT_red, used_cols = remove_null_cols(kT, silent=True)
    lu = splu(csc_matrix(kT_red))

    def solve_kT(b):
        x = np.zeros(b.shape[0], dtype=np.result_type(b, lu.U.dtype))
        x[used_cols] = lu.solve(b[used_cols])
        return x

    return solve_kT


def _solver_arc_length(an, method, silent=False):
    r"""Arc-length solver

    The equilibrium path `\{F_{int}(c)\} = \lambda \{F_{ext}\}` is traced with
    the arc-length constraint

    .. math::

        \frac{\{\Delta c\}^T\{\Delta c\}}{\{u_{ref}\}^T\{u_{ref}\}}
        + \psi^2 \Delta\lambda^2 = \Delta s^2

    where `\{\Delta c\}` and `\Delta\lambda` are the increments of the current
    step and `\{u_{ref}\}` is the linear solution for `\lambda=1`, which makes
    the arc length `\Delta s` independent of the units of the model. In the
    linear regime `\Delta s \approx \sqrt{1 + \psi^2} \Delta\lambda`.

    Each step starts from a tangent predictor, whose direction follows the
    previous increment, allowing limit points to be passed. At each
    iteration, the bordered system is solved with two solutions of the
    tangent stiffness matrix, `\{\delta c_R\} = [K_T]^{-1}\{R\}` and
    `\{\delta c_q\} = [K_T]^{-1}\{F_{ext}\}`, and the correction of the load
    factor `\delta\lambda` comes from the constraint:

    - ``method='riks'``: the constraint is linearized at the current
      increment (updated normal plane), which is Newton-Raphson applied to
      the augmented system and gives quadratic convergence.
    - ``method='crisfield'``: the quadratic (spherical) constraint is solved
      exactly, choosing the root with the smallest angle to the current
      increment. When the roots are complex, the linearized constraint is
      used.

    The arc length of each step is adapted according to the number of
    iterations of the previous step, limited by ``an.maxInc``. When a step
    fails, the arc length is reduced. The analysis stops when:

    - the load factor reaches 1.0, the step where it is exceeded is replaced
      by Newton-Raphson iterations at exactly `\lambda = 1`
    - the cumulative arc length reaches ``an.maxArcLength``
    - the arc length becomes smaller than ``an.minInc``

    The initial arc length corresponds to a load factor increment of
    ``an.initialInc`` along the initial tangent.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object, the converged load factors and solutions are appended
        to ``an.increments`` and ``an.cs``.
    method : str
        ``'riks'`` or ``'crisfield'``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.

    """
    if method not in ('riks', 'crisfield'):
        raise ValueError('Invalid arc-length method: %s' % method)
    modified_NR = an.modified_NR

    fext = an.calc_fext(inc=1., silent=True)
    kC0 = an.calc_kC(silent=silent)
    u_ref = solve(kC0, fext, silent=True)
    scale2 = u_ref.dot(u_ref)
    if not scale2 > 0:
        scale2 = 1.
    psi2 = PSI**2

    def inner(dc1, dlbd1, dc2, dlbd2):
        return dc1.dot(dc2)/scale2 + psi2*dlbd1*dlbd2

    c = np.zeros_like(u_ref)
    lbd = 0.
    dc_last = None
    dlbd_last = None
    arc_length = None
    total_arc_length = 0.
    step_cut = False
    solve_kT_conv = None

    step_num = 1

    while True:
        if step_num > MAX_NUM_STEPS:
            warn('Maximum number of steps of %d achieved! Analysis stopped at load factor %1.5f'
                 % (MAX_NUM_STEPS, lbd), level=1, silent=silent)
            break

        # tangent stiffness matrix at the last converged state
        if solve_kT_conv is None:
            if step_num == 1 and modified_NR and not an.kT_initial_state:
                kT = kC0
            else:
                kC = an.calc_kC(c=c, NLgeom=True, silent=True)
                kG = an.calc_kG(c=c, NLgeom=True, silent=True)
                kT = kC + kG
            try:
                solve_kT_conv = _factorize(kT)
            except RuntimeError:
                warn('Singular tangent stiffness matrix at the last converged state! Analysis stopped',
                     level=1, silent=silent)
                break
        dc_q_conv = solve_kT_conv(fext)
        norm_q = np.sqrt(inner(dc_q_conv, 1., dc_q_conv, 1.))
        if arc_length is None:
            arc_length = an.initialInc*norm_q
            max_arc_length_inc = max(an.maxInc, arc_length)

        # tangent predictor
        sign = 1.
        if dc_last is not None:
            sign = np.sign(inner(dc_q_conv, 1., dc_last, dlbd_last))
            if sign == 0:
                sign = 1.
        dlbd = sign*arc_length/norm_q
        dc = dlbd*dc_q_conv

        msg('Step %d, lbd %1.5f, arc-length increment %1.5f' % (step_num, lbd, arc_length),
            level=1, silent=silent)

        converged = False
        Rnorms = []
        solve_kT = solve_kT_conv
        dc_q = dc_q_conv
        iter_kT = 0
        iteration = 0
        while True:
            iteration += 1
            fint = an.calc_fint(c=(c + dc), silent=True)
            fext_total = (lbd + dlbd)*fext
            R = fext_total - fint
            conv, Rnorm, Rrel = _check_convergence(an, R, fint, fext_total,
                                                   fext_ref=dlbd*fext)
            msg('Iteration: %d, lbd %1.5f, max(|R|) = %1.3e, relative ||R|| = %1.3e'
                % (iteration, lbd + dlbd, np.abs(R).max(), Rrel), level=2, silent=silent)
            if conv:
                converged = True
                break
            if _check_divergence(an, Rnorm, Rnorms, iteration, silent=silent):
                break

            # with modified_NR the tangent of the last converged state is
            # used first, otherwise it is updated at every iteration
            if modified_NR and iter_kT < an.compute_every_n:
                iter_kT += 1
            else:
                kC = an.calc_kC(c=(c + dc), NLgeom=True, silent=True)
                kG = an.calc_kG(c=(c + dc), NLgeom=True, silent=True)
                try:
                    solve_kT = _factorize(kC + kG)
                except RuntimeError:
                    warn('Singular tangent stiffness matrix', level=2, silent=silent)
                    break
                dc_q = solve_kT(fext)
                iter_kT = 1
            dc_R = solve_kT(R)

            # linearized constraint
            den = inner(dc, dlbd, dc_q, 1.)
            g = 0.5*(inner(dc, dlbd, dc, dlbd) - arc_length**2)
            varlbd = None
            if den != 0:
                varlbd = -(g + inner(dc, dlbd, dc_R, 0.))/den
            if method == 'crisfield':
                u = dc + dc_R
                a = inner(dc_q, 1., dc_q, 1.)
                b = 2*inner(u, dlbd, dc_q, 1.)
                cc = inner(u, dlbd, u, dlbd) - arc_length**2
                disc = b**2 - 4*a*cc
                if disc >= 0:
                    sq = np.sqrt(disc)
                    q = -0.5*(b + np.copysign(sq, b))
                    roots = [q/a]
                    if q != 0:
                        roots.append(cc/q)
                    cosines = [inner(dc, dlbd, u + r*dc_q, dlbd + r) for r in roots]
                    varlbd = roots[int(np.argmax(cosines))]
                else:
                    msg('Complex roots, using the linearized constraint', level=3, silent=silent)
            if varlbd is None or not np.isfinite(varlbd):
                warn('Diverged - arc-length constraint cannot be satisfied', level=2, silent=silent)
                break
            dc = dc + dc_R + varlbd*dc_q
            dlbd = dlbd + varlbd

        finished = False
        if converged and lbd + dlbd > 1. and lbd < 1.:
            msg('Load factor 1.0 exceeded, correcting to load factor 1.0...', level=1, silent=silent)
            t = (1. - lbd)/dlbd
            converged, dc1 = _NR_iterations(an, c, t*dc, 1., fext, kT0=None, silent=silent)
            if converged:
                msg('Converged at load factor 1.00000', level=2, silent=silent)
                an.cs.append((c + dc1).copy())
                an.increments.append(1.)
                finished = True

        if finished:
            break

        if converged:
            c = c + dc
            lbd = lbd + dlbd
            total_arc_length += arc_length
            msg('Converged at load factor %1.5f, total arc length %1.5f' % (lbd, total_arc_length),
                level=2, silent=silent)
            an.cs.append(c.copy()) #NOTE copy required
            an.increments.append(lbd)
            if lbd == 1.:
                break
            if total_arc_length >= an.maxArcLength:
                msg('Maximum specified arc-length of %1.5f achieved' % an.maxArcLength,
                    level=1, silent=silent)
                break
            dc_last = dc
            dlbd_last = dlbd
            factor = np.sqrt(DESIRED_NUM_ITER/max(iteration - 1, 1))
            factor = min(max(factor, MIN_GROWTH_FACTOR), MAX_GROWTH_FACTOR)
            if step_cut:
                factor = min(factor, 1.)
            arc_length = min(factor*arc_length, max_arc_length_inc)
            step_cut = False
            solve_kT_conv = None
            step_num += 1

        else:
            arc_length *= INC_CUT_FACTOR
            step_cut = True
            msg('Reseting step with reduced arc-length increment %1.5f' % arc_length,
                level=1, silent=silent)
            if arc_length < an.minInc:
                warn('Minimum arc-length increment of %1.5f achieved! Analysis stopped at load factor %1.5f'
                     % (an.minInc, lbd), level=1, silent=silent)
                break

    msg('Finished Non-Linear Static Analysis', silent=silent)
    msg('    load factor %1.5f, total arc length %1.5f' % (lbd if not an.increments
        else an.increments[-1], total_arc_length), level=1, silent=silent)
