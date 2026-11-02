"""Arc-length solvers for non-linear static analyses

The Riks and Crisfield methods share the implementation of
:func:`._solver_arc_length`, and are selected with
``Analysis.NL_method = 'arc_length_riks'`` or
``Analysis.NL_method = 'arc_length_crisfield'``, as does the indirect
displacement control, ``Analysis.NL_method = 'displacement_control'``. The module-level constants
control the arc-length scaling and adaptation.

"""
import numpy as np

from .logger import msg, warn
from .static import solve, _factorize
from . import callbacks
from .newton_raphson import (_check_convergence, _check_divergence,
                             _NR_iterations, _log_iteration, _weights,
                             _abs_matrix, _norm, _converged_state,
                             INC_CUT_FACTOR)


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
#: Factors tried in turn by the line search of the corrections, with
#: ``Analysis.arc_length_line_search = True``
LINE_SEARCH_ETAS = (1., 0.5, 0.25)
#: Predictors of the arc-length methods, ``Analysis.arc_length_predictor``
PREDICTORS = ('tangent', 'secant')


def _solver_arc_length(an, method, silent=False):
    r"""Arc-length solver

    Traces the equilibrium path of the model of ``an`` with an arc-length
    method, in which the load factor is an unknown of each step besides the
    solution vector, such that limit points can be passed. ``method`` selects
    how the load factor of each iteration is computed, see
    :func:`.arc_length_riks._solver_arc_length_riks` and
    :func:`.arc_length_crisfield._solver_arc_length_crisfield`.

    The steps are controlled by the attributes ``initialInc``, ``minInc``,
    ``maxInc`` and ``maxArcLength`` of ``an``, and by the module-level
    constants; the iterations of each step by ``convergence``, ``relTOL``,
    ``absTOL``, ``maxNumIter`` and ``too_slow_TOL``, with the checks of
    :func:`.newton_raphson._check_convergence` and
    :func:`.newton_raphson._check_divergence`, where the convergence
    criterion also considers the external force vector of the increment of
    the load factor of the current step; and
    ``an.modified_NR``, ``an.compute_every_n`` and ``an.kT_initial_state``
    control how often the tangent stiffness matrix is updated. A step that
    fails is repeated with the arc length multiplied by ``INC_CUT_FACTOR``.
    The analysis stops when:

    - the load factor reaches 1.0, the last solution being computed at a load
      factor of exactly 1.0
    - the cumulative arc length reaches ``an.maxArcLength``
    - the arc length becomes smaller than ``an.minInc``
    - ``MAX_NUM_STEPS`` steps converged
    - the tangent stiffness matrix at the last converged state is singular

    The callables of ``an`` that declare the optional keyword arguments
    ``inc``, ``c`` or ``lbd`` receive the current load factor and
    configuration, see :mod:`.callbacks`.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object, the converged load factors and solutions are appended
        to ``an.increments`` and ``an.cs``.
    method : str
        ``'riks'``, ``'crisfield'`` or ``'displacement_control'``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.

    Notes
    -----
    The equilibrium path `\boldsymbol{F}_{int}(\boldsymbol{c}) = \lambda \boldsymbol{F}_{ext}` is traced with
    the arc-length constraint

    .. math::

        \frac{\Delta \boldsymbol{c}^\top \Delta \boldsymbol{c}}{\boldsymbol{u}_{ref}^\top \boldsymbol{u}_{ref}}
        + \psi^2 \Delta\lambda^2 = \Delta s^2

    where `\Delta \boldsymbol{c}` and `\Delta\lambda` are the increments of the current
    step and `\boldsymbol{u}_{ref}` is the linear solution for `\lambda=1`, which makes
    the arc length `\Delta s` independent of the units of the model. In the
    linear regime `\Delta s \approx \sqrt{1 + \psi^2} \Delta\lambda`. It is
    the spherical arc-length constraint of [4]_ (Section 9.3.2, Eq. (9.19),
    p. 271), where the displacement term is not scaled and the load term is
    `\psi^2 \Delta\lambda^2 \boldsymbol{F}_{ext}^\top \boldsymbol{F}_{ext}`. `\psi` is the
    constant ``PSI``,
    and `\boldsymbol{u}_{ref}^\top \boldsymbol{u}_{ref}` is
    replaced by 1 when it is null.

    With ``an.arc_length_weights`` `= \boldsymbol{w}`, the products of
    displacements in the constraint, and in the inner product below, are
    weighted, `\boldsymbol{a}^\top \boldsymbol{W} \boldsymbol{b}` with `\boldsymbol{W} =
    \mathrm{diag}(\boldsymbol{w})`, also in the scale `\boldsymbol{u}_{ref}^\top
    \boldsymbol{W} \boldsymbol{u}_{ref}`. When the unknowns mix translations with
    their derivatives or with rotations, of other units and magnitudes, the
    weights 1 for the translations and 0 for the others measure the arc
    length on the translations only. The arc length is then independent of
    the units of the degrees of freedom of null weight, as is Crisfield's
    convergence criterion, see :mod:`.convergence`.

    With the inner product

    .. math::

        \langle (a, \alpha), (b, \beta) \rangle =
        \frac{\boldsymbol{a}^\top \boldsymbol{b}}{\boldsymbol{u}_{ref}^\top \boldsymbol{u}_{ref}} + \psi^2 \alpha \beta

    the constraint reads `g = \frac{1}{2}(\langle (\Delta c, \Delta\lambda),
    (\Delta c, \Delta\lambda) \rangle - \Delta s^2) = 0`.

    Each step starts from a tangent predictor, whose direction follows the
    previous increment, allowing limit points to be passed. With
    `\delta \boldsymbol{c}_q = \boldsymbol{K}_T^{-1} \boldsymbol{F}_{ext}` at the last converged state,
    the predictor

    .. math::

        \Delta\lambda = \pm \frac{\Delta s}{\sqrt{\langle (\delta c_q, 1),
        (\delta c_q, 1) \rangle}}, \qquad
        \Delta \boldsymbol{c} = \Delta\lambda \delta \boldsymbol{c}_q

    satisfies the constraint, with the sign of `\langle (\delta c_q, 1),
    (\Delta c_{prev}, \Delta\lambda_{prev}) \rangle`, where
    `(\Delta c_{prev}, \Delta\lambda_{prev})` is the increment of the
    previous converged step, and a positive sign at the first step. The
    initial arc length, `\Delta s_0 = initialInc \sqrt{\langle (\delta \boldsymbol{c}_q,
    1), (\delta \boldsymbol{c}_q, 1) \rangle}` at the origin, corresponds to a load factor
    increment of ``an.initialInc`` along the initial tangent.

    With ``an.arc_length_predictor = 'secant'``, the steps after the first
    one start instead from the increment of the previous converged step
    scaled to the new arc length, `(\Delta c, \Delta\lambda) = (\Delta
    s/\Delta s_{prev}) (\Delta c_{prev}, \Delta\lambda_{prev})`, which
    continues the path in the direction already traced, without the sign
    rule. Beyond a limit point the tangent predictor `\delta
    \boldsymbol{c}_q` is dominated by the eigenvectors of `\boldsymbol{K}_T` whose
    eigenvalues are close to zero, which may turn the path back and forth
    when there are many of them, e.g. after the bifurcations of a cylinder
    under axial compression. With full Newton-Raphson the factorization of
    `\boldsymbol{K}_T` at the last converged state is then not needed and it is
    skipped, unless ``calc_fext`` accepts ``lbd``, whose `-\partial
    \boldsymbol{R}/\partial \lambda` is evaluated together with the tangent.

    At each iteration, the residual is `\boldsymbol{R} = \boldsymbol{F}_{int}(\boldsymbol{c} +
    \Delta \boldsymbol{c}) - (\lambda + \Delta\lambda) \boldsymbol{F}_{ext}`, with `\partial
    \boldsymbol{R}/\partial \lambda = -\boldsymbol{F}_{ext}`, and the bordered system
    `\boldsymbol{K}_T \delta \boldsymbol{c} - \delta\lambda \, \boldsymbol{F}_{ext} = -\boldsymbol{R}` is
    solved with two solutions of the tangent stiffness matrix,
    `\delta \boldsymbol{c}_R = -\boldsymbol{K}_T^{-1} \boldsymbol{R}` and `\delta \boldsymbol{c}_q =
    \boldsymbol{K}_T^{-1} \boldsymbol{F}_{ext}`; the increment is updated to `(\Delta \boldsymbol{c} +
    \delta \boldsymbol{c}_R + \delta\lambda \, \delta \boldsymbol{c}_q, \Delta\lambda +
    \delta\lambda)`, where the correction of the load factor
    `\delta\lambda` comes from the constraint:

    - ``method='riks'``: the constraint is linearized at the current
      increment (updated normal plane), which is Newton-Raphson applied to
      the augmented system of the equilibrium equations and the constraint
      ([4]_, Section 9.3.2, p. 272), with quadratic convergence when `\boldsymbol{K}_T`
      is updated at every iteration and is the exact Jacobian. It follows
      Riks [1]_, who applies Newton's method to the equilibrium equations
      augmented with an equation that controls the arc length, and states
      its quadratic convergence (Sections 3.2 and 3.3, Eqs. (3.7), (3.11)
      and (3.17), pp. 537-539). The equation of Riks is linear (Section 3.1,
      Eqs. (3.4) and (3.9), pp. 537-538): the plane normal to the unit
      tangent of the path at the last converged point, at the distance
      `\Delta s` from it, in the unscaled Euclidean space of the load
      parameter and the generalized coordinates; the plane is fixed during
      the iterations of a step and redefined at each converged point.
      Without the mismatch `g` of the constraint, it
      is the updated normal plane of Ramm [3]_ (p. 75); [4]_ (Section
      9.3.2.2, Eq. (9.29), pp. 274-275) attributes the mismatch term to
      Schweizerhof and Wriggers and to Riks:

      .. math::

          \delta\lambda = -\frac{g + \langle (\Delta c, \Delta\lambda),
          (\delta c_R, 0) \rangle}{\langle (\Delta c, \Delta\lambda),
          (\delta c_q, 1) \rangle}

    - ``method='crisfield'``: the quadratic (spherical) constraint is solved
      exactly, choosing the root with the smallest angle to the current
      increment. When the roots are complex, the linearized constraint is
      used, see [2]_. With `\boldsymbol{u} = \Delta \boldsymbol{c} + \delta \boldsymbol{c}_R`, the
      constraint
      `\langle (u + \delta\lambda \, \delta c_q, \Delta\lambda +
      \delta\lambda), (u + \delta\lambda \, \delta c_q, \Delta\lambda +
      \delta\lambda) \rangle = \Delta s^2` is the quadratic equation
      `a \, \delta\lambda^2 + b \, \delta\lambda + c = 0` with
      `a = \langle (\delta c_q, 1), (\delta c_q, 1) \rangle`,
      `b = 2 \langle (u, \Delta\lambda), (\delta c_q, 1) \rangle` and
      `c = \langle (u, \Delta\lambda), (u, \Delta\lambda) \rangle -
      \Delta s^2`. Its roots are computed as `q/a` and `c/q`, with
      `q = -\frac{1}{2}(b + \mathrm{sign}(b) \sqrt{b^2 - 4 a c})`, and the
      root that maximizes the inner product of the new increment with the
      current one `(\Delta c, \Delta\lambda)` is chosen, i.e. the smallest
      angle, since both have the length `\Delta s`.

    - ``method='displacement_control'``: indirect displacement control, the
      arc-length constraint being replaced by the linear constraint

      .. math::

          \frac{\boldsymbol{\psi}^\top \Delta \boldsymbol{c}}{s_{ref}} = \Delta s

      with `\boldsymbol{\psi}` = ``an.control_vector`` and `s_{ref}` =
      ``an.control_scale``, by default `|\boldsymbol{\psi}^\top
      \boldsymbol{u}_{ref}|`, such that in the linear regime `\Delta s =
      \Delta\lambda`; a unit vector `\boldsymbol{\psi}` controls the
      displacement of one degree of freedom, e.g. the end shortening, and a
      buckling mode controls its amplitude. The correction of the load
      factor satisfies the constraint exactly:

      .. math::

          \delta\lambda = \frac{s_{ref} \Delta s - \boldsymbol{\psi}^\top
          (\Delta \boldsymbol{c} + \delta \boldsymbol{c}_R)}{\boldsymbol{\psi}^\top
          \delta \boldsymbol{c}_q}

      and the predictors are scaled to satisfy it, the tangent predictor
      `\Delta\lambda = s_{ref} \Delta s/(\boldsymbol{\psi}^\top \delta
      \boldsymbol{c}_q)`, without the sign rule, the sign of the load factor
      increment following from the one of `\boldsymbol{\psi}^\top \delta
      \boldsymbol{c}_q`. The step length `\Delta s` plays the role of the arc
      length, with ``an.initialInc`` the first one, `\Delta s` keeping the
      sign for which the controlled quantity increases with the load factor
      at the origin. The path can be traced through the limit points of the
      load and of the other degrees of freedom, e.g. snap-backs, as long as
      the controlled quantity increases monotonically along it; at a limit
      point of the controlled quantity `\boldsymbol{\psi}^\top \delta
      \boldsymbol{c}_q` vanishes and the iterations fail or converge to
      another part of the path. For the indirect displacement control, see
      de Borst [5]_. The check against the doubling back below does not
      apply, since the previous converged point does not satisfy the
      constraint.

    With ``an.arc_length_line_search = True``, the whole correction
    `(\delta \boldsymbol{c}_R + \delta\lambda \, \delta \boldsymbol{c}_q, \delta\lambda)`
    is multiplied by the first factor `\eta` of ``LINE_SEARCH_ETAS``,
    `1, 1/2, 1/4`, for which the norm of the residual of the convergence
    criterion, see :func:`.newton_raphson._check_convergence`, decreases,
    or by the factor of the smallest norm when none decreases it. The
    internal force vector of the accepted point is reused in the residual
    of the next iteration, such that the full correction costs no
    additional evaluation. With `\eta < 1` the iterate leaves the
    constraint surface, the next correction returning to it; the converged
    point may thus be closer than `\Delta s` to the last one when the last
    correction was reduced. Contrary to the line search of
    :func:`.newton_raphson._NR_iterations`, there is no sufficient-decrease
    test: the residual of the equilibrium equations alone is not the merit
    function of the augmented system of the equilibrium equations and the
    constraint, along whose Newton direction it need not decrease.

    The step fails when `\delta\lambda` cannot be computed or is not
    finite, when the iterations are stopped by
    :func:`.newton_raphson._check_divergence`, earlier with
    ``an.early_divergence = True``, and when the converged increment points
    against the predictor `(\Delta c_{pred}, \Delta\lambda_{pred})` of the
    step,
    `\langle (\Delta c, \Delta\lambda), (\Delta c_{pred},
    \Delta\lambda_{pred}) \rangle \le 0`. The previous converged point is
    also at the distance `\Delta s` from the last one, and with long steps
    or near a limit point the iterations may converge to it and then trace
    the path backwards, the "doubling back" that [2]_ (p. 56) avoids with
    the choice of the root. The accepted points are thus on the side of the
    tangent predictor of the hyperplane normal to it, in the inner product
    above, through the last converged point, as the points of the plane of
    Riks [1]_, normal to the tangent at the distance `\Delta s` from the
    last converged point. With full Newton-Raphson, `\boldsymbol{K}_T` is updated at
    every iteration at the current iterate `(c + \Delta c, \lambda +
    \Delta\lambda)`; with
    ``an.modified_NR = True`` the factorization at the last converged state
    is used for the first ``an.compute_every_n`` iterations and the tangent
    is then updated every ``an.compute_every_n`` iterations; at the first
    step with ``an.kT_initial_state = False`` the linear constitutive
    stiffness matrix replaces the tangent of the initial state.

    After each converged step, the arc length is multiplied by
    ``sqrt(DESIRED_NUM_ITER/num_iter)``, bounded by ``MIN_GROWTH_FACTOR`` and
    ``MAX_GROWTH_FACTOR`` and limited by ``an.maxInc``, where ``num_iter`` is
    the number of corrections of the step. The factor is at most 1 after a
    step that was repeated, and the limit is in fact
    ``max(an.maxInc, initial arc length)``. When a step fails, the arc length
    is multiplied by ``INC_CUT_FACTOR`` and the step is repeated from the
    same tangent predictor direction. The square root of the ratio is the
    rule recommended by Ramm [3]_ (p. 76), who found that the ratio itself,
    as used by Crisfield [2]_ (Eq. 15), makes the number of iterations
    oscillate from step to step.

    The tangent stiffness matrix is factorized with ``an.linear_solver``,
    see :func:`.static._factorize`. With ``an.inertia = True``, it is
    factorized at every converged state, the factorization being that of
    the predictor of the next step, and its number of negative eigenvalues
    and the sign of its determinant are recorded in
    ``an.num_negative_eigvals`` and ``an.det_signs``, see
    :func:`.newton_raphson._converged_state`: on a stable branch without
    prestress-induced instabilities the count is zero, it increases by one
    at each limit point of the load beyond which the branch is unstable,
    and at each bifurcation point crossed without branching. A perturbation
    whose shape keeps the path on a branch of a given symmetry, e.g.
    axisymmetric, may thus leave many negative eigenvalues unnoticed
    without this count.

    When the load factor 1.0 is exceeded within a converged step, from
    `\lambda < 1` to `\lambda + \Delta\lambda > 1`, the step is replaced by
    Newton-Raphson iterations at exactly `\lambda = 1`, see
    :func:`.newton_raphson._NR_iterations`, from the predictor
    `t \Delta \boldsymbol{c}` with `t = (1 - \lambda)/\Delta\lambda`. When these
    iterations do not converge, the step is repeated with a shorter arc
    length.

    When the load depends on the configuration, i.e. ``calc_fext`` accepts
    the keyword argument ``c``, `\boldsymbol{F}_{ext}` in `\delta \boldsymbol{c}_q` and in the
    tangent predictor is the load vector of the current configuration
    ``calc_fext(inc=1., c=c)``, minus the derivative `\partial \boldsymbol{R}/\partial
    \lambda` of the residual, whereas `\boldsymbol{u}_{ref}` remains the linear
    solution of the reference vector. The callables that accept ``inc``
    receive the current load factor, see :mod:`.callbacks`.

    When ``calc_fext`` also accepts the keyword argument ``lbd``, e.g. for
    prescribed displacements imposed exactly, it receives the load factor
    together with the configuration, `\boldsymbol{F}_{ext}` above being then
    `-\partial \boldsymbol{R}/\partial \lambda` at the current `c` and `\lambda`;
    `\boldsymbol{u}_{ref}` is the linear solution of `-\partial \boldsymbol{R}/\partial \lambda`
    at `c = 0`, `\lambda = 0`, since the reference vector may be null, e.g.
    under a prescribed displacement alone; and `-\lambda \, \partial
    \boldsymbol{R}/\partial \lambda` is an additional reference of the relative
    convergence criterion. `-\partial \boldsymbol{R}/\partial \lambda` is evaluated
    together with the tangent stiffness matrix, at the same iterate, i.e.
    once per update of the tangent stiffness matrix, and used in the
    bordered solution; the last one evaluated is the reference of the next
    convergence checks. At the converged state it is kept with the tangent
    stiffness matrix when the step is repeated with a shorter arc length.

    References
    ----------
    .. [1] Riks, E., An incremental approach to the solution of snapping and
       buckling problems, International Journal of Solids and Structures,
       15(7), 529-551, 1979. https://doi.org/10.1016/0020-7683(79)90081-7
    .. [2] Crisfield, M.A., A fast incremental/iterative solution procedure
       that handles "snap-through", Computers & Structures, 13(1-3), 55-62,
       1981. https://doi.org/10.1016/0045-7949(81)90108-5
    .. [3] Ramm, E., Strategies for tracing the nonlinear response near
       limit points, in: Wunderlich, W., Stein, E., Bathe, K.J. (eds.),
       Nonlinear Finite Element Analysis in Structural Mechanics, Springer,
       63-89, 1981. https://doi.org/10.1007/978-3-642-81589-8_5
    .. [4] Crisfield, M.A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.
    .. [5] de Borst, R., Computation of post-bifurcation and post-failure
       behavior of strain-softening solids, Computers & Structures, 25(2),
       211-224, 1987. https://doi.org/10.1016/0045-7949(87)90144-1

    """
    if method not in ('riks', 'crisfield', 'displacement_control'):
        raise ValueError('Invalid arc-length method: %s' % method)
    control = method == 'displacement_control'
    if an.arc_length_predictor not in PREDICTORS:
        raise ValueError('Invalid arc_length_predictor: %s'
                         % an.arc_length_predictor)
    secant = an.arc_length_predictor == 'secant'
    modified_NR = an.modified_NR

    fext = an.calc_fext(inc=1., silent=True)
    kC0 = an.calc_kC(silent=silent)
    w = _weights(an, kC0)
    K0_abs = _abs_matrix(kC0)
    lbd_dependent = callbacks.is_load_factor_dependent(an)
    if lbd_dependent:
        # -dR/dlbd at c = 0 and lbd = 0
        q0 = callbacks.calc_load_vector(an, np.zeros_like(fext), fext, 0.)
        u_ref = solve(kC0, q0, silent=True)
    else:
        u_ref = solve(kC0, fext, silent=True)
    if an.arc_length_weights is None:
        wa = np.ones_like(u_ref)
    else:
        wa = np.asarray(an.arc_length_weights, dtype=float)
        if wa.shape != u_ref.shape:
            raise ValueError('Analysis.arc_length_weights must have the shape %s'
                             % (u_ref.shape,))
        if not np.all(wa >= 0):
            raise ValueError('Analysis.arc_length_weights must not be negative')
    scale2 = u_ref.dot(wa*u_ref)
    if not scale2 > 0:
        scale2 = 1.
    psi2 = PSI**2

    def inner(dc1, dlbd1, dc2, dlbd2):
        return dc1.dot(wa*dc2)/scale2 + psi2*dlbd1*dlbd2

    if control:
        if an.control_vector is None:
            raise ValueError("NL_method='displacement_control' requires "
                             "Analysis.control_vector")
        psi_c = np.asarray(an.control_vector, dtype=float)
        if psi_c.shape != u_ref.shape:
            raise ValueError('Analysis.control_vector must have the shape %s'
                             % (u_ref.shape,))
        if an.control_scale is None:
            s_ref = abs(psi_c.dot(u_ref))
            if not s_ref > 0:
                raise ValueError('The controlled quantity is null in the '
                                 'linear solution, Analysis.control_scale '
                                 'must be given')
        else:
            s_ref = float(an.control_scale)

        def ctrl(dc):
            # the controlled quantity, dimensionless
            return psi_c.dot(dc)/s_ref

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

        # the secant predictor with full Newton-Raphson does not use the
        # tangent stiffness matrix at the last converged state, whose
        # factorization is skipped, unless -dR/dlbd depends on the load
        # factor and must be evaluated together with the tangent
        use_secant = secant and dc_last is not None
        need_kT_conv = not use_secant or modified_NR or lbd_dependent

        # tangent stiffness matrix at the last converged state
        if solve_kT_conv is None and need_kT_conv:
            if step_num == 1 and modified_NR and not an.kT_initial_state:
                kT = kC0
            else:
                kT = callbacks.calc_kT(an, c, lbd)
            try:
                solve_kT_conv = _factorize(kT, linear_solver=an.linear_solver)
            except RuntimeError:
                warn('Singular tangent stiffness matrix at the last converged state! Analysis stopped',
                     level=1, silent=silent)
                break
            # load vector of the converged configuration, -dR/dlbd, kept with
            # the tangent stiffness matrix when the step is repeated
            if step_num == 1 and lbd_dependent:
                q_conv = q0
            else:
                q_conv = callbacks.calc_load_vector(an, c, fext, lbd)
        if solve_kT_conv is not None:
            dc_q_conv = solve_kT_conv(q_conv)
        if arc_length is None:
            if control:
                # the first step is initialInc in units of the scale, the
                # controlled quantity increasing with the load factor along
                # the initial tangent
                norm_q = 1.
                ctrl_sign = np.sign(ctrl(dc_q_conv)) or 1.
            else:
                norm_q = np.sqrt(inner(dc_q_conv, 1., dc_q_conv, 1.))
            arc_length = an.initialInc*norm_q
            max_arc_length_inc = max(an.maxInc, arc_length)

        if control:
            # psi^T dc = ds
            ds = ctrl_sign*arc_length
            if use_secant:
                den = ctrl(dc_last)
                dc_dir, dlbd_dir = dc_last, dlbd_last
            else:
                den = ctrl(dc_q_conv)
                dc_dir, dlbd_dir = dc_q_conv, 1.
            if den == 0:
                warn('The controlled quantity does not change along the predictor! Analysis stopped',
                     level=1, silent=silent)
                break
            dlbd = dlbd_dir*ds/den
            dc = (ds/den)*dc_dir
        elif use_secant:
            # secant predictor: the previous converged increment scaled to
            # the arc length
            factor = arc_length/np.sqrt(inner(dc_last, dlbd_last, dc_last,
                                              dlbd_last))
            dc = factor*dc_last
            dlbd = factor*dlbd_last
        else:
            # tangent predictor
            norm_q = np.sqrt(inner(dc_q_conv, 1., dc_q_conv, 1.))
            sign = 1.
            if dc_last is not None:
                sign = np.sign(inner(dc_q_conv, 1., dc_last, dlbd_last))
                if sign == 0:
                    sign = 1.
            dlbd = sign*arc_length/norm_q
            dc = dlbd*dc_q_conv
        # the predictor, the forward direction of the step
        dc_pred = dc
        dlbd_pred = dlbd

        msg('Step %d, lbd %1.5f, arc-length increment %1.5f' % (step_num, lbd, arc_length),
            level=1, silent=silent)

        converged = False
        Rnorms = []
        solve_kT = solve_kT_conv
        if solve_kT_conv is not None:
            dc_q = dc_q_conv
            # the last -dR/dlbd evaluated, with the tangent stiffness matrix
            q_iter = q_conv
        iter_kT = 0
        iteration = 0
        fint = None
        while True:
            iteration += 1
            if fint is None:
                fint = callbacks.calc_fint(an, c + dc, lbd + dlbd)
            fext_total = (lbd + dlbd)*fext
            R = fint - fext_total
            q_ref = None
            if lbd_dependent:
                q_ref = (lbd + dlbd)*q_iter
            conv, Rnorm, Rrel, roundoff = _check_convergence(an, R, fint,
                    fext_total, fext_ref=dlbd*fext, q_ref=q_ref, w=w,
                    Kc_abs=K0_abs @ np.abs(c + dc), full_output=True)
            _log_iteration(an, iteration, R, Rrel, silent, lbd=lbd + dlbd,
                           roundoff=roundoff)
            if conv:
                converged = True
                break
            if _check_divergence(an, Rnorm, Rnorms, iteration, silent=silent,
                                 Rrel=Rrel):
                break

            # with modified_NR the tangent of the last converged state is
            # used first, otherwise it is updated at every iteration
            if modified_NR and iter_kT < an.compute_every_n:
                iter_kT += 1
            else:
                try:
                    solve_kT = _factorize(callbacks.calc_kT(an, c + dc,
                                                            lbd + dlbd),
                                          linear_solver=an.linear_solver)
                except RuntimeError:
                    warn('Singular tangent stiffness matrix', level=2, silent=silent)
                    break
                if lbd_dependent:
                    # -dR/dlbd at the iterate of the tangent stiffness matrix
                    q_iter = callbacks.calc_load_vector(an, c + dc, fext,
                                                        lbd + dlbd)
                    dc_q = solve_kT(q_iter)
                else:
                    dc_q = solve_kT(callbacks.calc_load_vector(an, c + dc,
                                                               fext))
                iter_kT = 1
            dc_R = solve_kT(-R)

            varlbd = None
            if control:
                # psi^T (dc + dc_R + varlbd dc_q) = ds
                den = ctrl(dc_q)
                if den != 0:
                    varlbd = (ds - ctrl(dc + dc_R))/den
            else:
                # linearized constraint
                den = inner(dc, dlbd, dc_q, 1.)
                g = 0.5*(inner(dc, dlbd, dc, dlbd) - arc_length**2)
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
            dc_corr = dc_R + varlbd*dc_q
            eta = 1.
            fint = None
            if an.arc_length_line_search:
                # backtracking on the whole correction, the first factor
                # that decreases the norm of the residual is accepted,
                # otherwise the one with the smallest norm
                best = None
                for eta_try in LINE_SEARCH_ETAS:
                    lbd_try = lbd + dlbd + eta_try*varlbd
                    fint_try = callbacks.calc_fint(an, c + dc + eta_try*dc_corr,
                                                   lbd_try)
                    Rnorm_try = _norm(fint_try - lbd_try*fext, w)
                    if best is None or Rnorm_try < best[1]:
                        best = (eta_try, Rnorm_try, fint_try)
                    if Rnorm_try < Rnorm:
                        break
                eta, _, fint = best
                if eta != 1.:
                    msg('Line-search: eta = %1.5f' % eta, level=3, silent=silent)
            dc = dc + eta*dc_corr
            dlbd = dlbd + eta*varlbd

        if (converged and not control
                and inner(dc, dlbd, dc_pred, dlbd_pred) <= 0):
            # the increment points backwards, e.g. to the previous converged
            # point, which is also at the distance arc_length
            warn('Diverged - the step returned along the path already traced',
                 level=2, silent=silent)
            converged = False

        finished = False
        if converged and lbd + dlbd > 1. and lbd < 1.:
            msg('Load factor 1.0 exceeded, correcting to load factor 1.0...', level=1, silent=silent)
            t = (1. - lbd)/dlbd
            converged, dc1, _ = _NR_iterations(an, c, t*dc, 1., fext,
                    kT0=None, silent=silent, lbd_dependent=lbd_dependent,
                    q=q_iter, w=w, K0_abs=K0_abs)
            if converged:
                msg('Converged at load factor 1.00000', level=2, silent=silent)
                an.cs.append((c + dc1).copy())
                an.increments.append(1.)
                if an.inertia:
                    _converged_state(an, c + dc1, 1., silent=silent)
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
            solve_kT_conv = None
            if an.inertia:
                # the factorization at the converged state, reused by the
                # next step
                solve_kT_conv = _converged_state(an, c, lbd, silent=silent)
                if solve_kT_conv is not None:
                    q_conv = callbacks.calc_load_vector(an, c, fext, lbd)
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
