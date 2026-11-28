"""Stabilized Newton-Raphson solver for non-linear static analyses

Selected with ``Analysis.NL_method = 'stabilized_NR'``. A load-controlled
Newton-Raphson solver that passes the load factors at which the plain
iterations fail, e.g. local folds and mode jumps of the postbuckling of
stiffened panels, with viscous stabilization and a correction to the
undamped equilibrium. The implementation is shared with the
Newton-Raphson solver, see :mod:`.newton_raphson`.

"""
from .logger import msg
from .newton_raphson import _solver_NR


def _solver_stabilized_NR(an, silent=False):
    r"""Newton-Raphson solver with load control and viscous stabilization

    Used by :meth:`.Analysis.static` when ``NL_method='stabilized_NR'``.
    Runs :func:`.newton_raphson._solver_NR` with ``stabilized=True``: the
    load factor is incremented as with ``NL_method='NR'``, and every
    increment is solved in up to three stages:

    1. the Newton-Raphson iterations of
       :func:`.newton_raphson._NR_iterations` from the predictor
       extrapolated from the previous converged increment;
    2. when they fail, the iterations on the equilibrium with viscous
       forces proportional to the change of the solution since the last
       converged state, see
       :func:`.newton_raphson._stabilized_iterations`;
    3. when these converge, the Newton-Raphson iterations without viscous
       forces again, from the damped state.

    A state from stage 1 is recorded as ``'undamped'`` and one from stage 3
    as ``'corrected'`` in ``an.stabilization_status``; both satisfy the
    convergence criteria of the undamped equilibrium. When stage 3 fails,
    the damped state of stage 2 is accepted and recorded as ``'damped'``,
    with the relative residual of its undamped equilibrium in
    ``an.damped_residuals`` and a warning. When stage 2 fails, the load
    increment is multiplied by ``INC_CUT_FACTOR`` and the increment starts
    again at stage 1. The load increment grows by ``INC_GROWTH_FACTOR``
    only after an ``'undamped'`` state reached without a cut, and is kept
    after the other states.

    The viscous forces are scaled by ``an.stabilization_factor``. All the
    other options of ``NL_method='NR'`` apply to the three stages: the
    convergence criteria, the full or modified Newton-Raphson method, the
    line search, ``an.linear_solver``, the constraints and their values,
    the inertia, ``an.on_converged``, ``an.stop_criterion``,
    ``an.final_load_factor`` and ``an.restart_state``, the latter only
    from a ``path_state`` of ``'stabilized_NR'``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object, the accepted load factors and solutions are
        appended to ``an.increments`` and ``an.cs``, and their status to
        ``an.stabilization_status`` and ``an.damped_residuals``.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.

    Raises
    ------
    ValueError
        When ``an.stabilization_factor`` is not a positive finite number,
        when ``an.final_load_factor`` is ``None``, or for the invalid
        options of :func:`.newton_raphson._solver_NR`.

    Notes
    -----
    The equilibrium at the load factor `\lambda_{n+1} = \lambda_n +
    \Delta\lambda` is `\boldsymbol{R}(\boldsymbol{c}) =
    \boldsymbol{F}_{int}(\boldsymbol{c}) - \lambda_{n+1}
    \boldsymbol{F}_{ext} = \boldsymbol{0}`. Where the equilibrium path has
    a limit point or a fold in the load factor, or where the solution
    jumps to another buckling pattern, there is no equilibrium near the
    last converged state `\boldsymbol{c}_n`, and the Newton-Raphson
    iterations fail at every load increment beyond it, whereas the
    arc-length methods may turn back along the path instead of going past
    the jump.

    The stabilization treats the load factor as a pseudo-time `t` and adds
    to the internal force vector the viscous forces of the damping matrix
    `\mu \boldsymbol{D}`, discretized by the backward Euler method over the
    increment, `\Delta t = \Delta\lambda`:

    .. math::

        \boldsymbol{R}_v(\boldsymbol{c}) = \boldsymbol{F}_{int}(\boldsymbol{c})
        + \frac{\mu}{\Delta\lambda} \boldsymbol{D} (\boldsymbol{c} -
        \boldsymbol{c}_n) - \lambda_{n+1} \boldsymbol{F}_{ext} =
        \boldsymbol{0}, \qquad \boldsymbol{D} = \mathrm{diag}(|K_{C0,11}|,
        \ldots, |K_{C0,NN}|)

    with `\mu` = ``an.stabilization_factor``, dimensionless, and
    `\boldsymbol{K}_{C0}` the linear stiffness matrix ``calc_kC()``,
    whose diagonal makes the viscous forces independent of the units of
    each degree of freedom, as the weights of Crisfield's criterion, see
    :mod:`.convergence`. The Jacobian of `\boldsymbol{R}_v` is
    `\boldsymbol{K}_T + (\mu/\Delta\lambda) \boldsymbol{D}`: the diagonal
    term makes it regular where `\boldsymbol{K}_T` is singular or
    indefinite and resists a large change of the solution within one
    increment, such that the iterations converge to a nearby damped state
    instead of failing. A smaller `\Delta\lambda` increases the viscous
    coefficient `\mu/\Delta\lambda`. The damped iterations start from
    `\boldsymbol{c}_n`, or, with ``an.constraint_values``, from the
    predictor of stage 1, which satisfies the constraints at
    `\lambda_{n+1}`.

    The damped state does not satisfy `\boldsymbol{R} = \boldsymbol{0}`:
    the viscous forces are the residual of its undamped equilibrium. Stage
    3 removes them: the Newton-Raphson iterations of `\boldsymbol{R}` start
    from the damped state, which is closer to an equilibrium beyond the
    jump than the extrapolated predictor of stage 1. A corrected state is
    an equilibrium of `\boldsymbol{R}` to the tolerance of the convergence
    criteria, such that the viscous forces do not accumulate along the
    path. A damped-only state, ``'damped'`` in ``an.stabilization_status``,
    is not an equilibrium: its relative residual in ``an.damped_residuals``
    measures how far it is from one, and stresses or displacements of such
    a state must not be read as those of an equilibrium.

    The scheme does not guarantee:

    - that the equilibrium found after a jump is the one a dynamic process
      would reach: the viscous path is an artificial one, and another
      `\mu`, `\Delta\lambda` or predictor may give another equilibrium;
    - that the accepted states are stable: their stability is not checked,
      unless ``an.inertia`` records the inertia of the tangent stiffness
      matrix, see :func:`.newton_raphson._converged_state`;
    - that a state exists at every load factor: when the damped
      iterations fail, the increment is cut as with ``NL_method='NR'``,
      and the analysis stops at ``an.minInc``.

    The dissipated energy of the viscous forces relative to the strain
    energy or to the external work is not computed: the callables give
    neither the strain energy nor the work along the path between the
    converged states.

    The choices of this solver are the damping matrix from the diagonal of
    the linear stiffness matrix, the load factor as the pseudo-time, the
    start of the damped iterations from the last converged state and the
    correction of stage 3. The full and modified Newton-Raphson iterations
    of the three stages are those of [1]_ (Sections 1.2.2 and 1.2.3,
    pp. 8-13).

    References
    ----------
    .. [1] Crisfield, M.A., Non-linear Finite Element Analysis of Solids and
       Structures, Volume 1: Essentials, John Wiley & Sons, Chichester,
       1991.

    """
    msg('___________________________________________', level=1, silent=silent)
    msg('                                           ', level=1, silent=silent)
    msg('Stabilized Newton-Raphson solver', level=1, silent=silent)
    msg('___________________________________________', level=1, silent=silent)
    msg('Initializing...', level=1, silent=silent)
    _solver_NR(an, silent=silent, initialInc=None, stabilized=True)
