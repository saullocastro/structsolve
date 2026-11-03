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
                             _constraint_matrix, _linear_solve,
                             _residual_projector, _stop, _factorize_state,
                             _record_inertia, _check_max_inertia_jump,
                             _inertia_jump, _restart_state, _cache,
                             _constraint_values, INC_CUT_FACTOR)


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
#: Rules of the sign of the tangent predictor,
#: ``Analysis.arc_length_sign_rule``
SIGN_RULES = ('inner_product', 'determinant')
#: Methods of :func:`._solver_arc_length`
METHODS = ('riks', 'crisfield', 'displacement_control',
           'generalized_displacement_control')
#: Minimum cosine between the converged increment of a branch-switching
#: step and its predictor, the critical eigenvector
BRANCH_SWITCH_MIN_COS = 0.5
#: Number of attempts of a branch-switching step, each with the arc length
#: multiplied by ``INC_CUT_FACTOR``, before the switch is abandoned
BRANCH_SWITCH_MAX_TRIES = 3
#: Number of eigenpairs computed besides the negative eigenvalues by the
#: subspace iteration of the critical eigenvector
EIGVEC_NUM_GUARD = 2
#: Maximum number of iterations of the subspace iteration
EIGVEC_MAX_ITER = 100
#: Relative tolerance of the eigenvalues of the subspace iteration
EIGVEC_TOL = 1.e-10
#: ``Analysis.bifurcation_pinning``: largest cosine between the critical
#: eigenvector and the load vector of a bifurcation mode
PIN_MAX_COS = 1.e-3
#: ``Analysis.bifurcation_pinning``: maximum number of iterations of the
#: subspace iteration of the critical eigenvector
PIN_MAX_ITER = 20
#: Largest subspace of :func:`._critical_eigenvector`, whose size is
#: doubled until it holds the negative eigenvalues
EIGVEC_MAX_SIZE = 64


def _eigs_near_zero(solve_kT, size, p, max_iter=None, start=None):
    r"""The ``p`` eigenvalues of `\boldsymbol{K}_T` closest to zero and their
    eigenvectors, by the subspace iteration of :func:`._critical_eigenvector`,
    with at most ``max_iter`` iterations, ``EIGVEC_MAX_ITER`` by default,
    from the vectors ``start``, columns, completed by random vectors. With
    ``'pardiso'`` the solutions of the factorization are used without the
    check of their residual and GMRES, see :func:`.static._factorize`: the
    tangent stiffness matrix near a critical point is almost singular, the
    GMRES of its solutions took about one second each for the DOE09
    cylinder (57000 degrees of freedom), against 0.03 s for those of the
    factorization alone, and the subspace iteration converges to the
    eigenvectors closest to zero without them.

    Returns
    -------
    mus : array-like
        The eigenvalues, not sorted.
    vecs : array-like
        The eigenvectors, the columns, orthonormal.

    """
    solve = getattr(solve_kT, 'unrefined', solve_kT)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((p, size))
    if start is not None:
        k = min(start.shape[1], p)
        X[:k] = start[:, :k].T
    Y = np.column_stack([solve(x) for x in X])
    theta_old = None
    for _ in range(EIGVEC_MAX_ITER if max_iter is None else max_iter):
        Q, _ = np.linalg.qr(Y)
        Y = np.column_stack([solve(q) for q in Q.T])
        H = Q.T @ Y
        theta, V = np.linalg.eigh(0.5*(H + H.T))
        if (theta_old is not None
                and np.all(np.abs(theta - theta_old)
                           <= EIGVEC_TOL*np.abs(theta).max())):
            break
        theta_old = theta
    keep = theta != 0
    return 1./theta[keep], (Q @ V)[:, keep]


def _critical_eigenvector(solve_kT, size, num_neg, num_new):
    r"""Eigenvector of the tangent stiffness matrix whose eigenvalue just
    became negative

    Subspace iteration with the inverse of the factorization ``solve_kT``,
    see :func:`.static._factorize`, and the Rayleigh-Ritz projection of the
    inverse on the orthonormalized subspace, which gives the eigenvalues of
    `\boldsymbol{K}_T` closest to zero and their eigenvectors in the
    Euclidean inner product, see :func:`._eigs_near_zero`. The subspace has
    ``num_neg + EIGVEC_NUM_GUARD`` vectors, ``num_neg`` being the number of
    negative eigenvalues of `\boldsymbol{K}_T`, and is iterated until the
    Ritz values change by less than ``EIGVEC_TOL``, at most
    ``EIGVEC_MAX_ITER`` times. When fewer than ``num_neg`` of the Ritz
    values are negative, i.e. positive eigenvalues are closer to zero than
    some negative ones, e.g. of weakly stiff terms of a Ritz model, the
    size of the subspace is doubled and the iteration repeated, up to
    ``EIGVEC_MAX_SIZE`` vectors. With ``an.constraints`` the factorization
    is the bordered one, whose solutions satisfy the constraints, and the
    eigenpairs are those of `\boldsymbol{K}_T` restricted to them.

    Of the negative eigenvalues found, the ``num_new`` closest to zero are
    taken as those that became negative in the last step, and the most
    negative of them is the critical one, the first that became negative.
    Without negative eigenvalue, the one closest to zero is taken. With
    ``num_new < 0``, ``-num_new`` eigenvalues became positive in the last
    step, the subspace has ``-num_new`` more vectors, and the critical one
    is the largest of the ``-num_new`` smallest positive eigenvalues, the
    first that became positive.

    Returns
    -------
    phi : array-like
        The eigenvector, of unit Euclidean norm.
    mu : float
        Its eigenvalue.

    """
    p = min(max(num_neg, 1) + max(-num_new, 0) + EIGVEC_NUM_GUARD, size)
    vecs = None
    while True:
        mus, vecs = _eigs_near_zero(solve_kT, size, p, start=vecs)
        p_max = min(max(EIGVEC_MAX_SIZE, p), size)
        if (mus < 0).sum() >= num_neg or p >= p_max:
            break
        p = min(2*p, p_max)
    neg = np.flatnonzero(mus < 0)
    pos = np.flatnonzero(mus > 0)
    if num_new < 0 and pos.size > 0:
        # the positive eigenvalues sorted from the closest to zero
        new = pos[np.argsort(mus[pos])][:-num_new]
        i = int(new[np.argmax(mus[new])])
    elif neg.size == 0:
        i = int(np.argmin(np.abs(mus)))
    else:
        # the negative eigenvalues sorted from the closest to zero
        neg = neg[np.argsort(-mus[neg])]
        new = neg[:max(num_new, 1)]
        i = int(new[np.argmin(mus[new])])
    phi = vecs[:, i]
    return phi/np.linalg.norm(phi), float(mus[i])


def _pin_bifurcation_mode(an, c, lbd, fext, C, d, solve_conv=None,
                          q_conv=None, silent=False):
    r"""The constraints of ``Analysis.bifurcation_pinning`` at the converged
    state ``c``, ``lbd``

    The eigenvector `\boldsymbol{\phi}` of the eigenvalue of
    `\boldsymbol{K}_T` closest to zero, by the subspace iteration of
    :func:`._eigs_near_zero` with the factorization ``solve_conv`` of the
    converged state (computed when ``None``), is pinned when it is
    orthogonal to the load vector `\boldsymbol{q}` (``q_conv``),
    `|\boldsymbol{\phi}^\top \boldsymbol{q}| \le` ``PIN_MAX_COS``
    `\|\boldsymbol{q}\|`, i.e. a bifurcation mode: the constraint
    `\boldsymbol{\phi}^\top \Delta \boldsymbol{c} = 0` is added to the
    constraints ``C`` and their values ``d``.

    Returns
    -------
    pinned : tuple or None
        ``(phi, C_pin, d_pin, solve_pin, q_conv)``, ``solve_pin`` the
        factorization of the converged state with the constraints
        ``C_pin``; ``None`` when the eigenvector is not orthogonal to the
        load or the factorization failed.

    """
    from scipy.sparse import hstack, csc_matrix
    try:
        if solve_conv is None:
            solve_conv = _factorize(callbacks.calc_kT(an, c, lbd),
                                    linear_solver=an.linear_solver,
                                    constraints=C, cache=_cache(an))
        if q_conv is None:
            q_conv = callbacks.calc_load_vector(an, c, fext, lbd)
        nneg = solve_conv.num_negative_eigvals or 0
        size = c.shape[0]
        mus, vecs = _eigs_near_zero(solve_conv, size,
                                    min(max(nneg, 1) + EIGVEC_NUM_GUARD, size),
                                    max_iter=PIN_MAX_ITER)
        i = int(np.argmin(np.abs(mus)))
        phi = vecs[:, i]/np.linalg.norm(vecs[:, i])
        qnorm = np.linalg.norm(q_conv)
        cos = abs(phi.dot(q_conv))/qnorm if qnorm > 0 else 0.
        if not cos <= PIN_MAX_COS:
            msg('Bifurcation pinning: the critical eigenvector, of the '
                'eigenvalue %1.5e, is not orthogonal to the load (cosine '
                '%1.1e)' % (mus[i], cos), level=2, silent=silent)
            return None
        col = csc_matrix(phi[:, None])
        C_pin = col if C is None else hstack([C, col], format='csc')
        d_pin = None if d is None else np.r_[d, 0.]
        solve_pin = _factorize(callbacks.calc_kT(an, c, lbd),
                               linear_solver=an.linear_solver,
                               constraints=C_pin, cache=_cache(an))
    except RuntimeError:
        return None
    msg('Bifurcation pinning: the amplitude of the critical eigenvector, of '
        'the eigenvalue %1.5e and cosine %1.1e with the load, is held for the '
        'step' % (mus[i], cos), level=1, silent=silent)
    return phi, C_pin, d_pin, solve_pin, q_conv


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

    - the load factor reaches ``an.final_load_factor``, 1.0 by default, the
      last solution being computed at exactly this load factor; never with
      ``an.final_load_factor = None``
    - ``an.stop_criterion(an)`` returns ``True`` after a converged step, see
      :mod:`.stopping`
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
        ``'riks'``, ``'crisfield'``, ``'displacement_control'`` or
        ``'generalized_displacement_control'``.
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

    - ``method='generalized_displacement_control'``: the generalized
      displacement control of Yang and Shieh [6]_. With `\delta
      \boldsymbol{c}_q^{(k)}` the tangent `\boldsymbol{K}_T^{-1}
      \boldsymbol{F}_{ext}` at the start of step `k` and `\langle a, b
      \rangle_W = \boldsymbol{a}^\top \boldsymbol{W} \boldsymbol{b}`, with
      the weights of ``an.arc_length_weights``, the predictor of step `k`
      is `\Delta\lambda = s_k \Delta s \sqrt{|GSP_k|}`, `\Delta
      \boldsymbol{c} = \Delta\lambda \, \delta \boldsymbol{c}_q^{(k)}`,
      with the generalized stiffness parameter

      .. math::

          GSP_k = \frac{\langle \delta c_q^{(1)}, \delta c_q^{(1)}
          \rangle_W}{\langle \delta c_q^{(k-1)}, \delta c_q^{(k)}
          \rangle_W}, \qquad GSP_1 = 1

      which decreases with the stiffness along the path and changes sign
      where the tangent turns back, at the limit points of the load; the
      sign `s_k = s_{k-1} \, \mathrm{sign}(GSP_k)`, `s_1 = 1`, then
      reverses the load increment. The corrections satisfy

      .. math::

          \langle \delta c_q^{(k-1)}, \delta \boldsymbol{c}_R + \delta\lambda
          \, \delta \boldsymbol{c}_q \rangle_W = 0, \qquad
          \delta\lambda = -\frac{\langle \delta c_q^{(k-1)}, \delta
          c_R \rangle_W}{\langle \delta c_q^{(k-1)}, \delta c_q
          \rangle_W}

      with `\delta c_q^{(0)} = \delta c_q^{(1)}`, i.e. they are orthogonal
      to the tangent of the previous step, which does not depend on the
      units of the load nor on a scale of the displacements. The step
      length `\Delta s` is a load factor, ``an.initialInc`` the first one
      and the largest, cut and adapted as the arc length, but never
      increased beyond ``an.initialInc``, since growing it as the arc length
      would defeat the scaling by `GSP_k`, which shortens the steps where
      the structure softens: with longer steps the iterations may converge
      beyond a limit point on another part of the path. ``an.maxArcLength``
      bounds the sum of the `\Delta s`. With ``an.arc_length_weights =
      None``, `GSP_k`, the predictor and `\delta\lambda` are Eqs. (33),
      (34) and (30) of [6]_, with `\delta c_q^{(0)} = \delta c_q^{(1)}` as
      there. [6]_ multiplies the load increment by `-1` when `GSP_k < 0`,
      its step 2d, which keeps the displacements going forward only if the
      reversal is relative to the previous step, `s_k = s_{k-1} \,
      \mathrm{sign}(GSP_k)` as here, since `GSP_k` is negative only at
      the step after each limit point. The weights and the adaptation of
      `\Delta s` are additions of this implementation: in [6]_ `\Delta s`
      is the first load increment throughout. The check against the
      doubling back below does not apply.

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

    With ``an.max_inertia_jump`` `= j`, which requires ``an.inertia`` and
    ``an.linear_solver = 'pardiso'``, a converged step whose number of
    negative eigenvalues differs by more than `j` from that of the last
    converged state is repeated with a shorter arc length, as a step that
    failed, unless the shorter arc length would be smaller than
    ``an.minInc``. With `j = 1` the eigenvalues cross zero one at a time,
    and a predictor that steps over several critical points, e.g. the
    secant predictor on a cylinder under axial compression, is cut back
    to them.

    With ``an.arc_length_sign_rule = 'determinant'``, which requires
    ``an.inertia``, the sign of the tangent predictor is the product of the
    signs of the determinant of the tangent stiffness matrix at the last
    converged state and at the origin, instead of the sign of the inner
    product above: the load increases on the branches whose determinant has
    the sign of the initial, stable, state, and decreases on the others.
    The load is thus reversed at every critical point crossed, at the limit
    points as at the bifurcations, and the path cannot turn back on a
    branch, e.g. towards the origin, without crossing a critical point.
    It suits paths whose critical points are crossed one at a time, e.g.
    with ``an.max_inertia_jump = 1``, and whose bifurcations are switched,
    see below; at a bifurcation crossed without switching, the path
    returns to it. After a branch switch, and when a determinant is null,
    the inner product decides.

    With ``an.branch_switching`` `= n > 0`, which requires ``an.inertia``,
    the path is switched to the bifurcated branch at the first `n`
    bifurcation points crossed, with every method. A critical point is
    crossed when the number of negative eigenvalues changes between two
    converged states, increases or decreases, or, when it is not known
    (``an.linear_solver = 'splu'``), when the sign of the determinant
    changes. It is a bifurcation point when the load factor varies in the
    same direction along the tangent predictors of the inner-product rule
    at the two converged states, before and after the critical point, a
    limit point when the direction is reversed. At a bifurcation, the
    eigenvector `\boldsymbol{\phi}` of the eigenvalue that became negative,
    or positive when the count decreased, is computed with
    :func:`._critical_eigenvector`, with the sign for which
    `\boldsymbol{\phi}^\top \boldsymbol{c} \ge 0`, i.e. the direction
    of the deformation already present, e.g. of an imperfection, and the
    next step starts from the predictor `(\Delta s \hat{\phi}, 0)` along it,
    `\hat{\phi}` normalized in the inner product above, with `\Delta s`
    the larger of the current arc length and that of the last step, which
    crossed the critical point. With ``method='displacement_control'`` and
    ``'generalized_displacement_control'``, whose step lengths are not arc
    lengths, `\Delta s` is the length of the last increment in the inner
    product above, multiplied by the growth of the step length since then
    when it grew, as for the arc length. The step is corrected with the
    spherical constraint of ``method='crisfield'`` of radius `\Delta s`,
    whose root closest to the current
    increment is on the bifurcated branch, which passes through the critical
    point, at a distance smaller than `\Delta s`, and is tangent to
    `\boldsymbol{\phi}` at it, whereas the linearized constraint of
    ``method='riks'`` is singular there for a symmetric bifurcation. The
    step is accepted when the cosine between its increment and the
    predictor is at least ``BRANCH_SWITCH_MIN_COS``, otherwise it is
    repeated with a shorter `\Delta s`, at most ``BRANCH_SWITCH_MAX_TRIES``
    times, after which the switch is abandoned and the path continues on
    the branch already traced. The index in ``an.increments`` of the state
    from which the path was switched is appended to ``an.branch_points``.
    The step after the switch starts along the tangent in the direction of
    the switching increment: with the inner-product rule for
    ``method='riks'`` and ``'crisfield'``, also with
    ``an.arc_length_sign_rule = 'determinant'``; with
    ``'displacement_control'``, the sign of `\Delta s` for which the
    predictor makes a positive inner product with the switching increment;
    with ``'generalized_displacement_control'``, the same sign of the load
    increment, `GSP_k = \langle \delta c_q^{(1)}, \delta c_q^{(1)}
    \rangle_W / \langle \delta c_q^{(k)}, \delta c_q^{(k)} \rangle_W`, i.e.
    `\delta c_q^{(k-1)}` replaced by `\delta c_q^{(k)}`, and the
    corrections orthogonal to `\delta c_q^{(k)}`, as at the first step,
    since the tangent before the switch is that of the other branch.
    The detection by the count of negative pivots and the switch along the
    critical eigenvector follow Wagner and Wriggers [7]_, whose algorithm
    (p. 106, step (c)) detects a stability point when the number of
    negative diagonal elements of the factorization changes, and whose
    classification by the current stiffness parameter, zero at the limit
    points and not at the bifurcations, their Eq. (22), is replaced here by
    the direction of the load along the tangents. In [7]_ the critical
    point is first located by bisection, the eigenvector is added to the
    displacements of the last converged state, `\bar{\boldsymbol{v}} +
    \xi \boldsymbol{\phi}/\|\boldsymbol{\phi}\|` with `|\xi| =
    \|\bar{\boldsymbol{v}}\|/\tau` and `\tau` of the order of 100, Eqs.
    (24) and (25), and an equilibrium state is then found by the arc-length
    method, repeated with another `\tau`, up to about three times, when it
    falls back on the primary branch. Here the critical point is not
    located, the perturbation is the predictor of an arc-length step from
    the first state beyond the critical point, with the length of the
    arc-length steps instead of `|\xi|`, and the shorter steps of the
    repeated attempts play the role of the other values of `\tau`.

    With ``an.bifurcation_pinning`` `= n > 0`, a step whose iterations
    failed `n` times in a row, a step accepted without any correction not
    breaking the count, is repeated with the amplitude of the critical
    eigenvector `\boldsymbol{\phi}` held: the eigenvector of the eigenvalue
    of the tangent stiffness matrix at the last converged state closest to
    zero, by :func:`._eigs_near_zero`, when it is orthogonal to the load
    vector `\boldsymbol{q}`, `|\boldsymbol{\phi}^\top \boldsymbol{q}| \le`
    ``PIN_MAX_COS`` `\|\boldsymbol{q}\|`, i.e. a bifurcation mode, see
    :func:`._pin_bifurcation_mode`. The constraint `\boldsymbol{\phi}^\top
    \Delta \boldsymbol{c} = 0` is added to ``an.constraints`` for the
    iterations of that step only, its predictor included; the converged
    state satisfies the equilibrium equations to the tolerance, the
    residual along `\boldsymbol{\phi}` included, and the count of the
    negative eigenvalues is that of the unconstrained tangent. Near a
    bifurcation point the correction of the load factor of the arc-length
    and displacement control methods is `\delta\lambda \approx -r_\phi/
    q_\phi`, with `r_\phi` and `q_\phi` the components of the residual and
    of the load vector along `\boldsymbol{\phi}`, both small, of the order
    of the residual left by the tolerance and of the asymmetry of the load,
    and independent of the step length: shorter steps do not converge,
    and the steps accepted without correction creep along the predictor,
    their residual growing to the tolerance. Holding the amplitude of
    `\boldsymbol{\phi}` removes this division, and the step crosses the
    bifurcation point on the branch already traced, also when a tiny
    asymmetry of the model separates it from the crossing branch by less
    than the tolerance; the next steps are free. On the DOE09 cylinder
    (case 510, 57000 degrees of freedom, ``max_inertia_jump = 1``) the
    stall at 4526.7 N was such a point: the eigenvalue closest to zero went
    from `-9.3 \times 10^{-7}` to `-5.0 \times 10^{-8}` and `+4.8 \times
    10^{-9}` (scaled tangent) between the states of the stall, its
    eigenvector at `5.6 \times 10^{-8}` (cosine) from the load and, at the
    end, along the increments, which crept at the relative residual
    `10^{-4}` of the tolerance, the corrections going to `\delta\lambda
    \approx -9 \times 10^{-4}` at every step length.

    With ``an.restart_state``, the dictionary ``an.path_state`` saved after
    a converged step of an analysis with the same method, the analysis
    continues from that step, with its arc length and the directions of its
    predictors, as the analysis that saved it would have, appending the
    next converged increments to the empty ``an.increments`` and ``an.cs``.
    ``an.path_state`` holds arrays and numbers, e.g. to be saved with
    :mod:`pickle` and the analysis restarted with other options or after
    the end of a job.

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

    With ``an.constraints`` and ``an.constraint_values``, the constraints
    `\boldsymbol{C}^\top \boldsymbol{c} = \lambda \boldsymbol{d}`, e.g. a
    displacement prescribed in proportion to the load factor, linear in
    `(\boldsymbol{c}, \lambda)`: the load solutions `\delta
    \boldsymbol{c}_q` (and `\boldsymbol{u}_{ref}`) are those of the system
    bordered by `\boldsymbol{d}`, `\boldsymbol{C}^\top \delta
    \boldsymbol{c}_q = \boldsymbol{d}`, and the residual solutions `\delta
    \boldsymbol{c}_R` those of the null border, see
    :func:`.static._factorize`, such that every predictor, tangent, secant,
    of the displacement controls or the jump to `\lambda = 1`, and every
    correction `\delta \boldsymbol{c}_R + \delta\lambda \, \delta
    \boldsymbol{c}_q` satisfies `\boldsymbol{C}^\top \Delta \boldsymbol{c}
    = \Delta\lambda \boldsymbol{d}`; the eigenvector predictor of the
    branch switching, at a constant load, has the null border. Under a
    prescribed displacement alone, `\boldsymbol{F}_{ext} = \boldsymbol{0}`,
    `\boldsymbol{u}_{ref}` is the displacement of the linear solution, not
    null, and the reactions of the constraints, in the internal force
    vector, are the reference of the convergence criterion.

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
    .. [6] Yang, Y.-B., Shieh, M.-S., Solution method for nonlinear problems
       with multiple critical points, AIAA Journal, 28(12), 2110-2116, 1990.
       https://doi.org/10.2514/3.10529
    .. [7] Wagner, W., Wriggers, P., A simple method for the calculation of
       postcritical branches, Engineering Computations, 5(2), 103-109, 1988.
       https://doi.org/10.1108/eb023727

    """
    if method not in METHODS:
        raise ValueError('Invalid arc-length method: %s' % method)
    control = method == 'displacement_control'
    gdc = method == 'generalized_displacement_control'
    if an.arc_length_predictor not in PREDICTORS:
        raise ValueError('Invalid arc_length_predictor: %s'
                         % an.arc_length_predictor)
    secant = an.arc_length_predictor == 'secant'
    if gdc and secant:
        raise ValueError("arc_length_predictor='secant' is not valid with "
                         "NL_method='generalized_displacement_control'")
    if an.arc_length_sign_rule not in SIGN_RULES:
        raise ValueError('Invalid arc_length_sign_rule: %s'
                         % an.arc_length_sign_rule)
    det_rule = an.arc_length_sign_rule == 'determinant'
    if det_rule and not an.inertia:
        raise ValueError("Analysis.arc_length_sign_rule = 'determinant' "
                         "requires Analysis.inertia = True")
    max_jump = _check_max_inertia_jump(an)
    switches_left = int(an.branch_switching)
    if switches_left > 0 and not an.inertia:
        raise ValueError('Analysis.branch_switching requires '
                         'Analysis.inertia = True')
    modified_NR = an.modified_NR

    fext = an.calc_fext(inc=1., silent=True)
    kC0 = an.calc_kC(silent=silent)
    w = _weights(an, kC0)
    K0_abs = _abs_matrix(kC0)
    lbd_dependent = callbacks.is_load_factor_dependent(an)
    # the values of the constraints C^T c = lbd d, the border of the
    # solutions of the load vector
    d = _constraint_values(an)
    if lbd_dependent:
        # -dR/dlbd at c = 0 and lbd = 0
        q0 = callbacks.calc_load_vector(an, np.zeros_like(fext), fext, 0.)
        u_ref = _linear_solve(an, kC0, q0, d)
    else:
        u_ref = _linear_solve(an, kC0, fext, d)
    C = _constraint_matrix(an)
    proj = _residual_projector(an, w)

    def residual(fint, lbd_total):
        R = fint - lbd_total*fext
        return R if proj is None else proj(R)

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

    def wdot(a, b):
        return a.dot(wa*b)

    target = an.final_load_factor

    def target_reached(lbd0, dlbd0):
        # the step from lbd0 to lbd0 + dlbd0 exceeds the final load factor
        return (target is not None and lbd0 + dlbd0 > target
                and lbd0 < target)

    c = np.zeros_like(u_ref)
    lbd = 0.
    dc_last = None
    dlbd_last = None
    arc_length = None
    arc_length_last = None
    total_arc_length = 0.
    step_cut = False
    solve_kT_conv = None
    # factorization of the last converged state, with max_inertia_jump
    solve_last = None
    # signs of the determinant of the last converged state and of the origin
    det_conv = None
    det_origin = None
    # direction of the load along the tangent, with the inner-product rule,
    # at the last converged state and at the one before
    s_tan = 1.
    s_tan_prev = 1.
    # branch switching: the inertia changed in the last step, the pending
    # switch (eigenvector, eigenvalue, arc length before it, attempts)
    check_switch = 0
    switch = None
    after_switch = False
    # generalized displacement control
    dq_first = None
    dq_prev = None
    gdc_sign = 1.
    ctrl_sign = 1.

    step_num = 1
    # bifurcation pinning: consecutive failed attempts of the step, the
    # pinned eigenvector and the constraints and their values with it
    pinning = int(an.bifurcation_pinning or 0)
    num_fails = 0
    pin = None
    C_it = C
    d_it = d
    q_conv = None

    state = _restart_state(an, method)
    if state is not None:
        c = np.array(state['c'], dtype=float)
        lbd = state['lbd']
        dc_last = state['dc_last']
        dlbd_last = state['dlbd_last']
        arc_length = state['arc_length']
        arc_length_last = state['arc_length_last']
        max_arc_length_inc = state['max_arc_length_inc']
        total_arc_length = state['total_arc_length']
        det_origin = state['det_origin']
        s_tan_prev = state['s_tan_prev']
        dq_first = state['dq_first']
        dq_prev = state['dq_prev']
        gdc_sign = state['gdc_sign']
        ctrl_sign = state['ctrl_sign']
        switches_left = min(switches_left, state['switches_left'])
        after_switch = state.get('after_switch', False)
        step_num = state['step_num']
        msg('Restarting at load factor %1.5f' % lbd, level=1, silent=silent)

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
                solve_kT_conv = _factorize(kT, linear_solver=an.linear_solver,
                                           constraints=C, cache=_cache(an))
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
            if an.inertia:
                solve_last = solve_kT_conv
                det_conv = solve_kT_conv.det_sign
                if det_origin is None:
                    det_origin = det_conv
        if solve_kT_conv is not None:
            dc_q_conv = solve_kT_conv(q_conv, d_it)
        if solve_kT_conv is not None and dc_last is not None:
            s_tan = np.sign(inner(dc_q_conv, 1., dc_last, dlbd_last)) or 1.
        if check_switch:
            num_new = check_switch
            check_switch = 0
            # a bifurcation point when the load varies in the same direction
            # along the tangents before and after it, a limit point otherwise
            if s_tan == s_tan_prev:
                nneg = solve_kT_conv.num_negative_eigvals
                phi, mu = _critical_eigenvector(solve_kT_conv, c.shape[0],
                                                nneg or 0, num_new)
                if phi.dot(c) < 0:
                    phi = -phi
                msg('Bifurcation point crossed, critical eigenvalue %1.5e, '
                    'switching branch' % mu, level=1, silent=silent)
                if control or gdc:
                    # the step lengths are not arc lengths: the radius of
                    # the switching step is the length of the last step,
                    # with the growth of the step length since then, as the
                    # arc length of the Riks and Crisfield methods
                    radius = (np.sqrt(inner(dc_last, dlbd_last, dc_last,
                                            dlbd_last))
                              *max(arc_length/arc_length_last, 1.))
                else:
                    radius = None
                switch = [phi, mu, arc_length, 0, radius]
                if radius is None:
                    arc_length = max(arc_length, arc_length_last)
            else:
                msg('Limit point crossed', level=2, silent=silent)
        if arc_length is None:
            if gdc:
                norm_q = 1.
            elif control:
                # the first step is initialInc in units of the scale, the
                # controlled quantity increasing with the load factor along
                # the initial tangent
                norm_q = 1.
                ctrl_sign = np.sign(ctrl(dc_q_conv)) or 1.
            else:
                norm_q = np.sqrt(inner(dc_q_conv, 1., dc_q_conv, 1.))
            arc_length = an.initialInc*norm_q
            max_arc_length_inc = max(an.maxInc, arc_length)

        # radius of the spherical constraint of the corrections
        radius = arc_length
        if switch is not None:
            # branch switching: the critical eigenvector, at a constant load
            phi = switch[0]
            if switch[4] is not None:
                radius = switch[4]
            dc = (radius/np.sqrt(inner(phi, 0., phi, 0.)))*phi
            dlbd = 0.
        elif gdc:
            dq_step = dc_q_conv
            if dq_first is None:
                dq_first = dq_step
            if dq_prev is None:
                gsp = 1.
                dq_ref = dq_step
            elif after_switch:
                # first step on the bifurcated branch: the stiffness
                # parameter of the current tangent, the corrections
                # orthogonal to it, and the load in the direction of the
                # switching step
                gsp = wdot(dq_first, dq_first)/wdot(dq_step, dq_step)
                dq_ref = dq_step
                gdc_sign = np.sign(inner(dq_step, 1., dc_last,
                                         dlbd_last)) or 1.
            else:
                den = wdot(dq_prev, dq_step)
                if den == 0:
                    warn('Null generalized stiffness parameter! Analysis stopped',
                         level=1, silent=silent)
                    break
                gsp = wdot(dq_first, dq_first)/den
                dq_ref = dq_prev
            sign = gdc_sign*np.sign(gsp)
            dlbd = sign*arc_length*np.sqrt(abs(gsp))
            dc = dlbd*dq_step
            msg('Generalized stiffness parameter %1.5f' % gsp, level=2,
                silent=silent)
        elif control:
            if after_switch:
                # first step on the bifurcated branch: the controlled
                # quantity varies as along the predictor in the direction
                # of the switching step
                if use_secant:
                    s = np.sign(ctrl(dc_last))
                else:
                    s = np.sign(inner(dc_q_conv, 1., dc_last, dlbd_last)
                                *ctrl(dc_q_conv))
                ctrl_sign = s or ctrl_sign
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
                if det_rule and not after_switch and det_conv and det_origin:
                    # the load decreases when the determinant changed sign
                    sign = det_conv*det_origin
                else:
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
            R = residual(fint, lbd + dlbd)
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
                                          linear_solver=an.linear_solver,
                                          constraints=C_it, cache=_cache(an))
                except RuntimeError:
                    warn('Singular tangent stiffness matrix', level=2, silent=silent)
                    break
                if lbd_dependent:
                    # -dR/dlbd at the iterate of the tangent stiffness matrix
                    q_iter = callbacks.calc_load_vector(an, c + dc, fext,
                                                        lbd + dlbd)
                    dc_q = solve_kT(q_iter, d_it)
                else:
                    dc_q = solve_kT(callbacks.calc_load_vector(an, c + dc,
                                                               fext), d_it)
                iter_kT = 1
            dc_R = solve_kT(-R)

            varlbd = None
            if control and switch is None:
                # psi^T (dc + dc_R + varlbd dc_q) = ds
                den = ctrl(dc_q)
                if den != 0:
                    varlbd = (ds - ctrl(dc + dc_R))/den
            elif gdc and switch is None:
                # correction orthogonal to the tangent of the previous step
                den = wdot(dq_ref, dc_q)
                if den != 0:
                    varlbd = -wdot(dq_ref, dc_R)/den
            else:
                # linearized constraint
                den = inner(dc, dlbd, dc_q, 1.)
                g = 0.5*(inner(dc, dlbd, dc, dlbd) - radius**2)
                if den != 0:
                    varlbd = -(g + inner(dc, dlbd, dc_R, 0.))/den
            if method == 'crisfield' or switch is not None:
                u = dc + dc_R
                a = inner(dc_q, 1., dc_q, 1.)
                b = 2*inner(u, dlbd, dc_q, 1.)
                cc = inner(u, dlbd, u, dlbd) - radius**2
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
                    Rnorm_try = _norm(residual(fint_try, lbd_try), w)
                    if best is None or Rnorm_try < best[1]:
                        best = (eta_try, Rnorm_try, fint_try)
                    if Rnorm_try < Rnorm:
                        break
                eta, _, fint = best
                if eta != 1.:
                    msg('Line-search: eta = %1.5f' % eta, level=3, silent=silent)
            dc = dc + eta*dc_corr
            dlbd = dlbd + eta*varlbd

        # the iterations of the step failed, counted by the pinning
        iterations_failed = not converged
        if (converged and not control and not gdc
                and inner(dc, dlbd, dc_pred, dlbd_pred) <= 0):
            # the increment points backwards, e.g. to the previous converged
            # point, which is also at the distance arc_length
            warn('Diverged - the step returned along the path already traced',
                 level=2, silent=silent)
            converged = False

        if converged and switch is not None:
            cos = (inner(dc, dlbd, dc_pred, dlbd_pred)
                   /np.sqrt(inner(dc, dlbd, dc, dlbd)
                            *inner(dc_pred, dlbd_pred, dc_pred, dlbd_pred)))
            if cos < BRANCH_SWITCH_MIN_COS:
                warn('Branch switching: the step converged at an angle of '
                     '%1.1f degrees to the eigenvector'
                     % np.degrees(np.arccos(min(cos, 1.))), level=2,
                     silent=silent)
                converged = False

        solve_new = None
        if converged and an.inertia and not target_reached(lbd, dlbd):
            solve_new = _factorize_state(an, c + dc, lbd + dlbd,
                                         silent=silent)
            if (_inertia_jump(solve_last, solve_new, max_jump)
                    and arc_length*INC_CUT_FACTOR >= an.minInc):
                warn('The number of negative eigenvalues changed from %d to '
                     '%d, by more than max_inertia_jump'
                     % (solve_last.num_negative_eigvals,
                        solve_new.num_negative_eigvals), level=2,
                     silent=silent)
                converged = False

        finished = False
        if converged and target_reached(lbd, dlbd):
            msg('Load factor %1.5f exceeded, correcting to it...' % target,
                level=1, silent=silent)
            t = (target - lbd)/dlbd
            converged, dc1, _ = _NR_iterations(an, c, t*dc, target, fext,
                    kT0=None, silent=silent, lbd_dependent=lbd_dependent,
                    q=q_iter, w=w, K0_abs=K0_abs, proj=proj)
            if converged:
                msg('Converged at load factor %1.5f' % target, level=2,
                    silent=silent)
                an.cs.append((c + dc1).copy())
                an.increments.append(target)
                if an.inertia:
                    _converged_state(an, c + dc1, target, silent=silent)
                finished = True

        if finished:
            break

        if converged:
            c = c + dc
            lbd = lbd + dlbd
            if iteration > 1:
                # a step accepted without correction, at the residual of the
                # last converged state, is not counted as a success
                num_fails = 0
            if pin is not None:
                # released at the next step
                pin = None
                C_it = C
                d_it = d
            if switch is None or switch[4] is None:
                total_arc_length += arc_length
            msg('Converged at load factor %1.5f, total arc length %1.5f' % (lbd, total_arc_length),
                level=2, silent=silent)
            an.cs.append(c.copy()) #NOTE copy required
            an.increments.append(lbd)
            solve_kT_conv = None
            after_switch = switch is not None
            if after_switch:
                an.branch_points.append(len(an.increments) - 2)
                switches_left -= 1
                switch = None
            if an.inertia:
                # the factorization at the converged state, reused by the
                # next step
                _record_inertia(an, solve_new, silent=silent)
                solve_kT_conv = solve_new
                if solve_kT_conv is not None:
                    q_conv = callbacks.calc_load_vector(an, c, fext, lbd)
                    det_conv = solve_kT_conv.det_sign
                    if (switches_left > 0 and not after_switch
                            and solve_last is not None):
                        # number of eigenvalues that became negative, or
                        # positive when negative
                        n0 = solve_last.num_negative_eigvals
                        n1 = solve_kT_conv.num_negative_eigvals
                        if n0 is not None and n1 is not None:
                            check_switch = n1 - n0
                        else:
                            check_switch = int(det_conv
                                               != solve_last.det_sign)
                solve_last = solve_kT_conv
            if gdc and not after_switch:
                dq_prev = dq_step
                gdc_sign = sign
            s_tan_prev = s_tan
            dc_last = dc
            dlbd_last = dlbd
            arc_length_last = arc_length
            factor = np.sqrt(DESIRED_NUM_ITER/max(iteration - 1, 1))
            factor = min(max(factor, MIN_GROWTH_FACTOR), MAX_GROWTH_FACTOR)
            if step_cut:
                factor = min(factor, 1.)
            arc_length = min(factor*arc_length, max_arc_length_inc)
            if gdc:
                # the stiffness parameter scales the load increments
                arc_length = min(arc_length, an.initialInc)
            step_cut = False
            step_num += 1
            an.path_state = dict(method=method, c=c.copy(), lbd=lbd,
                    dc_last=dc_last.copy(), dlbd_last=dlbd_last,
                    arc_length=arc_length, arc_length_last=arc_length_last,
                    max_arc_length_inc=max_arc_length_inc,
                    total_arc_length=total_arc_length, det_origin=det_origin,
                    s_tan_prev=s_tan_prev,
                    dq_first=dq_first, dq_prev=dq_prev, gdc_sign=gdc_sign,
                    ctrl_sign=ctrl_sign, switches_left=switches_left,
                    after_switch=after_switch, step_num=step_num)
            if lbd == target:
                break
            if _stop(an, silent):
                break
            if total_arc_length >= an.maxArcLength:
                msg('Maximum specified arc-length of %1.5f achieved' % an.maxArcLength,
                    level=1, silent=silent)
                break

        elif switch is not None and switch[3] + 1 >= BRANCH_SWITCH_MAX_TRIES:
            warn('Branch switching abandoned, the path continues on the '
                 'branch already traced', level=1, silent=silent)
            arc_length = switch[2]
            switch = None
            switches_left -= 1

        elif switch is not None and switch[4] is not None:
            # the radius of the switching step is cut, not the step length
            switch[3] += 1
            switch[4] *= INC_CUT_FACTOR
            msg('Reseting the branch-switching step with the radius %1.5f'
                % switch[4], level=1, silent=silent)

        else:
            if switch is not None:
                switch[3] += 1
            arc_length *= INC_CUT_FACTOR
            step_cut = True
            msg('Reseting step with reduced arc-length increment %1.5f' % arc_length,
                level=1, silent=silent)
            if arc_length < an.minInc:
                warn('Minimum arc-length increment of %1.5f achieved! Analysis stopped at load factor %1.5f'
                     % (an.minInc, lbd), level=1, silent=silent)
                break
            num_fails += iterations_failed
            if (pinning and pin is None and switch is None
                    and num_fails >= pinning):
                pinned = _pin_bifurcation_mode(an, c, lbd, fext, C, d,
                                               solve_kT_conv, q_conv,
                                               silent=silent)
                if pinned is not None:
                    pin, C_it, d_it, solve_kT_conv, q_conv = pinned

    msg('Finished Non-Linear Static Analysis', silent=silent)
    msg('    load factor %1.5f, total arc length %1.5f' % (lbd if not an.increments
        else an.increments[-1], total_arc_length), level=1, silent=silent)
