r"""Calls to the callables of :class:`.Analysis` used by the non-linear solvers

The callables may depend on the load factor `\lambda` and on the
configuration through their optional keyword arguments, which are passed only
when the callable declares them explicitly in its signature (a ``**kwargs``
does not count), such that the callables written for the previous versions
are called exactly as before:

- ``inc``: the current load factor `\lambda`, passed to ``calc_fint``,
  ``calc_kC`` and ``calc_kG``. It allows configuration-dependent loads, e.g.
  follower pressures, whose configuration-dependent part enters
  ``calc_fint`` and whose load stiffness enters ``calc_kC`` or ``calc_kG``,
  both scaled by `\lambda`. The residual remains

  .. math::

      \{R(c, \lambda)\} = \lambda \{F_{ext}\} - \{F_{int}(c, \lambda)\}

  with ``calc_kC + calc_kG`` its exact Jacobian `-\partial R/\partial c`.

- ``c``: the current configuration, passed to ``calc_fext``, which must then
  return the load vector of that configuration for a unit load factor, i.e.
  `\partial R/\partial \lambda` at `c`. The arc-length solvers use it in the
  predictor and in the constraint instead of the constant reference vector
  ``calc_fext(inc=1.)``.

- ``lbd``: the current load factor `\lambda`, passed to ``calc_fext``
  together with ``c``, for a load vector `\partial R/\partial \lambda` that
  depends on the load factor as well, e.g. prescribed displacements imposed
  exactly as linear constraints `c = [T] c_r + \{c_{p0}\} + \lambda
  \{c_{p1}\}`, whose reduced residual

  .. math::

      \{R_r(c_r, \lambda)\} = [T]^T \left( \lambda \{F_{ext}\} -
      \{F_{int}([T] c_r + \{c_{p0}\} + \lambda \{c_{p1}\})\} \right)

  has `\partial R_r/\partial \lambda = [T]^T(\{F_{ext}\} - [K_T]
  \{c_{p1}\})`, with `[K_T]` evaluated at the current configuration and
  load factor. The keyword ``inc`` keeps its meaning, the factor that
  multiplies the returned vector, and the solvers keep calling
  ``calc_fext(inc=1., ...)``. When ``calc_fext`` declares ``lbd``:

  - the arc-length solvers pass ``lbd`` with ``c`` in the predictor and in
    the constraint;
  - the arc-length solvers scale the displacements with the linear solution
    of `\partial R/\partial \lambda` at `c = 0, \lambda = 0`, instead of
    the linear solution of the reference vector ``calc_fext(inc=1.)``, and
    the Newton-Raphson solver uses this linear solution as the predictor of
    its first step;
  - the relative convergence criterion also considers the norm of
    `\lambda \, \partial R/\partial \lambda`, see
    :func:`.newton_raphson._check_convergence`, since under a prescribed
    displacement alone the reference vector and the reduced internal force
    vector both vanish at equilibrium.

  ``lbd`` is never passed by the linear static analysis,
  ``Analysis.static(NLgeom=False)``.

"""
import inspect


def accepts_kwarg(func, name):
    """Tell whether ``func`` declares the keyword argument ``name``"""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    par = sig.parameters.get(name)
    return par is not None and par.kind in (par.POSITIONAL_OR_KEYWORD,
                                           par.KEYWORD_ONLY)


def calc_fint(an, c, lbd):
    r"""``an.calc_fint`` at ``c``, with ``inc=lbd`` when accepted"""
    kwargs = dict(inc=lbd) if accepts_kwarg(an.calc_fint, 'inc') else {}
    return an.calc_fint(c=c, silent=True, **kwargs)


def calc_kT(an, c, lbd):
    r"""Tangent stiffness matrix ``calc_kC + calc_kG`` at ``c``, with
    ``inc=lbd`` passed to the callables that accept it"""
    kwC = dict(inc=lbd) if accepts_kwarg(an.calc_kC, 'inc') else {}
    kwG = dict(inc=lbd) if accepts_kwarg(an.calc_kG, 'inc') else {}
    kC = an.calc_kC(c=c, NLgeom=True, silent=True, **kwC)
    kG = an.calc_kG(c=c, NLgeom=True, silent=True, **kwG)
    return kC + kG


def is_load_configuration_dependent(an):
    """Tell whether ``an.calc_fext`` accepts the configuration ``c``"""
    return accepts_kwarg(an.calc_fext, 'c')


def is_load_factor_dependent(an):
    """Tell whether ``an.calc_fext`` accepts the load factor ``lbd``"""
    return accepts_kwarg(an.calc_fext, 'lbd')


def calc_load_vector(an, c, fext, lbd=None):
    r"""`\partial R/\partial \lambda` at ``c`` and ``lbd``

    Returns ``calc_fext(inc=1., c=c)`` when ``calc_fext`` accepts ``c``,
    with ``lbd=lbd`` when it also accepts ``lbd`` and ``lbd`` is given, or
    ``calc_fext(inc=1., lbd=lbd)`` when it accepts only ``lbd``, otherwise
    the constant ``fext``.

    """
    kwargs = {}
    if is_load_configuration_dependent(an):
        kwargs['c'] = c
    if lbd is not None and is_load_factor_dependent(an):
        kwargs['lbd'] = lbd
    if not kwargs:
        return fext
    return an.calc_fext(inc=1., silent=True, **kwargs)
