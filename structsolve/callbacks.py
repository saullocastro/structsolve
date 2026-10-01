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


def calc_load_vector(an, c, fext):
    r"""`\partial R/\partial \lambda` at ``c``: ``calc_fext(inc=1., c=c)``
    when ``calc_fext`` accepts ``c``, otherwise the constant ``fext``"""
    if is_load_configuration_dependent(an):
        return an.calc_fext(inc=1., c=c, silent=True)
    return fext
