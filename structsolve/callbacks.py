r"""Calls to the callables of :class:`.Analysis` used by the non-linear solvers

The callables may depend on the load factor and on the configuration
through their optional keyword arguments, which are passed only when the
callable declares them explicitly in its signature (a ``**kwargs`` does not
count), such that the callables written for the previous versions are called
exactly as before:

- ``inc``: the current load factor, passed to ``calc_fint``, ``calc_kC`` and
  ``calc_kG``. It allows configuration-dependent loads, e.g. follower
  pressures, whose configuration-dependent part enters ``calc_fint`` and
  whose load stiffness enters ``calc_kC`` or ``calc_kG``, both scaled by the
  load factor.

- ``c``: the current configuration, passed to ``calc_fext``, which must then
  return the load vector of that configuration for a unit load factor. The
  arc-length solvers use it in the predictor and in the constraint instead
  of the constant reference vector ``calc_fext(inc=1.)``.

- ``lbd``: the current load factor, passed to ``calc_fext`` together with
  ``c``, for a load vector that depends on the load factor as well, e.g.
  prescribed displacements imposed exactly as linear constraints. The
  keyword ``inc`` keeps its meaning, the factor that multiplies the returned
  vector, and the solvers keep calling ``calc_fext(inc=1., ...)``. When
  ``calc_fext`` declares ``lbd``:

  - the arc-length solvers pass ``lbd`` with ``c`` in the predictor and in
    the constraint;
  - the arc-length solvers scale the displacements with the linear solution
    of the load vector at ``c = 0``, ``lbd = 0``, instead of the linear
    solution of the reference vector ``calc_fext(inc=1.)``, and the
    Newton-Raphson solver uses this linear solution as the predictor of its
    first step;
  - the convergence criterion also considers the norm of the load
    factor times the load vector, see
    :func:`.newton_raphson._check_convergence`;
  - ``calc_fext(c=c, lbd=lbd)`` is called at the configurations where the
    tangent stiffness matrix is updated, right after ``calc_kC`` and
    ``calc_kG`` at the same ``c`` and load factor, such that the callables
    may share their computations, and the last load vector evaluated is the
    reference of the convergence checks until the next update.

  ``lbd`` is never passed by the linear static analysis,
  ``Analysis.static(NLgeom=False)``.

Notes
-----
With ``inc``, the residual remains

.. math::

    \{R(c, \lambda)\} = \lambda \{F_{ext}\} - \{F_{int}(c, \lambda)\}

with ``calc_kC + calc_kG`` its exact Jacobian `-\partial R/\partial c`.

With ``c``, ``calc_fext(inc=1., c=c)`` is `\partial R/\partial \lambda` at
`c`.

With ``lbd``, prescribed displacements imposed exactly as linear
constraints `c = [T] c_r + \{c_{p0}\} + \lambda \{c_{p1}\}` give the
reduced residual

.. math::

    \{R_r(c_r, \lambda)\} = [T]^T \left( \lambda \{F_{ext}\} -
    \{F_{int}([T] c_r + \{c_{p0}\} + \lambda \{c_{p1}\})\} \right)

which has `\partial R_r/\partial \lambda = [T]^T(\{F_{ext}\} - [K_T]
\{c_{p1}\})`, with `[K_T]` evaluated at the current configuration and
load factor, the vector that ``calc_fext(inc=1., c=c, lbd=lbd)`` must
return. The convergence criterion also considers the norm of
`\lambda \, \partial R/\partial \lambda`, since under a prescribed
displacement alone the reference vector and the reduced internal force
vector both vanish at equilibrium, and the displacements of the arc-length
solvers are scaled with the linear solution of `\partial R/\partial
\lambda` at `c = 0, \lambda = 0`, since the reference vector
``calc_fext(inc=1.)`` may then be null.

"""
import inspect


def accepts_kwarg(func, name):
    """Tell whether ``func`` declares the keyword argument ``name``

    The argument is declared when it is a named parameter of the signature,
    positional-or-keyword or keyword-only; a ``**kwargs`` does not count. A
    callable whose signature cannot be read by :func:`inspect.signature` does
    not declare it.

    Parameters
    ----------
    func : callable
        Function or method to inspect.
    name : str
        Name of the keyword argument.

    Returns
    -------
    accepts : bool
        ``True`` when ``func`` has a parameter ``name`` that can be passed by
        keyword.

    See Also
    --------
    calc_fint, calc_kT, calc_load_vector : calls that depend on it.

    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    par = sig.parameters.get(name)
    return par is not None and par.kind in (par.POSITIONAL_OR_KEYWORD,
                                           par.KEYWORD_ONLY)


def calc_fint(an, c, lbd):
    r"""``an.calc_fint`` at ``c``, with ``inc=lbd`` when accepted

    Calls ``an.calc_fint(c=c, silent=True)``, adding ``inc=lbd`` only when
    ``an.calc_fint`` declares ``inc``, see :func:`accepts_kwarg`.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object providing ``calc_fint``.
    c : array-like
        Configuration, the vector of the degrees of freedom.
    lbd : float
        Current load factor, dimensionless, passed as ``inc``.

    Returns
    -------
    fint : array-like
        The internal force vector returned by ``an.calc_fint``.

    """
    kwargs = dict(inc=lbd) if accepts_kwarg(an.calc_fint, 'inc') else {}
    return an.calc_fint(c=c, silent=True, **kwargs)


def calc_kT(an, c, lbd):
    r"""Tangent stiffness matrix ``calc_kC + calc_kG`` at ``c``, with
    ``inc=lbd`` passed to the callables that accept it

    Calls ``an.calc_kC(c=c, NLgeom=True, silent=True)`` and
    ``an.calc_kG(c=c, NLgeom=True, silent=True)``, each with ``inc=lbd`` only
    when it declares ``inc``, see :func:`accepts_kwarg`, and returns their
    sum.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object providing ``calc_kC`` and ``calc_kG``.
    c : array-like
        Configuration, the vector of the degrees of freedom.
    lbd : float
        Current load factor, dimensionless, passed as ``inc``.

    Returns
    -------
    kT : sparse matrix or ndarray
        The sum of the matrices returned by ``an.calc_kC`` and ``an.calc_kG``,
        of the type given by them.

    """
    kwC = dict(inc=lbd) if accepts_kwarg(an.calc_kC, 'inc') else {}
    kwG = dict(inc=lbd) if accepts_kwarg(an.calc_kG, 'inc') else {}
    kC = an.calc_kC(c=c, NLgeom=True, silent=True, **kwC)
    kG = an.calc_kG(c=c, NLgeom=True, silent=True, **kwG)
    return kC + kG


def is_load_configuration_dependent(an):
    """Tell whether ``an.calc_fext`` accepts the configuration ``c``

    When it does, the solvers evaluate the load vector at the current
    configuration, see :func:`calc_load_vector`, and the linear static
    analysis iterates on the load, see :meth:`.Analysis.static`.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object providing ``calc_fext``.

    Returns
    -------
    dependent : bool
        ``True`` when ``an.calc_fext`` declares the keyword argument ``c``,
        see :func:`accepts_kwarg`.

    """
    return accepts_kwarg(an.calc_fext, 'c')


def is_load_factor_dependent(an):
    """Tell whether ``an.calc_fext`` accepts the load factor ``lbd``

    When it does, the solvers pass the current load factor to ``calc_fext``
    together with ``c``, see :func:`calc_load_vector` and the module
    docstring.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object providing ``calc_fext``.

    Returns
    -------
    dependent : bool
        ``True`` when ``an.calc_fext`` declares the keyword argument ``lbd``,
        see :func:`accepts_kwarg`.

    """
    return accepts_kwarg(an.calc_fext, 'lbd')


def calc_load_vector(an, c, fext, lbd=None):
    r"""Derivative of the residual with respect to the load factor

    Returns the load vector ``q`` given by ``an.calc_fext`` for a unit load
    factor at the configuration ``c`` and the load factor ``lbd``, used by
    the solvers in the predictors, in the arc-length constraint and in the
    convergence criterion. With prescribed displacements, ``q`` is not their
    reaction force: it has the size of the reduced problem and acts on the
    free degrees of freedom. ``structsolve`` does not compute the reaction;
    see for instance ``MultiDomain.reaction_pd`` of `panels`.

    ``an.calc_fext`` is called with ``inc=1.`` and only with the keyword
    arguments that it declares:

    - ``c=c``, when it declares ``c``;
    - ``lbd=lbd``, when it declares ``lbd`` and ``lbd`` is not ``None``.

    When none of them is passed, `\{q\}` is the constant reference vector,
    and ``fext`` is returned without calling ``an.calc_fext``.

    Parameters
    ----------
    an : :class:`.Analysis`
        Analysis object.
    c : array-like
        Configuration.
    fext : array-like
        Reference vector ``an.calc_fext(inc=1.)``.
    lbd : float or None, optional
        Load factor.

    Returns
    -------
    q : array-like
        `\partial R/\partial \lambda`.

    Notes
    -----
    `\{q\} = \partial R/\partial \lambda` is the external load vector only
    for loads that do not depend on the configuration; with prescribed
    displacements imposed exactly it also contains `-[T]^T [K_T]
    \{c_{p1}\}`, see the module docstring.

    With prescribed displacements, `\{q\}` is the out-of-balance force
    produced by a unit increment of the load factor at the current
    configuration, with the free degrees of freedom held fixed. The reaction
    is a quantity of the full problem, e.g. the generalized reaction
    work-conjugate to `\lambda`, `P = \{c_{p1}\}^T (\{F_{int}(c)\} - \lambda
    \{F_{ext}\})`.

    """

    kwargs = {}
    if is_load_configuration_dependent(an):
        kwargs['c'] = c
    if lbd is not None and is_load_factor_dependent(an):
        kwargs['lbd'] = lbd
    if not kwargs:
        return fext
    return an.calc_fext(inc=1., silent=True, **kwargs)
