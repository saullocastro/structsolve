r"""Criteria to stop a non-linear analysis (:mod:`structsolve.stopping`)

A criterion is a callable ``stop_criterion(an)``, assigned to
``Analysis.stop_criterion``, called by the Newton-Raphson, arc-length and
displacement control solvers after every converged increment, once the
load factor and the solution of the increment are appended to
``an.increments`` and ``an.cs``, and its inertia to
``an.num_negative_eigvals`` and ``an.det_signs`` with ``an.inertia =
True``. The analysis stops when it returns ``True``, keeping the converged
increments. Any callable with this signature is accepted; the factory
below gives a common one. Together with ``Analysis.final_load_factor =
None``, the arc-length methods then trace a post-buckling path until the
criterion is met, rather than until the load factor 1.

"""


def value_reached(func, value):
    r"""Stop when a quantity of the solution reaches a value

    Factory of a ``stop_criterion`` that evaluates ``func`` on the last
    converged solution ``an.cs[-1]`` and compares it with ``value``.

    Parameters
    ----------
    func : callable
        ``func(c)`` returns a scalar of the solution vector ``c``, e.g. a
        displacement measure such as the end shortening, ``psi @ c``.
    value : float
        The analysis stops at the first converged increment where ``func(c)
        >= value`` for a positive ``value``, ``func(c) <= value`` for a
        negative one.

    Returns
    -------
    stop_criterion : callable
        ``stop_criterion(an)``, see :mod:`.stopping`.

    """
    def stop_criterion(an):
        f = func(an.cs[-1])
        return f >= value if value >= 0 else f <= value

    return stop_criterion
