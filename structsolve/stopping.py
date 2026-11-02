r"""Criteria to stop a non-linear analysis (:mod:`structsolve.stopping`)

A criterion is a callable ``stop_criterion(an)``, assigned to
``Analysis.stop_criterion``, called by the Newton-Raphson, arc-length and
displacement control solvers after every converged increment, once the
load factor and the solution of the increment are appended to
``an.increments`` and ``an.cs``, and its inertia to
``an.num_negative_eigvals`` and ``an.det_signs`` with ``an.inertia =
True``. The analysis stops when it returns ``True``, keeping the converged
increments. Any callable with this signature is accepted; the factories
below give two common ones. Together with ``Analysis.final_load_factor =
None``, the arc-length methods then trace a post-buckling path until the
criterion is met, rather than until the load factor 1.

"""
import numpy as np


def beyond_limit_point(num_increments=0):
    r"""Stop a given number of increments beyond the first limit point of
    the load

    The first limit point is the last converged increment before the first
    decrease of the load factor along the path. The analysis stops after
    ``num_increments`` converged increments beyond it, i.e. when the
    increment of index `i_{lim}` + ``num_increments`` has converged, `i_{lim}`
    being the index of the limit point in ``an.increments``.

    Parameters
    ----------
    num_increments : int, optional
        Number of increments beyond the limit point. With ``0`` the analysis
        stops at the first increment whose load factor is smaller than the
        previous one, the limit point being the previous increment.

    Returns
    -------
    stop_criterion : callable
        ``stop_criterion(an)``, see :mod:`.stopping`.

    """
    def stop_criterion(an):
        lbds = np.asarray(an.increments)
        drops = np.flatnonzero(np.diff(lbds) < 0)
        if drops.size == 0:
            return False
        ilim = drops[0]
        return len(lbds) - 1 >= ilim + max(num_increments, 1)

    return stop_criterion


def value_reached(func, value):
    r"""Stop when a quantity of the solution reaches a value

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
