from __future__ import absolute_import

from .static import solve
from .logger import msg
from .newton_raphson import _solver_NR
from .arc_length_riks import _solver_arc_length_riks
from .arc_length_crisfield import _solver_arc_length_crisfield


class Analysis(object):
    r"""Class that embodies all data required for linear/non-linear analysis

    The structural model is defined by the callables passed to the
    constructor, which return the force vectors and stiffness matrices. The
    static analysis, linear or non-linear, is run with :meth:`.static`, whose
    solution is stored in the attributes ``increments`` and ``cs``.

    For non-linear analyses, the internal force vector
    `\{F_{int}(c)\}` must be in equilibrium with the external force
    vector scaled by the load factor `\lambda`:

    .. math::

        \{R\} = \lambda \{F_{ext}\} - \{F_{int}(c)\} = \{0\}

    and the tangent stiffness matrix `[K_T] = [K_C] + [K_G]`, given by the
    callables ``calc_kC`` and ``calc_kG``, should be the exact derivative of
    `\{F_{int}\}` with respect to `\{c\}`, such that the Newton-Raphson
    iterations converge quadratically.

    The analysis parameters are attributes of this class, described in the
    following tables together with their default values:

    ========================  ==================================================
    Non-Linear Algorithm      Description
    ========================  ==================================================
    ``NL_method``             ``str``, ``'NR'`` (default) for the
                              Newton-Raphson method, see
                              :func:`.newton_raphson._solver_NR`, or
                              ``'arc_length_riks'`` and
                              ``'arc_length_crisfield'`` for the arc-length
                              methods, see
                              :func:`.arc_length._solver_arc_length`
    ``line_search``           ``bool``, activates a safeguarding line-search,
                              for the Newton-Raphson method only. The full
                              step is tried first and only reduced when it
                              fails a sufficient-decrease test on the
                              residual norm. Default is ``False``
    ``max_iter_line_search``  ``int``, maximum number of iterations of the
                              line-search. Default is ``20``
    ``modified_NR``           ``bool``, activates the modified Newton-Raphson
                              method, where the tangent stiffness matrix is
                              not updated at every iteration. Default is
                              ``False``, i.e. full Newton-Raphson
    ``compute_every_n``       ``int``, if ``modified_NR=True``, the tangent
                              stiffness matrix is updated at every `n`
                              iterations. Default is ``6``
    ``kT_initial_state``      ``bool``, if ``modified_NR=True``, tells if the
                              tangent stiffness matrix should be calculated
                              already at the first iteration of the analysis,
                              which is required for example when initial
                              imperfections take place. Otherwise the linear
                              constitutive stiffness matrix is used. Default
                              is ``True``
    ========================  ==================================================

    ================   =================================================
    Incrementation     Description
    ================   =================================================
    ``initialInc``     initial load increment. In the arc-length
                       methods it defines the initial arc-length
                       increment, corresponding to a load factor
                       increment of ``initialInc`` along the initial
                       tangent. Default is ``0.1``
    ``minInc``         minimum increment; the analysis stops when the
                       load increment (Newton-Raphson) or the arc-length
                       increment (arc-length methods) becomes smaller
                       than ``minInc``. Default is ``1.e-4``
    ``maxInc``         maximum load increment, or maximum arc-length
                       increment for the arc-length methods. Default is
                       ``1.``
    ``maxArcLength``   maximum cumulative arc length covered by the
                       arc-length methods. The arc length is
                       dimensionless, with displacements scaled by the
                       linear solution for a load factor of 1, such
                       that in the linear regime an arc length of
                       about ``sqrt(2)`` corresponds to a load factor
                       increment of 1. Default is ``18``
    ================   =================================================

    ====================    ============================================
    Convergence Criteria    Description
    ====================    ============================================
    ``relTOL``              the convergence is achieved when the norm of
                            the residual force vector is smaller than
                            ``relTOL`` times the largest norm between the
                            external and internal force vectors. Not used
                            if ``None``. Default is ``1.e-6``
    ``absTOL``              the convergence is also achieved when the
                            maximum absolute residual force is smaller
                            than this value, which depends on the units of
                            the model. Not used if ``None``. Default is
                            ``None``
    ``maxNumIter``          maximum number of iterations (corrections) of a
                            step; if achieved the increment is reduced.
                            Default is ``30``
    ``too_slow_TOL``        a step is considered too slow when the smallest
                            residual norm is not reduced by this fraction
                            over the last iterations; the increment is then
                            reduced. Default is ``0.005``
    ====================    ============================================

    Parameters
    ----------
    calc_fext : callable, optional
        ``calc_fext(inc=1., silent=False)``, must return a 1-D array with the
        external force vector. Required for linear and non-linear static
        analyses. The non-linear solvers call it with ``inc=1.`` and scale the
        returned vector by the load factor.
    calc_fint : callable, optional
        ``calc_fint(c, silent=False)``, must return a 1-D array with the
        internal force vector for the solution vector ``c``. Required for
        non-linear analyses.
    calc_kC : callable, optional
        ``calc_kC(c=None, NLgeom=False, silent=False)``, must return a sparse
        matrix with the constitutive stiffness matrix. With ``c=None`` and
        ``NLgeom=False`` it must return the linear stiffness matrix, and with
        ``NLgeom=True`` the constitutive part of the tangent stiffness matrix
        at ``c``. Required for linear and non-linear static analyses.
    calc_kG : callable, optional
        ``calc_kG(c=None, NLgeom=False, silent=False)``, must return a sparse
        matrix with the geometric stiffness matrix at ``c``. It is called
        with ``NLgeom=True``. Required for non-linear analyses.

    Attributes
    ----------
    increments : list
        Load factors of the converged increments, filled by :meth:`.static`.
    cs : list
        Solution vectors of the converged increments, filled by
        :meth:`.static`.
    last_analysis : str
        Type of the last analysis run, ``'static'`` after :meth:`.static`.

    """
    __slots__ = ['NL_method', 'line_search', 'max_iter_line_search',
            'modified_NR', 'compute_every_n',
            'kT_initial_state', 'initialInc', 'minInc', 'maxInc',
            'maxArcLength', 'absTOL', 'relTOL', 'maxNumIter', 'too_slow_TOL',
            'increments', 'cs', 'last_analysis', 'calc_fext', 'calc_kC',
            'calc_fint', 'calc_kG']


    def __init__(self, calc_fext=None, calc_fint=None, calc_kC=None,
            calc_kG=None):
        # non-linear algorithm
        self.NL_method = 'NR'
        self.line_search = False
        self.max_iter_line_search = 20
        self.modified_NR = False
        self.compute_every_n = 6
        self.kT_initial_state = True
        # incrementation
        self.initialInc = 0.1
        self.minInc = 1.e-4
        self.maxInc = 1.
        self.maxArcLength = 18
        # convergence criteria
        self.absTOL = None
        self.relTOL = 1.e-6
        self.maxNumIter = 30
        self.too_slow_TOL = 0.005

        # required methods
        self.calc_fext = calc_fext
        self.calc_fint = calc_fint
        self.calc_kC = calc_kC
        self.calc_kG = calc_kG

        # outputs to be filled
        self.increments = None
        self.cs = None

        # flag telling the last analysis
        self.last_analysis = ''


    def static(self, NLgeom=False, silent=False):
        r"""General solver for static analyses

        The linear analysis solves `[K_C]\{c\} = \{F_{ext}\}` using
        :func:`.solve`. The non-linear analysis uses the solver selected by
        the ``NL_method`` attribute.

        Parameters
        ----------
        NLgeom : bool, optional
            Flag to indicate whether a linear or a non-linear analysis is to
            be performed.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.

        Returns
        -------
        increments : list
            Load factors of the converged increments. A linear analysis
            returns ``[1.]``. Non-linear analyses finish at a load factor of
            exactly ``1.`` unless they are stopped earlier, see the
            documentation of each solver.
        cs : list
            Solution vectors of the converged increments.

        """
        self.increments = []
        self.cs = []

        if NLgeom:
            self.maxInc = max(self.initialInc, self.maxInc)
            msg('Started Non-Linear Static Analysis', silent=silent)
            if self.NL_method == 'NR':
                _solver_NR(self, silent=silent)
            elif self.NL_method == 'arc_length_riks':
                _solver_arc_length_riks(self, silent=silent)
            elif self.NL_method == 'arc_length_crisfield':
                _solver_arc_length_crisfield(self, silent=silent)
            else:
                raise ValueError('{0} is an invalid NL_method'.format(self.NL_method))

        else:
            msg('Started Linear Static Analysis', silent=silent)
            fext = self.calc_fext(silent=silent)
            k0 = self.calc_kC(silent=silent)

            c = solve(k0, fext, silent=silent)

            self.cs.append(c)
            self.increments.append(1.)
            msg('Finished Linear Static Analysis', silent=silent)

        self.last_analysis = 'static'

        return self.increments, self.cs

