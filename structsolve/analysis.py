from __future__ import absolute_import

import numpy as np
from numpy import dot

from .static import solve
from .logger import msg
from .newton_raphson import _solver_NR
from .arc_length_riks import _solver_arc_length_riks
from .arc_length_crisfield import _solver_arc_length_crisfield


class Analysis(object):
    r"""Class that embodies all data required for linear/non-linear analysis

    The parameters are described in the following tables:

    ========================  ==================================================
    Non-Linear Algorithm      Description
    ========================  ==================================================
    ``NL_method``             ``str``, ``'NR'`` for the Newton-Raphson,
                              ``'arc_length_riks'`` or
                              ``'arc_length_crisfield'`` for the Arc-Length
                              methods
    ``line_search``           ``bool``, activate a safeguarding line-search
                              (for Newton-Raphson methods only). The full
                              step is tried first and only reduced when it
                              fails a sufficient-decrease test on the
                              residual norm
    ``max_iter_line_search``  ``int``, maximum number of iteration attempts
                              for the line-search algorithm
    ``modified_NR``           ``bool``, activates the modified Newton-Raphson
    ``compute_every_n``       ``int``, if ``modified_NR=True``, the non-linear
                              matrices will be updated at every `n` iterations
    ``kT_initial_state``      ``bool``, if ``modified_NR=True``, tells if the
                              tangent stiffness matrix should be calculated
                              already at the first iteration of the analysis,
                              which is required for example when initial
                              imperfections take place
    ========================  ==================================================

    ==============     =================================================
    Incrementation     Description
    ==============     =================================================
    ``initialInc``     initial load increment size. In the arc-length
                       methods it defines the initial arc-length
                       increment, corresponding to a load factor
                       increment of ``initialInc`` along the initial
                       tangent
    ``minInc``         minimum increment size; if achieved the analysis
                       is terminated. The arc-length methods will use
                       this parameter to terminate when the
                       arc-length increment is smaller than ``minInc``
    ``maxInc``         maximum increment size, for the arc-length methods
                       the maximum arc-length increment
    ``maxArcLength``   maximum cumulative arc length covered by the
                       arc-length methods. The arc length is
                       dimensionless, with displacements scaled by the
                       linear solution for a load factor of 1, such
                       that in the linear regime an arc length of
                       about ``sqrt(2)`` corresponds to a load factor
                       increment of 1
    ==============     =================================================

    ====================    ============================================
    Convergence Criteria    Description
    ====================    ============================================
    ``relTOL``              the convergence is achieved when the norm of
                            the residual force vector is smaller than
                            ``relTOL`` times the largest norm between the
                            external and internal force vectors. Not used
                            if ``None``
    ``absTOL``              the convergence is also achieved when the
                            maximum residual force is smaller than this
                            value, which depends on the units of the
                            model. Not used if ``None``
    ``maxNumIter``          maximum number of iterations; if achieved the
                            load increment is reduced
    ``too_slow_TOL``        tolerance that tells if the convergence is too
                            slow
    ====================    ============================================

    Parameters
    ----------
    calc_fext : callable, optional
        Must return a 1-D array containing the external forces. Required for
        linear/non-linear static analysis.
    calc_fint : callable, optional
        Must return a 1-D array containing the internal forces. Required for
        non-linear analysis.
    calc_kC : callable, optional
        Must return a sparse matrix containing the constitutive stiffness matrix.
        Required for linear/non-linear static analysis.
    calc_kG : callable, optional
        Must return a sparse matrix containing the geometric stiffness matrix.
        Required for non-linear analysis.

    Returns
    -------
    increments : list
        Each time increment that achieved convergence.
    cs : list
        The solution for each increment.

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
        """General solver for static analyses

        Selects the specific solver based on the ``NL_method`` parameter.

        Parameters
        ----------

        NLgeom : bool
            Flag to indicate whether a linear or a non-linear analysis is to
            be performed.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.

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
                raise ValueError('{0} is an invalid NL_method')

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

