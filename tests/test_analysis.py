"""Tests for structsolve.analysis: Analysis class (linear and non-linear)"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix

from structsolve import Analysis


def _make_linear_system(n=10):
    """Create simple linear system callables"""
    K = csc_matrix(np.eye(n) * 3.0)
    f = np.ones(n) * 6.0

    def calc_fext(inc=1., silent=True):
        return f

    def calc_kC(c=None, NLgeom=False, silent=True):
        return K

    return calc_fext, calc_kC, K, f


def test_analysis_linear_static():
    """Test Analysis class for linear static analysis"""
    n = 10
    calc_fext, calc_kC, K, f = _make_linear_system(n)

    an = Analysis(calc_fext=calc_fext, calc_kC=calc_kC)
    increments, cs = an.static(NLgeom=False, silent=True)

    assert len(increments) == 1
    assert increments[0] == 1.0
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0], 2.0 * np.ones(n))
    assert an.last_analysis == 'static'


def test_analysis_newton_raphson():
    """Test Analysis class with Newton-Raphson non-linear solver

    Using a trivially linear system so NR converges in one iteration.
    """
    n = 5
    K = csc_matrix(np.eye(n) * 2.0)
    f = np.ones(n) * 4.0

    def calc_fext(inc=1., silent=True):
        return f

    def calc_kC(c=None, NLgeom=False, silent=True):
        return K

    def calc_kG(c=None, NLgeom=False, silent=True):
        return csc_matrix((n, n))

    def calc_fint(c=None, silent=True):
        return K.dot(c)

    an = Analysis(
        calc_fext=calc_fext,
        calc_kC=calc_kC,
        calc_kG=calc_kG,
        calc_fint=calc_fint,
    )
    an.initialInc = 1.0
    an.modified_NR = False
    an.kT_initial_state = False

    increments, cs = an.static(NLgeom=True, silent=True)

    assert len(increments) >= 1
    # Final solution should satisfy K*u = f
    np.testing.assert_allclose(cs[-1], 2.0 * np.ones(n), rtol=1e-3)
