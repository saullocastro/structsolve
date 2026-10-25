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


def test_invalid_convergence_parameters():
    """An invalid convergence criterion, or no tolerance at all"""
    from structsolve.convergence import CRITERIA

    assert CRITERIA == ('crisfield', 'relative')
    n = 3
    K = csc_matrix(np.eye(n))
    an = Analysis(calc_fext=lambda inc=1., silent=True: np.ones(n),
                  calc_kC=lambda c=None, NLgeom=False, silent=True: K,
                  calc_kG=lambda c=None, NLgeom=False, silent=True:
                  csc_matrix((n, n)),
                  calc_fint=lambda c=None, silent=True: K @ c)
    assert an.convergence == 'crisfield'
    assert an.relTOL == 1.e-9
    an.convergence = 'euclidean'
    with pytest.raises(ValueError, match='invalid convergence'):
        an.static(NLgeom=True, silent=True)
    an.convergence = 'crisfield'
    an.relTOL = None
    with pytest.raises(ValueError, match='relTOL or absTOL'):
        an.static(NLgeom=True, silent=True)


def test_crisfield_weights_and_zeta():
    r"""`w_j = 1/|K_{jj}|` at the free degrees of freedom, zero at the null
    rows and columns; `\zeta = \|\boldsymbol{R}\|_{\boldsymbol{d}}/
    \max(\|\boldsymbol{F}_{ext}\|_{\boldsymbol{d}}, \|\boldsymbol{F}_{int}\|_{\boldsymbol{d}})`"""
    from scipy.sparse import csr_matrix
    from structsolve.convergence import crisfield_weights, crisfield_zeta

    K = csr_matrix(np.array([[4., -1., 0.], [-1., -2., 0.], [0., 0., 0.]]))
    w = crisfield_weights(K)
    np.testing.assert_allclose(w, [0.25, 0.5, 0.])
    fext = np.array([2., 0., 7.])
    fint = np.array([1., 1., 0.])
    R = fext - fint
    zeta = crisfield_zeta(R, fext, fint, w)
    ref = max(np.sqrt(4*0.25), np.sqrt(0.25 + 0.5))
    np.testing.assert_allclose(zeta, np.sqrt(0.25 + 0.5)/ref)
    # an additional reference, e.g. lbd*dR/dlbd
    assert crisfield_zeta(R, fext, fint, w, q=np.array([10., 0., 0.])) < zeta
    assert crisfield_zeta(np.zeros(3), np.zeros(3), np.zeros(3), w) == 0.
    assert crisfield_zeta(R, np.zeros(3), np.zeros(3), w) == np.inf
    with pytest.raises(ValueError, match='null diagonal'):
        crisfield_weights(csr_matrix(np.array([[0., 1.], [1., 2.]])))


def test_check_convergence_criteria():
    r"""Crisfield's `\zeta \le` ``relTOL`` and the relative criterion of the
    Euclidean norm, with ``absTOL`` an alternative of both"""
    from structsolve.newton_raphson import _check_convergence

    class An(object):
        convergence = 'crisfield'
        relTOL = 1.e-3
        absTOL = None
    an = An()
    w = np.array([1., 1.e-6])
    fext = np.array([1., 0.])
    fint = np.array([1., 0.5])
    R = fext - fint
    # the second component is a moment of a stiff rotation
    conv, Rnorm, zeta = _check_convergence(an, R, fint, fext, w=w)
    assert conv and zeta < 1.e-3
    an.convergence = 'relative'
    conv, Rnorm, zeta = _check_convergence(an, R, fint, fext)
    assert not conv and np.isclose(zeta, 0.5/np.linalg.norm(fint))
    an.absTOL = 0.6
    assert _check_convergence(an, R, fint, fext)[0]
    an.convergence = 'crisfield'
    with pytest.raises(ValueError, match='weights'):
        _check_convergence(an, R, fint, fext)


@pytest.mark.parametrize('convergence', ['crisfield', 'relative'])
def test_check_convergence_roundoff_floor(convergence, monkeypatch):
    r"""An iteration has also converged when the norm of the residual of
    the criterion is not larger than ``ROUNDOFF_FACTOR`` `\varepsilon \|
    |K_{C0}| |c| \|`, whatever ``relTOL``, and only when ``Kc_abs`` is
    given; `\zeta` is unchanged"""
    from structsolve import convergence as cv

    class An(object):
        relTOL = 1.e-20
        absTOL = None
    an = An()
    an.convergence = convergence
    eps = np.finfo(float).eps
    w = np.array([1., 1.e-6]) if convergence == 'crisfield' else None
    # |K_C0| |c| of a translation and of a stiff rotation
    Kc_abs = np.array([1.e3, 1.e9])
    fext = np.array([1., 0.])
    fint = np.array([1., 0.])
    if convergence == 'crisfield':
        norm_Kc = np.sqrt(1.e6 + 1.e-6*1.e18)
    else:
        norm_Kc = np.sqrt(1.e6 + 1.e18)
    floor = cv.ROUNDOFF_FACTOR*eps*norm_Kc
    assert cv.ROUNDOFF_FACTOR == 3.
    assert np.isclose(cv.roundoff_floor(Kc_abs, w=w), floor, rtol=1e-14)
    # residual on the first component only: its norm is |R_0| with both
    # criteria
    for factor, expected in ((0.99, True), (1.01, False)):
        R = np.array([factor*floor, 0.])
        conv, Rnorm, zeta, roundoff = cv.check_convergence(an, R, fint,
                fext, w=w, Kc_abs=Kc_abs, full_output=True)
        assert conv is expected and roundoff is expected
        assert np.isclose(Rnorm, factor*floor, rtol=1e-14)
        assert zeta > an.relTOL
        # without Kc_abs, the criterion of relTOL only
        assert cv.check_convergence(an, R, fint, fext, w=w) == (False,
                Rnorm, zeta)
    R = np.array([0.5*floor, 0.])
    # an iteration accepted by relTOL is not marked
    an.relTOL = 1.e-3
    assert cv.check_convergence(an, R, fint, fext, w=w, Kc_abs=Kc_abs,
                                full_output=True)[::3] == (True, False)
    # ROUNDOFF_FACTOR = 0 disables the floor
    an.relTOL = 1.e-20
    monkeypatch.setattr(cv, 'ROUNDOFF_FACTOR', 0.)
    assert not cv.check_convergence(an, R, fint, fext, w=w,
                                    Kc_abs=Kc_abs)[0]
