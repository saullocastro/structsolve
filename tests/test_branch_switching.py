"""Tests of the critical points along the path of the arc-length methods

``Analysis.branch_switching``, ``Analysis.max_inertia_jump`` and
``Analysis.arc_length_sign_rule``, see :func:`.arc_length._solver_arc_length`.
The tests of ``max_inertia_jump`` need oneMKL, see :mod:`structsolve.pardiso`.

"""
import numpy as np
import pytest

from analytic_problems import Problem, von_mises_truss
from test_arc_length import check_equilibrium
from test_inertia import needs_pardiso
from structsolve.arc_length import _critical_eigenvector
from structsolve.static import _factorize
from structsolve.pardiso import available as pardiso_available

METHODS = ['arc_length_riks', 'arc_length_crisfield']
CONTROLS = ['displacement_control', 'generalized_displacement_control']


def control_kwargs(method, sign=-1.):
    """The axial displacement controlled by ``displacement_control``,
    ``sign`` its direction under the load"""
    if method == 'displacement_control':
        return dict(control_vector=np.array([sign, 0.]))
    return {}


def pitchforks(betas=(0.5,), ks=(1.,), EA=10., P0=2.):
    """Columns with a symmetric bifurcation each, under a common axial load

    DOF 0 is the axial displacement `u`, DOFs ``1:`` the transverse
    displacements `v_i`, with the axial strain `\\epsilon = u + \\sum v_i^2/2`
    and the energy `EA \\epsilon^2/2 + \\sum (k_i v_i^2/2 + \\beta_i v_i^4/4)`
    under the compressive load `\\lambda P_0` on `u`. The fundamental path
    `v = 0` bifurcates at `\\lambda_i = k_i/P_0` into the branch `\\lambda P_0
    = k_i + \\beta_i v_i^2`, stable (supercritical) for `\\beta_i > 0`.

    """
    betas = np.asarray(betas, dtype=float)
    ks = np.asarray(ks, dtype=float)
    n = 1 + betas.size
    fext = np.zeros(n)
    fext[0] = -P0
    kC = np.diag(np.r_[EA, ks])

    def fint(c):
        u, v = c[0], c[1:]
        N = EA*(u + 0.5*(v**2).sum())
        return np.r_[N, N*v + ks*v + betas*v**3]

    def kG(c):
        u, v = c[0], c[1:]
        N = EA*(u + 0.5*(v**2).sum())
        K = np.zeros((n, n))
        K[0, 1:] = K[1:, 0] = EA*v
        K[1:, 1:] = EA*np.outer(v, v) + np.diag(N + 3*betas*v**2)
        return K
    p = Problem(fext, fint, kC, kG)
    p.lbd_cr = ks/P0
    p.P0 = P0
    p.ks = ks
    p.betas = betas
    return p


def bifurcated_residual(p, i):
    """Residual of the bifurcated branch of column ``i`` at every increment
    after the first branch point"""
    an = p.an
    j = an.branch_points[0] + 1
    lbds = np.asarray(an.increments[j:])
    v = np.array([c[1 + i] for c in an.cs[j:]])
    return lbds*p.P0 - p.ks[i] - p.betas[i]*v**2, v


@pytest.mark.parametrize('method', METHODS)
def test_without_switching_the_fundamental_path_is_traced(method):
    p = pitchforks()
    p.analysis(NL_method=method, initialInc=0.1, inertia=True)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    v = np.array([c[1] for c in p.an.cs])
    assert np.all(v == 0)
    lbds = np.asarray(p.an.increments)
    dets = np.asarray(p.an.det_signs)
    assert np.all(dets[lbds < 0.5] == 1)
    assert np.all(dets[lbds > 0.5] == -1)
    assert p.an.branch_points == []


@pytest.mark.parametrize('beta', [0.5, -0.5])
@pytest.mark.parametrize('method', METHODS)
def test_branch_switching_symmetric_bifurcation(method, beta):
    """The path switches to the bifurcated branch, rising (supercritical)
    or falling (subcritical), whose stability follows from the sign of
    the determinant"""
    p = pitchforks(betas=[beta])
    final = 1. if beta > 0 else None
    kwargs = {}
    if final is None:
        kwargs['stop_criterion'] = lambda an: len(an.increments) >= 30
    p.analysis(NL_method=method, initialInc=0.1, inertia=True,
               branch_switching=1, final_load_factor=final, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    an = p.an
    assert len(an.branch_points) == 1
    ib = an.branch_points[0]
    # switched from the first state beyond the critical point
    assert an.increments[ib] > 0.5 > an.increments[ib - 1]
    res, v = bifurcated_residual(p, 0)
    np.testing.assert_allclose(res, 0, atol=1e-6)
    assert np.all(v > 0)
    assert abs(v[-1]) > 0.5
    dets = np.asarray(an.det_signs[ib + 1:])
    if beta > 0:
        assert an.increments[-1] == 1.
        assert np.all(dets == 1)
    else:
        assert an.increments[-1] < 0.5
        assert np.all(dets == -1)


@pytest.mark.parametrize('method', METHODS)
def test_no_switching_at_a_limit_point(method):
    """A limit point is passed as without branch switching"""
    p = von_mises_truss(frac=1.5, ks=0.5)
    kwargs = dict(NL_method=method, initialInc=0.05, inertia=True)
    p.analysis(**kwargs)
    p.an.static(NLgeom=True, silent=True)
    ref = list(p.an.increments)
    p.analysis(branch_switching=1, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    assert p.an.branch_points == []
    assert p.an.increments == ref
    assert -1 in p.an.det_signs


@pytest.mark.parametrize('initialInc', [0.07, 0.13, 0.17])
@pytest.mark.parametrize('beta', [0.5, -0.5])
@pytest.mark.parametrize('method', CONTROLS)
def test_branch_switching_controls(method, beta, initialInc):
    """The displacement controls switch to the bifurcated branch and trace
    it away from the bifurcation point"""
    p = pitchforks(betas=[beta])
    p.analysis(NL_method=method, initialInc=initialInc, inertia=True,
               branch_switching=1, final_load_factor=None,
               stop_criterion=lambda an: len(an.increments) >= 25,
               **control_kwargs(method))
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    an = p.an
    assert len(an.branch_points) == 1
    ib = an.branch_points[0]
    assert an.increments[ib] > 0.5 > an.increments[ib - 1]
    res, v = bifurcated_residual(p, 0)
    np.testing.assert_allclose(res, 0, atol=1e-6)
    # on one side of the bifurcation, away from it
    v = np.abs(v)
    assert np.all(np.diff(v) > 0)
    assert v[-1] > (1. if method == 'displacement_control' else 0.15)
    lbds = np.asarray(an.increments[ib + 1:])
    dets = np.asarray(an.det_signs[ib + 1:])
    if beta > 0:
        assert np.all(lbds > 0.5) and np.all(dets == 1)
    else:
        assert np.all(lbds < 0.5) and np.all(dets == -1)


@pytest.mark.parametrize('linear_solver', ['splu', 'pardiso'])
@pytest.mark.parametrize('method', METHODS + CONTROLS)
def test_branch_switching_count_decrease(method, linear_solver):
    r"""A bifurcation point where an eigenvalue becomes positive: the column
    is unstable at the origin, `k < 0`, and the axial tension stabilizes it
    at `\lambda = 0.5`, where the branch of the buckled states, stable,
    falls back towards the origin"""
    if linear_solver == 'pardiso':
        if not pardiso_available():
            pytest.skip('oneMKL is not found')
    p = pitchforks(betas=[0.5], ks=[-1.], P0=-2.)
    p.analysis(NL_method=method, initialInc=0.13, inertia=True,
               branch_switching=1, final_load_factor=None,
               linear_solver=linear_solver,
               stop_criterion=lambda an: len(an.increments) >= 25,
               **control_kwargs(method, sign=1.))
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    an = p.an
    assert len(an.branch_points) == 1
    ib = an.branch_points[0]
    assert an.increments[ib] > 0.5 > an.increments[ib - 1]
    assert an.det_signs[ib - 1] == -1 and an.det_signs[ib] == 1
    res, v = bifurcated_residual(p, 0)
    np.testing.assert_allclose(res, 0, atol=1e-6)
    assert np.all(np.asarray(an.increments[ib + 1:]) < 0.5)
    assert np.all(np.asarray(an.det_signs[ib + 1:]) == 1)
    assert np.all(np.diff(np.abs(v)) > 0)


def test_branch_switching_requirements():
    p = pitchforks()
    p.analysis(NL_method='arc_length_riks', branch_switching=1)
    with pytest.raises(ValueError, match='inertia'):
        p.an.static(NLgeom=True, silent=True)


def test_critical_eigenvector():
    rng = np.random.default_rng(3)
    n = 12
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    mus = np.r_[-3., -0.2, -0.05, 0.1, np.linspace(1, 9, n - 4)]
    K = Q @ np.diag(mus) @ Q.T
    K = 0.5*(K + K.T)
    from scipy.sparse import csc_matrix
    solve = _factorize(csc_matrix(K))
    # two eigenvalues just became negative: the most negative of them
    phi, mu = _critical_eigenvector(solve, n, 3, 2)
    assert abs(mu - (-0.2)) < 1e-8
    assert abs(abs(phi @ Q[:, 1]) - 1) < 1e-8
    phi, mu = _critical_eigenvector(solve, n, 3, 1)
    assert abs(mu - (-0.05)) < 1e-8
    # one eigenvalue just became positive
    phi, mu = _critical_eigenvector(solve, n, 3, -1)
    assert abs(mu - 0.1) < 1e-8
    assert abs(abs(phi @ Q[:, 3]) - 1) < 1e-8
    # with a constraint orthogonal to the eigenvector of -0.05
    C = Q[:, 2:3]
    solve = _factorize(csc_matrix(K), constraints=C)
    phi, mu = _critical_eigenvector(solve, n, 2, 1)
    assert abs(mu - (-0.2)) < 1e-8
    assert abs(phi @ Q[:, 2]) < 1e-10


@pytest.mark.parametrize('num_weak', [3, 10])
def test_critical_eigenvector_weakly_stiff_modes(num_weak):
    """Positive eigenvalues closer to zero than the negative one, e.g. of
    weakly stiff terms of a Ritz model: the subspace grows until it holds
    the negative eigenvalue"""
    rng = np.random.default_rng(5)
    n = 40
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    mus = np.r_[-2e-3, np.full(num_weak, 1.5e-4)*(1 + 0.01*np.arange(num_weak)),
                np.linspace(1, 9, n - 1 - num_weak)]
    K = Q @ np.diag(mus) @ Q.T
    K = 0.5*(K + K.T)
    from scipy.sparse import csc_matrix
    solve = _factorize(csc_matrix(K))
    phi, mu = _critical_eigenvector(solve, n, 1, 1)
    assert abs(mu - (-2e-3)) < 1e-10
    assert abs(abs(phi @ Q[:, 0]) - 1) < 1e-8


@needs_pardiso
@pytest.mark.parametrize('method', METHODS + ['NR'])
def test_max_inertia_jump(method):
    """Two bifurcations close to each other are crossed one at a time"""
    p = pitchforks(betas=[0.5, 0.5], ks=[1., 1.02])
    kwargs = dict(NL_method=method, initialInc=0.2, inertia=True,
                  linear_solver='pardiso')
    p.analysis(**kwargs)
    p.an.static(NLgeom=True, silent=True)
    jumps = np.abs(np.diff(np.r_[0, p.an.num_negative_eigvals]))
    assert jumps.max() == 2
    p.analysis(max_inertia_jump=1, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    nneg = np.asarray(p.an.num_negative_eigvals)
    assert np.abs(np.diff(np.r_[0, nneg])).max() == 1
    assert nneg[-1] == 2
    assert p.an.increments[-1] == 1.


def test_max_inertia_jump_requirements():
    p = pitchforks()
    p.analysis(NL_method='arc_length_riks', max_inertia_jump=1, inertia=True)
    with pytest.raises(ValueError, match='pardiso'):
        p.an.static(NLgeom=True, silent=True)


@pytest.mark.parametrize('method', METHODS)
def test_determinant_sign_rule_limit_point(method):
    """The determinant rule reverses the load at the limit point as the
    inner-product rule does"""
    p = von_mises_truss(frac=1.5, ks=0.5)
    kwargs = dict(NL_method=method, initialInc=0.05, inertia=True)
    p.analysis(**kwargs)
    p.an.static(NLgeom=True, silent=True)
    ref = np.asarray(p.an.increments)
    p.analysis(arc_length_sign_rule='determinant', **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    np.testing.assert_allclose(p.an.increments, ref)


def test_determinant_sign_rule_bifurcation():
    """At a bifurcation crossed without switching, the determinant rule
    reverses the load, the inner-product rule does not"""
    p = pitchforks()
    kwargs = dict(NL_method='arc_length_riks', initialInc=0.1, inertia=True,
                  final_load_factor=None,
                  stop_criterion=lambda an: len(an.increments) >= 12)
    p.analysis(**kwargs)
    p.an.static(NLgeom=True, silent=True)
    assert np.all(np.diff(p.an.increments) > 0)
    p.analysis(arc_length_sign_rule='determinant', **kwargs)
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    lbds = np.asarray(p.an.increments)
    ilim = np.flatnonzero(np.diff(lbds) < 0)[0]
    assert lbds[ilim] > 0.5


def test_sign_rule_requirements():
    p = pitchforks()
    p.analysis(NL_method='arc_length_riks',
               arc_length_sign_rule='determinant')
    with pytest.raises(ValueError, match='inertia'):
        p.an.static(NLgeom=True, silent=True)
    p.analysis(NL_method='arc_length_riks', arc_length_sign_rule='abc')
    with pytest.raises(ValueError, match='arc_length_sign_rule'):
        p.an.static(NLgeom=True, silent=True)


def test_pin_bifurcation_mode():
    """The eigenvector of the eigenvalue closest to zero is pinned when it
    is orthogonal to the load: at the fundamental state of a column just
    below its bifurcation, the transverse displacement, whose eigenvalue is
    2e-4, not the axial one; the solutions of the pinned factorization keep
    it null"""
    from structsolve.arc_length import _pin_bifurcation_mode
    from structsolve.newton_raphson import _constraint_matrix
    p = pitchforks()
    an = p.analysis(NL_method='arc_length_riks')
    lbd = 0.4999
    c = np.array([-lbd*p.P0/10., 0.])
    fext = p.calc_fext()
    pinned = _pin_bifurcation_mode(an, c, lbd, fext, _constraint_matrix(an),
                                   None, silent=True)
    assert pinned is not None
    phi, C_pin, d_pin, solve_pin, q = pinned
    assert abs(abs(phi[1]) - 1) < 1e-12
    assert C_pin.shape == (2, 1) and d_pin is None
    x = solve_pin(np.array([1., 1.]))
    assert abs(x[1]) < 1e-14
    np.testing.assert_allclose(x[0], 1/10.)
    # with a transverse load the eigenvector is not a bifurcation mode
    p.fext[1] = 0.1
    assert _pin_bifurcation_mode(an, c, lbd, p.calc_fext(),
                                 _constraint_matrix(an), None,
                                 silent=True) is None


@pytest.mark.parametrize('method', METHODS + CONTROLS)
def test_bifurcation_pinning_default_path(method):
    """Without failed steps at a bifurcation the path is unchanged: the
    symmetric bifurcation crossed without switching, and the limit points
    of the von Mises truss, whose critical eigenvector is not orthogonal to
    the load"""
    for p, kwargs in ((pitchforks(), dict(final_load_factor=None,
                       stop_criterion=lambda an: len(an.increments) >= 12)),
                      (von_mises_truss(frac=1.5, ks=0.5), {})):
        kw = dict(NL_method=method, initialInc=0.1, **kwargs)
        if method == 'displacement_control':
            kw['control_vector'] = np.array([-1., 0.]) if p.fext.size == 2 \
                and p.fext[0] != 0 else np.array([0., -1.])
        p.analysis(**kw)
        p.an.static(NLgeom=True, silent=True)
        ref = list(p.an.increments)
        p.analysis(bifurcation_pinning=1, **kw)
        p.an.static(NLgeom=True, silent=True)
        assert p.an.increments == ref
