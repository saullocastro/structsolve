"""Tests of the stabilized Newton-Raphson solver,
``Analysis.NL_method = 'stabilized_NR'``

Against the plain Newton-Raphson solver on a problem without instability,
and on the von Mises truss of ``analytic_problems`` loaded beyond its limit
load, where the plain solver stops at the limit point: the accepted states,
their status, the undamped equilibrium of the accepted states, the residual
of the damped-only states, the linear solvers, the constraint values, the
restart, ``on_converged``, the iteration control of the stages, the
growth of the increments and the validation of the ``stabilization_*``
options.

"""
import numpy as np
import pytest

from structsolve.convergence import crisfield_weights, crisfield_zeta
from structsolve.pardiso import available as pardiso_available
from analytic_problems import hardening_springs, von_mises_truss, Problem


def zeta(p, lbd, c):
    """Crisfield's relative residual of the undamped equilibrium of the
    problem ``p`` at the state ``c`` and the load factor ``lbd``"""
    fint = p.fint(c)
    return crisfield_zeta(fint - lbd*p.fext, lbd*p.fext, fint,
                          crisfield_weights(p.kC))


def truss_beyond_limit(**kwargs):
    """The von Mises truss loaded at 1.3 times its limit load, with load
    increments starting at 0.1: the plain Newton-Raphson solver stops at
    the limit point, at a load factor of about 1/1.3"""
    p = von_mises_truss(frac=1.3)
    kwargs.setdefault('initialInc', 0.1)
    p.analysis(**kwargs)
    return p


def test_equals_newton_raphson_without_instability():
    """On the hardening springs every increment converges at the first
    stage within a few corrections: with the increment growth of
    ``NL_method='NR'``, ``stabilization_inc_growth=INC_GROWTH_FACTOR``, the
    increments, the solutions and the evaluations of the internal force
    vector are those of ``NL_method='NR'``, every state is
    ``'undamped'``"""
    from structsolve.newton_raphson import INC_GROWTH_FACTOR
    out = {}
    for method in ('NR', 'stabilized_NR'):
        p = hardening_springs()
        p.analysis(initialInc=0.1, NL_method=method,
                   stabilization_inc_growth=INC_GROWTH_FACTOR)
        p.an.static(NLgeom=True, silent=True)
        out[method] = p
    p0, p1 = out['NR'], out['stabilized_NR']
    assert p1.an.increments == p0.an.increments
    for c0, c1 in zip(p0.an.cs, p1.an.cs):
        np.testing.assert_array_equal(c1, c0)
    assert len(p1.fint_calls) == len(p0.fint_calls)
    assert p1.an.stabilization_status == ['undamped']*len(p1.an.increments)
    assert p1.an.damped_residuals == [None]*len(p1.an.increments)
    assert p0.an.stabilization_status == []
    assert p0.an.damped_residuals == []


def test_passes_limit_point_where_newton_raphson_stops():
    """Beyond the limit load the plain solver stops at the limit point;
    the stabilized solver reaches the load factor 1.0 through a corrected
    state, every accepted state is an undamped equilibrium to the
    tolerance, and the final state is the equilibrium on the inverted
    configuration given by the arc-length method"""
    p0 = truss_beyond_limit()
    p0.an.static(NLgeom=True, silent=True)
    assert p0.an.increments[-1] < 1/1.3

    p = truss_beyond_limit(NL_method='stabilized_NR')
    p.an.static(NLgeom=True, silent=True)
    an = p.an
    assert an.increments[-1] == 1.
    assert len(an.stabilization_status) == len(an.increments)
    assert 'corrected' in an.stabilization_status
    assert 'damped' not in an.stabilization_status
    for lbd, c in zip(an.increments, an.cs):
        assert zeta(p, lbd, c) <= 1e-8
    # the jump: the apex passes from before the limit point to the
    # inverted configuration
    w = np.array([c[0] for c in an.cs])
    assert np.diff(w).max() > 1.
    q = von_mises_truss(frac=1.3)
    q.analysis(NL_method='arc_length_riks', initialInc=0.05)
    q.an.static(NLgeom=True, silent=True)
    np.testing.assert_allclose(an.cs[-1], q.an.cs[-1], rtol=1e-6)


def test_damped_only_states_flagged(capsys):
    r"""On the truss at 1.05 times its limit load, with load increments
    starting at 0.2, the correction to the undamped equilibrium fails at
    some increments, whose
    damped states are accepted, flagged ``'damped'`` with a warning: their
    recorded residual is the relative residual of their undamped
    equilibrium, larger than the tolerance, and their undamped residual
    is minus the viscous force of the stabilization, Crisfield's
    criterion being met by the damped residual"""
    p = von_mises_truss(frac=1.05)
    p.analysis(NL_method='stabilized_NR', initialInc=0.2)
    mu = p.an.stabilization_factor
    p.an.static(NLgeom=True, silent=False)
    assert 'damped state accepted' in capsys.readouterr().out
    an = p.an
    assert 'damped' in an.stabilization_status
    D = np.abs(np.diag(p.kC))
    lbd_prev, c_prev = 0., np.zeros(2)
    for lbd, c, status, res in zip(an.increments, an.cs,
                                   an.stabilization_status,
                                   an.damped_residuals):
        if status == 'damped':
            assert res == pytest.approx(zeta(p, lbd, c), rel=1e-10)
            assert res > 1e3*an.relTOL
            viscous = mu/(lbd - lbd_prev)*D*(c - c_prev)
            R = p.fint(c) - lbd*p.fext
            np.testing.assert_allclose(R, -viscous, rtol=1e-5,
                                       atol=1e-5*np.abs(viscous).max())
        else:
            assert res is None
            assert zeta(p, lbd, c) <= 1e-8
        lbd_prev, c_prev = lbd, c


def test_damped_only_flag_when_correction_prevented(monkeypatch):
    """The correction to the undamped equilibrium, the iterations without
    viscous forces that start from the damped state, is made to fail: the
    state that the plain solver could not reach is accepted as
    ``'damped'``, with its residual, instead of ``'corrected'``"""
    from structsolve import newton_raphson
    NR_iterations = newton_raphson._NR_iterations

    def no_correction(an, c, dc, total, fext, **kwargs):
        if kwargs.get('viscous') is None and np.any(dc != 0) and \
                getattr(no_correction, 'damped', False):
            no_correction.damped = False
            return False, dc, kwargs.get('q')
        out = NR_iterations(an, c, dc, total, fext, **kwargs)
        no_correction.damped = kwargs.get('viscous') is not None and out[0]
        return out

    monkeypatch.setattr(newton_raphson, '_NR_iterations', no_correction)
    p = truss_beyond_limit(NL_method='stabilized_NR')
    p.an.static(NLgeom=True, silent=True)
    status = p.an.stabilization_status
    assert 'corrected' not in status
    i = status.index('damped')
    assert p.an.damped_residuals[i] > 1e3*p.an.relTOL
    assert p.an.damped_residuals[i] == pytest.approx(
        zeta(p, p.an.increments[i], p.an.cs[i]), rel=1e-10)


def linear_solvers():
    return ['splu', 'dense_lu'] + (['pardiso', 'pardiso_lu']
                                   if pardiso_available() else [])


@pytest.mark.parametrize('linear_solver', linear_solvers())
def test_linear_solvers(linear_solver):
    """Every linear solver gives the increments, the status and, to
    round-off, the solutions of ``'splu'`` on the truss beyond its limit
    load, with the inertia recorded for ``'splu'``, ``'dense_lu'`` and
    ``'pardiso'``"""
    out = {}
    for ls in ('splu', linear_solver):
        inertia = ls != 'pardiso_lu'
        p = truss_beyond_limit(NL_method='stabilized_NR', linear_solver=ls,
                               inertia=inertia)
        p.an.static(NLgeom=True, silent=True)
        out[ls] = p.an
        if inertia:
            assert len(p.an.det_signs) == len(p.an.increments)
    a0, a1 = out['splu'], out[linear_solver]
    assert a1.stabilization_status == a0.stabilization_status
    np.testing.assert_allclose(a1.increments, a0.increments, rtol=1e-12)
    for c0, c1 in zip(a0.cs, a1.cs):
        np.testing.assert_allclose(c1, c0, rtol=1e-8, atol=1e-12)
    if linear_solver != 'pardiso_lu':
        assert a1.det_signs == a0.det_signs


def test_constraint_values_snap_under_displacement_control():
    """The loaded point of the truss with a soft spring, ``ks=0.2``, is
    displaced by a constraint with a value and no external load: the apex
    snaps, the plain solver stops with load increments starting at 0.1,
    and the stabilized solver reaches the final displacement with every
    accepted state satisfying the constraint and, except the damped ones,
    the equilibrium"""
    u = 3.
    out = {}
    for method in ('NR', 'stabilized_NR'):
        p = von_mises_truss(frac=0., ks=0.2)
        p.analysis(NL_method=method, initialInc=0.1,
                   constraints=np.array([0., 1.]), constraint_values=[u])
        p.an.static(NLgeom=True, silent=True)
        out[method] = p
    assert out['NR'].an.increments[-1] < 1.
    p = out['stabilized_NR']
    an = p.an
    assert an.increments[-1] == 1.
    assert set(an.stabilization_status) - {'undamped'}
    for lbd, c, status in zip(an.increments, an.cs, an.stabilization_status):
        assert c[1] == pytest.approx(lbd*u, rel=1e-12)
        if status != 'damped':
            # the residual at the apex, the residual at the loaded point
            # being the reaction of the constraint, compared with the force
            # of the spring at the final displacement
            assert abs(p.fint(c)[0]) <= 1e-8*0.2*u


def test_restart_and_on_converged():
    """``on_converged`` is called once per accepted state, damped or
    not; a run stopped after the jump and restarted from its
    ``path_state`` finishes with the states of the uninterrupted run; a
    ``path_state`` of ``'NR'`` is rejected"""
    p = truss_beyond_limit(NL_method='stabilized_NR')
    seen = []
    p.an.on_converged = lambda an: seen.append(an.stabilization_status[-1])
    p.an.static(NLgeom=True, silent=True)
    full = p.an
    assert seen == full.stabilization_status
    n = full.stabilization_status.index('corrected') + 1

    q = truss_beyond_limit(NL_method='stabilized_NR')
    q.an.stop_criterion = lambda an: len(an.increments) == n
    q.an.static(NLgeom=True, silent=True)
    assert len(q.an.increments) == n
    r = truss_beyond_limit(NL_method='stabilized_NR',
                           restart_state=q.an.path_state)
    r.an.static(NLgeom=True, silent=True)
    np.testing.assert_allclose(r.an.increments, full.increments[n:],
                               rtol=1e-14)
    assert r.an.stabilization_status == full.stabilization_status[n:]
    np.testing.assert_allclose(r.an.cs[-1], full.cs[-1], rtol=1e-12)

    s = truss_beyond_limit()
    s.an.stop_criterion = lambda an: True
    s.an.static(NLgeom=True, silent=True)
    t = truss_beyond_limit(NL_method='stabilized_NR',
                           restart_state=s.an.path_state)
    with pytest.raises(ValueError, match='restart_state'):
        t.an.static(NLgeom=True, silent=True)


@pytest.mark.parametrize('mu', [0., -1e-5, np.nan, np.inf, True, None,
                                [1e-5]])
def test_invalid_stabilization_factor(mu):
    """A stabilization factor that is not a positive finite number is a
    ``ValueError``"""
    p = truss_beyond_limit(NL_method='stabilized_NR', stabilization_factor=mu)
    with pytest.raises(ValueError, match='stabilization_factor'):
        p.an.static(NLgeom=True, silent=True)


def test_dense_matrices():
    """Callables that return dense arrays instead of sparse matrices give
    the states of the sparse ones"""
    p = von_mises_truss(frac=1.3)
    d = Problem(p.fext, p.fint, p.kC, p.kG)
    d.calc_kC = lambda c=None, NLgeom=False, silent=True: np.array(p.kC)
    d.calc_kG = lambda c=None, NLgeom=False, silent=True: np.array(p.kG(c))
    out = []
    for q in (p, d):
        q.analysis(NL_method='stabilized_NR', initialInc=0.1)
        q.an.static(NLgeom=True, silent=True)
        out.append(q.an)
    assert out[1].stabilization_status == out[0].stabilization_status
    np.testing.assert_allclose(out[1].cs[-1], out[0].cs[-1], rtol=1e-10)


def test_increment_growth():
    """After an increment solved by the plain iterations within
    ``stabilization_inc_growth_max_iter`` corrections the load increment is
    multiplied by ``stabilization_inc_growth``, and kept otherwise"""
    p = hardening_springs()
    p.analysis(NL_method='stabilized_NR', initialInc=0.1)
    p.an.static(NLgeom=True, silent=True)
    steps = np.diff(np.concatenate(([0.], p.an.increments)))
    np.testing.assert_allclose(steps[1:-1]/steps[:-2], 1.5, rtol=1e-12)
    assert p.an.increments[-1] == 1.
    q = hardening_springs()
    q.analysis(NL_method='stabilized_NR', initialInc=0.1,
               stabilization_inc_growth_max_iter=1)
    q.an.static(NLgeom=True, silent=True)
    np.testing.assert_allclose(q.an.increments, np.linspace(0.1, 1., 10),
                               rtol=1e-12)


def test_stage_iteration_control():
    """The stages stop at a non-finite residual, after
    ``stabilization_max_iter`` corrections, or when the residual exceeds
    ``stabilization_divergence_factor`` times its first value, at any
    iteration, but not when the residual stagnates, which
    ``NL_method='NR'`` stops as too slow"""
    from structsolve.newton_raphson import (_check_divergence_stabilized,
                                            _check_divergence)

    def run(norms, control):
        Rnorms = []
        for i, r in enumerate(norms, start=1):
            if _check_divergence_stabilized(r, Rnorms, i, control,
                                            silent=True):
                return i
        return None

    stagnating = [1.] + [0.9]*20
    assert run(stagnating, (25, 1e3)) is None
    assert run(stagnating, (10, 1e3)) == 11
    assert run([1., 2., 1001.], (25, 1e3)) == 3
    assert run([1., np.nan], (25, 1e3)) == 2
    an = hardening_springs().analysis()
    Rnorms = []
    stops = [_check_divergence(an, r, Rnorms, i, silent=True)
             for i, r in enumerate(stagnating, start=1)]
    assert any(stops)


@pytest.mark.parametrize('name, value', [
    ('stabilization_max_iter', 0), ('stabilization_max_iter', 2.5),
    ('stabilization_inc_growth_max_iter', -1),
    ('stabilization_divergence_factor', 0.5),
    ('stabilization_divergence_factor', np.inf),
    ('stabilization_inc_growth', 0.9), ('stabilization_inc_growth', None)])
def test_invalid_stabilization_options(name, value):
    """Invalid iteration control or increment growth is a ``ValueError``"""
    p = truss_beyond_limit(NL_method='stabilized_NR', **{name: value})
    with pytest.raises(ValueError, match=name):
        p.an.static(NLgeom=True, silent=True)
