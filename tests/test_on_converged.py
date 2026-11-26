"""Tests of ``Analysis.on_converged``

The callable is called once per converged increment, with the new state
already appended, and never for a rejected or cut increment. A
path-dependent spring whose internal variable is updated only by
``on_converged`` follows its exact equilibrium path beyond the peak of the
force, under displacement control and with the Riks method, whereas with
the variable updated during the iterations the Riks method leaves the exact
path by a derived amount.

"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from structsolve import Analysis
from analytic_problems import hardening_springs, von_mises_truss
from test_stopping import ARC_METHODS, kwargs_of


@pytest.mark.parametrize('method', ['NR'] + ARC_METHODS)
def test_called_once_per_converged_increment(method):
    seen = []

    def on_converged(an):
        seen.append((len(an.cs), len(an.increments), an.increments[-1],
                     an.cs[-1].copy()))

    p = hardening_springs()
    kwargs = {}
    if method == 'displacement_control':
        kwargs['control_vector'] = np.array([0., 0., 0., 1.])
    p.analysis(NL_method=method, initialInc=0.1, final_load_factor=1.,
               on_converged=on_converged, **kwargs)
    p.an.static(NLgeom=True, silent=True)
    # one call per converged increment, the state already appended
    assert len(seen) == len(p.an.increments) > 1
    for k, (ncs, ninc, lbd, c) in enumerate(seen):
        assert ncs == ninc == k + 1
        assert lbd == p.an.increments[k]
        np.testing.assert_array_equal(c, p.an.cs[k])


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks'])
def test_not_called_for_cut_increments(method):
    """Increments that do not converge are cut and retried: the calls
    match the converged increments only"""
    calls = []
    p = von_mises_truss(frac=0.99 if method == 'NR' else 1.5)
    p.analysis(NL_method=method, initialInc=0.5, maxInc=0.5,
               on_converged=lambda an: calls.append(len(an.cs)))
    p.an.static(NLgeom=True, silent=True)
    assert calls == list(range(1, len(p.an.cs) + 1))


def _softening_spring(update='converged'):
    r"""One spring with irreversible secant softening under a force
    proportional to the load factor

    The internal variable ``state['d']``, the damage, never decreases. With
    ``update='converged'`` (default) it is a function of the largest
    displacement of the converged states only, updated by the
    ``on_converged`` returned; with ``update='iterations'`` it is updated
    by every call of ``calc_fint``, i.e. also with the iterates that are
    not converged, and ``on_converged`` does nothing. Returns ``state``
    and the callables ``calc_fext``, ``calc_fint``, ``calc_kC``,
    ``calc_kG`` and ``on_converged`` of the analysis.

    Notes
    -----
    The force of the spring is `f(u) = k (1 - d) u` with
    `d = \max_{past} g(|u|)`, `g(u) = (u - u_0)/(2 u_0)` limited to
    `[0, 1/2]`, with `k = 1` and `u_0 = 1`; the external force is the load
    factor `\lambda`. During the iterations the damage is
    `\max(d_{conv}, g(|u|))`, with `d_{conv}` that of the last converged
    state. Under a monotonically increasing displacement, `d = g(u)` and
    the equilibrium path is

    .. math::

        \lambda(u) = \begin{cases} u & u \le 1 \\ (3 u - u^2)/2 & 1 \le u
        \le 2 \\ u/2 & u \ge 2 \end{cases}

    with the peak `\lambda = 9/8` at `u = 3/2`, the softening down to
    `\lambda = 1` at `u = 2` and a second elastic range of half the
    stiffness.

    """
    k, u0 = 1., 1.

    def g(u):
        return np.clip(0.5*(abs(u) - u0)/u0, 0., 0.5)

    state = {'d': 0.}

    def dmg(u):
        return max(state['d'], g(u))

    def calc_fint(c, inc=1., silent=True):
        if update == 'iterations':
            state['d'] = dmg(c[0])
        return np.array([k*(1 - dmg(c[0]))*c[0]])

    def calc_kC(c=None, NLgeom=False, inc=1., silent=True):
        if c is None:
            return csr_matrix(np.array([[k]]))
        u = c[0]
        d = dmg(u)
        kt = k*(1 - d)
        if update == 'iterations':
            # the damage of calc_fint at u is already in the state
            loading = g(u) >= state['d']
        else:
            loading = g(u) > state['d']
        if loading and 0. < g(u) < 0.5:
            kt -= k*u*np.sign(u)*0.5/u0
        return csr_matrix(np.array([[kt]]))

    def calc_kG(c=None, NLgeom=False, inc=1., silent=True):
        return csr_matrix((1, 1))

    def calc_fext(inc=1., silent=True):
        return np.array([inc*1.])

    def on_converged(an):
        if update == 'converged':
            state['d'] = dmg(an.cs[-1][0])

    return state, calc_fext, calc_fint, calc_kC, calc_kG, on_converged


def _exact_load_factor(u):
    """Load factor of the equilibrium path of :func:`_softening_spring` at
    the displacements ``u``, monotonically increasing"""
    u = np.asarray(u)
    return (1 - np.clip(0.5*(u - 1.), 0., 0.5))*u


def _run_softening_spring(NL_method, update):
    """Analysis of :func:`_softening_spring` up to the load factor 1.2, in
    the second elastic range, with increments of 0.1

    Returns the displacements, the load factors and the damage of the
    converged increments.
    """
    state, fext, fint, kC, kG, on_converged = _softening_spring(update)
    damage = []

    def record(an):
        on_converged(an)
        damage.append(state['d'])

    an = Analysis(fext, fint, kC, kG)
    an.NL_method = NL_method
    an.initialInc = 0.1
    an.maxInc = 0.1
    an.final_load_factor = 1.2
    if NL_method == 'displacement_control':
        an.control_vector = np.array([1.])
    an.on_converged = record
    an.static(NLgeom=True, silent=True)
    u = np.array([c[0] for c in an.cs])
    return u, np.array(an.increments), np.array(damage)


def test_history_updated_only_at_converged_states():
    r"""The softening spring of :func:`_softening_spring`, loaded by the
    Newton-Raphson method up to the onset of its damage with the internal
    variable updated by ``on_converged``, ends without damage at the exact
    displacement, within ``1e-8``

    Notes
    -----
    With `d = 0` the spring is elastic, `u = \lambda/k`, up to the onset of
    the damage at `u = u_0`, i.e. at the force `k u_0 = 1`, the load
    factor `\lambda = 1` at which the analysis finishes: `u = 1` and
    `d = g(u_0) = 0`.

    """
    state, fext, fint, kC, kG, on_converged = _softening_spring()
    an = Analysis(fext, fint, kC, kG)
    an.NL_method = 'NR'
    an.initialInc = 0.05
    an.maxInc = 0.05
    an.final_load_factor = 1.
    an.on_converged = on_converged
    an.static(NLgeom=True, silent=True)
    # at the onset of the damage: no damage, u = 1
    assert state['d'] == 0.
    np.testing.assert_allclose(an.cs[-1], [1.], rtol=1e-8)


def test_softening_path_displacement_control():
    r"""The softening spring of :func:`_softening_spring`, with its damage
    updated by ``on_converged``, follows the exact equilibrium path beyond
    the peak of the force under displacement control: the load factors of
    all the increments and the damage, within ``1e-10``, through the peak,
    the softening and up to the load factor 1.2 of the second elastic range

    Notes
    -----
    The displacement, controlled, grows monotonically, by increments of
    0.1 or less, such that `d_i = g(u_i)` and the load factor is that of
    the path of :func:`_softening_spring`, `\lambda(u_i)`: increasing up
    to the peak `\lambda = 9/8` at `u = 3/2`, decreasing down to
    `\lambda = 1` at `u = 2`, where the damage reaches `1/2`, and
    increasing again up to the end at `\lambda = 1.2`, where
    `u = 2 \lambda = 2.4`.

    """
    u, lbd, damage = _run_softening_spring('displacement_control',
                                           'converged')
    assert np.all(np.diff(u) > 0) and np.all(np.diff(u) < 0.1 + 1e-10)
    np.testing.assert_allclose(lbd, _exact_load_factor(u), rtol=1e-10)
    np.testing.assert_allclose(damage, np.clip(0.5*(u - 1.), 0., 0.5),
                               rtol=0, atol=1e-10)
    # up to the peak, the softening, the second elastic range
    rising = u <= 1.5 + 1e-10
    softening = (u >= 1.5 - 1e-10) & (u <= 2. + 1e-10)
    assert np.all(np.diff(lbd[rising]) > 0)
    assert 1.12 < lbd[rising].max() <= 1.125 + 1e-10
    assert np.count_nonzero(softening) >= 5
    assert np.all(np.diff(lbd[softening]) < 0)
    np.testing.assert_allclose(lbd[softening].min(), 1., atol=0.05)
    np.testing.assert_allclose([u[-1], lbd[-1], damage[-1]], [2.4, 1.2, 0.5],
                               rtol=1e-10)


def test_damage_updated_during_the_iterations_is_not_exact():
    r"""With the Riks method, the softening spring of
    :func:`_softening_spring` follows the exact equilibrium path, within
    ``1e-8``, when its damage is updated by ``on_converged``; when it is
    updated during the iterations instead, the damage of the first
    increment beyond the onset is 39/560, that of an iterate that is not
    the converged one, instead of the exact 0.0657, the load factor is
    0.0045 below the exact path there, and up to 0.04 along the softening

    Notes
    -----
    Ten increments of 0.1 reach the onset of the damage,
    `(u, \lambda) = (1, 1)`, with `d = 0`. The eleventh starts with the
    elastic tangent predictor `(\Delta u, \Delta\lambda) = (0.1, 0.1)`,
    the displacements being scaled by the linear solution of a unit load
    factor, 1. At `u = 1.1` the internal force is
    `(3 \cdot 1.1 - 1.1^2)/2 = 1.045` and the tangent stiffness
    `3/2 - u = 0.4`. The first correction solves the equilibrium equation
    linearized there with the constraint normal to the predictor,

    .. math::

        0.4 \, \delta u - \delta\lambda = 1.1 - 1.045, \qquad
        0.1 \, \delta u + 0.1 \, \delta\lambda = 0

    i.e. `\delta u = 0.055/1.4 = 11/280`: the iterate `u = 319/280 = 1.1393`
    has a larger displacement than the state to which the increment
    converges, `u = 1.1297` with the damage updated by ``on_converged``,
    reached from above by the next corrections. Updated during the
    iterations, the damage keeps the value of that iterate,

    .. math::

        d = g(319/280) = 39/560 = 0.0696

    The increment then converges to a state of the spring of damage
    39/560, `\lambda = (1 - 39/560) u` at `u = 1.1314`, whereas the exact
    damage is that of the converged displacement, `g(1.1314) = 0.0657`:
    the load factor is below the exact path `\lambda(u)` by
    `(39/560 - g(u)) u = 0.0045`. Updated by ``on_converged``, the damage
    of the iterates is discarded and every converged state has `d = g(u)`
    and `\lambda = \lambda(u)`.

    """
    u, lbd, damage = _run_softening_spring('arc_length_riks', 'converged')
    assert np.all(np.diff(u) > 0)
    np.testing.assert_allclose(lbd, _exact_load_factor(u), rtol=1e-8)
    np.testing.assert_allclose(damage, np.clip(0.5*(u - 1.), 0., 0.5),
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose([u[9], u[10], u[-1]], [1., 1.1296752778, 2.4],
                               rtol=1e-8)

    u, lbd, damage = _run_softening_spring('arc_length_riks', 'iterations')
    assert np.all(np.diff(u) > 0)
    np.testing.assert_allclose([u[9], lbd[9], damage[9]], [1., 1., 0.],
                               rtol=1e-12, atol=1e-12)
    # the damage of the first corrected iterate, not of the converged state
    np.testing.assert_allclose(damage[10], 39/560, rtol=1e-12)
    np.testing.assert_allclose(u[10], 1.1313575641, rtol=1e-8)
    exact_damage = 0.5*(u[10] - 1.)
    np.testing.assert_allclose(lbd[10], (1 - 39/560)*u[10], rtol=1e-8)
    np.testing.assert_allclose(_exact_load_factor(u[10]) - lbd[10],
                               (39/560 - exact_damage)*u[10], rtol=1e-6)
    np.testing.assert_allclose(_exact_load_factor(u[10]) - lbd[10], 0.0045,
                               atol=1e-4)
    # along the softening the path stays below the exact one
    softening = (u > 1.) & (u < 1.95)
    error = _exact_load_factor(u) - lbd
    assert np.all(error[softening] > 4e-3)
    np.testing.assert_allclose(error.max(), 0.038, atol=1e-3)
