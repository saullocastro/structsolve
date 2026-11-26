"""Tests of ``Analysis.on_converged``

The callable is called once per converged increment, with the new state
already appended, and never for a rejected or cut increment. A
path-dependent spring whose internal variable is updated only by
``on_converged`` follows the exact solution, which it would not if the
variable were updated during the iterations.

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


def _softening_spring():
    r"""One spring with irreversible secant softening, loaded by a force
    beyond its peak along a displacement-controlled path

    The internal variable ``state['d']``, the damage, is a function of the
    largest displacement of the converged states only, updated by the
    ``on_converged`` returned. Returns ``state`` and the callables
    ``calc_fext``, ``calc_fint``, ``calc_kC``, ``calc_kG`` and
    ``on_converged`` of the analysis.

    Notes
    -----
    The force of the spring is `f(u) = k (1 - d) u` with
    `d = \max_{past} g(|u|)`, `g` growing linearly from 0 at `u_0` to 0.5
    at `2 u_0`, with `k = 1` and `u_0 = 1`; the external force is the load
    factor `\lambda`. During the iterations the damage is
    `\max(d_{conv}, g(|u|))`, with `d_{conv}` that of the last converged
    state.

    """
    k, u0 = 1., 1.

    def g(u):
        return np.clip(0.5*(abs(u) - u0)/u0, 0., 0.5)

    state = {'d': 0.}

    def dmg(u):
        return max(state['d'], g(u))

    def calc_fint(c, inc=1., silent=True):
        return np.array([k*(1 - dmg(c[0]))*c[0]])

    def calc_kC(c=None, NLgeom=False, inc=1., silent=True):
        if c is None:
            return csr_matrix(np.array([[k]]))
        u = c[0]
        d = dmg(u)
        kt = k*(1 - d)
        if g(u) > state['d'] and 0. < g(u) < 0.5:
            kt -= k*u*np.sign(u)*0.5/u0
        return csr_matrix(np.array([[kt]]))

    def calc_kG(c=None, NLgeom=False, inc=1., silent=True):
        return csr_matrix((1, 1))

    def calc_fext(inc=1., silent=True):
        return np.array([inc*1.])

    def on_converged(an):
        state['d'] = dmg(an.cs[-1][0])

    return state, calc_fext, calc_fint, calc_kC, calc_kG, on_converged


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
