"""Tests of the options for the robustness of the arc-length methods

``Analysis.arc_length_line_search``, ``Analysis.arc_length_predictor`` and
``Analysis.early_divergence``, see :func:`.arc_length._solver_arc_length`
and :func:`.newton_raphson._check_divergence`, on the analytic problems of
``test_arc_length.py``: alone and together, the snap-through, the
snap-back and the long steps through the limit point are traced as with
the defaults.

"""
import numpy as np
import pytest

from structsolve import Analysis
from structsolve.newton_raphson import (_check_divergence,
                                        EARLY_DIVERGENCE_FACTOR,
                                        EARLY_DIVERGENCE_MAX_RREL,
                                        EARLY_DIVERGENCE_NUM_ITER)
from analytic_problems import hardening_springs, von_mises_truss
from test_arc_length import check_equilibrium

METHODS = ['arc_length_riks', 'arc_length_crisfield']

OPTIONS = {
    'line_search': dict(arc_length_line_search=True),
    'secant': dict(arc_length_predictor='secant'),
    'early_divergence': dict(early_divergence=True),
    'all': dict(arc_length_line_search=True, arc_length_predictor='secant',
                early_divergence=True),
}


def count_kT(p):
    """Wrap ``p.calc_kG`` to count the evaluations of the tangent stiffness
    matrix"""
    calls = []
    calc_kG = p.calc_kG

    def wrapped(c=None, NLgeom=False, silent=True):
        calls.append(len(p.an.increments))
        return calc_kG(c=c, NLgeom=NLgeom, silent=silent)
    p.calc_kG = wrapped
    return calls


@pytest.mark.parametrize('options', list(OPTIONS))
@pytest.mark.parametrize('method', METHODS)
def test_snap_through(method, options):
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05, **OPTIONS[options])
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    w = np.array([c[0] for c in p.an.cs])
    assert inc[-1] == 1.0
    # limit load factor 1/1.5, before the minimum -1/1.5 of the path
    assert inc[:inc.argmin()].max() < 1/1.5 + 1e-6
    assert inc.min() < -0.5
    assert np.all(np.diff(w) > 0)
    assert w[-1] > 2.


@pytest.mark.parametrize('options', list(OPTIONS))
@pytest.mark.parametrize('method', METHODS)
def test_snap_back(method, options):
    p = von_mises_truss(frac=1.5, ks=0.5)
    p.analysis(NL_method=method, initialInc=0.05, **OPTIONS[options])
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    inc = np.asarray(p.an.increments)
    cs = np.asarray(p.an.cs)
    assert inc[-1] == 1.0
    assert inc.min() < -0.5
    assert np.any(np.diff(cs[:, 1]) < 0)
    assert np.all(np.diff(cs[:, 0]) > 0)


@pytest.mark.parametrize('options', list(OPTIONS))
@pytest.mark.parametrize('method', METHODS)
def test_no_doubling_back(method, options):
    p = von_mises_truss(frac=10., ks=1.)
    p.analysis(NL_method=method, initialInc=0.2, **OPTIONS[options])
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    cs = np.asarray(p.an.cs)
    assert p.an.increments[-1] == 1.0
    assert np.all(np.diff(np.r_[0., cs[:, 0]]) > 0)
    np.testing.assert_allclose(cs[-1],
                               p.solve_reference(c0=cs[-1], tol=1e-12),
                               rtol=1e-6)


@pytest.mark.parametrize('method', METHODS)
def test_defaults_unchanged(method):
    """The defaults are the former behaviour: no line search, tangent
    predictor, the divergence checks of ``NUM_ITER_BEFORE_DIVERGENCE_CHECK``"""
    an = Analysis()
    assert an.arc_length_line_search is False
    assert an.arc_length_predictor == 'tangent'
    assert an.early_divergence is False
    p = von_mises_truss(frac=1.5)
    p.analysis(NL_method=method, initialInc=0.05)
    p.an.static(NLgeom=True, silent=True)
    q = von_mises_truss(frac=1.5)
    q.analysis(NL_method=method, initialInc=0.05, **OPTIONS['all'])
    q.an.static(NLgeom=True, silent=True)
    assert len(p.fint_calls) != len(q.fint_calls)


@pytest.mark.parametrize('method', METHODS)
def test_line_search_full_correction_no_extra_cost(method):
    """On the hardening springs every full correction decreases the
    residual: the line search accepts it and reuses its internal force
    vector, with the same path and number of evaluations as without"""
    results = []
    for line_search in (False, True):
        p = hardening_springs()
        p.analysis(NL_method=method, initialInc=0.1,
                   arc_length_line_search=line_search)
        p.an.static(NLgeom=True, silent=True)
        check_equilibrium(p)
        results.append((np.asarray(p.an.increments), np.asarray(p.an.cs),
                        len(p.fint_calls)))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-12)
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12)
    assert results[1][2] == results[0][2]


@pytest.mark.parametrize('method', METHODS)
def test_line_search_reduces_correction(method, capsys):
    """Long steps through the limit point of the von Mises truss: some
    corrections are reduced and the path is traced to the load factor 1"""
    p = von_mises_truss(frac=10., ks=1.)
    p.analysis(NL_method=method, initialInc=0.2, arc_length_line_search=True)
    p.an.static(NLgeom=True, silent=False)
    out = capsys.readouterr().out
    check_equilibrium(p)
    assert p.an.increments[-1] == 1.0
    assert out.count('Line-search: eta') > 0


@pytest.mark.parametrize('method', METHODS)
def test_secant_predictor(method):
    """From the second step on, the first iterate of a step is the last
    converged point plus the previous increment scaled to the arc length,
    and the tangent stiffness matrix is not evaluated at the converged
    state: one evaluation less per step than with the tangent predictor"""
    p = hardening_springs()
    kT_calls = count_kT(p)
    p.analysis(NL_method=method, initialInc=0.1, arc_length_predictor='secant')
    p.an.static(NLgeom=True, silent=True)
    check_equilibrium(p)
    cs = [np.zeros(4)] + list(p.an.cs)
    lbds = [0.] + list(p.an.increments)
    inc = np.asarray(p.an.increments)
    assert inc[-1] == 1.0
    assert np.all(np.diff(inc) > 0)
    np.testing.assert_allclose(p.an.cs[-1], p.solve_reference(), rtol=1e-5)
    for step in range(1, len(p.an.increments) - 1):
        first = [c for s, c in p.fint_calls if s == step][0]
        d_prev = cs[step] - cs[step - 1]
        d_pred = first - cs[step]
        # parallel and in the same direction
        cos = d_pred.dot(d_prev)/np.linalg.norm(d_pred)/np.linalg.norm(d_prev)
        assert cos > 1 - 1e-12, (step, cos)

    q = hardening_springs()
    kT_calls_tangent = count_kT(q)
    q.analysis(NL_method=method, initialInc=0.1)
    q.an.static(NLgeom=True, silent=True)
    # the steps of the two predictors differ in their number of corrections,
    # the tangent predictor evaluates kT once more per step
    for calls, pp in ((kT_calls, p), (kT_calls_tangent, q)):
        per_step = [calls.count(i) for i in range(len(pp.an.increments))]
        corr = [n - 1 for n in pp.calls_per_step()]
        extra = 0 if pp is p else 1
        # the first step always starts from the tangent; the last step
        # through lbd = 1 adds the iterations at lbd = 1
        for i in range(1, len(per_step) - 1):
            assert per_step[i] == corr[i] + extra, (pp is p, i, per_step, corr)


def test_invalid_predictor():
    p = hardening_springs()
    p.analysis(NL_method='arc_length_riks', arc_length_predictor='chord')
    with pytest.raises(ValueError, match='arc_length_predictor'):
        p.an.static(NLgeom=True, silent=True)


def test_early_divergence_check():
    """A residual that grows past ``EARLY_DIVERGENCE_FACTOR`` times its
    minimum after ``EARLY_DIVERGENCE_NUM_ITER`` iterations, or a relative
    residual above ``EARLY_DIVERGENCE_MAX_RREL``, stops the step only with
    ``early_divergence=True``"""
    for early in (False, True):
        an = Analysis()
        an.early_divergence = early
        Rnorms = []
        seq = [1., 0.5, 0.4] + [1.01*EARLY_DIVERGENCE_FACTOR*0.4]
        stops = [_check_divergence(an, r, Rnorms, i + 1, silent=True,
                                   Rrel=1.)
                 for i, r in enumerate(seq)]
        assert len(seq) == EARLY_DIVERGENCE_NUM_ITER + 1
        assert stops == [False]*3 + [early]
        an = Analysis()
        an.early_divergence = early
        assert _check_divergence(an, 1., [], 1, silent=True,
                                 Rrel=2*EARLY_DIVERGENCE_MAX_RREL) is early
        # below the factor, or before the number of iterations: no stop
        assert not _check_divergence(an, 9., [1.], 2, silent=True, Rrel=1.)
        assert not _check_divergence(an, 3.9, [1., 0.5, 0.4], 4, silent=True,
                                     Rrel=1.)


@pytest.mark.parametrize('method', METHODS + ['NR'])
def test_early_divergence_cuts_earlier(method):
    """With few iterations allowed per step, the steps that diverge are cut
    before the iterations are spent: fewer evaluations for the same
    result"""
    frac = 1.5 if method != 'NR' else 0.99
    calls = []
    for early in (False, True):
        p = von_mises_truss(frac=frac, ks=1.)
        p.analysis(NL_method=method, initialInc=0.6, maxInc=0.6,
                   early_divergence=early)
        p.an.static(NLgeom=True, silent=True)
        check_equilibrium(p)
        assert p.an.increments[-1] == 1.0
        calls.append(len(p.fint_calls))
    assert calls[1] <= calls[0], calls
