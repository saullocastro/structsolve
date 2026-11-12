r"""Post-buckling of an axially compressed cylinder, Simo, Fox and Rifai (1990)

Verifies the Riks method, the detection of a limit load and the branch
switching of :class:`structsolve.Analysis` with both linear solvers on the
benchmark of Section 6.3.4 of Simo, Fox and Rifai (1990), with the Ritz
model of panels built by the module ``simo1990_utils.py`` of the tests, in
small models of 6 x 6 and 5 x 5 terms: the model of the imperfect shell, the path
of the imperfect shell through its limit load, and the switch of the
perfect shell to its bifurcated branch, against the eigenvalues of the
dense tangent stiffness matrix and values of the code. The tests are
skipped where panels is not installed.

References
----------
.. [1] Simo, J.C., Fox, D.D. and Rifai, M.S., On a stress resultant
   geometrically exact shell model. Part III: Computational aspects of the
   nonlinear theory, Computer Methods in Applied Mechanics and Engineering,
   79(1), 21-70, 1990. https://doi.org/10.1016/0045-7825(90)90094-3

"""
import os
import sys

import numpy as np
import pytest

from structsolve.pardiso import available as pardiso_available

# panels (which requires structsolve>=0.10.0) is not a dependency of
# structsolve: these tests run where panels is installed, and are skipped
# otherwise
pytest.importorskip('panels')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simo1990_utils as su # noqa: E402


def test_imperfection_identities():
    """The model of the imperfect shell of ``simo1990_utils.py``: its
    geometric stiffness matrix of the linear membrane resultants against
    ``calc_kG`` of panels, null internal forces at ``c = 0``, the perfect
    shell without imperfection, and the tangent stiffness matrix and the
    internal forces against central finite differences of the internal
    forces and of the strain energy"""
    s, lbd_cr, mode = su.normalized_shell(6, 6)
    size = s.get_size()
    rng = np.random.default_rng(1)
    # the geometric stiffness of the linear resultants is panels'
    # calc_kG(c), which has no resultant of the non-linear strains
    c = rng.standard_normal(size)*1e-2
    kG = s.calc_kG(c=c, NLgeom=True).toarray()
    kG_L = su.calc_kG_N(s, su.membrane_stresses(s, c, NLgeom=False)).toarray()
    kG_N = su.calc_kG_N(s, su.membrane_stresses(s, c, NLgeom=True)).toarray()
    assert np.abs(kG_L - kG).max() <= 1e-12*np.abs(kG).max()
    assert np.abs(kG_N - kG).max() >= 1e-4*np.abs(kG).max()
    c0 = su.imperfection(s, mode, 0.5)
    model = su.ImperfectShell(s, c0)
    # the imperfect shell is unstressed at c = 0
    assert np.linalg.norm(model.calc_fint(np.zeros(size))) == 0
    # without imperfection the model is the perfect one
    model0 = su.ImperfectShell(s, 0*c0)
    c = np.zeros(size)
    c[2::3] = rng.standard_normal(size//3)
    c *= 0.3*su.t/np.abs(s.uvw(c, gridx=21, gridy=21)[1]['w']).max()
    c[0::3] = rng.standard_normal(size//3)*1e-3
    c[1::3] = rng.standard_normal(size//3)*1e-3
    f = s.calc_fint(c)
    assert np.linalg.norm(model0.calc_fint(c) - f) <= 1e-14*np.linalg.norm(f)
    # the tangent stiffness matrix is the derivative of the internal forces
    v = rng.standard_normal(size)
    v[np.asarray(abs(model.kT0).sum(axis=1)).ravel() == 0] = 0
    h = 1e-5
    dfd = (model.calc_fint(c + h*v) - model.calc_fint(c - h*v))/(2*h)
    kTv = model.calc_kT(c) @ v
    assert np.linalg.norm(dfd - kTv) <= 1e-9*np.linalg.norm(kTv)
    # the internal forces are the gradient of the strain energy of the
    # strains eps(c + c0) - eps(c0)
    dU = (su.strain_energy_imp(s, c + h*v, c0)
          - su.strain_energy_imp(s, c - h*v, c0))/(2*h)
    fv = model.calc_fint(c) @ v
    assert abs(dU - fv) <= 1e-7*abs(fv)
    # which fails with panels' calc_kG(c0) in place of kG(N0)
    f_L = (s.calc_fint(c + c0) - model.fint0
           - s.calc_kG(c=c0, NLgeom=True) @ c) @ v
    assert abs(dU - f_L) >= 1e-6*abs(fv)


def _num_negative_eigvals(kT):
    """Negative eigenvalues of the dense ``kT`` below the round-off of its
    null rows"""
    ev = np.linalg.eigvalsh(kT.toarray())
    return int((ev < -1e-9*np.abs(ev).max()).sum())


def test_riks_imperfect_cylinder_limit_load():
    """Riks method on the imperfect shell, 50 % of the thickness, 5 x 5
    terms: the path passes the limit load, where the load factor decreases
    for the first time, every increment in equilibrium; the tangent
    stiffness matrix of the model has no negative eigenvalue before the
    limit load and one at it; the linear buckling load and the limit load
    and deflection at it are those of the code

    Notes
    -----
    The limit load with 5 x 5 terms, 0.419 at the maximum axial deflection
    0.0113 of the loaded edge, converges towards that of 8 x 8 terms, 0.393
    at 0.0107 (0.401 with 6 x 6 terms), against 0.411 at 0.0108 read from
    Fig. 6.3.4 (a) of Simo, Fox and Rifai (1990).

    """
    from structsolve import callbacks

    r = su.run_riks(0.5, 5, 5, umax=0.0125, silent=True, save=False,
                    verbose=False)
    assert r['max_residual'] <= 1e-9
    an, lbds, u_max = r['an'], r['lbds'], r['u'][:, 1]
    # value of the code, 5 x 5 terms
    assert abs(r['lbd_cr'] - 1.019207) <= 1e-5
    ilim = np.flatnonzero(np.diff(lbds) < 0)[0]
    # values of the code, 5 x 5 terms, see the Notes
    assert abs(lbds[ilim] - 0.41896) <= 1e-4
    assert abs(u_max[ilim] - 0.011292) <= 1e-5
    assert u_max[-1] > u_max[ilim] and lbds[-1] < lbds[ilim] - 0.02
    kT = [callbacks.calc_kT(an, an.cs[i], lbds[i]) for i in (ilim - 1, ilim)]
    assert [_num_negative_eigvals(k) for k in kT] == [0, 1]


@pytest.mark.parametrize('linear_solver', ['splu', 'pardiso'])
def test_riks_perfect_cylinder_branch_switching(linear_solver):
    """Riks method on the perfect shell, 5 x 5 terms, unknowns scaled to a
    unit diagonal, ``inertia=True``, ``branch_switching=1``: the bifurcation
    is detected by the sign of the determinant with ``'splu'`` and by the
    count of negative eigenvalues with ``'pardiso'`` at the first increment
    where the dense tangent stiffness matrix has a negative eigenvalue, and
    the path switches there along the critical eigenvector to the
    bifurcated branch, on which the load decreases; every increment is in
    equilibrium and the bifurcation load is that of the code, the same with
    both linear solvers

    Notes
    -----
    The bifurcation load with 5 x 5 terms, 0.908 at the maximum axial
    deflection 0.0200 of the loaded edge, converges towards that of 8 x 8
    terms, 0.827 at 0.0182 (0.840 with 6 x 6 terms), against 0.833 at
    0.0183 read from Fig. 6.3.4 (d) of Simo, Fox and Rifai (1990).

    """
    from structsolve import callbacks

    if linear_solver == 'pardiso':
        if not pardiso_available():
            pytest.skip('oneMKL is not found')

    def stop(an):
        # eight increments on the bifurcated branch
        return (len(an.branch_points) > 0
                and len(an.increments) >= an.branch_points[0] + 9)

    r = su.run_riks(0., 5, 5, branch_switching=1, silent=True, save=False,
                    verbose=False, linear_solver=linear_solver,
                    stop_criterion=stop)
    assert r['max_residual'] <= 1e-9
    an, lbds, u_max = r['an'], r['lbds'], r['u'][:, 1]
    assert len(r['branch_points']) == 1
    ib = r['branch_points'][0]
    # values of the code, 5 x 5 terms, see the Notes
    assert abs(lbds[ib] - 0.90796) <= 1e-4
    assert abs(u_max[ib] - 0.019969) <= 1e-5
    # fundamental path up to the bifurcation, then the bifurcated branch
    assert np.all(np.diff(lbds[:ib + 1]) > 0)
    assert np.all(np.diff(lbds[ib:]) < 0)
    if linear_solver == 'pardiso':
        assert an.num_negative_eigvals[ib - 1:ib + 1] == [0, 1]
    assert an.det_signs[ib - 1:ib + 1] == [1, -1]
    kT = [callbacks.calc_kT(an, an.cs[i], lbds[i]) for i in (ib - 1, ib)]
    assert [_num_negative_eigvals(k) for k in kT] == [0, 1]
