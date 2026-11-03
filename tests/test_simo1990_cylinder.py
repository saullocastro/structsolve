r"""Post-buckling of an axially compressed cylinder, Simo, Fox and Rifai (1990)

Benchmark of Section 6.3.4, Fig. 6.3.4 of Simo, Fox and Rifai (1990), with
the model of ``notebooks/simo1990_utils.py``, see the notebook
``notebooks/simo1990_cylinder_postbuckling.ipynb`` and the documentation
page ``ex_cylinder_postbuckling``. The values of the figure were read with
a precision of about +-0.0003 in the axial deflection and +-0.01 in the
load normalized by the linear buckling load.

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'notebooks'))
import simo1990_utils as su # noqa: E402


def test_imperfection_identities():
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


def test_riks_imperfect_cylinder_50pct():
    # Fig. 6.3.4 (a), imperfection of 50 % of the thickness, 8 x 8 terms
    r = su.run_riks(0.5, 8, 8, umax=0.021, silent=True, save=False,
                    verbose=False)
    # linear buckling load of the diamond mode, 0.47 % above the classical
    # one with 8 x 8 terms (0.42 % with 10 x 10 to 14 x 14)
    assert abs(r['lbd_cr'] - 1.00472) <= 1e-5
    assert r['max_residual'] <= 1e-9
    v = su.path_values(r['lbds'], r['u'][:, 1])
    # limit load: 0.411 at 0.0108 read for Simo et al., 0.402 at 0.0106 for
    # Stanley; 0.393 at 0.0107 (0.390 with 12 x 12 terms)
    assert abs(v['P_lim'] - 0.411) <= 0.025
    assert abs(v['P_lim'] - 0.402) <= 0.015
    assert abs(v['u_lim'] - 0.0108) <= 0.0005
    # post-buckling plateau: 0.315 at 0.015 to 0.020 read for Simo et al.,
    # 0.297 at 0.0175 for Stanley; 0.323 at 0.0173
    assert abs(v['P_min'] - 0.315) <= 0.015
    assert abs(v['P(0.02)'] - 0.315) <= 0.015


@pytest.mark.parametrize('linear_solver', ['splu', 'pardiso'])
def test_riks_perfect_cylinder_branch_switching(linear_solver):
    # Fig. 6.3.4 (d), the perfect cylinder, 8 x 8 terms, unknowns scaled to
    # a unit diagonal, switch at the first bifurcation along the critical
    # eigenvector: by the sign of the determinant with 'splu', by the count
    # of negative eigenvalues with 'pardiso'
    if linear_solver == 'pardiso':
        if not pardiso_available():
            pytest.skip('oneMKL is not found')
    r = su.run_riks(0., 8, 8, umax=0.022, branch_switching=1, silent=True,
                    save=False, verbose=False, linear_solver=linear_solver)
    assert r['max_residual'] <= 1e-9
    assert len(r['branch_points']) == 1
    lbds, u_max = r['lbds'], r['u'][:, 1]
    ib = r['branch_points'][0]
    # bifurcation: 0.833 at 0.0183 read for Simo et al.; 0.827 at 0.0182,
    # where a dense eigenvalue analysis of the tangent stiffness matrix has
    # its first negative eigenvalue (0.8273, 0.8270 and 0.8267 with 8 x 8,
    # 10 x 10 and 12 x 12 terms)
    assert abs(lbds[ib] - 0.827) <= 0.005
    assert abs(u_max[ib] - 0.0183) <= 0.0003
    v = su.path_values(lbds, u_max)
    assert v['P_lim'] == lbds[ib]
    # snap-back of the bifurcated branch: 0.0149 at the load 0.457 read for
    # Simo et al.; 0.0145 at 0.453
    assert abs(v['u_back'] - 0.0149) <= 0.0006
    assert abs(v['P_back'] - 0.457) <= 0.01
    # minimum post-buckling load: 0.333 at 0.020 read for Simo et al.; 0.327
    # at 0.0208
    assert abs(v['P_min'] - 0.327) <= 0.003
    assert abs(v['P_min'] - 0.333) <= 0.01
    assert abs(v['u_min'] - 0.020) <= 0.001
