import numpy as np
from numpy import dot
from scipy.sparse import csr_matrix, vstack as spvstack, hstack as sphstack

from .static import solve
from .logger import msg, warn


TOO_SLOW = 0.001

def _solver_arc_length_riks(an, silent=False):
    r"""Arc-Length solver using the Riks method

    """
    msg('Initialization...', level=1)
    lbd = 0.
    arc_length = an.initialInc
    length = arc_length
    dlbd = arc_length
    max_arc_length = an.maxArcLength

    modified_NR = an.modified_NR
    fext = an.calc_fext(inc=1., silent=silent)
    kC = an.calc_kC(silent=silent)
    kT = kC
    c = solve(kC, arc_length*fext, silent=silent)
    fint = kC*c
    dc = c
    c_last = c.copy()

    step_num = 1

    if modified_NR:
        compute_NL_matrices = False
    else:
        compute_NL_matrices = True

    while step_num < 1000:
        msg('Step %d, lbd %1.5f, arc-length %1.5f' % (step_num, lbd, arc_length), level=1)
        min_Rmax = 1.e6
        prev_Rmax = 1.e6
        converged = False
        iteration = 0
        varlbd = 0
        varc = 0
        phi = 1 # spheric arc-length

        while True:
            iteration += 1
            if iteration > an.maxNumIter:
                warn('Maximum number of iterations achieved!', level=2)
                break
            q = fext
            TMP = sphstack((kT, -q[:, None]), format='csr')
            dcext = np.concatenate((dc, [0.]))
            TMP = spvstack((TMP, 2*dcext[None, :]), format='csr')
            TMP[-1, -1] = 2*phi**2*dlbd*np.dot(q, q)
            right_vec = np.zeros(q.shape[0]+1, dtype=q.dtype)

            R = fint - (lbd + dlbd)*q
            A = - (np.dot(dc, dc) + phi**2*dlbd**2*np.dot(q, q) - arc_length**2)
            right_vec[:-1] = -R
            right_vec[-1] = A
            solution = solve(TMP, right_vec, silent=silent)
            varc = solution[:-1]
            varlbd = solution[-1]

            dlbd = dlbd + varlbd
            dc = dc + varc

            msg('iter %d, lbd+dlbd %1.5f' % (iteration, lbd+dlbd), level=2)

            # computing the Non-Linear matrices
            if compute_NL_matrices:
                kC = an.calc_kC(c=(c + dc), silent=silent, NLgeom=True)
                kG = an.calc_kG(c=(c + dc), silent=silent, NLgeom=True)
                kT = kC + kG
                if modified_NR:
                    compute_NL_matrices = False
            else:
                if not modified_NR:
                    compute_NL_matrices = True

            # calculating the residual
            fint = an.calc_fint(c + dc, silent=silent)
            Rmax = np.abs((lbd + dlbd)*fext - fint).max()
            if Rmax <= an.absTOL:
                converged = True
                break
            if (Rmax > min_Rmax and Rmax > prev_Rmax and iteration > 3):
                warn('Diverged - Rmax value significantly increased', level=2)
                break
            else:
                min_Rmax = min(min_Rmax, Rmax)
            change_rate_Rmax = abs(1 - Rmax/prev_Rmax)
            if (iteration > 2 and change_rate_Rmax < TOO_SLOW):
                warn('Diverged - convergence too slow', level=2)
                break
            prev_Rmax = Rmax

        if converged:
            step_num += 1
            msg('Converged at lbd+dlbd of %1.5f, total length %1.5f' % (lbd + dlbd, length), level=2)
            length += arc_length

            lbd = lbd + dlbd

            arc_length *= 1.1111

            dlbd = arc_length
            c_last = c.copy()
            c = c + dc

            an.increments.append(lbd)
            an.cs.append(c.copy())

        else:
            msg('Reseting step with reduced arc-length', level=2)
            arc_length *= 0.90

        if length >= max_arc_length:
            msg('Maximum specified arc-length of %1.5f achieved' % max_arc_length, level=2)
            break

        dc = c_last - c
        dlbd = arc_length

        kC = an.calc_kC(c=c, silent=silent, NLgeom=True)
        kG = an.calc_kG(c=c, silent=silent, NLgeom=True)
        kT = kC + kG
        fint = an.calc_fint(c=c, silent=silent)
        compute_NL_matrices = False

    msg('Finished Non-Linear Static Analysis')
    msg('    total arc-length %1.5f' % length, level=1)
