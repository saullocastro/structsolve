# Notebooks

Worked examples, tutorials and validation notebooks of `structsolve`, stored
with their outputs and rendered in the documentation. Each explains its steps
and compares its results with reference values (closed-form solutions, the
literature or an independent solver) in a table with the relative
differences. They are run from this folder and import the model helpers of
`tests/` (`ritz.py`, `analytic_problems.py`, `simo1990_utils.py`).

The validation against results of the literature uses structural models
of [`panels`](https://github.com/saullocastro/panels), which requires
`structsolve>=0.10.0` and is not a dependency of `structsolve`: install it
(`pip install panels`, or its repository built in place with
`python setup.py build_ext --inplace` and on the `PYTHONPATH`) to run them,
with oneMKL (`pip install mkl`) for `linear_solver='pardiso'`. Their model is
`tests/simo1990_utils.py`, shared with `tests/test_simo1990_cylinder.py`,
which is skipped where `panels` is not installed.

| File | Content |
|---|---|
| `ex_linear_static.ipynb` | Linear static analysis with `solve`, `static` and `Analysis`: the tapered bar of Bathe (2014), Example 3.22, and its mesh convergence |
| `ex_linear_buckling.ipynb` | Linear buckling of the two-bar column of Bathe (2014), Example 3.14, and of simply supported plates, Ritz model (FSDT) with Bardell's functions, against Timoshenko and Woinowsky-Krieger (1959), Eq. (227) |
| `ex_frequency.ipynb` | Frequency analysis with `freq`: the clamped-free rod of Petyt (2010), Example 3.3, options, sorting, prestressed column (Bathe 2014, Example 3.14) and plate |
| `ex_newton_raphson.ipynb` | Non-linear static analysis with the Newton-Raphson method, hardening springs |
| `ex_arc_length.ipynb` | Arc-length methods through the limit point of a von Mises truss |
| `ex_follower_loads.ipynb` | Follower loads and unsymmetric matrices: non-linear analyses, inertia, linear buckling, flutter |
| `ex_prescribed_displacements.ipynb` | Prescribed displacements imposed exactly, with and without `constraint_values` |
| `ex_cylinder_postbuckling.ipynb` | Simo, Fox and Rifai (1990): model, imperfection, convergence study in the Ritz terms and figure (`ex_cylinder_postbuckling.png`, written by the command `plot` of `tests/simo1990_utils.py`) |
| `tutorial_von_mises_truss.ipynb` | Tutorial: static, buckling, frequency and non-linear analyses of a von Mises truss |
| `simo1990_cylinder_postbuckling.ipynb` | Simo, Fox and Rifai (1990): post-buckling of an axially compressed cylinder with imperfections of 10 % and 50 % of the thickness, Riks method, and of the perfect one by branch switching with `linear_solver='pardiso'` and `'splu'`; runs `panels` and `structsolve` at each execution (about 6 min on one thread) |
