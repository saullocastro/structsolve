# Notebooks

Usage examples, tutorials and validation notebooks of `structsolve`, stored
with their outputs and rendered in the documentation. They are run from this
folder; the examples run the code of the unit tests and import their helper
functions from `tests/`.

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
| `ex_linear_static.ipynb` | Linear static analysis with `solve` and `Analysis` |
| `ex_linear_buckling.ipynb` | Linear buckling of a simply supported plate, Ritz model (FSDT) with Bardell's functions |
| `ex_frequency.ipynb` | Frequency analysis with `freq`: options, sorting, prestressed plate |
| `ex_newton_raphson.ipynb` | Non-linear static analysis with the Newton-Raphson method, hardening springs |
| `ex_arc_length.ipynb` | Arc-length methods through the limit point of a von Mises truss |
| `ex_follower_loads.ipynb` | Follower loads and unsymmetric matrices: non-linear analyses, inertia, linear buckling, flutter |
| `ex_prescribed_displacements.ipynb` | Prescribed displacements imposed exactly, with and without `constraint_values` |
| `ex_cylinder_postbuckling.ipynb` | Simo, Fox and Rifai (1990): model, imperfection, convergence study in the Ritz terms and figure (`ex_cylinder_postbuckling.png`, written by the command `plot` of `tests/simo1990_utils.py`) |
| `tutorial_von_mises_truss.ipynb` | Tutorial: static, buckling, frequency and non-linear analyses of a von Mises truss |
| `simo1990_cylinder_postbuckling.ipynb` | Simo, Fox and Rifai (1990): post-buckling of an axially compressed cylinder with imperfections of 10 % and 50 % of the thickness, Riks method, and of the perfect one by branch switching with `linear_solver='pardiso'` and `'splu'`; runs `panels` and `structsolve` at each execution (about 3.5 min on one thread) |
