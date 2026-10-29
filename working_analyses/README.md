for anyone working with jupyter notebooks, use nbstripout to make sure local outputs of jupyter notebooks are not published to the main github

initial setup within your conda env:
```
conda install nbstripout
nbstripout --install
```
the filters are stored in the ```.gitattributes``` file.

checks to see if it is working:

```
nbstripout --status

git check-attr filter -- working_analyses/hjResearch_2025/data/simulated_data/data_simulation_code/concentric_shells.ipynb 
```

---

notes for G:

- some conflicts in numpy1 / numpy2 versioning, needs fixing in toml?

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.5.4 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
