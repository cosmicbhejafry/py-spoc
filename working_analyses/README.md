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