
- each notebook generates datasets with varying parameters and varying N, P with some fixed random seeds
- the data are saved in ../all_data/

- note for user-- the data save function takes in a parameters dict and an extra_array option

- the parameters dict is tracked across all datasets together in configs.json, and extra_array can have the full value dumped within it
- sometimes you might want to reduce the size of the parameters dict, and use text 
(see the anisotropic_gaussian for example, where parameters stores how the cov matrix was generated and extra_array stores the actual cov matrix)
- the parameters need to be unique to each data generating output (eg you might have to include the seed to make the dictionary unique)

- also, data is appended to the manifest.csv and configs.json after each new run / or each new save.
- if deleting and regenerating any data then rerunning all the things per data generating mechanism is the easiest
- there's a delete_dataset function to safely delete the dataset folder, and remove all rows corresponding to a data generator mechanism, rather than doing it manually

folder structure for data saving:

all_data/data_generation_type/params{i}/params.json
stores parameters for the different seeds and N, P's
some datasets might include N,P in the parameters implicitly (by the size of params used to generate for eg.)

all_data/data_generation_type/params{i}/seed{x}_N{n}_P{p}.npy
data matrix

all_data/manifest.csv 
csv of all generated data, can be used to fetch relative file paths, and filter on N, P

all_data/configs.json (NOT WRITTEN BY DEFAULT BECAUSE OF SIZE)
maps params{i} with the parameter used for a specific data_generation_type

implementation notes:
- HJ currently using a lightweight conda env called hcda-datagen to run the notebooks
- saving all data in ephemeral space

```
conda create -n hcda-datagen -c conda-forge python=3.11 numpy scipy scikit-learn pandas matplotlib jupyterlab ipykernel -y
conda activate datagen
python -m ipykernel install --user --name hcda-datagen
```

quick instructions to get this to work on imperial hpc:

go to https://jupyter.cx3.rcs.ic.ac.uk/hub/home and click on 'token' (top left)
generate api token, use that to tunnel jupyter from vscode to the server
i followed this tutorial- https://www.anaconda.com/docs/anaconda-platform/admin/integrations/jupyterhub/jupyter-vscode

